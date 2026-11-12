"""RVM 形状的整链实验: 固定大小输出块 + 每 core 自己的行数 + 完全由描述符驱动的 core 循环 + OC / IC 分块

  source npu/env.sh && python -m npu.chain_c16                 # 正确性 + 计时
  python -m npu.chain_c16 --spec enc --iters 50                # 只跑一条链 (见 CHAINS), 打印每层切分
  python -m npu.chain_c16 --compute enc                        # 纯计算 / 额外开销拆分

接 chain_hetero.py (desc-loop spike)。那边留下的问题: (1) 输出 fifo 元素大小要求所有层 OCB·WT 相等;
(2) 45/23/12 行切不成两段相等、宽 40/20 切不成 2 条 8 的倍数; (3) core 代码随 (行数, stride) 类别数增长。
这里的做法:
  · 激活布局改成"16 通道像素" [H+2][CP][PAD + Ws + PAD][16] (内核 npu/kernels/conv_c16.cc)。
    一个输出块 = 1 个通道组 × CW (=40) 像素 = 640 字节, 是 DDR 里连续的一段 → 任何层的一行输出都是整数个块,
    drain 的 TAP 不用变 ([行][通道组][WT·16])。
  · 存储宽度 Ws 补到 CW 的倍数 (12×20 存成 40 宽), 内核把超出真实宽度的输出写 0 → 补的列就是下一层的零填充。
  · 每层自己选切分 (宽 t 条 × 高 4/t 段), 行数不能均分时各 core 行数不同 (23 = 6+6+6+5); 用不满 4 个 core 的层,
    空闲 core 这一层什么都不做。每 core 的行数 / 有效宽度 / 选择字节都在权重头里按 core 给。
  · core 循环完全由描述符驱动: 行数、每行块数都是运行时 range_ 边界。
    规则 (desc-loop spike 后续试出来的): ObjectFifo 的 acquire/release 可以放进运行时次数的循环,
    只要循环体相对于进入时的持有数是平衡的。s1 滑窗先 acquire(2), 循环体 acquire(3)/release(1) 就是平衡的;
    s2 每行 acquire(3)/release(3); 块循环 acquire(1)/release(1)。
  · 输入元素固定 E_IN 字节 = [ICP_pad][PLANE], 每层 PLANE = E_IN / ICP_pad (≥ 真实条带宽度), 多读的是垃圾。

"步骤" (一个权重元素 = 一步, core 的外层循环按步走, 每步的权重头说这一步干什么):
  · full  整层一步。
  · oc    OC 分块: 权重放不进一个元素 (W_MAX) 时, 按输出通道组切成 n_wb 块, 每块把同一份输入再喂一遍
          (fill 的 TAP 最外维步长 0 重复 n_wb 次, 还是一个 BD), 输出写到对应的通道组 (drain 一个 BD)。
  · ict   IC 分块 (只用于 1×1): 一行输入放不进一个元素时, 每步只带 ≤ 8 个输入通道组, 部分和存在 core 本地的
          int32 acc 缓冲区里 (Buffer); 中间步 (sel_acc) 不出输出, 每个 OC 子块的最后一步 (sel_fin) 量化输出。
          acc 只有 ACC_CHUNKS 个块那么大, 所以 OC 也要切成 k 个通道组一份 (每份把输入再读一遍)。
  同一层的步按 OC 块/份分成几个 task group (每组每个 shim 通道 ≤ FILLS_PER_TG 个 fill, 多了会挂死)。
  · OC 并行: 一条带 (宽 ≤ 40) 的 OC 分块层不再按行切给 4 个 core, 而是每个 core 算不同的 OC 块 (c, c+4, …)、
    处理全部行 → 每 core 只拿 1/4 的权重。权重缓冲区按 [槽][core] 存, 每个 core 读自己那份
    (所以 e_w 补到 256 的倍数, fill 写成 [槽][e_w/256][256]: 跳着读时最内维不能超过 1023 个字)。
  · 补宽列不算也不读: 整组 8 像素都在有效宽度之外时内核只写零; PLANE 按有效宽度 (向上取 8) 定,
    12×20 的层每块只要 512 B → 一个输入元素放 15 个通道组, 少了很多 IC 分块。

还没做: depthwise、5×5、空洞; 3×3 的 IC 分块 (RVM 里没有需要的层)。
层间激活约定同前: 内核输出 uint8 (饱和=ReLU), 下一层把同一字节当 int8 读。
"""
from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction

from .fused import KERNEL_INC
from .harness import SelfTestResult

SRC = str(Path(__file__).resolve().parent / "kernels" / "conv_c16.cc")
PAD, HDR, CW = 4, 128, 40
E_IN = 7680                    # 输入元素字节数 (所有层相同; = 10 组 × (40+8)·16)
W_MAX = 18432                  # 一个权重元素里最多的权重字节 (= 3×3 128→16)
ACC_CHUNKS = 3                 # acc 缓冲区能放几个输出块的部分和 (每块 CW·16 个 int32 = 2.5 KB)
N_CORES = 4
W_DEPTH = 1                    # 权重 fifo 深度 (2 = 算这一步时预取下一步)
FILLS_PER_TG = 4               # 一个 task group 里同一个 shim 通道最多排几个 fill
MAX_PLANE = 4064               # shim BD 最内维上限之内


def cdiv(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class Layer:
    ic: int
    oc: int
    k: int
    s: int
    h: int
    w: int

    @property
    def ho(self): return cdiv(self.h, self.s)
    @property
    def wo(self): return cdiv(self.w, self.s)
    @property
    def icp(self): return self.ic // 16
    @property
    def ocp(self): return self.oc // 16
    @property
    def n_w(self): return self.k * self.k * self.ic * self.oc
    @property
    def macs(self): return self.ho * self.wo * self.oc * self.ic * self.k * self.k


# RVM 编码器形状 (通道补到 16 的倍数; dw 卷积暂时用普通 3×3 代替), 输入 16 通道 180×320
R = {
    "e1": Layer(16, 16, 3, 2, 180, 320),   # → 90×160            (≈ Conv_8 / Conv_17)
    "e2": Layer(16, 64, 1, 1, 90, 160),    # 1×1 升维            (≈ Conv_15)
    "e3": Layer(64, 32, 3, 2, 90, 160),    # → 45×80, 45 行奇数  (≈ Conv_17 + Conv_19)
    "e4": Layer(32, 80, 1, 1, 45, 80),     # 1×1 32→80          (≈ Conv_20, 72→80)
    "e5": Layer(80, 32, 1, 1, 45, 80),     # 1×1 80→32, ICP 5   (≈ Conv_24)
    "e6": Layer(32, 32, 3, 1, 45, 80),     # 3×3 s1 @45×80, 行 23+22
    "e7": Layer(32, 48, 3, 2, 45, 80),     # → 23×40             (≈ Conv_28)
    "e8": Layer(48, 128, 1, 1, 23, 40),    # 1×1 48→128 @23×40  (≈ Conv_37)
    "e9": Layer(128, 48, 1, 1, 23, 40),    # 1×1 128→48         (≈ Conv_47)
    "e10": Layer(48, 32, 3, 2, 23, 40),    # → 12×20, 宽 20 存成 40
    "e11": Layer(32, 128, 1, 1, 12, 20),   # 1×1 @12×20
    "e12": Layer(128, 32, 1, 1, 12, 20),
    "e13": Layer(32, 32, 3, 1, 12, 20),    # 3×3 s1 @12×20 (补宽列必须是零)
}
# RVM 编码器 12×20 段的真实形状 (跳过 depthwise 和 SE; 200/184 补到 208/192), 输入 240 通道
_T = [(240, 80, 1, "Conv_67"), (80, 208, 1, "Conv_68"), (208, 80, 1, "Conv_74"), (80, 480, 1, "Conv_92"),
      (480, 112, 1, "Conv_104"), (112, 672, 1, "Conv_105"), (672, 160, 1, "Conv_131"),
      (160, 960, 1, "Conv_132"), (960, 160, 1, "Conv_144"), (160, 960, 1, "Conv_146"),
      (960, 128, 1, "Conv_163"), (128, 128, 3, "Conv_176"), (128, 64, 3, "Conv_181")]
for _ic, _oc, _k, _n in _T:
    R[_n] = Layer(_ic, _oc, _k, 1, 12, 20)
ENC = ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "e11", "e12", "e13"]
DEEP = [n for *_, n in _T]
CHAINS = {
    "enc": ENC,
    "head": ENC[:3],
    "low": ENC + ["e11", "e12", "e13"] * 4,           # 12×20 的三层重复
    "deep": DEEP,                                     # 12×20 高通道段 (OC / IC 分块)
}


def chain_layers(spec: str) -> list[Layer]:
    if spec in CHAINS:
        return [R[k] for k in CHAINS[spec]]
    return [R[k] for k in spec.split(",")]


def spec_str(layers) -> str:
    return ";".join(f"{l.ic},{l.oc},{l.k},{l.s},{l.h},{l.w}" for l in layers)


def from_spec_str(s: str) -> list[Layer]:
    return [Layer(*map(int, p.split(","))) for p in s.split(";")]


@dataclass
class Tensor:
    c: int
    h: int
    w: int        # 真实宽度
    ws: int       # 存储宽度 (≥ w)

    @property
    def cplane(self): return (self.ws + 2 * PAD) * 16
    @property
    def frow(self): return (self.c // 16) * self.cplane
    @property
    def size(self): return (self.h + 2) * self.frow
    @property
    def key(self): return (self.c, self.h, self.ws)


@dataclass
class Tiling:
    n_tw: int
    n_bands: int
    wt: int
    band_y0: list
    band_rows: list
    mode: str = "full"          # full / oc / ict
    icp_pad: int = 0            # 输入元素里的通道组数 (full/oc) 或每步通道组数 m (ict)
    plane: int = 0
    nocp: int = 0               # 每个 OC 块 (oc) / 每份 (ict) 的输出通道组数
    n_ob: int = 1               # OC 块数 / 份数
    n_icb: int = 1              # IC 步数 (ict)
    steps: list = field(default_factory=list)   # [(ocp0, nocp, icp0, nicp, first, last)]
    ocpar: bool = False         # OC 并行: 4 个 core 各算不同的 OC 块、都算全部行 (低分辨率的 OC 分块层)

    def cores(self):
        """[(core id, 条 t, 段 b)]"""
        if self.ocpar:
            return [(c, 0, 0) for c in range(min(N_CORES, self.n_ob))]
        return [(t * self.n_bands + b, t, b) for t in range(self.n_tw) for b in range(self.n_bands)]

    def blocks_of(self, c):
        """OC 并行时 core c 负责的 OC 块数 (块 c, c+4, c+8, …)"""
        return len(range(c, self.n_ob, N_CORES))

    def slots(self):
        """core 外层循环的每一步 = 一个槽, 槽里每个 core 一个步骤 (None = 这个 core 这一步空闲)"""
        if not self.ocpar:
            return [[st] * N_CORES for st in self.steps]
        return [[self.steps[s * N_CORES + c] if s * N_CORES + c < self.n_ob else None for c in range(N_CORES)]
                for s in range(cdiv(self.n_ob, N_CORES))]

    @property
    def nxc(self): return self.wt // CW


def _divisor_le(n, cap):
    return max(d for d in range(1, n + 1) if n % d == 0 and d <= cap)


def choose_tiling(l: Layer) -> Tiling:
    best = None
    for t in (4, 2, 1):
        wt = cdiv(cdiv(l.wo, t), CW) * CW
        bands = N_CORES // t
        if l.ho < bands or (t - 1) * wt >= l.wo:
            continue
        score = (t * wt / l.wo, -t)
        if best is None or score < best[0]:
            best = (score, t, wt, bands)
    _, t, wt, bands = best
    q, r = divmod(l.ho, bands)
    rows = [q + (1 if b < r else 0) for b in range(bands)]
    tl = Tiling(t, bands, wt, [sum(rows[:b]) for b in range(bands)], rows)
    we = min(wt, cdiv(min(l.wo, wt), 8) * 8)          # 内核实际会读到的输出宽度 (整组在 wvalid 外的不读)
    need = (l.s * we + 2 * PAD) * 16
    per_ocp = l.k * l.k * l.ic * 16
    for p in range(l.icp, 64):
        if E_IN % p == 0 and need <= E_IN // p <= MAX_PLANE and (E_IN // p) % 64 == 0:
            tl.icp_pad, tl.plane = p, E_IN // p
            break
    if tl.plane:                                     # 一行输入放得下
        if per_ocp > W_MAX:
            raise ValueError(f"{l}: 一个输出通道组的权重 {per_ocp} B > {W_MAX} (要 3×3 的 IC 分块)")
        tl.nocp = _divisor_le(l.ocp, W_MAX // per_ocp)
        tl.n_ob = l.ocp // tl.nocp
        tl.mode = "full" if tl.n_ob == 1 else "oc"
        if tl.mode == "oc" and l.s == 1 and tl.n_tw == 1:
            # 低分辨率 (一条带) 的 OC 分块层: 改成 OC 并行, 每 core 只拿自己那 1/4 的权重, 处理全部行。
            # 块大小取 "每 core 最多要算的通道组数 + 每槽固定开销" 最小的
            cap = W_MAX // per_ocp
            d = min((dd for dd in range(1, l.ocp + 1) if l.ocp % dd == 0 and dd <= cap),
                    key=lambda dd: (cdiv(l.ocp // dd, N_CORES) * (dd + 1), -dd))
            tl.nocp, tl.n_ob, tl.ocpar = d, l.ocp // d, True
            tl.n_bands, tl.band_y0, tl.band_rows = 1, [0], [l.ho]
        tl.steps = [(o * tl.nocp, tl.nocp, 0, l.icp, 1, 1) for o in range(tl.n_ob)]
        if tl.mode == "oc" and l.s == 2 and tl.n_ob > 8:
            raise ValueError(f"{l}: s2 的 OC 分块要每块一个 fill BD, 块太多")
        return tl
    if l.k != 1:
        raise ValueError(f"{l}: 3×3 的一行输入放不下 (要 3×3 的 IC 分块)")
    # IC 分块: 每步的通道组数 m 取放得下的最大值
    m = max(p for p in range(1, 64) if E_IN % p == 0 and need <= E_IN // p <= MAX_PLANE and (E_IN // p) % 64 == 0)
    tl.mode, tl.icp_pad, tl.plane = "ict", m, E_IN // m
    tl.n_icb = cdiv(l.icp, m)
    k_cap = ACC_CHUNKS // (max(rows) * tl.nxc)
    if k_cap < 1:
        raise ValueError(f"{l}: acc 放不下一行 ({max(rows)} 行 × {tl.nxc} 块 > {ACC_CHUNKS})")
    tl.nocp = _divisor_le(l.ocp, min(k_cap, W_MAX // (m * 256)))
    tl.n_ob = l.ocp // tl.nocp
    tl.steps = [(o * tl.nocp, tl.nocp, i * m, min(m, l.icp - i * m), int(i == 0), int(i == tl.n_icb - 1))
                for o in range(tl.n_ob) for i in range(tl.n_icb)]
    return tl


class Plan:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
        self.tilings = [choose_tiling(l) for l in layers]
        l0 = layers[0]
        cur = Tensor(l0.ic, l0.h, l0.w, l0.w)
        self.tensors = [cur]
        for l, tl in zip(layers, self.tilings):
            assert l.ic == cur.c and l.h == cur.h and l.w == cur.w, f"接不上: {cur} → {l}"
            assert l.ic % 16 == 0 and l.oc % 16 == 0
            cur = Tensor(l.oc, l.ho, l.wo, tl.n_tw * tl.wt)
            self.tensors.append(cur)
        self.n_steps = sum(len(tl.slots()) for tl in self.tilings)       # core 外层循环次数
        # 补到 256 的倍数: 权重按 [槽][core] 交错存, fill 是跳着读的, 最内维不能超过 1023 个字
        self.e_w = cdiv(HDR + max(self._step_wbytes(l, st) for l, tl in zip(layers, self.tilings)
                                  for st in tl.steps), 256) * 256
        self.step0 = np.cumsum([0] + [len(tl.slots()) for tl in self.tilings]).tolist()
        self.slot_off, off = {}, 0
        for t in self.tensors[1:-1]:
            if t.key not in self.slot_off:
                self.slot_off[t.key] = (off, off + t.size)
                off += 2 * t.size
        toggle, prev, self.io = {}, ("act", 0), []
        for k, t in enumerate(self.tensors[1:]):
            if k == len(layers) - 1:
                dst = ("out", 0)
            else:
                i = toggle.get(t.key, 0)
                toggle[t.key] = i ^ 1
                dst = ("s", self.slot_off[t.key][i])
            self.io.append((prev, dst))
            prev = dst
        ends = {"act": self.tensors[0].size, "s": max(off, 1), "out": self.tensors[-1].size}
        for k, l in enumerate(layers):
            (src, soff), _ = self.io[k]
            ti, tl = self.tensors[k], self.tilings[k]
            for c, t, b in tl.cores():
                for o, sz, st in self.fill_taps(k, soff, c, t, b):
                    end = o + sum((n - 1) * s_ for n, s_ in zip(sz, st)) + 1
                    ends[src] = max(ends[src], end)
        self.size = {k: cdiv(v, 64) * 64 for k, v in ends.items()}

    @staticmethod
    def _step_wbytes(l, st):
        _, nocp, _, nicp, _, _ = st
        return l.k * l.k * nicp * 16 * nocp * 16

    @staticmethod
    def ob_groups(tl, l):
        """一层按 OC 块/份分成几个 task group, 每组每个 shim 通道最多 FILLS_PER_TG 个 fill
        (实测一个 task group 里同一通道排 7 个以上 DMA 任务就挂死, 5 个没事)"""
        if tl.ocpar:
            return [(0, tl.n_ob)]
        per = 1 if tl.mode == "ict" else (0 if l.s == 1 else cdiv(max(tl.band_rows), 64))
        g = tl.n_ob if per == 0 else max(1, FILLS_PER_TG // per)
        return [(o, min(o + g, tl.n_ob)) for o in range(0, tl.n_ob, g)]

    def fill_taps(self, k, base, c, t, b, ob0=0, ob1=None):
        """core c 这一层 (OC 块/份 ob0..ob1) 的输入 fill: [(偏移, sizes, strides)]"""
        l, tl, ti = self.layers[k], self.tilings[k], self.tensors[k]
        ob1 = tl.n_ob if ob1 is None else ob1
        nob = tl.blocks_of(c) if tl.ocpar else ob1 - ob0
        rows = tl.band_rows[b]
        off = base + l.s * tl.band_y0[b] * ti.frow + t * l.s * tl.wt * 16
        if tl.mode == "ict":     # 不要上下边行; 每份 OC 一个 BD: [IC 步][行][m 组][PLANE]
            o1 = off + ti.frow
            return [(o1, [tl.n_icb, rows, tl.icp_pad, tl.plane], [tl.icp_pad * ti.cplane, ti.frow, ti.cplane, 1])
                    for _ in range(nob)]
        if l.s == 1:             # OC 块: 最外维步长 0, 同一份输入重复 nob 次
            return [(off, [nob, rows + 2, tl.icp_pad, tl.plane], [0, ti.frow, ti.cplane, 1])]
        return [(off + 2 * r0 * ti.frow, [min(64, rows - r0), 3, tl.icp_pad, tl.plane],
                 [2 * ti.frow, ti.frow, ti.cplane, 1])
                for _ in range(nob) for r0 in range(0, rows, 64)]

    def drain_tap(self, k, base, c, t, b, ob0=0, ob1=None):
        tl, to = self.tilings[k], self.tensors[k + 1]
        if tl.ocpar:             # core c 的块 c, c+4, … → 通道组 (c + 4i)·nocp
            off = base + to.frow + PAD * 16 + c * tl.nocp * to.cplane
            return off, [tl.blocks_of(c), tl.band_rows[0], tl.nocp, tl.wt * 16], \
                [N_CORES * tl.nocp * to.cplane, to.frow, to.cplane, 1]
        ob1 = tl.n_ob if ob1 is None else ob1
        off = base + (1 + tl.band_y0[b]) * to.frow + (PAD + t * tl.wt) * 16 + ob0 * tl.nocp * to.cplane
        return off, [ob1 - ob0, tl.band_rows[b], tl.nocp, tl.wt * 16], [tl.nocp * to.cplane, to.frow, to.cplane, 1]

    def describe(self):
        out = []
        for l, tl in zip(self.layers, self.tilings):
            kind = "1×1" if l.k == 1 else f"3×3 s{l.s}"
            extra = {"full": "", "oc": f" OC 分 {tl.n_ob} 块×{tl.nocp} 组" + (" (OC 并行)" if tl.ocpar else ""),
                     "ict": f" IC 分 {tl.n_icb} 步×{tl.icp_pad} 组, OC 分 {tl.n_ob} 份×{tl.nocp} 组"}[tl.mode]
            out.append(f"{l.ic}→{l.oc} {kind} {l.h}×{l.w}→{l.ho}×{l.wo}: {tl.n_tw}条×{tl.n_bands}段 "
                       f"WT {tl.wt} 行 {tl.band_rows} ICPpad {tl.icp_pad} PLANE {tl.plane}{extra}")
        return out


def _tap(n_total, off, sizes, strides):
    from aie.helpers.taplib import TensorAccessPattern
    return TensorAccessPattern((1, n_total), off, sizes, strides)


def _u16(buf, off):
    """core 里读权重头的一个 u16 → index"""
    from aie.dialects import arith, memref
    from aie.ir import IndexType, IntegerType
    i32 = IntegerType.get_signless(32)
    ld = lambda o: arith.extui(i32, memref.load(buf, [arith.constant(IndexType.get(), o)]))
    v = arith.addi(ld(off), arith.shli(ld(off + 1), arith.constant(i32, 8)))
    return arith.index_cast(IndexType.get(), v)


def _i32(iv):
    from aie.dialects import arith
    from aie.ir import IntegerType
    return arith.index_cast(IntegerType.get_signless(32), iv)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def conv_c16(act: In, wts: In, scratch: InOut, out: Out, *, spec: CompileTime[str],
             reps: CompileTime[int] = 1):
    layers = from_spec_str(spec)
    P = Plan(layers)
    i8, u8, i32 = np.dtype[np.int8], np.dtype[np.uint8], np.dtype[np.int32]
    in_ty = np.ndarray[(E_IN,), i8]
    out_ty = np.ndarray[(CW * 16,), u8]
    w_ty = np.ndarray[(P.e_w,), i8]
    acc_ty = np.ndarray[(ACC_CHUNKS * CW * 16,), i32]
    kern = ExternalFunction("convC16Chunk", source_file=SRC,
                            arg_types=[in_ty, in_ty, in_ty, w_ty, out_ty, acc_ty, np.int32, np.int32, np.int32],
                            include_dirs=[KERNEL_INC], compile_flags=[f"-DCW={CW}"])

    def make_core(cid):
        hc = 32 + 16 * cid          # 本 core 在权重头里的那段

        def call(k, win, wv, o, acc, j, r):
            for _ in range(reps):
                k(win[0], win[1], win[2], wv, o, acc, _i32(j), _i32(r), cid)

        def core(f_in, f_w, f_out, k, acc, dummy):
            for _ in range_(P.n_steps):
                wv = f_w.acquire(1)
                rows, nch = _u16(wv, hc), _u16(wv, hc + 8)
                for _ in range_(_u16(wv, hc + 2)):          # stride 1 (含 1×1): 滑窗
                    f_in.acquire(2)
                    for r in range_(rows):
                        win = f_in.acquire(3)
                        for j in range_(nch):
                            o = f_out.acquire(1)
                            call(k, win, wv, o, acc, j, r)
                            f_out.release(1)
                        f_in.release(1)
                    f_in.release(2)
                for _ in range_(_u16(wv, hc + 4)):          # stride 2: 每行 3 行
                    for r in range_(rows):
                        win = f_in.acquire(3)
                        for j in range_(nch):
                            o = f_out.acquire(1)
                            call(k, win, wv, o, acc, j, r)
                            f_out.release(1)
                        f_in.release(3)
                for _ in range_(_u16(wv, hc + 10)):         # IC 分块中间步: 只累加
                    for r in range_(rows):
                        win = f_in.acquire(1)
                        for j in range_(nch):
                            k(win, win, win, wv, dummy, acc, _i32(j), _i32(r), cid)
                        f_in.release(1)
                for _ in range_(_u16(wv, hc + 12)):         # IC 分块最后一步: 出输出
                    for r in range_(rows):
                        win = f_in.acquire(1)
                        for j in range_(nch):
                            o = f_out.acquire(1)
                            k(win, win, win, wv, o, acc, _i32(j), _i32(r), cid)
                            f_out.release(1)
                        f_in.release(1)
                f_w.release(1)
        return core

    fifos, workers = [], []
    for c in range(N_CORES):
        f_in = ObjectFifo(in_ty, depth=4, name=f"c16_{c}i")
        f_w = ObjectFifo(w_ty, depth=W_DEPTH, name=f"c16_{c}w")
        f_out = ObjectFifo(out_ty, depth=2, name=f"c16_{c}o")
        acc = Buffer(acc_ty, name=f"c16_{c}acc")
        dummy = Buffer(out_ty, name=f"c16_{c}dummy")
        fifos.append((f_in, f_w, f_out))
        workers.append(Worker(make_core(c), [f_in.cons(), f_w.cons(), f_out.prod(), kern, acc, dummy]))

    rt = Runtime()
    nw = P.n_steps * N_CORES * P.e_w                # 权重缓冲区 [槽][core][e_w]
    with rt.sequence(np.ndarray[(P.size["act"],), i8], np.ndarray[(nw,), i8],
                     np.ndarray[(P.size["s"],), i8], np.ndarray[(P.size["out"],), u8]) as (a, w, s, o):
        bufs = {"act": a, "s": s, "out": o}
        rt.start(*workers)
        for k, l in enumerate(layers):
            (src, soff), (dst, doff) = P.io[k]
            tl = P.tilings[k]
            spo = len(tl.steps) // tl.n_ob                   # 每个 OC 块/份的步数
            for ob0, ob1 in P.ob_groups(tl, l):
                if tl.ocpar:
                    ns, s0 = len(tl.slots()), P.step0[k]
                else:
                    ns, s0 = (ob1 - ob0) * spo, P.step0[k] + ob0 * spo
                tg = rt.task_group()
                for c, (f_in, f_w, _) in enumerate(fifos):
                    rt.fill(f_w.prod(), w, tap=_tap(nw, (s0 * N_CORES + c) * P.e_w, [1, ns, P.e_w // 256, 256],
                                                     [0, N_CORES * P.e_w, 256, 1]), task_group=tg)
                for c, t, b in tl.cores():
                    for o_, sz, st in P.fill_taps(k, soff, c, t, b, ob0, ob1):
                        rt.fill(fifos[c][0].prod(), bufs[src], tap=_tap(P.size[src], o_, sz, st), task_group=tg)
                for c, t, b in tl.cores():
                    o_, sz, st = P.drain_tap(k, doff, c, t, b, ob0, ob1)
                    rt.drain(fifos[c][2].cons(), bufs[dst], tap=_tap(P.size[dst], o_, sz, st), wait=True,
                             task_group=tg)
                rt.finish_task_group(tg)
    return Program(iron.get_current_device(), rt).resolve_program()


# ---------------------------------------------------------------- 主机端

def to_layout(x: np.ndarray, t: Tensor, size: int) -> np.ndarray:
    c, h, w = x.shape
    p = np.zeros((c, h + 2, t.ws + 2 * PAD), x.dtype)
    p[:, 1:-1, PAD:PAD + w] = x
    flat = p.reshape(c // 16, 16, h + 2, t.ws + 2 * PAD).transpose(2, 0, 3, 1).reshape(-1)
    buf = np.zeros(size, x.dtype)
    buf[:flat.size] = flat
    return buf


def from_layout(buf: np.ndarray, t: Tensor) -> np.ndarray:
    """返回 (C, H+2, Ws+2PAD) 的整幅 (含边), 方便同时检查真实区域和补宽列是否为零"""
    wp = t.ws + 2 * PAD
    return buf[:t.size].reshape(t.h + 2, t.c // 16, wp, 16).transpose(1, 3, 0, 2).reshape(t.c, t.h + 2, wp)


def pack_layer(l: Layer, tl: Tiling, Wq: np.ndarray, shift: int, e_w: int) -> list[np.ndarray]:
    """一层的所有槽 × 每个 core 一个 e_w 字节的权重元素 [头][权重], 顺序 [槽][core]"""
    ocp, icp = l.ocp, l.icp
    if l.k == 3:   # [ocp][ky][icp][kx][h][ob][ic8][oc8]
        full = Wq.reshape(ocp, 2, 8, icp, 2, 8, 3, 3).transpose(0, 6, 3, 7, 4, 1, 5, 2)
    else:          # [ocp][icp][h][ob][ic8][oc8]
        full = Wq.reshape(ocp, 2, 8, icp, 2, 8).transpose(0, 3, 4, 1, 5, 2)
    out = []
    for slot in tl.slots():
        for c_own, st in enumerate(slot):
            out.append(_pack_step(l, tl, full, shift, e_w, st, c_own))
    return out


def _pack_step(l, tl, full, shift, e_w, st, c_own):
    buf = np.zeros(e_w, np.int8)
    if st is None:               # 这个 core 这一槽空闲: 全零头 (所有选择字节为 0)
        return buf
    ocp0, nocp, icp0, nicp, first, last = st
    if True:
        if l.k == 3:
            w = full[ocp0:ocp0 + nocp, :, icp0:icp0 + nicp]
        else:
            w = full[ocp0:ocp0 + nocp, icp0:icp0 + nicp]
        hdr = np.zeros(32, np.int32)
        hdr[:8] = [nicp, nocp, tl.wt, l.s, l.k, shift, tl.nxc, tl.plane]
        hdr[24:26] = [first, last]
        buf[:HDR] = hdr.view(np.int8)
        for c, t, b in tl.cores():
            if tl.ocpar and c != c_own:
                continue
            rows = tl.band_rows[b]
            wvalid = min(max(l.wo - t * tl.wt, 0), tl.wt)
            ict = tl.mode == "ict"
            per = np.array([rows, int(not ict and l.s == 1), int(not ict and l.s == 2), wvalid, nocp * tl.nxc,
                            int(ict and not last), int(ict and last), 0], np.uint16)
            buf[32 + 16 * c: 48 + 16 * c] = per.view(np.int8)
        flat = np.ascontiguousarray(w).reshape(-1)
        buf[HDR:HDR + flat.size] = flat
    return buf


def conv_acc(l: Layer, x_i8: np.ndarray, Wq: np.ndarray) -> np.ndarray:
    p = (l.k - 1) // 2
    xp = np.zeros((l.ic, l.h + 2 * p, l.w + 2 * p), np.int32)
    xp[:, p:p + l.h, p:p + l.w] = x_i8
    acc = np.zeros((l.oc, l.ho * l.wo), np.int32)
    Wi = Wq.astype(np.int32)
    for ky in range(l.k):
        for kx in range(l.k):
            patch = xp[:, ky:ky + l.s * (l.ho - 1) + 1:l.s, kx:kx + l.s * (l.wo - 1) + 1:l.s]
            acc += Wi[:, :, ky, kx] @ patch.reshape(l.ic, -1)
    return acc.reshape(l.oc, l.ho, l.wo)


def requant(acc, shift):
    return np.clip((acc + (1 << (shift - 1))) >> shift, 0, 255).astype(np.uint8)


def make_case(layers, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(-100, 101, (layers[0].ic, layers[0].h, layers[0].w), dtype=np.int64).astype(np.int8)
    params, cur = [], x
    for l in layers:
        Wq = rng.integers(-12, 13, (l.oc, l.ic, l.k, l.k)).astype(np.int8)
        acc = conv_acc(l, cur.view(np.int8), Wq)
        shift = min(range(3, 16), key=lambda s: abs(requant(acc, s).mean() - 24))
        params.append((Wq, shift))
        cur = requant(acc, shift)
    return x, params, cur


class C16Chain:
    def __init__(self, layers, params, reps=1):
        self.layers, self.reps = layers, reps
        self.P = P = Plan(layers)
        self.spec = spec_str(layers)
        self.t_act = iron.zeros(P.size["act"], dtype=np.int8, device="npu")
        steps = [b for l, tl, (Wq, s) in zip(layers, P.tilings, params) for b in pack_layer(l, tl, Wq, s, P.e_w)]
        self.t_w = iron.tensor(np.concatenate(steps), dtype=np.int8, device="npu")
        self.t_s = iron.zeros(P.size["s"], dtype=np.int8, device="npu")
        self.t_out = iron.zeros(P.size["out"], dtype=np.uint8, device="npu")

    def set_input(self, x):
        self.t_act.data[:] = to_layout(x, self.P.tensors[0], self.P.size["act"])
        self.t_act._sync_to_device()

    def call(self):
        conv_c16(self.t_act, self.t_w, self.t_s, self.t_out, spec=self.spec, reps=self.reps)

    def result_full(self):
        return from_layout(self.t_out.numpy(), self.P.tensors[-1])


def check(full: np.ndarray, y: np.ndarray, t: Tensor) -> int:
    """真实区域与参考的最大差; 边和补宽列必须是 0 (否则返回 999)"""
    real = full[:, 1:-1, PAD:PAD + t.w]
    d = int(np.abs(real.astype(int) - y.astype(int)).max())
    mask = np.ones(full.shape, bool)
    mask[:, 1:-1, PAD:PAD + t.w] = False
    return d if not full[mask].any() else 999


def _cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def run_one(spec, iters=50, reps=1, seed=0, verbose=False):
    layers = chain_layers(spec)
    x, params, y = make_case(layers, seed)
    ch = C16Chain(layers, params, reps)
    if verbose:
        for line in ch.P.describe():
            print(f"    {line}", file=sys.stderr)
    ch.set_input(x)
    t0 = time.perf_counter()
    ch.call()
    jit_s = time.perf_counter() - t0
    d = check(ch.result_full(), y, ch.P.tensors[-1])
    walls, cpus = [], []
    for _ in range(iters):
        c0, t0 = _cpu(), time.perf_counter()
        ch.call()
        walls.append(time.perf_counter() - t0)
        cpus.append(_cpu() - c0)
    d = max(d, check(ch.result_full(), y, ch.P.tensors[-1]))
    walls.sort()
    return dict(spec=spec, n=len(layers), steps=ch.P.n_steps, max_diff=d, med=walls[len(walls) // 2] * 1e3,
                best=walls[0] * 1e3, cpu=float(np.mean(cpus)) * 1e3, jit=jit_s,
                gmac=sum(l.macs for l in layers) / 1e9,
                stats=f"末层 均值 {y.mean():.1f} 零 {(y == 0).mean() * 100:.0f}%")


def report(r):
    print(f"[c16] {r['spec']:<14s} N={r['n']:<3d} 步 {r['steps']:<3d} 最大差 {r['max_diff']}  中位 {r['med']:.2f} ms  "
          f"最快 {r['best']:.2f} ms  CPU {r['cpu']:.3f} ms/次  ({r['gmac']:.3f} GMAC, 首次 {r['jit']:.1f} s; "
          f"{r['stats']})", file=sys.stderr, flush=True)


def compute_split(spec, reps=(1, 2, 3), iters=50):
    res = [run_one(spec, iters, r) for r in reps]
    for r, x in zip(reps, res):
        print(f"[c16] {spec} reps={r} 最大差 {x['max_diff']} 中位 {x['med']:.2f} ms 最快 {x['best']:.2f} ms",
              file=sys.stderr, flush=True)
    for key in ("med", "best"):
        slope, icpt = np.polyfit(np.array(reps, float), np.array([x[key] for x in res]), 1)
        n = res[0]["n"]
        print(f"[c16] ({key}) {spec}: 纯计算 {slope:.3f} ms; 截距 {icpt:.3f} ms → 每层额外 ≈ "
              f"{(icpt - 0.25) / n:.3f} ms (固定按 0.25)", file=sys.stderr)


def selftest(spec="enc", iters=10) -> SelfTestResult:
    """npu.check 用: 一条链逐字节一致 (含补宽列为零), 返回中位耗时"""
    r = run_one(spec, iters)
    report(r)
    return SelfTestResult(r["max_diff"] == 0, r["max_diff"], r["med"])


def main(iters=50):
    ok = True
    for spec in ["e1", "e1,e2", "head", "enc", "low", "deep"]:
        r = run_one(spec, iters, verbose=(spec in ("enc", "deep")))
        report(r)
        ok &= r["max_diff"] == 0
    return SelfTestResult(ok, 0 if ok else 1, r["med"])


if __name__ == "__main__":
    def arg(name, default):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
    iters = int(arg("--iters", 50))
    if "--compute" in sys.argv:
        compute_split(arg("--compute", "enc"), iters=iters)
        sys.exit(0)
    if "--spec" in sys.argv:
        r = run_one(arg("--spec", "enc"), iters, verbose=True)
        report(r)
        sys.exit(0 if r["max_diff"] == 0 else 1)
    sys.exit(0 if main(iters) else 1)
