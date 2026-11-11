"""RVM 形状的整链实验: 固定大小输出块 + 每 core 自己的行数 + 完全由描述符驱动的 core 循环

  source npu/env.sh && python -m npu.chain_c16                 # 正确性 + 计时 + 开销拟合
  python -m npu.chain_c16 --spec enc --iters 50                # 只跑一条链 (见 CHAINS)
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
  · core 循环完全由描述符驱动: 只有 s1、s2 两段, 行数、每行块数都是运行时 range_ 边界。
    规则 (desc-loop spike 后续试出来的): ObjectFifo 的 acquire/release 可以放进运行时次数的循环,
    只要循环体相对于进入时的持有数是平衡的。s1 滑窗先 acquire(2), 循环体 acquire(3)/release(1) 就是平衡的;
    s2 每行 acquire(3)/release(3); 块循环 acquire(1)/release(1)。
  · 输入元素固定 E_IN 字节 = [ICP_pad][PLANE], 每层 PLANE = E_IN / ICP_pad (≥ 真实条带宽度), 多读的是垃圾。

还没做 (下一步): 权重/输入放不下单个 core 时的 OC / IC 分块 (RVM 里 960 通道的 1×1)、depthwise、5×5、空洞。
层间激活约定同前: 内核输出 uint8 (饱和=ReLU), 下一层把同一字节当 int8 读。
"""
from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, InOut, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction

from .fused import KERNEL_INC
from .harness import SelfTestResult

SRC = str(Path(__file__).resolve().parent / "kernels" / "conv_c16.cc")
PAD, HDR, CW = 4, 128, 40
E_IN = 6144                    # 输入元素字节数 (所有层相同)
N_CORES = 4
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
ENC = ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "e11", "e12", "e13"]
CHAINS = {
    "enc": ENC,
    "head": ENC[:3],
    "low": ENC + ["e11", "e12", "e13"] * 4,           # 12×20 的三层重复
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
    ws: int       # 存储宽度 (≥ w, CW 的倍数或首层的原宽)

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
    icp_pad: int
    plane: int

    def cores(self):
        """[(core id, t, b)]"""
        return [(t * self.n_bands + b, t, b) for t in range(self.n_tw) for b in range(self.n_bands)]


def choose_tiling(l: Layer) -> Tiling:
    best = None
    for t in (4, 2, 1):
        wt = cdiv(cdiv(l.wo, t), CW) * CW
        bands = N_CORES // t
        if l.ho < bands or (t - 1) * wt >= l.wo:      # 每条带至少有真实像素
            continue
        score = (t * wt / l.wo, -t)                   # 先少补宽, 再多切条
        if best is None or score < best[0]:
            best = (score, t, wt, bands)
    _, t, wt, bands = best
    q, r = divmod(l.ho, bands)
    rows = [q + (1 if b < r else 0) for b in range(bands)]
    y0 = [sum(rows[:b]) for b in range(bands)]
    need = (l.s * wt + 2 * PAD) * 16
    for p in range(l.icp, 64):
        if E_IN % p == 0 and E_IN // p >= need and E_IN // p <= MAX_PLANE and (E_IN // p) % 64 == 0:
            return Tiling(t, bands, wt, y0, rows, p, E_IN // p)
    raise ValueError(f"{l}: 输入元素放不下 (需要 {l.icp}×{need} 字节 > {E_IN})")


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
        self.e_w = HDR + max(l.n_w for l in layers)
        # scratch: 每种中间张量形状两个槽
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
            for _, t, b in tl.cores():
                o = self.fill_off(k, soff, t, b)
                span = (tl.band_rows[b] - 1) * l.s + 3
                ends[src] = max(ends[src], o + span * ti.frow + tl.icp_pad * ti.cplane + E_IN)
        self.size = {k: cdiv(v, 64) * 64 for k, v in ends.items()}

    def fill_off(self, k, base, t, b):
        l, tl, ti = self.layers[k], self.tilings[k], self.tensors[k]
        return base + l.s * tl.band_y0[b] * ti.frow + t * l.s * tl.wt * 16

    def drain_off(self, k, base, t, b):
        tl, to = self.tilings[k], self.tensors[k + 1]
        return base + (1 + tl.band_y0[b]) * to.frow + (PAD + t * tl.wt) * 16

    def describe(self):
        out = []
        for l, tl in zip(self.layers, self.tilings):
            kind = "1×1" if l.k == 1 else f"3×3 s{l.s}"
            out.append(f"{l.ic}→{l.oc} {kind} {l.h}×{l.w}→{l.ho}×{l.wo}: {tl.n_tw}条×{tl.n_bands}段 "
                       f"WT {tl.wt} 行 {tl.band_rows} ICPpad {tl.icp_pad} PLANE {tl.plane}")
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
    n = len(layers)
    i8, u8 = np.dtype[np.int8], np.dtype[np.uint8]
    in_ty = np.ndarray[(E_IN,), i8]
    out_ty = np.ndarray[(CW * 16,), u8]
    w_ty = np.ndarray[(P.e_w,), i8]
    kern = ExternalFunction("convC16Chunk", source_file=SRC,
                            arg_types=[in_ty, in_ty, in_ty, w_ty, out_ty, np.int32, np.int32],
                            include_dirs=[KERNEL_INC], compile_flags=[f"-DCW={CW}"])

    def make_core(cid):
        hc = 32 + 16 * cid          # 本 core 在权重头里的那段

        def core(f_in, f_w, f_out, k):
            for _ in range_(n):
                wv = f_w.acquire(1)
                rows, nch = _u16(wv, hc), _u16(wv, hc + 8)
                for _ in range_(_u16(wv, hc + 2)):          # stride 1 (含 1×1): 滑窗
                    f_in.acquire(2)
                    for _ in range_(rows):
                        win = f_in.acquire(3)
                        for j in range_(nch):
                            o = f_out.acquire(1)
                            for _ in range(reps):
                                k(win[0], win[1], win[2], wv, o, _i32(j), cid)
                            f_out.release(1)
                        f_in.release(1)
                    f_in.release(2)
                for _ in range_(_u16(wv, hc + 4)):          # stride 2: 每行 3 行
                    for _ in range_(rows):
                        win = f_in.acquire(3)
                        for j in range_(nch):
                            o = f_out.acquire(1)
                            for _ in range(reps):
                                k(win[0], win[1], win[2], wv, o, _i32(j), cid)
                            f_out.release(1)
                        f_in.release(3)
                f_w.release(1)
        return core

    fifos, workers = [], []
    for c in range(N_CORES):
        f_in = ObjectFifo(in_ty, depth=4, name=f"c16_{c}i")
        f_w = ObjectFifo(w_ty, depth=1, name=f"c16_{c}w")
        f_out = ObjectFifo(out_ty, depth=2, name=f"c16_{c}o")
        fifos.append((f_in, f_w, f_out))
        workers.append(Worker(make_core(c), [f_in.cons(), f_w.cons(), f_out.prod(), kern]))

    rt = Runtime()
    with rt.sequence(np.ndarray[(P.size["act"],), i8], np.ndarray[(n * P.e_w,), i8],
                     np.ndarray[(P.size["s"],), i8], np.ndarray[(P.size["out"],), u8]) as (a, w, s, o):
        bufs = {"act": a, "s": s, "out": o}
        rt.start(*workers)
        for k, l in enumerate(layers):
            (src, soff), (dst, doff) = P.io[k]
            tl, ti, to = P.tilings[k], P.tensors[k], P.tensors[k + 1]
            tg = rt.task_group()
            for c, (f_in, f_w, _) in enumerate(fifos):
                rt.fill(f_w.prod(), w, tap=_tap(n * P.e_w, k * P.e_w, [1, 1, 1, P.e_w], [0, 0, 0, 1]),
                        task_group=tg)
            for c, t, b in tl.cores():
                rows = tl.band_rows[b]
                off = P.fill_off(k, soff, t, b)
                if l.s == 1:
                    taps = [([1, rows + 2, tl.icp_pad, tl.plane], [0, ti.frow, ti.cplane, 1], off)]
                else:   # 最高维 (迭代维) 上限 64 → 每 64 个输出行一个 fill
                    taps = [([min(64, rows - r0), 3, tl.icp_pad, tl.plane], [2 * ti.frow, ti.frow, ti.cplane, 1],
                             off + 2 * r0 * ti.frow) for r0 in range(0, rows, 64)]
                for sz, st, o_ in taps:
                    rt.fill(fifos[c][0].prod(), bufs[src], tap=_tap(P.size[src], o_, sz, st), task_group=tg)
            for c, t, b in tl.cores():
                rt.drain(fifos[c][2].cons(), bufs[dst],
                         tap=_tap(P.size[dst], P.drain_off(k, doff, t, b),
                                  [1, tl.band_rows[b], l.ocp, tl.wt * 16], [0, to.frow, to.cplane, 1]),
                         wait=True, task_group=tg)
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
    """返回 (C, H, Ws+2PAD) 的整幅 (含边), 方便同时检查真实区域和补宽列是否为零"""
    wp = t.ws + 2 * PAD
    return buf[:t.size].reshape(t.h + 2, t.c // 16, wp, 16).transpose(1, 3, 0, 2).reshape(t.c, t.h + 2, wp)


def pack_weights(l: Layer, tl: Tiling, Wq: np.ndarray, shift: int, e_w: int) -> np.ndarray:
    ocp, icp = l.ocp, l.icp
    if l.k == 3:   # → [ocp][ky][icp][kx][h][ob][ic8][oc8]
        w = Wq.reshape(ocp, 2, 8, icp, 2, 8, 3, 3).transpose(0, 6, 3, 7, 4, 1, 5, 2)
    else:          # → [ocp][icp][h][ob][ic8][oc8]
        w = Wq.reshape(ocp, 2, 8, icp, 2, 8).transpose(0, 3, 4, 1, 5, 2)
    buf = np.zeros(e_w, np.int8)
    nxc = tl.wt // CW
    buf[:32] = np.array([l.icp, l.ocp, tl.wt, l.s, l.k, shift, nxc, tl.plane], np.int32).view(np.int8)
    for c, t, b in tl.cores():
        rows = tl.band_rows[b]
        wvalid = min(max(l.wo - t * tl.wt, 0), tl.wt)
        per = np.array([rows, int(l.s == 1), int(l.s == 2), wvalid, l.ocp * nxc, 0, 0, 0], np.uint16)
        buf[32 + 16 * c: 48 + 16 * c] = per.view(np.int8)
    buf[HDR:HDR + l.n_w] = np.ascontiguousarray(w).reshape(-1)
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
        shift = min(range(3, 14), key=lambda s: abs(requant(acc, s).mean() - 24))
        params.append((Wq, shift))
        cur = requant(acc, shift)
    return x, params, cur


class C16Chain:
    def __init__(self, layers, params, reps=1):
        self.layers, self.reps = layers, reps
        self.P = P = Plan(layers)
        self.spec = spec_str(layers)
        self.t_act = iron.zeros(P.size["act"], dtype=np.int8, device="npu")
        self.t_w = iron.tensor(np.concatenate([pack_weights(l, tl, Wq, s, P.e_w)
                                               for l, tl, (Wq, s) in zip(layers, P.tilings, params)]),
                               dtype=np.int8, device="npu")
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
    return dict(spec=spec, n=len(layers), max_diff=d, med=walls[len(walls) // 2] * 1e3, best=walls[0] * 1e3,
                cpu=float(np.mean(cpus)) * 1e3, jit=jit_s, gmac=sum(l.macs for l in layers) / 1e9,
                stats=f"末层 均值 {y.mean():.1f} 零 {(y == 0).mean() * 100:.0f}%")


def report(r):
    print(f"[c16] {r['spec']:<14s} N={r['n']:<3d} 最大差 {r['max_diff']}  中位 {r['med']:.2f} ms  最快 {r['best']:.2f} ms  "
          f"CPU {r['cpu']:.3f} ms/次  ({r['gmac']:.3f} GMAC, 首次 {r['jit']:.1f} s; {r['stats']})",
          file=sys.stderr, flush=True)


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


def main(iters=50):
    ok = True
    for i, spec in enumerate(["e1", "e1,e2", "head", "enc", "low"]):
        r = run_one(spec, iters, verbose=(spec == "enc"))
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
