"""异构层串联实验: 一次 NPU 调用里串不同形状的卷积层 (3×3 s1 / 3×3 s2 / 1×1, 两种分辨率), 只用一个通用内核

  source npu/env.sh && python -m npu.chain_hetero            # 正确性 + 开销拟合 + 纯计算拆分
  python -m npu.chain_hetero --spec A,B,C,D --iters 50       # 只跑一条指定的链

接 npu/chain_spike.py (同形状 N 层, 每层额外开销 ~0.04 ms)。这里回答: 层形状各不相同时还能不能
"一个 runtime sequence + 一个内核"串起来, 每层额外开销是否仍 ≤ 0.1 ms。

做法 = "通用内核 + 运行时层描述符 + 固定大小的块":
  · 内核 npu/kernels/conv_generic.cc 只编一份; 每层的 ICB/OCB/WT/stride/核大小/shift 写在该层权重块的
    头 64 字节里, 内核运行时读。
  · 三路 ObjectFifo 的元素大小对所有层相同:
      输入元素 = [ICB_E][PLANE_E] 字节 (取各层最大值): fill 的 TAP 每个通道块读 PLANE_E 字节、读 ICB_E 块,
               多出来的是邻近像素/下一行的垃圾, 内核不看 (只读真实部分)。DDR 缓冲区尾部留了余量。
      输出元素 = OCB·WT·8 字节: 每层把宽度切条数选成让它相等 (这里全是 2560 = 2 块×160 = 4 块×80)。
               输出不能带垃圾 (drain 会把垃圾写进 DDR), 所以这是"固定块"的硬约束。
      权重元素 = 64 + 最大层权重字节, 各层在主机侧补齐到同样大小。
  · 激活仍是整幅带边布局 [H+2][ICB][PAD+W+PAD][8]; 每种形状两个 scratch 槽乒乓 (不同形状不能共用槽,
    否则上一层写的内部像素会落在下一种形状的"边"上)。
  · core 侧 (MODE, --mode):
      class  (默认) 描述符驱动: 每个 (每 core 行数, stride) 类别在 core 上只有一段静态行循环, 每层从权重头读
             一个 0/1 选择字节当 range_ 的次数, 决定走哪段 → core 代码大小只和类别数有关, 与层数/顺序无关
             (实测 3 个类别 .text 6672 B, 3 层到 85 层不变)。
      static 相邻、行数和 stride 相同的层合成一个 range_ 循环; 代码随层组数增长 (每组 0.7–1.5 KB)。
      dyn    行数也从描述符读: 编译器拒绝 ("cannot statically analyze cyclostatic acquire pattern"),
             滑窗的 acquire/release 不能放进运行时次数的循环 → 行数必须是编译期常量, 所以用 class。
    1×1 层也按 3 行窗口喂 (多读上下两行), 这样所有层同一种 fifo 用法。

层间激活约定与 chain_spike 相同: 内核输出 uint8 (饱和=ReLU), 下一层把同一字节当 int8 读。
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

SRC = str(Path(__file__).resolve().parent / "kernels" / "conv_generic.cc")
PAD, HDR = 4, 64
SEL_OFF, ROWS_OFF = 32, 48     # 描述符里给 core 读的字节 (前 24 字节给内核)
N_TW, N_BANDS = 2, 2           # 每层都是 宽 2 条 × 高 2 段 = 4 个 core
IN_DEPTH = 4                   # 输入 fifo 深度
GROUP = 1                      # 0 = core 上每层单独展开 (调试用)
MODE = "class"                 # core 循环: static = 按层组展开 / class = 描述符选类别 / dyn = 行数也读描述符


@dataclass(frozen=True)
class Layer:
    ic: int
    oc: int
    k: int        # 1 或 3
    s: int        # 1 或 2 (只有 3×3 支持 2)
    h: int        # 输入高
    w: int        # 输入宽

    @property
    def ho(self): return self.h // self.s
    @property
    def wo(self): return self.w // self.s
    @property
    def icb(self): return self.ic // 8
    @property
    def ocb(self): return self.oc // 8
    @property
    def wt(self): return self.wo // N_TW          # 每个 core 的输出宽度
    @property
    def rows(self): return self.ho // N_BANDS     # 每个 core 的输出行数
    @property
    def in_rows(self):       # fill 发的行数: s1 = 滑窗 (R+2 行, 相邻窗口共享 2 行);
        # s2 = 每个输出行单独发 3 行 (共享的 1 行重复发), 见 core() 里关于 ObjectFifo 下降 bug 的说明
        return self.rows + 2 if self.s == 1 else 3 * self.rows
    @property
    def in_span(self): return (self.rows - 1) * self.s + 3    # 实际覆盖的输入行数
    @property
    def plane_e(self): return (self.s * self.wt + 2 * PAD) * 8   # 每块真实需要的字节
    @property
    def n_w(self): return self.k * self.k * self.ic * self.oc
    @property
    def in_shape(self): return (self.ic, self.h, self.w)
    @property
    def out_shape(self): return (self.oc, self.ho, self.wo)
    @property
    def macs(self): return self.ho * self.wo * self.oc * self.ic * self.k * self.k


# 实验用的四种层 (RVM 编码器里都有这几类)
LAYERS = {
    "A": Layer(16, 16, 3, 1, 180, 320),   # 3×3 s1 16→16 @180×320 (同 chain_spike)
    "B": Layer(16, 32, 3, 2, 180, 320),   # 3×3 s2 16→32 → 90×160
    "C": Layer(32, 32, 1, 1, 90, 160),    # 1×1 32→32 @90×160
    "D": Layer(32, 32, 3, 1, 90, 160),    # 3×3 s1 32→32 @90×160
}


def parse_spec(spec: str) -> list[Layer]:
    return [LAYERS[c] for c in spec.split(",") if c]


def spec_str(layers) -> str:
    return ";".join(f"{l.ic},{l.oc},{l.k},{l.s},{l.h},{l.w}" for l in layers)


def from_spec_str(s: str) -> list[Layer]:
    return [Layer(*map(int, p.split(","))) for p in s.split(";")]


def layout_dims(shape):
    c, h, w = shape
    plane = (w + 2 * PAD) * 8
    full_row = (c // 8) * plane
    return plane, full_row, (h + 2) * full_row


class Plan:
    """主机和设计共用的静态规划: 元素大小、各层的输入/输出位置、缓冲区大小"""

    def __init__(self, layers: list[Layer]):
        for a, b in zip(layers, layers[1:]):
            assert a.out_shape == b.in_shape, f"层形状接不上: {a} → {b}"
        for l in layers:
            assert l.wo % (8 * N_TW) == 0 and l.ho % N_BANDS == 0 and l.ic % 8 == 0 and l.oc % 16 == 0
            assert l.k == 3 or l.s == 1
        self.layers = layers
        self.icb_e = max(l.icb for l in layers)
        self.plane_e = max(l.plane_e for l in layers)
        self.e_in = self.icb_e * self.plane_e
        outs = {l.ocb * l.wt * 8 for l in layers}
        assert len(outs) == 1, f"各层输出元素大小必须相同, 现在是 {outs}"
        self.e_out = outs.pop()
        self.e_w = HDR + max(l.n_w for l in layers)
        # scratch: 每种中间形状两个槽
        self.slot_off = {}
        off = 0
        for l in layers[:-1]:
            if l.out_shape not in self.slot_off:
                n = layout_dims(l.out_shape)[2]
                self.slot_off[l.out_shape] = (off, off + n)
                off += 2 * n
        # 每层的 (src, src_off) / (dst, dst_off); src/dst ∈ {"act","s","out"}
        toggle = {}
        self.io = []
        prev = ("act", 0)
        for k, l in enumerate(layers):
            if k == len(layers) - 1:
                dst = ("out", 0)
            else:
                t = toggle.get(l.out_shape, 0)
                toggle[l.out_shape] = t ^ 1
                dst = ("s", self.slot_off[l.out_shape][t])
            self.io.append((prev, dst))
            prev = dst
        # 缓冲区大小: 布局本身 + fill 多读的尾巴
        ends = {"act": layout_dims(layers[0].in_shape)[2], "s": max(off, 1),
                "out": layout_dims(layers[-1].out_shape)[2]}
        for l, ((src, soff), _) in zip(layers, self.io):
            plane, full_row, _ = layout_dims(l.in_shape)
            for t, b in self.cores():
                o = self.fill_off(l, soff, t, b)
                end = o + (l.in_span - 1) * full_row + (self.icb_e - 1) * plane + self.plane_e
                ends[src] = max(ends[src], end)
        self.size = {k: (v + 63) // 64 * 64 for k, v in ends.items()}

    @staticmethod
    def cores():
        return [(t, b) for t in range(N_TW) for b in range(N_BANDS)]

    @staticmethod
    def fill_off(l: Layer, base: int, t: int, b: int) -> int:
        _, full_row, _ = layout_dims(l.in_shape)
        return base + l.s * l.rows * b * full_row + t * l.s * l.wt * 8

    @staticmethod
    def drain_off(l: Layer, base: int, t: int, b: int) -> int:
        _, full_row, _ = layout_dims(l.out_shape)
        return base + (1 + l.rows * b) * full_row + (PAD + t * l.wt) * 8


def layer_classes(layers):
    """core 上的代码段类别 = 链里出现过的 (每 core 行数, stride), 排好序"""
    return sorted({(l.rows, l.s) for l in layers})


def _hdr_byte(buf, off):
    """core 里读权重块的一个字节 (0..127) → index, 用作 range_ 的运行时次数"""
    from aie.dialects import arith, memref
    from aie.ir import IndexType
    v = memref.load(buf, [arith.constant(IndexType.get(), off)])
    return arith.index_cast(IndexType.get(), v)


def _hdr_u16(buf, off):
    from aie.dialects import arith, memref
    from aie.ir import IndexType, IntegerType
    i32 = IntegerType.get_signless(32)
    lo = arith.extui(i32, memref.load(buf, [arith.constant(IndexType.get(), off)]))
    hi = arith.extui(i32, memref.load(buf, [arith.constant(IndexType.get(), off + 1)]))
    v = arith.addi(lo, arith.shli(hi, arith.constant(i32, 8)))
    return arith.index_cast(IndexType.get(), v)


def _tap(n_total, off, sizes, strides):
    from aie.helpers.taplib import TensorAccessPattern
    return TensorAccessPattern((1, n_total), off, sizes, strides)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def conv_hetero(act: In, wts: In, scratch: InOut, out: Out, *, spec: CompileTime[str],
                reps: CompileTime[int] = 1, group: CompileTime[int] = 1,
                mode: CompileTime[str] = "static"):
    layers = from_spec_str(spec)
    P = Plan(layers)
    n = len(layers)
    i8, u8 = np.dtype[np.int8], np.dtype[np.uint8]
    in_ty = np.ndarray[(P.e_in,), i8]
    out_ty = np.ndarray[(P.e_out,), u8]
    w_ty = np.ndarray[(P.e_w,), i8]
    kern = ExternalFunction("convGenericLine", source_file=SRC,
                            arg_types=[in_ty, in_ty, in_ty, w_ty, out_ty],
                            include_dirs=[KERNEL_INC], compile_flags=[f"-DPLANE_E={P.plane_e}"])

    # 相邻、(行数, stride) 相同的层合成一组 → core 上一个 range_ 循环
    groups = []
    for l in layers:
        key = (l.rows, l.s)
        if group and groups and groups[-1][0] == key:
            groups[-1][1] += 1
        else:
            groups.append([key, 1])

    # stride 2 不用 acquire 3 / release 2 的滑窗: IRON 1.3.4 的 ObjectFifo 下降在这种循环前面还有别的
    # range_(≥2) 循环时 (如 A,A,B), 不剥第一次迭代, 每次迭代都按"再取 3 个"上锁 → 多等 1 个对象 → 挂死
    # (ERT_CMD_STATE_TIMEOUT)。改成每行由 DMA 发完整 3 行、core acquire 3 / release 3, 循环边界持有数恒为 0。
    classes = layer_classes(layers)

    def row_block(f_in, f_out, k, wv, rows, s):
        """一层的行循环; 进出时 f_in 持有数都是 0。rows 可以是 int 或运行时 Value"""
        for _ in range_(rows):
            win = f_in.acquire(3)
            o = f_out.acquire(1)
            for _ in range(reps):        # reps>1 只用于测纯计算时间
                k(win[0], win[1], win[2], wv, o)
            f_in.release(1 if s == 1 else 3)
            f_out.release(1)
        if s == 1:
            f_in.release(2)

    def core_class(f_in, f_w, f_out, k):
        # 描述符驱动: core 代码里每个 (行数, stride) 类别只有一段, 每层读选择字节决定走哪段
        # (range_(0 或 1)); 代码大小只和类别数有关, 和层数、层的顺序无关。
        for _ in range_(n):
            wv = f_w.acquire(1)
            for c, (rows, s) in enumerate(classes):
                for _ in range_(_hdr_byte(wv, SEL_OFF + c)):
                    row_block(f_in, f_out, k, wv, rows, s)
            f_w.release(1)

    def core_dyn(f_in, f_w, f_out, k):
        # 更激进: 行数也从描述符读, 只按 stride 分两段
        for _ in range_(n):
            wv = f_w.acquire(1)
            rows = _hdr_u16(wv, ROWS_OFF)
            for s in (1, 2):
                for _ in range_(_hdr_byte(wv, SEL_OFF + 8 + s)):
                    row_block(f_in, f_out, k, wv, rows, s)
            f_w.release(1)

    def core(f_in, f_w, f_out, k):
        for (rows, s), cnt in groups:
            for _ in range_(cnt):
                wv = f_w.acquire(1)
                for _ in range_(rows):
                    win = f_in.acquire(3)
                    o = f_out.acquire(1)
                    for _ in range(reps):        # reps>1 只用于测纯计算时间
                        k(win[0], win[1], win[2], wv, o)
                    f_in.release(1 if s == 1 else 3)
                    f_out.release(1)
                if s == 1:
                    f_in.release(2)
                f_w.release(1)

    fifos, workers = [], []
    for t, b in P.cores():
        f_in = ObjectFifo(in_ty, depth=IN_DEPTH, name=f"hx{t}_{b}i")
        f_w = ObjectFifo(w_ty, depth=1, name=f"hx{t}_{b}w")
        f_out = ObjectFifo(out_ty, depth=2, name=f"hx{t}_{b}o")
        fifos.append((t, b, f_in, f_w, f_out))
        body = {"static": core, "class": core_class, "dyn": core_dyn}[mode]
        workers.append(Worker(body, [f_in.cons(), f_w.cons(), f_out.prod(), kern]))

    rt = Runtime()
    with rt.sequence(np.ndarray[(P.size["act"],), i8], np.ndarray[(n * P.e_w,), i8],
                     np.ndarray[(P.size["s"],), i8], np.ndarray[(P.size["out"],), u8]) as (a, w, s, o):
        bufs = {"act": a, "s": s, "out": o}
        rt.start(*workers)
        for k, l in enumerate(layers):
            (src, soff), (dst, doff) = P.io[k]
            pin, frow_in, _ = layout_dims(l.in_shape)
            pout, frow_out, _ = layout_dims(l.out_shape)
            tg = rt.task_group()
            for t, b, f_in, f_w, _ in fifos:
                rt.fill(f_w.prod(), w, tap=_tap(n * P.e_w, k * P.e_w, [1, 1, 1, P.e_w], [0, 0, 0, 1]),
                        task_group=tg)
                if l.s == 1:
                    sizes, strides = [1, l.in_rows, P.icb_e, P.plane_e], [0, frow_in, pin, 1]
                else:      # 输出行 y ← 输入行 2y-1, 2y, 2y+1
                    sizes, strides = [l.rows, 3, P.icb_e, P.plane_e], [2 * frow_in, frow_in, pin, 1]
                rt.fill(f_in.prod(), bufs[src], tap=_tap(P.size[src], P.fill_off(l, soff, t, b), sizes, strides),
                        task_group=tg)
            for t, b, _, _, f_out in fifos:
                rt.drain(f_out.cons(), bufs[dst],
                         tap=_tap(P.size[dst], P.drain_off(l, doff, t, b),
                                  [1, l.rows, l.ocb, l.wt * 8], [0, frow_out, pout, 1]),
                         wait=True, task_group=tg)
            rt.finish_task_group(tg)
    return Program(iron.get_current_device(), rt).resolve_program()


# ---------------------------------------------------------------- 主机端

def to_layout(x: np.ndarray, size: int) -> np.ndarray:
    """(C, H, W) → [H+2][CB][PAD+W+PAD][8] (边为零), 再补零到 size 字节"""
    c, h, w = x.shape
    p = np.zeros((c, h + 2, w + 2 * PAD), x.dtype)
    p[:, 1:-1, PAD:PAD + w] = x
    flat = p.reshape(c // 8, 8, h + 2, w + 2 * PAD).transpose(2, 0, 3, 1).reshape(-1)
    buf = np.zeros(size, x.dtype)
    buf[:flat.size] = flat
    return buf


def from_layout(buf: np.ndarray, shape) -> np.ndarray:
    c, h, w = shape
    n = (h + 2) * c * (w + 2 * PAD)
    b = buf[:n].reshape(h + 2, c // 8, w + 2 * PAD, 8).transpose(1, 3, 0, 2).reshape(c, h + 2, w + 2 * PAD)
    return b[:, 1:-1, PAD:PAD + w]


def pack_weights(l: Layer, Wq: np.ndarray, shift: int, e_w: int, classes=()) -> np.ndarray:
    """(OC, IC, k, k) → [描述符 64 B][权重], 补齐到 e_w 字节

    描述符: int32 [ICB, OCB, WT, stride, k, shift] (内核读);
            字节 SEL_OFF+c = 1 若本层属于类别 c (core 读, mode=class);
            字节 SEL_OFF+8+s = 1 若 stride = s, ROWS_OFF 起 u16 = 行数 (core 读, mode=dyn)"""
    ocp, icb = l.oc // 16, l.icb
    if l.k == 3:   # → [ocp][ky][icb][kx][ob][ic8][oc8]
        w = Wq.reshape(ocp, 2, 8, icb, 8, 3, 3).transpose(0, 5, 3, 6, 1, 4, 2)
    else:          # → [ocp][icb][ob][ic8][oc8]
        w = Wq.reshape(ocp, 2, 8, icb, 8).transpose(0, 3, 1, 4, 2)
    hdr = np.zeros(HDR // 4, np.int32)
    hdr[:6] = [l.icb, l.ocb, l.wt, l.s, l.k, shift]
    buf = np.zeros(e_w, np.int8)
    buf[:HDR] = hdr.view(np.int8)
    for c, key in enumerate(classes):
        buf[SEL_OFF + c] = int(key == (l.rows, l.s))
    buf[SEL_OFF + 8 + l.s] = 1
    buf[ROWS_OFF] = l.rows & 0xFF
    buf[ROWS_OFF + 1] = l.rows >> 8
    buf[HDR:HDR + l.n_w] = np.ascontiguousarray(w).reshape(-1)
    return buf


def conv_acc(l: Layer, x_i8: np.ndarray, Wq: np.ndarray) -> np.ndarray:
    """int8 (IC,H,W) → int32 累加 (OC, Ho, Wo); pad = (k-1)/2"""
    p = (l.k - 1) // 2
    xp = np.zeros((l.ic, l.h + 2 * p, l.w + 2 * p), np.int32)
    xp[:, p:p + l.h, p:p + l.w] = x_i8
    acc = np.zeros((l.oc, l.ho * l.wo), np.int32)
    Wi = Wq.astype(np.int32)
    for ky in range(l.k):
        for kx in range(l.k):
            patch = xp[:, ky:ky + l.s * l.ho:l.s, kx:kx + l.s * l.wo:l.s]
            acc += Wi[:, :, ky, kx] @ patch.reshape(l.ic, -1)
    return acc.reshape(l.oc, l.ho, l.wo)


def requant(acc: np.ndarray, shift: int) -> np.ndarray:
    return np.clip((acc + (1 << (shift - 1))) >> shift, 0, 255).astype(np.uint8)


def make_case(layers: list[Layer], seed: int = 0):
    """随机输入 + 随机权重 ±12; 每层 shift 自动选成输出均值最接近 24 (分布不塌缩、不饱和)。
    返回 x, [(W, shift)], 各层参考输出"""
    rng = np.random.default_rng(seed)
    x = rng.integers(-100, 101, layers[0].in_shape, dtype=np.int64).astype(np.int8)
    params, outs, cur = [], [], x
    for l in layers:
        Wq = rng.integers(-12, 13, (l.oc, l.ic, l.k, l.k)).astype(np.int8)
        acc = conv_acc(l, cur.view(np.int8), Wq)
        shift = min(range(3, 12), key=lambda s: abs(requant(acc, s).mean() - 24))
        y = requant(acc, shift)
        params.append((Wq, shift))
        outs.append(y)
        cur = y
    return x, params, outs


class HeteroChain:
    def __init__(self, layers, params, reps=1):
        self.layers, self.reps = layers, reps
        self.P = P = Plan(layers)
        self.spec = spec_str(layers)
        self.t_act = iron.zeros(P.size["act"], dtype=np.int8, device="npu")
        cls = layer_classes(layers)
        self.t_w = iron.tensor(np.concatenate([pack_weights(l, Wq, s, P.e_w, cls)
                                               for l, (Wq, s) in zip(layers, params)]),
                               dtype=np.int8, device="npu")
        self.t_s = iron.zeros(P.size["s"], dtype=np.int8, device="npu")      # 边永远为零
        self.t_out = iron.zeros(P.size["out"], dtype=np.uint8, device="npu")

    def set_input(self, x):
        self.t_act.data[:] = to_layout(x, self.P.size["act"])
        self.t_act._sync_to_device()

    def call(self):
        conv_hetero(self.t_act, self.t_w, self.t_s, self.t_out, spec=self.spec, reps=self.reps,
                    group=GROUP, mode=MODE)

    def result(self):
        return from_layout(self.t_out.numpy(), self.layers[-1].out_shape)


def _cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def run_one(spec: str, iters: int = 50, reps: int = 1, seed: int = 0):
    layers = parse_spec(spec)
    x, params, outs = make_case(layers, seed)
    y = outs[-1]
    ch = HeteroChain(layers, params, reps)
    ch.set_input(x)
    t0 = time.perf_counter()
    ch.call()
    jit_s = time.perf_counter() - t0
    d = int(np.abs(ch.result().astype(int) - y.astype(int)).max())
    walls, cpus = [], []
    for _ in range(iters):
        c0, t0 = _cpu(), time.perf_counter()
        ch.call()
        walls.append(time.perf_counter() - t0)
        cpus.append(_cpu() - c0)
    d = max(d, int(np.abs(ch.result().astype(int) - y.astype(int)).max()))
    walls.sort()
    stats = f"末层 均值 {y.mean():.1f} 零 {(y == 0).mean() * 100:.0f}%; shift {[s for _, s in params]}"
    return dict(spec=spec, n=len(layers), max_diff=d, med=walls[len(walls) // 2] * 1e3,
                best=walls[0] * 1e3, cpu=float(np.mean(cpus)) * 1e3, jit=jit_s, stats=stats,
                gmac=sum(l.macs for l in layers) / 1e9)


def report(r):
    print(f"[hetero] {r['spec']:<28s} N={r['n']:<3d} 最大差 {r['max_diff']}  中位 {r['med']:.2f} ms  "
          f"最快 {r['best']:.2f} ms  CPU {r['cpu']:.3f} ms/次  ({r['gmac']:.2f} GMAC, 首次 {r['jit']:.1f} s; "
          f"{r['stats']})", file=sys.stderr, flush=True)


def fit(res, label):
    ns = np.array([r["n"] for r in res], float)
    out = {}
    for key in ("med", "best"):
        slope, icpt = np.polyfit(ns, np.array([r[key] for r in res]), 1)
        out[key] = (slope, icpt)
        print(f"[hetero] 拟合 {label} ({key}): T = {icpt:.3f} + {slope:.3f}·N ms", file=sys.stderr)
    return out


def compute_split(spec, reps=(1, 2, 3), iters=50):
    """内核每行重复 r 次: T(r) 对 r 的斜率 = 整条链的纯计算时间"""
    res = [run_one(spec, iters, r) for r in reps]
    for r, x in zip(reps, res):
        print(f"[hetero] {spec} reps={r} 最大差 {x['max_diff']} 中位 {x['med']:.2f} ms 最快 {x['best']:.2f} ms",
              file=sys.stderr, flush=True)
    out = {}
    for key in ("med", "best"):
        slope, icpt = np.polyfit(np.array(reps, float), np.array([x[key] for x in res]), 1)
        out[key] = (slope, icpt)
        print(f"[hetero] ({key}) {spec}: 纯计算 {slope:.3f} ms;  T(reps=1) - 计算 = {icpt:.3f} ms "
              f"(= 固定 + Σ每层额外)", file=sys.stderr)
    return out


def main(iters=50):
    ok = True
    # 1) 正确性: 四种层的各种排列
    for spec in ["A", "B", "A,B", "B,C", "B,D", "A,B,C,D", "A,A,B,D,C,D"]:
        r = run_one(spec, iters)
        report(r)
        ok &= r["max_diff"] == 0
    # 2) 开销: A,B 后面接 K 对 (C, D), 斜率 = 一对 C+D 的 (计算 + 额外)
    res = []
    for k in (0, 1, 2, 4, 8):
        r = run_one("A,B" + ",C,D" * k, iters)
        report(r)
        ok &= r["max_diff"] == 0
        res.append(r)
    fit(res, "A,B,(C,D)×K")
    # 3) 同形状对照: 通用内核跑 A×N, 和 chain_spike 的专用内核 (0.236 ms/层) 比
    res = []
    for k in (1, 2, 5, 10):
        r = run_one(",".join("A" * k), iters)
        report(r)
        ok &= r["max_diff"] == 0
        res.append(r)
    fit(res, "A×N")
    return SelfTestResult(ok, 0 if ok else 1, res[-1]["med"])


if __name__ == "__main__":
    def arg(name, default):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
    iters = int(arg("--iters", 50))
    MODE = arg("--mode", MODE)
    if "--compute" in sys.argv:
        compute_split(arg("--compute", "A,B,C,D"), iters=iters)
        sys.exit(0)
    if "--spec" in sys.argv:
        r = run_one(arg("--spec", "A,B,C,D"), iters)
        report(r)
        sys.exit(0 if r["max_diff"] == 0 else 1)
    sys.exit(0 if main(iters) else 1)
