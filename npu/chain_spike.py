"""串联验证实验: 一次 NPU 调用 (一个 runtime sequence) 里依次跑 N 层 16→16 的 int8 3×3 卷积 + ReLU

  source npu/env.sh && python -m npu.chain_spike [--layers 1,2,5,10] [--iters 30] [--tw 2] [--bands 2]

目的: 量化"多层串联"的每层额外开销 (DMA 重新编程 + 层间屏障 + 流水线排空/灌满), 判断 RVM 整网
(85 个卷积) 放 NPU 是否可行。内核复用 npu/kernels/conv3x3.cc (WT=宽/n_tw, ICB=2)。

数据流 (全在 DDR, 不经主机):
  所有激活用同一种"整幅宽度、带边"的布局 [H+2][ICB][PAD + W + PAD][8] int8,
  上下各一行零、左右各 PAD 像素零 —— 这些零永远不被写, 所以层间不需要主机重新打包。
  · fill: 条带 t / 段 b 用 TAP 直接从整幅布局里切出 (rows+2) 行 × [ICB][WT+2·PAD][8]
          (相邻条带的邻居列自然就在里面)
  · drain: 输出行 [OCB][WT][8] 用 TAP 写回下一层输入布局的内部 (跳过边)
  · 层 k: act/scratch[k-1 槽] → scratch[k 槽]/out, 两个 scratch 槽乒乓;
          每层一个 task group, 等全部 drain 完才开始下一层 (层间数据依赖 = 屏障)。
  · worker 在 core 上循环 N 层, 每层取一次权重 (权重各层一段, 由同一路 DMA 按偏移喂)。

层间激活: 内核输出 uint8 (饱和 = ReLU), 下一层把同一字节当 int8 读 (>127 的值变负)。
这只是实验的约定 (真实网络会按 per-tensor 零点处理), 参考实现照做, 逐字节比对仍然严格。
"""
from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, InOut, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction

from .fused import KERNEL_INC
from .harness import SelfTestResult

SRC = str(Path(__file__).resolve().parent / "kernels" / "conv3x3.cc")
PAD, C, CB = 4, 16, 2          # 左右边像素; 通道数 (输入 = 输出); 通道块数
H, W = 180, 320                # Conv_292 规模
SHIFT = 6


def _tap(n_total, off, sizes, strides):
    from aie.helpers.taplib import TensorAccessPattern
    return TensorAccessPattern((1, n_total), off, sizes, strides)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def conv_chain(act: In, wts: In, scratch: InOut, out: Out, *, n_layers: CompileTime[int],
               n_tw: CompileTime[int], n_bands: CompileTime[int], shift: CompileTime[int],
               reps: CompileTime[int] = 1):
    wt = W // n_tw
    rows = H // n_bands
    plane = (W + 2 * PAD) * 8                    # 整幅布局里一个通道块一行的字节数
    full_row = CB * plane                        # 整幅布局一行
    n_act = (H + 2) * full_row
    in_row = CB * (wt + 2 * PAD) * 8             # 每个 core 看到的一行
    out_row = CB * wt * 8
    n_w = 9 * CB * 8 * CB * 8
    i8, u8 = np.dtype[np.int8], np.dtype[np.uint8]
    in_ty, out_ty, w_ty = np.ndarray[(in_row,), i8], np.ndarray[(out_row,), u8], np.ndarray[(n_w,), i8]
    kern = ExternalFunction("conv3x3I8Line", source_file=SRC,
                            arg_types=[in_ty, in_ty, in_ty, w_ty, out_ty, np.int32],
                            include_dirs=[KERNEL_INC], compile_flags=[f"-DWT={wt}", f"-DICB={CB}"])

    def core(f_in, f_w, f_out, k):
        for _ in range_(n_layers):
            wv = f_w.acquire(1)
            for _ in range_(rows):
                win = f_in.acquire(3)
                o = f_out.acquire(1)
                for _ in range(reps):        # reps>1 只用于测纯计算时间 (结果相同, 重复写)
                    k(win[0], win[1], win[2], wv, o, shift)
                f_in.release(1)
                f_out.release(1)
            f_in.release(2)
            f_w.release(1)

    fifos, workers = [], []
    for t in range(n_tw):
        for b in range(n_bands):
            f_in = ObjectFifo(in_ty, depth=4, name=f"ch{t}_{b}i")
            f_w = ObjectFifo(w_ty, depth=1, name=f"ch{t}_{b}w")
            f_out = ObjectFifo(out_ty, depth=2, name=f"ch{t}_{b}o")
            fifos.append((t, b, f_in, f_w, f_out))
            workers.append(Worker(core, [f_in.cons(), f_w.cons(), f_out.prod(), kern]))

    def fill_tap(slot, t, b):    # (rows+2) 行 × CB 块 × (WT+2·PAD)·8 字节, 从整幅布局里切
        return _tap(2 * n_act, slot * n_act + b * rows * full_row + t * wt * 8,
                    [1, rows + 2, CB, (wt + 2 * PAD) * 8], [0, full_row, plane, 1])

    def drain_tap(slot, t, b):   # rows 行 × CB 块 × WT·8 字节, 写进整幅布局内部
        return _tap(2 * n_act, slot * n_act + (1 + b * rows) * full_row + (PAD + t * wt) * 8,
                    [1, rows, CB, wt * 8], [0, full_row, plane, 1])

    rt = Runtime()
    with rt.sequence(np.ndarray[(n_act,), i8], np.ndarray[(n_layers * n_w,), i8],
                     np.ndarray[(2 * n_act,), i8], np.ndarray[(n_act,), u8]) as (a, w, s, o):
        rt.start(*workers)
        for k in range(n_layers):
            src, src_slot = (a, 0) if k == 0 else (s, (k - 1) % 2)
            dst, dst_slot = (o, 0) if k == n_layers - 1 else (s, k % 2)
            tg = rt.task_group()
            for t, b, f_in, f_w, _ in fifos:
                rt.fill(f_w.prod(), w, tap=_tap(n_layers * n_w, k * n_w, [1, 1, 1, n_w], [0, 0, 0, 1]),
                        task_group=tg)
                rt.fill(f_in.prod(), src, tap=fill_tap(src_slot, t, b), task_group=tg)
            for t, b, _, _, f_out in fifos:
                rt.drain(f_out.cons(), dst, tap=drain_tap(dst_slot, t, b), wait=True, task_group=tg)
            rt.finish_task_group(tg)
    return Program(iron.get_current_device(), rt).resolve_program()


# ---------------------------------------------------------------- 主机端

def to_layout(x: np.ndarray) -> np.ndarray:
    """(C, H, W) int8/uint8 → [H+2][CB][PAD+W+PAD][8] (边为零)"""
    p = np.zeros((C, H + 2, W + 2 * PAD), x.dtype)
    p[:, 1:-1, PAD:PAD + W] = x
    return np.ascontiguousarray(p.reshape(CB, 8, H + 2, W + 2 * PAD).transpose(2, 0, 3, 1))


def from_layout(buf: np.ndarray) -> np.ndarray:
    b = buf.reshape(H + 2, CB, W + 2 * PAD, 8).transpose(1, 3, 0, 2).reshape(C, H + 2, W + 2 * PAD)
    return b[:, 1:-1, PAD:PAD + W]


def pack_weights(Wq: np.ndarray) -> np.ndarray:
    """(16, 16, 3, 3) → [ky][icb][kx][ocb][ic8][oc8]"""
    w = Wq.reshape(CB, 8, CB, 8, 3, 3).transpose(4, 2, 5, 0, 3, 1)
    return np.ascontiguousarray(w).reshape(-1).astype(np.int8)


def ref_layer(x_i8: np.ndarray, Wq: np.ndarray, shift: int) -> np.ndarray:
    """int8 (C,H,W) → uint8 (C,H,W) = sat_u8((Σ x·w + 2^(s-1)) >> s)"""
    xp = np.zeros((C, H + 2, W + 2), np.int32)
    xp[:, 1:-1, 1:-1] = x_i8
    acc = np.zeros((C, H * W), np.int32)
    Wi = Wq.astype(np.int32)
    for ky in range(3):
        for kx in range(3):
            acc += Wi[:, :, ky, kx] @ xp[:, ky:ky + H, kx:kx + W].reshape(C, -1)
    return np.clip((acc + (1 << (shift - 1))) >> shift, 0, 255).astype(np.uint8).reshape(C, H, W)


def make_case(n_layers: int, seed: int = 0):
    """随机输入 + n 层随机权重; 权重幅度选成每层输出分布大致稳定 (不全零、不全饱和)"""
    rng = np.random.default_rng(seed)
    x = rng.integers(-100, 101, (C, H, W), dtype=np.int64).astype(np.int8)
    Ws = [rng.integers(-12, 13, (C, C, 3, 3)).astype(np.int8) for _ in range(n_layers)]
    return x, Ws


class ConvChain:
    def __init__(self, Ws, n_tw=2, n_bands=2, shift=SHIFT, reps=1):
        assert W % (8 * n_tw) == 0 and H % n_bands == 0
        self.n, self.n_tw, self.n_bands, self.shift, self.reps = len(Ws), n_tw, n_bands, shift, reps
        n_act = (H + 2) * CB * (W + 2 * PAD) * 8
        self.t_act = iron.zeros(n_act, dtype=np.int8, device="npu")
        self.t_w = iron.tensor(np.concatenate([pack_weights(w) for w in Ws]), dtype=np.int8, device="npu")
        self.t_s = iron.zeros(2 * n_act, dtype=np.int8, device="npu")      # 边永远为零
        self.t_out = iron.zeros(n_act, dtype=np.uint8, device="npu")

    def set_input(self, x: np.ndarray) -> None:
        self.t_act.data[:] = to_layout(x).reshape(-1)
        self.t_act._sync_to_device()

    def call(self) -> None:
        conv_chain(self.t_act, self.t_w, self.t_s, self.t_out, n_layers=self.n, n_tw=self.n_tw,
                   n_bands=self.n_bands, shift=self.shift, reps=self.reps)

    def result(self) -> np.ndarray:
        return from_layout(self.t_out.numpy())


def _cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def run_one(n: int, n_tw: int, n_bands: int, iters: int, reps: int = 1):
    x, Ws = make_case(n)
    exp = [x]
    for w in Ws:
        exp.append(ref_layer(exp[-1].view(np.int8), w, SHIFT))
    y = exp[-1]
    chain = ConvChain(Ws, n_tw, n_bands, reps=reps)
    chain.set_input(x)
    t0 = time.perf_counter()
    chain.call()
    jit_s = time.perf_counter() - t0
    got = chain.result()
    d = int(np.abs(got.astype(int) - y.astype(int)).max())
    stats = f"输出 均值 {y.mean():.1f}, 零 {(y == 0).mean() * 100:.0f}%, 255 {(y == 255).mean() * 100:.1f}%"
    walls, cpus = [], []
    for _ in range(iters):
        c0, t0 = _cpu(), time.perf_counter()
        chain.call()
        walls.append(time.perf_counter() - t0)
        cpus.append(_cpu() - c0)
    # 计时后再比一次 (确认多次调用、scratch 乒乓不串)
    d = max(d, int(np.abs(chain.result().astype(int) - y.astype(int)).max()))
    walls.sort()
    return dict(n=n, max_diff=d, med=walls[len(walls) // 2] * 1e3, best=walls[0] * 1e3,
                cpu=float(np.mean(cpus)) * 1e3, jit=jit_s, stats=stats)


def main(layers=(1, 2, 5, 10), n_tw=2, n_bands=2, iters=30):
    res = []
    for n in layers:
        r = run_one(n, n_tw, n_bands, iters)
        res.append(r)
        print(f"[chain] N={n:<3d} {n_tw}×{n_bands} core  最大差 {r['max_diff']}  中位 {r['med']:.2f} ms  "
              f"最快 {r['best']:.2f} ms  主机CPU {r['cpu']:.3f} ms/次  (首次 {r['jit']:.1f} s; {r['stats']})",
              file=sys.stderr, flush=True)
    if len(res) >= 2:
        ns = np.array([r["n"] for r in res], float)
        for key in ("med", "best"):
            ts = np.array([r[key] for r in res])
            slope, icpt = np.polyfit(ns, ts, 1)
            print(f"[chain] 拟合 ({key}): T(N) = {icpt:.3f} + {slope:.3f}·N ms", file=sys.stderr)
    ok = all(r["max_diff"] == 0 for r in res)
    return SelfTestResult(ok, max(r["max_diff"] for r in res), res[-1]["med"])


def compute_split(n=10, reps=(1, 2, 3), n_tw=2, n_bands=2, iters=50):
    """内核每行重复调用 r 次: T(r) 对 r 的斜率 / n = 每层纯计算时间, 截距部分 = DMA/屏障/固定开销"""
    res = [run_one(n, n_tw, n_bands, iters, r) for r in reps]
    for r, x in zip(reps, res):
        print(f"[chain] N={n} reps={r}  最大差 {x['max_diff']}  中位 {x['med']:.2f} ms  最快 {x['best']:.2f} ms",
              file=sys.stderr, flush=True)
    for key in ("med", "best"):
        slope, icpt = np.polyfit(np.array(reps, float), np.array([x[key] for x in res]), 1)
        print(f"[chain] ({key}) 纯计算 {slope / n:.3f} ms/层;  T(reps=1) - N·计算 = {icpt:.3f} ms "
              f"(= 固定 + N·每层额外)", file=sys.stderr)
    return res


if __name__ == "__main__":
    if "--compute" in sys.argv:
        compute_split(int(sys.argv[sys.argv.index("--compute") + 1]))
        sys.exit(0)
    def arg(name, default):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
    ls = tuple(int(v) for v in arg("--layers", "1,2,5,10").split(","))
    sys.exit(0 if main(ls, int(arg("--tw", 2)), int(arg("--bands", 2)), int(arg("--iters", 30))) else 1)
