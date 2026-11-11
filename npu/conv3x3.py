"""int8 3×3 卷积 + ReLU 的 NPU 实现 (自写内核 npu/kernels/conv3x3.cc, 替代上游 conv2dk3)

  source npu/env.sh && python -m npu.conv3x3 [--iters 50]

拿 RVM 的 Conv_290 (35→16, 3×3, 180×320) 真实数据做验收: 与 QConv.reference (int8 整数
参考, npu/rvm/conv_proto.py) 逐字节比对, 并与 ONNX Runtime fp32 (CPU) 比时间。

切分: 图像按宽度切成 N_TW 条 (每条 WT 像素, 左右各带 PAD 像素的邻居), 按高度切成 N_BANDS 段;
每个 core 处理一条 × 一段, 行走 3 行滑动窗口的 ObjectFifo (acquire 3 / release 1)。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction

from .fused import KERNEL_INC
from .harness import SelfTestResult

SRC = str(Path(__file__).resolve().parent / "kernels" / "conv3x3.cc")
PAD = 4
OCB = 2


def _tap(n_total, off, size):
    from aie.helpers.taplib import TensorAccessPattern
    return TensorAccessPattern((1, n_total), off, [1, 1, 1, size], [0, 0, 0, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def conv3x3_relu(act: In, wts: In, out: Out, *, height: CompileTime[int], wt: CompileTime[int],
                 n_tw: CompileTime[int], n_bands: CompileTime[int], icb: CompileTime[int],
                 shift: CompileTime[int]):
    rows = height // n_bands                         # 每个 core 的输出行数
    in_row = icb * (wt + 2 * PAD) * 8
    out_row = OCB * wt * 8
    n_w = 9 * icb * 8 * OCB * 8
    i8, u8 = np.dtype[np.int8], np.dtype[np.uint8]
    in_ty, out_ty, w_ty = np.ndarray[(in_row,), i8], np.ndarray[(out_row,), u8], np.ndarray[(n_w,), i8]
    kern = ExternalFunction("conv3x3I8Line", source_file=SRC,
                            arg_types=[in_ty, in_ty, in_ty, w_ty, out_ty, np.int32],
                            include_dirs=[KERNEL_INC], compile_flags=[f"-DWT={wt}", f"-DICB={icb}"])

    def core(f_in, f_w, f_out, k):
        wv = f_w.acquire(1)
        for _ in range_(rows):
            win = f_in.acquire(3)
            o = f_out.acquire(1)
            k(win[0], win[1], win[2], wv, o, shift)
            f_in.release(1)
            f_out.release(1)
        f_in.release(2)
        f_w.release(1)

    fifos, workers = [], []
    for t in range(n_tw):
        for b in range(n_bands):
            f_in = ObjectFifo(in_ty, depth=4, name=f"cv{t}_{b}i")
            f_w = ObjectFifo(w_ty, depth=1, name=f"cv{t}_{b}w")
            f_out = ObjectFifo(out_ty, depth=2, name=f"cv{t}_{b}o")
            fifos.append((t, b, f_in, f_w, f_out))
            workers.append(Worker(core, [f_in.cons(), f_w.cons(), f_out.prod(), kern]))

    n_act = n_tw * (height + 2) * in_row
    n_out = n_tw * height * out_row
    rt = Runtime()
    with rt.sequence(np.ndarray[(n_act,), i8], w_ty, np.ndarray[(n_out,), u8]) as (a, w, o):
        rt.start(*workers)
        tg = rt.task_group()
        for t, b, f_in, f_w, _ in fifos:
            rt.fill(f_w.prod(), w, task_group=tg)
            rt.fill(f_in.prod(), a, tap=_tap(n_act, (t * (height + 2) + b * rows) * in_row, (rows + 2) * in_row),
                    task_group=tg)
        for t, b, _, _, f_out in fifos:
            rt.drain(f_out.cons(), o, tap=_tap(n_out, (t * height + b * rows) * out_row, rows * out_row),
                     wait=True, task_group=tg)
        rt.finish_task_group(tg)
    return Program(iron.get_current_device(), rt).resolve_program()


# ---------------------------------------------------------------- 主机端打包

def pack_weights(Wq: np.ndarray) -> np.ndarray:
    """(OC=16, ICP, 3, 3) → [ky][icb][kx][ocb][ic8][oc8] int8"""
    oc, icp = Wq.shape[:2]
    w = Wq.reshape(oc // 8, 8, icp // 8, 8, 3, 3)            # [ocb][oc8][icb][ic8][ky][kx]
    w = w.transpose(4, 2, 5, 0, 3, 1)                        # [ky][icb][kx][ocb][ic8][oc8]
    return np.ascontiguousarray(w).reshape(-1).astype(np.int8)


def pack_act(xq: np.ndarray, wt: int, n_tw: int) -> np.ndarray:
    """(ICP, H, W) int8 → (n_tw, H+2, ICB, WT+2·PAD, 8): 上下各补一行零, 每条左右带 PAD 列邻居/零"""
    icp, h, w = xq.shape
    padded = np.zeros((icp, h + 2, n_tw * wt + 2 * PAD), np.int8)
    padded[:, 1:-1, PAD:PAD + w] = xq
    tiles = np.stack([padded[:, :, t * wt: t * wt + wt + 2 * PAD] for t in range(n_tw)])  # (T, ICP, H+2, WT+2P)
    t = tiles.reshape(n_tw, icp // 8, 8, h + 2, wt + 2 * PAD).transpose(0, 3, 1, 4, 2)
    return np.ascontiguousarray(t)


def unpack_out(o: np.ndarray, h: int, w: int, wt: int, n_tw: int) -> np.ndarray:
    """(n_tw, H, OCB, WT, 8) → (OC, H, W)"""
    o = o.reshape(n_tw, h, OCB, wt, 8).transpose(2, 4, 1, 0, 3).reshape(OCB * 8, h, n_tw * wt)
    return o[:, :, :w]


class Conv3x3:
    def __init__(self, q, h: int, w: int, n_tw: int = 2, n_bands: int = 2):
        self.q, self.h, self.w = q, h, w
        self.n_tw, self.n_bands = n_tw, n_bands
        self.wt = -(-w // (8 * n_tw)) * 8
        self.icb = q.icp // 8
        assert q.oc == 16 and h % n_bands == 0
        self.t_act = iron.zeros(n_tw * (h + 2) * self.icb * (self.wt + 2 * PAD) * 8, dtype=np.int8, device="npu")
        self.t_w = iron.tensor(pack_weights(q.Wq), dtype=np.int8, device="npu")
        self.t_out = iron.zeros(n_tw * h * OCB * self.wt * 8, dtype=np.uint8, device="npu")

    def set_input(self, xq: np.ndarray) -> None:
        self.t_act.data[:] = pack_act(xq, self.wt, self.n_tw).reshape(-1)

    def run(self) -> np.ndarray:
        self.t_act._sync_to_device()
        conv3x3_relu(self.t_act, self.t_w, self.t_out, height=self.h, wt=self.wt, n_tw=self.n_tw,
                     n_bands=self.n_bands, icb=self.icb, shift=self.q.shift)
        return unpack_out(self.t_out.numpy(), self.h, self.w, self.wt, self.n_tw)


def selftest(n_tw=2, n_bands=2, iters=30):
    from .rvm.conv_proto import QConv, float_reference, load_case
    x, W, B, _, calib = load_case()
    ic, h, w = x.shape
    yf = float_reference(x, W, B)
    q = QConv(W, B, x_absmax=float(np.abs(calib).max()), y_max=float(np.percentile(yf, 99.99)))
    xq = q.quant_x(x)
    exp = q.reference(xq)
    conv = Conv3x3(q, h, w, n_tw, n_bands)
    conv.set_input(xq)
    t0 = time.perf_counter()
    got = conv.run()
    print(f"[conv3x3] 首次调用 (含 JIT 编译) {time.perf_counter() - t0:.1f} s", file=sys.stderr)
    d = np.abs(got.astype(int) - exp.astype(int))
    print(f"[conv3x3] 与 int8 参考: 最大差 {d.max()}  不一致 {(d > 0).mean() * 100:.3f}%", file=sys.stderr)
    if d.max():
        bad = np.argwhere(d > 0)
        print(f"   例: {[tuple(b) for b in bad[:6]]}", file=sys.stderr)
        for ax, name in ((0, "通道"), (1, "行"), (2, "列")):
            vals = np.unique(bad[:, ax])
            print(f"   出错的{name}: {vals[:20]}{' …' if len(vals) > 20 else ''} (共 {len(vals)})", file=sys.stderr)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        conv.run()
        times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2]
    macs = W.size * h * w
    print(f"[conv3x3] {ic}→16 3×3 @ {h}×{w}, {n_tw}×{n_bands}={n_tw * n_bands} core: 中位 {med * 1000:.2f} ms, "
          f"最快 {times[0] * 1000:.2f} ms ({macs / med / 1e9:.0f} GMAC/s, 含同步)", file=sys.stderr)
    return SelfTestResult(d.max() == 0, d.max(), med * 1000)


if __name__ == "__main__":
    def arg(name, default):
        return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
    sys.exit(0 if selftest(arg("--tw", 2), arg("--bands", 2), arg("--iters", 30)) else 1)
