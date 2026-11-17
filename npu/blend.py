"""NPU 版逐像素 alpha 混合 (IRON / mlir-aie, AIE2 Phoenix)

数据流: 主机把每行打包成 [fg | bg | mask] → Shim DMA → 计算核心 alphaBlendPackedLine → Shim → 主机
用 n_cols 列并行: 图像按行均分给每一列, 每列一个计算核心。

  python -m npu.blend --selftest          # 用随机数据对比 NPU 与 numpy 结果
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.kernel import ExternalFunction

HERE = Path(__file__).resolve().parent
KERNEL = str(HERE / "kernels" / "alpha_blend.cc")
MLIR_AIE = Path(os.environ.get("MLIR_AIE_INSTALL_DIR", "")) if os.environ.get("MLIR_AIE_INSTALL_DIR") \
    else Path(iron.__file__).resolve().parents[2]
KERNEL_INC = str(MLIR_AIE / "include" / "aie_kernels")


@iron.jit
def alpha_blend(packed: In, out: Out, *,
                width: CompileTime[int], height: CompileTime[int], n_cols: CompileTime[int] = 1):
    line = width * 3
    rows_per_col = height // n_cols
    packed_ty = np.ndarray[(line * 3,), np.dtype[np.uint8]]
    line_ty = np.ndarray[(line,), np.dtype[np.uint8]]

    kern = ExternalFunction(
        "alphaBlendPackedLine", source_file=KERNEL,
        arg_types=[packed_ty, line_ty, np.int32], include_dirs=[KERNEL_INC])

    def core(of_in, of_out, k):
        e_in = of_in.acquire(1)
        e_out = of_out.acquire(1)
        k(e_in, e_out, line)
        of_in.release(1)
        of_out.release(1)

    ins, outs, workers = [], [], []
    for c in range(n_cols):
        of_in = ObjectFifo(packed_ty, depth=2, name=f"in{c}")
        of_out = ObjectFifo(line_ty, depth=2, name=f"out{c}")
        ins.append(of_in)
        outs.append(of_out)
        workers.append(Worker(core, [of_in.cons(), of_out.prod(), kern]))

    in_ty = np.ndarray[(line * 3 * height,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(line * height,), np.dtype[np.uint8]]
    chunk_in, chunk_out = line * 3 * rows_per_col, line * rows_per_col

    in_h = [f.prod() for f in ins]
    out_h = [f.cons() for f in outs]

    def seq(a, b, *handles):
        tg = TaskGroup()
        for c in range(n_cols):
            handles[c].fill(a, tap=_slice(in_ty, c * chunk_in, chunk_in), group=tg)
        for c in range(n_cols):
            handles[n_cols + c].drain(b, tap=_slice(out_ty, c * chunk_out, chunk_out),
                                      wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [in_ty, out_ty, *in_h, *out_h])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _slice(tensor_ty, offset: int, size: int):
    from aie.helpers.taplib import TensorAccessPattern
    total = tensor_ty.__args__[0][0]
    return TensorAccessPattern((1, total), offset, [1, 1, 1, size], [0, 0, 0, 1])


def pack(fg: np.ndarray, bg: np.ndarray, mask: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """fg/bg: (H,W,3) uint8, mask: (H,W) uint8 → (H, 3, W*3) 连续数组"""
    h, w = mask.shape
    if out is None:
        out = np.empty((h, 3, w * 3), np.uint8)
    out[:, 0] = fg.reshape(h, w * 3)
    out[:, 1] = bg.reshape(h, w * 3)
    out[:, 2] = np.repeat(mask, 3, axis=1)
    return out


def reference(fg, bg, mask):
    m = np.repeat(mask, 3, axis=1).reshape(fg.shape).astype(np.uint32)
    return ((fg.astype(np.uint32) * m + bg.astype(np.uint32) * (255 - m)) >> 8).astype(np.uint8)


def selftest(width=1280, height=720, n_cols=1, iters=20):
    rng = np.random.default_rng(0)
    fg = rng.integers(0, 256, (height, width, 3), np.uint8)
    bg = rng.integers(0, 256, (height, width, 3), np.uint8)
    mask = rng.integers(0, 256, (height, width), np.uint8)

    packed = pack(fg, bg, mask)
    t_in = iron.tensor(packed.reshape(-1), dtype=np.uint8, device="npu")
    t_out = iron.zeros(width * height * 3, dtype=np.uint8, device="npu")

    t0 = time.perf_counter()
    alpha_blend(t_in, t_out, width=width, height=height, n_cols=n_cols)
    print(f"[blend] 首次调用 (含 JIT 编译) {time.perf_counter() - t0:.1f} s", file=sys.stderr)

    got = t_out.numpy().reshape(height, width, 3)
    exp = reference(fg, bg, mask)
    diff = np.abs(got.astype(int) - exp.astype(int))
    print(f"[blend] 最大误差 {diff.max()}  不一致像素 {(diff > 1).mean()*100:.3f}%", file=sys.stderr)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        alpha_blend(t_in, t_out, width=width, height=height, n_cols=n_cols)
        times.append(time.perf_counter() - t0)
    times.sort()
    print(f"[blend] {width}x{height} n_cols={n_cols}: 中位数 {times[len(times)//2]*1000:.2f} ms, "
          f"最快 {times[0]*1000:.2f} ms", file=sys.stderr)
    return diff.max() <= 1


if __name__ == "__main__":
    cols = int(sys.argv[sys.argv.index("--cols") + 1]) if "--cols" in sys.argv else 1
    ok = selftest(n_cols=cols)
    sys.exit(0 if ok else 1)
