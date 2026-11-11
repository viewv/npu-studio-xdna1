"""NPU 融合合成: 全分辨率的 "放大 + 导向滤波遮罩 + 背景放大 + 合成" 一次完成

CPU 只准备低分辨率数据 (宽 W/4, 已垂直插值到全高度):
  Aq  int16  = round(64·a)                 导向滤波系数 a
  Bq  int16  = round(64·255·(b + a·128/255))
  bg  BGRA   模糊后的背景
NPU 每行读入原图 BGRA + 上述低分辨率行, 输出合成后的 BGRA 行。

  python -m npu.fused --selftest [--cols 4]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, Out, Program, Runtime

from .harness import RowPipeline, external_row_kernel

HERE = Path(__file__).resolve().parent
KERNEL = str(HERE / "kernels" / "fused_composite.cc")
LIB = str(HERE / "lib")                 # npu/lib/row_ops.h — 各逐行内核共用
MLIR_AIE = Path(os.environ["MLIR_AIE_INSTALL_DIR"]) if os.environ.get("MLIR_AIE_INSTALL_DIR") \
    else Path(iron.__file__).resolve().parents[2]
KERNEL_INC = str(MLIR_AIE / "include" / "aie_kernels")

def low_stride(width: int) -> int:
    """低分辨率行每段的元素数: 留出内核会读的尾巴, 并按 64 字节对齐 (int16 平面 2*LS)

    可以用 NPU_LOW_STRIDE 覆盖 (调参用: 某些取值会触发 AIE 编译器的问题, 见 README)。
    """
    v = os.environ.get("NPU_LOW_STRIDE")
    return int(v) if v else -(-(width // 4 + 16) // 32) * 32


def low_row_bytes(ls: int) -> int:
    """Aq + Bq (int16) + bg (BGRA) 一行字节数"""
    return ls * 2 * 2 + ls * 4

WP = np.array([3, 1, 0, 0]); WC = np.array([5, 7, 7, 5]); WN = np.array([0, 0, 1, 3])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def fused_composite(px: In, low: In, out: Out, *,
                    width: CompileTime[int], height: CompileTime[int],
                    low_stride: CompileTime[int], n_cols: CompileTime[int] = 4):
    row = width * 4
    low_row = low_stride * 2 * 2 + low_stride * 4
    row_ty = np.ndarray[(row,), np.dtype[np.uint8]]
    low_ty = np.ndarray[(low_row,), np.dtype[np.uint8]]

    kern = external_row_kernel(
        "fusedCompositeLine", KERNEL, [row_ty, low_ty], row_ty,
        scalars=[np.int32], include_dirs=[KERNEL_INC, LIB],
        compile_flags=[f"-DLOW_STRIDE={low_stride}"])

    pl = RowPipeline(kern, [row, low_row], row, height=height, n_cols=n_cols,
                     extra_args=[width], prefix="fx")
    pl.build()

    rt = Runtime()
    with rt.sequence(*pl.tensor_types()) as bufs:
        pl.stream(rt, bufs)
    return Program(iron.get_current_device(), rt).resolve_program()


# ---------------------------------------------------------------- 主机端封装

class LowRowViews:
    """把 (H, low_row) 字节缓冲区按段切成 Aq / Bq / bg 的 numpy 视图 (含左右各 1 个填充)"""

    def __init__(self, buf: np.ndarray, lw: int, ls: int):
        h = buf.shape[0]
        self.lw, self.ls = lw, ls
        self.aq = buf[:, : ls * 2].view(np.int16)          # (H, ls)
        self.bq = buf[:, ls * 2: ls * 4].view(np.int16)
        self.bg = buf[:, ls * 4:].reshape(h, ls, 4)

    def set(self, aq: np.ndarray, bq: np.ndarray, bg: np.ndarray) -> None:
        """aq/bq: (H, LW) int16, bg: (H, LW, 4) uint8 — 写入并做左右复制填充"""
        lw = self.lw
        for dst, src in ((self.aq, aq), (self.bq, bq), (self.bg, bg)):
            dst[:, 1: lw + 1] = src
            dst[:, 0] = src[:, 0]
            dst[:, lw + 1] = src[:, -1]


class FusedCompositor:
    def __init__(self, width: int, height: int, n_cols: int = 4):
        if width % 32 or height % n_cols:
            raise ValueError(f"不支持的尺寸 {width}x{height} (宽需为32的倍数, 高需能被 {n_cols} 整除)")
        self.w, self.h, self.n_cols = width, height, n_cols
        self.ls = low_stride(width)
        lr = low_row_bytes(self.ls)
        self.t_px = iron.zeros(width * height * 4, dtype=np.uint8, device="npu")
        self.t_low = iron.zeros(lr * height, dtype=np.uint8, device="npu")
        self.t_out = iron.zeros(width * height * 4, dtype=np.uint8, device="npu")
        self.px = self.t_px.data.reshape(height, width, 4)        # 直接写进 NPU 缓冲区
        self.low = LowRowViews(self.t_low.data.reshape(height, lr), width // 4, self.ls)
        t0 = time.perf_counter()
        self.run()                       # 预热: 触发 JIT 编译 / 加载 xclbin (首次约 2 s)
        self.warmup_s = time.perf_counter() - t0

    def run(self) -> np.ndarray:
        self.t_px._sync_to_device()
        self.t_low._sync_to_device()
        fused_composite(self.t_px, self.t_low, self.t_out,
                        width=self.w, height=self.h, low_stride=self.ls, n_cols=self.n_cols)
        return self.t_out.numpy().reshape(self.h, self.w, 4)


# ---------------------------------------------------------------- numpy 参考实现 (与内核整数运算一致)

def _up_h(v: np.ndarray) -> np.ndarray:
    """v: (H, LW, ...) → (H, 4·LW, ...) 水平 ×4, 与内核相同的 /8 权重与向下取整"""
    pad = np.concatenate([v[:, :1], v, v[:, -1:]], axis=1).astype(np.int32)
    lw = v.shape[1]
    out = np.empty((v.shape[0], lw * 4) + v.shape[2:], np.int32)
    for r in range(4):
        out[:, r::4] = (WP[r] * pad[:, 0:lw] + WC[r] * pad[:, 1:lw + 1] + WN[r] * pad[:, 2:lw + 2]) >> 3
    return out


def reference(px: np.ndarray, aq: np.ndarray, bq: np.ndarray, bg: np.ndarray) -> np.ndarray:
    p = px.astype(np.int32)
    gray = ((29 * p[..., 0] + 150 * p[..., 1] + 77 * p[..., 2]) >> 8) - 128
    A, B = _up_h(aq), _up_h(bq)
    m = np.clip((A * gray + B) >> 6, 0, 255)[..., None]
    bgu = _up_h(bg)
    return ((p * m + bgu * (255 - m)) >> 8).astype(np.uint8)


def selftest(width=1280, height=720, n_cols=4, iters=30):
    rng = np.random.default_rng(1)
    lw = width // 4
    comp = FusedCompositor(width, height, n_cols)
    px = rng.integers(0, 256, (height, width, 4), np.uint8)
    aq = rng.integers(-400, 400, (height, lw)).astype(np.int16)
    bq = rng.integers(-8000, 24000, (height, lw)).astype(np.int16)
    bg = rng.integers(0, 256, (height, lw, 4), np.uint8)
    comp.px[:] = px
    comp.low.set(aq, bq, bg)

    t0 = time.perf_counter()
    got = comp.run().copy()
    print(f"[fused] 首次调用 (含 JIT 编译) {time.perf_counter() - t0:.1f} s", file=sys.stderr)
    exp = reference(px, aq, bq, bg)
    diff = np.abs(got.astype(int) - exp.astype(int))
    print(f"[fused] 最大误差 {diff.max()}  误差>1 的字节 {(diff > 1).mean() * 100:.3f}%", file=sys.stderr)
    if diff.max() > 1:
        bad = np.argwhere(diff > 1)[:5]
        for y, x, c in bad:
            print(f"   ({y},{x},{c}) npu={got[y, x, c]} ref={exp[y, x, c]}", file=sys.stderr)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        comp.run()
        times.append(time.perf_counter() - t0)
    times.sort()
    print(f"[fused] {width}x{height} n_cols={n_cols}: 中位数 {times[len(times) // 2] * 1000:.2f} ms, "
          f"最快 {times[0] * 1000:.2f} ms (含同步)", file=sys.stderr)
    return diff.max() <= 1


if __name__ == "__main__":
    def arg(name, default):
        return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
    sys.exit(0 if selftest(arg("--width", 1280), arg("--height", 720),
                            arg("--cols", 4), arg("--iters", 30)) else 1)
