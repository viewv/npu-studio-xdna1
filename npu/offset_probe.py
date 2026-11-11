"""编译器回归探针: 常量平面偏移 >= 1024 B 的非对齐加载 (见 npu/lib/row_ops.h 文件头)

  python -m npu.offset_probe        # 打印 原样 / plane_stride 两种写法的错误率

原样写法错 = 编译器 bug 还在 (预期, 内核都已经绕过);
原样写法对 = 新编译器修好了, plane_stride 可以保留也可以去掉;
plane_stride 写法错 = 绕过失效, 所有多平面内核都要重新排查。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, Out, Program, Runtime

from .fused import KERNEL_INC, LIB, _up_h
from .harness import RowPipeline, external_row_kernel

SRC = str(Path(__file__).resolve().parent / "kernels" / "offset_probe.cc")
W, H = 256, 4


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def _probe(inp: In, out: Out, *, plane: CompileTime[int], launder: CompileTime[int]):
    in_b, out_b = plane * 4, W * 2
    k = external_row_kernel("offsetProbeLine", SRC, [np.ndarray[(in_b,), np.dtype[np.uint8]]],
                            np.ndarray[(out_b,), np.dtype[np.uint8]], scalars=[np.int32],
                            include_dirs=[KERNEL_INC, LIB],
                            compile_flags=[f"-DPLANE={plane}", f"-DLAUNDER={launder}"])
    pl = RowPipeline(k, [in_b], out_b, height=H, n_cols=1, extra_args=[W], prefix=f"op{launder}")
    pl.build()
    rt = Runtime()
    with rt.sequence(*pl.tensor_types()) as bufs:
        pl.stream(rt, bufs)
    return Program(iron.get_current_device(), rt).resolve_program()


def wrong_fraction(plane: int = 512, launder: bool = False) -> float:
    """返回 up4(B) 与参考不一致的元素比例 (0 = 正确)"""
    lw = W // 4
    rng = np.random.default_rng(plane)
    buf = np.zeros((H, plane * 4), np.uint8)
    va, vb = buf[:, :plane * 2].view(np.int16), buf[:, plane * 2:].view(np.int16)
    A = rng.integers(-400, 400, (H, lw)).astype(np.int16)
    B = rng.integers(-400, 400, (H, lw)).astype(np.int16)
    for dst, src in ((va, A), (vb, B)):
        dst[:, 1:lw + 1] = src
        dst[:, 0] = src[:, 0]
        dst[:, lw + 1] = src[:, -1]
    t_in = iron.tensor(buf.reshape(-1), dtype=np.uint8, device="npu")
    t_out = iron.zeros(H * W * 2, dtype=np.uint8, device="npu")
    _probe(t_in, t_out, plane=plane, launder=int(launder))
    got = t_out.numpy().reshape(H, W * 2).view(np.int16)
    return float((got != _up_h(B)).mean())


if __name__ == "__main__":
    raw, fixed = wrong_fraction(512, False), wrong_fraction(512, True)
    print(f"常量偏移 1024 B, 原样写法: 错 {raw * 100:.1f}%   plane_stride 写法: 错 {fixed * 100:.1f}%")
    sys.exit(0 if fixed == 0 else 1)
