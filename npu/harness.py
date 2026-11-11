"""NPU 行流流水线脚手架 (IRON / mlir-aie)

把一个"逐行"内核切成 n_cols 段并行: 每列一个 core, 每列从同一块输入缓冲区里
取自己那段行, 结果写回同一块输出缓冲区的对应段。

行内核的约定 (C 函数签名):

    void kernel(<输入行 1>, <输入行 2>, ..., <输出行>, <标量参数...>);

每个"行"是一块固定字节数的缓冲区 (低分辨率的多通道数据也按字节打包, 用视图解释)。
这样同一个内核既能吃 BGRA 行, 也能吃 int16 平面行 (见 npu/lib/row_ops.h)。

用法 (在 @iron.jit 的函数里):

    kern = external_row_kernel("myKernel", SRC, [px_row_ty, low_row_ty], out_row_ty,
                               scalars=[np.int32], include_dirs=[KERNEL_INC, LIB],
                               compile_flags=["-DXXX=1"])
    pl = RowPipeline(kern, [px_bytes, low_bytes], out_bytes,
                     height=H, n_cols=4, extra_args=[width], prefix="fx")
    pl.build()

    rt = Runtime()
    with rt.sequence(*pl.tensor_types()) as bufs:
        pl.stream(rt, bufs)          # 内部完成 start / fill / drain / finish_task_group
    return Program(iron.get_current_device(), rt).resolve_program()
"""
from __future__ import annotations

import os
import sys

import os
import sys

import numpy as np

from aie.iron import ObjectFifo, Worker

_HARNESS = os.path.abspath(__file__)


# 注意: IRON 的 JIT 磁盘缓存键只看内核 .cc 的 mtime + 编译参数, 改 npu/lib/*.h 或
# 本文件 (切段逻辑会编进 host 端) 都不会失效, 会静默跑旧二进制。改完内核相关代码后:
#     rm -rf ~/.npu/cache/*
# 表现为"改了代码但结果不变"或结果诡异时, 先清缓存再查。


class SelfTestResult:
    """selftest 的结果: 真值 = 是否通过 (兼容旧的 `sys.exit(0 if selftest() else 1)`)"""

    def __init__(self, ok: bool, max_diff: int, ms: float):
        self.ok, self.max_diff, self.ms = bool(ok), int(max_diff), float(ms)

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        return f"SelfTestResult(ok={self.ok}, max_diff={self.max_diff}, ms={self.ms:.2f})"


def _slice(n_total: int, offset: int, size: int):
    from aie.helpers.taplib import TensorAccessPattern
    return TensorAccessPattern((1, n_total), offset, [1, 1, 1, size], [0, 0, 0, 1])


def external_row_kernel(name: str, source_file: str, in_row_tys, out_row_ty, *,
                        scalars=(), include_dirs=(), compile_flags=()):
    """构造行内核的 ExternalFunction (参数顺序: 输入行..., 输出行, 标量...)"""
    from aie.iron.kernel import ExternalFunction
    flags = list(compile_flags)
    return ExternalFunction(name, source_file=source_file,
                            arg_types=list(in_row_tys) + [out_row_ty] + list(scalars),
                            include_dirs=list(include_dirs),
                            compile_flags=flags)


class RowPipeline:
    """把逐行内核铺到 n_cols 个 core 上, 并负责 fill / drain 的切段"""

    def __init__(self, kern, in_row_bytes, out_row_bytes, *, height: int, n_cols: int,
                 extra_args=(), prefix: str = "r", fifo_depth: int = 2,
                 stack_size: int | None = None):
        if height % n_cols:
            raise ValueError(f"高 {height} 必须能被列数 {n_cols} 整除")
        self.kern = kern
        self.in_bytes = [int(n) for n in in_row_bytes]
        self.out_bytes = int(out_row_bytes)
        self.h = height
        self.n_cols = n_cols
        self.rows_per_col = height // n_cols
        self.extra = list(extra_args)
        self.prefix = prefix
        self.depth = fifo_depth
        self.stack_size = stack_size
        self._workers: list | None = None
        self._fifos: list | None = None

    def tensor_types(self):
        """返回 (输入类型..., 输出类型), 直接展开给 rt.sequence(*...)"""
        u8 = np.dtype[np.uint8]
        return tuple([np.ndarray[(n * self.h,), u8] for n in self.in_bytes]
                     + [np.ndarray[(self.out_bytes * self.h,), u8]])

    def build(self):
        """创建 FIFO 和 Worker (必须在 @iron.jit 的追踪里调用)"""
        n_in = len(self.in_bytes)
        extra = self.extra

        def core(*args):
            fis = args[:n_in]
            fo, k = args[n_in], args[n_in + 1]
            vals = [f.acquire(1) for f in fis]
            o = fo.acquire(1)
            k(*vals, o, *extra)
            for f in fis:
                f.release(1)
            fo.release(1)

        fifos, workers = [], []
        for c in range(self.n_cols):
            fs_in = [ObjectFifo(np.ndarray[(n,), np.dtype[np.uint8]], depth=self.depth,
                                name=f"{self.prefix}{c}i{i}")
                     for i, n in enumerate(self.in_bytes)]
            fs_out = ObjectFifo(np.ndarray[(self.out_bytes,), np.dtype[np.uint8]],
                                depth=self.depth, name=f"{self.prefix}{c}o")
            workers.append(Worker(core, [f.cons() for f in fs_in] + [fs_out.prod(), self.kern],
                                  stack_size=self.stack_size))
            fifos.append((*fs_in, fs_out))
        self._workers, self._fifos = workers, fifos
        return workers, fifos

    def stream(self, rt, bufs):
        """start → 各列 fill → 各列 drain → finish (bufs 是 rt.sequence 解出来的)"""
        if self._workers is None:
            raise RuntimeError("先调用 build()")
        ins, outs = tuple(bufs[:-1]), bufs[-1]
        rows = self.rows_per_col
        if os.environ.get("NPU_HARNESS_DEBUG"):
            print(f"[harness] {self.prefix}: {len(bufs)} 个缓冲区 ({len(ins)} 入 + 1 出), "
                  f"{self.n_cols} 列 × {rows} 行, 行字节 {self.in_bytes} → {self.out_bytes}",
                  file=sys.stderr)
        rt.start(*self._workers)
        tg = rt.task_group()
        for c, fifo in enumerate(self._fifos):
            for i, (f, buf) in enumerate(zip(fifo[:-1], ins)):
                step = self.in_bytes[i]
                off, size = c * rows * step, rows * step
                if os.environ.get("NPU_HARNESS_DEBUG"):
                    print(f"[harness]   fill 入{i} 列{c}: n={step * self.h} off={off} size={size} "
                          f"buf={getattr(buf, 'shape', '?')}", file=sys.stderr)
                rt.fill(f.prod(), buf, tap=_slice(step * self.h, off, size),
                        task_group=tg)
        for c, fifo in enumerate(self._fifos):
            step = self.out_bytes
            off, size = c * rows * step, rows * step
            if os.environ.get("NPU_HARNESS_DEBUG"):
                print(f"[harness]   drain 出 列{c}: n={step * self.h} off={off} size={size}",
                      file=sys.stderr)
            rt.drain(fifo[-1].cons(), outs,
                     tap=_slice(step * self.h, off, size),
                     wait=True, task_group=tg)
        rt.finish_task_group(tg)
