"""LN 定点阶梯的 NPU 探针（独立 IRON 程序, 不依赖 chain_c16）

存档说明: 这是"探针先验证、再集成"的那一步留下的脚本 —— 单 token (512 通道) 的完整 LN 定点阶梯,
在真机上与 numpy 的同一套阶梯**逐位一致**。集成进 chain 之后, 512 通道又出问题 (见
docs/npu-handoff.md §6A), 所以这份探针是判断"数学对不对 / 框架的调度对不对"的分界线:
如果这里一致、chain 里不一致, 问题就在调度/打包, 不在定点阶梯。

跑法 (需要先 source npu/env.sh):
    .venv-iron/bin/python npu/probes/ln_probe/probe.py
"""
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

D = 512
from pathlib import Path
INCLUDE = str(next(Path(__file__).resolve().parents[3].glob(".venv-iron/lib/python*/site-packages/mlir_aie/include/aie_kernels")))
SRC = str(Path(__file__).resolve().parent / "ln_probe.cc")

def tap(n, off, size):
    return TensorAccessPattern((1, n), off, [1, 1, 1, size], [0, 0, 0, 1])

@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def ln_kernel(x: In, p: In, y: Out, *, d: CompileTime[int]):
    i8, i32 = np.dtype[np.int8], np.dtype[np.int32]
    x_ty = np.ndarray[(d,), i8]
    p_ty = np.ndarray[(5 * d,), i32]
    y_ty = np.ndarray[(d,), i8]
    kern = ExternalFunction("lnProbe", source_file=SRC, arg_types=[x_ty, p_ty, y_ty],
                            include_dirs=[INCLUDE], compile_flags=[f"-DD={d}"])
    f_in = ObjectFifo(x_ty, depth=2, name="lni")
    f_p = ObjectFifo(p_ty, depth=1, name="lnp")
    f_out = ObjectFifo(y_ty, depth=2, name="lno")
    def core(fi, fp, fo, k):
        pv = fp.acquire(1)
        for _ in range_(1):
            win = fi.acquire(1)
            o = fo.acquire(1)
            k(win, pv, o)
            fi.release(1)
            fo.release(1)
        fp.release(1)
    w = Worker(core, [f_in.cons(), f_p.cons(), f_out.prod(), kern])
    rt = Runtime()
    with rt.sequence(x_ty, p_ty, y_ty) as (a, b, c):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(f_p.prod(), b, task_group=tg)
        rt.fill(f_in.prod(), a, tap=tap(d, 0, d), task_group=tg)
        rt.drain(f_out.cons(), c, tap=tap(d, 0, d), wait=True, task_group=tg)
        rt.finish_task_group(tg)
    return Program(iron.get_current_device(), rt).resolve_program()

def ref(x, m1, m2, g10, b12, gq):
    """与 kernel 完全同一套定点阶梯（逐位参考）"""
    x = x.astype(np.int64)
    s1 = (x * m1).sum(); s2 = (x * x * m2).sum()
    mean_q = s1 // D
    var = s2 / (D * 524288.0) - float(mean_q) ** 2 / (4096.0 ** 2)
    r12 = int(1.0 / np.sqrt(var + 1e-6) * 4096.0 + 0.5)
    dev = x * m1 - mean_q
    n12 = (dev * r12) >> 12
    y12 = (n12 * g10) >> 10
    o = ((y12 + b12) * gq) >> 12
    return np.clip(o, -128, 127).astype(np.int8)

def main():
    dev = iron.get_current_device()
    rng = np.random.default_rng(0)
    x = rng.integers(-120, 121, D).astype(np.int8)
    m1 = np.full(D, 82, np.int32); m2 = np.full(D, 210, np.int32)
    g10 = np.full(D, 1024, np.int32)      # gamma = 1.0
    b12 = np.zeros(D, np.int32)           # beta = 0
    gq = np.full(D, int(round(1/0.02 * 4096)), np.int32)   # 输出比例 0.02
    t_x = iron.tensor(x, dtype=np.int8, device="npu")
    t_p = iron.tensor(np.concatenate([m1, m2, g10, b12, gq]), dtype=np.int32, device="npu")
    t_y = iron.zeros(D, dtype=np.int8, device="npu")
    fn = ln_kernel
    fn(t_x, t_p, t_y, d=D)
    got = t_y.numpy().astype(np.int8)
    exp = ref(x, m1, m2, g10, b12, gq)
    d = int(np.abs(got.astype(int) - exp.astype(int)).max())
    print(f"LN 定点探针: 最大差 {d}  样本 {got[:8]} vs {exp[:8]}")
    return 0 if d == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
