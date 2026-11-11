"""Conv_290 (35→16, 3×3, 180×320, +ReLU) 在 Phoenix NPU 上运行, 并与 numpy 参考 / CPU 对比

  source npu/env.sh && python3 -m npu.rvm.conv_npu [--cols 4] [--iters 50]

设计: 图像上下各补一行零 → 按行切成 n_cols 段 (每段多带上下各 1 行重叠) → 每列一个核心,
     3 行滑动窗口调用 conv2dk3_i8 (check=middle), 权重通过第二路 DMA 每次调用送一份。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction

from ..fused import KERNEL_INC
from .conv_proto import QConv, float_reference, load_case

MIDDLE = 1      # conv2dk3 的 region 枚举: top=0, middle=1, bottom=2
# 上游 int8 向量版有编译错误, 用修补过的副本 (见文件头注释)
CONV_SRC = str(Path(__file__).resolve().parents[1] / "kernels" / "conv2dk3_fixed.cc")


def conv2dk3_i8(width: int, ic: int, oc: int) -> ExternalFunction:
    line_ty = np.ndarray[(width * ic,), np.dtype[np.int8]]
    wt_ty = np.ndarray[(9 * ic * oc,), np.dtype[np.int8]]
    out_ty = np.ndarray[(width * oc,), np.dtype[np.uint8]]
    return ExternalFunction("conv2dk3_i8", source_file=CONV_SRC,
                            arg_types=[line_ty, line_ty, line_ty, wt_ty, out_ty] + [np.int32] * 8,
                            include_dirs=[KERNEL_INC, KERNEL_INC + "/aie2"], compile_flags=["-DINT8_ACT"])


def _slice(n_total: int, offset: int, size: int):
    from aie.helpers.taplib import TensorAccessPattern
    return TensorAccessPattern((1, n_total), offset, [1, 1, 1, size], [0, 0, 0, 1])


TILE = 32      # 上游向量内核把宽度写死为 32, 图像切成 32 宽条带 (见 QConv.pack_input_tiles)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def conv3x3_relu(act: In, wts: In, out: Out, *,
                 height: CompileTime[int], n_tiles: CompileTime[int], ic: CompileTime[int],
                 oc: CompileTime[int], shift: CompileTime[int], n_cols: CompileTime[int] = 4):
    tiles_per_col = n_tiles // n_cols
    line_in, line_out, n_w = TILE * ic, TILE * oc, 9 * ic * oc
    in_ty = np.ndarray[(line_in,), np.dtype[np.int8]]
    out_ty = np.ndarray[(line_out,), np.dtype[np.uint8]]
    w_ty = np.ndarray[(n_w,), np.dtype[np.int8]]
    kern = conv2dk3_i8(TILE, ic, oc)

    def core(of_in, of_w, of_out, k):
        for _ in range_(sys.maxsize):
            w = of_w.acquire(1)
            for _ in range_(tiles_per_col):              # 每个条带是一张 (H+2)×32 的小图
                for _ in range_(height):
                    win = of_in.acquire(3)
                    o = of_out.acquire(1)
                    k(win[0], win[1], win[2], w, o, TILE, ic, oc, 3, 3, MIDDLE, shift, 0)
                    of_in.release(1)
                    of_out.release(1)
                of_in.release(2)
            of_w.release(1)

    fifos, workers = [], []
    for c in range(n_cols):
        f_in = ObjectFifo(in_ty, depth=4, name=f"act{c}")
        f_w = ObjectFifo(w_ty, depth=1, name=f"wts{c}")
        f_out = ObjectFifo(out_ty, depth=2, name=f"out{c}")
        fifos.append((f_in, f_w, f_out))
        workers.append(Worker(core, [f_in.cons(), f_w.cons(), f_out.prod(), kern], while_true=False))

    tile_in, tile_out = (height + 2) * line_in, height * line_out
    n_act, n_out = n_tiles * tile_in, n_tiles * tile_out
    act_ty = np.ndarray[(n_act,), np.dtype[np.int8]]
    wts_ty = np.ndarray[(n_w,), np.dtype[np.int8]]
    res_ty = np.ndarray[(n_out,), np.dtype[np.uint8]]

    rt = Runtime()
    with rt.sequence(act_ty, wts_ty, res_ty) as (a, w, o):
        rt.start(*workers)
        tg = rt.task_group()
        for c, (f_in, f_w, _) in enumerate(fifos):
            rt.fill(f_w.prod(), w, task_group=tg)
            rt.fill(f_in.prod(), a, tap=_slice(n_act, c * tiles_per_col * tile_in, tiles_per_col * tile_in),
                    task_group=tg)
        for c, (_, _, f_out) in enumerate(fifos):
            rt.drain(f_out.cons(), o, tap=_slice(n_out, c * tiles_per_col * tile_out, tiles_per_col * tile_out),
                     wait=True, task_group=tg)
        rt.finish_task_group(tg)
    return Program(iron.get_current_device(), rt).resolve_program()


def cpu_baseline(x, W, B, threads=2, iters=30):
    """同一个 Conv+ReLU 用 ONNX Runtime (fp32) 在 CPU 上跑, 作为对比基线"""
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper
    g = helper.make_graph(
        [helper.make_node("Conv", ["x", "W", "B"], ["c"], pads=[1, 1, 1, 1]),
         helper.make_node("Relu", ["c"], ["y"])],
        "conv", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, *x.shape])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    xi = x[None].astype(np.float32)
    for _ in range(5):
        s.run(None, {"x": xi})
    t, c = time.perf_counter(), time.process_time()
    for _ in range(iters):
        s.run(None, {"x": xi})
    return (time.perf_counter() - t) / iters * 1000, (time.process_time() - c) / iters * 1000


def main():
    cols = int(sys.argv[sys.argv.index("--cols") + 1]) if "--cols" in sys.argv else 4
    iters = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 50
    x, W, B, _, calib = load_case()
    ic_real, h, w = x.shape
    yf = float_reference(x, W, B)
    q = QConv(W, B, x_absmax=float(np.abs(calib).max()), y_max=float(np.percentile(yf, 99.99)))
    xq = q.quant_x(x)
    y_ref = q.reference(xq)                                   # 与内核逐位一致的期望输出

    n_tiles = q.n_tiles(w, multiple=cols)
    t_act = iron.tensor(q.pack_input_tiles(xq, n_tiles).reshape(-1), dtype=np.int8, device="npu")
    t_w = iron.tensor(q.pack_weights(), dtype=np.int8, device="npu")
    t_out = iron.zeros(n_tiles * h * q.oc * TILE, dtype=np.uint8, device="npu")
    kw = dict(height=h, n_tiles=n_tiles, ic=q.icp, oc=q.oc, shift=q.shift, n_cols=cols)
    print(f"[npu] {n_tiles} 个 32 宽条带 (有效 {n_tiles * 30} 列, 原图 {w} 列)", file=sys.stderr)

    t0 = time.perf_counter()
    conv3x3_relu(t_act, t_w, t_out, **kw)
    print(f"[npu] 首次调用 (含 JIT 编译) {time.perf_counter() - t0:.1f} s", file=sys.stderr)

    y_npu = q.unpack_output_tiles(t_out.numpy(), n_tiles, h, w)
    d = np.abs(y_npu.astype(int) - y_ref.astype(int))
    print(f"[npu] 与 int8 参考: 最大差 {d.max()}, 不一致 {(d > 0).mean()*100:.3f}%", file=sys.stderr)
    yh = q.dequant_y(y_npu)
    rel = np.linalg.norm(yh - yf) / np.linalg.norm(yf)
    print(f"[npu] 与 float32 参考: 相对误差 {rel*100:.2f}%", file=sys.stderr)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        conv3x3_relu(t_act, t_w, t_out, **kw)
        times.append(time.perf_counter() - t0)
    times.sort()
    macs = W.size * h * w
    med = times[len(times) // 2]
    print(f"[npu] {ic_real}→{q.oc} 3x3 @ {h}x{w}, {cols} 列: 中位数 {med*1000:.2f} ms, 最快 {times[0]*1000:.2f} ms "
          f"({macs / med / 1e9:.0f} GMAC/s, 含同步)", file=sys.stderr)
    for th in (1, 2):
        wall, cpu = cpu_baseline(x, W, B, threads=th)
        print(f"[cpu] ONNX Runtime fp32, {th} 线程: {wall:.2f} ms/次 (CPU 时间 {cpu:.2f} ms)", file=sys.stderr)


if __name__ == "__main__":
    main()
