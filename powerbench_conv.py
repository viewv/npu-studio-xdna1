"""微基准: 同一份卷积在 CPU / NPU 上的 时间 + 整包功率 + J/帧

  source npu/env.sh && python powerbench_conv.py [--secs 20] [--rounds 2] [--threads 2] [--cols 4]

用来回答"这个内核到底省不省电": NPU 的边际功率低, 但慢会把省下的电吃掉。
能量判据:  E = (P_idle + ΔP) × t      →  NPU 打平 CPU 的条件  t_npu < E_cpu / (P_idle + ΔP_npu)

默认负载是 RVM 解码器最重的 Conv_290 (35→16, 3×3, 180×320, +ReLU), 与 npu/rvm/conv_npu.py 同一份数据。
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SECS = int(sys.argv[sys.argv.index("--secs") + 1]) if "--secs" in sys.argv else 20
ROUNDS = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 2
THREADS = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 2
COLS = int(sys.argv[sys.argv.index("--cols") + 1]) if "--cols" in sys.argv else 4
WARM = 8.0


def _ppt_path() -> str | None:
    import glob
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            if open(f"{h}/name").read().strip() == "amdgpu" and os.path.exists(f"{h}/power1_average"):
                return f"{h}/power1_average"
        except OSError:
            pass
    return None


PPT = _ppt_path()


def watts() -> float | None:
    try:
        return int(open(PPT).read()) / 1e6
    except (OSError, TypeError, ValueError):
        return None


def phase(name: str, run, secs: float = SECS, warm: float = WARM) -> dict:
    n, samples, t0 = 0, [], time.monotonic()
    while True:
        el = time.monotonic() - t0
        if el >= secs:
            break
        run()
        n += 1
        if el >= warm:
            w = watts()
            if w is not None:
                samples.append(w)
    el = time.monotonic() - t0
    ms = el / n * 1000
    p = statistics.mean(samples)
    print(f"  {name:12} {n:6d} 帧  {ms:7.2f} ms/帧  {p:6.2f} W  {p * ms / 1000:7.3f} J/帧", flush=True)
    return {"ms": ms, "w": p, "n": n}


def main():
    if not PPT:
        sys.exit("找不到 amdgpu power1_average, 无法测功率")

    from npu.rvm.conv_proto import QConv, float_reference, load_case
    x, W, B, _, calib = load_case()
    yf = float_reference(x, W, B)
    q = QConv(W, B, x_absmax=float(np.abs(calib).max()), y_max=float(np.percentile(yf, 99.99)))
    xq = q.quant_x(x)
    h, w = x.shape[1:]
    n_tiles = q.n_tiles(w, multiple=COLS)
    macs = W.size * h * w
    print(f"负载: Conv_290 35→16 3x3 @ {h}x{w}  {macs / 1e6:.0f} MMAC\n")

    # CPU: 同一个卷积, ONNX Runtime fp32
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
    so.intra_op_num_threads = THREADS
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    xi = x[None].astype(np.float32)
    sess.run(None, {"x": xi})

    # NPU: 现有的 conv2dk3 (修补版), n_cols 列
    import aie.iron as iron
    from npu.rvm.conv_npu import TILE, conv3x3_relu
    t_act = iron.tensor(q.pack_input_tiles(xq, n_tiles).reshape(-1), dtype=np.int8, device="npu")
    t_w = iron.tensor(q.pack_weights(), dtype=np.int8, device="npu")
    t_out = iron.zeros(n_tiles * h * q.oc * TILE, dtype=np.uint8, device="npu")
    kw = dict(height=h, n_tiles=n_tiles, ic=q.icp, oc=q.oc, shift=q.shift, n_cols=COLS)
    conv3x3_relu(t_act, t_w, t_out, **kw)     # 预热 / JIT

    print(f"顺序: 空闲 → (CPU×{THREADS} / NPU×{COLS}) × {ROUNDS} → 空闲\n")
    r = {"idle": [], f"cpu": [], "npu": []}
    r["idle"].append(phase("idle", lambda: time.sleep(0.05), secs=SECS, warm=4))
    for _ in range(ROUNDS):
        r["cpu"].append(phase(f"cpu×{THREADS}", lambda: sess.run(None, {"x": xi})))
        r["npu"].append(phase(f"npu×{COLS}", lambda: conv3x3_relu(t_act, t_w, t_out, **kw)))
    r["idle"].append(phase("idle", lambda: time.sleep(0.05), secs=max(8, SECS // 2), warm=4))

    def mean(k, f):
        return statistics.mean(v[f] for v in r[k])

    p_idle = mean("idle", "w")
    cpu, npu = {f: mean("cpu", f) for f in ("ms", "w")}, {f: mean("npu", f) for f in ("ms", "w")}
    e_cpu, e_npu = cpu["w"] * cpu["ms"] / 1000, npu["w"] * npu["ms"] / 1000
    be = 1000 * e_cpu / (p_idle + max(0.0, npu["w"] - p_idle))   # NPU 打平 CPU 所需的耗时 (ms)

    print(f"\n整机底噪 {p_idle:.2f} W")
    print(f"CPU: {cpu['ms']:6.2f} ms  {cpu['w']:5.2f} W  边际 {cpu['w'] - p_idle:5.2f} W  "
          f"{e_cpu * 1000:6.1f} mJ/帧   {macs / cpu['ms'] / 1e6:6.0f} MMAC/s")
    print(f"NPU: {npu['ms']:6.2f} ms  {npu['w']:5.2f} W  边际 {npu['w'] - p_idle:5.2f} W  "
          f"{e_npu * 1000:6.1f} mJ/帧   {macs / npu['ms'] / 1e6:6.0f} MMAC/s")
    print(f"→ 能量比 NPU/CPU = {e_npu / e_cpu:.2f}×  "
          f"({'NPU 省' if e_npu < e_cpu else 'NPU 费'} {abs(1 - e_npu / e_cpu) * 100:.0f}%)")
    print(f"→ NPU 要做到 ≤ {be:.2f} ms 才能在整机能量上打平 CPU "
          f"(现在 {npu['ms']:.2f} ms, 还差 {npu['ms'] / be:.1f}×)")


if __name__ == "__main__":
    main()
