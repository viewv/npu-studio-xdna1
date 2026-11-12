"""RVM 编码器 (MobileNetV3, 320×180) 的整包功耗: NPU (npu/rvm/encoder_npu) vs onnxruntime CPU

  source npu/env.sh && python powerbench_enc.py [--secs 15] [--rounds 6] [--fps 30] [--max-rounds 2]
                                               [--backends idle,npu,cpu×1,cpu×4,gpu,cpu×1-full]

后端: npu = NPU 编码器; npu-full = NPU 编码器 + 解码器的卷积 (decoder_probe, 不含逐元素部分, 6 次调用); cpu×N = onnxruntime 编码器 N 线程; gpu = studio 用的 RVM 整网 @780M (ncnn + Vulkan, 常驻 worker 进程,
它的 CPU 时间也算进每帧 CPU); cpu×1-full = onnxruntime 单线程 RVM 整网 (看编码器 / 解码器各占多少)。

两种负载, 都在同一个进程里按轮交替 (每轮内顺序轮换), 跟 README 说的一样: 桌面在用时 PPT 漂移很大,
分开跑的几分钟对比不可信, 只看同一轮里和空闲成对相减的差。
  1. 定速 (默认 30 fps, 摄像头的真实情形): 每帧一次编码器, 剩下的时间 sleep → 每种后端比空闲多几瓦、每帧多少 CPU 时间;
  2. 连续跑满: ms/帧、整包 W、J/帧 (= W × t, 含底噪)。
每帧的工作: NPU = 写入量化好的输入 (set_input, 含同步到设备) + 一次调用; CPU = sess.run (fp32, 输出 f1..f4)。
两边都不含前处理 (归一化 / 量化), 也不含解码器。onnxruntime 关掉 spinning (否则线程空转, 定速下白白耗电)。
"""
from __future__ import annotations

import os
import resource
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def arg(name, default, typ=float):
    return typ(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


SECS = arg("--secs", 15.0)
ROUNDS = arg("--rounds", 6, int)
FPS = arg("--fps", 30.0)
MAX_ROUNDS = arg("--max-rounds", 2, int)
BACKENDS = arg("--backends", "idle,npu,cpu×1,cpu×4", str).split(",")
CHILD = []          # 子进程 pid (gpu worker), 它们的 CPU 时间也算
WARM = 3.0


def _ppt_path():
    import glob
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            if open(f"{h}/name").read().strip() == "amdgpu" and os.path.exists(f"{h}/power1_average"):
                return f"{h}/power1_average"
        except OSError:
            pass
    return None


PPT = _ppt_path()


def watts():
    return int(open(PPT).read()) / 1e6


def cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    t = r.ru_utime + r.ru_stime
    for pid in CHILD:
        try:
            f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            t += (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
        except (OSError, ValueError, IndexError):
            pass
    return t


def phase(run, paced: bool):
    """跑 SECS 秒 → (平均 W, 帧数, 每帧墙钟 ms, 每帧 CPU ms); 前 WARM 秒不采样"""
    period = 1.0 / FPS
    t0 = time.monotonic()
    nxt, n, samples, c0, n0, tw = t0, 0, [], None, 0, None
    while True:
        now = time.monotonic()
        el = now - t0
        if el >= SECS:
            break
        if el >= WARM:
            if c0 is None:
                c0, n0, tw = cpu_s(), n, now
            samples.append(watts())
        if run is not None:
            run()
        n += 1
        if paced:
            nxt += period
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.monotonic()
    t1 = time.monotonic()
    nf = max(1, n - n0)
    return statistics.mean(samples), nf, (t1 - tw) / nf * 1e3, (cpu_s() - c0) / nf * 1e3


def main():
    if not PPT:
        sys.exit("找不到 amdgpu power1_average, 无法测功率")
    import onnx
    import onnx.utils
    import onnxruntime as ort

    from npu import chain_c16 as c
    from npu.rvm import encoder_npu as e
    from npu.rvm.encoder_ref import FEATS, ONNX, images, preprocess, run_float

    print("校准 + 建 NPU 链 (约 40 s) ……", flush=True)
    qe, ops = e.prepare()
    im = images(1, offset=16)[0]
    xf = preprocess(im)
    ref = run_float(ops, xf)
    qe.run(xf)
    shapes = {k: v.shape for k, v in ref.items()}
    layers, params, _ = e.build(qe, shapes)
    ch = c.C16Chain(layers, params)
    xq = np.zeros((layers[0].ic, *shapes["x"][1:]), np.int8)
    xq[:shapes["x"][0]] = qe.last_int["x"][0]
    ch.set_input(xq)
    ch.call()

    def run_npu():
        ch.set_input(xq)
        ch.call()

    # onnxruntime: 只要编码器 (Conv_8 的输入 → f1..f4)
    m = onnx.load(str(ONNX))
    x_name = next(n for n in m.graph.node if n.name == "Conv_8").input[0]
    known = {v.name for v in m.graph.value_info}
    for nm, shp in [(x_name, [1, *xf.shape])] + [(f, None) for f in FEATS.values()]:
        if nm not in known:        # 图里形状是动态的 (downsample_ratio), Extractor 要 value_info
            m.graph.value_info.append(onnx.helper.make_tensor_value_info(nm, onnx.TensorProto.FLOAT, shp))
    sub =onnx.utils.Extractor(m).extract_model([x_name], list(FEATS.values()))
    xi = xf[None].astype(np.float32)
    sessions = {}
    for th in (1, 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = th
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        s = ort.InferenceSession(sub.SerializeToString(), so, providers=["CPUExecutionProvider"])
        s.run(None, {x_name: xi})
        sessions[th] = s
    runs = {"idle": None, "npu": run_npu}
    for th, s in sessions.items():
        runs[f"cpu×{th}"] = (lambda s=s: s.run(None, {x_name: xi}))
    if "npu-full" in BACKENDS:                            # 编码器 + 解码器的卷积 (npu/rvm/decoder_probe, 随机权重)
        from npu.rvm.decoder_probe import build_all
        dec = [d[1] for d in build_all()]

        def run_npu_full():
            run_npu()
            for d in dec:
                d.call()
        runs["npu-full"] = run_npu_full
    frames = images(8)                                   # 整网的后端轮流喂几张不同的图
    if "gpu" in BACKENDS:
        import cv2
        from studio.rvm_gpu import RvmGpuMatting
        g = RvmGpuMatting()
        CHILD.append(g.proc.pid)
        bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
        k_g = [0]

        def run_gpu():
            g(bgr[k_g[0] % len(bgr)])
            k_g[0] += 1
        runs["gpu"] = run_gpu
    if "cpu×1-full" in BACKENDS:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sf = ort.InferenceSession(str(ONNX), so, providers=["CPUExecutionProvider"])
        h, w = xf.shape[1:]
        st = {f"r{i + 1}i": np.zeros((1, ch_, -(-h // 2 ** (i + 1)), -(-w // 2 ** (i + 1))), np.float32)
              for i, ch_ in enumerate((16, 20, 40, 64))}
        srcs = [(f.astype(np.float32) / 255).transpose(2, 0, 1)[None] for f in frames]
        k_f = [0]

        def run_full():
            out = sf.run(["pha", "r1o", "r2o", "r3o", "r4o"],
                         {"src": srcs[k_f[0] % len(srcs)], "downsample_ratio": np.array([1.0], np.float32), **st})
            st.update({f"r{i + 1}i": r for i, r in enumerate(out[1:])})
            k_f[0] += 1
        runs["cpu×1-full"] = run_full
    runs = {k: runs[k] for k in BACKENDS}

    names = list(runs)
    print(f"\n[1] 定速 {FPS:.0f} fps, {ROUNDS} 轮 × {len(names)} 种 × {SECS:.0f} s (轮内顺序轮换)", flush=True)
    res = {k: [] for k in names}
    for r in range(ROUNDS):
        order = names[r % len(names):] + names[:r % len(names)]
        line = []
        for k in order:
            w, nf, ms, cpu = phase(runs[k], paced=True)
            res[k].append((w, ms, cpu))
            line.append(f"{k} {w:5.2f} W")
        print(f"  轮 {r + 1}: " + "  ".join(line), flush=True)
    idle = [v[0] for v in res["idle"]]
    print(f"\n  空闲 {statistics.mean(idle):.2f} W (各轮 {min(idle):.2f}–{max(idle):.2f})")
    for k in names[1:]:
        d = [v[0] - i for v, i in zip(res[k], idle)]
        se = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")
        cpu = statistics.mean(v[2] for v in res[k])
        print(f"  {k:6s} 比空闲 {statistics.mean(d):+5.2f} W (中位 {statistics.median(d):+5.2f}, 标准误 {se:.2f})   "
              f"每帧 CPU {cpu:5.2f} ms  ({cpu * FPS / 10:.1f}% 一个核)", flush=True)

    print(f"\n[2] 连续跑满, {MAX_ROUNDS} 轮", flush=True)
    full = {k: [] for k in names[1:]}
    idle2 = []
    for r in range(MAX_ROUNDS):
        idle2.append(phase(None, paced=True)[0])
        for k in names[1:]:
            full[k].append(phase(runs[k], paced=False))
    p0 = statistics.mean(idle2)
    print(f"  空闲 {p0:.2f} W")
    for k, v in full.items():
        w = statistics.mean(x[0] for x in v)
        ms = statistics.mean(x[2] for x in v)
        cpu = statistics.mean(x[3] for x in v)
        print(f"  {k:6s} {ms:6.2f} ms/帧  {w:5.2f} W (边际 {w - p0:+5.2f})  {w * ms:6.1f} mJ/帧 "
              f"(边际 {(w - p0) * ms:5.1f})  每帧 CPU {cpu:5.2f} ms", flush=True)


if __name__ == "__main__":
    main()
