"""量化编码器对最终 alpha 的影响: 浮点 RVM 整网 vs (NPU int8 编码器 → 反量化 → 浮点解码器), 连续帧、带时序状态

  source npu/env.sh && python -m npu.rvm.alpha_check 帧.npz [--cal wall|cam|both] [--n-cal 16] [--out 目录] [--sim]

帧.npz: rgb = (N, 180, 320, 3) uint8 (RGB, 已缩到工作分辨率)。两路各自带自己的循环状态 r1..r4 逐帧往下传。
  --cal wall  用系统壁纸校准 (encoder_npu.prepare 的默认, 没有人像)
  --cal cam   用帧.npz (或 --cal-from 另一段.npz) 里均匀抽的 n-cal 帧校准; 不给 --cal-from 时评估帧包含校准帧
  --sim       不上 NPU, 用整数模拟 (和 NPU 逐字节一致, 快很多也不占设备)
指标 (alpha 在 0..1, 报告 ×255):
  MAE 全图 / 边缘带 (参考 alpha 在 (0.02, 0.98) 之间再膨胀 3 像素) / 前景 IoU (阈值 0.5) /
  时序: 帧间变化之差 mean |Δq − Δref| (q 比参考多出来的闪烁)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .. import chain_c16 as c
from . import encoder_npu as e
from .encoder_q import Cfg, QEncoder, calibrate, col
from .encoder_ref import FEATS, ONNX, images, load_graph, preprocess, run_float


def arg(name, default, typ=str):
    return typ(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def sessions():
    import onnx
    import onnx.utils
    import onnxruntime as ort
    m = onnx.load(str(ONNX))
    full = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    known = {v.name for v in m.graph.value_info}
    for nm in FEATS.values():
        if nm not in known:
            m.graph.value_info.append(onnx.helper.make_tensor_value_info(nm, onnx.TensorProto.FLOAT, None))
    ins = ["src", "r1i", "r2i", "r3i", "r4i", "downsample_ratio", *FEATS.values()]
    dec = onnx.utils.Extractor(m).extract_model(ins, ["pha", "r1o", "r2o", "r3o", "r4o"])
    used = {i.name for i in dec.graph.input}
    return full, ort.InferenceSession(dec.SerializeToString(), providers=["CPUExecutionProvider"]), used


def zero_state(h, w):
    return {f"r{i + 1}i": np.zeros((1, ch, -(-h // 2 ** (i + 1)), -(-w // 2 ** (i + 1))), np.float32)
            for i, ch in enumerate((16, 20, 40, 64))}


def prepare(ops, cal_imgs):
    qe = QEncoder(ops, calibrate(ops, cal_imgs), Cfg(aq="channel", pct=0, rq="m16"))
    for im in cal_imgs:
        qe.run(preprocess(im))
    qe.frozen = True
    return qe


def edge_band(a, r=3):
    import cv2
    m = ((a > 0.02) & (a < 0.98)).astype(np.uint8)
    return cv2.dilate(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0


def main():
    frames = np.load(sys.argv[1])["rgb"]
    cal, n_cal, out, sim = arg("--cal", "wall"), arg("--n-cal", 16, int), arg("--out", None), "--sim" in sys.argv
    ops = load_graph()
    cf = np.load(arg("--cal-from", sys.argv[1]))["rgb"]     # 校准帧来源 (默认就是评估的这段)
    cam = [cf[i] for i in np.linspace(0, len(cf) - 1, n_cal).astype(int)]
    cal_imgs = {"wall": images(n_cal), "cam": cam, "both": images(n_cal // 2) + cam[::2]}[cal]
    qe = prepare(ops, cal_imgs)
    full, dec, dec_ins = sessions()

    h, w = frames.shape[1:3]
    xf0 = preprocess(frames[0])
    shapes = {k: v.shape for k, v in run_float(ops, xf0).items()}
    qe.run(xf0)
    s_x = qe.scale("x")
    scales = {k: qe.last_int[nm][1].reshape(-1, 1, 1) for k, nm in FEATS.items()}
    ch = tid = None
    if not sim:
        layers, params, tid = e.build(qe, shapes)
        ch = c.C16Chain(layers, params)

    st_ref, st_q = zero_state(h, w), zero_state(h, w)
    dr = np.array([1.0], np.float32)
    A_ref, A_q = [], []
    for n, im in enumerate(frames):
        src = (im.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        pha, *rs = full.run(["pha", "r1o", "r2o", "r3o", "r4o"], {"src": src, "downsample_ratio": dr, **st_ref})
        st_ref = {f"r{i + 1}i": r for i, r in enumerate(rs)}
        A_ref.append(pha[0, 0])
        xf = preprocess(im)
        if sim:
            qe.run(xf)
            feats = {k: qe.last_int[nm][0] for k, nm in FEATS.items()}
        else:
            xi = np.clip(np.round(xf / col(s_x)), -qe.qa, qe.qa).astype(np.int8)
            x = np.zeros((ch.layers[0].ic, h, w), np.int8)
            x[:xi.shape[0]] = xi
            ch.set_input(x)
            ch.call()
            feats = {k: e.read_tensor(ch, tid[nm])[:shapes[nm][0]] for k, nm in FEATS.items()}
        feed = {"src": src, "downsample_ratio": dr, **st_q}
        for k, nm in FEATS.items():
            feed[nm] = (feats[k] * scales[k]).astype(np.float32)[None]
        pha_q, *rs = dec.run(["pha", "r1o", "r2o", "r3o", "r4o"], {k: v for k, v in feed.items() if k in dec_ins})
        st_q = {f"r{i + 1}i": r for i, r in enumerate(rs)}
        A_q.append(pha_q[0, 0])
        if n % 50 == 0:
            print(f"  帧 {n}/{len(frames)}", flush=True)

    R, Q = np.stack(A_ref), np.stack(A_q)
    d = np.abs(Q - R)
    band = np.stack([edge_band(a) for a in R])
    iou = ((Q > .5) & (R > .5)).sum((1, 2)) / np.maximum(((Q > .5) | (R > .5)).sum((1, 2)), 1)
    flick = np.abs(np.diff(Q, axis=0) - np.diff(R, axis=0))
    print(f"\n{len(frames)} 帧, 校准 {cal} ({len(cal_imgs)} 张){', 整数模拟' if sim else ', NPU'}")
    print(f"  MAE 全图 {d.mean() * 255:.2f}/255   边缘带 {d[band].mean() * 255:.2f}/255   "
          f"非边缘 {d[~band].mean() * 255:.2f}/255   |差| > 0.1 的像素 {(d > .1).mean() * 100:.2f}%")
    print(f"  前景 IoU 平均 {iou.mean():.4f} (最差 {iou.min():.4f})   前 10 帧 MAE {d[:10].mean() * 255:.2f}  "
          f"后 50 帧 MAE {d[-50:].mean() * 255:.2f}")
    print(f"  时序: 参考帧间变化 {np.abs(np.diff(R, axis=0)).mean() * 255:.2f}/255, "
          f"量化版 {np.abs(np.diff(Q, axis=0)).mean() * 255:.2f}/255, 多出来的闪烁 {flick.mean() * 255:.2f}/255")
    if out:
        import cv2
        Path(out).mkdir(parents=True, exist_ok=True)
        idx = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
        g = lambda a: cv2.cvtColor((np.clip(a, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        rows = [np.concatenate([cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR), g(R[i]), g(Q[i]),
                                g(np.clip(d[i] * 4, 0, 1))], 1) for i in idx]
        cv2.imwrite(f"{out}/alpha_{cal}.png", np.concatenate(rows, 0))
        np.savez_compressed(f"{out}/alpha_{cal}.npz", ref=R, q=Q)
        print(f"  对比图 {out}/alpha_{cal}.png (每行: 画面 | 浮点 alpha | 量化 alpha | |差|×4)")


if __name__ == "__main__":
    main()
