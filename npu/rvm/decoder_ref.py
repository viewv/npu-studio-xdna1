"""RVM (MobileNetV3) 解码器的浮点参考 (numpy), 结构手写、权重从 ONNX 按节点名取, 对上 onnxruntime

  source npu/env.sh && python -m npu.rvm.decoder_ref [帧.npz]      # 连续帧 (带循环状态) 和 onnxruntime 整网比

输入 (都是 (C, H, W) float32): src = 0..1 的 RGB (工作分辨率, downsample_ratio = 1 时等于 ONNX 里的 389),
f1..f4 = 编码器特征 (encoder_ref.FEATS), r1..r4 = 上一帧的循环状态。
结构 (BN 已经折进卷积):
  s1..s3 = src 的 2×2 平均池化 (ceil, 边上不足的按实际个数平均) 1/2, 1/4, 1/8
  decode4 @1/16: x = relu(Conv_163(f4)) · sigmoid(Conv_166(GAP(f4)))  (LRASPP);  a, b = x 的前 / 后 64 通道
                 h = GRU(b, r4);  out4 = [a ‖ h]
  decode3..1:    up = 双线性 ×2 (pytorch_half_pixel) 再裁到下一级大小;  x = relu(conv([up ‖ f_k ‖ s_k]))
                 a, b = 前 / 后一半;  h = GRU(b, r_k);  out = [a ‖ h]
  GRU(b, h):     r, z = sigmoid(conv_ih([b ‖ h])) 的前 / 后一半;  c = tanh(conv_hh([b ‖ r·h]));  h' = (1 − z)·h + z·c
  输出 @1/1:     up = ×2 → relu(Conv_290([up ‖ src])) → relu(Conv_292) = hid → Conv_294 → [fgr 残差 3, pha 1]
  细化 (DGF, ratio = 1 时低分辨率 = 高分辨率, 两个 Resize 是恒等):
     X = [src ‖ mean_c(src)], Y = [fgr 残差 ‖ pha];  box = 3×3 均值 (零填充, /9)
     cov = box(X·Y) − box(X)·box(Y),  var = box(X·X) − box(X)²
     A = Conv_316(relu(Conv_314(relu(Conv_312([cov ‖ var ‖ hid])))));  B = box(Y) − A·box(X)
     out = A·X + B → fgr = clip(out[:3] + src, 0, 1), pha = clip(out[3], 0, 1)
"""
from __future__ import annotations

import sys

import numpy as np

from .encoder_ref import FEATS, ONNX, conv2d

CONVS = ["Conv_163", "Conv_166", "Conv_176", "Conv_181", "Conv_200", "Conv_206", "Conv_211", "Conv_230", "Conv_236",
         "Conv_241", "Conv_260", "Conv_266", "Conv_271", "Conv_290", "Conv_292", "Conv_294", "Conv_301", "Conv_312",
         "Conv_314", "Conv_316"]
# 每一级: (上采样输入块, 融合卷积, GRU ih, GRU hh)
BLOCKS = {3: ("Conv_200", "Conv_206", "Conv_211"), 2: ("Conv_230", "Conv_236", "Conv_241"),
          1: ("Conv_260", "Conv_266", "Conv_271")}


def load_weights(path=ONNX) -> dict:
    """{卷积节点名: (W, b 或 None)}"""
    import onnx
    from onnx import numpy_helper as nh
    m = onnx.load(str(path))
    init = {i.name: nh.to_array(i) for i in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    init[n.output[0]] = nh.to_array(a.t)
    out = {}
    for n in m.graph.node:
        if n.name in CONVS:
            out[n.name] = (init[n.input[1]].astype(np.float32),
                           init[n.input[2]].astype(np.float32) if len(n.input) > 2 else None)
    assert len(out) == len(CONVS), set(CONVS) - set(out)
    return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def avgpool2(x):
    """2×2 stride 2, ceil_mode, 不补零 (边上不足 2 个的按实际个数平均)"""
    c, h, w = x.shape
    ho, wo = -(-h // 2), -(-w // 2)
    xp = np.zeros((c, 2 * ho, 2 * wo), np.float64)
    n = np.zeros((2 * ho, 2 * wo))
    xp[:, :h, :w], n[:h, :w] = x, 1
    s = xp.reshape(c, ho, 2, wo, 2).sum((2, 4))
    return (s / n.reshape(ho, 2, wo, 2).sum((1, 3))).astype(np.float32)


def up2(x, h, w):
    """双线性 ×2 (pytorch_half_pixel: 输出 i 取输入 i/2 − 1/4, 即 3:1 加权、边上夹住), 再裁到 (h, w)"""
    def axis(a, ax):
        n = a.shape[ax]
        lo = np.take(a, np.r_[0, np.arange(n - 1)], ax)      # 前一个 (夹住)
        hi = np.take(a, np.r_[np.arange(1, n), n - 1], ax)   # 后一个
        even, odd = 0.75 * a + 0.25 * lo, 0.75 * a + 0.25 * hi
        return np.stack([even, odd], ax + 1).reshape(*a.shape[:ax], 2 * n, *a.shape[ax + 1:])
    return axis(axis(x, 1), 2)[:, :h, :w].astype(np.float32)


def gru(Wt, ih, hh, b, h):
    rz = sigmoid(conv2d(np.concatenate([b, h]), *Wt[ih]))
    r, z = np.split(rz, 2)
    c = np.tanh(conv2d(np.concatenate([b, r * h]), *Wt[hh]))
    return ((1 - z) * h + z * c).astype(np.float32)


def run(Wt, src, f, r) -> dict:
    """src (3,H,W) 0..1; f = {f1..f4}; r = {r1..r4} → 所有中间量 (含 pha, fgr, r1o..r4o)"""
    t = {}
    s = {1: avgpool2(src)}
    s[2], s[3] = avgpool2(s[1]), None
    s[3] = avgpool2(s[2])
    x = np.maximum(conv2d(f["f4"], *Wt["Conv_163"]), 0) * sigmoid(conv2d(f["f4"].mean((1, 2), keepdims=True),
                                                                           *Wt["Conv_166"]))
    a, b = np.split(x.astype(np.float32), 2)
    t["r4o"] = gru(Wt, "Conv_176", "Conv_181", b, r["r4"])
    out = np.concatenate([a, t["r4o"]])
    for k in (3, 2, 1):
        fk = f[f"f{k}"]
        up = up2(out, *fk.shape[1:])
        conv, ih, hh = BLOCKS[k]
        x = np.maximum(conv2d(np.concatenate([up, fk, s[k]]), *Wt[conv]), 0).astype(np.float32)
        a, b = np.split(x, 2)
        t[f"r{k}o"] = gru(Wt, ih, hh, b, r[f"r{k}"])
        out = np.concatenate([a, t[f"r{k}o"]])
    up = up2(out, *src.shape[1:])
    hid = np.maximum(conv2d(np.concatenate([up, src]), *Wt["Conv_290"]), 0)
    hid = np.maximum(conv2d(hid, *Wt["Conv_292"]), 0).astype(np.float32)
    y = conv2d(hid, *Wt["Conv_294"]).astype(np.float32)
    t["hid"], t["y"] = hid, y
    # DGF 细化
    X = np.concatenate([src, src.mean(0, keepdims=True)])
    box = lambda v: conv2d(v, Wt["Conv_301"][0], None, g=4)
    mx, my = box(X), box(y)
    cov, var = box(X * y) - mx * my, box(X * X) - mx * mx
    A = np.maximum(conv2d(np.concatenate([cov, var, hid]), *Wt["Conv_312"]), 0)
    A = np.maximum(conv2d(A, *Wt["Conv_314"]), 0)
    A = conv2d(A, *Wt["Conv_316"])
    B = my - A * mx
    o = A * X + B
    t["fgr"] = np.clip(o[:3] + src, 0, 1).astype(np.float32)
    t["pha"] = np.clip(o[3:], 0, 1).astype(np.float32)
    t["pha_raw"] = np.clip(y[3:], 0, 1).astype(np.float32)          # 不经过细化的 alpha (对照用)
    return t


def ort_full():
    import onnx
    import onnxruntime as ort
    m = onnx.load(str(ONNX))
    have = {o.name for o in m.graph.output}
    for nm in FEATS.values():
        if nm not in have:
            m.graph.output.append(onnx.helper.make_tensor_value_info(nm, onnx.TensorProto.FLOAT, None))
    return ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])


def zero_state(h, w):
    return {f"r{i + 1}": np.zeros((ch, -(-h // 2 ** (i + 1)), -(-w // 2 ** (i + 1))), np.float32)
            for i, ch in enumerate((16, 20, 40, 64))}


def main():
    from .encoder_ref import images
    frames = np.load(sys.argv[1])["rgb"][:20] if len(sys.argv) > 1 else np.stack(images(4))
    Wt = load_weights()
    sess = ort_full()
    h, w = frames.shape[1:3]
    st_o, st_n = zero_state(h, w), zero_state(h, w)
    names = ["fgr", "pha", "r1o", "r2o", "r3o", "r4o", *FEATS.values()]
    worst = {}
    for im in frames:
        src = (im.astype(np.float32) / 255).transpose(2, 0, 1)
        feed = {"src": src[None], "downsample_ratio": np.array([1.0], np.float32),
                **{f"{k}i": v[None] for k, v in st_o.items()}}
        o = dict(zip(names, (v[0] for v in sess.run(names, feed))))
        f = {k: o[nm] for k, nm in FEATS.items()}               # 编码器特征用 onnxruntime 的, 只验解码器
        t = run(Wt, src, f, st_n)
        for k in ("pha", "fgr", "r1o", "r2o", "r3o", "r4o"):
            worst[k] = max(worst.get(k, 0), float(np.abs(t[k] - o[k]).max()))
        worst["pha_raw vs pha"] = max(worst.get("pha_raw vs pha", 0), float(np.abs(t["pha_raw"] - o["pha"]).max()))
        st_o = {f"r{i}": o[f"r{i}o"] for i in range(1, 5)}
        st_n = {f"r{i}": t[f"r{i}o"] for i in range(1, 5)}
    print(f"{len(frames)} 帧, numpy 解码器 vs onnxruntime 最大差: " + "  ".join(f"{k} {v:.2e}" for k, v in worst.items()))


if __name__ == "__main__":
    main()
