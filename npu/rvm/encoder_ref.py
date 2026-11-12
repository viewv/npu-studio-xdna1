"""RVM (MobileNetV3) 编码器: 从 ONNX 里抽出算子图 + 权重, numpy 浮点参考, 对 onnxruntime 的中间结果

  source npu/env.sh && python -m npu.rvm.encoder_ref          # 浮点参考 vs onnxruntime

编码器 = Resize(1:1) → (x - mean) / std → Conv_8 … Mul_162 (张量 "594", 960 × H/16 × W/16)。
解码器要的 4 个特征: f1 = "403" (16 @ H/2), f2 = "420" (24 @ H/4), f3 = "464" (40 @ H/8), f4 = "594" (960 @ H/16)。
ONNX 里的 BN 已经折进卷积 (每个 Conv 都带 bias)。

图表示 (Op 列表, 拓扑序), 每个 Op 一个输出张量:
  conv   (x, W[oc, ic/g, k, k], b[oc], s, d, g)
  relu / hswish (x)            hardswish 在 ONNX 里是 HardSigmoid(alpha=1/6) + Mul(x, ·)
  hsig   (x)                   SE 的 HardSigmoid (后面的 Mul 是 scale)
  gap    (x)                   GlobalAveragePool
  mul    (x, s)                SE: x · s[c]
  add    (a, b)                残差
"""
from __future__ import annotations

import glob
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parents[2] / "models"
ONNX = MODELS / "rvm_mobilenetv3_fp32.onnx"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FEATS = {"f1": "403", "f2": "420", "f3": "464", "f4": "594"}


@dataclass
class Op:
    kind: str
    out: str
    ins: list
    name: str = ""
    W: np.ndarray | None = None
    b: np.ndarray | None = None
    s: int = 1
    d: int = 1
    g: int = 1
    attrs: dict = field(default_factory=dict)


def load_graph(path=ONNX, last="594") -> list[Op]:
    """Conv_8 起到张量 last 为止的编码器子图。输入张量叫 "x" (= 归一化后的 3 通道图)"""
    import onnx
    from onnx import numpy_helper as nh
    m = onnx.load(str(path))
    W = {i.name: nh.to_array(i) for i in m.graph.initializer}
    nodes = list(m.graph.node)
    start = next(i for i, n in enumerate(nodes) if n.name == "Conv_8")
    rename = {nodes[start].input[0]: "x"}
    ops, i = [], start
    while True:
        n = nodes[i]
        ins = [rename.get(x, x) for x in n.input]
        a = {at.name: onnx.helper.get_attribute_value(at) for at in n.attribute}
        if n.op_type == "Conv":
            ops.append(Op("conv", n.output[0], ins[:1], n.name, W[n.input[1]],
                          W[n.input[2]] if len(n.input) > 2 else np.zeros(W[n.input[1]].shape[0], np.float32),
                          a["strides"][0], a["dilations"][0], a["group"]))
        elif n.op_type == "Relu":
            ops.append(Op("relu", n.output[0], ins, n.name))
        elif n.op_type == "HardSigmoid":
            nxt = nodes[i + 1]
            assert abs(a.get("alpha", 0.2) - 1 / 6) < 1e-6 and nxt.op_type == "Mul"
            if set(nxt.input) == {n.input[0], n.output[0]}:          # x · hsig(x) = hardswish
                ops.append(Op("hswish", nxt.output[0], ins, n.name))
            else:                                                     # SE: hsig(fc2) · x
                other = [x for x in nxt.input if x != n.output[0]][0]
                ops.append(Op("hsig", n.output[0], ins, n.name))
                ops.append(Op("mul", nxt.output[0], [rename.get(other, other), n.output[0]], nxt.name))
            i += 1
        elif n.op_type == "GlobalAveragePool":
            ops.append(Op("gap", n.output[0], ins, n.name))
        elif n.op_type == "Add":
            ops.append(Op("add", n.output[0], ins, n.name))
        else:
            raise ValueError(f"编码器里没想到的算子 {n.op_type} {n.name}")
        if ops[-1].out == last:
            return ops
        i += 1


def conv2d(x: np.ndarray, W: np.ndarray, b, s=1, d=1, g=1) -> np.ndarray:
    """x (C, H, W) 浮点或整数; same padding = d·(k-1)/2; 返回 (OC, Ho, Wo)"""
    c, h, w = x.shape
    oc, icg, k, _ = W.shape
    p = d * (k - 1) // 2
    ho, wo = -(-h // s), -(-w // s)
    acc_t = np.float64 if np.issubdtype(x.dtype, np.floating) else np.int64
    xp = np.zeros((c, h + 2 * p, w + 2 * p), acc_t)
    xp[:, p:p + h, p:p + w] = x
    out = np.zeros((oc, ho * wo), acc_t)
    Wt = W.astype(acc_t)
    ocg = oc // g
    for ky in range(k):
        for kx in range(k):
            patch = xp[:, ky * d: ky * d + s * (ho - 1) + 1: s, kx * d: kx * d + s * (wo - 1) + 1: s].reshape(c, -1)
            if g == 1:
                out += Wt[:, :, ky, kx] @ patch
            elif g == c and icg == 1:                                 # depthwise
                out += Wt[:, 0, ky, kx][:, None] * patch
            else:
                for gi in range(g):
                    out[gi * ocg:(gi + 1) * ocg] += Wt[gi * ocg:(gi + 1) * ocg, :, ky, kx] @ patch[gi * icg:(gi + 1) * icg]
    if b is not None:
        out += np.asarray(b, acc_t)[:, None]
    return out.reshape(oc, ho, wo)


def hardswish(x):
    return x * np.clip(x / 6 + 0.5, 0, 1)


def run_float(ops: list[Op], x: np.ndarray) -> dict[str, np.ndarray]:
    """x: (3, H, W) 归一化后的图 → 所有张量 (float32)"""
    t = {"x": x.astype(np.float32)}
    for op in ops:
        a = t[op.ins[0]]
        if op.kind == "conv":
            r = conv2d(a, op.W, op.b, op.s, op.d, op.g)
        elif op.kind == "relu":
            r = np.maximum(a, 0)
        elif op.kind == "hswish":
            r = hardswish(a)
        elif op.kind == "hsig":
            r = np.clip(a / 6 + 0.5, 0, 1)
        elif op.kind == "gap":
            r = a.mean(axis=(1, 2), keepdims=True)
        elif op.kind == "mul":
            r = a * t[op.ins[1]]
        elif op.kind == "add":
            r = a + t[op.ins[1]]
        t[op.out] = r.astype(np.float32)
    return t


def preprocess(rgb_u8: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB uint8 → (3, H, W) 归一化 (等价于 ONNX 里 Resize(1:1) + Sub + Div)"""
    x = rgb_u8.astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)


def images(n=None, w=320, h=180, offset=0) -> list[np.ndarray]:
    """系统壁纸 (每个主题一张) 缩到工作分辨率, RGB uint8。没有人像: 只用来看数值误差"""
    import cv2
    files = []
    for d in sorted(glob.glob("/usr/share/wallpapers/*/contents/images")):
        fs = sorted(glob.glob(d + "/*.jpg") + glob.glob(d + "/*.png"))
        if fs:
            files.append(fs[len(fs) // 2])
    files = files[offset:][:n] if n else files[offset:]
    out = []
    for f in files:
        im = cv2.imread(f)
        if im is None:
            continue
        sc = max(w / im.shape[1], h / im.shape[0])                    # 等比缩放 + 中心裁剪
        im = cv2.resize(im, (round(im.shape[1] * sc), round(im.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        y0, x0 = (im.shape[0] - h) // 2, (im.shape[1] - w) // 2
        out.append(cv2.cvtColor(im[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2RGB))
    return out


def ort_tensors(rgb_u8: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """onnxruntime 跑整个模型, 把 names 里的中间张量也拿出来"""
    import onnx
    import onnxruntime as ort
    m = onnx.load(str(ONNX))
    have = {o.name for o in m.graph.output}
    for nm in names:
        if nm not in have:
            m.graph.output.append(onnx.helper.make_tensor_value_info(nm, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    h, w = rgb_u8.shape[:2]
    src = (rgb_u8.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    feed = {"src": src, "downsample_ratio": np.array([1.0], np.float32)}
    for i, c in enumerate((16, 20, 40, 64)):
        feed[f"r{i + 1}i"] = np.zeros((1, c, -(-h // 2 ** (i + 1)), -(-w // 2 ** (i + 1))), np.float32)
    outs = sess.run(names, feed)
    return {nm: o[0] for nm, o in zip(names, outs)}


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ops = load_graph()
    kinds = {}
    for op in ops:
        kinds[op.kind] = kinds.get(op.kind, 0) + 1
    print(f"编码器 {len(ops)} 个算子: {kinds}")
    im = images(1)[0]
    t = run_float(ops, preprocess(im))
    ref = ort_tensors(im, list(FEATS.values()))
    for k, nm in FEATS.items():
        d = np.abs(t[nm] - ref[nm]).max()
        print(f"  {k} {nm} {t[nm].shape}: 最大差 {d:.2e}  余弦 {cos(t[nm], ref[nm]):.7f}  范围 ±{np.abs(ref[nm]).max():.2f}")


if __name__ == "__main__":
    sys.exit(main())
