"""PP-OCRv4 Mobile 文本识别 (中英文) 在 XDNA1 NPU 上运行:

  source npu/env.sh && .venv-iron/bin/python -m npu.ocr.rec_npu [--iters 20]

结构:
  1. PP-LCNet 主干 (ONNX nodes[234..611], 33 个 Conv + 2 个 SE, 含 s=(2,1)/(1,2) 非对称步长):
     全部折叠 BN / mobile_one_block / whswish_b 仿射后, 用对称 int8 量化跑在 4 核 NPU (chain_c16) 上。
  2. SVTR + CTC 头 (ONNX nodes[612..859], AveragePool(3,2) + 2 层 D=120 SVTR + 6625 类线性):
     用 onnxruntime 子图在 CPU 上解码, 对比全浮点 onnxruntime 的识别文本与 NPU 耗时。
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnx
from onnx import helper, numpy_helper as nh, TensorProto
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

from .. import chain_c16 as c

REC_ONNX = "/home/viewv/.npu/ocr/ch_PP-OCRv4_rec.onnx"
POOL_S = 22
ACT = {"relu": 4, "hswish": 3, "hsig": 1, None: 2}
FONT_CANDIDATES = [
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/TTF/LXGWWenKai-Regular.ttf",
    "/usr/share/fonts/noto-cjk/NotoSerifCJK-Black.ttc",
]


def pad16(n: int) -> int:
    return -(-n // 16) * 16


def rnd_shift(v, s):
    return (v + (1 << (s - 1))) >> s if s > 0 else v << (-s)


def col(a):
    return np.asarray(a, np.float64).reshape(-1, 1, 1)


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@dataclass
class Op:
    kind: str
    out: str
    ins: list[str]
    name: str
    W: np.ndarray | None = None
    b: np.ndarray | None = None
    sy: int = 1
    sx: int = 1
    d: int = 1
    g: int = 1


def load_dict(m: onnx.ModelProto) -> list[str]:
    for p in m.metadata_props:
        if p.key == "character":
            chars = p.value.splitlines()
            return ["blank"] + chars + [" "]
    raise ValueError("ONNX metadata_props 中未找到 character 词表")


def load_rec_backbone(path: str = REC_ONNX):
    """解析 nodes[234..611] 为 Op 列表, 折叠 BN / mobile_one_block / whswish_b 仿射。
    返回 (ops, last_tensor_name, tail_affine (w, b), dict_chars)"""
    m = onnx.load(path)
    chars = load_dict(m)
    nodes = list(m.graph.node)
    consts = {i.name: nh.to_array(i).astype(np.float64) for i in m.graph.initializer}
    for n in nodes:
        if n.op_type == "Constant":
            consts[n.output[0]] = nh.to_array(n.attribute[0].t).astype(np.float64)

    ops: list[Op] = []
    # affine[tensor_name] = (base_tensor_name, w_scalar_or_vec, b_scalar_or_vec)
    affine: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}

    def resolve(t: str):
        if t in affine:
            return affine[t]
        return t, np.array(1.0), np.array(0.0)

    i = 234
    while i < 612:
        n = nodes[i]
        if n.op_type == "Constant":
            i += 1
            continue
        if n.op_type == "Conv":
            W = consts[n.input[1]].copy()
            b = consts[n.input[2]].copy() if len(n.input) > 2 else np.zeros(W.shape[0], np.float64)
            a = {at.name: helper.get_attribute_value(at) for at in n.attribute}
            sy, sx = a.get("strides", [1, 1])
            d = a.get("dilations", [1, 1])[0]
            g = a.get("group", 1)
            src, w_in, b_in = resolve(n.input[0])
            # 折叠前置仿射 y = w_in * x + b_in
            if np.any(b_in != 0):
                b = b + (W * b_in.reshape(1, -1, 1, 1) if g == 1 else W * b_in.reshape(-1, 1, 1, 1)).sum(axis=(1, 2, 3))
            if np.any(w_in != 1):
                W = W * w_in.reshape(1, -1, 1, 1) if g == 1 else W * w_in.reshape(-1, 1, 1, 1)
            out_name = n.output[0]
            i += 1
            # 折叠紧随其后的 BatchNormalization 或 Mul+Add
            if i < 612 and nodes[i].op_type == "BatchNormalization":
                bn = nodes[i]
                gamma = consts[bn.input[1]]
                beta = consts[bn.input[2]]
                mean = consts[bn.input[3]]
                var = consts[bn.input[4]]
                eps = {at.name: helper.get_attribute_value(at) for at in bn.attribute}.get("epsilon", 1e-5)
                scale = gamma / np.sqrt(var + eps)
                W = W * scale.reshape(-1, 1, 1, 1)
                b = (b - mean) * scale + beta
                out_name = bn.output[0]
                i += 1
            elif i + 1 < 612 and nodes[i].op_type == "Mul" and nodes[i + 1].op_type == "Add":
                mul_n, add_n = nodes[i], nodes[i + 1]
                if out_name in mul_n.input and mul_n.output[0] in add_n.input:
                    w_name = [x for x in mul_n.input if x != out_name][0]
                    b_name = [x for x in add_n.input if x != mul_n.output[0]][0]
                    wm = consts[w_name].reshape(-1)
                    ba = consts[b_name].reshape(-1)
                    W = W * wm.reshape(-1, 1, 1, 1)
                    b = b * wm + ba
                    out_name = add_n.output[0]
                    i += 2
            ops.append(Op("conv", out_name, [src], n.name, W.astype(np.float32), b.astype(np.float32), sy, sx, d, g))
            continue

        if n.op_type == "Add" and i + 6 < 612 and nodes[i + 3].op_type == "Clip" and nodes[i + 4].op_type == "Mul" and nodes[i + 5].op_type == "Div":
            # HardSwish: Add(+3) -> Clip(0,6) -> Mul -> Div(6)
            x_in = n.input[0] if n.input[0] not in consts else n.input[1]
            div_n = nodes[i + 5]
            h_out = div_n.output[0]
            ops.append(Op("hswish", h_out, [x_in], div_n.name))
            i += 6
            # 检查紧随其后的 whswish_b: Mul(w_0) + Add(w_1)
            if i + 1 < 612 and nodes[i].op_type == "Mul" and nodes[i + 1].op_type == "Add":
                mul_n, add_n = nodes[i], nodes[i + 1]
                if h_out in mul_n.input and mul_n.output[0] in add_n.input:
                    w_name = [x for x in mul_n.input if x != h_out][0]
                    b_name = [x for x in add_n.input if x != mul_n.output[0]][0]
                    wa = float(consts[w_name].reshape(-1)[0])
                    ba = float(consts[b_name].reshape(-1)[0])
                    if abs(ba) <= 1e-4:
                        affine[add_n.output[0]] = (h_out, np.array(wa), np.array(0.0))
                    else:
                        # 若后续是 DWConv 或 SE (非 1x1 普通卷积), 插入 1x1 对角卷积保持边界与 SE 精确
                        nxt_op = nodes[i + 2] if i + 2 < 612 else None
                        is_1x1 = (
                            nxt_op is not None and nxt_op.op_type == "Conv"
                            and consts[nxt_op.input[1]].shape[2] == 1
                            and {at.name: helper.get_attribute_value(at) for at in nxt_op.attribute}.get("group", 1) == 1
                        )
                        if is_1x1 or i + 2 >= 612:
                            affine[add_n.output[0]] = (h_out, np.array(wa), np.array(ba))
                        else:
                            ch = ops[-1].W.shape[0] if ops[-1].kind == "conv" else len(ops[-1].ins)
                            # 从上一个 conv 获取通道数
                            for prev_op in reversed(ops):
                                if prev_op.kind == "conv":
                                    ch = prev_op.W.shape[0]
                                    break
                            W_id = np.eye(ch, dtype=np.float32).reshape(ch, ch, 1, 1) * wa
                            b_id = np.full(ch, ba, dtype=np.float32)
                            ops.append(Op("conv", add_n.output[0], [h_out], add_n.name + ".aff", W_id, b_id, 1, 1, 1, 1))
                    i += 2
            continue

        if n.op_type == "GlobalAveragePool":
            src, w_in, b_in = resolve(n.input[0])
            ops.append(Op("gap", n.output[0], [src], n.name))
            if np.any(w_in != 1) or np.any(b_in != 0):
                affine[n.output[0]] = (n.output[0], w_in, b_in)
            i += 1
            continue

        if n.op_type == "Relu":
            ops.append(Op("relu", n.output[0], list(n.input), n.name))
            i += 1
            continue

        if n.op_type == "HardSigmoid":
            nxt = nodes[i + 1]
            assert nxt.op_type == "Mul"
            other = [x for x in nxt.input if x != n.output[0]][0]
            src_other, w_in, b_in = resolve(other)
            ops.append(Op("hsig", n.output[0], list(n.input), n.name))
            ops.append(Op("mul", nxt.output[0], [src_other, n.output[0]], nxt.name))
            # SE 分支中的 b_in * se: 将 w_in 折叠到下一层 Conv, b_in * 0.5 近似偏置
            if np.any(w_in != 1) or np.any(b_in != 0):
                affine[nxt.output[0]] = (nxt.output[0], w_in, b_in * 0.5)
            i += 2
            continue

        raise ValueError(f"未预期的节点 {i}: {n.op_type} {n.name}")

    tail_name, w_tail, b_tail = resolve("p2o.Add.231")
    return ops, tail_name, (w_tail.astype(np.float32), b_tail.astype(np.float32)), chars


def build_rec_head(path: str = REC_ONNX) -> ort.InferenceSession:
    """提取 nodes[612..859] (AveragePool + SVTR + CTC) 作为独立 ONNX 会话"""
    m = onnx.load(path)
    nodes = list(m.graph.node)
    consts = [n for n in nodes[:612] if n.op_type == "Constant"]
    needed_inits = set()
    for n in nodes[612:]:
        for inp in n.input:
            needed_inits.add(inp)
    inits = [i for i in m.graph.initializer if i.name in needed_inits]
    head_consts = [n for n in consts if n.output[0] in needed_inits]

    inp = helper.make_tensor_value_info("p2o.Add.231", TensorProto.FLOAT, [1, 480, 3, 80])
    out = m.graph.output[0]
    g = helper.make_graph(head_consts + nodes[612:], "rec_head", [inp], [out], initializer=inits)
    hm = helper.make_model(g, opset_imports=m.opset_import, ir_version=m.ir_version)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(hm.SerializeToString(), opts, providers=["CPUExecutionProvider"])


def conv2d(x: np.ndarray, W: np.ndarray, b, sy=1, sx=1, d=1, g=1) -> np.ndarray:
    c_in, h, w = x.shape
    oc, icg, k, _ = W.shape
    p = d * (k - 1) // 2
    ho, wo = -(-h // sy), -(-w // sx)
    acc_t = np.float64 if np.issubdtype(x.dtype, np.floating) else np.int64
    xp = np.zeros((c_in, h + 2 * p, w + 2 * p), acc_t)
    xp[:, p:p + h, p:p + w] = x
    out = np.zeros((oc, ho * wo), acc_t)
    Wt = W.astype(acc_t)
    for ky in range(k):
        for kx in range(k):
            patch = xp[:, ky * d: ky * d + sy * (ho - 1) + 1: sy, kx * d: kx * d + sx * (wo - 1) + 1: sx].reshape(c_in, -1)
            if g == 1:
                out += Wt[:, :, ky, kx] @ patch
            else:
                out += Wt[:, 0, ky, kx][:, None] * patch
    if b is not None:
        out += np.asarray(b, acc_t)[:, None]
    return out.reshape(oc, ho, wo)


def hardswish(x):
    return x * np.clip(x / 6.0 + 0.5, 0.0, 1.0)


def run_float(ops: list[Op], x: np.ndarray) -> dict[str, np.ndarray]:
    T = {"x": x.astype(np.float64)}
    for op in ops:
        if op.kind == "conv":
            T[op.out] = conv2d(T[op.ins[0]], op.W, op.b, op.sy, op.sx, op.d, op.g)
        elif op.kind == "relu":
            T[op.out] = np.maximum(T[op.ins[0]], 0.0)
        elif op.kind == "hswish":
            T[op.out] = hardswish(T[op.ins[0]])
        elif op.kind == "hsig":
            T[op.out] = np.clip(T[op.ins[0]] / 6.0 + 0.5, 0.0, 1.0)
        elif op.kind == "gap":
            T[op.out] = T[op.ins[0]].mean(axis=(1, 2), keepdims=True)
        elif op.kind == "mul":
            T[op.out] = T[op.ins[0]] * T[op.ins[1]]
    return T


class QRec:
    """对称 int8 + m16 重量化模拟与参数冻结"""

    def __init__(self, ops: list[Op], calib: list[dict[str, np.ndarray]], pct: float = 99.95):
        self.ops = ops
        self.byname = {op.out: op for op in ops}
        self.users: dict[str, list[Op]] = {}
        for op in ops:
            for inp in op.ins:
                self.users.setdefault(inp, []).append(op)
        self.amax: dict[str, np.ndarray] = {}
        self.accmax: dict[str, int] = {}
        self.params: dict[str, tuple] = {}
        self.export: dict[str, dict] = {}
        self.frozen = False

        for nm in calib[0]:
            ch = calib[0][nm].shape[0]
            v = np.abs(np.stack([c_d[nm] for c_d in calib], 1)).reshape(ch, -1)
            a = np.percentile(v, pct, axis=1) if pct < 100.0 else v.max(1)
            self.amax[nm] = np.maximum(a, a.max() * 1e-3 + 1e-8)

    def scale(self, nm: str) -> np.ndarray:
        return self.amax[nm] / 127.0

    def requant(self, acc: np.ndarray, m: np.ndarray, y_lim=None, key: str | None = None) -> np.ndarray:
        lo, hi = y_lim or (-128, 127)
        m = col(m)
        if key is not None and not self.frozen:
            self.accmax[key] = max(self.accmax.get(key, 1), int(np.abs(acc).max()))
        amax = 2 * self.accmax[key] if (key in self.accmax and self.frozen) else max(int(np.abs(acc).max()), 1)
        s1 = max(0, int(np.ceil(np.log2(amax / 32767.0))))
        t = np.clip(rnd_shift(acc, s1), -32768, 32767)
        mm = m * (1 << s1)
        s2 = max(0, int(np.floor(np.log2(32767.0 / max(float(mm.max()), 1e-12)))))
        M = np.clip(np.round(mm * (1 << s2)), 0, 32767).astype(np.int64)
        y = rnd_shift(t * M, s2) if s2 > 0 else t * M
        if key is not None:
            self.params[key] = (s1, M.ravel(), s2)
        return np.clip(y, lo, hi).astype(np.int64)

    def wquant(self, W: np.ndarray):
        oc = W.shape[0]
        amax_c = np.abs(W.reshape(oc, -1)).max(1)
        amax_c = np.maximum(amax_c, amax_c.max() * 1e-3 + 1e-30)
        s_w = amax_c / 127.0
        Wq = np.clip(np.round(W / s_w.reshape(-1, 1, 1, 1)), -127, 127).astype(np.int64)
        return Wq, s_w

    def run(self, x_float: np.ndarray) -> dict[str, np.ndarray]:
        s_in = self.scale("x")
        T = {"x": (np.clip(np.round(x_float / col(s_in)), -127, 127).astype(np.int64), s_in)}
        done = set()
        for op in self.ops:
            if op.out in done:
                continue
            if op.kind == "conv":
                self._conv(op, T, done)
            elif op.kind == "gap":
                q, s = T[op.ins[0]]
                h, w = q.shape[1:]
                M = round((1 << POOL_S) / (h * w))
                T[op.out] = (np.clip(rnd_shift(q.reshape(q.shape[0], -1).sum(1) * M, POOL_S), -128, 127).reshape(-1, 1, 1), s)
            elif op.kind == "mul":
                pass
            else:
                raise ValueError(f"未融合的算子: {op.kind} {op.name}")
            done.add(op.out)
        self.last_int = T
        return {k: T[k][0] * col(T[k][1]) for k in T}

    def _conv(self, op: Op, T, done):
        src = op.ins[0]
        se = None
        if src in self.byname and self.byname[src].kind == "mul":
            mul = self.byname[src]
            src, se = mul.ins[0], T[mul.ins[1]][0].ravel()
            done.add(mul.out)
        xq, s_x = T[src]
        users = self.users.get(op.out, [])
        act = users[0] if len(users) == 1 and users[0].kind in ("relu", "hswish", "hsig") else None
        final = (act or op).out

        W = op.W.astype(np.float64)
        W = W * s_x.reshape(1, -1, 1, 1) if op.g == 1 else W * s_x.reshape(-1, 1, 1, 1)
        Wq, s_w = self.wquant(W)
        Wq_raw = Wq
        if se is not None:
            Wq = rnd_shift(Wq * se.reshape(1, -1, 1, 1), 8)
        bq = np.round(op.b / s_w).astype(np.int64)
        acc = conv2d(xq, Wq, bq, op.sy, op.sx, op.d, op.g)

        ex = self.export.setdefault(op.name, {})
        ex.update(W=Wq_raw, b=bq, src=src, se=(self.byname[op.ins[0]].ins[1] if se is not None else None),
                  act=(act.kind if act is not None else None), out=final)

        if act is not None and act.kind == "relu":
            s_p = self.scale(act.out)
            T[act.out] = (self.requant(acc, s_w / s_p, (0, 127), op.name), s_p)
            done.add(act.out)
        elif act is not None and act.kind == "hswish":
            q = np.clip(np.floor(127.0 / self.amax[op.out]), 1, 4096).astype(np.int64)
            ex.update(q=q)
            y8 = self.requant(acc, s_w * q, None, op.name)
            qc = col(q).astype(np.int64)
            yt = y8 * np.clip(y8 + 3 * qc, 0, 6 * qc)
            s_o = self.scale(act.out)
            T[act.out] = (self.requant(yt, 1.0 / (6.0 * q * q) / s_o, None, op.name + ".hs"), s_o)
            done.add(act.out)
        elif act is not None and act.kind == "hsig":
            y8 = self.requant(acc, s_w / (6.0 / 256.0), (-128, 127), op.name)
            T[act.out] = (y8 + 128, np.full(len(s_w), 1.0 / 256.0))
            done.add(act.out)
        else:
            s_p = self.scale(final)
            T[final] = (self.requant(acc, s_w / s_p, None, op.name), s_p)
        T[op.out] = T[final]


def _pad(v, n):
    out = np.zeros(n, np.int64)
    v = np.asarray(v, np.int64).ravel()
    out[:v.size] = v
    return out


def build_chain(qr: QRec, shapes: dict[str, tuple[int, int, int]]) -> tuple[list, list, dict]:
    tid, chans = {"x": 0}, {"x": shapes["x"][0]}
    layers, params = [], []
    for op in qr.ops:
        k = len(layers)
        if op.kind == "gap":
            src = op.ins[0]
            ch, h, w = shapes[src]
            layers.append(c.Layer(pad16(ch), pad16(ch), 1, 1, h, w, op=1, src=tid[src] - (k + 1)))
            params.append(c.LParam(None))
            tid[op.out], chans[op.out] = k + 1, ch
            continue
        if op.kind != "conv":
            continue
        ex = qr.export[op.name]
        src, se = ex["src"], ex["se"]
        ci, (h, w) = chans[src], shapes[src][1:]
        co, kk = op.W.shape[0], op.W.shape[2]
        dw = op.g > 1
        ica, oc = pad16(ci), pad16(co)
        layers.append(c.Layer(ica, oc, kk, op.sx, h, w, dw=int(dw), d=op.d, act=ACT[ex["act"]],
                              src=tid[src] - (k + 1), sc=tid[se] - (k + 1) if se else 0,
                              sy=op.sy if op.sy != op.sx else 0))
        W = np.zeros((oc, 1 if dw else ica, kk, kk), np.int8)
        W[:co, :ex["W"].shape[1]] = ex["W"]
        s1, M, s2 = qr.params[op.name]
        p = c.LParam(W, s1=s1, s2=s2, bias=_pad(ex["b"], oc), M=_pad(M, oc))
        if ex["act"] == "hswish":
            t1, M2, t2 = qr.params[op.name + ".hs"]
            q = ex["q"]
            p.q3, p.q6, p.M2, p.t1, p.t2 = _pad(3 * q, oc), _pad(6 * q, oc), _pad(M2, oc), t1, t2
        assert int(np.abs(p.bias).max()) < 2 ** 31 and int(p.M.max()) < 2 ** 15
        params.append(p)
        tid[ex["out"]], chans[ex["out"]] = k + 1, co
        tid[op.out] = k + 1
    return layers, params, tid


def read_tensor(ch: c.C16Chain, i: int) -> np.ndarray:
    P = ch.P
    t = P.tensors[i]
    where, off = P.loc[i]
    buf = {"s": ch.t_s, "out": ch.t_out, "act": ch.t_act}[where].numpy().view(np.int8)
    root = off - t.g0 * t.cplane
    ctot = t.ctot or t.c
    full = buf[root: root + (t.h + 2 * c.BR) * (ctot // 16) * t.cplane].reshape(t.h + 2 * c.BR, ctot // 16, -1, 16)
    g = full[c.BR:c.BR + t.h, t.g0:t.g0 + t.c // 16, c.PAD:c.PAD + t.w]
    return g.transpose(1, 3, 0, 2).reshape(t.c, t.h, t.w)


def render_text_strip(text: str, h: int = 48, w: int = 320) -> np.ndarray:
    """渲染白底黑字中英文测试图 (BGR uint8, HxWx3)"""
    font_path = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
    img = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    font_size = 30
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if tw > w - 12:
        font_size = max(14, int(font_size * (w - 12) / tw))
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    y = max(2, (h - th) // 2 - bbox[1])
    draw.text((8, y), text, fill=(15, 15, 15), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def preprocess_rec(bgr: np.ndarray, h: int = 48, w: int = 320) -> np.ndarray:
    """PP-OCRv4 Rec 预处理: 保持纵横比缩放到高 h, 右侧零填充至 w, 归一化到 [-1, 1]"""
    ih, iw = bgr.shape[:2]
    rw = min(w, max(1, int(np.ceil(h * iw / ih))))
    img = cv2.resize(bgr, (rw, h), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
    out = np.zeros((3, h, w), np.float32)
    out[:, :, :rw] = x.transpose(2, 0, 1)
    return out


def ctc_decode(probs: np.ndarray, chars: list[str]) -> tuple[str, float]:
    """probs: (T, 6625) -> (text, mean_conf)"""
    idx = probs.argmax(axis=-1)
    confs = probs.max(axis=-1)
    out_chars, out_confs = [], []
    prev = 0
    for i, c_id in enumerate(idx):
        if c_id != 0 and c_id != prev and c_id < len(chars):
            out_chars.append(chars[c_id])
            out_confs.append(float(confs[i]))
        prev = int(c_id)
    return "".join(out_chars), (float(np.mean(out_confs)) if out_confs else 0.0)


def main():
    iters = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 20
    texts = [
        "Ryzen 7840HS NPU测试",
        "中英文混合 OCR 2026",
        "XDNA1 神经网络加速",
        "Hello World! 你好世界",
        "PP-OCRv4 Mobile INT8",
        "算子库性能与精度验证",
    ]
    imgs = [render_text_strip(t) for t in texts]
    xs = [preprocess_rec(im) for im in imgs]

    ops, tail_name, (w_tail, b_tail), chars = load_rec_backbone()
    calib = [run_float(ops, x) for x in xs]
    shapes = {k: v.shape for k, v in calib[0].items()}

    qr = QRec(ops, calib, pct=99.95)
    for x in xs:
        qr.run(x)
    qr.frozen = True
    qr.run(xs[0])

    layers, params, tid = build_chain(qr, shapes)
    print(f"PP-OCRv4 Rec 主干: {len(layers)} 层 (NPU C16Chain), tail={tail_name} shape={shapes[tail_name]}")

    ch = c.C16Chain(layers, params, reps=1)
    s_in = qr.scale("x").reshape(-1, 1, 1)
    xq = np.clip(np.round(xs[0] / s_in), -127, 127).astype(np.int8)
    xq16 = np.zeros((16, 48, 320), np.int8)
    xq16[:3] = xq
    ch.set_input(xq16)
    ch.call()

    npu_tail = read_tensor(ch, tid[tail_name])
    ref_tail = qr.last_int[tail_name][0].astype(np.int8)
    diff = int(np.abs(npu_tail.astype(np.int16) - ref_tail.astype(np.int16)).max())
    print(f"NPU vs 整数模拟逐字节误差 (tail {tail_name}): max_diff={diff}")
    assert diff == 0, f"NPU 逐字节不一致! max_diff={diff}"

    # 性能测试
    t0 = time.perf_counter()
    for _ in range(iters):
        ch.call()
    npu_ms = (time.perf_counter() - t0) * 1000.0 / iters
    print(f"NPU 主干推理耗时 ({iters} 次平均): {npu_ms:.2f} ms")

    # 对比全浮点 onnxruntime vs NPU 主干 + SVTR 头
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess_full = ort.InferenceSession(REC_ONNX, opts, providers=["CPUExecutionProvider"])
    sess_head = build_rec_head()

    s_tail = qr.scale(tail_name).reshape(-1, 1, 1).astype(np.float32)
    print("\n--- 中英文识别对比 (CPU float32 vs NPU int8) ---")
    exact = 0
    for txt, x in zip(texts, xs):
        t_c0 = time.perf_counter()
        cpu_probs = sess_full.run(None, {"x": x[None]})[0][0]
        cpu_ms = (time.perf_counter() - t_c0) * 1000.0
        cpu_txt, cpu_conf = ctc_decode(cpu_probs, chars)

        xq = np.clip(np.round(x / s_in), -127, 127).astype(np.int8)
        xq16[:3] = xq
        ch.set_input(xq16)
        ch.call()
        npu_feat = read_tensor(ch, tid[tail_name]).astype(np.float32) * s_tail
        npu_feat = npu_feat * w_tail.reshape(-1, 1, 1) + b_tail.reshape(-1, 1, 1)

        t_h0 = time.perf_counter()
        npu_probs = sess_head.run(None, {"p2o.Add.231": npu_feat[None]})[0][0]
        head_ms = (time.perf_counter() - t_h0) * 1000.0
        npu_txt, npu_conf = ctc_decode(npu_probs, chars)
        match = "✅" if npu_txt == cpu_txt else "❌"
        exact += int(npu_txt == cpu_txt)
        print(f"  GT:  {txt}\n  CPU: {cpu_txt} ({cpu_conf:.3f}, {cpu_ms:.1f} ms)\n  NPU: {npu_txt} ({npu_conf:.3f}, npu={npu_ms:.2f}+head={head_ms:.1f} ms) {match}")
    print(f"与 CPU 一致率: {exact}/{len(texts)}")


if __name__ == "__main__":
    main()
