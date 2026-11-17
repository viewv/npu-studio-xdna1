"""PP-OCRv4 Mobile 文本检测 (DBNet, 640×640) 在 XDNA1 NPU 上运行:

  source npu/env.sh && .venv-iron/bin/python -m npu.ocr.det_npu [--iters 10]

结构:
  1. MobileNetV3 + FPN 主干 (ONNX nodes[234..664], 62 个 Conv + 10 个 SE + 6 个 Resize×2 + 4 路 Concat):
     全部在 4 核 NPU (chain_c16) 上以对称 int8 运行, 输出 24×160×160 特征图 (batch_norm_0.tmp_4)。
  2. DBHead (ONNX nodes[665..671], 2 次 ConvTranspose2d 2×2 s=2 + Sigmoid):
     用 onnxruntime 子图在 CPU 上还原 1×640×640 概率图并提取文本框。
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np
import onnx
from onnx import helper, numpy_helper as nh, TensorProto
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

from .. import chain_c16 as c
from .rec_npu import ACT, FONT_CANDIDATES, POOL_S, col, conv2d, cos, hardswish, pad16, rnd_shift

DET_ONNX = "/home/viewv/.npu/ocr/ch_PP-OCRv4_det.onnx"
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


@dataclass
class DOp:
    kind: str  # conv, relu, hswish, hsig, gap, mul, add, resize, concat
    out: str
    ins: list[str]
    name: str
    W: np.ndarray | None = None
    b: np.ndarray | None = None
    s: int = 1
    d: int = 1
    g: int = 1
    scale_factor: int = 1


def load_det_backbone(path: str = DET_ONNX) -> tuple[list[DOp], str]:
    """解析 nodes[234..664] (止于 batch_norm_0.tmp_4), 折叠 BN / mobile_one_block / whswish_b"""
    m = onnx.load(path)
    nodes = list(m.graph.node)
    consts = {i.name: nh.to_array(i).astype(np.float64) for i in m.graph.initializer}
    for n in nodes:
        if n.op_type == "Constant":
            consts[n.output[0]] = nh.to_array(n.attribute[0].t).astype(np.float64)

    ops: list[DOp] = []
    affine: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}

    def resolve(t: str):
        return affine[t] if t in affine else (t, np.array(1.0), np.array(0.0))

    # 统计每个张量被多少非 Constant 节点使用
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for n in nodes[234:665]:
        if n.op_type != "Constant":
            for inp in n.input:
                consumers.setdefault(inp, []).append(n)

    i = 234
    while i < 665:
        n = nodes[i]
        if n.op_type == "Constant":
            i += 1
            continue

        if n.op_type == "Conv":
            W = consts[n.input[1]].copy()
            b = consts[n.input[2]].copy() if len(n.input) > 2 else np.zeros(W.shape[0], np.float64)
            a = {at.name: helper.get_attribute_value(at) for at in n.attribute}
            s = a.get("strides", [1, 1])[0]
            d = a.get("dilations", [1, 1])[0]
            g = a.get("group", 1)
            src, w_in, b_in = resolve(n.input[0])
            if np.any(b_in != 0):
                b = b + (W * b_in.reshape(1, -1, 1, 1) if g == 1 else W * b_in.reshape(-1, 1, 1, 1)).sum(axis=(1, 2, 3))
            if np.any(w_in != 1):
                W = W * w_in.reshape(1, -1, 1, 1) if g == 1 else W * w_in.reshape(-1, 1, 1, 1)
            out_name = n.output[0]
            i += 1
            if i < 665 and nodes[i].op_type == "BatchNormalization":
                bn = nodes[i]
                gamma, beta = consts[bn.input[1]], consts[bn.input[2]]
                mean, var = consts[bn.input[3]], consts[bn.input[4]]
                eps = {at.name: helper.get_attribute_value(at) for at in bn.attribute}.get("epsilon", 1e-5)
                sc = gamma / np.sqrt(var + eps)
                W = W * sc.reshape(-1, 1, 1, 1)
                b = (b - mean) * sc + beta
                out_name = bn.output[0]
                i += 1
            elif i + 1 < 665 and nodes[i].op_type == "Mul" and nodes[i + 1].op_type == "Add":
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
            ops.append(DOp("conv", out_name, [src], n.name, W.astype(np.float32), b.astype(np.float32), s, d, g))
            continue

        if n.op_type == "Add" and i + 5 < 665 and nodes[i + 3].op_type == "Clip" and nodes[i + 4].op_type == "Mul" and nodes[i + 5].op_type == "Div":
            x_in = n.input[0] if n.input[0] not in consts else n.input[1]
            div_n = nodes[i + 5]
            h_out = div_n.output[0]
            ops.append(DOp("hswish", h_out, [x_in], div_n.name))
            i += 6
            if i + 1 < 665 and nodes[i].op_type == "Mul" and nodes[i + 1].op_type == "Add":
                mul_n, add_n = nodes[i], nodes[i + 1]
                if h_out in mul_n.input and mul_n.output[0] in add_n.input:
                    w_name = [x for x in mul_n.input if x != h_out][0]
                    b_name = [x for x in add_n.input if x != mul_n.output[0]][0]
                    wa = float(consts[w_name].reshape(-1)[0])
                    ba = float(consts[b_name].reshape(-1)[0])
                    if abs(ba) <= 1e-4:
                        affine[add_n.output[0]] = (h_out, np.array(wa), np.array(0.0))
                    else:
                        uses = consumers.get(add_n.output[0], [])
                        all_1x1 = len(uses) > 0 and all(
                            u.op_type == "Conv"
                            and consts[u.input[1]].shape[2] == 1
                            and {at.name: helper.get_attribute_value(at) for at in u.attribute}.get("group", 1) == 1
                            for u in uses
                        )
                        if all_1x1:
                            affine[add_n.output[0]] = (h_out, np.array(wa), np.array(ba))
                        else:
                            ch = next(op.W.shape[0] for op in reversed(ops) if op.kind == "conv")
                            W_id = np.eye(ch, dtype=np.float32).reshape(ch, ch, 1, 1) * wa
                            b_id = np.full(ch, ba, dtype=np.float32)
                            ops.append(DOp("conv", add_n.output[0], [h_out], add_n.name + ".aff", W_id, b_id, 1, 1, 1))
                    i += 2
            continue

        if n.op_type == "GlobalAveragePool":
            src, w_in, b_in = resolve(n.input[0])
            ops.append(DOp("gap", n.output[0], [src], n.name))
            if np.any(w_in != 1) or np.any(b_in != 0):
                affine[n.output[0]] = (n.output[0], w_in, b_in)
            i += 1
            continue

        if n.op_type == "Relu":
            ops.append(DOp("relu", n.output[0], list(n.input), n.name))
            i += 1
            continue

        if n.op_type == "HardSigmoid":
            nxt = nodes[i + 1]
            assert nxt.op_type == "Mul"
            other = [x for x in nxt.input if x != n.output[0]][0]
            src_other, w_in, b_in = resolve(other)
            assert np.all(w_in == 1) and np.all(b_in == 0)
            ops.append(DOp("hsig", n.output[0], list(n.input), n.name))
            ops.append(DOp("mul", nxt.output[0], [src_other, n.output[0]], nxt.name))
            i += 2
            continue

        if n.op_type == "Add":
            ops.append(DOp("add", n.output[0], list(n.input), n.name))
            i += 1
            continue

        if n.op_type == "Resize":
            sc_arr = consts[n.input[2]]
            sf = int(round(float(sc_arr[-1])))
            cur = n.input[0]
            # 拆成多次 2× 上采样以匹配 NPU op=2
            steps = sf.bit_length() - 1
            assert (1 << steps) == sf, f"不支持的 Resize 倍数: {sf}"
            for st in range(steps):
                out_t = n.output[0] if st == steps - 1 else f"{n.output[0]}_x2_{st}"
                ops.append(DOp("resize", out_t, [cur], f"{n.name}_{st}", scale_factor=2))
                cur = out_t
            i += 1
            continue

        if n.op_type == "Concat":
            ops.append(DOp("concat", n.output[0], list(n.input), n.name))
            i += 1
            continue

        raise ValueError(f"未预期的节点 {i}: {n.op_type} {n.name}")

    return ops, "batch_norm_0.tmp_4"


def build_fpn_head(path: str = DET_ONNX) -> ort.InferenceSession:
    """提取 nodes[572..671] (FPN + RSE + Resize + DBHead) 为独立 CPU ONNX 会话"""
    m = onnx.load(path)
    nodes = list(m.graph.node)
    needed = set()
    for n in nodes[572:]:
        needed.update(n.input)
    inits = [i for i in m.graph.initializer if i.name in needed]
    consts = [n for n in nodes[:572] if n.op_type == "Constant" and n.output[0] in needed]
    ins = [
        helper.make_tensor_value_info("p2o.Add.43", TensorProto.FLOAT, [1, 48, None, None]),
        helper.make_tensor_value_info("p2o.Add.71", TensorProto.FLOAT, [1, 96, None, None]),
        helper.make_tensor_value_info("p2o.Add.147", TensorProto.FLOAT, [1, 192, None, None]),
        helper.make_tensor_value_info("p2o.Add.215", TensorProto.FLOAT, [1, 384, None, None]),
    ]
    g = helper.make_graph(consts + nodes[572:], "det_fpn_head", ins, list(m.graph.output), initializer=inits)
    hm = helper.make_model(g, opset_imports=m.opset_import, ir_version=m.ir_version)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(hm.SerializeToString(), opts, providers=["CPUExecutionProvider"])


def build_det_head(path: str = DET_ONNX) -> ort.InferenceSession:
    """提取 nodes[665..671] (DBHead: ConvTranspose×2 + Sigmoid) 为独立 ONNX 会话"""
    m = onnx.load(path)
    nodes = list(m.graph.node)
    needed = set()
    for n in nodes[665:]:
        needed.update(n.input)
    inits = [i for i in m.graph.initializer if i.name in needed]
    consts = [n for n in nodes[:665] if n.op_type == "Constant" and n.output[0] in needed]
    inp = helper.make_tensor_value_info("batch_norm_0.tmp_4", TensorProto.FLOAT, [1, 24, None, None])
    out = m.graph.output[0]
    g = helper.make_graph(consts + nodes[665:], "det_head", [inp], [out], initializer=inits)
    hm = helper.make_model(g, opset_imports=m.opset_import, ir_version=m.ir_version)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(hm.SerializeToString(), opts, providers=["CPUExecutionProvider"])


def run_float(ops: list[DOp], x: np.ndarray) -> dict[str, np.ndarray]:
    T = {"x": x.astype(np.float64)}
    for op in ops:
        if op.kind == "conv":
            T[op.out] = conv2d(T[op.ins[0]], op.W, op.b, op.s, op.s, op.d, op.g)
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
        elif op.kind == "add":
            T[op.out] = T[op.ins[0]] + T[op.ins[1]]
        elif op.kind == "resize":
            T[op.out] = np.repeat(np.repeat(T[op.ins[0]], 2, axis=1), 2, axis=2)
        elif op.kind == "concat":
            T[op.out] = np.concatenate([T[inp] for inp in op.ins], axis=0)
    return T


class QDet:
    """对称 int8 + m16 量化模拟 (支持 SE, RSE y + y*se, FPN Add, Resize×2, Concat)"""

    def __init__(self, ops: list[DOp], calib: list[dict[str, np.ndarray]], pct: float = 99.95):
        self.ops = ops
        self.byname = {op.out: op for op in ops}
        self.users: dict[str, list[DOp]] = {}
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

    def wquant(self, W: np.ndarray, floor=None):
        oc = W.shape[0]
        amax_c = np.abs(W.reshape(oc, -1)).max(1)
        amax_c = np.maximum(amax_c, amax_c.max() * 1e-3 + 1e-30)
        s_w = amax_c / 127.0
        if floor is not None:
            s_w = np.maximum(s_w, floor)
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
            elif op.kind == "resize":
                q, s = T[op.ins[0]]
                T[op.out] = (np.repeat(np.repeat(q, 2, axis=1), 2, axis=2), s)
            elif op.kind == "mul":
                # 检查是否为 RSE: mul(y, se) -> add(y, mul_out)
                users = self.users.get(op.out, [])
                if len(users) == 1 and users[0].kind == "add":
                    add_op = users[0]
                    y_name, se_name = op.ins[0], op.ins[1]
                    yq, sy = T[y_name]
                    seq = T[se_name][0].ravel()
                    # 用 1x1 单位卷积 (src=y, sc=se, cat=y) 计算 y * se + y
                    c_dim = yq.shape[0]
                    W_se = np.eye(c_dim, dtype=np.float64).reshape(c_dim, c_dim, 1, 1) * sy.reshape(1, -1, 1, 1)
                    Wq, s_w = self.wquant(W_se, floor=sy / 127.0)
                    w_r = np.clip(np.round(sy / s_w), 0, 127).astype(np.int64)
                    Wq_eff = rnd_shift(Wq * seq.reshape(1, -1, 1, 1), 8)
                    acc = conv2d(yq, Wq_eff, None, 1, 1, 1, 1) + col(w_r).astype(np.int64) * yq
                    s_out = self.scale(add_op.out)
                    T[add_op.out] = (self.requant(acc, s_w / s_out, None, add_op.name), s_out)
                    self.export[add_op.name] = dict(kind="rse", W=Wq, w_r=w_r, src=y_name, se=se_name, cat=y_name, ch=c_dim)
                    done.add(add_op.out)
            elif op.kind == "add":
                # FPN top-down 残差加法: a + b → 1x1 cat 卷积
                a_name, b_name = op.ins[0], op.ins[1]
                (aq, sa), (bq, sb) = T[a_name], T[b_name]
                c_dim = aq.shape[0]
                W_a = np.eye(c_dim, dtype=np.float64).reshape(c_dim, c_dim, 1, 1) * sa.reshape(1, -1, 1, 1)
                Wq, s_w = self.wquant(W_a, floor=sb / 127.0)
                w_r = np.clip(np.round(sb / s_w), 0, 127).astype(np.int64)
                acc = conv2d(aq, Wq, None, 1, 1, 1, 1) + col(w_r).astype(np.int64) * bq
                s_out = self.scale(op.out)
                T[op.out] = (self.requant(acc, s_w / s_out, None, op.name), s_out)
                self.export[op.name] = dict(kind="add", W=Wq, w_r=w_r, src=a_name, cat=b_name, ch=c_dim)
            elif op.kind == "concat":
                qs = [T[inp][0] for inp in op.ins]
                ss = [T[inp][1] for inp in op.ins]
                T[op.out] = (np.concatenate(qs, axis=0), np.concatenate(ss, axis=0))
            else:
                raise ValueError(f"未处理的算子: {op.kind} {op.name}")
            done.add(op.out)
        self.last_int = T
        return {k: T[k][0] * col(T[k][1]) for k in T}

    def _conv(self, op: DOp, T, done):
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
        acc = conv2d(xq, Wq, bq, op.s, op.s, op.d, op.g)

        ex = self.export.setdefault(op.name, {})
        ex.update(kind="conv", W=Wq_raw, b=bq, src=src,
                  se=(self.byname[op.ins[0]].ins[1] if se is not None else None),
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


def build_chain(qd: QDet, shapes: dict[str, tuple[int, int, int]]) -> tuple[list, list, dict]:
    tid, chans = {"x": 0}, {"x": shapes["x"][0]}
    layers, params = [], []

    # 预先记录 concat 成员到下一成员的 nx 链接目标
    concat_op = next(op for op in qd.ops if op.kind == "concat")
    concat_ins = concat_op.ins  # [T0, T1, T2, T3]

    # 先按拓扑顺序生成层, 再把 concat_ins 的 nx 链接补上去
    op_to_layer_idx: dict[str, int] = {}

    for op in qd.ops:
        k = len(layers)
        if op.kind == "gap":
            src = op.ins[0]
            ch, h, w = shapes[src]
            layers.append(c.Layer(pad16(ch), pad16(ch), 1, 1, h, w, op=1, src=tid[src] - (k + 1)))
            params.append(c.LParam(None))
            tid[op.out], chans[op.out] = k + 1, ch
            continue

        if op.kind == "resize":
            src = op.ins[0]
            ch, h, w = shapes[src]
            layers.append(c.Layer(pad16(ch), pad16(ch), 1, 1, h, w, op=2, src=tid[src] - (k + 1), oh=2 * h))
            params.append(c.LParam(None))
            tid[op.out], chans[op.out] = k + 1, ch
            op_to_layer_idx[op.out] = k
            continue

        if op.kind == "concat":
            # 4 路拼接 [T0, T1, T2, T3], 每个 padded 到 32 通道, 总 128 通道
            tid[op.out] = tid[concat_ins[0]]
            chans[op.out] = sum(pad16(chans[inp]) for inp in concat_ins)
            continue

        if op.kind in ("mul",):
            continue

        if op.kind == "add":
            ex = qd.export[op.name]
            src, cat, ch = ex["src"], ex["cat"], ex["ch"]
            h, w = shapes[src][1:]
            cp = pad16(ch)
            if ex["kind"] == "rse":
                # 避免 src == cat 自环: 先复制一份 y_copy, 再以 [y_copy || y] 计算 y*se + y
                layers.append(c.Layer(cp, cp, 1, 1, h, w, act=2, src=tid[src] - (k + 1)))
                W_id = np.zeros((cp, cp, 1, 1), np.int8)
                W_id[np.arange(ch), np.arange(ch), 0, 0] = 1
                params.append(c.LParam(W_id, s1=0, s2=0, bias=_pad(np.zeros(ch), cp), M=_pad(np.ones(ch), cp)))
                k_copy = k + 1
                k = len(layers)
                src_ref = -1
                sc_ref = tid[ex["se"]] - (k + 1)
                cat_ref = tid[cat] - (k + 1)
            else:
                src_ref = tid[src] - (k + 1)
                sc_ref = 0
                cat_ref = tid[cat] - (k + 1)
            layers.append(c.Layer(2 * cp, cp, 1, 1, h, w, act=2, src=src_ref, sc=sc_ref, cat=cat_ref))
            W = np.zeros((cp, 2 * cp, 1, 1), np.int8)
            W[:ch, :ch] = ex["W"]
            W[np.arange(ch), cp + np.arange(ch), 0, 0] = ex["w_r"]
            s1, M, s2 = qd.params[op.name]
            params.append(c.LParam(W, s1=s1, s2=s2, bias=_pad(np.zeros(ch), cp), M=_pad(M, cp)))
            tid[op.out], chans[op.out] = k + 1, ch
            op_to_layer_idx[op.out] = k
            continue

        if op.kind == "conv":
            ex = qd.export[op.name]
            src, se = ex["src"], ex["se"]
            _, h, w = shapes[src]
            co, kk = op.W.shape[0], op.W.shape[2]
            dw = op.g > 1
            oc = pad16(co)
            if src == concat_op.out:
                # concat 输出由 4 个 pad16(24)=32 通道块拼成 128 通道
                ica = sum(pad16(chans[inp]) for inp in concat_ins)
                W = np.zeros((oc, ica, kk, kk), np.int8)
                off = 0
                in_off = 0
                for inp in concat_ins:
                    c_part = chans[inp]
                    W[:co, off:off + c_part] = ex["W"][:, in_off:in_off + c_part]
                    off += pad16(c_part)
                    in_off += c_part
            else:
                ci = chans[src]
                ica = pad16(ci)
                W = np.zeros((oc, 1 if dw else ica, kk, kk), np.int8)
                W[:co, :ex["W"].shape[1]] = ex["W"]

            layers.append(c.Layer(ica, oc, kk, op.s, h, w, dw=int(dw), d=op.d, act=ACT[ex["act"]],
                                  src=tid[src] - (k + 1), sc=tid[se] - (k + 1) if se else 0))
            s1, M, s2 = qd.params[op.name]
            p = c.LParam(W, s1=s1, s2=s2, bias=_pad(ex["b"], oc), M=_pad(M, oc))
            if ex["act"] == "hswish":
                t1, M2, t2 = qd.params[op.name + ".hs"]
                q = ex["q"]
                p.q3, p.q6, p.M2, p.t1, p.t2 = _pad(3 * q, oc), _pad(6 * q, oc), _pad(M2, oc), t1, t2
            assert int(np.abs(p.bias).max()) < 2 ** 31 and int(p.M.max()) < 2 ** 15
            params.append(p)
            tid[ex["out"]], chans[ex["out"]] = k + 1, co
            tid[op.out] = k + 1
            op_to_layer_idx[ex["out"]] = k

    # 设置 concat 4 路链接: T1 接在 T0 后, T2 接在 T1 后, T3 接在 T2 后
    for idx in range(1, len(concat_ins)):
        prev_t = tid[concat_ins[idx - 1]]
        cur_k = op_to_layer_idx[concat_ins[idx]]
        layers[cur_k] = replace(layers[cur_k], nx=prev_t - (cur_k + 1))

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


def render_doc_image(lines: list[str], h: int = 480, w: int = 640) -> np.ndarray:
    """渲染白底黑字多行中英文文档图 (BGR uint8, h×w)"""
    font_path = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
    img = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 32) if font_path else ImageFont.load_default()
    y = 45
    for ln in lines:
        draw.text((45, y), ln, fill=(15, 15, 15), font=font)
        y += 95
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def preprocess_det(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (rgb.transpose(2, 0, 1) - MEAN) / STD
    return x


def extract_boxes(prob: np.ndarray, thresh: float = 0.3, min_size: int = 5) -> list[tuple[int, int, int, int]]:
    mask = (prob > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= min_size and h >= min_size:
            pad_x, pad_y = max(8, int(h * 0.45)), max(6, int(h * 0.35))
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1, y1 = min(prob.shape[1], x + w + pad_x), min(prob.shape[0], y + h + pad_y)
            boxes.append((x0, y0, x1, y1))
    boxes.sort(key=lambda b: b[1])
    return boxes


def main():
    iters = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 10
    h = int(sys.argv[sys.argv.index("--h") + 1]) if "--h" in sys.argv else 480
    w = int(sys.argv[sys.argv.index("--w") + 1]) if "--w" in sys.argv else 640
    lines = [
        "Ryzen 7840HS NPU 测试",
        "中英文混合 OCR 2026",
        "XDNA1 神经网络加速",
        "Hello World! 你好世界",
    ]
    doc = render_doc_image(lines, h, w)
    x = preprocess_det(doc)

    ops, _ = load_det_backbone()
    # 异构切分: NPU 跑 49 层 MobileNetV3 主干 (止于 conv2d_469 之前), CPU 跑 FPN+DBHead
    cut = next(i for i, o in enumerate(ops) if o.out == "conv2d_469.tmp_0")
    ops_bb = ops[:cut]
    fl = run_float(ops_bb, x)
    shapes = {k: v.shape for k, v in fl.items()}

    # 确定 4 个多尺度阶段在主干中的张量名及其可选的 whswish_b 仿射
    stage_names = ["p2o.Add.43", "p2o.Add.71", "p2o.Add.147", "p2o.Add.215"]
    # 注意: load_det_backbone 中若这 4 个是 whswish_b 输出, 已在 consumers 里被视为非 1x1?
    # 它们直接喂给 conv2d_469..472 (1x1), 所以其前驱是 hardswish 输出, 或直接存在于 fl 中
    m_raw = onnx.load(DET_ONNX)
    consts_raw = {i.name: nh.to_array(i).astype(np.float32) for i in m_raw.graph.initializer}
    for n in m_raw.graph.node:
        if n.op_type == "Constant":
            consts_raw[n.output[0]] = nh.to_array(n.attribute[0].t).astype(np.float32)
    stage_map = {}
    for sn in stage_names:
        if sn in fl:
            stage_map[sn] = (sn, 1.0, 0.0)
        else:
            # sn 是 Add(Mul(h_out, w), b)
            add_n = next(n for n in m_raw.graph.node if sn in n.output)
            mul_out = [i for i in add_n.input if i not in consts_raw][0]
            b_nm = [i for i in add_n.input if i in consts_raw][0]
            mul_n = next(n for n in m_raw.graph.node if mul_out in n.output)
            h_out = [i for i in mul_n.input if i not in consts_raw][0]
            w_nm = [i for i in mul_n.input if i in consts_raw][0]
            stage_map[sn] = (h_out, float(consts_raw[w_nm].reshape(-1)[0]), float(consts_raw[b_nm].reshape(-1)[0]))

    qd = QDet(ops_bb, [fl], pct=99.95)
    qd.run(x)
    qd.frozen = True
    qd.run(x)

    # 构建 49 层主干链 (按 4 个阶段分段, 避免 t_s 复写 C2/C3/C4)
    tid, chans = {"x": 0}, {"x": 3}
    layers, params = [], []
    for op in ops_bb:
        k = len(layers)
        if op.kind == "gap":
            src = op.ins[0]
            ch, hh, ww = shapes[src]
            layers.append(c.Layer(pad16(ch), pad16(ch), 1, 1, hh, ww, op=1, src=tid[src] - (k + 1)))
            params.append(c.LParam(None))
            tid[op.out], chans[op.out] = k + 1, ch
        elif op.kind == "conv":
            ex = qd.export[op.name]
            src, se = ex["src"], ex["se"]
            ci, (hh, ww) = chans[src], shapes[src][1:]
            co, kk = op.W.shape[0], op.W.shape[2]
            dw = op.g > 1
            ica, oc = pad16(ci), pad16(co)
            layers.append(c.Layer(ica, oc, kk, op.s, hh, ww, dw=int(dw), d=op.d, act=ACT[ex["act"]],
                                  src=tid[src] - (k + 1), sc=tid[se] - (k + 1) if se else 0))
            W = np.zeros((oc, 1 if dw else ica, kk, kk), np.int8)
            W[:co, :ex["W"].shape[1]] = ex["W"]
            s1, M, s2 = qd.params[op.name]
            p = c.LParam(W, s1=s1, s2=s2, bias=_pad(ex["b"], oc), M=_pad(M, oc))
            if ex["act"] == "hswish":
                t1, M2, t2 = qd.params[op.name + ".hs"]
                q = ex["q"]
                p.q3, p.q6, p.M2, p.t1, p.t2 = _pad(3 * q, oc), _pad(6 * q, oc), _pad(M2, oc), t1, t2
            params.append(p)
            tid[ex["out"]], chans[ex["out"]] = k + 1, co
            tid[op.out] = k + 1

    stage_tids = [tid[stage_map[sn][0]] for sn in stage_names]
    parts = [(0, stage_tids[0]), (stage_tids[0], stage_tids[1]),
             (stage_tids[1], stage_tids[2]), (stage_tids[2], len(layers))]
    print(f"PP-OCRv4 Det NPU 主干: {len(layers)} 层 (4 段读取 C2..C5), CPU FPN+DBHead")

    ch = c.C16Chain(layers, params, reps=1, parts=parts)
    s_in = qd.scale("x").reshape(-1, 1, 1)
    xq16 = np.zeros((16, h, w), np.int8)
    xq16[:3] = np.clip(np.round(x / s_in), -127, 127).astype(np.int8)
    ch.set_input(xq16)

    def run_npu_backbone():
        feats = {}
        for (k0, k1), sn in zip(parts, stage_names):
            c.conv_c16(ch.t_act, ch.t_w, ch.t_s, ch.t_out, spec=ch.spec, reps=1, part=f"{k0},{k1}")
            base_nm, wa, ba = stage_map[sn]
            sc = qd.scale(base_nm).reshape(-1, 1, 1).astype(np.float32)
            arr = read_tensor(ch, tid[base_nm]).astype(np.float32) * sc
            if wa != 1.0 or ba != 0.0:
                arr = arr * wa + ba
            feats[sn] = arr[None]
        return feats

    npu_feats = run_npu_backbone()
    t0 = time.perf_counter()
    for _ in range(iters):
        npu_feats = run_npu_backbone()
    npu_ms = (time.perf_counter() - t0) * 1000.0 / iters
    print(f"NPU Det 主干推理耗时 ({iters} 次平均, 含读出): {npu_ms:.2f} ms")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess_full = ort.InferenceSession(DET_ONNX, opts, providers=["CPUExecutionProvider"])
    sess_head = build_fpn_head()

    t_c0 = time.perf_counter()
    cpu_prob = sess_full.run(None, {"x": x[None]})[0][0, 0]
    cpu_ms = (time.perf_counter() - t_c0) * 1000.0

    t_h0 = time.perf_counter()
    npu_prob = sess_head.run(None, npu_feats)[0][0, 0]
    head_ms = (time.perf_counter() - t_h0) * 1000.0

    cpu_boxes = extract_boxes(cpu_prob)
    npu_boxes = extract_boxes(npu_prob)
    print(f"Prob map cos: {cos(npu_prob, cpu_prob):.4f}")
    print(f"CPU {h}×{w} 检测: {len(cpu_boxes)} 框, {cpu_ms:.1f} ms -> {cpu_boxes}")
    print(f"NPU+CPU {h}×{w} 检测: {len(npu_boxes)} 框, npu={npu_ms:.2f}+fpn_head={head_ms:.1f} ms -> {npu_boxes}")


if __name__ == "__main__":
    main()
