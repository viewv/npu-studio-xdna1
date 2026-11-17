"""PP-OCRv6_medium 文本识别 (HGNetV2 + SVTR + 18710-CTC, 48×320) 在 XDNA1 NPU 上运行:

  source npu/env.sh && .venv-iron/bin/python -m npu.ocr.v6_rec_npu [--iters 10]

异构切分:
  1. CPU Stem (onnxruntime, nodes[0..22]): HGStem + Conv3x3 s=2 (3×48×320 → 64×12×80)
  2. NPU Backbone (chain_c16, nodes[23..330]):
     55 个 Conv (1×1 / DW 3×3 / 非对称步长 [2,1] / ReLU / GELU act=5 / SE / 残差加)
     全部在 4 核 NPU 上以 int8 运行, 输出 768×3×80 特征图 (p2o.pd_op.add.69.0)。
  3. CPU Tail (onnxruntime, nodes[331..507]): AveragePool + 1×7 DW + SVTR + 18710-CTC。
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper, numpy_helper as nh
import onnxruntime as ort

from .. import chain_c16 as c
from .det_npu import read_tensor
from .rec_npu import POOL_S, col, conv2d, cos, pad16, preprocess_rec, render_text_strip, rnd_shift
from .v6_det_npu import V6_ACT, _fast_erf, build_sub_ort

V6_REC_ONNX = "/home/viewv/.npu/ocr/v6/medium_rec.onnx"
V6_REC_YML = "/home/viewv/.npu/ocr/v6/small_rec.yml"
REC_STEM_OUT = "p2o.pd_op.relu.3.0"
REC_BB_OUT = "p2o.pd_op.add.69.0"


def load_v6_dict(path: str = V6_REC_YML) -> list[str]:
    """解析 PP-OCRv6 18,710 类字典 (index 0 = CTC blank, 1..18708 = 字符, 18709 = 空格)。"""
    with open(path, encoding="utf-8") as f:
        raw = [l.rstrip("\r\n") for l in f]
    idx = next(i for i, l in enumerate(raw) if l.strip().startswith("character_dict:"))
    chars = [""]
    for l in raw[idx + 1:]:
        if l.startswith("  - "):
            v = l[4:]
            if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                v = v[1:-1]
            chars.append(v)
        elif l.strip() and not l.startswith("  "):
            break
    chars.append(" ")
    return chars


def ctc_decode_v6(logits: np.ndarray, chars: list[str]) -> tuple[str, float]:
    """CTC greedy decode (logits: (1, T, 18710)) -> (text, avg_conf)。"""
    p = logits[0]
    ids = p.argmax(axis=-1)
    if p.min() >= -1e-4 and abs(float(p.sum(axis=-1).mean()) - 1.0) < 0.05:
        probs = p.max(axis=-1)
    else:
        m = p.max(axis=-1, keepdims=True)
        e = np.exp(p - m)
        probs = (e / e.sum(axis=-1, keepdims=True)).max(axis=-1)
    out, confs, prev = [], [], 0
    for idx, sc in zip(ids, probs):
        if idx != 0 and idx != prev and idx < len(chars):
            out.append(chars[idx])
            confs.append(float(sc))
        prev = int(idx)
    return "".join(out), (float(np.mean(confs)) if confs else 0.0)


@dataclass
class V6RecOp:
    kind: str  # conv, relu, gelu, hsig, gap, mul, add
    out: str
    ins: list[str]
    name: str
    W: np.ndarray | None = None
    b: np.ndarray | None = None
    sy: int = 1
    sx: int = 1
    d: int = 1
    g: int = 1


def load_v6_rec_backbone(path: str = V6_REC_ONNX) -> list[V6RecOp]:
    """解析 medium_rec.onnx nodes[23..330], 跳过 Identity 并折叠 Conv 后的常偏置 Add。"""
    m = onnx.load(path)
    nodes = list(m.graph.node)
    consts = {i.name: nh.to_array(i).astype(np.float64) for i in m.graph.initializer}
    alias: dict[str, str] = {}

    def res(nm: str) -> str:
        while nm in alias:
            nm = alias[nm]
        return nm

    ops: list[V6RecOp] = []
    i = 23
    while i <= 330:
        n = nodes[i]
        if n.op_type == "Identity":
            alias[n.output[0]] = res(n.input[0])
            i += 1
        elif n.op_type == "Conv":
            W = consts[n.input[1]].astype(np.float32)
            b = consts[n.input[2]].astype(np.float32) if len(n.input) > 2 else np.zeros(W.shape[0], np.float32)
            a = {at.name: helper.get_attribute_value(at) for at in n.attribute}
            st = a.get("strides", [1, 1])
            d = a.get("dilations", [1, 1])[0]
            g = a.get("group", 1)
            out_nm = n.output[0]
            src_nm = res(n.input[0])
            i += 1
            # 折叠 Conv 后的 Add(conv_out, bias_const) -> Identity
            if i <= 330 and nodes[i].op_type == "Add":
                an = nodes[i]
                c_in = [x for x in an.input if x in consts]
                if c_in and res(an.input[0]) == out_nm or (c_in and res(an.input[1]) == out_nm):
                    b = b + consts[c_in[0]].reshape(-1).astype(np.float32)
                    out_nm = an.output[0]
                    i += 1
                    while i <= 330 and nodes[i].op_type == "Identity":
                        alias[nodes[i].output[0]] = out_nm
                        i += 1
            ops.append(V6RecOp("conv", out_nm, [src_nm], n.name, W, b, st[0], st[1], d, g))
        elif n.op_type == "Relu":
            ops.append(V6RecOp("relu", n.output[0], [res(n.input[0])], n.name))
            i += 1
        elif n.op_type == "HardSigmoid":
            ops.append(V6RecOp("hsig", n.output[0], [res(n.input[0])], n.name))
            i += 1
        elif n.op_type == "ReduceMean":
            ops.append(V6RecOp("gap", n.output[0], [res(n.input[0])], n.name))
            i += 1
        elif n.op_type == "Mul":
            ops.append(V6RecOp("mul", n.output[0], [res(x) for x in n.input], n.name))
            i += 1
        elif n.op_type == "Add":
            ops.append(V6RecOp("add", n.output[0], [res(x) for x in n.input], n.name))
            i += 1
        elif (n.op_type == "Div" and i + 4 <= 330
              and nodes[i + 1].op_type == "Erf" and nodes[i + 2].op_type == "Add"
              and nodes[i + 3].op_type == "Mul" and nodes[i + 4].op_type == "Mul"):
            mul_out = nodes[i + 4].output[0]
            ops.append(V6RecOp("gelu", mul_out, [res(n.input[0])], nodes[i + 4].name))
            i += 5
        else:
            raise ValueError(f"未识别节点 {i}: {n.op_type} {n.name}")
    # 末尾若经过 Identity 输出到 REC_BB_OUT, 将最后一个 op.out 映射过去
    if res(REC_BB_OUT) != REC_BB_OUT:
        last_src = res(REC_BB_OUT)
        for op in ops:
            if op.out == last_src:
                op.out = REC_BB_OUT
            op.ins = [REC_BB_OUT if x == last_src else x for x in op.ins]
    return ops


def run_float_v6_rec(ops: list[V6RecOp], x_in: np.ndarray) -> dict[str, np.ndarray]:
    t = {REC_STEM_OUT: x_in.astype(np.float32)}
    inv_s2 = 1.0 / math.sqrt(2.0)
    for op in ops:
        if op.kind == "conv":
            x = t[op.ins[0]]
            W, b = op.W, op.b
            kh, kw = W.shape[2], W.shape[3]
            pad = (kh // 2) * op.d
            c_in, h, w = x.shape
            if pad:
                x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
            ho = (h + 2 * pad - op.d * (kh - 1) - 1) // op.sy + 1
            wo = (w + 2 * pad - op.d * (kw - 1) - 1) // op.sx + 1
            if op.g == 1 and kh == 1 and op.sy == 1 and op.sx == 1:
                y = (W.reshape(W.shape[0], c_in) @ x.reshape(c_in, h * w)).reshape(W.shape[0], h, w)
            elif op.g > 1:
                y = np.zeros((W.shape[0], ho, wo), np.float32)
                for ky in range(kh):
                    for kx in range(kw):
                        ys, xs = ky * op.d, kx * op.d
                        y += W[:, 0, ky, kx].reshape(-1, 1, 1) * x[:, ys:ys + op.sy * ho:op.sy, xs:xs + op.sx * wo:op.sx]
            else:
                cols = np.empty((c_in * kh * kw, ho * wo), np.float32)
                idx = 0
                for ky in range(kh):
                    for kx in range(kw):
                        ys, xs = ky * op.d, kx * op.d
                        cols[idx:idx + c_in] = x[:, ys:ys + op.sy * ho:op.sy, xs:xs + op.sx * wo:op.sx].reshape(c_in, -1)
                        idx += c_in
                y = (W.reshape(W.shape[0], -1) @ cols).reshape(W.shape[0], ho, wo)
            if b is not None:
                y = y + b.reshape(-1, 1, 1)
            t[op.out] = y
        elif op.kind == "relu":
            t[op.out] = np.maximum(t[op.ins[0]], 0)
        elif op.kind == "gelu":
            x = t[op.ins[0]]
            t[op.out] = (0.5 * x * (1.0 + _fast_erf(x * inv_s2))).astype(np.float32)
        elif op.kind == "hsig":
            t[op.out] = np.clip(t[op.ins[0]] / 6.0 + 0.5, 0.0, 1.0)
        elif op.kind == "gap":
            t[op.out] = t[op.ins[0]].mean(axis=(1, 2), keepdims=True)
        elif op.kind == "mul":
            t[op.out] = t[op.ins[0]] * t[op.ins[1]]
        elif op.kind == "add":
            t[op.out] = t[op.ins[0]] + t[op.ins[1]]
    return t


class QV6Rec:
    """PP-OCRv6_medium Rec 主干 int8 整数模拟器 (与 NPU 逐字节等价)。"""

    def __init__(self, ops: list[V6RecOp], calib_fl: list[dict[str, np.ndarray]], pct: float = 99.95):
        self.ops = ops
        self.in_name = REC_STEM_OUT
        self.luts: dict[str, np.ndarray] = {}
        self.params: dict[str, tuple[int, np.ndarray, int]] = {}
        self.export: dict[str, dict] = {}
        self.frozen = False
        self.last_int: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.ranges: dict[str, np.ndarray] = {}
        for nm in {self.in_name} | {op.out for op in ops}:
            vals = [fl[nm] for fl in calib_fl if nm in fl]
            if vals:
                abs_m = np.maximum.reduce([np.percentile(np.abs(v), pct, axis=(1, 2)) for v in vals])
                self.ranges[nm] = np.maximum(abs_m, float(np.max(abs_m)) * 1e-2 + 1e-6)
        self.users: dict[str, list[V6RecOp]] = {}
        for op in ops:
            for inp in op.ins:
                self.users.setdefault(inp, []).append(op)

    def scale(self, name: str) -> np.ndarray:
        return self.ranges[name] / 127.0

    def wquant(self, W: np.ndarray, floor: np.ndarray | float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        w_max = np.abs(W).reshape(W.shape[0], -1).max(axis=1)
        w_max = np.maximum(w_max, float(w_max.max()) * 1e-3 + 1e-30)
        s_w = np.maximum(w_max / 127.0, floor)
        s_w = np.maximum(s_w, 1e-12)
        Wq = np.clip(np.round(W / s_w.reshape(-1, 1, 1, 1)), -127, 127).astype(np.int8)
        return Wq, s_w

    def requant(self, acc: np.ndarray, ratio: np.ndarray, act: str, key: str,
                lut: np.ndarray | None = None) -> np.ndarray:
        if not self.frozen:
            s1 = max(0, int(np.ceil(np.log2(max(float(np.abs(acc).max()) / 32000.0, 1e-9)))))
            mm = ratio * (1 << s1)
            s2 = max(0, min(30, int(np.floor(np.log2(32767.0 / max(float(mm.max()), 1e-12))))))
            M = np.clip(np.round(mm * (1 << s2)), 0, 32767).astype(np.int64)
            self.params[key] = (s1, M, s2)
        else:
            s1, M, s2 = self.params[key]
        t = np.clip(rnd_shift(acc.astype(np.int64), s1), -32768, 32767)
        r = rnd_shift(t * col(M).astype(np.int64), s2)
        if act == "relu":
            return np.clip(r, 0, 127).astype(np.uint8)
        y = np.clip(r, -128, 127)
        if act == "hsig":
            return (y + 128).astype(np.uint8)
        if act == "gelu":
            assert lut is not None
            return lut[y + 128].astype(np.int8).view(np.uint8)
        return y.astype(np.int8).view(np.uint8)

    def run(self, x_in: np.ndarray) -> dict[str, np.ndarray]:
        s_in = self.scale(self.in_name)
        xq = np.clip(np.round(x_in / col(s_in)), 0, 127).astype(np.uint8)
        T: dict[str, tuple[np.ndarray, np.ndarray]] = {self.in_name: (xq, s_in)}
        done: set[str] = set()
        inv_s2 = 1.0 / math.sqrt(2.0)

        for idx, op in enumerate(self.ops):
            if op.out in done:
                continue
            if op.kind == "gap":
                x_u8, sx = T[op.ins[0]]
                h, w = x_u8.shape[1], x_u8.shape[2]
                m_pool = round((1 << POOL_S) / (h * w))
                pq = np.clip(rnd_shift(x_u8.view(np.int8).astype(np.int64).sum(axis=(1, 2), keepdims=True) * m_pool, POOL_S), -128, 127).astype(np.int8)
                T[op.out] = (pq.view(np.uint8), sx)
            elif op.kind == "mul":
                y_name, se_name = op.ins[0], op.ins[1]
                yq, sy = T[y_name]
                seq = T[se_name][0].ravel()
                c_dim = yq.shape[0]
                W_id = np.eye(c_dim, dtype=np.float64).reshape(c_dim, c_dim, 1, 1) * sy.reshape(1, -1, 1, 1)
                Wq, s_w = self.wquant(W_id)
                Wq_eff = rnd_shift(Wq * seq.reshape(1, -1, 1, 1), 8)
                acc = conv2d(yq.view(np.int8), Wq_eff, None, 1, 1, 1, 1)
                s_out = self.scale(op.out)
                T[op.out] = (self.requant(acc, s_w / s_out, "linear", op.name), s_out)
                self.export[op.name] = dict(kind="se_mul", W=Wq, src=y_name, se=se_name, ch=c_dim)
            elif op.kind == "conv":
                xq, sx = T[op.ins[0]]
                W, b = op.W.astype(np.float64), op.b.astype(np.float64)
                W_eff = W * (sx.reshape(-1, 1, 1, 1) if op.g > 1 else sx.reshape(1, -1, 1, 1))
                nxt = self.ops[idx + 1] if idx + 1 < len(self.ops) else None

                if nxt is not None and nxt.kind in ("relu", "hsig", "gelu") and nxt.ins[0] == op.out and len(self.users.get(op.out, [])) == 1:
                    act_kind, out_name = nxt.kind, nxt.out
                    done.add(nxt.out)
                    Wq, s_w = self.wquant(W_eff)
                    bq = np.round(b / s_w).astype(np.int64)
                    acc = conv2d(xq.view(np.int8), Wq, bq, op.sy, op.sx, op.d, op.g)
                    if act_kind == "hsig":
                        s_out = np.full(W.shape[0], 1.0 / 255.0)
                        q_out = self.requant(acc, s_w * (255.0 / 6.0), "hsig", op.name)
                    elif act_kind == "gelu":
                        s_pre = float(np.max(self.scale(op.out)))
                        s_post = float(np.max(self.scale(out_name)))
                        if op.name not in self.luts:
                            g_idx = np.arange(-128, 128, dtype=np.float64)
                            x_fp = g_idx * s_pre
                            g_fp = 0.5 * x_fp * (1.0 + _fast_erf(x_fp * inv_s2))
                            lut = np.clip(np.round(g_fp / s_post), -128, 127).astype(np.int8)
                            lut[128] = 0
                            self.luts[op.name] = lut
                        lut = self.luts[op.name]
                        q_out = self.requant(acc, s_w / s_pre, "gelu", op.name, lut=lut)
                        s_out = np.full(W.shape[0], s_post)
                    else:
                        s_out = self.scale(out_name)
                        q_out = self.requant(acc, s_w / s_out, "relu", op.name)
                    T[out_name] = (q_out, s_out)
                    self.export[op.name] = dict(kind="conv", out_name=out_name, act=act_kind,
                                                W=Wq, bias=bq, src=op.ins[0], cat=None,
                                                lut=self.luts.get(op.name))
                elif (nxt is not None and nxt.kind == "add" and op.out in nxt.ins
                      and op.W.shape[2] == 1 and op.g == 1 and len(self.users.get(op.out, [])) == 1):
                    skip_name = nxt.ins[0] if nxt.ins[1] == op.out else nxt.ins[1]
                    sq, ss = T[skip_name]
                    out_name = nxt.out
                    done.add(nxt.out)
                    Wq, s_w = self.wquant(W_eff, floor=ss / 127.0)
                    w_r = np.clip(np.round(ss / s_w), -127, 127).astype(np.int64)
                    bq = np.round(b / s_w).astype(np.int64)
                    acc = conv2d(xq.view(np.int8), Wq, bq, op.sy, op.sx, op.d, op.g) + col(w_r).astype(np.int64) * sq.view(np.int8).astype(np.int64)
                    s_out = self.scale(out_name)
                    q_out = self.requant(acc, s_w / s_out, "linear", op.name)
                    T[out_name] = (q_out, s_out)
                    self.export[op.name] = dict(kind="conv_add", out_name=out_name, act="linear",
                                                W=Wq, w_r=w_r, bias=bq, src=op.ins[0], cat=skip_name)
                else:
                    Wq, s_w = self.wquant(W_eff)
                    bq = np.round(b / s_w).astype(np.int64)
                    acc = conv2d(xq.view(np.int8), Wq, bq, op.sy, op.sx, op.d, op.g)
                    s_out = self.scale(op.out)
                    q_out = self.requant(acc, s_w / s_out, "linear", op.name)
                    T[op.out] = (q_out, s_out)
                    self.export[op.name] = dict(kind="conv", out_name=op.out, act="linear",
                                                W=Wq, bias=bq, src=op.ins[0], cat=None, lut=None)
            else:
                raise ValueError(f"未融合/未处理算子: {op.kind} {op.name}")
            done.add(op.out)

        self.last_int = T
        return {k: T[k][0].view(np.int8).astype(np.float32) * col(T[k][1]) for k in T}


def build_v6_rec_chain(qr: QV6Rec, ops: list[V6RecOp], shapes: dict[str, tuple[int, ...]]):
    c0, h0, w0 = shapes[REC_STEM_OUT]
    layers = [c.Layer(pad16(c0), pad16(c0), 1, 1, h0, w0, op=3)]
    params = [c.LParam(None)]
    tid = {REC_STEM_OUT: 1}
    chans = {REC_STEM_OUT: c0}

    for op in ops:
        k = len(layers)
        if op.kind == "gap":
            src = op.ins[0]
            ch, hh, ww = shapes[src]
            layers.append(c.Layer(pad16(ch), pad16(ch), 1, 1, hh, ww, op=1, src=tid[src] - (k + 1)))
            params.append(c.LParam(None))
            tid[op.out], chans[op.out] = k + 1, ch
        elif op.kind == "mul" and op.name in qr.export:
            ex = qr.export[op.name]
            src, se, ch = ex["src"], ex["se"], ex["ch"]
            hh, ww = shapes[src][1:]
            cp = pad16(ch)
            layers.append(c.Layer(cp, cp, 1, 1, hh, ww, act=2, src=tid[src] - (k + 1), sc=tid[se] - (k + 1)))
            W = np.zeros((cp, cp, 1, 1), np.int8)
            W[:ch, :ch] = ex["W"]
            s1, M, s2 = qr.params[op.name]
            params.append(c.LParam(W, s1=s1, s2=s2, bias=np.zeros(cp, np.int64),
                                   M=np.pad(M, (0, cp - ch), constant_values=1)))
            tid[op.out], chans[op.out] = k + 1, ch
        elif op.kind == "conv" and op.name in qr.export:
            ex = qr.export[op.name]
            src, cat = ex["src"], ex["cat"]
            ci, (hh, ww) = chans[src], shapes[src][1:]
            co, kk = op.W.shape[0], op.W.shape[2]
            dw = op.g > 1
            ica, oca = pad16(ci), pad16(co)
            sy_arg = op.sy if op.sy != op.sx else 0
            if cat is not None:
                c_cat = pad16(chans[cat])
                layers.append(c.Layer(ica + c_cat, oca, kk, op.sx, hh, ww, act=V6_ACT[ex["act"]],
                                      src=tid[src] - (k + 1), cat=tid[cat] - (k + 1), sy=sy_arg))
                W = np.zeros((oca, ica + c_cat, 1, 1), np.int8)
                W[:co, :ci] = ex["W"]
                W[np.arange(co), ica + np.arange(co), 0, 0] = ex["w_r"]
            else:
                layers.append(c.Layer(ica, oca, kk, op.sx, hh, ww, dw=int(dw), d=op.d, act=V6_ACT[ex["act"]],
                                      src=tid[src] - (k + 1), sy=sy_arg))
                W = np.zeros((oca, 1 if dw else ica, kk, kk), np.int8)
                W[:co, :ex["W"].shape[1]] = ex["W"]
            s1, M, s2 = qr.params[op.name]
            bias = np.pad(ex["bias"], (0, oca - co))
            M_pad = np.pad(M, (0, oca - co), constant_values=1)
            params.append(c.LParam(W, s1=s1, s2=s2, bias=bias, M=M_pad, gelu=ex.get("lut")))
            tid[ex["out_name"]], chans[ex["out_name"]] = k + 1, co

    return layers, params, tid


def main():
    iters = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 10
    chars = load_v6_dict()
    text_gt = "Ryzen NPU 中文测试 2026"
    img = render_text_strip(text_gt, 48, 320)
    x = preprocess_rec(img)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess_full = ort.InferenceSession(V6_REC_ONNX, so, providers=["CPUExecutionProvider"])
    in_nm_full = sess_full.get_inputs()[0].name
    for _ in range(3):
        logits_cpu = sess_full.run(None, {in_nm_full: x[None]})[0]
    t0 = time.perf_counter()
    for _ in range(iters):
        logits_cpu = sess_full.run(None, {in_nm_full: x[None]})[0]
    cpu_ms = (time.perf_counter() - t0) * 1000 / iters
    txt_cpu, conf_cpu = ctc_decode_v6(logits_cpu, chars)

    sess_stem = build_sub_ort(V6_REC_ONNX, ["x"], [REC_STEM_OUT], [[1, 3, 48, 320]], [[1, 64, 12, 80]])
    sess_tail = build_sub_ort(V6_REC_ONNX, [REC_BB_OUT], [sess_full.get_outputs()[0].name],
                              [[1, 768, 3, 80]], [[1, 40, 18710]])

    stem_feat = sess_stem.run(None, {"x": x[None]})[0][0]
    ops = load_v6_rec_backbone()
    fl = run_float_v6_rec(ops, stem_feat)
    shapes = {k: v.shape for k, v in fl.items()}

    qr = QV6Rec(ops, [fl], pct=99.95)
    qr.run(stem_feat)
    qr.frozen = True
    qr.run(stem_feat)

    layers, params, tid = build_v6_rec_chain(qr, ops, shapes)
    print(f"PP-OCRv6_medium Rec NPU 主干: {len(layers) - 1} 层 (64×12×80 → 768×3×80), CPU Stem + 1×7 DW + SVTR + CTC")

    ch = c.C16Chain(layers, params)
    ch.write_tensor(1, qr.last_int[REC_STEM_OUT][0].astype(np.uint8))
    c.conv_c16(ch.t_act, ch.t_w, ch.t_s, ch.t_out, spec=ch.spec, reps=1)
    t0 = time.perf_counter()
    for _ in range(iters):
        c.conv_c16(ch.t_act, ch.t_w, ch.t_s, ch.t_out, spec=ch.spec, reps=1)
    npu_ms = (time.perf_counter() - t0) * 1000 / iters

    got_u8 = read_tensor(ch, tid[REC_BB_OUT])
    ref_u8 = qr.last_int[REC_BB_OUT][0].view(np.int8)
    max_diff = int(np.abs(got_u8.astype(np.int32) - ref_u8.astype(np.int32)).max())
    bb_fp = (got_u8.astype(np.float32) * col(qr.last_int[REC_BB_OUT][1]).astype(np.float32))[None]

    for _ in range(3):
        logits_npu = sess_tail.run(None, {REC_BB_OUT: bb_fp})[0]
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = sess_stem.run(None, {"x": x[None]})[0][0]
        logits_npu = sess_tail.run(None, {REC_BB_OUT: bb_fp})[0]
    cpu_pre_post_ms = (time.perf_counter() - t0) * 1000 / iters
    txt_npu, conf_npu = ctc_decode_v6(logits_npu, chars)

    print(f"NPU 逐字节对比整数参考: max_diff={max_diff}")
    print(f"NPU 主干推理耗时 ({iters} 次平均): {npu_ms:.2f} ms, CPU Stem+SVTR+CTC: {cpu_pre_post_ms:.2f} ms (合计 {npu_ms + cpu_pre_post_ms:.2f} ms vs 纯 CPU {cpu_ms:.2f} ms)")
    print(f"Logits cos: {cos(logits_cpu, logits_npu):.4f}")
    print(f"GT : {text_gt!r}")
    print(f"CPU: {txt_cpu!r} (conf={conf_cpu:.3f})")
    print(f"NPU: {txt_npu!r} (conf={conf_npu:.3f})")
    assert max_diff == 0, f"NPU 输出与整数参考不一致: max_diff={max_diff}"


if __name__ == "__main__":
    main()
