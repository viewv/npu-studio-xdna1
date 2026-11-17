"""PP-OCRv6_medium 文本检测 (HGNetV2 + RepLKFPN + DBHead, 480×640) 在 XDNA1 NPU 上运行:

  source npu/env.sh && .venv-iron/bin/python -m npu.ocr.v6_det_npu [--iters 10]

异构切分:
  1. CPU Stem (onnxruntime, nodes[0..9]): HGStem + Conv3x3 s=2 (3×480×640 → 64×120×160)
  2. NPU Backbone + 4 路 FPN 1×1 投影 (chain_c16, nodes[10..159]):
     全部 Conv 1×1 / DW 3×3 / ReLU / GELU (act=5 LUT) / SE / 残差加在 4 核 NPU 上以 int8 运行,
     分 4 段产出 4 个 256 通道多尺度特征图 (conv2d.44/43/42/41)。
  3. CPU RepLKFPN + DBHead (onnxruntime, nodes[160..275]): 9×9 DW + 上采样 + DBHead → 1×480×640 概率图。
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper as nh
import onnxruntime as ort

from .. import chain_c16 as c
from .det_npu import extract_boxes, preprocess_det, read_tensor, render_doc_image
from .rec_npu import POOL_S, col, conv2d, cos, pad16, rnd_shift

V6_ACT = {"relu": 4, "hswish": 3, "hsig": 1, "linear": 2, "gelu": 5, None: 2}

V6_DET_ONNX = "/home/viewv/.npu/ocr/v6/medium_det.onnx"
STAGE_PROJ_OUTS = [
    "p2o.pd_op.conv2d.44.0",  # 256×120×160 (from Add.33)
    "p2o.pd_op.conv2d.43.0",  # 256×60×80   (from Add.65)
    "p2o.pd_op.conv2d.42.0",  # 256×30×40   (from Add.121)
    "p2o.pd_op.conv2d.41.0",  # 256×15×20   (from Add.153)
]
STAGE_IN_TENSORS = ["Add.33", "Add.65", "Add.121", "Add.153"]


@dataclass
class V6Op:
    kind: str  # conv, relu, gelu, hsig, gap, mul, add
    out: str
    ins: list[str]
    name: str
    W: np.ndarray | None = None
    b: np.ndarray | None = None
    s: int = 1
    d: int = 1
    g: int = 1


def load_v6_det_backbone(path: str = V6_DET_ONNX) -> list[V6Op]:
    """解析 medium_det.onnx nodes[10..159] (输入为 p2o.pd_op.relu.3.0)。"""
    m = onnx.load(path)
    nodes = list(m.graph.node)
    consts = {i.name: nh.to_array(i).astype(np.float64) for i in m.graph.initializer}
    ops: list[V6Op] = []
    i = 10
    while i < 160:
        n = nodes[i]
        if n.op_type == "Conv":
            W = consts[n.input[1]].astype(np.float32)
            b = consts[n.input[2]].astype(np.float32) if len(n.input) > 2 else np.zeros(W.shape[0], np.float32)
            a = {at.name: helper.get_attribute_value(at) for at in n.attribute}
            s = a.get("strides", [1, 1])[0]
            d = a.get("dilations", [1, 1])[0]
            g = a.get("group", 1)
            ops.append(V6Op("conv", n.output[0], [n.input[0]], n.name, W, b, s, d, g))
            i += 1
        elif n.op_type == "Relu":
            ops.append(V6Op("relu", n.output[0], [n.input[0]], n.name))
            i += 1
        elif n.op_type == "HardSigmoid":
            ops.append(V6Op("hsig", n.output[0], [n.input[0]], n.name))
            i += 1
        elif n.op_type == "ReduceMean":
            ops.append(V6Op("gap", n.output[0], [n.input[0]], n.name))
            i += 1
        elif n.op_type == "Mul":
            ops.append(V6Op("mul", n.output[0], list(n.input), n.name))
            i += 1
        elif n.op_type == "Add":
            ops.append(V6Op("add", n.output[0], list(n.input), n.name))
            i += 1
        elif (n.op_type == "Div" and i + 4 < 160
              and nodes[i + 1].op_type == "Erf" and nodes[i + 2].op_type == "Add"
              and nodes[i + 3].op_type == "Mul" and nodes[i + 4].op_type == "Mul"):
            mul_out = nodes[i + 4].output[0]
            ops.append(V6Op("gelu", mul_out, [n.input[0]], nodes[i + 4].name))
            i += 5
        else:
            raise ValueError(f"未识别节点 {i}: {n.op_type} {n.name}")

    # 将 4 个 FPN 1x1 投影挪到各自 Stage 输出刚算完之后 (方便按 4 段直接导出)
    stage_proj_map = dict(zip(STAGE_IN_TENSORS, STAGE_PROJ_OUTS))
    projs = {op.out: op for op in ops if op.out in STAGE_PROJ_OUTS}
    reordered = [op for op in ops if op.out not in STAGE_PROJ_OUTS]
    final_ops: list[V6Op] = []
    for op in reordered:
        final_ops.append(op)
        if op.out in stage_proj_map:
            final_ops.append(projs[stage_proj_map[op.out]])
    return final_ops


def _fast_erf(x: np.ndarray) -> np.ndarray:
    """向量化 erf 近似 (Abramowitz & Stegun 7.1.26, 最大误差 < 1.5e-7, 比 np.vectorize 快百倍)。"""
    s = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return s * y


def run_float_v6(ops: list[V6Op], x_in: np.ndarray, in_name: str = "p2o.pd_op.relu.3.0") -> dict[str, np.ndarray]:
    """浮点参考执行 (验证解析与重排无误)。"""
    t = {in_name: x_in.astype(np.float32)}
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
            ho, wo = (h + 2 * pad - op.d * (kh - 1) - 1) // op.s + 1, (w + 2 * pad - op.d * (kw - 1) - 1) // op.s + 1
            if op.g == 1 and kh == 1 and op.s == 1:
                y = (W.reshape(W.shape[0], c_in) @ x.reshape(c_in, h * w)).reshape(W.shape[0], h, w)
            elif op.g > 1:
                y = np.zeros((W.shape[0], ho, wo), np.float32)
                for ky in range(kh):
                    for kx in range(kw):
                        ys, xs = ky * op.d, kx * op.d
                        y += W[:, 0, ky, kx].reshape(-1, 1, 1) * x[:, ys:ys + op.s * ho:op.s, xs:xs + op.s * wo:op.s]
            else:
                cols = np.empty((c_in * kh * kw, ho * wo), np.float32)
                idx = 0
                for ky in range(kh):
                    for kx in range(kw):
                        ys, xs = ky * op.d, kx * op.d
                        cols[idx:idx + c_in] = x[:, ys:ys + op.s * ho:op.s, xs:xs + op.s * wo:op.s].reshape(c_in, -1)
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


class QV6Det:
    """PP-OCRv6_medium 主干与 FPN 投影的 int8 整数模拟器 (与 NPU 逐字节等价)。"""

    def __init__(self, ops: list[V6Op], calib_fl: list[dict[str, np.ndarray]],
                 in_name: str = "p2o.pd_op.relu.3.0", pct: float = 99.95):
        self.ops = ops
        self.in_name = in_name
        self.acts: dict[str, tuple[np.ndarray, float]] = {}
        self.luts: dict[str, np.ndarray] = {}
        self.params: dict[str, tuple[int, np.ndarray, int]] = {}
        self.export: dict[str, dict] = {}
        self.frozen = False
        self.last_int: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # 统计每层激活范围 (用标量统一 scale 或每通道)
        self.ranges: dict[str, np.ndarray] = {}
        all_names = {in_name} | {op.out for op in ops}
        for nm in all_names:
            vals = [fl[nm] for fl in calib_fl if nm in fl]
            if vals:
                abs_m = np.maximum.reduce([np.percentile(np.abs(v), pct, axis=(1, 2)) for v in vals])
                self.ranges[nm] = np.maximum(abs_m, float(np.max(abs_m)) * 1e-2 + 1e-6)

        self.users: dict[str, list[V6Op]] = {}
        for op in ops:
            for inp in op.ins:
                self.users.setdefault(inp, []).append(op)

    def scale(self, name: str, div: float = 127.0) -> np.ndarray:
        return self.ranges[name] / div

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
                mean_q = np.round(x_u8.astype(np.int8).astype(np.float64).sum(axis=(1, 2), keepdims=True)
                                  * POOL_S / 65536.0)
                # 注: pool 输出是 uint8 位模式 (有符号 int8), chain_c16 pool 输出是 int8.view(uint8)
                # 这里直接使用实际 NPU pool_ref: acc_sum * round(65536/(h*w)) >> 16
                h, w = x_u8.shape[1], x_u8.shape[2]
                m_pool = round((1 << POOL_S) / (h * w))
                pq = np.clip(rnd_shift(x_u8.view(np.int8).astype(np.int64).sum(axis=(1, 2), keepdims=True) * m_pool, POOL_S), -128, 127).astype(np.int8)
                T[op.out] = (pq.view(np.uint8), sx)
            elif op.kind == "mul":
                # SE: mul(y, se) -> 1x1 单位卷积 (src=y, sc=se) 实体化到 DDR
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
                if op.g > 1:
                    W_eff = W * sx.reshape(-1, 1, 1, 1)
                else:
                    W_eff = W * sx.reshape(1, -1, 1, 1)

                # 检查下一个算子
                nxt = self.ops[idx + 1] if idx + 1 < len(self.ops) else None
                nxt2 = self.ops[idx + 2] if idx + 2 < len(self.ops) else None

                # Case A: conv + relu / hsig / gelu
                if nxt is not None and nxt.kind in ("relu", "hsig", "gelu") and nxt.ins[0] == op.out and len(self.users.get(op.out, [])) == 1:
                    act_kind = nxt.kind
                    out_name = nxt.out
                    done.add(nxt.out)
                    Wq, s_w = self.wquant(W_eff)
                    bq = np.round(b / s_w).astype(np.int64)
                    acc = conv2d(xq.view(np.int8), Wq, bq, op.s, op.s, op.d, op.g)
                    if act_kind == "hsig":
                        s_out = np.full(W.shape[0], 1.0 / 255.0)
                        ratio = s_w * (255.0 / 6.0)
                        q_out = self.requant(acc, ratio, "hsig", op.name)
                    elif act_kind == "gelu":
                        # 先量化到 pre-GELU scale (op.out), 再查 256 项 LUT 到 post-GELU scale (nxt.out)
                        s_pre = float(np.max(self.scale(op.out)))
                        s_post = float(np.max(self.scale(out_name)))
                        if op.name not in self.luts:
                            g_idx = np.arange(-128, 128, dtype=np.float64)
                            x_fp = g_idx * s_pre
                            erf_v = np.vectorize(math.erf)(x_fp * inv_s2)
                            g_fp = 0.5 * x_fp * (1.0 + erf_v)
                            lut = np.clip(np.round(g_fp / s_post), -128, 127).astype(np.int8)
                            lut[128] = 0
                            self.luts[op.name] = lut
                        lut = self.luts[op.name]
                        q_out = self.requant(acc, s_w / s_pre, "gelu", op.name, lut=lut)
                        s_out = np.full(W.shape[0], s_post)
                    else:  # relu
                        s_out = self.scale(out_name)
                        q_out = self.requant(acc, s_w / s_out, "relu", op.name)
                    T[out_name] = (q_out, s_out)
                    self.export[op.name] = dict(kind="conv", out_name=out_name, act=act_kind,
                                                W=Wq, bias=bq, src=op.ins[0], cat=None,
                                                lut=self.luts.get(op.name))
                # Case B: 1x1 conv + residual add(skip, conv_out) -> 折叠进 cat 1x1 conv
                elif (nxt is not None and nxt.kind == "add" and op.out in nxt.ins
                      and op.W.shape[2] == 1 and op.g == 1 and len(self.users.get(op.out, [])) == 1):
                    skip_name = nxt.ins[0] if nxt.ins[1] == op.out else nxt.ins[1]
                    sq, ss = T[skip_name]
                    out_name = nxt.out
                    done.add(nxt.out)
                    Wq, s_w = self.wquant(W_eff, floor=ss / 127.0)
                    w_r = np.clip(np.round(ss / s_w), -127, 127).astype(np.int64)
                    bq = np.round(b / s_w).astype(np.int64)
                    acc = conv2d(xq.view(np.int8), Wq, bq, op.s, op.s, op.d, op.g) + col(w_r).astype(np.int64) * sq.astype(np.int8).astype(np.int64)
                    s_out = self.scale(out_name)
                    q_out = self.requant(acc, s_w / s_out, "linear", op.name)
                    T[out_name] = (q_out, s_out)
                    self.export[op.name] = dict(kind="conv_add", out_name=out_name, act="linear",
                                                W=Wq, w_r=w_r, bias=bq, src=op.ins[0], cat=skip_name)
                # Case C: 普通线性 conv (如 DW 3x3 或 FPN 1x1 投影)
                else:
                    Wq, s_w = self.wquant(W_eff)
                    bq = np.round(b / s_w).astype(np.int64)
                    acc = conv2d(xq.view(np.int8), Wq, bq, op.s, op.s, op.d, op.g)
                    s_out = self.scale(op.out)
                    q_out = self.requant(acc, s_w / s_out, "linear", op.name)
                    T[op.out] = (q_out, s_out)
                    self.export[op.name] = dict(kind="conv", out_name=op.out, act="linear",
                                                W=Wq, bias=bq, src=op.ins[0], cat=None, lut=None)
            else:
                raise ValueError(f"未融合/未处理算子: {op.kind} {op.name}")
            done.add(op.out)

        self.last_int = T
        return {k: T[k][0].astype(np.int8).astype(np.float32) * col(T[k][1]) for k in T}


def build_v6_det_chain(qd: QV6Det, ops: list[V6Op], shapes: dict[str, tuple[int, ...]]):
    """将 QV6Det 导出为 chain_c16 的 (layers, params, parts, stage_layer_indices)。"""
    in_nm = qd.in_name
    c0, h0, w0 = shapes[in_nm]
    layers = [c.Layer(pad16(c0), pad16(c0), 1, 1, h0, w0, op=3)]
    params = [c.LParam(None)]
    tid = {in_nm: 1}
    chans = {in_nm: c0}
    stage_lidx: dict[str, int] = {}

    for op in ops:
        k = len(layers)
        if op.kind == "gap":
            src = op.ins[0]
            ch, hh, ww = shapes[src]
            layers.append(c.Layer(pad16(ch), pad16(ch), 1, 1, hh, ww, op=1, src=tid[src] - (k + 1)))
            params.append(c.LParam(None))
            tid[op.out], chans[op.out] = k + 1, ch
        elif op.kind == "mul" and op.name in qd.export:
            ex = qd.export[op.name]
            src, se, ch = ex["src"], ex["se"], ex["ch"]
            hh, ww = shapes[src][1:]
            cp = pad16(ch)
            layers.append(c.Layer(cp, cp, 1, 1, hh, ww, act=2, src=tid[src] - (k + 1), sc=tid[se] - (k + 1)))
            W = np.zeros((cp, cp, 1, 1), np.int8)
            W[:ch, :ch] = ex["W"]
            s1, M, s2 = qd.params[op.name]
            params.append(c.LParam(W, s1=s1, s2=s2, bias=np.zeros(cp, np.int64),
                                   M=np.pad(M, (0, cp - ch), constant_values=1)))
            tid[op.out], chans[op.out] = k + 1, ch
        elif op.kind == "conv" and op.name in qd.export:
            ex = qd.export[op.name]
            src, cat = ex["src"], ex["cat"]
            ci, (hh, ww) = chans[src], shapes[src][1:]
            co, kk = op.W.shape[0], op.W.shape[2]
            dw = op.g > 1
            ica, oca = pad16(ci), pad16(co)
            if cat is not None:
                c_cat = pad16(chans[cat])
                layers.append(c.Layer(ica + c_cat, oca, kk, op.s, hh, ww, act=V6_ACT[ex["act"]],
                                      src=tid[src] - (k + 1), cat=tid[cat] - (k + 1)))
                W = np.zeros((oca, ica + c_cat, 1, 1), np.int8)
                W[:co, :ci] = ex["W"]
                W[np.arange(co), ica + np.arange(co), 0, 0] = ex["w_r"]
            else:
                layers.append(c.Layer(ica, oca, kk, op.s, hh, ww, dw=int(dw), d=op.d, act=V6_ACT[ex["act"]],
                                      src=tid[src] - (k + 1)))
                W = np.zeros((oca, 1 if dw else ica, kk, kk), np.int8)
                W[:co, :ex["W"].shape[1]] = ex["W"]
            s1, M, s2 = qd.params[op.name]
            bias = np.pad(ex["bias"], (0, oca - co))
            M_pad = np.pad(M, (0, oca - co), constant_values=1)
            params.append(c.LParam(W, s1=s1, s2=s2, bias=bias, M=M_pad, gelu=ex.get("lut")))
            tid[ex["out_name"]], chans[ex["out_name"]] = k + 1, co
            if ex["out_name"] in STAGE_PROJ_OUTS:
                stage_lidx[ex["out_name"]] = tid[ex["out_name"]]

    # 分 4 段运行, 每段末尾直接从 t_out 读出 1 个 FPN 投影特征图
    cuts = [stage_lidx[nm] for nm in STAGE_PROJ_OUTS]
    parts = [(0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], cuts[2]), (cuts[2], cuts[3])]
    return layers, params, parts, stage_lidx


def build_sub_ort(onnx_path: str, input_names: list[str], output_names: list[str],
                  in_shapes: list[list[int | str]], out_shapes: list[list[int | str]]) -> ort.InferenceSession:
    """从 ONNX 抽取子图构建 CPU onnxruntime Session。"""
    m = onnx.load(onnx_path)
    inits = {t.name for t in m.graph.initializer}
    needed = set(output_names)
    selected = []
    for n in reversed(m.graph.node):
        if any(o in needed for o in n.output):
            selected.append(n)
            for inp in n.input:
                if inp and inp not in inits and inp not in input_names:
                    needed.add(inp)
    selected.reverse()
    used_inits = {inp for n in selected for inp in n.input if inp in inits}
    g = helper.make_graph(
        selected, "sub",
        [helper.make_tensor_value_info(nm, TensorProto.FLOAT, sh) for nm, sh in zip(input_names, in_shapes)],
        [helper.make_tensor_value_info(nm, TensorProto.FLOAT, sh) for nm, sh in zip(output_names, out_shapes)],
        [t for t in m.graph.initializer if t.name in used_inits],
    )
    sm = helper.make_model(g, opset_imports=m.opset_import)
    sm.ir_version = m.ir_version
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    return ort.InferenceSession(sm.SerializeToString(), so, providers=["CPUExecutionProvider"])


def main():
    iters = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 10
    h, w = 480, 640
    lines = [
        "Ryzen 7840HS NPU 测试",
        "中英文混合 OCR 2026",
        "XDNA1 神经网络加速",
        "Hello World! 你好世界",
    ]
    doc = render_doc_image(lines, h, w)
    x = preprocess_det(doc)

    # 1. CPU full ORT 基线
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess_full = ort.InferenceSession(V6_DET_ONNX, so, providers=["CPUExecutionProvider"])
    in_nm_full = sess_full.get_inputs()[0].name
    for _ in range(3):
        prob_cpu = sess_full.run(None, {in_nm_full: x[None]})[0][0, 0]
    t0 = time.perf_counter()
    for _ in range(iters):
        prob_cpu = sess_full.run(None, {in_nm_full: x[None]})[0][0, 0]
    cpu_ms = (time.perf_counter() - t0) * 1000 / iters

    # 2. Stem & FPN+DBHead CPU 子图
    sess_stem = build_sub_ort(
        V6_DET_ONNX, ["x"], ["p2o.pd_op.relu.3.0"],
        [[1, 3, h, w]], [[1, 64, h // 4, w // 4]],
    )
    sess_tail = build_sub_ort(
        V6_DET_ONNX, STAGE_PROJ_OUTS, ["fetch_name_0"],
        [[1, 256, 120, 160], [1, 256, 60, 80], [1, 256, 30, 40], [1, 256, 15, 20]],
        [[1, 1, h, w]],
    )

    stem_feat = sess_stem.run(None, {"x": x[None]})[0][0]
    ops = load_v6_det_backbone()
    fl = run_float_v6(ops, stem_feat)
    shapes = {k: v.shape for k, v in fl.items()}

    qd = QV6Det(ops, [fl], pct=99.95)
    qd.run(stem_feat)
    qd.frozen = True
    qd.run(stem_feat)

    layers, params, parts, stage_tid = build_v6_det_chain(qd, ops, shapes)
    print(f"PP-OCRv6_medium Det NPU 主干: {len(layers) - 1} 层 ({len(parts)} 段导出 4 路 FPN), CPU Stem + RepLKFPN+DBHead")

    ch = c.C16Chain(layers, params, parts=parts)
    ch.write_tensor(1, qd.last_int["p2o.pd_op.relu.3.0"][0].astype(np.uint8))

    stage_feats: list[np.ndarray] = []
    max_diff_all = 0
    t_npu_sum = 0.0
    for p_idx, (k0, k1) in enumerate(parts):
        # 预热 (触发 JIT 编译 / XCLBIN 加载)
        c.conv_c16(ch.t_act, ch.t_w, ch.t_s, ch.t_out, spec=ch.spec, reps=1, part=f"{k0},{k1}")
        t_s = time.perf_counter()
        for _ in range(iters):
            c.conv_c16(ch.t_act, ch.t_w, ch.t_s, ch.t_out, spec=ch.spec, reps=1, part=f"{k0},{k1}")
        t_npu_sum += (time.perf_counter() - t_s) * 1000 / iters
        nm = STAGE_PROJ_OUTS[p_idx]
        got_u8 = read_tensor(ch, stage_tid[nm])
        ref_u8 = qd.last_int[nm][0].view(np.int8)
        diff = int(np.abs(got_u8.astype(np.int32) - ref_u8.astype(np.int32)).max())
        max_diff_all = max(max_diff_all, diff)
        s_out = qd.last_int[nm][1]
        fp = (got_u8.astype(np.float32) * col(s_out).astype(np.float32))[None]
        stage_feats.append(fp)

    for _ in range(3):
        prob_npu = sess_tail.run(None, dict(zip(STAGE_PROJ_OUTS, stage_feats)))[0][0, 0]
    t0 = time.perf_counter()
    for _ in range(iters):
        stem_f = sess_stem.run(None, {"x": x[None]})[0][0]
        prob_npu = sess_tail.run(None, dict(zip(STAGE_PROJ_OUTS, stage_feats)))[0][0, 0]
    cpu_pre_post_ms = (time.perf_counter() - t0) * 1000 / iters

    boxes_cpu = extract_boxes(prob_cpu)
    boxes_npu = extract_boxes(prob_npu)
    print(f"NPU 逐字节对比整数参考: max_diff={max_diff_all}")
    print(f"NPU 主干推理耗时 ({iters} 次平均): {t_npu_sum:.2f} ms, CPU Stem+RepLKFPN+DBHead: {cpu_pre_post_ms:.2f} ms (合计 {t_npu_sum + cpu_pre_post_ms:.2f} ms vs 纯 CPU {cpu_ms:.2f} ms)")
    print(f"Prob map cos: {cos(prob_cpu, prob_npu):.4f}")
    print(f"CPU 480×640 检测: {len(boxes_cpu)} 框 -> {boxes_cpu}")
    print(f"NPU+CPU 480×640 检测: {len(boxes_npu)} 框 -> {boxes_npu}")
    assert max_diff_all == 0, f"NPU 输出与整数参考不一致: max_diff={max_diff_all}"


if __name__ == "__main__":
    main()
