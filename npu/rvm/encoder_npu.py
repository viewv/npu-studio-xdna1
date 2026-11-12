"""RVM 编码器在 NPU 上跑真实权重: 量化模拟 (encoder_q) 冻结的整数参数 → chain_c16 的层 + LParam

  source npu/env.sh && python -m npu.rvm.encoder_npu [--iters 50]

检查两件事:
  1. NPU 的输出和 numpy 整数模拟逐字节一致 (f1..f4 四个特征张量);
  2. 反量化后和浮点参考 (= onnxruntime) 比余弦相似度 —— 量化方案本身的精度。

映射规则 (encoder_q 的融合方式和 NPU 一一对应):
  conv (+ relu / hswish / hsig) (+ add)  → 一个 Layer; act: relu 4, hswish 3, hsig 1, 无 2 (线性)
  SE 的 mul                               → 投影层的 sc (缩放张量 = FC2 / hsig 的输出)
  残差 add                                → 投影层的 cat (块的输入拼在 dw 输出后面, 单位权重 w_r[c])
  gap                                     → Layer(op=1)
  通道补到 16 的倍数: 补出来的通道权重 / bias / M 都是 0, 输出就是 0。
"""
from __future__ import annotations

import sys
import time

import numpy as np

from .. import chain_c16 as c
from .encoder_q import Cfg, QEncoder, calibrate
from .encoder_ref import FEATS, cos, images, load_graph, preprocess, run_float

ACT = {"relu": 4, "hswish": 3, "hsig": 1, None: 2}


def pad16(n):
    return -(-n // 16) * 16


def build(qe: QEncoder, shapes: dict) -> tuple[list, list, dict]:
    """→ (layers, params, {张量名: 链里的张量号})"""
    tid, chans = {"x": 0}, {"x": shapes["x"][0]}
    layers, params = [], []
    for op in qe.ops:
        k = len(layers)
        if op.kind == "gap":
            src = op.ins[0]
            ch, h, w = shapes[src]
            layers.append(c.Layer(pad16(ch), pad16(ch), 1, 1, h, w, op=1, src=tid[src] - (k + 1)))
            params.append(c.LParam(None))
            tid[op.out], chans[op.out] = k + 1, ch
            continue
        if op.kind != "conv":
            continue                                   # 激活 / mul / add 都融合进卷积了
        ex = qe.export[op.name]
        src, se, res = ex["src"], ex["se"], ex.get("res")
        ci, (h, w) = chans[src], shapes[src][1:]
        co, kk = op.W.shape[0], op.W.shape[2]
        dw = op.g > 1
        ica = pad16(ci)
        icb = pad16(chans[res]) if res else 0
        oc = pad16(co)
        layers.append(c.Layer(ica + icb, oc, kk, op.s, h, w, dw=int(dw), d=op.d, act=ACT[ex["act"]],
                              src=tid[src] - (k + 1), sc=tid[se] - (k + 1) if se else 0,
                              cat=tid[res] - (k + 1) if res else 0))
        W = np.zeros((oc, 1 if dw else ica + icb, kk, kk), np.int8)
        W[:co, :ex["W"].shape[1]] = ex["W"]
        if res:                                       # 残差: 输入通道 ica + c → 输出通道 c, 权重 w_r[c]
            assert chans[res] == co
            W[np.arange(co), ica + np.arange(co), 0, 0] = ex["w_r"]
        s1, M, s2 = qe.params[op.name]
        p = c.LParam(W, s1=s1, s2=s2, bias=_pad(ex["b"], oc), M=_pad(M, oc))
        if ex["act"] == "hswish":
            t1, M2, t2 = qe.params[op.name + ".hs"]
            q = ex["q"]
            p.q3, p.q6, p.M2, p.t1, p.t2 = _pad(3 * q, oc), _pad(6 * q, oc), _pad(M2, oc), t1, t2
        assert int(np.abs(p.bias).max()) < 2 ** 31 and int(p.M.max()) < 2 ** 15
        assert p.q6 is None or (int(p.q6.max()) < 2 ** 15 and int(p.M2.max()) < 2 ** 15)
        params.append(p)
        tid[ex["out"]], chans[ex["out"]] = k + 1, co
        tid[op.out] = k + 1
    return layers, params, tid


def _pad(v, n):
    out = np.zeros(n, np.int64)
    v = np.asarray(v, np.int64).ravel()
    out[:v.size] = v
    return out


def read_tensor(ch: c.C16Chain, i: int) -> np.ndarray:
    """链里张量 i 的真实区域 (C, H, W) int8 (从 scratch / 输出缓冲区读回)"""
    P = ch.P
    t = P.tensors[i]
    where, off = P.loc[i]
    buf = {"s": ch.t_s, "out": ch.t_out, "act": ch.t_act}[where].numpy().view(np.int8)
    root = off - t.g0 * t.cplane
    ctot = t.ctot or t.c
    full = buf[root: root + (t.h + 2 * c.BR) * (ctot // 16) * t.cplane].reshape(t.h + 2 * c.BR, ctot // 16, -1, 16)
    g = full[c.BR:c.BR + t.h, t.g0:t.g0 + t.c // 16, c.PAD:c.PAD + t.w]          # (H, CP, W, 16)
    return g.transpose(1, 3, 0, 2).reshape(t.c, t.h, t.w)


def prepare(n_cal=16):
    """校准 + 冻结 → (QEncoder, ops)"""
    cfg = Cfg(aq="channel", pct=0, rq="m16")
    ops = load_graph()
    cal = images(n_cal)
    qe = QEncoder(ops, calibrate(ops, cal), cfg)
    for im in cal:                                    # 冻结 m16 的 S1
        qe.run(preprocess(im))
    qe.frozen = True
    return qe, ops


def selftest(n_img=2, iters=10):
    """npu.check 用: 真实权重的编码器 (70 层) 和整数模拟逐字节一致"""
    from ..harness import SelfTestResult
    r = main(iters, n_img, quiet=True)
    return SelfTestResult(r[0] == 0, r[0], r[1])


def main(iters=20, n_img=8, quiet=False):
    qe, ops = prepare()
    evs = images(n_img, offset=16)
    ch = tid = shapes = None
    worst, coss = 0, {k: [] for k in FEATS}
    for n, im in enumerate(evs):
        ref = run_float(ops, preprocess(im))
        deq = qe.run(preprocess(im))
        T = qe.last_int
        if ch is None:
            shapes = {nm: v.shape for nm, v in ref.items()}
            layers, params, tid = build(qe, shapes)
            ch = c.C16Chain(layers, params)
            if not quiet:
                P = ch.P
                print(f"编码器 → {len(layers)} 层, {P.n_steps} 步, e_w {P.e_w} B, scratch {P.size['s'] / 2**20:.1f} MB",
                      flush=True)
        x = np.zeros((ch.layers[0].ic, *shapes["x"][1:]), np.int8)
        x[:shapes["x"][0]] = T["x"][0]
        ch.set_input(x)
        ch.call()
        for k, nm in FEATS.items():
            got = read_tensor(ch, tid[nm])[:shapes[nm][0]].astype(np.int64)
            worst = max(worst, int(np.abs(got - T[nm][0]).max()))
            coss[k].append(cos(got * T[nm][1].reshape(-1, 1, 1), ref[nm]))
    walls = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ch.call()
        walls.append(time.perf_counter() - t0)
    walls.sort()
    med = walls[len(walls) // 2] * 1e3
    if not quiet:
        print(f"{n_img} 张图: NPU vs 整数模拟 最大差 {worst}")
        print("  NPU 反量化 vs 浮点 (onnxruntime) 余弦: " +
              "  ".join(f"{k} {np.mean(v):.4f} (最差 {np.min(v):.4f})" for k, v in coss.items()))
        print(f"NPU 编码器 {len(ch.layers)} 层: 中位 {med:.2f} ms  最快 {walls[0] * 1e3:.2f} ms")
    return worst, med


if __name__ == "__main__":
    it = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 20
    sys.exit(0 if main(it)[0] == 0 else 1)
