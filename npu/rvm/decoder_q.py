"""RVM 解码器的量化方案模拟 (伪量化: 浮点算, 每个张量边界按整数规则落到网格), 用来选 NPU 上的做法

  source npu/env.sh && python -m npu.rvm.decoder_q 校准.npz 评估.npz [方案 ...]
  方案是 Cfg 的字段, 例: "state='channel'" "dgf='float'" "enc='float'"; 不给就跑一组对照

和 decoder_ref.run 同一个结构 (见那里的文件头), 量化点 (激活都是对称 int8, 零点 0, 除非另说):
  · 编码器特征 f1..f4: 逐通道 (enc='q8', MSE 截断; 模拟编码器的 int8 输出) 或不量化 (enc='float')
    注意 enc='float' 时特征按"比例 1"折进权重, 和其他输入的比例差几个数量级, 权重量化会把别的通道压成 0 ——
    只能和 wq=False 一起用 (诊断)
  · src: 0..1 → round(·127) (比例 1/127); s1..s3 = 主机做平均池化再按 1/127 量化
  · 卷积: 输入逐通道比例折进权重后按输出通道 int8 量化 (和编码器一样); 输出 (relu 后) 逐通道 MSE
  · 上采样 ×2: 整数 (9a + 3b + 3c + d + 8) >> 4 (= 双线性的值四舍五入), 比例不变
  · sigmoid / tanh: 输入按整层一个比例 (sigmoid ±8、tanh ±4 封顶) 量化成 int8 → 查表;
    sigmoid 输出 uint8, 比例 1/256; tanh 输出 int8, 比例 1/127
  · LRASPP: x · σ = (x·σ_u8 + 128) >> 8, 比例不变
  · GRU: 状态 h 的比例 state='fixed' (所有通道 1/127; |h| ≤ 1 恒成立) 或 'channel' (逐通道 MSE, 输入输出同一个比例,
         c 先 requant 到状态比例); r·h = (r_u8·h + 128) >> 8; h' = h + ((z_u8·(c − h) + 128) >> 8)
  · DGF 细化: dgf='float' 不量化; 'q8' = cov / var / A / 中间的 1×1 都 int8 (逐通道 MSE), box 和乘积在 int32 里精确算
  · 输出 pha: round(·255) (8 位 alpha)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from .decoder_ref import BLOCKS, avgpool2, conv2d, load_weights, run, sigmoid, up2, zero_state
from .encoder_q import QEncoder
from .encoder_ref import FEATS


@dataclass
class Cfg:
    enc: str = "q8"          # 编码器特征: q8 / float
    state: str = "fixed"     # GRU 状态比例: fixed / channel
    dgf: str = "q8"          # 细化: q8 / float
    lut: bool = True         # sigmoid / tanh 的输入量化 (False = 浮点激活, 诊断用)
    wq: bool = True          # 权重 int8
    aq: bool = True          # 卷积输出 / src / 上采样落到 int8 网格 (False = 浮点, 诊断用)

    def __str__(self):
        return (f"编码器特征 {self.enc:5s} 状态 {self.state:7s} 细化 {self.dgf:5s}"
                + ("" if self.lut else " 激活不量化") + ("" if self.wq else " 权重浮点"))


def rhu(x):
    """四舍五入 (半数向上, 和 (v + 2^(s-1)) >> s 一样)"""
    return np.floor(x + 0.5)


class QDecoder:
    def __init__(self, Wt, cfg: Cfg):
        self.W, self.cfg = Wt, cfg
        self.cal, self.amax, self.calibrating = {}, {}, True

    # -------- 量化点
    def act(self, nm, x, levels=127):
        """逐通道对称 int8 → 返回 (整数, 逐通道比例)"""
        if self.calibrating:
            v = np.abs(x).reshape(x.shape[0], -1)
            self.cal.setdefault(nm, []).append(v)
            return x, np.ones(x.shape[0])
        if not self.cfg.aq:
            return x, np.ones(x.shape[0])
        s = self.amax[nm] / levels
        return np.clip(rhu(x / s.reshape(-1, 1, 1)), -levels, levels), s

    def freeze(self):
        for nm, vs in self.cal.items():
            v = np.concatenate(vs, 1)
            a = np.array([QEncoder._mse_clip(row, 127) for row in v])
            self.amax[nm] = np.maximum(a, a.max() * 1e-3 + 1e-8)
        self.calibrating = False

    def conv(self, xs, name):
        """xs = [(整数或浮点, 比例)] 拼接后做卷积 → 浮点结果 (反量化的 acc + bias)"""
        W, b = self.W[name]
        x = np.concatenate([a for a, _ in xs])
        s_x = np.concatenate([s for _, s in xs])
        if self.calibrating or not self.cfg.wq:
            return conv2d(x * s_x.reshape(-1, 1, 1), W, b)
        We = W * s_x.reshape(1, -1, 1, 1)
        oc = W.shape[0]
        s_w = np.abs(We.reshape(oc, -1)).max(1) / 127
        s_w = np.maximum(s_w, s_w.max() * 1e-3 + 1e-30)
        Wq = np.clip(np.round(We / s_w.reshape(-1, 1, 1, 1)), -127, 127)
        return conv2d(x, Wq, None) * s_w.reshape(-1, 1, 1) + (0 if b is None else b.reshape(-1, 1, 1))

    def lut_in(self, nm, x, cap):
        """sigmoid / tanh 的输入: 整层一个比例 (校准最大值, 封顶 cap) → 落到 int8 网格后的浮点值"""
        if self.calibrating or not self.cfg.lut:
            if self.calibrating:
                self.cal.setdefault("lut:" + nm, []).append(np.abs(x).reshape(1, -1))
            return x
        s = min(cap, self.amax["lut:" + nm].max()) / 127
        return np.clip(rhu(x / s), -127, 127) * s

    def sig_u8(self, nm, x):
        y = sigmoid(self.lut_in(nm, x, 8.0))
        return y if self.calibrating or not self.cfg.lut else np.clip(rhu(y * 256), 0, 255)   # 单位 1/256

    def tanh(self, nm, x):
        return np.tanh(self.lut_in(nm, x, 4.0))

    # -------- GRU
    def gru(self, k, ih, hh, b, h):
        """b = (整数, 比例); h = 状态 (整数, 比例) → 新状态 (整数, 比例)"""
        cal = self.calibrating
        rz = self.sig_u8(ih, self.conv([b, h], ih))
        r, z = np.split(rz, 2)
        hi, sh = h
        if cal or not self.cfg.lut:
            rh = (r * hi, sh)
            c = self.tanh(hh, self.conv([b, rh], hh))
            if cal:
                hn = (1 - z) * hi + z * c
                self.act(f"r{k}", hn)
                return hn, sh
            zf = z / 256 if self.cfg.lut else z
        else:
            rh = (rhu((r * hi) / 256), sh)
            c = self.tanh(hh, self.conv([b, rh], hh))
            zf = None
        c_i = np.clip(rhu(c / sh.reshape(-1, 1, 1)), -127, 127)        # tanh 输出落到状态比例
        if zf is None:
            hn = hi + np.floor((z * (c_i - hi) + 128) / 256)
        else:
            hn = (1 - zf) * hi + zf * c_i
        return hn, sh

    # -------- 整个解码器
    def state_scale(self, k, ch):
        if self.calibrating:
            return np.ones(ch)                 # 校准时张量都是浮点, 比例记 1
        if self.cfg.state == "fixed":
            return np.full(ch, 1 / 127)
        return self.amax[f"r{k}"] / 127

    def run(self, src, f, r):
        """src (3,H,W) 0..1; f = {f1..f4} 浮点; r = {r1..r4}: (整数, 比例) 或 None (= 0) → (pha, 新状态)"""
        cal = self.calibrating
        cal = cal or not self.cfg.aq                     # aq=False: 下面按浮点走 (和校准一样)
        q_src = (src, np.ones(3)) if cal else (rhu(src * 127), np.full(3, 1 / 127))
        s = {1: avgpool2(src)}
        s[2] = avgpool2(s[1])
        s[3] = avgpool2(s[2])
        s = {k: (v, np.ones(3)) if cal else (rhu(v * 127), np.full(3, 1 / 127)) for k, v in s.items()}
        fq = {k: self.act(k, v) if self.cfg.enc == "q8" else (v, np.ones(v.shape[0])) for k, v in f.items()}
        st = {}
        for k in (1, 2, 3, 4):
            if r[k] is None:
                ch = (16, 20, 40, 64)[k - 1]
                hh, ww = f[f"f{k}"].shape[1:]
                r[k] = (np.zeros((ch, hh, ww)), self.state_scale(k, ch))
        # decode4: LRASPP
        x, sx = self.act("x4", np.maximum(self.conv([fq["f4"]], "Conv_163"), 0))
        g = fq["f4"][0].mean((1, 2), keepdims=True)
        sig = self.sig_u8("Conv_166", self.conv([(g, fq["f4"][1])], "Conv_166"))
        x = x * sig if cal or not self.cfg.lut else rhu(x * sig / 256)
        a, b = (x[:64], sx[:64]), (x[64:], sx[64:])
        st[4] = self.gru(4, "Conv_176", "Conv_181", b, r[4])
        out = [a, st[4]]
        for k in (3, 2, 1):
            fk = fq[f"f{k}"]
            o = np.concatenate([v for v, _ in out])
            so = np.concatenate([v for _, v in out])
            up = up2(o, *fk[0].shape[1:])
            up = (up if cal else rhu(up), so)
            conv, ih, hh = BLOCKS[k]
            x, sx = self.act(f"x{k}", np.maximum(self.conv([up, fk, s[k]], conv), 0))
            n = x.shape[0] // 2
            st[k] = self.gru(k, ih, hh, (x[n:], sx[n:]), r[k])
            out = [(x[:n], sx[:n]), st[k]]
        o = np.concatenate([v for v, _ in out])
        so = np.concatenate([v for _, v in out])
        up = up2(o, *src.shape[1:])
        up = (up if cal else rhu(up), so)
        h1 = self.act("h290", np.maximum(self.conv([up, q_src], "Conv_290"), 0))
        hid = self.act("hid", np.maximum(self.conv([h1], "Conv_292"), 0))
        y = self.conv([hid], "Conv_294")
        pha = self.dgf(src, q_src, y, hid)
        return pha, st

    def dgf(self, src, q_src, y, hid):
        cal, fl = self.calibrating, self.calibrating or self.cfg.dgf == "float"
        hidf = hid[0] * hid[1].reshape(-1, 1, 1)
        if not fl:
            y, sy = self.act("y", y)
            y = y * sy.reshape(-1, 1, 1)
        elif cal:
            self.act("y", y)
        X = np.concatenate([src, src.mean(0, keepdims=True)])
        if not cal:
            X = rhu(X * 127) / 127
        box = lambda v: conv2d(v, self.W["Conv_301"][0], None, g=4)
        mx, my = box(X), box(y)
        cov, var = box(X * y) - mx * my, box(X * X) - mx * mx
        if fl:
            if cal:
                self.act("cov", cov), self.act("var", var)
            A = np.maximum(conv2d(np.concatenate([cov, var, hidf]), *self.W["Conv_312"]), 0)
            if cal:
                self.act("a312", A)
            A = np.maximum(conv2d(A, *self.W["Conv_314"]), 0)
            if cal:
                self.act("a314", A)
            A = conv2d(A, *self.W["Conv_316"])
            if cal:
                self.act("A", A)
        else:
            cq, vq = self.act("cov", cov), self.act("var", var)
            A = self.act("a312", np.maximum(self.conv([cq, vq, hid], "Conv_312"), 0))
            A = self.act("a314", np.maximum(self.conv([A], "Conv_314"), 0))
            A, sA = self.act("A", self.conv([A], "Conv_316"))
            A = A * sA.reshape(-1, 1, 1)
        o = A * X + (my - A * mx)
        pha = np.clip(o[3], 0, 1)
        return pha if cal else rhu(pha * 255) / 255


def ort_feats(frames):
    """onnxruntime 整网逐帧跑 (带状态) → [(src, {f1..f4})] (浮点编码器特征)"""
    from .decoder_ref import ort_full
    sess = ort_full()
    h, w = frames.shape[1:3]
    st = {f"{k}i": v[None] for k, v in zero_state(h, w).items()}
    names = ["r1o", "r2o", "r3o", "r4o", *FEATS.values()]
    out = []
    for im in frames:
        src = (im.astype(np.float32) / 255).transpose(2, 0, 1)
        o = dict(zip(names, (v[0] for v in sess.run(names, {"src": src[None], "downsample_ratio": np.array([1.0], np.float32), **st}))))
        st = {f"r{i}i": o[f"r{i}o"][None] for i in range(1, 5)}
        out.append((src, {k: o[nm] for k, nm in FEATS.items()}))
    return out


def evaluate(Wt, cal, ev, cfg: Cfg):
    qd = QDecoder(Wt, cfg)
    r = {k: None for k in (1, 2, 3, 4)}
    for src, f in cal:
        _, r = qd.run(src, f, r)
    qd.freeze()
    rq = {k: None for k in (1, 2, 3, 4)}
    rf = zero_state(*ev[0][0].shape[1:])
    errs, band_e, flick, prev = [], [], [], None
    for src, f in ev:
        t = run(Wt, src, f, rf)
        rf = {f"r{k}": t[f"r{k}o"] for k in (1, 2, 3, 4)}
        p, rq = qd.run(src, f, rq)
        ref = t["pha"][0]
        d = np.abs(p - ref)
        band = (ref > 0.02) & (ref < 0.98)
        import cv2
        band = cv2.dilate(band.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        errs.append(d.mean())
        band_e.append(d[band].mean())
        if prev is not None:
            flick.append(np.abs((p - prev[0]) - (ref - prev[1])).mean())
        prev = (p, ref)
    return np.mean(errs) * 255, np.mean(band_e) * 255, np.mean(flick) * 255


def main():
    cal_frames = np.load(sys.argv[1])["rgb"]
    ev_frames = np.load(sys.argv[2])["rgb"]
    Wt = load_weights()
    idx = np.linspace(0, len(cal_frames) - 1, 16).astype(int)
    cal = ort_feats(cal_frames)
    cal = [cal[i] for i in idx]
    ev = ort_feats(ev_frames)
    cfgs = [eval(f"Cfg({', '.join(sys.argv[3:])})")] if len(sys.argv) > 3 else [
        Cfg(aq=False, wq=False, lut=False, dgf="float", enc="float"), Cfg(dgf="float", lut=False),
        Cfg(dgf="float"), Cfg(), Cfg(state="channel")]
    print(f"校准 {len(cal)} 帧 (只取帧, 循环状态按顺序), 评估 {len(ev)} 帧; 对浮点 RVM 的 alpha 误差 (/255)")
    for cfg in cfgs:
        m, b, fl = evaluate(Wt, cal, ev, cfg)
        print(f"  {cfg}:  MAE {m:5.2f}  边缘带 {b:5.2f}  多出来的闪烁 {fl:5.2f}", flush=True)


if __name__ == "__main__":
    main()
