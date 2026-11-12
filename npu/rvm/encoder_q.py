"""RVM 编码器的 int8 量化模拟 (numpy, 整数逐位定义), 用来选 NPU 上的量化方案

  source npu/env.sh && python -m npu.rvm.encoder_q                     # 几种方案对浮点参考的误差
  python -m npu.rvm.encoder_q "aq='channel'" "aq='tensor',pct=99.9"    # 指定方案 (Cfg 的字段)

约定: 张量都是对称 int8 (零点 0), 比例 s 按通道 (aq=channel) 或整个张量 (aq=tensor, 就是每个通道同一个值)。
  conv   输入的逐通道比例先折进权重 (W_eff[o, c] = W[o, c]·s_x[c], 精确), 再按输出通道 (或整层) 对称量化:
         acc = Σ W_q·x_q + b_q (int32), acc 的单位 = s_w[o] → 重新量化 (requant) 到下一个张量 → 激活:
           relu    y = clamp(·, 0, 127)                           (ReLU 输出只用 0..127)
           hswish  先量化到 s = 1/q_c (q_c 整数, "3" 和 "6" 是精确的整数 3q / 6q), y8;
                   yt = y8 · clamp(y8 + 3q, 0, 6q) (单位 1/(6q²)) → 再 requant 一次到输出比例
           hsig    (SE 的 FC2) 量化到 s = 6/256, out = y8 + 128 (uint8, 比例 1/256) —— 正好是 clamp(x/6 + 0.5)
  gap    均值 (Σ·M + 2^21) >> 22, 比例不变
  mul    SE: 折进下一层投影卷积的权重 W' = (W_q·s + 128) >> 8 (se=wfold) 或精确乘 (se=exact)
  add    残差: 投影层的累加器里加 w_r[o]·r_q (单位权重, w_r = round(s_r[o] / s_w[o]) ≤ 127) (res=identity)
         或各自反量化后相加再量化 (res=exact)

requant (acc → int8, 乘子 m[o] = s_acc[o] / s_y[o]):
  exact   y = round(acc · m)                                       (浮点乘, 理想情况)
  m16     t = sat16((acc + 2^(S1-1)) >> S1), y = sat8((t · M[o] + 2^(S2-1)) >> S2), M int16 (内核候选)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from .encoder_ref import FEATS, Op, conv2d, cos, images, load_graph, preprocess, run_float

POOL_S = 22


def rnd_shift(v, s):
    """(v + 2^(s-1)) >> s, s ≥ 1 (向正无穷舍入, 和内核的 srs 一样)"""
    return (v + (1 << (s - 1))) >> s if s > 0 else v << (-s)


def col(a):
    return np.asarray(a, np.float64).reshape(-1, 1, 1)


@dataclass
class Cfg:
    aq: str = "tensor"        # 激活比例: tensor / channel
    wq: str = "channel"       # 权重比例: channel / tensor
    rq: str = "exact"         # exact / m16
    res: str = "identity"     # identity / exact
    se: str = "wfold"         # wfold / exact
    pct: float = 100.0        # 激活范围取 |x| 的百分位 (100 = 最大值; 0 = 逐通道找量化 MSE 最小的截断)
    relu_u8: bool = False     # ReLU 输出用 0..255 (多 1 位; 内核要支持 uint8 输入)
    abits: int = 8            # 激活位宽 (诊断用: 16 位 ≈ 只剩权重量化误差)
    wbits: int = 8

    def __str__(self):
        return (f"激活 {self.aq:<7s} 权重 {self.wq:<7s} 重量化 {self.rq:<5s} 残差 {self.res:<8s} SE {self.se:<5s} "
                f"范围 {'mse' if self.pct == 0 else f'p{self.pct:g}'}" + (" relu8" if self.relu_u8 else "")
                + (f" a{self.abits}" if self.abits != 8 else "")
                + (f" w{self.wbits}" if self.wbits != 8 else ""))


class QEncoder:
    def __init__(self, ops: list[Op], calib: list[dict], cfg: Cfg):
        self.ops, self.cfg = ops, cfg
        self.byname = {op.out: op for op in ops}
        self.users = {}
        for op in ops:
            for i in op.ins:
                self.users.setdefault(i, []).append(op)
        self.stats, self.accmax, self.params, self.frozen, self.export = {}, {}, {}, False, {}
        self.relu_out = {op.out for op in ops if op.kind == "relu"}
        # 每个张量每个通道的 |x| 范围 (校准集上的百分位); aq=tensor 时所有通道取整个张量的值
        self.amax = {}
        relu_out = {op.out for op in ops if op.kind == "relu"}
        for nm in calib[0]:
            v = np.abs(np.stack([c[nm] for c in calib], 1)).reshape(calib[0][nm].shape[0], -1)
            levels = 255 if (cfg.relu_u8 and nm in relu_out) else self.qa
            if cfg.aq != "channel":
                v = v.reshape(1, -1)
            if cfg.pct == 0:
                a = np.array([self._mse_clip(row, levels) for row in v])
            elif cfg.pct < 100:
                a = np.percentile(v, cfg.pct, axis=1)
            else:
                a = v.max(1)
            a = np.broadcast_to(a, (calib[0][nm].shape[0],)).copy()
            self.amax[nm] = np.maximum(a, a.max() * 1e-3 + 1e-8)   # 全零通道也给个比例

    @staticmethod
    def _mse_clip(v, levels, n=48):
        """|x| 的样本 v → 截断值 c (在 max·[0.2, 1] 里找 Σ (x - Q_c(x))² 最小的)"""
        mx = float(v.max())
        if mx <= 0:
            return 1e-8
        if v.size > 200000:
            v = np.random.default_rng(0).choice(v, 200000, replace=False)
        best, bc = None, mx
        for r in np.linspace(0.2, 1.0, n):
            c = mx * r
            st = c / levels
            e = float(((np.minimum(np.round(v / st), levels) * st - v) ** 2).sum())
            if best is None or e < best:
                best, bc = e, c
        return bc

    @property
    def qa(self): return (1 << (self.cfg.abits - 1)) - 1
    @property
    def qw(self): return (1 << (self.cfg.wbits - 1)) - 1

    def scale(self, nm):
        if self.cfg.relu_u8 and nm in self.relu_out:
            return self.amax[nm] / 255
        return self.amax[nm] / self.qa

    def requant(self, acc, m, y_lim=None, key=None):
        """acc int64 (C, …), m (C,) 浮点乘子 → int (clip 到 y_lim)。
        key: 这次 requant 的名字; 校准时 (accmax 没冻结) 记下 |acc| 的最大值, m16 的 S1 用它定 (留 2 倍余量)"""
        lo, hi = y_lim or (-self.qa - 1, self.qa)
        m = col(m)
        if key is not None and not self.frozen:
            self.accmax[key] = max(self.accmax.get(key, 1), int(np.abs(acc).max()))
        if self.cfg.rq == "exact":
            y = np.floor(acc * m + 0.5)
        else:  # m16
            amax = 2 * self.accmax[key] if (key in self.accmax and self.frozen) else max(int(np.abs(acc).max()), 1)
            s1 = max(0, int(np.ceil(np.log2(amax / 32767))))            # acc >> S1 放进 int16
            if key is not None:
                self.params[key] = s1
            t = np.clip(rnd_shift(acc, s1), -32768, 32767)
            mm = m * (1 << s1)                                           # t 上的乘子
            s2 = max(0, int(np.floor(np.log2(32767 / mm.max()))))        # M ≤ 32767, 尽量大
            M = np.clip(np.round(mm * (1 << s2)), 0, 32767).astype(np.int64)
            if mm.max() > 32767:
                self.stats.setdefault("m16_sat", []).append(float(mm.max()))
            y = rnd_shift(t * M, s2) if s2 > 0 else t * M
            if key is not None:
                self.params[key] = (s1, M.ravel(), s2)
        return np.clip(y, lo, hi).astype(np.int64)

    def wquant(self, W, floor=None):
        """W 浮点 (已折入输入比例) → (W_q, s_w[o]); floor: s_w 的下限 (残差单位权重要放得进 int8)"""
        oc = W.shape[0]
        amax_c = np.abs(W.reshape(oc, -1)).max(1)
        # 死通道 (权重全 0) / 极小的通道: 比例太小会让 b_q 溢出 int32, 设个下限
        amax_c = np.maximum(amax_c, amax_c.max() * 1e-3 + 1e-30)
        s_w = amax_c / self.qw if self.cfg.wq == "channel" else np.full(oc, amax_c.max() / self.qw)
        if floor is not None:
            s_w = np.maximum(s_w, floor)
        Wq = np.clip(np.round(W / s_w.reshape(-1, 1, 1, 1)), -self.qw, self.qw).astype(np.int64)
        return Wq, s_w

    # ---------------------------------------------------------------- 执行
    def run(self, x_float: np.ndarray) -> dict[str, np.ndarray]:
        """返回 {张量名: 反量化后的浮点}"""
        s_in = self.scale("x")
        T = {"x": (np.clip(np.round(x_float / col(s_in)), -self.qa, self.qa).astype(np.int64), s_in)}
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
                T[op.out] = (np.clip(rnd_shift(q.reshape(q.shape[0], -1).sum(1) * M, POOL_S), -self.qa - 1, self.qa)
                             .reshape(-1, 1, 1), s)
            elif op.kind == "mul":
                pass                                     # 折进下一层 (在 _conv 里处理)
            elif op.kind == "add":                       # exact 残差 (identity 时已在 _conv 里融合)
                (a, sa), (b, sb) = T[op.ins[0]], T[op.ins[1]]
                so = self.scale(op.out)
                T[op.out] = (np.clip(np.round((a * col(sa) + b * col(sb)) / col(so)), -self.qa - 1, self.qa)
                             .astype(np.int64), so)
            else:
                raise ValueError(f"{op.kind} 应该已经融合进卷积: {op.name}")
            done.add(op.out)
        self.last_int = T
        return {k: T[k][0] * col(T[k][1]) for k in T}

    def _conv(self, op: Op, T, done):
        cfg = self.cfg
        src = op.ins[0]
        se = None
        if src in self.byname and self.byname[src].kind == "mul":        # SE 之后的投影层
            mul = self.byname[src]
            src, se = mul.ins[0], T[mul.ins[1]][0].ravel()               # uint8 0..255 = s/256
            done.add(mul.out)
        xq, s_x = T[src]
        users = self.users.get(op.out, [])
        act = users[0] if len(users) == 1 and users[0].kind in ("relu", "hswish", "hsig") else None
        add = users[0] if len(users) == 1 and users[0].kind == "add" and cfg.res == "identity" else None
        final = (act or add or op).out
        # 输入比例折进权重
        W = op.W.astype(np.float64)
        if op.g == 1:
            W = W * s_x.reshape(1, -1, 1, 1)
        else:
            assert op.g == W.shape[0] == xq.shape[0] and W.shape[1] == 1, f"{op.name}: 只支持普通卷积和 depthwise"
            W = W * s_x.reshape(-1, 1, 1, 1)
        if se is not None and cfg.se == "exact":
            W = W * (se / 256.0).reshape(1, -1, 1, 1)
        floor, rq = None, None
        if add is not None:
            r_nm = [i for i in add.ins if i != op.out][0]
            rq, s_r = T[r_nm]
            floor = s_r / self.qw                                        # w_r = s_r / s_w ≤ 127
        Wq, s_w = self.wquant(W, floor)
        Wq_raw = Wq                                  # 导出给 NPU 的是乘 SE 之前的 (内核每步自己折)
        if se is not None and cfg.se == "wfold":
            Wq = rnd_shift(Wq * se.reshape(1, -1, 1, 1), 8)
        bq = np.round(op.b / s_w).astype(np.int64)
        acc = conv2d(xq, Wq, bq, op.s, op.d, op.g)
        ex = self.export.setdefault(op.name, {})
        ex.update(W=Wq_raw, b=bq, src=src, se=(self.byname[op.ins[0]].ins[1] if se is not None else None),
                  act=(act.kind if act is not None else None), out=final)
        if add is not None:
            w_r = np.clip(np.round(s_r / s_w), 0, self.qw).astype(np.int64)
            acc = acc + col(w_r).astype(np.int64) * rq
            ex.update(res=r_nm, w_r=w_r)
            self.stats.setdefault("w_r", []).append((op.name, int(w_r.min()), int(w_r.max())))
        if act is not None and act.kind == "relu":
            s_p = self.scale(act.out)
            T[act.out] = (self.requant(acc, s_w / s_p, (0, 255 if cfg.relu_u8 else self.qa), op.name), s_p)
            done.add(act.out)
        elif act is not None and act.kind == "hswish":
            # 每通道 1.0 = q; 封顶 4096: 内核里 q6 = 6q 是 int16 (死通道的范围极小, q 会大得离谱)
            q = np.clip(np.floor(self.qa / self.amax[op.out]), 1, 4096).astype(np.int64)
            ex.update(q=q)
            y8 = self.requant(acc, s_w * q, None, op.name)
            qc = col(q).astype(np.int64)
            yt = y8 * np.clip(y8 + 3 * qc, 0, 6 * qc)                    # 单位 1 / (6q²)
            s_o = self.scale(act.out)
            T[act.out] = (self.requant(yt, 1.0 / (6.0 * q * q) / s_o, None, op.name + ".hs"), s_o)
            done.add(act.out)
        elif act is not None and act.kind == "hsig":
            y8 = self.requant(acc, s_w / (6.0 / 256), (-128, 127), op.name)
            T[act.out] = (y8 + 128, np.full(len(s_w), 1.0 / 256))
            done.add(act.out)
        else:
            s_p = self.scale(final)
            T[final] = (self.requant(acc, s_w / s_p, None, op.name), s_p)
            if add is not None:
                done.add(add.out)
        T[op.out] = T[final]


def calibrate(ops, imgs):
    return [run_float(ops, preprocess(im)) for im in imgs]


def evaluate(ops, calib, evals, cfg: Cfg, cal_imgs=None):
    qe = QEncoder(ops, calib, cfg)
    if cal_imgs is not None:                 # 冻结: 先在校准图上记下每层累加器范围 (m16 的 S1 用)
        for im in cal_imgs:
            qe.run(preprocess(im))
        qe.frozen = True
    res = {k: [] for k in FEATS}
    for im, ref in evals:
        out = qe.run(preprocess(im))
        for k, nm in FEATS.items():
            r, o = ref[nm], out[nm]
            res[k].append((cos(o, r), float(np.sqrt(((o - r) ** 2).mean()) / (np.sqrt((r ** 2).mean()) + 1e-12))))
    return {k: (np.mean([a for a, _ in v]), np.mean([b for _, b in v])) for k, v in res.items()}, qe


def main():
    ops = load_graph()
    cal_imgs, ev_imgs = images(16), images(8, offset=16)
    calib = calibrate(ops, cal_imgs)
    evals = [(im, run_float(ops, preprocess(im))) for im in ev_imgs]
    print(f"校准 {len(cal_imgs)} 张, 评估 {len(ev_imgs)} 张 (系统壁纸, 320×180)")
    cfgs = [Cfg(), Cfg(pct=99.9), Cfg(aq="channel"), Cfg(aq="channel", pct=99.99), Cfg(aq="channel", pct=99.9),
            Cfg(aq="channel", rq="m16"), Cfg(aq="channel", res="exact"), Cfg(aq="channel", se="exact")]
    if len(sys.argv) > 1:
        cfgs = [eval(f"Cfg({a})") for a in sys.argv[1:]]
    for cfg in cfgs:
        r, qe = evaluate(ops, calib, evals, cfg, cal_imgs if cfg.rq == "m16" else None)
        print(f"{cfg}  " + "  ".join(f"{k} cos {c:.4f} rel {e:.3f}" for k, (c, e) in r.items()), flush=True)
    if "w_r" in qe.stats:
        print("残差单位权重 w_r (min, max):", qe.stats["w_r"][:10])
    if "m16_sat" in qe.stats:
        print("m16 乘子饱和:", qe.stats["m16_sat"][:5])


if __name__ == "__main__":
    sys.exit(main())
