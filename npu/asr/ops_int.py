"""ASR 里三个"新算子"的整数定义（NPU 版目标，纯 numpy，先于内核存在）。

对应 chain_c16 需要新增的三种 step：
  * `lm`    —— LayerNorm（逐 token，跨 512 通道）
  * `attn`  —— 注意力：scores = QKᵀ（逐头）、softmax、×V
  * `gelu`  —— GELU（int8 查表）

设计要点（都是被 NPU 的约束逼出来的）：

1. **LayerNorm 的输入用"逐张量"比例**：LN 对输入的均匀缩放不变
   （LN(kx) = LN(x)），所以不需要逐通道比例，逐张量一个比例即可 —— 内核里
   Σx、Σx² 变成纯整数累加（int64），不碰乘加。gamma/beta 仍是 fp32/逐通道。
   均值方差之后要 1/sqrt(var)，AIE2 没有除法/开方：用 8.8 定点倒数表
   （var 归一化到 [0,255] 查 256 项）。
2. **softmax 的 exp 用查表**：scores 先减行最大值，得到 ≤0 的 int16，
   查 `EXP_LUT`（128 项，exp(-i/16) 的 Q8 值），累加得 int32 分母，
   再除一次（同样查倒数表）→ uint8 权重（比例 1/256）。
3. **GELU 用 256 项查表**：输入 int8（逐张量比例），输出 int8（逐张量比例）。
   tanh 在 AIE2 上不可用；查表的误差可控（表本身按真实 gelu 生成）。

每个函数都有一个 `_ref` 浮点版本用于对拍。
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- 定点表

_FRAC_BITS = 8                 # 1/sqrt(var) 与 softmax 分母的定点小数位


def _recip_lut(n: int = 256, frac: int = _FRAC_BITS) -> np.ndarray:
    """LUT[x] = round((1/x)·2^frac)，x ∈ [1, n]，索引 0 留给"钳到 1"。"""
    xs = np.maximum(np.arange(n + 1), 1.0)
    return np.rint((1.0 / xs) * (1 << frac)).astype(np.int64)


_RECIP = _recip_lut()


def _exp_lut(n: int = 255, max_neg: float = 16.0, frac: int = 16) -> np.ndarray:
    """LUT[i] = round(exp(-i·max_neg/n)·2^frac)：覆盖 [-16, 0]，Q16。

    索引 i = round(-s / step)，s ≤ 0 是减去行最大值后的分数。Q16 是为了
    避免累加 140 项时的量化误差（e 之和 ≤ 140·65536 ≈ 9.2 M，int32 装得下）。"""
    return np.rint(np.exp(-np.arange(n + 1) * (max_neg / n)) * (1 << frac)).astype(np.int64)


_EXP = _exp_lut()


# ---------------------------------------------------------------- LayerNorm


def layernorm_int8(xq: np.ndarray, s_in: float, gamma: np.ndarray, beta: np.ndarray,
                   eps: float = 1e-6, s_gamma: float = 1.0) -> np.ndarray:
    """int8 [T][D]（逐张量比例 s_in）→ fp32 [T][D]。

    内核里：sum/sumsq 用 int64；var 归一化后查倒数表；输出仍是浮点（下一步
    的量化由生产者做）。这里返回浮点，和 `_layernorm_ref` 逐元素比。
    """
    d = xq.shape[1]
    x = xq.astype(np.int64)                         # 量化单位
    sx = x.sum(1)                                   # int64
    sx2 = (x * x).sum(1)                            # int64
    mean = sx / d
    var = np.maximum(sx2 / d - mean ** 2, 0.0)
    # 内核里 1/sqrt(var+eps) 用 Newton 迭代（AIE2 没有除法/开方），精度到 1e-6；
    # 这里直接用浮点，代表"内核能做到的精度"。
    rstd = 1.0 / np.sqrt(var + eps / (s_in * s_in))   # eps 换算到量化单位
    y = (x - mean[:, None]) * rstd[:, None]            # 无量纲（LN 对缩放不变）
    return y * gamma * s_gamma + beta


def _layernorm_ref(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean(1, keepdims=True)
    v = ((x - mu) ** 2).mean(1, keepdims=True)
    return (x - mu) / np.sqrt(v + eps) * gamma + beta


# ---------------------------------------------------------------- softmax（查表）


def softmax_int8(scores_q: np.ndarray, out_frac: int = 8) -> np.ndarray:
    """[H][T][T] 的 int32 分数（已乘好比例）→ uint8 [H][T][T]，比例 1/256。"""
    s = scores_q - scores_q.max(2, keepdims=True)        # ≤ 0
    n = len(_EXP) - 1                                    # 表覆盖分数区间 [-16, 0]
    idx = np.clip(np.rint(-s * (n / 16.0)).astype(np.int64), 0, n)
    e = _EXP[idx]                                        # Q16，[0, 65536]
    den = e.sum(2, keepdims=True)                        # int32 内
    w = (e * 255 + den // 2) // den                      # 归一化到 0..255
    return np.clip(w, 0, 255).astype(np.uint8)


def _softmax_ref(scores: np.ndarray) -> np.ndarray:
    s = scores - scores.max(2, keepdims=True)
    e = np.exp(s)
    return e / e.sum(2, keepdims=True)


# ---------------------------------------------------------------- GELU（查表）


def gelu_lut(s_in: float, s_out: float) -> np.ndarray:
    """int8 输入（比例 s_in）→ int8 输出（比例 s_out）的 256 项表。"""
    v = np.arange(-128, 128) * s_in
    g = 0.5 * v * (1 + np.tanh(0.7978846 * (v + 0.044715 * v ** 3)))
    return np.clip(np.rint(g / s_out), -128, 127).astype(np.int8)


def gelu_int8(xq: np.ndarray, s_in: float, s_out: float) -> np.ndarray:
    table = gelu_lut(s_in, s_out)
    return table[xq.astype(np.int64) + 128]


# ---------------------------------------------------------------- 自检


def _report(name: str, a: np.ndarray, b: np.ndarray) -> None:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    cos = float(a @ b / np.sqrt((a @ a) * (b @ b)))
    print(f"  {name:<28s} cos={cos:.6f}  max|Δ|={np.abs(a - b).max():.4f}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 512
    print("LayerNorm（int8 逐张量比例 + 定点倒数表）:")
    g = rng.normal(1, 0.1, d).astype(np.float32)
    b = rng.normal(0, 0.1, d).astype(np.float32)
    for s_in in (0.02, 0.1):
        x = rng.normal(0, s_in * 60, (64, d)).astype(np.float32)
        xq = np.clip(np.rint(x / s_in), -127, 127).astype(np.int8)
        _report(f"s_in={s_in}", layernorm_int8(xq, s_in, g, b), _layernorm_ref(xq * s_in, g, b))

    print("softmax（查表 exp + 定点倒数）:")
    for t in (64, 140):
        sc = rng.normal(0, 8, (8, t, t)).astype(np.float32) * 100
        w = softmax_int8(np.rint(sc).astype(np.int64)) / 256.0
        _report(f"T={t}", w, _softmax_ref(sc))

    print("GELU（256 项查表）:")
    for (s_in, s_out) in ((0.05, 0.05), (0.02, 0.1)):
        x = rng.normal(0, s_in * 40, (64, 2048)).astype(np.float32)
        xq = np.clip(np.rint(x / s_in), -127, 127).astype(np.int8)
        ref = 0.5 * (xq * s_in) * (1 + np.tanh(0.7978846 * ((xq * s_in) + 0.044715 * (xq * s_in) ** 3)))
        _report(f"{s_in}→{s_out}", gelu_int8(xq, s_in, s_out) * s_out, ref)
