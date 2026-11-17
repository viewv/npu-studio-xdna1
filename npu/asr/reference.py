"""ASR 声学模型的 numpy 参考实现（浮点 + 逐行动态量化两条路径）。

与 `wechat-asr-gpu` 的 Rust CPU 路径逐算子对齐：
fbank（HTK 风格 39+1 维、CMS 50 帧）→ 3 层卷积前端 → 40 层 pre-norm
Transformer → CTC argmax。`forward(int=True)` 复刻 Rust 的量化语义
（权重 per-tensor int8、激活逐行 absmax int8、int32 累加），用于和
NPU 结果逐字节比对；`int=False` 是纯 fp32 参考（权重量化误差的上界）。

    python -m npu.asr.reference --wav xx.wav       # 转写（和 wasr 对拍）
    python -m npu.asr.reference --random           # 随机窗自检两条路径
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

from npu.asr.weights import MODEL, OUT as NPZ

D, HEADS, HD, FF, LAYERS = 512, 8, 64, 2048, 40
FLAT = 192 * 20
WIN, HOP, NFFT, NCH, CMS_WIN = 400, 160, 512, 39, 50


# ------------------------------------------------------------------ fbank


def _tables():
    pi = np.pi
    ham = 0.54 - 0.46 * np.cos(2 * pi * np.arange(WIN) / (WIN - 1))
    mel = lambda f: 1127.0 * np.log1p(f / 700.0)
    hi = mel(8000.0)
    cf = hi * np.arange(1, NCH + 2) / (NCH + 1)          # 40 个中心
    ks = np.arange(1, NFFT // 2 + 1)
    m = mel(ks * 16000.0 / NFFT)
    c = (cf[None, :] < m[:, None]).sum(1)                # 每个频点落进哪个通道
    lo = np.zeros(len(ks), np.int64)
    wl = np.zeros(len(ks))
    hi_b = np.zeros(len(ks), np.int64)
    wh = np.zeros(len(ks))
    for i in range(len(ks)):
        cc = c[i]
        if cc == 0:
            lo[i], wl[i] = 0, m[i] / cf[0]
        elif cc < NCH:
            w = (cf[cc] - m[i]) / (cf[cc] - cf[cc - 1])
            lo[i], wl[i], hi_b[i], wh[i] = cc - 1, w, cc, 1.0 - w
        elif cc == NCH:
            lo[i] = NCH - 1
            wl[i] = (cf[NCH] - m[i]) / (cf[NCH] - cf[NCH - 1])
    return ham, lo, wl, hi_b, wh


_T = None


def _t():
    global _T
    if _T is None:
        _T = _tables()
    return _T


def fbank(pcm: np.ndarray) -> np.ndarray:
    """i16 PCM → [t][40] f32 帧（滑动均值归一化）。"""
    ham, lo, wl, hi_b, wh = _t()
    n = len(pcm)
    nfr = (n - WIN) // HOP + 1 if n >= WIN else 0
    frames = np.zeros((nfr, NCH + 1))
    hist = np.zeros((0, NCH + 1))
    for k in range(nfr):
        x = pcm[k * HOP:k * HOP + WIN].astype(np.float64)
        e = (x * x).sum()
        pe = np.empty(WIN)
        pe[0] = x[0] * (1.0 - 0.97)
        pe[1:] = x[1:] - 0.97 * x[:-1]
        re = pe * ham
        spec = np.fft.rfft(re, NFFT)
        p = (spec.real[1:NFFT // 2 + 1] ** 2 + spec.imag[1:NFFT // 2 + 1] ** 2)
        out = np.zeros(NCH + 1)
        np.add.at(out, lo, p * wl)
        np.add.at(out, hi_b, p * wh)
        out[:NCH] = np.log(np.maximum(out[:NCH], 1.0))
        out[NCH] = np.log(max(e, 1e-10))
        hist = np.vstack([hist[-(CMS_WIN - 1):], out])
        frames[k] = out - hist.mean(0)
    return frames.astype(np.float32)


# ------------------------------------------------------------------ 前端（3 层卷积）


def frontend(x: np.ndarray, z) -> tuple[int, np.ndarray]:
    """[t][40] 帧 → (t1, [t1][3840] 展平特征)。布局按 Rust frontend()。"""
    t = len(x)
    t1 = (t - 5) // 5 + 1

    # conv1: 5×5, 1→32, 时间/频率各补 2 → y1 [t][32][40]
    xp = np.zeros((t + 4, 44))
    xp[2:2 + t, 2:42] = x
    y1 = np.zeros((t, 32, 40))
    for i in range(5):
        for j in range(5):
            y1 += np.einsum("o,tf->tof", z["c1"][:, 0, i, j], xp[i:i + t, j:j + 40])
    y1 = np.maximum(y1, 0)

    # conv2: 7×7, 32→64, 时间步长 5（补 1）、频率步长 1（补 3）→ [t1][64][41]（末列 0）
    y1p = np.zeros((t + 2, 32, 46))                       # 时间补 1、频率补 3
    y1p[1:1 + t, :, 3:43] = y1
    y2 = np.zeros((t1, 64, 41))
    for ti in range(t1):
        win = np.empty((40, 32, 7, 7))
        for i in range(7):
            for j in range(7):
                win[:, :, i, j] = y1p[ti * 5 + i, :, j:j + 40].T   # [f][c]
        y2[ti, :, :40] = np.maximum(np.einsum("ocij,fcij->of", z["c2"], win), 0)

    # conv3: 3×3, 64→192, 时间补 1 步长 1、频率步长 2 → [t1][192][20]
    y2p = np.zeros((t1 + 2, 64, 41))
    y2p[1:1 + t1] = y2
    y3 = np.zeros((t1, 192, 20))
    for ti in range(t1):
        win = np.empty((20, 64, 3, 3))
        for i in range(3):
            for j in range(3):
                win[:, :, i, j] = y2p[ti + i, :, j:j + 39:2].T       # 频率 2f+j
        y3[ti] = np.maximum(np.einsum("ocij,fcij->of", z["c3"], win), 0)
    return t1, y3.reshape(t1, FLAT)


# ------------------------------------------------------------------ transformer


def _quant(x: np.ndarray):
    m = np.abs(x).max(1)
    sc = np.where(m > 0, m / 127.0, 1.0)
    q = np.clip(np.rint(x / sc[:, None]), -127, 127).astype(np.int8)
    return q, sc


def _mm(q: np.ndarray, xs: np.ndarray, W: np.ndarray, ws: float, bias: np.ndarray | None,
        int_path: bool) -> np.ndarray:
    if int_path:
        acc = (q.astype(np.int32) @ W.T.astype(np.int32))
        y = acc * xs[:, None] * ws
    else:
        y = q @ (W.T * ws)
    return y + bias if bias is not None else y


def _norm(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    mu = x.mean(1, keepdims=True)
    v = ((x - mu) ** 2).mean(1, keepdims=True)
    return (x - mu) / np.sqrt(v + 1e-6) * g + b


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(0.7978846 * (x + 0.044715 * x ** 3)))


def _attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, t: int) -> np.ndarray:
    qh = q.reshape(t, HEADS, HD).transpose(1, 0, 2)               # [h][t][hd]
    kh = k.reshape(t, HEADS, HD).transpose(1, 0, 2)
    vh = v.reshape(t, HEADS, HD).transpose(1, 0, 2)
    s = qh @ kh.transpose(0, 2, 1) * 0.125
    s -= s.max(2, keepdims=True)
    a = np.exp(s)
    a /= a.sum(2, keepdims=True)
    o = a @ vh                                                    # [h][t][hd]
    return o.transpose(1, 0, 2).reshape(t, D)


def forward(frames: np.ndarray, z, int_path: bool = True, layers: int = LAYERS) -> np.ndarray:
    """fbank 帧 → 每 token 的 argmax id。"""
    t1, y3 = frontend(frames, z)
    if int_path:
        q, sc = _quant(y3)
        h = _mm(q, sc, z["lin_w"], float(z["lin_s"]), z["lin_b"], True)
    else:
        h = y3 @ (z["lin_w"].T * float(z["lin_s"])) + z["lin_b"]
    for l in range(layers):
        p = f"l{l}_"
        y = _norm(h, z[p + "png"], z[p + "pnb"])
        if int_path:
            qq, sc = _quant(y)
            qm = _mm(qq, sc, z[p + "q"], float(z[p + "q_s"]), None, True)
            km = _mm(qq, sc, z[p + "k"], float(z[p + "k_s"]), None, True)
            vm = _mm(qq, sc, z[p + "v"], float(z[p + "v_s"]), None, True)
        else:
            qm = y @ (z[p + "q"].T * float(z[p + "q_s"]))
            km = y @ (z[p + "k"].T * float(z[p + "k_s"]))
            vm = y @ (z[p + "v"].T * float(z[p + "v_s"]))
        att = _attention(qm, km, vm, t1)
        if int_path:
            aq, sc = _quant(att)
            h = h + _mm(aq, sc, z[p + "o"], float(z[p + "o_s"]), None, True)
            nf = _norm(h, z[p + "mng"], z[p + "mnb"])
            nq, sc = _quant(nf)
            f = _mm(nq, sc, z[p + "f1"], float(z[p + "f1_s"]), z[p + "f1b"], True)
        else:
            h = h + att @ (z[p + "o"].T * float(z[p + "o_s"]))
            f = _norm(h, z[p + "mng"], z[p + "mnb"]) @ (z[p + "f1"].T * float(z[p + "f1_s"])) + z[p + "f1b"]
        f = _gelu(f)
        if int_path:
            fq, sc = _quant(f)
            h = h + _mm(fq, sc, z[p + "f2"], float(z[p + "f2_s"]), z[p + "f2b"], True)
        else:
            h = h + f @ (z[p + "f2"].T * float(z[p + "f2_s"])) + z[p + "f2b"]
    if int_path:
        hq, sc = _quant(h)
        lo = _mm(hq, sc, z["out_w"], float(z["out_s"]), z["out_b"], True)
    else:
        lo = h @ (z["out_w"].T * float(z["out_s"])) + z["out_b"]
    return lo.argmax(1).astype(np.uint32)


def text(ids: np.ndarray, vocab: list[str]) -> str:
    out, prev = [], 2 ** 32
    for i in ids:
        i = int(i)
        if i != prev and i != 0:
            t = vocab[i]
            if t == "<UNK>":
                pass
            elif t == "<SPACE>":
                out.append(" ")
            elif len(t) > 1 and t.startswith("|"):
                if out and out[-1][-1:].isascii() and out[-1][-1:].isalnum():
                    out.append(" ")
                out.append(t[1:])
            else:
                out.append(t)
        prev = i
    return "".join(out)


def main() -> int:
    z = dict(np.load(NPZ))
    vocab = (MODEL / "dict.decoder.utf8.txt").read_text().splitlines()
    # 字典行格式：`id token 标志`（标志 W/P 只对解码策略有用，不是词的一部分）
    vocab = [l.split()[1] if len(l.split()) >= 2 else "" for l in vocab]
    if "--random" in sys.argv:
        rng = np.random.default_rng(0)
        frames = rng.normal(0, 1, (70, 40)).astype(np.float32)
        a = forward(frames, z, int_path=True)
        b = forward(frames, z, int_path=False)
        agree = (a == b).mean()
        print(f"随机窗自检：int/fp32 token 一致率 {agree * 100:.1f}%  ids[:12] int={a[:12]} fp32={b[:12]}")
        return 0 if agree > 0.5 else 1
    if "--dump" in sys.argv:
        """和 asrd/examples/refdump.rs 的 frames 模式对拍：同一 LCG、同一 forward。"""
        n = int(sys.argv[sys.argv.index("--dump") + 1]) if len(sys.argv) > sys.argv.index("--dump") + 1 else 200
        scale = float(sys.argv[sys.argv.index("--dump") + 2]) if len(sys.argv) > sys.argv.index("--dump") + 2 else 100.0
        s = 0x243F6A8885A308D3
        frames = np.zeros((n, 40), np.float32)
        for t in range(n):
            for f in range(40):
                s = (s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
                u = (s >> 11) / (1 << 53)
                frames[t, f] = (u * 2 - 1) * scale
        ids = forward(frames, z, int_path=True)
        mine = ",".join(str(int(i)) for i in ids)
        ref = Path(sys.argv[sys.argv.index("--dump") + 3] if len(sys.argv) > sys.argv.index("--dump") + 3 else
                   "/tmp/opencode/rust_ids.txt").read_text().strip()
        ref = ref.removeprefix("ids=").rstrip(",")
        ok = mine == ref
        print(f"对拍 {'✅ 完全一致' if ok else '❌ 不一致'}：{n} 帧 scale={scale}")
        if not ok:
            a = np.array([int(x) for x in ref.split(",")])
            b = np.array([int(i) for i in ids])
            diff = (a != b)
            print(f"  一致率 {(~diff).mean() * 100:.1f}%  首个不同位置 {int(np.argmax(diff)) if diff.any() else -1}")
            print(f"  rust[:12] {a[:12]}\n  mine[:12] {b[:12]}")
        return 0 if ok else 1
    if "--wav" in sys.argv:
        path = sys.argv[sys.argv.index("--wav") + 1]
        expect = sys.argv[sys.argv.index("--expect") + 1] if "--expect" in sys.argv else None
        with wave.open(path) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        frames = fbank(pcm)
        out = {}
        for tag, logits in (("int", None), ("npu", None)):
            pass
        ref = text(forward(frames, z, int_path=True), vocab)
        npu = text(forward_npu(frames, z).argmax(1).astype(np.uint32), vocab)
        print(f"int : {ref}")
        print(f"npu : {npu}")
        if expect:
            import difflib
            for tag, got in (("int", ref), ("npu", npu)):
                cer = 1 - difflib.SequenceMatcher(None, expect, got).ratio()
                print(f"  {tag} CER {cer * 100:.1f}%")
        return 0 if ref == npu and (not expect or ref == expect) else 1
    print(__doc__)
    return 1


# ------------------------------------------------------------------ 伪量化（NPU 方案评估）
#
# 目标：NPU 上激活是"静态逐通道"（比例在校准集上定好、折进下一层权重），
# 而 Rust 引擎是"逐行动态"（每 token 自己算 absmax，NPU 上做不到）。
# 这里用浮点伪量化（照 encoder_q/decoder_q 的方法论）评估两种方案的差距：
#   把 x 量化成 round(x/s)·s 再算，权重同理；其余算子（LN/softmax/GELU）仍是 fp32。


def _q_per_row(x: np.ndarray) -> np.ndarray:
    m = np.abs(x).max(1)
    s = np.where(m > 0, m / 127.0, 1.0)
    return np.rint(x / s[:, None]).clip(-127, 127) * s[:, None]


def _q_per_channel(x: np.ndarray, cal: np.ndarray | None = None) -> np.ndarray:
    m = np.abs(cal if cal is not None else x).max(0)
    s = np.where(m > 0, m / 127.0, 1.0)
    return np.rint(x / s).clip(-127, 127) * s


def _q_per_tensor(x: np.ndarray) -> np.ndarray:
    m = np.abs(x).max()
    s = m / 127.0 if m > 0 else 1.0
    return np.rint(x / s).clip(-127, 127) * s


def forward_fq(frames: np.ndarray, z, mode: str = "pc", cal_frames: np.ndarray | None = None,
               layers: int = LAYERS) -> np.ndarray:
    """伪量化前向：mode = row（动态逐行）/ pc（静态逐通道）/ pt（静态逐张量）。
    pc 时用 cal_frames（默认用 frames 自己）按通道取 absmax 当校准。"""
    qf = {"row": _q_per_row, "pc": _q_per_channel, "pt": _q_per_tensor}[mode]
    # 校准：把整段 fbank 先过一遍前端拿特征，按通道统计（简化：用当前窗口）
    t1, y3 = frontend(frames, z)
    if mode == "pc" and cal_frames is not None:
        _, y3c = frontend(cal_frames, z)
    else:
        y3c = y3
    acc = {"cal": y3c}

    def Q(x: np.ndarray) -> np.ndarray:
        if mode == "pc":
            return _q_per_channel(x, acc["cal"])
        return qf(x)

    def mm(x: np.ndarray, W: np.ndarray, ws: float, b) -> np.ndarray:
        y = Q(x) @ (W.T * ws)
        return y + b if b is not None else y

    h = mm(y3, z["lin_w"], float(z["lin_s"]), z["lin_b"])
    acc["cal"] = h      # 之后的校准用同形状的上一窗口近似（仅为量级检查）
    for l in range(layers):
        p = f"l{l}_"
        y = _norm(h, z[p + "png"], z[p + "pnb"])
        acc["cal"] = y
        qm = mm(y, z[p + "q"], float(z[p + "q_s"]), None)
        km = mm(y, z[p + "k"], float(z[p + "k_s"]), None)
        vm = mm(y, z[p + "v"], float(z[p + "v_s"]), None)
        att = _attention(qm, km, vm, t1)
        acc["cal"] = att
        h = h + mm(att, z[p + "o"], float(z[p + "o_s"]), None)
        acc["cal"] = _norm(h, z[p + "mng"], z[p + "mnb"])
        f = mm(acc["cal"], z[p + "f1"], float(z[p + "f1_s"]), z[p + "f1b"])
        f = _gelu(f)
        acc["cal"] = f
        h = h + mm(f, z[p + "f2"], float(z[p + "f2_s"]), z[p + "f2b"])
    acc["cal"] = h
    lo = mm(h, z["out_w"], float(z["out_s"]), z["out_b"])
    return lo


# ------------------------------------------------------------------ NPU 方案端到端


def forward_npu(frames: np.ndarray, z, layers: int = LAYERS) -> np.ndarray:
    """整条 NPU 方案的伪量化前向：

    激活 静态逐通道比例（逐张量比例给 LN 的输入，因为 LN 对缩放不变）、
    LayerNorm / softmax / GELU 走整数定义（npu/asr/ops_int.py）。
    返回 logits（float，可和 row 参考比余弦/argmax）。
    """
    from npu.asr.ops_int import gelu_int8, layernorm_int8, softmax_int8

    def qpc(x):
        m = np.abs(x).max(0)
        s = np.where(m > 0, m / 127.0, 1.0)
        return np.rint(x / s).clip(-127, 127) * s

    def mm(x, W, ws, b):
        y = qpc(x) @ (W.T * ws)
        return y + b if b is not None else y

    def ln(x, g, b):
        # LN 输入：逐通道比例（实测逐张量会在 40 层后丢字，见 ops_int 的说明）
        return layernorm_int8(x, g, b)

    t1, y3 = frontend(frames, z)
    h = mm(y3, z["lin_w"], float(z["lin_s"]), z["lin_b"])
    for l in range(layers):
        p = f"l{l}_"
        y = ln(h, z[p + "png"], z[p + "pnb"])
        qm = mm(y, z[p + "q"], float(z[p + "q_s"]), None)
        km = mm(y, z[p + "k"], float(z[p + "k_s"]), None)
        vm = mm(y, z[p + "v"], float(z[p + "v_s"]), None)
        # ── 注意力：int32 分数（·0.125 折进量化比例）→ 查表 softmax → uint8 权重 × V
        qh = qm.reshape(t1, HEADS, HD).transpose(1, 0, 2)
        kh = km.reshape(t1, HEADS, HD).transpose(1, 0, 2)
        vh = vm.reshape(t1, HEADS, HD).transpose(1, 0, 2)
        sc = (qh @ kh.transpose(0, 2, 1)) * 0.125
        w = softmax_int8(sc).astype(np.float64) / 255.0
        att = (w @ vh).transpose(1, 0, 2).reshape(t1, D)   # [h][t][hd] → [t][D]
        h = h + mm(att, z[p + "o"], float(z[p + "o_s"]), None)
        nf = ln(h, z[p + "mng"], z[p + "mnb"])
        f = mm(nf, z[p + "f1"], float(z[p + "f1_s"]), z[p + "f1b"])
        # ── GELU：int8 查表（输入/输出各一个逐张量比例）
        mf = np.abs(f).max()
        sf = mf / 127.0 if mf > 0 else 1.0
        fq = np.clip(np.rint(f / sf), -127, 127).astype(np.int8)
        # 输出比例按 gelu 实际输出范围校准（负半轴会被压到 -0.17 附近）
        g = 0.5 * f * (1 + np.tanh(0.7978846 * (f + 0.044715 * f ** 3)))
        mg = np.abs(g).max()
        so = mg / 127.0 if mg > 0 else 1.0
        f = gelu_int8(fq, sf, so).astype(np.float64) * so
        h = h + mm(f, z[p + "f2"], float(z[p + "f2_s"]), z[p + "f2b"])
    return mm(h, z["out_w"], float(z["out_s"]), z["out_b"])


if __name__ == "__main__":
    sys.exit(main())
