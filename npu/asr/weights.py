#!/usr/bin/env python3
"""ASR 声学模型（embed_140m…xnet）→ npz 权重包。

按 wechat-asr-gpu `Model::load` 的记录名把权重抽出来：int8 矩阵保持 i8、
per-tensor scale 单独存（`*_s`），norm/bias 是 fp32。核对每种记录的份数和形状。

    python -m npu.asr.weights            # 解析 + 核对 → work/asr/wtmodel.npz
"""
from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import numpy as np

ELEM = {6: 1, 2: 1, 3: 1, 4: 2, 5: 2, 1: 4, 7: 4, 8: 4, 11: 4, 9: 8, 10: 8, 12: 8}
MODEL = Path.home() / ".local/share/wetype/asr/model"
XNET = MODEL / "embed_140m_kld43wh_test_78000_1_merge_4.kv.conv1d.dyQu.onnx.xnet"
OUT = MODEL / "wtmodel.npz"

D, HEADS, HD, FF, LAYERS, VOCAB = 512, 8, 64, 2048, 40, 10502
FLAT = 192 * 20


class _Bad(Exception):
    pass


def chain(d: bytes, p: int):
    """记录表解析（对应 wechat-asr-gpu src/xnet.rs 的 chain）。"""
    out, L = {}, len(d)

    def u32():
        nonlocal p
        if p + 4 > L:
            raise _Bad()
        v = struct.unpack_from("<I", d, p)[0]
        p += 4
        return v

    try:
        while True:
            n = u32()
            if n == 0:
                rest = d[p:]
                if len(rest) <= 16 and all(b == 0 for b in rest) and len(out) > 100:
                    return out
                raise _Bad()
            if n > 256 or p + n > L:
                raise _Bad()
            name = d[p:p + n].decode("latin1")
            p += n
            dtype, quant = u32(), u32()
            for _ in range(3):
                u32()
            ns = u32()
            u32()
            if ns > 4096:
                raise _Bad()
            scale = None
            if ns:
                raw = d[p:p + 4]
                if len(raw) == 4 and quant:
                    scale = struct.unpack("<f", raw)[0]
            p += ns * 4
            nz = u32()
            p += nz
            nd = u32()
            if nd > 8:
                raise _Bad()
            dims = [u32() for _ in range(nd)]
            ln = u32()
            es = ELEM.get(dtype)
            if es is None:
                raise _Bad()
            prod = 1
            for v in dims:
                prod *= v
            if ln != prod * es or p + ln > L:
                raise _Bad()
            out[name] = dict(dtype=dtype, quant=quant, ns=ns, dims=dims, off=p, len=ln, scale=scale)
            p += ln
    except (_Bad, struct.error):
        return None


def parse_xnet(d: bytes):
    assert d[:4] == b"XNET"
    pat = re.compile(rb"[\x01-\x80]\x00\x00\x00")
    for m in pat.finditer(d, 8, min(len(d), 1 << 20)):
        at = m.start()
        name = d[at + 4:at + 4 + d[at]]
        if name and all(32 <= b < 127 for b in name):
            r = chain(d, at)
            if r:
                return r
    return None


def main() -> int:
    d = XNET.read_bytes()
    recs = parse_xnet(d)
    if not recs:
        print("解析失败", file=sys.stderr)
        return 1
    print(f"{XNET.name}: {len(recs)} 条记录")

    z: dict[str, np.ndarray] = {}

    def f32(name: str, dims: tuple) -> np.ndarray:
        r = recs[name]
        assert r["dtype"] == 11 and tuple(r["dims"]) == dims, (name, r["dims"])
        return np.frombuffer(d, np.float32, r["len"] // 4, r["off"]).reshape(dims)

    def i8(name: str, n: int, k: int) -> tuple[np.ndarray, float]:
        r = recs[name]
        assert r["dtype"] == 6 and r["dims"][:2] == [n, k], (name, r["dims"])
        assert r["scale"] is not None, f"{name} 缺 scale"
        return (np.frombuffer(d, np.int8, n * k, r["off"]).reshape(n, k), float(r["scale"]))

    z["c1"] = f32("frontend.conv2d.weight", (32, 1, 5, 5))
    z["c2"] = f32("frontend.conv2d_1.weight", (64, 32, 7, 7))
    z["c3"] = f32("frontend.conv2d_2.weight", (192, 64, 3, 3))
    (z["lin_w"], z["lin_s"]) = i8("frontend.linear.weight_quantized_s8", D, FLAT)
    z["lin_b"] = f32("frontend.linear.bias", (D,))

    for l in range(LAYERS):
        p = f"layers.{l}."
        for tag, nm, n, k in (("q", "attention.q_conv", D, D), ("k", "attention.k_conv", D, D),
                              ("v", "attention.v_conv", D, D), ("o", "attention.output_transform_conv", D, D),
                              ("f1", "feed_forward.filter_conv", FF, D), ("f2", "feed_forward.output_conv", D, FF)):
            z[f"l{l}_{tag}"], z[f"l{l}_{tag}_s"] = i8(p + nm + ".weight_quantized_s8", n, k)
        z[f"l{l}_f1b"] = f32(p + "feed_forward.filter_conv.bias", (FF,))
        z[f"l{l}_f2b"] = f32(p + "feed_forward.output_conv.bias", (D,))
        z[f"l{l}_png"] = f32(p + "prior_norm.weight", (D,))
        z[f"l{l}_pnb"] = f32(p + "prior_norm.bias", (D,))
        z[f"l{l}_mng"] = f32(p + "middle_norm.weight", (D,))
        z[f"l{l}_mnb"] = f32(p + "middle_norm.bias", (D,))

    (z["out_w"], z["out_s"]) = i8("output_conv.weight_quantized_s8", VOCAB, D)
    z["out_b"] = f32("output_conv.bias", (VOCAB,))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **z)
    arrays = [v for v in z.values() if isinstance(v, np.ndarray)]
    scalars = [v for v in z.values() if not isinstance(v, np.ndarray)]
    i8b = sum(v.nbytes for v in arrays if v.dtype == np.int8)
    f32b = sum(v.nbytes for v in arrays if v.dtype == np.float32)
    print(f"写出 {OUT}（{(i8b + f32b) / 1e6:.1f} MB，矩阵 {len(arrays)} 个 + scale {len(scalars)} 个："
          f"int8 {i8b / 1e6:.1f} + fp32 {f32b / 1e6:.1f}）")
    # 抽查 scale 的数值量级
    for k in ("lin_s", "l0_q_s", "l0_f1_s", "out_s"):
        print(f"  {k} = {float(z[k]):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
