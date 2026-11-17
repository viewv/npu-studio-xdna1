"""导出 4 核 ASR NPU + 前/后处理权重到扁平二进制包，供 C++ XRT 常驻后端直接加载。

用法:
  source npu/env.sh && .venv-iron/bin/python -m npu.asr.export_c_bundle
"""
from __future__ import annotations

import os
import shutil
import struct
import tempfile
from pathlib import Path

# 在临时 JIT 缓存里现编 7 秒档 (W=35) 的 NPU 图, 编完只会有这一个目录 —— 不再写死某个缓存哈希。
# ~/.npu/cache 会被 npu.check 和 rm -rf ~/.npu/cache/* 清掉, 所以导出包和权重缓存放在 ~/.npu/asr/。
# 必须在 import aie (经 npu.chain_c16) 之前设。
_JIT_TMP = tempfile.mkdtemp(prefix="asr_export_jit_")
os.environ["NPU_CACHE_HOME"] = _JIT_TMP

import numpy as np

from npu import chain_c16 as c
from npu.asr.encoder_npu import (
    BLOCKS_PER_CALL, D, FLAT, MODEL, N_CALLS, NPZ, NCH, NFFT, T_WIN,
    FastFrontend, NPUTransformer40, _mel_mat, make_block_layers,
)
from npu.asr.reference import _t

OUT_DIR = Path.home() / ".npu" / "asr" / "asr_4c_bundle"
CACHE_NPZ = Path.home() / ".npu" / "asr" / "asr_4c_weights.npz"


def _compile_xclbin(z) -> Path:
    """校准 (或读权重缓存) + 跑一次 40 层, 触发 JIT 编译; 返回临时缓存里 final.xclbin 所在目录。"""
    r = NPUTransformer40.from_cache(z, CACHE_NPZ)
    h0 = np.zeros((D, 4, 35), np.uint8)
    r.run_hidden(h0, 10)
    dirs = [p.parent for p in Path(_JIT_TMP).rglob("final.xclbin")]
    assert len(dirs) == 1, f"临时 JIT 缓存里应只有一个 xclbin: {dirs}"
    return dirs[0]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    z0 = dict(np.load(NPZ))
    xdir = _compile_xclbin(z0)
    shutil.copy2(xdir / "final.xclbin", OUT_DIR / "final.xclbin")
    shutil.copy2(xdir / "insts.bin", OUT_DIR / "insts.bin")

    vocab = [l.split()[1] if len(l.split()) >= 2 else "" for l in (MODEL / "dict.decoder.utf8.txt").read_text().splitlines()]
    (OUT_DIR / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")

    z = dict(np.load(NPZ))
    head = FastFrontend(z)
    d = np.load(CACHE_NPZ)

    # 计算 vt_idx
    layers = make_block_layers(BLOCKS_PER_CALL)
    P = c.Plan(layers)
    vt_idx = []
    for k, l in enumerate(layers):
        if l.op == 6:
            s0 = P.step0[k]
            for step_i, st in enumerate(P.tilings[k].steps):
                if st[7] == 0:
                    for core_c in range(c.N_CORES):
                        byte_off = ((s0 + step_i) * c.N_CORES + core_c) * P.e_w + 62 * 4
                        vt_idx.append(byte_off // 4)
    vt_arr = np.array(vt_idx, dtype=np.int64)

    ham, _, _, _, _ = _t()
    mel = _mel_mat()  # [256, 40] float64
    vocab_sz = head.out_w.shape[1]
    w_bytes = int(d["w0"].nbytes)

    with open(OUT_DIR / "weights.bin", "wb") as f:
        # Header: magic, vocab_sz, n_vt, act_bytes, s_bytes, out_bytes, w_bytes
        f.write(struct.pack("<8I", 0x41535234, vocab_sz, len(vt_arr),
                            P.size["act"], P.size["s"], P.size["out"], w_bytes, N_CALLS))
        f.write(vt_arr.tobytes())
        f.write(d["sh0"].astype(np.float32).tobytes())
        f.write(d["sh_final"].astype(np.float32).tobytes())
        f.write(head.c1_w.astype(np.float32).tobytes())
        f.write(head.c2_w.astype(np.float32).tobytes())
        f.write(head.c3_w.astype(np.float32).tobytes())
        f.write(head.lin_w.astype(np.float32).tobytes())
        f.write(head.lin_b.astype(np.float32).tobytes())
        f.write(head.out_w.astype(np.float32).tobytes())
        f.write(head.out_b.astype(np.float32).tobytes())
        f.write(np.ascontiguousarray(ham, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(mel, dtype=np.float64).tobytes())
        for g in range(N_CALLS):
            f.write(np.ascontiguousarray(d[f"w{g}"], dtype=np.int8).tobytes())

    print(f"[export] exported bundle to {OUT_DIR} (vocab={vocab_sz}, w_bytes={w_bytes})")
    shutil.rmtree(_JIT_TMP, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
