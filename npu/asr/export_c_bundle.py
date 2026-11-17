"""导出 4 核 ASR NPU (2/3/5/7 秒四档) + 前/后处理权重到扁平二进制包，供 C++ XRT 常驻后端直接加载。

用法:
  source npu/env.sh && .venv-iron/bin/python -m npu.asr.export_c_bundle
"""
from __future__ import annotations

import os
import shutil
import struct
import tempfile
from pathlib import Path

# 在临时 JIT 缓存里现编各档 NPU 图, 编完只会有当次目录 —— 不再写死某个缓存哈希。
# ~/.npu/cache 会被 npu.check 和 rm -rf ~/.npu/cache/* 清掉, 所以导出包和权重缓存放在 ~/.npu/asr/。
# 必须在 import aie (经 npu.chain_c16) 之前设。
_JIT_TMP = tempfile.mkdtemp(prefix="asr_export_jit_")
os.environ["NPU_CACHE_HOME"] = _JIT_TMP

import numpy as np

from npu import chain_c16 as c
from npu.asr.encoder_npu import (
    BLOCKS_PER_CALL, BUCKETS, D, MODEL, N_CALLS, NPZ,
    FastFrontend, NPUTransformer40, _mel_mat, calibrate_and_freeze, load_wav_window, make_block_layers,
    SAMPLES_DIR,
)
from npu.asr.reference import _t

OUT_DIR = Path.home() / ".npu" / "asr" / "asr_4c_bundle"
CACHE_NPZ = Path.home() / ".npu" / "asr" / "asr_4c_weights.npz"


def _compile_bucket_xclbin(z, sh0, sh_final, all_params, W: int) -> Path:
    """在独立临时目录编译指定 W 档的 xclbin + insts.bin，返回其所在目录。"""
    for p in Path(_JIT_TMP).iterdir():
        shutil.rmtree(p, ignore_errors=True)
    r = NPUTransformer40(z, sh0, sh_final, all_params, W=W)
    h0 = np.zeros((D, 4, W), np.uint8)
    r.run_hidden(h0, min(10, 4 * W))
    dirs = [p.parent for p in Path(_JIT_TMP).rglob("final.xclbin")]
    assert len(dirs) == 1, f"W={W} 临时 JIT 缓存里应只有一个 xclbin: {dirs}"
    return dirs[0]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    z = dict(np.load(NPZ))

    vocab = [l.split()[1] if len(l.split()) >= 2 else "" for l in (MODEL / "dict.decoder.utf8.txt").read_text().splitlines()]
    (OUT_DIR / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")

    cal_wins = [load_wav_window(SAMPLES_DIR / fn) for fn in ("funasr_asr_example.wav", "ming16k.wav")]
    sh0, sh_final, all_params = calibrate_and_freeze(cal_wins, z)

    head = FastFrontend(z)
    ham, _, _, _, _ = _t()
    mel = _mel_mat()  # [256, 40] float64
    vocab_sz = head.out_w.shape[1]

    # 共享前/后处理参数 (与 W 无关)
    P0 = c.Plan(make_block_layers(BLOCKS_PER_CALL, BUCKETS[-1]))
    with open(OUT_DIR / "common.bin", "wb") as f:
        f.write(struct.pack("<6I", 0x41535235, vocab_sz,
                            P0.size["act"], P0.size["s"], P0.size["out"], N_CALLS))
        f.write(sh0.astype(np.float32).tobytes())
        f.write(sh_final.astype(np.float32).tobytes())
        f.write(head.c1_w.astype(np.float32).tobytes())
        f.write(head.c2_w.astype(np.float32).tobytes())
        f.write(head.c3_w.astype(np.float32).tobytes())
        f.write(head.lin_w.astype(np.float32).tobytes())
        f.write(head.lin_b.astype(np.float32).tobytes())
        f.write(head.out_w.astype(np.float32).tobytes())
        f.write(head.out_b.astype(np.float32).tobytes())
        f.write(np.ascontiguousarray(ham, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(mel, dtype=np.float64).tobytes())

    # 逐档导出 xclbin, insts.bin, bucket.bin
    for W in BUCKETS:
        bdir = OUT_DIR / f"w{W}"
        bdir.mkdir(parents=True, exist_ok=True)
        xdir = _compile_bucket_xclbin(z, sh0, sh_final, all_params, W)
        shutil.copy2(xdir / "final.xclbin", bdir / "final.xclbin")
        shutil.copy2(xdir / "insts.bin", bdir / "insts.bin")

        layers = make_block_layers(BLOCKS_PER_CALL, W)
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

        w_arrs = []
        for g in range(N_CALLS):
            p_grp = all_params[g * BLOCKS_PER_CALL * 9 : (g + 1) * BLOCKS_PER_CALL * 9]
            w_arr = np.concatenate([
                w for k, (l, tl, p) in enumerate(zip(layers, P.tilings, p_grp))
                for w in c.pack_layer(l, tl, p, P.e_w, P.tin[k].ws, c.signed_in(layers, k),
                                      P.tensors[k + 1 + l.src].c // 16)
            ])
            w_arrs.append(np.ascontiguousarray(w_arr, dtype=np.int8))

        w_bytes = int(w_arrs[0].nbytes)
        with open(bdir / "bucket.bin", "wb") as f:
            f.write(struct.pack("<4I", W, len(vt_arr), w_bytes, N_CALLS))
            f.write(vt_arr.tobytes())
            for w_arr in w_arrs:
                f.write(w_arr.tobytes())
        print(f"[export] W={W:2d} (T={4*W:3d}) -> {bdir} (w_bytes={w_bytes})")

    shutil.rmtree(_JIT_TMP, ignore_errors=True)
    print(f"[export] all buckets exported to {OUT_DIR} (vocab={vocab_sz})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
