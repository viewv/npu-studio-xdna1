# XDNA1 AIE2 C16 Kernel & Chain Library (`npu/`)

This project's NPU backend (`npu/chain_c16.py` + `npu/kernels/conv_c16.cc`) forms a reusable, bit-exact **int8 Neural Network Kernel & Chain Compiler** for AMD Phoenix XDNA1 (AIE2) NPUs via `mlir-aie` (IRON) and XRT.

All operators are verified bit-exact (`max_diff == 0`) against numpy integer references via `python -m npu.check --quick` (`19/19` passing).

---

## 1. Supported Operators (`chain_c16.Layer`)

### Vision / CNN Operators (Used by RVM, Ready for DBNet / MobileNet / PPLCNet)
| Operator | `Layer` Config | Implementation (`conv_c16.cc`) | Notes |
|---|---|---|---|
| **1×1 Conv / Linear** | `op=0, k=1` | `aie::mmul<4, 8, 8>` (int8 × int8 → int32) | Supports per-channel int16 requantization (`s1, M, s2`) + bias |
| **3×3 Conv (s=1, s=2)** | `op=0, k=3, s=1|2` | `conv3s1`, `conv3s2` | Uses `aie::shuffle_up_fill` / `filter_even` |
| **Depthwise Conv** | `dw=True, k=3|5, d=1|2` | `dwconv` | 3×3, 5×5, and 5×5 `dilation=2` |
| **Squeeze-and-Excite (SE)** | `op=1` (pool) + `sc=-k` | `se_pool`, `se_load`, `se_wscale` | Global avg pool → FC → HardSigmoid weight scaling |
| **2× Bilinear Upsample** | `op=2` | `upsample2` / `upsample_step` | Align-corners=False bilinear ×2 |
| **Multi-input / Concat / Alias** | `src`, `cat`, `op=3` (ext input) | DDR `Plan` zero-copy layout | `nx=-1` skips redundant fill; in-place alias supported |
| **Activations** | `act=0|1|2|3|5` | `post_act` | `0` none, `1` ReLU, `2` ReLU6/clip, `3` HardSwish, `5` **GELU** (`aie::parallel_lookup` 2KB LUT) |

### Transformer / Sequence Operators (Used by 40-Layer ASR, Ready for SVTR / Conformer)
| Operator | `Layer` Config | Implementation (`conv_c16.cc`) | Notes |
|---|---|---|---|
| **LayerNorm (512-ch)** | `op=5` | `ln_step` + `rsqrt_q12` | 16-ch SIMD `acc64` mean/var + Newton-Raphson `rsqrt` (`Q12`), bit-exact with int64 ref |
| **Multi-Head Self-Attention** | `op=6, ic=3*oc` | `attn_step` | `Q@K^T` and `W@V` via `aie::mmul<4,8,8>`, `EXP_LUT` softmax, runtime `valid_t` & variable `W` |

---

## 2. Tensor Layout & Hardware Constraints

- **Channel-packed `C16` layout**: `[H + 2*BR][C/16][Ws + 2*PAD][16]` (`BR=4`, `PAD=4`, `CW=40`).
- **4-Core Row-Parallel Chain (`C16Chain`)**:
  - Splits height `H` across 4 AIE2 tiles (`Tile(0, 2..5)`).
  - **`Plan.w_bcast`**: When all 4 rows share identical weights (1×1 Conv, LN, Attn), DDR broadcasts one copy across all 4 cores (`4×` weight bandwidth reduction).
- **16 KB Tile Program Memory**:
  - `Plan.kernel_flags(k0, k1)` automatically strips unused branches via `-DNO_K3`, `-DNO_DW`, `-DNO_SE`, `-DNO_UP`, `-DNO_PA`, `-DNO_GELU`, `-DNO_LM`, `-DNO_ATTN`.
- **DMA & Buffer Rules**:
  - Single DMA BD must be `< 16,384 B` (`W_BD_MAX`); `TG_INFLIGHT = 2` (≤ 4 in-flight tasks per Shim DMA channel).
  - Weights + headers + requantization tables fit within 64 KB local tile data memory (`e_w` dynamically sized).

---

## 3. Deployment Modes

1. **Python JIT (`C16Chain`)**: Compiles MLIR-AIE graphs on demand, caches `.xclbin`/`.bin` in `~/.npu/cache/`.
2. **Standalone C++ XRT Daemon**: Export `.xclbin` + `insts.bin` + flat packed weights (see `npu/asr/export_c_bundle.py` and `npu/asr/asr_npu.cpp`) for `< 1s` cold start and zero Python runtime overhead.

---

## 4. Verification

```bash
source npu/env.sh
.venv-iron/bin/python -m npu.check --quick   # 19/19 unit & chain tests (max_diff == 0)
```
