# AGENTS.md — npu-studio-effects

Experimental Ryzen AI (Phoenix / Ryzen 7 7840HS, XDNA1 AIE2, ~10 TOPS) NPU playground & desktop toolbox for Linux.

## Repository Map

- `npu/chain_c16.py`: 4-core / 16-core MLIR-AIE (`IRON`) chain compiler & packer (`Conv1x1`, `DWConv`, `Upsample2x`, `SE`, `LayerNorm`, `Attn`, `GELU`, `Residual`).
- `npu/kernels/conv_c16.cc`: AIE2 C++ tile kernels (`aie::mmul<4,8,8>`, `parallel_lookup`, fixed-point `ln_step`, `attn_step`).
- `npu/check.py`: Full kernel/chain self-test suite (`20/20`, bit-exact `max_diff == 0` against numpy integer reference).
- `npu/rvm/`: RobustVideoMatting (RVM) MobileNetV3 int8 NPU encoder (`encoder_npu.py`).
- `npu/ocr/`: PP-OCRv4 Mobile heterogeneous Chinese+English OCR (`det_npu.py`, `rec_npu.py`, `calibrate.py`, persistent `worker.py` with 1:1 wide-image tiling).
- `npu/asr/`:
  - `encoder_npu.py`: 4-core 40-layer Transformer ASR (`2/3/5/7s` buckets, `132/146/204/290 ms` NPU-only, `from_cache`).
  - `encoder_16c.py`: Experimental 16-core (`4×4`) executor (`~216 ms` NPU-only).
  - `export_c_bundle.py`: Exports `w{10,15,25,35}` `.xclbin`, `insts.bin`, `bucket.bin`, `common.bin`, `vocab.txt` to `~/.npu/asr/asr_4c_bundle/`.
  - `asr_npu.cpp`: Standalone C++ XRT + OpenBLAS daemon (`136.7 / 159.5 / 219.4 / 320.0 ms` end-to-end, `~0.7s` cold start, `PCM <n>` streaming protocol).
  - `shell_asr.py`: Interactive microphone CLI (`--cpp` and `--cpu-daemon`).
- `studio/`: Python virtual camera + RVM/MediaPipe matting + `v4l2loopback` output.
- `app/`: Rust + Slint desktop Toolbox GUI (`NPU Studio`), supporting English/Chinese toggle, Video Blur, Live ASR, and Screen Snip OCR.
- `docs/`: Technical notes & benchmarks (`docs/asr-npu.md`, `docs/npu-spikes/`).

## Developer & Agent Rules

1. **Before editing `npu/`**: read `.claude/skills/npu-dev/SKILL.md` (hardware constraints, 16 KB program memory / 64 KB data memory limits, DMA BD limits, debugging timeouts).
2. **Always run the test suite after kernel/chain changes**:
   ```bash
   source npu/env.sh
   .venv-iron/bin/python -m npu.check --quick
   ```
   NPU bugs typically manifest as silent numerical corruption rather than crashes — verify `max_diff == 0` against the numpy integer reference.
3. **Clear JIT cache** (`rm -rf ~/.npu/cache/*`) if you modify `conv_c16.cc` headers or Python constants that aren't part of the JIT cache key. Exported ASR bundles live in `~/.npu/asr/` so `npu.check` won't wipe them.
4. **Desktop GUI**: Build & install with `./app/install.sh` (requires `cargo` + `slint`).
