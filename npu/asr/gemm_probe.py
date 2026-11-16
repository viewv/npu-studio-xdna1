"""ASR transformer 的 GEMM 在 chain_c16 上的形状探针。

序列 GEMM（ic=512/2048, oc=512/2048, T 个 token）必然走 IC 分块（一行输入放不进
7680 B 的元素），ict 又要求 rows·nocp·nxc ≤ ACC_CHUNKS(3)，所以 (h, w) 的选择
决定步数。先在主机侧用 choose_tiling 枚举布局，再上硬件测最好的那个。

    python -m npu.asr.gemm_probe            # 只枚举布局（不需要 NPU）
    python -m npu.asr.gemm_probe --run 175  # T=175 上硬件：正确性 + 计时
"""
from __future__ import annotations

import sys
import time

import numpy as np

from npu.chain_c16 import C16Chain, Layer, check, make_case

# 一个 transformer 块的三个 GEMM: (ic, oc)  —— 探针默认（75% 的块内 GEMM 算力）
GEMMS = [(512, 512), (512, 2048), (2048, 512)]
# 真实形状：q/k/v/output_transform 四个 512×512 + filter/output 两个 (2048,512)/(512,2048)
GEMMS_REAL = [(512, 512), (512, 512), (512, 512), (512, 512), (512, 2048), (2048, 512)]


def macs_per_block(T: int, gems=None) -> int:
    return sum(T * ic * oc for ic, oc in (gems or GEMMS))


def probe_layouts(T: int):
    """对若干 (h, w) 组合打印每种 GEMM 的 tiling（步数、模式）。"""
    from npu.chain_c16 import choose_tiling

    layouts = []
    for h in (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 16, 20, 22, 25, 44):
        if T % h or T // h > 400:
            continue
        w = T // h
        layouts.append((h, w))
    print(f"{'布局':>10s} {'GEMM':>12s} {'模式':>6s} {'步数':>6s}  说明")
    best = {}
    for h, w in layouts:
        total = 0
        desc = []
        for ic, oc in GEMMS:
            l = Layer(ic, oc, 1, 1, h, w)
            try:
                tl = choose_tiling(l)
                n = len(tl.steps)
                total += n
                desc.append(f"{tl.mode}/{n}步")
                best.setdefault((h, w), []).append((tl.mode, n))
            except ValueError as e:
                desc.append(f"✗ {str(e)[:28]}")
                total = 10 ** 9
                best.setdefault((h, w), []).append(None)
        print(f"{f'{h}×{w}':>10s} {'':>12s} {'':>6s} {total:>6d}  {' '.join(desc)}")
    return best


def build_chain(T: int, blocks: int, h: int, gems=None):
    """blocks 个 transformer 块的 GEMM 链（不含注意力/layernorm），序列铺成 h×w。"""
    assert T % h == 0
    w = T // h
    gems = gems or GEMMS
    layers = []
    for _ in range(blocks):
        for ic, oc in gems:
            layers.append(Layer(ic, oc, 1, 1, h, w))
    return layers


def run(T: int, h: int, blocks: int, iters: int = 30, real: bool = False):
    gems = GEMMS_REAL if real else GEMMS
    layers = build_chain(T, blocks, h, gems)
    print(f"[probe] T={T} 铺成 {h}×{T // h}, {blocks} 块 × 3 GEMM = {len(layers)} 层, "
          f"{macs_per_block(T, gems) * blocks / 1e9:.3f} GMAC")
    t0 = time.perf_counter()
    x, params, y = make_case(layers, seed=1)
    ch = C16Chain(layers, params)
    ch.set_input(x)
    ch.call()
    jit = time.perf_counter() - t0
    d = check(ch.result_full(), y, ch.P.tensors[-1])
    print(f"[probe] JIT {jit:.1f}s  最大差 {d}  步数 {ch.P.n_steps}")
    walls = []
    for _ in range(iters):
        t = time.perf_counter()
        ch.call()
        walls.append(time.perf_counter() - t)
    walls.sort()
    med, best_t = walls[len(walls) // 2], walls[0]
    chunk_ms = med * 1e3
    print(f"[probe] 一次 {len(layers)} 层: 中位 {chunk_ms:.2f} ms 最快 {best_t * 1e3:.2f} ms")
    # 换算成 40 块（整个声学模型）和一个流式窗口
    per40 = chunk_ms / blocks * 40
    gmac = macs_per_block(T, gems) * blocks / 1e9
    print(f"[probe] 吞吐 {gmac / med:.1f} GMAC/s；40 块全模型 ≈ {per40:.1f} ms"
          f"（每次调用推进 4.0 s 新音频 → {4.0 / (per40 / 1e3):.0f}× 实时）")
    return d


if __name__ == "__main__":
    if "--run" in sys.argv:
        T = int(sys.argv[sys.argv.index("--run") + 1])
        h = int(sys.argv[sys.argv.index("-h") + 1]) if "-h" in sys.argv else 1
        blocks = int(sys.argv[sys.argv.index("-b") + 1]) if "-b" in sys.argv else 4
        sys.exit(0 if run(T, h, blocks, real="--real" in sys.argv) == 0 else 1)
    import os
    for t in [int(x) for x in os.environ.get('TS', '120 176 180').split()]:
        probe_layouts(t)
