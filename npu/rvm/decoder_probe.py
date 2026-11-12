"""探路: RVM 解码器里的大卷积 (3×3 为主) 用现在的 chain_c16 在 NPU 上要多久 —— 决定解码器值不值得上 NPU

  source npu/env.sh && python -m npu.rvm.decoder_probe [--iters 50]

只是卷积本身 (随机权重, 和 numpy 整数参考逐字节比), 不含上采样 / 拼接 / ConvGRU 的逐元素门控 / 最后的投影。
解码器的卷积按分辨率串成 5 条链 (同一分辨率里前一层的输出接下一层; 通道补到 16 的倍数):
  · 12×20 : Conv_163 (LRASPP 1×1 960→128), Conv_176 (3×3 128→128), Conv_181 (3×3 128→64)
  · 23×40 : Conv_200 171→80 (3×3 IC 分块还没做, 用 128→80 代替, 按 MAC 折算), Conv_206 80→80, Conv_211 80→40
  · 45×80 : Conv_230 107→40, Conv_236 40→40, Conv_241 40→20
  · 90×160: Conv_260 59→32, Conv_266 32→32, Conv_271 32→16
  · 180×320: Conv_290 35→16, Conv_292 16→16
真实解码器的 MAC 见 docs/npu-spikes/rvm_convs_180x320.tsv (1.41 GMAC; 这里覆盖 ~1.36)。
"""
from __future__ import annotations

import sys
import time

import numpy as np

from .. import chain_c16 as c

L = c.Layer


def p16(n):
    return -(-n // 16) * 16


# (名字, [(ic, oc, k)], h, w, 真实 MAC / 这里的 MAC 的折算系数)
GROUPS = [
    ("12×20", [(960, 128, 1), (128, 128, 3), (128, 64, 3)], 12, 20),
    ("23×40", [(128, 80, 3), (80, 80, 3), (80, 40, 3)], 23, 40),
    ("45×80", [(107, 40, 3), (40, 40, 3), (40, 20, 3)], 45, 80),
    ("90×160", [(59, 32, 3), (32, 32, 3), (32, 16, 3)], 90, 160),
    ("180×320", [(35, 16, 3), (16, 16, 3)], 180, 320),
]
REAL_MAC = {"12×20": 81e6, "23×40": 191e6, "45×80": 214e6, "90×160": 442e6, "180×320": 422e6}  # tsv 里对应几层的和


def group_layers(spec, h, w):
    out, prev = [], None
    for ic, oc, k in spec:
        ic = p16(ic) if prev is None else prev
        out.append(L(ic, p16(oc), k, 1, h, w))
        prev = p16(oc)
    return out


def build_all():
    """→ [(名字, C16Chain, 输入, 参考, 层)]"""
    res = []
    for name, spec, h, w in GROUPS:
        layers = group_layers(spec, h, w)
        x, params, y = c.make_case(layers, 0)
        ch = c.C16Chain(layers, params)
        ch.set_input(x)
        res.append((name, ch, x, y, layers))
    return res


def main(iters=50):
    chains = build_all()
    tot_best = tot_mac = tot_real = 0
    for name, ch, x, y, layers in chains:
        ch.call()
        d = c.check(ch.result_full(), y, ch.P.tensors[-1])
        w = []
        for _ in range(iters):
            t0 = time.perf_counter()
            ch.call()
            w.append(time.perf_counter() - t0)
        w.sort()
        mac = sum(l.macs for l in layers)
        print(f"{name:8s} {len(layers)} 层 {ch.P.n_steps:3d} 步 e_w {ch.P.e_w:5d} 深度 {ch.P.w_depth}  最大差 {d}  "
              f"中位 {w[len(w) // 2] * 1e3:6.2f}  最快 {w[0] * 1e3:6.2f} ms  {mac / 1e6:5.0f} MMAC (补齐后)  "
              f"{mac / w[0] / 1e9:5.0f} GMAC/s", flush=True)
        tot_best += w[0] * 1e3
        tot_mac += mac
        tot_real += REAL_MAC[name]
    fixed = 0.25 * (len(chains) - 1)          # 合成一次调用时省掉的固定成本 (每次调用 ~0.25 ms)
    print(f"\n合计 {tot_best:.2f} ms (5 次调用); 合成一次调用约 {tot_best - fixed:.2f} ms; "
          f"{tot_mac / 1e6:.0f} MMAC 补齐后 / 真实 {tot_real / 1e6:.0f} MMAC")


if __name__ == "__main__":
    main(int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 50)
