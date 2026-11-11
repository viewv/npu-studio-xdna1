"""第二阶段: CPU 只做低分辨率计算, 全分辨率的 放大/遮罩/合成 交给 NPU 融合内核

CPU (1/4 分辨率): studio/lowres.py 的 LowResPrep, 再定点化 + 垂直插值到全高度
NPU (全分辨率, 4 列并行): 水平 ×4 放大 + 遮罩 + 背景放大 + 合成  (npu/kernels/fused_composite.cc)

需要在 IRON 环境里运行: source npu/env.sh
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from npu.fused import FusedCompositor

from .lowres import LowResPrep

# NPU 合成内核做过逐字节验证的尺寸。别的尺寸没验证过 (内核和 AIE 编译器都可能有坑,
# 见 README「分辨率」), 所以在这里挡住, 而不是让用户拿到一帧一帧的错像素。
# 确要实验请设 NPU_ALLOW_UNVERIFIED=1。
VERIFIED_SIZES = {(1280, 720), (1920, 1080)}


class NpuBackgroundFx:
    def __init__(self, width: int, height: int, mode: str = "blur", blur: int = 61,
                 smooth: float = 0.5, threshold: float = 0.0, bg_image: str | None = None,
                 n_cols: int = 4):
        import os
        if (width, height) not in VERIFIED_SIZES and not os.environ.get("NPU_ALLOW_UNVERIFIED"):
            ok = ", ".join(f"{w}x{h}" for w, h in sorted(VERIFIED_SIZES))
            raise ValueError(
                f"NPU 合成只在 {ok} 上逐字节验证过；{width}x{height} 没验证过"
                f"（见 README「分辨率」）。换成 --gpu 用核显合成，"
                f"或设 NPU_ALLOW_UNVERIFIED=1 只做实验。")
        self.prep = LowResPrep(width, height, mode, blur, smooth, threshold, bg_image)
        self.h = height
        self.comp = FusedCompositor(width, height, n_cols)
        self.t_prep = self.t_npu = 0.0

    def apply(self, frame: np.ndarray, alpha_small: np.ndarray) -> np.ndarray:
        """返回 (H, W, 4) BGRA 合成结果 (NPU 缓冲区视图, 下一帧会被覆盖)"""
        t0 = time.perf_counter()
        a, b, bg = self.prep.compute(frame, alpha_small)
        lw, H = self.prep.lw, self.h

        # 定点化: q·255·64 = Aq·(I_u8-128) + Bq
        aq = np.clip(np.rint(a * 64.0), -32768, 32767).astype(np.int16)
        bq = np.clip(np.rint((b + a * (128.0 / 255.0)) * (255.0 * 64.0)), -32768, 32767).astype(np.int16)

        # 垂直插值到全高度 (水平方向由 NPU 完成)
        aq_v = cv2.resize(aq, (lw, H), interpolation=cv2.INTER_LINEAR)
        bq_v = cv2.resize(bq, (lw, H), interpolation=cv2.INTER_LINEAR)
        bg_v = cv2.resize(bg, (lw, H), interpolation=cv2.INTER_LINEAR)
        self.comp.low.set(aq_v, bq_v, bg_v)
        cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA, dst=self.comp.px)
        t1 = time.perf_counter()

        out = self.comp.run()
        t2 = time.perf_counter()
        self.t_prep, self.t_npu = t1 - t0, t2 - t1
        return out
