"""第二阶段: CPU 只做低分辨率计算, 全分辨率的 放大/遮罩/合成 交给 NPU 融合内核

CPU (1/4 分辨率): studio/lowres.py 的 LowResPrep, 再定点化 (只写 H/4 行)
NPU (全分辨率, 多列并行): 垂直+水平 ×4 放大 + 遮罩 + 背景放大 + 合成
     (npu/kernels/fused_composite.cc 的 fusedCompositeLineV)

需要在 IRON 环境里运行: source npu/env.sh
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from npu.fused import FusedCompositorV

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
        # 每列处理 H/n_cols 行, 且必须是 4 的倍数 (一个低分辨率行窗口出 4 行): 1080p 用 3 列
        while n_cols > 1 and height % (4 * n_cols):
            n_cols -= 1
        self.n_cols = n_cols
        self.comp = FusedCompositorV(width, height, n_cols)
        self.t_prep = self.t_npu = 0.0
        self._oracle = bool(os.environ.get("NPU_PREP_ORACLE"))
        self._primed = False

    def apply(self, frame: np.ndarray, alpha_small: np.ndarray,
              small: np.ndarray | None = None) -> np.ndarray:
        """返回 (H, W, 4) BGRA 合成结果 (NPU 缓冲区视图, 下一帧会被覆盖)"""
        t0 = time.perf_counter()
        if self._oracle and self._primed:
            # 功耗上限实验 (NPU_PREP_ORACLE=1): 低分辨率准备只做第一帧, 之后复用,
            # 模拟"准备全部搬到 NPU"后 CPU 侧剩下的工作。画面会冻在第一帧的遮罩上。
            cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA, dst=self.comp.px)
            t1 = time.perf_counter()
            out = self.comp.run()
            self.t_prep, self.t_npu = t1 - t0, time.perf_counter() - t1
            return out
        self._primed = True
        a, b, bg = self.prep.compute(frame, alpha_small, small)

        # 定点化: q·255·64 = Aq·(I_u8-128) + Bq。垂直/水平插值都在 NPU 上
        aq = np.clip(np.rint(a * 64.0), -32768, 32767).astype(np.int16)
        bq = np.clip(np.rint((b + a * (128.0 / 255.0)) * (255.0 * 64.0)), -32768, 32767).astype(np.int16)
        self.comp.low.set(aq, bq, bg)
        cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA, dst=self.comp.px)
        t1 = time.perf_counter()

        out = self.comp.run()
        t2 = time.perf_counter()
        self.t_prep, self.t_npu = t1 - t0, t2 - t1
        return out
