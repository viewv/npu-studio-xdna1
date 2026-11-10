"""第二阶段: CPU 只做低分辨率计算, 全分辨率的 放大/遮罩/合成 交给 NPU 融合内核

CPU (1/4 分辨率, 320×180):
  - 遮罩时间平滑 + 软阈值 (与 CPU 版一致)
  - 导向滤波系数 a, b
  - 无光晕背景模糊 (归一化卷积)
  - 垂直方向插值到全高度, 打包成 Aq / Bq / bg
NPU (全分辨率, 1280×720, 4 列并行):
  - 水平 ×4 放大 + 遮罩 = a·I + b + 背景放大 + alpha 合成  (npu/kernels/fused_composite.cc)

需要在 IRON 环境里运行: source npu/env.sh
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from npu.fused import FusedCompositor

SCALE = 4


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


class NpuBackgroundFx:
    def __init__(self, width: int, height: int, mode: str = "blur", blur: int = 61,
                 smooth: float = 0.5, threshold: float = 0.0, bg_image: str | None = None,
                 n_cols: int = 4, r: int = 4, eps: float = 2e-3):
        if mode not in ("blur", "image", "color"):
            raise ValueError(f"NPU 路径不支持模式 {mode}")
        self.mode = mode
        self.w, self.h = width, height
        self.lw, self.lh = width // SCALE, height // SCALE
        self.blur = max(3, blur) | 1
        self.smooth = min(max(smooth, 0.0), 0.95)
        self.threshold = threshold
        self.r, self.eps = r, eps
        self.comp = FusedCompositor(width, height, n_cols)
        self._prev = None
        self._bg_const = None
        if mode == "image":
            img = cv2.imread(bg_image) if bg_image else None
            if img is None:
                raise ValueError(f"cannot read background image: {bg_image}")
            self._bg_const = cv2.cvtColor(cv2.resize(img, (self.lw, self.lh), interpolation=cv2.INTER_AREA),
                                          cv2.COLOR_BGR2BGRA)
        elif mode == "color":
            self._bg_const = np.empty((self.lh, self.lw, 4), np.uint8)
            self._bg_const[:] = (0, 177, 64, 255)
        self.t_prep = self.t_npu = 0.0

    def _alpha(self, alpha: np.ndarray) -> np.ndarray:
        if self._prev is not None and self._prev.shape == alpha.shape and self.smooth > 0:
            alpha = self.smooth * self._prev + (1.0 - self.smooth) * alpha
        self._prev = alpha
        if self.threshold > 0:
            t = self.threshold
            alpha = np.clip((alpha - t) / max(1e-3, 1.0 - 2 * t), 0.0, 1.0)
        return alpha

    def _blur_bg(self, small: np.ndarray, ms: np.ndarray) -> np.ndarray:
        """低分辨率无光晕模糊; 返回 (lh, lw, 4) uint8 BGRA"""
        k = max(3, (self.blur // SCALE) | 1)
        f = small.astype(np.float32)
        wgt = np.clip(1.0 - ms * 1.5, 0.0, 1.0)
        wgt = cv2.erode(wgt, np.ones((3, 3), np.uint8))
        num = cv2.GaussianBlur(f * wgt[..., None], (k, k), 0)
        den = cv2.GaussianBlur(wgt, (k, k), 0)[..., None]
        plain = cv2.GaussianBlur(f, (k, k), 0)
        t = np.clip(den * 8.0, 0.0, 1.0)
        out = t * (num / np.maximum(den, 1e-4)) + (1.0 - t) * plain
        bgra = np.empty((self.lh, self.lw, 4), np.uint8)
        bgra[..., :3] = np.clip(out, 0, 255)
        bgra[..., 3] = 255
        return bgra

    def apply(self, frame: np.ndarray, alpha_small: np.ndarray) -> np.ndarray:
        """返回 (H, W, 4) BGRA 合成结果 (NPU 缓冲区视图, 下一帧会被覆盖)"""
        t0 = time.perf_counter()
        lw, lh, H = self.lw, self.lh, self.h
        alpha = self._alpha(alpha_small)

        small = cv2.resize(frame, (lw, lh), interpolation=cv2.INTER_AREA)
        I = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) * (1.0 / 255.0)
        p = cv2.resize(alpha, (lw, lh), interpolation=cv2.INTER_LINEAR)

        # 导向滤波系数 (低分辨率)
        r, eps = self.r, self.eps
        mI, mp = _box(I, r), _box(p, r)
        a = (_box(I * p, r) - mI * mp) / (_box(I * I, r) - mI * mI + eps)
        b = mp - a * mI
        a, b = _box(a, r), _box(b, r)
        q_low = np.clip(a * I + b, 0.0, 1.0)

        bg = self._bg_const if self._bg_const is not None else self._blur_bg(small, q_low)

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
