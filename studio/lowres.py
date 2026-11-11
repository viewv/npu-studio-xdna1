"""NPU / GPU 后端共用的 CPU 低分辨率准备 (1/4 分辨率)

  - 遮罩时间平滑 + 软阈值 (与 CPU 版一致)
  - 导向滤波系数 a, b
  - 无光晕背景模糊 (归一化卷积), 或固定的图片/纯色背景

全分辨率的 放大 + 遮罩 (a·I + b) + 背景放大 + 合成 由各后端完成。
"""
from __future__ import annotations

import cv2
import numpy as np

SCALE = 4


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


class LowResPrep:
    def __init__(self, width: int, height: int, mode: str = "blur", blur: int = 61,
                 smooth: float = 0.5, threshold: float = 0.0, bg_image: str | None = None,
                 r: int = 4, eps: float = 2e-3):
        if mode not in ("blur", "image", "color"):
            raise ValueError(f"加速后端不支持模式 {mode}")
        self.mode = mode
        self.w, self.h = width, height
        self.lw, self.lh = width // SCALE, height // SCALE
        self.blur = max(3, blur) | 1
        self.smooth = min(max(smooth, 0.0), 0.95)
        self.threshold = threshold
        self.r, self.eps = r, eps
        self._prev = None
        self.bg_const = None
        if mode == "image":
            img = cv2.imread(bg_image) if bg_image else None
            if img is None:
                raise ValueError(f"cannot read background image: {bg_image}")
            self.bg_const = cv2.cvtColor(cv2.resize(img, (self.lw, self.lh), interpolation=cv2.INTER_AREA),
                                         cv2.COLOR_BGR2BGRA)
        elif mode == "color":
            self.bg_const = np.empty((self.lh, self.lw, 4), np.uint8)
            self.bg_const[:] = (0, 177, 64, 255)

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

    def compute(self, frame: np.ndarray, alpha_small: np.ndarray):
        """返回 (a, b, bg): a/b 为 (lh, lw) float32 导向滤波系数, bg 为 (lh, lw, 4) BGRA"""
        lw, lh = self.lw, self.lh
        alpha = self._alpha(alpha_small)
        small = cv2.resize(frame, (lw, lh), interpolation=cv2.INTER_AREA)
        I = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) * (1.0 / 255.0)
        p = cv2.resize(alpha, (lw, lh), interpolation=cv2.INTER_LINEAR)

        r, eps = self.r, self.eps
        mI, mp = _box(I, r), _box(p, r)
        a = (_box(I * p, r) - mI * mp) / (_box(I * I, r) - mI * mI + eps)
        b = mp - a * mI
        a, b = _box(a, r), _box(b, r)

        if self.bg_const is not None:
            bg = self.bg_const
        else:
            bg = self._blur_bg(small, np.clip(a * I + b, 0.0, 1.0))
        return a, b, bg
