"""NPU / GPU 后端共用的 1/4 分辨率准备

  - 遮罩时间平滑 + 软阈值 (与 CPU 版一致)
  - 导向滤波系数 a, b
  - 无光晕背景模糊 (归一化卷积), 或固定的图片/纯色背景

两种实现:
  LowResPrep     CPU (numpy/OpenCV)
  GpuLowResPrep  同一套算法全部走 cv2.UMat (OpenCL)

⚠ 实测 (7840HS, 1280x720 -> 320x180, rusticl 与 rocm 两个驱动都试了):
     CPU 版  墙钟 2.8 ms  CPU 2.8 ms
     GPU 版  墙钟 6.2~7.2 ms  CPU 2.7~7.7 ms
   这块太小, OpenCL 的 kernel 启动开销 + 回传把收益吃光了, **CPU 时间也没降**。
   留着是为了以后换更弱的 CPU / 更大的分辨率时再量一次, 默认不用。

全分辨率的 放大 + 遮罩 (a·I + b) + 背景放大 + 合成 由各后端完成。
"""
from __future__ import annotations

import cv2
import numpy as np

SCALE = 4


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def _clip01(x):
    """逐像素 clamp 到 [0,1] (UMat 上用 threshold, 省掉一次 host 往返)"""
    _, x = cv2.threshold(x, 1.0, 1.0, cv2.THRESH_TRUNC)
    _, x = cv2.threshold(x, 0.0, 0.0, cv2.THRESH_TOZERO)
    return x


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
        self._erode_k = np.ones((3, 3), np.uint8)
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
        """低分辨率无光晕模糊; 返回 (lh, lw, 4) BGRA

        归一化卷积: out = blur(f·w) / blur(w), 背景像素少的地方逐渐退回普通模糊。
        模糊够强时 (k >= 9) 在再降一半的分辨率上做, 然后放大回来: CPU 从 ~1.3–2.3 ms
        降到 ~0.5 ms, 与全分辨率版在背景区域平均差 0.26/255 (95% 的像素 ≤1), 看不出来。
        """
        k = max(3, (self.blur // SCALE) | 1)
        if k >= 9:
            h2, w2 = self.lh // 2, self.lw // 2
            src = cv2.resize(small, (w2, h2), interpolation=cv2.INTER_AREA)
            ms = cv2.resize(ms, (w2, h2), interpolation=cv2.INTER_AREA)
            k = max(3, (k // 2) | 1)
        else:
            src = small
        f = src.astype(np.float32)
        wgt = np.clip(1.0 - ms * 1.5, 0.0, 1.0)
        wgt = cv2.erode(wgt, self._erode_k)
        num = cv2.GaussianBlur(f * wgt[..., None], (k, k), 0)
        den = cv2.GaussianBlur(wgt, (k, k), 0)[..., None]
        plain = cv2.GaussianBlur(f, (k, k), 0)
        t = np.clip(den * 8.0, 0.0, 1.0)
        out = t * (num / np.maximum(den, 1e-4)) + (1.0 - t) * plain
        if src is not small:
            out = cv2.resize(out, (self.lw, self.lh), interpolation=cv2.INTER_LINEAR)
        bgra = np.empty((self.lh, self.lw, 4), np.uint8)
        bgra[..., :3] = np.clip(out, 0, 255)
        bgra[..., 3] = 255
        return bgra

    def compute(self, frame: np.ndarray, alpha_small: np.ndarray, small: np.ndarray | None = None):
        """返回 (a, b, bg): a/b 为 (lh, lw) float32 导向滤波系数, bg 为 (lh, lw, 4) BGRA

        small: 可选, 已经缩好的 (lh, lw, 3) 原图 (RVM 为了喂模型已经算过同一张 INTER_AREA 缩放)
        """
        lw, lh = self.lw, self.lh
        alpha = self._alpha(alpha_small)
        if small is None or small.shape[:2] != (lh, lw):
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


class GpuLowResPrep(LowResPrep):
    """同一套算法, 全部在 GPU 上 (cv2.UMat / OpenCL)。

    CPU 只负责 180x320 的遮罩平滑 (太小, 搬来搬去不值) 和最后把 a/b/bg 取回来
    (约 0.5 MB)。输出与 LowResPrep 的差在 1e-6 量级, 定点化之后基本看不到。
    """

    def compute(self, frame: np.ndarray, alpha_small: np.ndarray):
        lw, lh = self.lw, self.lh
        r, eps = self.r, self.eps
        alpha = self._alpha(alpha_small)                      # CPU, 很小
        small = cv2.resize(cv2.UMat(frame), (lw, lh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        I = cv2.multiply(gray, 1.0 / 255.0, dtype=cv2.CV_32F)
        p = cv2.resize(cv2.UMat(alpha.astype(np.float32)), (lw, lh), interpolation=cv2.INTER_LINEAR)

        k = 2 * r + 1

        def box(x):
            return cv2.boxFilter(x, -1, (k, k), borderType=cv2.BORDER_REFLECT)

        mI, mp = box(I), box(p)
        mIp = box(cv2.multiply(I, p, dtype=cv2.CV_32F))
        mII = box(cv2.multiply(I, I, dtype=cv2.CV_32F))
        num = cv2.subtract(mIp, cv2.multiply(mI, mp, dtype=cv2.CV_32F), dtype=cv2.CV_32F)
        den = cv2.add(cv2.subtract(mII, cv2.multiply(mI, mI, dtype=cv2.CV_32F), dtype=cv2.CV_32F),
                      eps)
        a = cv2.divide(num, den, dtype=cv2.CV_32F)
        b = cv2.subtract(mp, cv2.multiply(a, mI, dtype=cv2.CV_32F), dtype=cv2.CV_32F)
        a, b = box(a), box(b)

        if self.bg_const is not None:
            bg = self.bg_const
        else:
            kb = max(3, (self.blur // SCALE) | 1)
            m = _clip01(cv2.add(cv2.multiply(a, I, dtype=cv2.CV_32F), b, dtype=cv2.CV_32F))
            wgt = _clip01(cv2.subtract(1.0, cv2.multiply(m, 1.5, dtype=cv2.CV_32F), dtype=cv2.CV_32F))
            wgt = cv2.erode(wgt, np.ones((3, 3), np.uint8))
            w3 = cv2.cvtColor(wgt, cv2.COLOR_GRAY2BGR)
            num_b = cv2.GaussianBlur(cv2.multiply(small, w3, dtype=cv2.CV_32F), (kb, kb), 0)
            den_b = cv2.GaussianBlur(wgt, (kb, kb), 0)
            plain = cv2.GaussianBlur(small, (kb, kb), 0)
            den_g = cv2.add(den_b, 1e-4)
            t3 = cv2.cvtColor(_clip01(cv2.multiply(den_g, 8.0, dtype=cv2.CV_32F)), cv2.COLOR_GRAY2BGR)
            den_b = cv2.cvtColor(den_g, cv2.COLOR_GRAY2BGR)
            one = cv2.subtract(1.0, t3, dtype=cv2.CV_32F)
            out = cv2.add(cv2.multiply(t3, cv2.divide(num_b, den_b, dtype=cv2.CV_32F), dtype=cv2.CV_32F),
                          cv2.multiply(one, plain, dtype=cv2.CV_32F), dtype=cv2.CV_32F)
            bgra = np.empty((lh, lw, 4), np.uint8)
            bgra[..., :3] = np.clip(out.get(), 0, 255)
            bgra[..., 3] = 255
            bg = bgra

        return (a.get() if isinstance(a, cv2.UMat) else a,
                b.get() if isinstance(b, cv2.UMat) else b,
                bg)
