"""人像分割 + 背景处理 (第一阶段: CPU)

Segmenter    : MediaPipe selfie segmentation (ONNX, 144x256 横屏版), 输出 0..1 人物遮罩
BackgroundFx : 遮罩平滑(时间域EMA + 羽化) 与 背景虚化/替换 合成

后续阶段会把 BackgroundFx 的模糊与合成换成 NPU 实现, 接口保持不变。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODELS = Path(__file__).resolve().parent.parent / "models"


class Segmenter:
    def __init__(self, model: str = "mediapipe_selfie_segmentation_landscape", threads: int = 2):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        # 默认工作线程会空转等待下一次推理, 白占 CPU; 每帧只推理一次, 关掉空转
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(str(MODELS / f"{model}.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        _, _, self.h, self.w = inp.shape
        self.output_name = self.sess.get_outputs()[0].name

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        """返回模型分辨率 (h, w) 的 float32 遮罩, 1=人物"""
        small = cv2.resize(frame_bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) * (1.0 / 255.0)).transpose(2, 0, 1)[None]
        alpha = self.sess.run([self.output_name], {self.input_name: x})[0]
        return alpha[0, 0]


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def fast_guided_filter(guide_full: np.ndarray, p_low: np.ndarray, scale: int,
                       r: int = 4, eps: float = 1e-3) -> np.ndarray:
    """Fast Guided Filter (He & Sun 2015): 在 1/scale 分辨率算线性系数, 再放大到全分辨率应用。
    让低分辨率遮罩的边缘贴合原图里的真实边缘 (头发、肩膀轮廓)。
    guide_full: (H,W) float32 0..1 灰度; p_low: 任意分辨率的遮罩"""
    H, W = guide_full.shape
    w, h = W // scale, H // scale
    I = cv2.resize(guide_full, (w, h), interpolation=cv2.INTER_AREA)
    p = cv2.resize(p_low, (w, h), interpolation=cv2.INTER_LINEAR)
    mI, mp = _box(I, r), _box(p, r)
    cov = _box(I * p, r) - mI * mp
    var = _box(I * I, r) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    A = cv2.resize(_box(a, r), (W, H), interpolation=cv2.INTER_LINEAR)
    B = cv2.resize(_box(b, r), (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(A * guide_full + B, 0.0, 1.0)


class BackgroundFx:
    """quality="hq": 导向滤波贴边 + 无光晕背景模糊 (默认)
       quality="basic": 第一版的简单做法 (直接放大遮罩 + 整帧模糊), 方便对比"""

    def __init__(self, mode: str = "blur", blur: int = 61, smooth: float = 0.5,
                 feather: int = 5, threshold: float = 0.0, bg_image: str | None = None,
                 quality: str = "hq"):
        if mode not in ("blur", "image", "color", "mask"):
            raise ValueError(f"unknown mode {mode}")
        if quality not in ("hq", "basic"):
            raise ValueError(f"unknown quality {quality}")
        self.mode = mode
        self.quality = quality
        self.blur = max(3, blur) | 1          # 等效于全分辨率下的高斯核大小, 需为奇数
        self.smooth = min(max(smooth, 0.0), 0.95)
        self.feather = max(1, feather) | 1
        self.threshold = threshold
        self.bg_src = cv2.imread(bg_image) if bg_image else None
        if mode == "image" and self.bg_src is None:
            raise ValueError(f"cannot read background image: {bg_image}")
        self._prev = None
        self._bg_cache = None

    def refine(self, alpha: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """时间平滑 + 软阈值 + 放大到输出分辨率; 返回 (H, W) float32 0..1"""
        if self._prev is not None and self._prev.shape == alpha.shape and self.smooth > 0:
            alpha = self.smooth * self._prev + (1.0 - self.smooth) * alpha
        self._prev = alpha
        if self.threshold > 0:
            # 软阈值: 把 [t, 1-t] 拉伸到 [0, 1], 边缘更干净
            t = self.threshold
            alpha = np.clip((alpha - t) / max(1e-3, 1.0 - 2 * t), 0.0, 1.0)
        h, w = frame.shape[:2]
        if self.quality == "hq":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) * (1.0 / 255.0)
            m = fast_guided_filter(gray, alpha, scale=4, r=4, eps=2e-3)
            return cv2.GaussianBlur(m, (3, 3), 0) if self.feather > 1 else m
        m = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(m, (self.feather, self.feather), 0)

    def _blur_bg(self, frame: np.ndarray, m: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = 4 if self.blur <= 63 else 8
        sw, sh = w // scale, h // scale
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        k = max(3, (self.blur // scale) | 1)
        if self.quality == "basic":
            out = cv2.GaussianBlur(small, (k, k), 0)
        else:
            # 归一化卷积: 只用背景像素参与模糊, 避免人物颜色被抹到背景里形成"光晕"
            ms = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA)
            wgt = np.clip(1.0 - ms * 1.5, 0.0, 1.0)            # 人物边缘也排除在外
            wgt = cv2.erode(wgt, np.ones((3, 3), np.uint8))
            num = cv2.GaussianBlur(small * wgt[..., None], (k, k), 0)
            den = cv2.GaussianBlur(wgt, (k, k), 0)[..., None]
            plain = cv2.GaussianBlur(small, (k, k), 0)
            # 背景像素太少的地方(被人挡住的区域)退回普通模糊
            t = np.clip(den * 8.0, 0.0, 1.0)
            out = t * (num / np.maximum(den, 1e-4)) + (1.0 - t) * plain
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

    def background(self, frame: np.ndarray, m: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.mode == "blur":
            return self._blur_bg(frame, m)
        if self._bg_cache is None or self._bg_cache.shape != frame.shape:
            if self.mode == "image":
                self._bg_cache = cv2.resize(self.bg_src, (w, h), interpolation=cv2.INTER_AREA)
            else:  # color / mask
                self._bg_cache = np.full_like(frame, (0, 177, 64) if self.mode == "color" else 0)
        return self._bg_cache

    def apply(self, frame: np.ndarray, alpha_small: np.ndarray) -> np.ndarray:
        m = self.refine(alpha_small, frame)
        if self.mode == "mask":
            return cv2.cvtColor((m * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        bg = self.background(frame, m)
        # 人物 = frame×m + 背景×(1-m); blendLinear 比 numpy 浮点运算快 ~10 倍
        return cv2.blendLinear(frame, bg, m, 1.0 - m)
