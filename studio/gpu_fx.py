"""GPU 后端 (Radeon 780M, OpenCL via OpenCV T-API): 与 NPU 后端相同的分工, 用于对比

CPU (1/4 分辨率): studio/lowres.py 的 LowResPrep
GPU (全分辨率): 灰度 → 双线性放大 a、b → 遮罩 = clamp(a·I + b) → 放大背景 → blendLinear 合成

驱动 (OpenCL 平台) 通过环境变量 OPENCV_OPENCL_DEVICE 选择, 见 select_device()。
"""
from __future__ import annotations

import os
import time

import numpy as np

from .lowres import LowResPrep

DEVICES = {
    "rusticl": "rusticl:GPU:",          # Mesa (radeonsi), 需 RUSTICL_ENABLE=radeonsi
    "rocm": "AMD:GPU:",                 # ROCm OpenCL ("AMD Accelerated Parallel Processing")
}


def select_device(name: str) -> None:
    """必须在 OpenCV 首次使用 OpenCL 之前调用"""
    if name not in DEVICES:
        raise ValueError(f"未知 GPU 驱动 {name} (可选: {', '.join(DEVICES)})")
    os.environ["OPENCV_OPENCL_DEVICE"] = DEVICES[name]
    if name == "rusticl":
        os.environ.setdefault("RUSTICL_ENABLE", "radeonsi")


class GpuBackgroundFx:
    def __init__(self, width: int, height: int, mode: str = "blur", blur: int = 61,
                 smooth: float = 0.5, threshold: float = 0.0, bg_image: str | None = None):
        import cv2
        if not cv2.ocl.haveOpenCL():
            raise RuntimeError("OpenCV 没有找到 OpenCL")
        cv2.ocl.setUseOpenCL(True)
        dev = cv2.ocl.Device.getDefault()
        if dev.type() != cv2.ocl.Device_TYPE_GPU and "gfx" not in dev.name().lower():
            raise RuntimeError(f"OpenCL 默认设备不是 GPU: {dev.name()}")
        self.device_name = f"{dev.name()} ({dev.vendorName()}, {dev.version()})"
        self.cv2 = cv2
        self.prep = LowResPrep(width, height, mode, blur, smooth, threshold, bg_image)
        self.w, self.h = width, height
        self.t_prep = self.t_gpu = 0.0

    def apply(self, frame: np.ndarray, alpha_small: np.ndarray) -> np.ndarray:
        cv2 = self.cv2
        W, H = self.w, self.h
        t0 = time.perf_counter()
        a, b, bg = self.prep.compute(frame, alpha_small)
        bg_bgr = cv2.cvtColor(bg, cv2.COLOR_BGRA2BGR)
        t1 = time.perf_counter()

        f = cv2.UMat(frame)
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        I = cv2.addWeighted(gray, 1.0 / 255.0, gray, 0.0, 0.0, dtype=cv2.CV_32F)
        A = cv2.resize(cv2.UMat(a), (W, H), interpolation=cv2.INTER_LINEAR)
        B = cv2.resize(cv2.UMat(b), (W, H), interpolation=cv2.INTER_LINEAR)
        m = cv2.add(cv2.multiply(A, I), B)
        _, m = cv2.threshold(m, 1.0, 1.0, cv2.THRESH_TRUNC)      # > 1 → 1
        _, m = cv2.threshold(m, 0.0, 0.0, cv2.THRESH_TOZERO)     # < 0 → 0
        inv = cv2.addWeighted(m, -1.0, m, 0.0, 1.0)              # 1 - m
        bgu = cv2.resize(cv2.UMat(bg_bgr), (W, H), interpolation=cv2.INTER_LINEAR)
        out = cv2.blendLinear(f, bgu, m, inv).get()               # 下载结果 (会等待 GPU 完成)
        t2 = time.perf_counter()
        self.t_prep, self.t_gpu = t1 - t0, t2 - t1
        return out
