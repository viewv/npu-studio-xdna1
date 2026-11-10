"""v4l2loopback 虚拟摄像头输出 (纯 ctypes ioctl, 不依赖 ffmpeg/pyfakewebcam)

格式用 I420 (YU12): OpenCV 能一步从 BGR 转换, Chrome/Firefox/OBS/Zoom 都认。
v4l2loopback 需要以 exclusive_caps=1 加载, 浏览器才会把它当成摄像头。
"""
from __future__ import annotations

import ctypes
import fcntl
import os

import cv2
import numpy as np

V4L2_BUF_TYPE_VIDEO_OUTPUT = 2
V4L2_FIELD_NONE = 1
V4L2_COLORSPACE_SRGB = 8


def fourcc(s: str) -> int:
    return ord(s[0]) | ord(s[1]) << 8 | ord(s[2]) << 16 | ord(s[3]) << 24


class v4l2_pix_format(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32), ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32), ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32), ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32), ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32), ("xfer_func", ctypes.c_uint32),
    ]


class _fmt_union(ctypes.Union):
    # 内核里这个union含指针成员, 因此按8字节对齐、总长200字节
    _fields_ = [("pix", v4l2_pix_format), ("raw_data", ctypes.c_uint8 * 200),
                ("_align", ctypes.c_void_p)]


class v4l2_format(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("fmt", _fmt_union)]


def _iowr(nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (ord("V") << 8) | nr


VIDIOC_S_FMT = _iowr(5, ctypes.sizeof(v4l2_format))
assert ctypes.sizeof(v4l2_format) == 208, ctypes.sizeof(v4l2_format)


class VirtualCamera:
    def __init__(self, device: str, width: int, height: int):
        if width % 2 or height % 2:
            raise ValueError("I420 需要偶数宽高")
        self.width, self.height = width, height
        self.fd = os.open(device, os.O_WRONLY)
        fmt = v4l2_format()
        fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT
        p = fmt.fmt.pix
        p.width, p.height = width, height
        p.pixelformat = fourcc("YU12")
        p.field = V4L2_FIELD_NONE
        p.bytesperline = width
        p.sizeimage = width * height * 3 // 2
        p.colorspace = V4L2_COLORSPACE_SRGB
        fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt)
        self.device = device

    def send(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        os.write(self.fd, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420).tobytes())

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def find_loopback() -> str | None:
    """找到第一个 v4l2loopback 设备 (/sys/devices/virtual/video4linux 下的)"""
    base = "/sys/devices/virtual/video4linux"
    try:
        for name in sorted(os.listdir(base), key=lambda s: int(s[5:]) if s[5:].isdigit() else 0):
            return f"/dev/{name}"
    except OSError:
        return None
    return None
