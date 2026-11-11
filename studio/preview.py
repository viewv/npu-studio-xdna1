"""给图形前端的预览共享内存 (/dev/shm), 无锁 seqlock 协议

文件布局 (小端):
  0  u32 magic  = 0x5355504E ("NPUS")
  4  u32 seq    写入前置为奇数, 写完置为偶数; 读者看到奇数或前后不一致就重读
  8  u32 width
 12  u32 height
 16  RGBA 像素 width*height*4
"""
from __future__ import annotations

import mmap
import os
import struct

import cv2
import numpy as np

MAGIC = 0x5355504E
HEADER = 16


class PreviewShm:
    def __init__(self, path: str, width: int = 640, height: int = 360):
        self.w, self.h = width, height
        size = HEADER + width * height * 4
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.ftruncate(fd, size)
            self.mm = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        self.seq = 0
        self.mm[0:HEADER] = struct.pack("<IIII", MAGIC, self.seq, width, height)
        self.px = np.frombuffer(self.mm, dtype=np.uint8, count=width * height * 4,
                                offset=HEADER).reshape(height, width, 4)

    def write(self, frame: np.ndarray) -> None:
        """frame: BGR 或 BGRA 全分辨率结果"""
        small = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        code = cv2.COLOR_BGRA2RGBA if small.shape[2] == 4 else cv2.COLOR_BGR2RGBA
        self.seq += 1                                   # 奇数: 正在写
        self.mm[4:8] = struct.pack("<I", self.seq)
        cv2.cvtColor(small, code, dst=self.px)
        if small.shape[2] == 4:
            self.px[..., 3] = 255
        self.seq += 1                                   # 偶数: 写完
        self.mm[4:8] = struct.pack("<I", self.seq)

    def close(self) -> None:
        self.px = None
        self.mm.close()
