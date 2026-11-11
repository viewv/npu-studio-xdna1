"""RVM (Robust Video Matting) 作为抠像后端

和 MediaPipe 分割的区别:
  - 头发丝级边缘, 自带时序记忆 (r1..r4 循环状态), 不会闪也不会拖影
  - 不需要再做时间平滑 (EMA), 那反而会拖影

关键设计: **把摄像头帧直接缩到 180x320 再喂给模型**。
  RVM 的输出分辨率等于输入分辨率, 所以 pha 天然就是 180x320 —— 正好是下游
  (LowResPrep + NPU 合成) 需要的 1/4 分辨率遮罩:
    - 不用改 ONNX 图 (不删尾部 Resize, 它在这里是 1:1 的 no-op)
    - 不用把全分辨率 fgr/pha 传回来 (720p 每帧 15-33 MB)
    - 不用 fgr: 合成仍然用原始画面作前景, 只借 RVM 的 alpha
  代价是放弃 RVM 的 fgr (它主要修边缘颜色), 后续想要可以再加。

  from studio.rvm import RvmMatting
  m = RvmMatting()                 # 工作分辨率 320x180
  alpha = m(frame_bgr)             # (180, 320) float32, 1=人物
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODELS = Path(__file__).resolve().parent.parent / "models"


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class RvmMatting:
    """RVM 抠像: __call__(frame_bgr) -> (h, w) float32 遮罩

    model:      onnx 文件 (fp32 或 fp16 都行, ORT 会在内部转换)
    width/height: 工作分辨率 (默认 320x180, 对应 720p 输出的 1/4)
    threads:    ORT 线程数; CPU 上这个模型相当重, 建议 2-4
    """

    def __init__(self, width: int = 320, height: int = 180,
                 model: str = "rvm_mobilenetv3_fp32.onnx", threads: int = 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(str(MODELS / model), so,
                                         providers=["CPUExecutionProvider"])
        self.w, self.h = width, height
        self.inputs = [i.name for i in self.sess.get_inputs()]
        self.outputs = [o.name for o in self.sess.get_outputs()]
        # 循环状态: r1..r4 的空间尺寸是工作分辨率的 1/2, 1/4, 1/8, 1/16 (向上取整)
        self._state_ch = (16, 20, 40, 64)
        self.reset()
        # downsample_ratio=1: 我们自己已经把画面缩到工作分辨率了
        self._ratio = np.array([1.0], np.float32)
        self._run(np.zeros((self.h, self.w, 3), np.uint8))    # 预热 (JIT/分配)
        self.reset()          # 预热跑出来的状态丢掉, 从零开始

    def reset(self) -> None:
        self._state = [np.zeros((1, c, _ceil_div(self.h, 2 ** (i + 1)),
                                 _ceil_div(self.w, 2 ** (i + 1))), np.float32)
                       for i, c in enumerate(self._state_ch)]

    def _run(self, small_bgr: np.ndarray):
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) * (1.0 / 255.0)).transpose(2, 0, 1)[None]
        feed = {"src": x, "r1i": self._state[0], "r2i": self._state[1],
                "r3i": self._state[2], "r4i": self._state[3],
                "downsample_ratio": self._ratio}
        out = self.sess.run(self.outputs, {k: v for k, v in feed.items() if k in self.inputs})
        d = dict(zip(self.outputs, out))
        self._state = [d[f"r{i}o"] for i in (1, 2, 3, 4)]
        return d["pha"][0, 0]

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        """frame_bgr: 任意分辨率的 BGR 帧 -> (h, w) float32 遮罩 (1=人物)"""
        if frame_bgr.shape[1] != self.w or frame_bgr.shape[0] != self.h:
            frame_bgr = cv2.resize(frame_bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return self._run(frame_bgr)


def preview(frame_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """把遮罩叠成一张灰度图, 方便看边缘 (alpha 放大到原图尺寸)"""
    h, w = frame_bgr.shape[:2]
    a = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return (g.astype(np.float32) * 0.35 + a * 255.0 * 0.65).astype(np.uint8)
