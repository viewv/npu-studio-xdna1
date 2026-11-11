"""RVM 跑在核显上 (ncnn + Vulkan), 通过一个常驻 worker 进程和 Python 通信

为什么是子进程而不是 Python 绑定: ncnn 没有可用的 Python 包, 但这个模型只需要
"喂一帧图、取回一张遮罩", 进程边界反而干净——Python 侧完全不碰模型状态。

构建 / 转换 (只需一次):

    # 1) 转换模型 (pnnx, 纯 pip, 不需要编译 ncnn)
    pnnx models/rvm_mobilenetv3_fp32.onnx

    # 2) 编译 worker (用 ncnn 的 Vulkan 构建; Arch 的包里就带, 不需要 sudo)
    bash studio/native/build_rvm_worker.sh

    # 3) 跑
    python -m studio --matting rvm --rvm-gpu --npu

  from studio.rvm_gpu import RvmGpuMatting
  m = RvmGpuMatting()
  alpha = m(frame_bgr)          # (180, 320) float32
"""
from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORKER = ROOT / "third_party" / "rvm_worker"
PARAM_CANDIDATES = ("rvm_mobilenetv3_fp32.ncnn.param", "rvm.ncnn.param")
BIN_CANDIDATES = ("rvm_mobilenetv3_fp32.ncnn.bin", "rvm.ncnn.bin")


def _find(names, where: Path) -> Path | None:
    for n in names:
        p = where / n
        if p.exists():
            return p
    return None


class RvmGpuMatting:
    """ncnn+Vulkan 版 RVM。接口与 studio.rvm.RvmMatting 完全一致。"""

    def __init__(self, width: int = 320, height: int = 180, model_dir: Path | None = None):
        model_dir = Path(model_dir) if model_dir else ROOT / "models"
        if not WORKER.exists():
            raise FileNotFoundError(
                f"找不到 {WORKER}\n先跑: bash studio/native/build_rvm_worker.sh")
        param = _find(PARAM_CANDIDATES, model_dir)
        binf = _find(BIN_CANDIDATES, model_dir)
        if param is None or binf is None:
            raise FileNotFoundError(
                f"{model_dir} 里没有 ncnn 模型 (需要 {PARAM_CANDIDATES[0]} / {BIN_CANDIDATES[0]})\n"
                f"先跑: pnnx models/rvm_mobilenetv3_fp32.onnx")
        self.w, self.h = width, height
        self.proc = subprocess.Popen([str(WORKER), str(param), str(binf)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     bufsize=0)
        self._run(np.zeros((height, width, 3), np.uint8))     # 预热 (分配/首帧 shader)

    def reset(self) -> None:
        """worker 里状态常驻; 需要重置就重启进程"""
        raise NotImplementedError

    def _run(self, small_bgr: np.ndarray) -> np.ndarray:
        assert self.proc.stdin and self.proc.stdout
        h, w = small_bgr.shape[:2]
        self.proc.stdin.write(struct.pack("<II", w, h))
        self.proc.stdin.write(np.ascontiguousarray(small_bgr).tobytes())
        self.proc.stdin.flush()
        buf = self.proc.stdout.read(w * h * 4)
        if len(buf) != w * h * 4:
            raise RuntimeError("RVM worker 挂了 (检查 third_party/rvm_worker 的 stderr)")
        return np.frombuffer(buf, np.float32).reshape(h, w).copy()

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.shape[1] != self.w or frame_bgr.shape[0] != self.h:
            frame_bgr = cv2.resize(frame_bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return self._run(frame_bgr)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
