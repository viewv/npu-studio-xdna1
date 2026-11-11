"""RVM 跑在核显上 (ncnn + Vulkan), 通过一个常驻 worker 进程和 Python 通信

为什么是子进程: ncnn 没有可用的 Python 包, 而这里只需要"喂一帧图、取回一张遮罩",
进程边界反而干净 —— Python 侧完全不碰模型状态, 也不占 CPU。

准备 (只需一次):

    # 1) 模型: ONNX -> 定形 -> ncnn (见 studio/native/convert_rvm.py)
    .venv/bin/python studio/native/convert_rvm.py

    # 2) worker (用 Arch 仓库里带 Vulkan 的 ncnn, 不需要 sudo)
    bash studio/native/build_rvm_worker.sh

实测 (7840HS, 720p, 摄像头 30fps, 与 NPU 合成同时运行): RVM 约 19 ms/帧 (单独跑约 12 ms);
整条流水线 30 fps, CPU 约 0.40 核 (其中 worker 自己约 0.08 核, 已计入引擎统计),
对比 CPU 版 RVM 的 61 ms CPU (2.4 核)。

精度: worker 打开了 ncnn 的 fp16 存储/运算, 与 ORT fp32 相比 pha 平均差约 5e-4,
最大约 0.12 (个别边缘像素), 量化到 8 位后约 2% 像素差 1~几级 —— 肉眼看不出, 但不是逐位相同。

  from studio.rvm_gpu import RvmGpuMatting
  m = RvmGpuMatting()
  alpha = m(frame_bgr)          # (180, 320) float32
"""
from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORKER = ROOT / "third_party" / "rvm_worker"
PARAM = ROOT / "models" / "rvm_mobilenetv3_fp32.ncnn.param"
BIN = ROOT / "models" / "rvm_mobilenetv3_fp32.ncnn.bin"


class RvmGpuMatting:
    """ncnn+Vulkan 版 RVM。接口与 studio.rvm.RvmMatting 一致, 多一个 device 属性。"""

    def __init__(self, width: int = 320, height: int = 180,
                 param: Path | str = PARAM, model_bin: Path | str = BIN):
        missing = [str(p) for p in (WORKER, Path(param), Path(model_bin)) if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                "RVM GPU 后端缺文件:\n  " + "\n  ".join(missing) +
                "\n先跑 (一次即可):\n"
                "  .venv/bin/python studio/native/convert_rvm.py\n"
                "  bash studio/native/build_rvm_worker.sh")
        self.w, self.h = width, height
        self.device = "VULKAN"
        # start_new_session: 让 worker 自己成一组, 终端 Ctrl-C 只打主进程;
        # 退出时由 close() 负责收掉
        self.proc = subprocess.Popen(
            [str(WORKER), str(param), str(model_bin)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
            start_new_session=True)
        self._run(np.zeros((height, width, 3), np.uint8))     # 预热: Vulkan 初始化 + 首帧

    def reset(self) -> None:
        raise NotImplementedError("worker 的状态常驻进程; 要重置请重新构造")

    def child_cpu_time(self) -> float:
        """worker 子进程累计用掉的 CPU 秒数 (time.process_time() 不包含子进程)"""
        try:
            with open(f"/proc/{self.proc.pid}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")   # utime + stime
        except (OSError, ValueError, IndexError):
            return 0.0

    def _read_exact(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            chunk = self.proc.stdout.read(n - len(out))
            if not chunk:
                raise RuntimeError(
                    "RVM GPU worker 退出了 (third_party/rvm_worker); "
                    "看终端上面有没有 Vulkan 报错")
            out += chunk
        return bytes(out)

    def _run(self, small_bgr: np.ndarray) -> np.ndarray:
        h, w = small_bgr.shape[:2]
        self.proc.stdin.write(struct.pack("<II", w, h))
        self.proc.stdin.write(np.ascontiguousarray(small_bgr).tobytes())
        self.proc.stdin.flush()
        buf = self._read_exact(w * h * 4)
        return np.frombuffer(buf, np.float32).reshape(h, w).copy()

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.shape[1] != self.w or frame_bgr.shape[0] != self.h:
            frame_bgr = cv2.resize(frame_bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        self.last_small = frame_bgr      # 低分辨率准备可以直接复用这张 (省一次 720p 缩放)
        return self._run(frame_bgr)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()          # ncnn 的 Vulkan 析构偶尔会卡, 直接 kill
            self.proc.wait(timeout=3)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
