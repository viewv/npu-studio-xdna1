"""npu-studio-effects: Linux 上的背景虚化虚拟摄像头

  python -m studio                       # 摄像头 → 背景虚化 → 自动找到的 v4l2loopback 设备
  python -m studio --mode image --bg 海滩.jpg
  python -m studio --bench 150           # 不输出, 只测性能
  python -m studio --snapshot out.png    # 处理一帧存图 (原图|结果 并排)
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

import cv2
import numpy as np

from .effects import BackgroundFx, Segmenter
from .vcam import VirtualCamera, find_loopback


class Camera:
    """后台线程持续读取, 只保留最新一帧 (避免缓冲造成延迟)"""

    def __init__(self, device: str, width: int, height: int, fps: int):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            sys.exit(f"打不开摄像头 {device}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._frame, self._seq = None, 0
        self._cv = threading.Condition()
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cv:
                self._frame, self._seq = f, self._seq + 1
                self._cv.notify_all()

    def read(self, last_seq: int, timeout: float = 2.0):
        with self._cv:
            if not self._cv.wait_for(lambda: self._seq != last_seq, timeout):
                return None, last_seq
            return self._frame, self._seq

    def close(self):
        self._stop = True
        self.cap.release()


def parse_size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m studio", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument("--size", default="1280x720", type=parse_size)
    ap.add_argument("--fps", default=30, type=int)
    ap.add_argument("--output", default="auto", help="v4l2loopback 设备, auto=自动查找, none=不输出")
    ap.add_argument("--mode", default="blur", choices=["blur", "image", "color", "mask"])
    ap.add_argument("--blur", default=61, type=int, help="背景模糊强度, 全分辨率等效核大小 (默认61, 可到150+)")
    ap.add_argument("--quality", default="hq", choices=["hq", "basic"],
                    help="hq=导向滤波贴边+无光晕模糊 (默认); basic=第一版简单算法")
    ap.add_argument("--bg", help="--mode image 用的背景图")
    ap.add_argument("--smooth", default=0.5, type=float, help="遮罩时间平滑 0-0.95 (越大越稳但越拖影)")
    ap.add_argument("--threshold", default=0.15, type=float, help="遮罩软阈值 0-0.4 (越大边缘越硬)")
    ap.add_argument("--feather", default=5, type=int, help="边缘羽化像素 (basic 模式)")
    ap.add_argument("--threads", default=2, type=int, help="分割模型 CPU 线程数")
    ap.add_argument("--bench", type=int, metavar="N", help="处理 N 帧后输出性能统计并退出")
    ap.add_argument("--snapshot", metavar="PNG", help="处理一帧并保存 (原图|结果 并排) 后退出")
    a = ap.parse_args()

    seg = Segmenter(threads=a.threads)
    fx = BackgroundFx(a.mode, a.blur, a.smooth, a.feather, a.threshold, a.bg, a.quality)
    cam = Camera(a.camera, *a.size, a.fps)
    w, h = cam.size
    print(f"[studio] 摄像头 {a.camera} {w}x{h}@{cam.fps:.0f}  模型 {seg.w}x{seg.h}  模式 {a.mode}",
          file=sys.stderr)

    out = None
    if not (a.bench or a.snapshot) and a.output != "none":
        dev = find_loopback() if a.output == "auto" else a.output
        if not dev:
            sys.exit("没有找到 v4l2loopback 设备。先运行:\n"
                     "  sudo modprobe v4l2loopback devices=1 exclusive_caps=1 "
                     "card_label=\"NPU Studio Camera\"\n或用 --output none 只测试")
        out = VirtualCamera(dev, w, h)
        print(f"[studio] 输出到虚拟摄像头 {dev}  (会议软件里选择 \"NPU Studio Camera\")", file=sys.stderr)

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    seq, n = 0, 0
    t_seg = t_fx = t_out = 0.0
    t_start = t_report = time.perf_counter()
    try:
        while running:
            frame, seq = cam.read(seq)
            if frame is None:
                print("[studio] 摄像头没有新画面", file=sys.stderr)
                continue
            t0 = time.perf_counter()
            alpha = seg(frame)
            t1 = time.perf_counter()
            result = fx.apply(frame, alpha)
            t2 = time.perf_counter()
            if out:
                out.send(result)
            t3 = time.perf_counter()
            n += 1
            t_seg += t1 - t0
            t_fx += t2 - t1
            t_out += t3 - t2

            if a.snapshot and n >= 15:   # 跳过前几帧, 等自动曝光和时间平滑稳定
                cv2.imwrite(a.snapshot, np.hstack([frame, result]))
                print(f"[studio] 已保存 {a.snapshot}  人物占比 {float(alpha.mean())*100:.0f}%",
                      file=sys.stderr)
                break
            if a.bench and n >= a.bench:
                break
            now = time.perf_counter()
            if now - t_report >= 5:
                print(f"[studio] {n/(now-t_start):5.1f} fps | 分割 {t_seg/n*1000:5.1f} ms | "
                      f"合成 {t_fx/n*1000:5.1f} ms | 输出 {t_out/n*1000:4.1f} ms", file=sys.stderr)
                t_report = now
    finally:
        cam.close()
        if out:
            out.close()

    if n and (a.bench or not a.snapshot):
        el = time.perf_counter() - t_start
        print(f"[studio] 共 {n} 帧, {n/el:.1f} fps  "
              f"(分割 {t_seg/n*1000:.1f} ms, 合成 {t_fx/n*1000:.1f} ms, 输出 {t_out/n*1000:.1f} ms; "
              f"每帧处理合计 {(t_seg+t_fx+t_out)/n*1000:.1f} ms)", file=sys.stderr)


if __name__ == "__main__":
    main()
