"""npu-studio-effects: Linux 上的背景虚化虚拟摄像头

  python -m studio                       # 摄像头 → 背景虚化 → 自动找到的 v4l2loopback 设备
  python -m studio --mode image --bg 海滩.jpg
  python -m studio --bench 150           # 不输出, 只测性能
  python -m studio --snapshot out.png    # 处理一帧存图 (原图|结果 并排)
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
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
        # 不要设 CAP_PROP_BUFFERSIZE=1: 只有 1 个 V4L2 缓冲时驱动会丢帧 (30→20 fps);
        # 读取线程本身只保留最新一帧, 不会积压延迟
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


class Status:
    """--status-json: 给前端 (npu-studio) 输出一行一个 JSON 的事件"""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def emit(self, event: str, **kw) -> None:
        if self.enabled:
            print(json.dumps({"event": event, **kw}, ensure_ascii=False), flush=True)


class DynamicFramerate:
    """--high-fps: 关掉 UVC 摄像头的 exposure_dynamic_framerate (暗光时会降到 ~15fps), 退出时还原"""

    def __init__(self, device: str):
        self.device, self.old = device, None
        try:
            out = subprocess.run(["v4l2-ctl", "-d", device, "-C", "exposure_dynamic_framerate"],
                                 capture_output=True, text=True, timeout=3).stdout
            self.old = out.split(":")[1].strip() if ":" in out else None
            if self.old is not None:
                self._set("0")
        except (OSError, subprocess.SubprocessError):
            self.old = None

    def _set(self, v: str) -> None:
        subprocess.run(["v4l2-ctl", "-d", self.device, "-c", f"exposure_dynamic_framerate={v}"],
                       capture_output=True, timeout=3)

    def restore(self) -> None:
        if self.old is not None:
            try:
                self._set(self.old)
            except (OSError, subprocess.SubprocessError):
                pass


def main() -> None:
    status = Status("--status-json" in sys.argv)
    try:
        _main(status)
    except SystemExit as e:
        if isinstance(e.code, str):          # sys.exit("错误信息")
            status.emit("error", message=e.code)
            if status.enabled:
                print(e.code, file=sys.stderr)
                raise SystemExit(1)
        raise
    except Exception as e:                   # noqa: BLE001 — 前端需要看到原因
        status.emit("error", message=f"{type(e).__name__}: {e}")
        raise


def _main(status: Status) -> None:
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
    ap.add_argument("--matting", default="mediapipe", choices=["mediapipe", "rvm"],
                    help="抠像模型: mediapipe=轻量分割 (默认); rvm=RVM, 头发丝级 + 时序记忆, "
                         "CPU 上约 19 ms/帧 (4 线程)")
    ap.add_argument("--threads", default=2, type=int, help="分割模型 CPU 线程数")
    ap.add_argument("--rvm-threads", default=4, type=int, help="RVM 的 CPU 线程数")
    ap.add_argument("--rvm-gpu", action="store_true",
                    help="RVM 跑在核显上 (ncnn+Vulkan), CPU 只剩缩放和合成; 需要先转换模型并编译 worker")
    ap.add_argument("--bench", type=int, metavar="N", help="处理 N 帧后输出性能统计并退出")
    ap.add_argument("--snapshot", metavar="PNG", help="处理一帧并保存 (原图|结果 并排) 后退出")
    ap.add_argument("--npu", action="store_true",
                    help="全分辨率的放大/遮罩/合成交给 NPU (需先 source npu/env.sh)")
    ap.add_argument("--npu-cols", default=4, type=int, help="NPU 并行列数 (默认4)")
    ap.add_argument("--gpu", choices=["rusticl", "rocm"],
                    help="全分辨率部分交给 780M 核显 (OpenCL), 选择驱动: rusticl=Mesa, rocm=ROCm")
    ap.add_argument("--cv-threads", default=1, type=int,
                    help="OpenCV 线程数 (默认1; 这种小图运算多线程只会白白占满所有核)")
    ap.add_argument("--high-fps", action="store_true",
                    help="关闭摄像头的动态帧率, 保证 30fps (画面会暗一些), 退出时还原")
    ap.add_argument("--status-json", action="store_true",
                    help="每秒向 stdout 输出一行 JSON 状态 (供图形前端使用)")
    ap.add_argument("--preview-shm", metavar="PATH",
                    help="把 640x360 预览写入共享内存文件 (供图形前端使用, 见 studio/preview.py)")
    a = ap.parse_args()
    if a.npu and a.gpu:
        sys.exit("--npu 和 --gpu 只能选一个")
    if a.gpu:
        from .gpu_fx import select_device
        select_device(a.gpu)            # 必须在 OpenCV 首次初始化 OpenCL 之前
    else:
        cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(a.cv_threads)

    status.emit("starting", backend="npu" if a.npu else (f"gpu-{a.gpu}" if a.gpu else "cpu"))
    if a.matting == "rvm":
        if a.rvm_gpu:
            from .rvm_gpu import RvmGpuMatting
            seg = RvmGpuMatting()
            seg_desc = "RVM (GPU, ncnn+Vulkan)"
        else:
            from .rvm import RvmMatting
            seg = RvmMatting(threads=a.rvm_threads)
            seg_desc = f"RVM (CPU ×{a.rvm_threads})"
        if "--smooth" not in sys.argv:
            a.smooth = 0.0          # RVM 自带时序记忆, 再叠 EMA 只会拖影
        if "--threshold" not in sys.argv:
            a.threshold = 0.0       # 遮罩已经很准, 再软阈值会把头发丝切掉
    else:
        seg = Segmenter(threads=a.threads)
        seg_desc = "MediaPipe 分割"
    cam = Camera(a.camera, *a.size, a.fps)
    dynfps = DynamicFramerate(a.camera) if a.high_fps else None   # 摄像头打开成功后再改设置
    w, h = cam.size
    if a.npu:
        from .npu_fx import NpuBackgroundFx
        fx = NpuBackgroundFx(w, h, a.mode, a.blur, a.smooth, a.threshold, a.bg, n_cols=a.npu_cols)
        backend = f"NPU ×{a.npu_cols}列"
    elif a.gpu:
        from .gpu_fx import GpuBackgroundFx
        fx = GpuBackgroundFx(w, h, a.mode, a.blur, a.smooth, a.threshold, a.bg)
        fx.t_npu = 0.0
        backend = f"GPU {fx.device_name}"
    else:
        fx = BackgroundFx(a.mode, a.blur, a.smooth, a.feather, a.threshold, a.bg, a.quality)
        backend = "CPU"
    accel = "NPU" if a.npu else ("GPU" if a.gpu else None)
    print(f"[studio] 摄像头 {a.camera} {w}x{h}@{cam.fps:.0f}  模型 {seg.w}x{seg.h} {seg_desc}  "
          f"模式 {a.mode}  后端 {backend}", file=sys.stderr)

    out = None
    if not (a.bench or a.snapshot) and a.output != "none":
        dev = find_loopback() if a.output == "auto" else a.output
        if not dev:
            sys.exit("没有找到 v4l2loopback 设备。先运行:\n"
                     "  sudo modprobe v4l2loopback devices=1 exclusive_caps=1 "
                     "card_label=\"NPU Studio Camera\"\n或用 --output none 只测试")
        out = VirtualCamera(dev, w, h)
        print(f"[studio] 输出到虚拟摄像头 {dev}  (会议软件里选择 \"NPU Studio Camera\")", file=sys.stderr)
    preview = None
    if a.preview_shm:
        from .preview import PreviewShm
        preview = PreviewShm(a.preview_shm)
    status.emit("ready", device=out.device if out else None, backend=backend,
                width=w, height=h, camera_fps=round(cam.fps), mode=a.mode)

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    seq, n = 0, 0
    t_seg = t_fx = t_out = t_prep = t_npu = 0.0
    t_start = t_report = time.perf_counter()
    cpu_start = time.process_time()
    # 每秒一个窗口的统计 (给前端的实时数字)
    win = dict(n=0, seg=0.0, fx=0.0, out=0.0, prep=0.0, accel=0.0)
    win_t, win_cpu = time.perf_counter(), time.process_time()
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
            if preview and n % 2 == 0:          # 预览 15 fps 足够
                preview.write(result)
            n += 1
            t_seg += t1 - t0
            t_fx += t2 - t1
            t_out += t3 - t2
            if accel:
                t_prep += fx.t_prep
                t_npu += fx.t_npu if a.npu else fx.t_gpu
            if status.enabled:
                win["n"] += 1
                win["seg"] += t1 - t0
                win["fx"] += t2 - t1
                win["out"] += t3 - t2
                if accel:
                    win["prep"] += fx.t_prep
                    win["accel"] += fx.t_npu if a.npu else fx.t_gpu
                now = time.perf_counter()
                if now - win_t >= 1.0:
                    k = max(1, win["n"])
                    cpu_now = time.process_time()
                    status.emit("stats", fps=round(win["n"] / (now - win_t), 1),
                                seg_ms=round(win["seg"] / k * 1000, 2),
                                fx_ms=round(win["fx"] / k * 1000, 2),
                                prep_ms=round(win["prep"] / k * 1000, 2),
                                accel_ms=round(win["accel"] / k * 1000, 2),
                                out_ms=round(win["out"] / k * 1000, 2),
                                cpu_pct=round((cpu_now - win_cpu) / (now - win_t) * 100, 1),
                                person=round(float(alpha.mean()) * 100, 1), frames=n)
                    win = dict(n=0, seg=0.0, fx=0.0, out=0.0, prep=0.0, accel=0.0)
                    win_t, win_cpu = now, cpu_now

            if a.snapshot and n >= 15:   # 跳过前几帧, 等自动曝光和时间平滑稳定
                shown = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR) if result.shape[2] == 4 else result
                cv2.imwrite(a.snapshot, np.hstack([frame, shown]))
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
        if dynfps:
            dynfps.restore()
        if preview:
            preview.close()
        status.emit("stopped", frames=n)

    if n and (a.bench or not a.snapshot):
        el = time.perf_counter() - t_start
        cpu = time.process_time() - cpu_start
        detail = f" [CPU准备 {t_prep/n*1000:.1f} + {accel} {t_npu/n*1000:.1f}]" if accel else ""
        print(f"[studio] 共 {n} 帧, {n/el:.1f} fps  "
              f"(分割 {t_seg/n*1000:.1f} ms, 合成 {t_fx/n*1000:.1f} ms{detail}, 输出 {t_out/n*1000:.1f} ms; "
              f"每帧处理合计 {(t_seg+t_fx+t_out)/n*1000:.1f} ms)  "
              f"CPU 时间 {cpu/n*1000:.1f} ms/帧 (约 {cpu/el*100:.0f}% 单核)", file=sys.stderr)


if __name__ == "__main__":
    main()
