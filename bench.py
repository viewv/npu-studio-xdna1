"""离线性能对比: 同一帧反复处理, 不受摄像头帧率限制

  source npu/env.sh && python3 bench.py [图片] [--frames N]

测量每帧墙钟时间 (可达到的最高帧率) 和进程 CPU 时间 (所有线程合计)。
"""
import sys
import time

import cv2
import numpy as np


def run(name, seg, fx, frame, n, to_bgr=False):
    for _ in range(10):                               # 预热
        fx.apply(frame, seg(frame))
    w0, c0 = time.perf_counter(), time.process_time()
    t_seg = t_fx = 0.0
    for _ in range(n):
        a = time.perf_counter()
        alpha = seg(frame)
        b = time.perf_counter()
        out = fx.apply(frame, alpha)
        if to_bgr:
            out = cv2.cvtColor(out, cv2.COLOR_BGRA2YUV_I420)
        else:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2YUV_I420)
        c = time.perf_counter()
        t_seg += b - a
        t_fx += c - b
    wall = (time.perf_counter() - w0) / n
    cpu = (time.process_time() - c0) / n
    print(f"{name:10} 每帧 {wall*1000:6.2f} ms (最高 {1/wall:5.1f} fps) | 分割 {t_seg/n*1000:5.2f} ms  "
          f"效果+输出 {t_fx/n*1000:5.2f} ms | CPU 时间 {cpu*1000:6.2f} ms/帧 ({cpu/wall:.1f} 核)")
    return wall, cpu


def main():
    path = next((x for x in sys.argv[1:] if not x.startswith("--") and not x.isdigit()), None)
    n = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 300
    if path:
        img = cv2.imread(path)
        frame = np.ascontiguousarray(img[:720, :1280])
    else:
        rng = np.random.default_rng(0)
        frame = cv2.GaussianBlur(rng.integers(0, 256, (720, 1280, 3), np.uint8), (31, 31), 0)

    from studio.effects import BackgroundFx, Segmenter
    from studio.npu_fx import NpuBackgroundFx
    seg = Segmenter()
    print(f"OpenCV 线程数 {cv2.getNumThreads()}, {n} 帧, 1280x720\n")
    w_cpu, c_cpu = run("CPU (hq)", seg, BackgroundFx("blur", 61, 0.5, 5, 0.15, None, "hq"), frame, n)
    w_npu, c_npu = run("NPU", seg, NpuBackgroundFx(1280, 720, "blur", 61, 0.5, 0.15), frame, n, to_bgr=True)
    print(f"\n每帧耗时 {w_cpu*1000:.1f} → {w_npu*1000:.1f} ms  (节省 {(1-w_npu/w_cpu)*100:.0f}% 时间, "
          f"{w_cpu/w_npu:.2f}× 帧率)")
    print(f"CPU 时间 {c_cpu*1000:.1f} → {c_npu*1000:.1f} ms/帧  ({(c_npu/c_cpu-1)*100:+.0f}%)")


if __name__ == "__main__":
    main()
