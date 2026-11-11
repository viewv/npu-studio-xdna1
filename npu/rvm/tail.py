"""RVM 尾部融合: 1/4 分辨率的 (refiner 残差, alpha) → 全分辨率 fgr/pha + 合成

实验状态: selftest 在 1280×720 / 1920×1080 与整数参考逐字节一致 (曾因 llvm-aie 丢掉
   >=1024 B 的常量平面偏移而错, 见 kernels/rvm_tail.cc 文件头)。还没接进 studio:
   需要带 refiner 的 RVM 部署, 且主机侧 8 个平面的垂直插值太贵, 见 README。

对应 ONNX 图的最后一段 (Resize_332 / Resize_346 之后):

    up1 = Resize(Conv_316 输出)        4 通道, 180×320 → 全分辨率 (fp32)
    up2 = Resize(Sub_318 输出)         4 通道, 180×320 → 全分辨率 (fp32)
    src4 = Concat(src, ReduceMean(src))              全分辨率
    x   = up1 * src4 + up2                           ← 全分辨率乘加
    fgr = Clip(x[:3] + src)          pha = Clip(x[3:])
    out = fgr·pha + bg·(1-pha)

原来这一段要在全分辨率上写两个 4 通道 fp32 张量 (720p 每帧约 24 MB, 1080p 约 54 MB),
再算乘加/Split/Clip, 才能合成。这里整段压进一个 NPU 行内核:

    CPU: 1/4 分辨率 → 垂直插值 + 定点化 (每帧约 1.8 MB)
    NPU: 水平 ×4 放大 + refiner 残差 + 导向合成 (一遍扫过全分辨率行)

  source npu/env.sh && python -m npu.rvm.tail [--width 1280] [--height 720] [--cols 4] [--iters 30]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, Out, Program, Runtime

from ..fused import KERNEL_INC, LIB, WP, WC, WN, _up_h
from ..harness import RowPipeline, external_row_kernel

HERE = Path(__file__).resolve().parent
KERNEL = str(HERE.parent / "kernels" / "rvm_tail.cc")

# 低分辨率行的定点比例: real = q / 256
Q_SCALE = 256.0


def _round_up(v: int, m: int = 32) -> int:
    return -(-v // m) * m


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def rvm_tail(src: In, low: In, out: Out, *, width: CompileTime[int],
             height: CompileTime[int], low_stride: CompileTime[int],
             n_cols: CompileTime[int] = 4):
    """行内核: src(BGRA 行) + low(P/Q 平面 + bg) → out(BGRA 行)"""
    row = width * 4
    low_row = low_stride * 20                  # 8 个 int16 平面 + 1 个 BGRA 平面
    row_ty = np.ndarray[(row,), np.dtype[np.uint8]]
    low_ty = np.ndarray[(low_row,), np.dtype[np.uint8]]

    kern = external_row_kernel("rvmTailLine", KERNEL, [row_ty, low_ty], row_ty,
                               scalars=[np.int32], include_dirs=[KERNEL_INC, LIB],
                               compile_flags=[f"-DLOW_STRIDE={low_stride}"])

    pl = RowPipeline(kern, [row, low_row], row, height=height, n_cols=n_cols,
                     extra_args=[width], prefix="rv", stack_size=12288)
    pl.build()

    rt = Runtime()
    with rt.sequence(*pl.tensor_types()) as bufs:
        pl.stream(rt, bufs)
    return Program(iron.get_current_device(), rt).resolve_program()


# ---------------------------------------------------------------- 主机端

class RvmTail:
    """把 1/4 分辨率的 RVM 尾部张量送到 NPU, 得到全分辨率 BGRA

        tail = RvmTail(width, height)
        out  = tail.run(frame_bgra, res, alpha_res, bg_low)
            res      : (4, h/4, w/4) float32   Conv_316 输出 (refiner.conv.6)
            alpha_res: (4, h/4, w/4) float32   Sub_318 输出 (refiner 的 box_filter 分支)
            bg_low   : (h/4, w/4, 4) uint8     低分辨率背景
    """

    def __init__(self, width: int, height: int, n_cols: int = 4, low_stride: int | None = None):
        if width % 32:
            raise ValueError("宽必须是 32 的倍数")
        if height % n_cols:
            raise ValueError(f"高 {height} 必须能被列数 {n_cols} 整除")
        self.w, self.h, self.n_cols = width, height, n_cols
        self.lw = width // 4
        # 内核从 j0 往后读 16 个 int16 (最后一个块会读到 LW+9), 所以留够余量
        self.ls = low_stride or _round_up(self.lw + 16, 32)
        if self.ls < self.lw + 10:
            raise ValueError(f"low_stride={self.ls} 太小 (需要 >= {self.lw + 10})")

        row, low_row = width * 4, self.ls * 20
        self.t_src = iron.zeros(row * height, dtype=np.uint8, device="npu")
        self.t_low = iron.zeros(low_row * height, dtype=np.uint8, device="npu")
        self.t_out = iron.zeros(row * height, dtype=np.uint8, device="npu")
        self.src = self.t_src.data.reshape(height, width, 4)

        b = self.t_low.data.reshape(height, low_row)
        ls = self.ls
        self._p = [b[:, i * 2 * ls:(i + 1) * 2 * ls].view(np.int16) for i in range(4)]
        self._q = [b[:, (8 * ls + i * 2 * ls):(8 * ls + (i + 1) * 2 * ls)].view(np.int16)
                   for i in range(4)]
        self._bg = b[:, 16 * ls:].reshape(height, ls, 4)

        self.run(np.zeros((height, width, 4), np.uint8),
                 np.zeros((4, height // 4, self.lw), np.float32),
                 np.zeros((4, height // 4, self.lw), np.float32),
                 np.zeros((height // 4, self.lw, 4), np.uint8))     # 预热 / JIT

    # ---------- 垂直插值 (水平方向由 NPU 做) ----------
    def _vert(self, x: np.ndarray, channels_first: bool) -> np.ndarray:
        """垂直插值到全高度; 水平方向留给 NPU (可分离, 结果与二维双线性一致)"""
        import cv2
        if channels_first:                    # (C, h, lw)
            if x.shape[1] == self.h:
                return x
            return np.stack([cv2.resize(c, (self.lw, self.h), interpolation=cv2.INTER_LINEAR)
                             for c in x])
        if x.shape[0] == self.h:              # (h, lw, 4)
            return x
        return cv2.resize(x, (self.lw, self.h), interpolation=cv2.INTER_LINEAR)

    def _pack(self, res: np.ndarray, alpha_res: np.ndarray, bg: np.ndarray) -> None:
        lw, ls = self.lw, self.ls
        for planes, src in ((self._p, res), (self._q, alpha_res)):
            for i, dst in enumerate(planes):
                v = src[i]
                dst[:, 1:lw + 1] = v
                dst[:, 0] = v[:, 0]
                dst[:, lw + 1] = v[:, -1]
        self._bg[:, 1:lw + 1] = bg
        self._bg[:, 0] = bg[:, 0]
        self._bg[:, lw + 1] = bg[:, -1]

    def run(self, frame_bgra: np.ndarray, res: np.ndarray, alpha_res: np.ndarray,
            bg_low: np.ndarray) -> np.ndarray:
        """res / alpha_res: (4, h, lw) float32 → 内部按 1/256 定点化; bg_low: (h, lw, 4) uint8"""
        self.src[:] = frame_bgra
        q = lambda x: np.clip(np.rint(self._vert(x, True) * Q_SCALE), -32768, 32767).astype(np.int16)
        self._pack(q(res), q(alpha_res), self._vert(bg_low, False))
        t0 = time.perf_counter()
        self.t_src._sync_to_device()
        self.t_low._sync_to_device()
        self.t1 = time.perf_counter()
        rvm_tail(self.t_src, self.t_low, self.t_out, width=self.w, height=self.h,
                 low_stride=self.ls, n_cols=self.n_cols)
        self.t2 = time.perf_counter()
        return self.t_out.numpy().reshape(self.h, self.w, 4)


# ---------------------------------------------------------------- numpy 参考

def reference(frame_bgra: np.ndarray, res_q: np.ndarray, alpha_q: np.ndarray,
              bg_low: np.ndarray) -> np.ndarray:
    """与内核逐字节一致的整数参考 (res_q/alpha_q 已经是垂直插值 + 定点化的结果)"""
    p = frame_bgra.astype(np.int32)
    B, G, R = p[..., 0], p[..., 1], p[..., 2]
    g = ((B + G + R) * 21846) >> 16
    planes = (B, G, R)
    fgr = np.empty((*p.shape[:2], 3), np.int32)
    for c in range(3):
        up = _up_h(res_q[c]) * planes[c] + _up_h(alpha_q[c]) * 255
        fgr[..., c] = np.clip((up >> 8) + planes[c], 0, 255)
    m = np.clip((_up_h(res_q[3]) * g + _up_h(alpha_q[3]) * 255) >> 8, 0, 255)
    bgu = _up_h(bg_low)
    out = np.empty_like(p)
    out[..., :3] = (fgr * m[..., None] + bgu[..., :3] * (255 - m[..., None])) >> 8
    out[..., 3] = (255 * m + bgu[..., 3] * (255 - m)) >> 8
    return out.astype(np.uint8)


# ---------------------------------------------------------------- 自检

def selftest(width=1280, height=720, n_cols=4, iters=30):
    rng = np.random.default_rng(2)
    lw = width // 4
    tail = RvmTail(width, height, n_cols)
    print(f"[tail] 低分辨率 {lw} 宽, low_stride={tail.ls}, "
          f"每帧低分辨率数据 {4 * lw * height * 2 * 2 / 1e6:.2f} MB (P+Q int16)", file=sys.stderr)

    frame = rng.integers(0, 256, (height, width, 4), np.uint8)
    res_f = rng.normal(0, 2.0, (4, height, lw)).astype(np.float32)    # 已全高, 只测水平放大
    alpha_f = rng.normal(0, 0.5, (4, height, lw)).astype(np.float32)
    bg = rng.integers(0, 256, (height, lw, 4), np.uint8)
    tail.src[:] = frame
    res_q = np.clip(np.rint(res_f * Q_SCALE), -32768, 32767).astype(np.int16)
    alpha_q = np.clip(np.rint(alpha_f * Q_SCALE), -32768, 32767).astype(np.int16)
    tail._pack(res_q, alpha_q, bg)
    t0 = time.perf_counter()
    got = tail.run(frame, res_f, alpha_f, bg).copy()
    print(f"[tail] 首次调用 (含 JIT 编译) {time.perf_counter() - t0:.1f} s", file=sys.stderr)
    exp = reference(frame, res_q, alpha_q, bg)
    diff = np.abs(got.astype(int) - exp.astype(int))
    print(f"[tail] 与整数参考: 最大差 {diff.max()}  不一致 {(diff > 0).mean() * 100:.3f}%",
          file=sys.stderr)
    if diff.max() > 0:
        for y, x, c in np.argwhere(diff > 0)[:5]:
            print(f"   ({y},{x},{c}) npu={got[y, x, c]} ref={exp[y, x, c]}", file=sys.stderr)

    times, t_npu = [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        tail.run(frame, res_f, alpha_f, bg)
        times.append(time.perf_counter() - t0)
        t_npu.append(tail.t2 - tail.t1)
    times.sort()
    t_npu.sort()
    print(f"[tail] {width}x{height} n_cols={n_cols}: 中位数 {times[len(times) // 2] * 1000:.2f} ms "
          f"(其中 NPU {t_npu[len(t_npu) // 2] * 1000:.2f} ms)", file=sys.stderr)
    return diff.max() == 0


if __name__ == "__main__":
    def arg(name, default):
        return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
    ok = selftest(arg("--width", 1280), arg("--height", 720), arg("--cols", 4),
                  arg("--iters", 30))
    sys.exit(0 if ok else 1)
