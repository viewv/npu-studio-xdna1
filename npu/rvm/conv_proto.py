"""NPU 原型: 把 RVM 解码器里最重的卷积 (Conv_290: 35→16, 3×3, 180×320, +ReLU) 放到 Phoenix NPU

使用 mlir-aie 自带的 int8 conv2dk3 内核 (aie_kernels/aie2/conv2dk3.cc):
  激活布局  [C/8][W][8]  (每行), int8
  权重布局  [OC/8][IC/8][ky][kx][ic8][oc8], int8
  输出      [OC/8][W][8] uint8 = clamp((acc + 2^(s-1)) >> s, 0, 255)   ← 自带 ReLU
  左右边界补零; 上下边界由 check 参数控制 (这里用主机端补零行 + 统一 middle)
内核没有 bias: 35 个输入通道补到 40, 用空闲通道放常数 127, 把 bias 写进其中心权重。

  source npu/env.sh && python3 -m npu.rvm.conv_proto [--cols 4]

数据: npu/rvm/conv290_case.npz (真实画面跑 RVM 时截下的输入/权重/输出)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE = HERE / "conv290_case.npz"

BIAS_K = 127            # 常数通道的值


# ------------------------------------------------------------------ 量化

class QConv:
    """把 float 卷积 (W, B) 量化成 conv2dk3 内核能用的 int8 形式"""

    def __init__(self, W: np.ndarray, B: np.ndarray, x_absmax: float, y_max: float):
        oc, ic, kh, kw = W.shape
        assert (kh, kw) == (3, 3) and oc % 8 == 0
        self.oc, self.ic = oc, ic
        self.icp = ((ic + 1 + 7) // 8) * 8          # 至少留 1 个通道放 bias
        self.s_x = x_absmax / 127.0
        self.s_w = float(np.abs(W).max()) / 127.0
        acc_unit = self.s_x * self.s_w              # 累加器 1 个单位对应的实数
        # 选 shift: 让 y_max 落在 uint8 的 ~240 以内
        self.shift = max(1, int(np.floor(np.log2(240 * 1.0 / (y_max / acc_unit) + 1e-30) * -1)))
        while y_max / (acc_unit * 2 ** self.shift) > 250 and self.shift < 30:
            self.shift += 1
        while self.shift > 1 and y_max / (acc_unit * 2 ** (self.shift - 1)) <= 250:
            self.shift -= 1
        self.s_y = acc_unit * 2 ** self.shift
        Wq = np.clip(np.rint(W / self.s_w), -127, 127).astype(np.int32)
        # bias → 空闲通道的中心权重, 必要时分摊到多个空闲通道
        bq = np.rint(B / acc_unit).astype(np.int64)
        spare = list(range(ic, self.icp))
        Wp = np.zeros((oc, self.icp, 3, 3), np.int32)
        Wp[:, :ic] = Wq
        remaining = bq.copy()
        self.bias_channels = []
        for ch in spare:
            if not remaining.any():
                break
            part = np.clip(np.rint(remaining / BIAS_K), -127, 127).astype(np.int64)
            Wp[:, ch, 1, 1] = part
            remaining = remaining - part * BIAS_K
            self.bias_channels.append(ch)
        self.bias_residual = int(np.abs(remaining).max())
        self.Wq = Wp                                # (OC, ICP, 3, 3) int32 (值在 int8 范围)

    def quant_x(self, x: np.ndarray) -> np.ndarray:
        """x: (IC, H, W) float → (ICP, H, W) int8, 含 bias 常数通道"""
        ic, h, w = x.shape
        q = np.zeros((self.icp, h, w), np.int8)
        q[:ic] = np.clip(np.rint(x / self.s_x), -127, 127)
        for ch in self.bias_channels:
            q[ch] = BIAS_K
        return q

    def dequant_y(self, yq: np.ndarray) -> np.ndarray:
        return yq.astype(np.float32) * self.s_y

    # ---------- 内核布局 ----------
    def pack_weights(self) -> np.ndarray:
        oc, icp = self.oc, self.icp
        w = self.Wq.reshape(oc // 8, 8, icp // 8, 8, 3, 3)          # [ocb][oc8][icb][ic8][ky][kx]
        w = w.transpose(0, 2, 4, 5, 3, 1)                          # [ocb][icb][ky][kx][ic8][oc8]
        return np.ascontiguousarray(w.reshape(-1)).astype(np.int8)

    def pack_input(self, xq: np.ndarray) -> np.ndarray:
        """(ICP, H, W) → 上下各补一行零 → (H+2, ICP/8, W, 8)"""
        icp, h, w = xq.shape
        pad = np.zeros((icp, h + 2, w), np.int8)
        pad[:, 1:-1] = xq
        for ch in self.bias_channels:       # 补的零行里常数通道也得是 0 (就是零填充)
            pad[ch, 0] = pad[ch, -1] = 0
        return np.ascontiguousarray(pad.reshape(icp // 8, 8, h + 2, w).transpose(2, 0, 3, 1))

    def unpack_output(self, out: np.ndarray, h: int, w: int) -> np.ndarray:
        """(H, OC/8, W, 8) → (OC, H, W)"""
        return out.reshape(h, self.oc // 8, w, 8).transpose(1, 3, 0, 2).reshape(self.oc, h, w)

    # ---------- 32 像素宽条带 (上游向量内核把宽度写死为 32) ----------
    TILE, STRIDE = 32, 30          # 每条带左右各 1 列重叠, 中间 30 列有效

    def n_tiles(self, w: int, multiple: int = 1) -> int:
        t = -(-w // self.STRIDE)
        return -(-t // multiple) * multiple

    def pack_input_tiles(self, xq: np.ndarray, n_tiles: int) -> np.ndarray:
        """(ICP, H, W) → (T, H+2, ICP/8, 32, 8); 条带 t 覆盖原图列 [30t-1, 30t+31)"""
        icp, h, w = xq.shape
        pad = np.zeros((icp, h + 2, self.STRIDE * n_tiles + 2), np.int8)
        pad[:, 1:-1, 1:w + 1] = xq
        for ch in self.bias_channels:
            pad[ch, 0] = pad[ch, -1] = 0
        cols = np.arange(n_tiles)[:, None] * self.STRIDE + np.arange(self.TILE)[None]   # (T, 32)
        t = pad[:, :, cols]                                     # (ICP, H+2, T, 32)
        t = t.reshape(icp // 8, 8, h + 2, n_tiles, self.TILE).transpose(3, 2, 0, 4, 1)
        return np.ascontiguousarray(t)

    def unpack_output_tiles(self, out: np.ndarray, n_tiles: int, h: int, w: int) -> np.ndarray:
        """(T, H, OC/8, 32, 8) → (OC, H, W), 每条带取中间 30 列"""
        o = out.reshape(n_tiles, h, self.oc // 8, self.TILE, 8)[:, :, :, 1:self.TILE - 1]
        o = o.transpose(2, 4, 1, 0, 3).reshape(self.oc, h, n_tiles * self.STRIDE)
        return o[:, :, :w]

    # ---------- 与内核逐位一致的参考实现 ----------
    def reference(self, xq: np.ndarray) -> np.ndarray:
        icp, h, w = xq.shape
        xp = np.zeros((icp, h + 2, w + 2), np.int64)
        xp[:, 1:-1, 1:-1] = xq
        acc = np.zeros((self.oc, h, w), np.int64)
        Wq = self.Wq.astype(np.int64)
        for ky in range(3):
            for kx in range(3):
                patch = xp[:, ky:ky + h, kx:kx + w].reshape(icp, -1)
                acc += (Wq[:, :, ky, kx] @ patch).reshape(self.oc, h, w)
        s = self.shift
        return np.clip((acc + (1 << (s - 1))) >> s, 0, 255).astype(np.uint8)


def float_reference(x, W, B):
    """float32 Conv + ReLU (用于衡量量化误差)"""
    ic, h, w = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    y = np.zeros((W.shape[0], h, w), np.float32)
    for ky in range(3):
        for kx in range(3):
            y += (W[:, :, ky, kx] @ xp[:, ky:ky + h, kx:kx + w].reshape(ic, -1)).reshape(-1, h, w)
    return np.maximum(y + B[:, None, None], 0)


def load_case():
    d = np.load(CASE)
    return d["x"], d["W"], d["B"], d["y"], d["calib"]


def quant_report(q: QConv, x, y_float):
    xq = q.quant_x(x)
    yq = q.reference(xq)
    y_hat = q.dequant_y(yq)
    err = np.abs(y_hat - y_float)
    rel = np.linalg.norm(y_hat - y_float) / max(1e-9, np.linalg.norm(y_float))
    print(f"[量化] s_x={q.s_x:.5f} s_w={q.s_w:.5f} shift={q.shift} s_y={q.s_y:.5f} "
          f"bias 通道 {q.bias_channels} (残差 {q.bias_residual} 单位)", file=sys.stderr)
    print(f"[量化] int8 vs float32: 相对误差 {rel*100:.2f}%  平均绝对误差 {err.mean():.4f}  "
          f"(输出均值 {y_float.mean():.4f}, 最大 {y_float.max():.3f})  饱和像素 {(yq == 255).mean()*100:.3f}%",
          file=sys.stderr)
    return xq, yq


if __name__ == "__main__" and "--quant-only" in sys.argv:
    x, W, B, y, calib = load_case()
    yf = float_reference(x, W, B)
    print(f"[参考] 自算 float 与 ONNX 输出最大差 {np.abs(yf - y).max():.2e}", file=sys.stderr)
    q = QConv(W, B, x_absmax=float(np.abs(calib).max()), y_max=float(np.percentile(yf, 99.99)))
    quant_report(q, x, yf)
