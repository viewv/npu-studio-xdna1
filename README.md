# npu-studio-effects

Linux 上的「Windows Studio Effects」：摄像头背景虚化 / 替换，输出成虚拟摄像头，
任何会议软件（浏览器、Zoom、Teams、OBS）都能直接选用。

目标硬件：AMD Ryzen 7040 (Phoenix, XDNA NPU)。分阶段把计算从 CPU 挪到 NPU：

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | 人像分割 (MediaPipe ONNX) + 背景模糊/合成, 全部 CPU | ✅ 可用 |
| 2 | 全分辨率的 放大 + 导向滤波遮罩 + 合成 换成 NPU 融合内核 (IRON) | ✅ 可用 (`--npu`) |
| 3 | 分割模型 INT8 量化后整体搬到 NPU | 计划中 |

## NPU 后端 (`--npu`)

CPU 只做 1/4 分辨率 (320×180) 的计算：遮罩平滑、导向滤波系数、无光晕模糊；
全分辨率 (1280×720) 的逐像素工作由一个 NPU 融合内核
[`npu/kernels/fused_composite.cc`](npu/kernels/fused_composite.cc) 一次完成：

```
原图一行 (BGRA) + 低分辨率 A/B/背景行
  → 拆通道算灰度 → A、B、背景水平 ×4 双线性放大 → 遮罩 = A·(I−128)+B (定点)
  → 遮罩扩展到 4 通道 → alpha 合成 → 输出一行
```

全部用 2 路 `interleave_zip/unzip` 向量操作, 每次迭代 32 像素; 4 列 NPU 并行, 每列处理 180 行。

| 实测 (7840HS, 720p) | CPU 后端 | NPU 后端 |
|---|---|---|
| 全分辨率阶段 | ~10 ms | **1.6 ms** (NPU, 含同步) |
| 合成总计 (含 CPU 低分辨率准备) | 10.3 ms | 8.0 ms (CPU 5.6 + NPU 2.0) |
| **进程 CPU 时间 / 帧** | 122 ms | **70 ms (−43%)** |
| 与 CPU 版输出差异 | — | 平均 1.2/255, PSNR 46 dB |

NPU 版的主要收益是**省 CPU**（OpenCV 在全分辨率上的多线程运算被整段卸掉了），帧率仍受摄像头限制。

### 环境

```bash
uv venv --python /usr/bin/python3.14 --system-site-packages .venv-iron   # 需要系统 pyxrt (Arch: xrt 包)
uv pip install --python .venv-iron/bin/python -r requirements.txt mlir_aie \
    -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4
# Peano (llvm-aie): 直接装与 mlir_aie 1.3.4 配套的 nightly wheel
uv pip install --python .venv-iron/bin/python \
    'llvm-aie @ https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-21.0.0.2026062301%2Bcb664e8c-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'

source npu/env.sh
python3 -m npu.fused --selftest         # NPU 与 numpy 参考逐字节对比
python3 -m studio --npu                  # NPU 后端虚拟摄像头
```

首次运行会 JIT 编译 NPU 程序（约 2 秒），结果缓存后再启动很快。

## 安装

```bash
uv venv --python /usr/bin/python3.14 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 虚拟摄像头内核模块
sudo pacman -S v4l2loopback-dkms
sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label="NPU Studio Camera"
```

开机自动加载：

```bash
echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo 'options v4l2loopback devices=1 exclusive_caps=1 card_label="NPU Studio Camera"' | sudo tee /etc/modprobe.d/v4l2loopback.conf
```

## 使用

```bash
.venv/bin/python -m studio                        # 背景虚化 → 虚拟摄像头
.venv/bin/python -m studio --mode image --bg 背景.jpg
.venv/bin/python -m studio --mode color           # 绿幕
.venv/bin/python -m studio --blur 51              # 更强的模糊
.venv/bin/python -m studio --bench 150            # 只测性能
.venv/bin/python -m studio --snapshot out.png     # 存一张 原图|结果 对比
```

然后在会议软件的摄像头列表里选 **NPU Studio Camera**。

常用参数：`--blur`（模糊强度，默认 61，可到 121+）、`--smooth`（遮罩时间平滑，减少闪烁）、
`--threshold`（边缘软硬）、`--quality basic`（切回第一版简单算法做对比）。

默认的 `hq` 质量做了两件事：

- **快速导向滤波**：以原图为参照修正 256×144 的模型遮罩，轮廓贴合头发和衣服边缘
- **无光晕背景模糊**：归一化卷积，只用背景像素参与模糊，人物颜色不会被抹进背景

## 性能（7840HS, 1280×720, CPU 阶段）

| 步骤 | 耗时 |
|---|---|
| 人像分割 (256×144) | ~5–8 ms |
| 导向滤波 + 模糊 + 合成 (hq) | ~12–16 ms |
| 合计 | ~18–22 ms / 帧 |

实际帧率通常受摄像头限制：光线暗时 UVC 摄像头会自动降到 15 fps 左右
（`exposure_dynamic_framerate`）。

## 模型

`models/` 里的人像分割模型来自 [onnx-community/mediapipe_selfie_segmentation](https://huggingface.co/onnx-community/mediapipe_selfie_segmentation)
与其横屏版，原始模型为 Google MediaPipe Selfie Segmentation，Apache-2.0 许可。
