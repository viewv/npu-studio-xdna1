# npu-studio-effects

Linux 上的「Windows Studio Effects」：摄像头背景虚化 / 替换，输出成虚拟摄像头，
任何会议软件（浏览器、Zoom、Teams、OBS）都能直接选用。

目标硬件：AMD Ryzen 7040 (Phoenix, XDNA NPU)。分阶段把计算从 CPU 挪到 NPU：

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | 人像分割 (MediaPipe ONNX) + 背景模糊/合成, 全部 CPU | ✅ 可用 |
| 2 | 全分辨率的 放大 + 导向滤波遮罩 + 合成 换成 NPU 融合内核 (IRON) | ✅ 可用 (`--npu`) |
| 3 | 抠像换成 RVM（时序记忆、头发丝边缘）：RVM 跑在 780M，合成仍在 NPU | ✅ 可用 (`--matting rvm --rvm-gpu --npu`) |

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

### 分辨率

NPU 合成内核**只在 1280×720 做过逐字节验证**（`--npu` 默认就是这个尺寸）。

| 分辨率 | NPU 合成 |
|---|---|
| 1280×720 | ✅ 1.3 ms，与 numpy 参考逐字节一致 |
| 1920×1080 | ❌ 23% 像素错，且错法固定（x%4∈{0,1} 的那些 lane）|

1080p 的错是 AIE 编译器的问题，不是内核算法：1280 过、1920 挂；改 `LOW_STRIDE`（512/544/576）、
去掉流水线 pragma、加软件流水，结果都一样或更糟。同样的坑也在 RVM 尾部内核上出现过（见下）。

所以 720p 是当前的支持范围：`--size 1920x1080 --npu` 会**直接报错**并建议 `--gpu`
（`NPU_ALLOW_UNVERIFIED=1` 可绕过，仅限实验）。720p 也正好是会议软件的主流发送分辨率，
而且让 RVM 的 180×320 → 720p 恰好是 ×4，与内核里写死的倍率一致。

**规则**：改内核、改分辨率、换编译器版本之后，都要在目标分辨率上跑
`python -m npu.fused --selftest --width W --height H` 再发布——"上次是好的"不算数。

## 可复用的内核骨架

阶段 2 的融合内核被拆成了三块，后面任何"逐行"的 NPU 内核都直接复用：

| 文件 | 作用 |
|---|---|
| [`npu/harness.py`](npu/harness.py) | `RowPipeline`：把行内核铺到 n_cols 个 core 上，负责 FIFO、按列切段、fill/drain、`stack_size` |
| [`npu/lib/row_ops.h`](npu/lib/row_ops.h) | 设备端算子：×4 水平放大（int16 / BGRA）、通道拆分与重组、灰度、mask 扩展、alpha 合成 |
| [`npu/kernels/fused_composite.cc`](npu/kernels/fused_composite.cc) | 消费者 1：全分辨率合成（与 numpy 参考逐字节一致） |
| [`npu/kernels/rvm_tail.cc`](npu/kernels/rvm_tail.cc) | 消费者 2：RVM 尾部融合（实验，见下） |

新增一个行内核只需要：

1. 在 `npu/kernels/` 写一个 `void myLine(输入行..., 输出行, 标量...)`，开头 `rowops::init()`；
2. 用 `external_row_kernel("myLine", SRC, [in_row_ty...], out_row_ty, scalars=[np.int32], include_dirs=[KERNEL_INC, LIB])` 建 `ExternalFunction`；
3. `pl = RowPipeline(kern, [输入行字节...], 输出行字节, height=H, n_cols=4, extra_args=[width]); pl.build()`；
4. `with rt.sequence(*pl.tensor_types()) as bufs: pl.stream(rt, bufs)`；
5. 主机侧仿照 `FusedCompositor` 管理缓冲区（直接 `iron.zeros(...).data.reshape(...)` 拿 numpy 视图，零拷贝写入）。

两个必须知道的坑：

- **JIT 缓存**：缓存键只看内核 `.cc` 的 mtime 和编译参数，改 `row_ops.h` / `harness.py` 不会失效，会静默跑旧二进制。改完执行 `rm -rf ~/.npu/cache/*`。症状是"改了代码结果不变"。
- **Worker 栈**：默认只有 1 KB。内核里同时活着的向量多（比如 8 个 int16 平面）会溢出，而且不报错，只是结果错乱。用 `RowPipeline(..., stack_size=8192)`。上限大约 16 KB（再大资源分配会失败）。

## RVM 尾部融合（实验，未接入）

`npu/rvm/tail.py` 把 RVM ONNX 图最后一段（两个 4 通道全分辨率 `Resize` + 乘加 + `Split`/`Clip`）
压成一个 NPU 行扫描，输入只有 1/4 分辨率的两个 4 通道张量（720p 每帧 ~1.8 MB，而不是 ~24 MB fp32）：

```
fgr_c = clamp((P_c·src_c + Q_c·255) >> 8 + src_c)
pha   = clamp((P_3·g     + Q_3·255) >> 8)          g = (B+G+R)/3
out   = (fgr·pha + up(bg)·(255-pha)) >> 8
```

当前状态：**编译、运行、接口都通了，但结果不对**——每 4 个输出像素里第 0、1 个错（`x % 4 ∈ {0,1}`），
`python -m npu.rvm.tail` 会报告不一致。

定位过程（每一步都用随机数据，常量数据复现不出来）：

| 试验 | 结果 |
|---|---|
| `up4_i16` 单独输出（含负数、斜坡） | 与参考逐元素相等 ✅ |
| `mul(P,u) >> 8` 单独转 uint8 | 每一 lane 都对 ✅ |
| `u` 单独输出（P=0） | 与源图逐像素相等 ✅ |
| 累加器上再做**一次**运算（`mac` 或 `add` 都一样） | lane ≡0 (mod 4) 恒 +95、≡1 (mod 4) 恒 +31 ❌ |
| 权重按值传参 / 权重改局部数组 / 结果过一遍本地内存 / 换 `u` 的宽化方式 | 无效 ❌ |
| 把 `+u·256` 折进 P（等价变换）/ `mac` 换 `add` / 加或去流水线 pragma | 无效 ❌ |

也就是说：同一个 32 lane 的 `acc32`，`mul` 之后直接 `to_vector<uint8_t>` 是对的，
但只要中间多一次累加，转换出来两个 lane 类的值就整体偏移——偏移量只跟 lane 位置有关，跟数据无关。
所以是 AIE 编译器的 `acc32 → uint8` 规约问题，不是算法、打包或量化。

**已找到一个能修好的做法，但有容量上限。** 把放大结果做成**跨迭代存活**的变量（手工软件流水：
本轮用上一轮算好的值，同时把下一轮算好），编译器就没法重算那一份，结果就对了：

| 试验 | 结果 |
|---|---|
| 单通道 + 2 个跨迭代向量 | **最大差 0** ✅ |
| 两通道 + 4 个跨迭代向量 | **最大差 0**（通道 0 验证）✅ |
| 本内核需要的 8 个跨迭代向量（32 lane） | 仍错 ❌ |
| 同样 8 个，改成 16 lane 向量（每个只占一半寄存器） | 仍错 ❌ |

所以不是寄存器宽度问题，是"跨迭代存活向量"的个数上限在 4 左右。
**下一步（明确）**：把内核拆成每遍 ≤4 个跨迭代向量的多遍实现——第一遍算 B/G，
第二遍算 R/alpha，第三遍合成，中间结果存一行 scratch。代价是多一遍行扫描和 2–4 W 字节的行缓冲。

在解决之前**不要接进 studio**。给这件事定个上限：即使全部修好，收益是 ~2 ms NPU 时间
换掉主机侧 30–90 mJ 的搬运，值钱但不是无底洞。


### 功耗测量

两把尺子，都以**整包 PPT**（`amdgpu/power1_average`，含 CPU+iGPU+NPU）为准：

```bash
python powerbench.py  --backends cpu,npu --secs 60     # 端到端: 各后端的整机瓦数
python powerbench_conv.py --secs 20 --rounds 2          # 微基准: 同一份卷积的 ms / W / J/帧
```

`powerbench_conv.py` 会直接算出 **J/帧**，以及"NPU 要做到多少 ms 才能在能量上打平 CPU"。
判据是 `E = (P_空闲 + ΔP) × t`。7840HS 上实测（720p，Conv_290 0.29 GMAC）：

| | ms/帧 | 整包 W | 边际 W | J/帧 |
|---|---|---|---|---|
| CPU ×2 | 4.1 | 34.6 | +13.4 | 0.143 |
| NPU ×4 | 11.1 | 22.7 | +1.4 | 0.252 |

NPU 的**边际**功率只有 1.4 W（CPU 13.4 W），但慢 2.7 倍，整机每帧能量反而高 76%。
打平线是 6.7 ms —— 也就是内核效率再提 1.7 倍。

⚠ 这台机器的 `BAT0` 没有 `power_now`，`current_now` 的量纲是坏的（1.2 W vs 整包 19 W），
所以脚本以 PPT 为准、电池值只做交叉验证。

## 环境

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

### RVM 抠像 (`--matting rvm`)

默认是轻量的 MediaPipe 分割。加 `--matting rvm` 换成 RVM（Robust Video Matting）：
头发丝级边缘 + 自带时序记忆（r1..r4 循环状态），所以不再需要 `--smooth` 的 EMA（那反而会拖影），
`--threshold` 也自动关掉，免得把细发丝切掉。

实现上**不碰合成内核**：摄像头帧直接缩到 320×180 喂给模型，它的 `pha` 天然就是 320×180 ——
正好是下游需要的 1/4 分辨率遮罩。所以不用截断 ONNX 图，也不用把全分辨率 `fgr`/`pha`
传回来（那会是每帧 15–33 MB）。只用 `pha`，前景仍然取原始画面。

**RVM 跑在 780M 核显上（ncnn + Vulkan）**，CPU 只剩缩放、1/4 准备和合成：

```bash
# 一次性准备。产物 (models/*.ncnn.*、third_party/rvm_worker) 在 .gitignore 里、只存在本机,
# 新克隆或换机器后需要重跑这两步
.venv/bin/python studio/native/convert_rvm.py --pnnx <pnnx> --onnxsim <onnxsim>
bash studio/native/build_rvm_worker.sh      # 用 Arch 仓库的 ncnn，不需要 sudo/编译 ncnn

# 跑
source npu/env.sh
python -m studio --matting rvm --rvm-gpu --npu --high-fps
```

实测（7840HS，1280×720，RVM@780M + NPU 合成，实测 2291 帧）：

| | 每帧 |
|---|---|
| RVM（320×180，Vulkan） | 18.8 ms |
| 合成（CPU 准备 4.6 + NPU 1.7） | 6.4 ms |
| 输出 | 0.8 ms |
| **合计 / 帧率** | **26.1 ms → 30.0 fps** |
| **CPU 时间** | **13 ms/帧（0.39 核，含 rvm_worker 子进程约 0.08 核）** |

对比同一台机器上的三种抠像方案：

| | 帧率 | CPU |
|---|---|---|
| MediaPipe 分割（阶段 1/2） | 30 fps | ~17 ms（0.5 核） |
| RVM，CPU 4 线程 | 39 fps | 61 ms（2.4 核） |
| **RVM，780M** | **30 fps** | **13 ms（0.39 核）** |

> CPU 时间口径：引擎统计 = 主进程 `process_time()` + 抠像子进程（`rvm_worker`）的 utime+stime。
> 早期版本只算了主进程，少算了 worker 的 ~0.08 核，报成了 0.30 核。
>
> 精度：worker 打开了 ncnn 的 fp16 路径（快 ~7%）。与 ORT fp32 相比，`pha` 平均差 ~5e-4、
> 最大 ~0.12（个别边缘像素），量化到 8 位后约 2% 像素差几级 —— 看不出来，但**不是逐位相同**。

两个实现都保留：`studio/rvm.py` 是 ORT 版（有 ROCm 时自动走核显，否则 CPU），
`studio/rvm_gpu.py` 是 ncnn+Vulkan 版（`--rvm-gpu`）。

模型转换是这条路上最麻烦的一步：RVM 的 ONNX 是动态形状（内部尺寸由 `Shape(src)` +
`downsample_ratio` 运行时算出来），ncnn/pnnx 处理不了。做法是把部署分辨率彻底焊死
（ratio 变常量 → onnxsim 折叠形状链 → Expand 换 Identity → ORT 逐字节验证 → pnnx），
全部封装在 `studio/native/convert_rvm.py` 里。

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
