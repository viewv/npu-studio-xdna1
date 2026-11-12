# NPU 实验记录（历史）

把 RVM 整网搬上 Phoenix NPU 过程中每一轮实验的原始记录：做法、数字、踩的坑。按时间顺序写的，
**后面的轮次会修正前面的猜测**（例如 tiling spike 第一轮猜"权重没预取"，第二轮实测否定了）。

**要开发 NPU，先读 `.claude/skills/npu-dev/SKILL.md`**——那里是整理过、目前仍然成立的规则和流程。这里只用来查当时的数字和来龙去脉。

| 顺序 | 文件 | 内容 | 对应代码 |
|---|---|---|---|
| 0 | `npu-roadmap-analysis.md` | 后续功能分析 + llvm-aie 常量偏移 bug 的定位（探针、汇编对比在 `aie-compiler-bug/`） | `npu/offset_probe.py`、`row_ops.h` 的 `plane_stride` |
| 1 | `rvm-npu-chain-spike.md` | 同形状 N 层串在一次调用里，每层额外 ~0.04 ms | `npu/chain_spike.py` |
| 2 | `rvm-npu-hetero-spike.md` | 通用内核 + 运行时层描述符串异构层；stride 2 滑窗的 ObjectFifo 下降 bug | `npu/chain_hetero.py` |
| 3 | `rvm-npu-desc-loop-spike.md` | core 层循环改为描述符分派，代码大小与层数无关；RVM 形状表 | `npu/chain_hetero.py --mode` |
| 4 | `rvm-npu-c16-spike.md` | 16 通道像素布局 + 固定 640 B 输出块 + 每 core 行数；平衡循环规则 | `npu/chain_c16.py` |
| 5 | `rvm-npu-tiling-spike.md` | 高通道层的 OC / IC 分块、OC 并行、补宽列不读；task group 里 fill 太多会挂死 | `npu/chain_c16.py` |
| 6 | `rvm-npu-dw-spike.md` | depthwise 3×3/5×5、stride 2、dilation 2；栈溢出 | `npu/chain_c16.py`、`conv_c16.cc` |
| 7 | `rvm-npu-se-spike.md` | SE：全局平均池化、FC、HardSigmoid、逐通道乘折进权重；层间分支 | `npu/chain_c16.py`、`conv_c16.cc` |
| 8 | `rvm-npu-act-res-spike.md` | 省程序内存；hardswish / 线性输出；残差加 = 通道拼接 + 单位权重；IC 切行 | `npu/chain_c16.py`、`conv_c16.cc` |
| 9 | `rvm-npu-quant-spike.md` | 真实权重：量化方案（逐通道激活 + bias + 逐通道 M）、真实编码器 70 层逐字节一致；每层开销占 75% | `npu/rvm/encoder_*.py`、`conv_c16.cc` |
| 10 | `rvm-npu-overhead-spike.md` | 拆"每层开销"：按步 / 按 task group 算，不是按字节；权重预取 + task group 流水 + 层间权重预取，11.4 → 9.2 ms | `npu/chain_c16.py` |
| 11 | `rvm-npu-alpha-spike.md` | 摄像头实录看最终 alpha：int8 编码器 + 浮点解码器，真实画面校准后边缘带 2.4/255、IoU 0.997 | `npu/rvm/alpha_check.py` |

`rvm_convs_180x320.tsv`：RVM（MobileNetV3）85 个卷积在工作分辨率 320×180 下的实测形状和 MAC 数（onnxruntime 跑出来的）。

文中"未提交"之类的状态是写的当时；对应改动后来都已提交到 `phase2-npu`。
