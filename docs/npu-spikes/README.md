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

`rvm_convs_180x320.tsv`：RVM（MobileNetV3）85 个卷积在工作分辨率 320×180 下的实测形状和 MAC 数（onnxruntime 跑出来的）。

文中"未提交"之类的状态是写的当时；对应改动后来都已提交到 `phase2-npu`。
