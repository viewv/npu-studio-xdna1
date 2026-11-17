---
name: npu-dev
description: 在这个仓库里开发、修改、调试、测量 Phoenix XDNA1 NPU 内核（IRON / mlir-aie 1.3.4 + Peano llvm-aie）时使用——包括 npu/ 下的逐行内核（fused / rvm_tail）、自写卷积，以及把 RVM 整网搬上 NPU 的串联框架 npu/chain_c16.py（加新算子、新层类型、分块方式、性能优化）。遇到 ERT_CMD_STATE_TIMEOUT 挂死、"不报错只算错"、编译资源分配失败、改了代码结果不变，也先读这个。
---

# Phoenix NPU 开发手册

这台机器：Ryzen 7 7840HS，XDNA1（AIE2，`npu1`），4 列 compute tile 可用；每个 core 64 KB 数据内存、16 KB 程序内存。
工具链：IRON（mlir-aie 1.3.4）+ Peano（llvm-aie 21.0.0.2026062301），都装在 `.venv-iron`。

## 先做这三件事

```bash
source npu/env.sh                 # 每个 shell 都要
python -m npu.check               # 全部回归: 逐字节自检 + 编译器探针; 改任何 NPU 代码前后都跑
python -m npu.check --quick       # 只跑 720p 和较短的链, 快一半
```

`npu.check` 默认会清掉 JIT 缓存（`~/.npu/cache`）再跑。全部 ✅ 才算数——NPU 出错的典型样子是**不报错、只算错**。
新功能一定要在 `npu/check.py` 的 `_cases()` 里加一项。

## 代码地图

| 文件 | 是什么 |
|---|---|
| `npu/env.sh` | 环境变量 |
| `npu/check.py` | 一键回归 |
| `npu/harness.py` + `npu/lib/row_ops.h` | 逐行流水线脚手架 `RowPipeline` 和设备端行算子（合成类内核用） |
| `npu/fused.py`、`npu/kernels/fused_composite.cc` | studio 在用的全分辨率合成 |
| `npu/rvm/tail.py`、`npu/kernels/rvm_tail.cc` | RVM 尾部融合 |
| `npu/conv3x3.py`、`npu/kernels/conv3x3.cc` | 专用 int8 3×3 卷积（Conv_290），全展开，最快的卷积写法 |
| `npu/offset_probe.py` | llvm-aie 常量偏移 bug 的探针 |
| **`npu/chain_c16.py` + `npu/kernels/conv_c16.cc`** | **整网串联框架**：一次 NPU 调用跑 N 层，描述符驱动。RVM 整网的主线 |
| `npu/rvm/encoder_ref.py` | 从 ONNX 抽 RVM 编码器（135 个算子）+ numpy 浮点参考（和 onnxruntime 一致到 3e-5） |
| `npu/rvm/encoder_q.py` | int8 量化模拟（整数逐位定义）：几种方案对浮点的误差；冻结后就是 NPU 的逐字节参考 |
| `npu/rvm/encoder_npu.py` | 冻结的整数参数 → `chain_c16` 的层 + `LParam`，NPU 跑真实编码器（70 层），和模拟逐字节比 |
| `npu/ksize.py` | 单独编译内核、按函数列程序内存（`python -m npu.ksize`） |
| `npu/chain_spike.py`、`npu/chain_hetero.py`、`npu/kernels/conv_generic.cc` | 串联框架的早期版本，只用于复现历史数字，别在上面加功能 |
| `docs/npu-spikes/` | 每轮实验的原始记录和数字（历史；结论以本 skill 为准） |

## 整网串联框架一页纸（细节见 `references/chain-framework.md`）

- 一次调用 = 一个 runtime sequence；每层一个或几个 task group，层间靠 drain `wait=True` 当屏障。激活全在 DDR，主机不参与。
  层可以读任意更早的张量（`Layer.src`，SE 的投影层读 4 个张量之前的 dw 输出），每个存储张量一个自己的 scratch 槽。
  **通道拼接**（`Layer.cat`）：两个张量放进同一块存储（后一个排在前一个的通道组后面），读的层把它们当一个更宽的输入。
- 激活布局 **16 通道像素** `[H+2·BR][C/16][PAD + Ws + PAD][16]`，上下 BR=4 行零边、左右 PAD=4 像素零边，存储宽度 Ws 补到 40 的倍数；
  **边和补宽列永远是零、从不写**（内核把有效宽度外的输出写 0）。fill/drain 用 TAP 直接从整幅布局里切条带。
- **ObjectFifo 元素大小对所有层固定**：输入 7680 B（多读的是垃圾，内核不看）、输出 640 B（1 个通道组 × 40 像素）、权重 e_w（补到 256 的倍数）。
- **一个权重元素 = 一步**。权重头（128 B）写着这一步干什么：形状、每 core 的行数/有效宽度/选哪段循环。
  core 程序与层的形状、层数都无关（ELF `.text` ≈ 15.25 KB，其中 core 程序 3.2 KB），只有 4 段循环：s1 滑窗、s2、累加、收尾。
  池化、装 SE 缩放向量这类"不是卷积"的步也塞进累加 / 收尾段，靠权重头里的标志让内核分派——**加新步类型时先这么做，别加段**（每段约 0.6 KB）。
- 层模式：`full`（整层一步）、`oc`（权重太大，按输出通道切块，输入用步长 0 重读）、`ict`（1×1 输入太宽，按输入通道切步，部分和在 core 本地 int32）、
  `dw`（depthwise，DMA 把每个输出行要的 k 行打进一个元素）、`pool`（全局平均池化，按通道组分 core）。
  低分辨率的 oc 层、以及 1 行的 FC 层（含 ict）用 **OC 并行**（每 core 不同通道块、全部行）。
- 当前支持：普通 3×3（s1/s2）、1×1、depthwise 3×3/5×5（s1/s2、dilation 2）、**双线性 ×2 上采样**（`Layer(op=2, oh=…)`，解码器用）、**SE**（池化 → FC1 → FC2 HardSigmoid →
  投影层逐通道乘，乘法折进投影层权重）、**残差加**（拼接 + 单位权重），输出激活 ReLU / HardSigmoid / 线性 / **hardswish**，
  **重新量化**：累加器初值 = bias，`y = sat8((sat16(acc >> s1) · M[c]) >> s2)`（M 逐通道 int16）；bias / M / hardswish 的逐通道参数
  放在每步权重后面的重新量化表里（`RQB` = 576 B / 输出通道组，`LParam`、`rq_table`）。激活都是对称 int8（ReLU 0..127）。
  HardSigmoid / hardswish 在入口函数末尾对整个输出块做（`post_act`）。

## 硬规则（每条都踩过；症状和来龙去脉见 `references/pitfalls.md`）

1. **运行时次数的循环里，ObjectFifo 的 acquire/release 必须相对平衡**（循环体净取 = 净放）。stride 1 滑窗要先 `acquire(2)`，
   循环体 `acquire(3)/release(1)`，循环后 `release(2)`。不平衡 → 编译器报 `cannot statically analyze cyclostatic acquire pattern`。
2. **不要写 `acquire(3)/release(2)` 的 stride 2 滑窗**（IRON 1.3.4 下降 bug，前面有 `range_(≥2)` 时会挂死）。改成 DMA 每行发 3 行、`acquire(3)/release(3)`。
3. **一个 task group 里同一个 shim 通道最多排 4 个 fill**（7 个以上挂死）。多了就拆成多个 task group。
4. **shim BD：最高维（迭代维）≤ 64；跳着读时最内维 ≤ 1023 个字（4092 B）**。超了编译报 `Size N exceeds the [..] range`。
5. **内核里别放向量 / 累加器数组**，别让大栈帧被内联进入口函数。core 默认栈只有 1 KB，溢出不报错，**所有调用**都会算错或挂死。大函数标 `noinline`；
   **不要把累加器（`MM&` / `accum&`）按引用传给 noinline 函数**（调用方会把 4 个累加器放到栈上，同样溢出、挂死）——先 `to_vector` 再按值传向量。
   需要更大的栈就在 Worker / `RowPipeline` 上设 `stack_size`（上限约 16 KB）。
6. **一行里多个平面的指针不能写 `base + 编译期常量`**（偏移 ≥ 1024 B 时 llvm-aie 读错地址）。用 `rowops::plane_stride(N)` 或运行时值。`npu.check` 的探针会告诉你 bug 还在不在。
7. **`acquire(1)` 返回的是元素本身，不是列表**；`win[0]` 会变成取一个字节（编译报 operand type mismatch）。
8. **JIT 缓存键不含头文件、`harness.py`、模块级常量和 Plan 逻辑**。改了这些（例如 `chain_c16.py` 里的 `E_IN`、`W_MAX`）必须 `rm -rf ~/.npu/cache/*`，否则静默跑旧二进制。
   要长期保留的东西（ASR 权重缓存、C++ 导出包和二进制）放 `~/.npu/asr/`，别放 `~/.npu/cache/`：`npu.check` 会整个清掉它，`rm -rf ~/.npu/cache/[0-9a-f]*` 也会误删 `asr_…`。
   要隔离的编译用 `NPU_CACHE_HOME=<临时目录>`（`export_c_bundle` 就这么找 xclbin）。
9. **内存预算**：每 core 数据约 61–63/64 KB（输入 4×7.5 + 权重 2×9.7 或 1×18.9 + acc 7.5 + 输出/dummy 1.9 + SE 缩放 1），
   **程序 16 KB**：所有分支一起编约 16.0 KB（装不下），所以**按链编译**——`Plan.kernel_flags()` 给链里用不到的分支加 `-DNO_SE / NO_UP / NO_DW / NO_DW3 / NO_DW5 / NO_DW5D2 / NO_S2 / NO_PA`
   （编码器 ≈ 14.9 KB，解码器类 ≈ 7.5 KB）。加新分支：在 `convC16Chunk` 里用 `#ifndef NO_xxx` 包起来，在 `kernel_flags` 里加判断。
   `python -m npu.ksize -DCW=40 -DNO_UP …` 看某种组合的大小。
   超了 aiecc 报 `Overflow of program memory` / `XAie_LoadElf failed with XAIE_INVALID_ELF`；`python -m npu.ksize` 看是哪个函数。
   加缓冲区、加内核分支前先算账；`llvm-nm --size-sort -S -C main_core_0_2.elf` 看是哪个函数涨了（`core_0_2` 是 core 程序，`convC16Chunk` 是入口）。
10. **激活函数在 0 处不为 0 时（HardSigmoid、以后的 bias）要显式把有效宽度外的像素写 0**：ReLU 版靠"零输入 → 零输出"，换了激活就会把补宽列写脏（`check` 报 999）。
11. **Peano 不给循环做软件流水**：短的逐元素循环（load → 几条运算 → mul → srs → store）每轮都串着等延迟（实测约 22 周期 / 32 B）。
    手动每轮处理 4 个互不依赖的向量（小的定长数组、完全展开，会进寄存器——用 `llvm-objdump -d` 确认没有 sp 访问），快 2 倍。

12. **ASR 链的程序内存只剩约 100 B**（LM + ATTN + GELU + GEMM 一起编）。加代码前先 `python -m npu.ksize -DCW=40 -DNO_SE -DNO_UP -DNO_DW -DNO_DW3 -DNO_DW5 -DNO_DW5D2 -DNO_S2 -DNO_K3 -DNO_PA`
    （zsh 里别把这串放进一个变量再 `$F` 传——不会按空格拆开，ksize 会按全部分支算）。腾空间：不在热路径上的函数加 `__attribute__((minsize))`。
    超了报 `Overflow of program memory` / `XAIE_INVALID_ELF`。
13. **权重广播 `Plan.w_bcast`**：链里所有层都是行并行（没有 OC 并行 / 池化 / SE 装载）时，一个权重 fifo 广播给 4 个 core，只读 core 0 那份。
    加新模式时如果各 core 的权重元素可能不同，要在 `w_bcast` 的条件里排除它（`C16Chain` 打包时会 assert 4 份相同）。

## 加一个新算子 / 新层类型的流程

1. **先写 numpy 整数参考**（照 `chain_c16.conv_acc` / `requant` 的样子），所有比较都是逐字节。
2. 决定它属于哪种模式、要不要新模式；画出输入元素里的数据排布和 fill/drain 的 TAP（≤ 4 维，见规则 3、4）。
3. 描述符：优先复用现有字段和 core 的 4 段循环（depthwise 就完全复用了"收尾段"）。core 程序每加一段约 0.7–1.5 KB。
4. 内核：在 `conv_c16.cc` 的 `convC16Chunk` 里按描述符分派；性能敏感的用模板把核大小 / stride 等做成编译期常量（运行时版慢 2–3 倍）。
5. 测试顺序：单层 → 和前后层串成链 → 加进 `CHAINS` 和 `npu.check`。随机权重 ±12、每层 shift 自动选成输出均值约 24，检查输出不是全零/全饱和。
6. 测性能用**链里逐层加前缀的边际耗时**，不要用单层耗时（见下）。

## 接真实权重 / 改量化方案（细节见 `references/quant.md`）

1. 先在 `npu/rvm/encoder_q.py` 里改（numpy，整数逐位定义），看对浮点的余弦；**16 位激活 + 16 位权重应该接近 1.0**，不是就是有 bug。
2. 方案要改内核时，先给 `chain_c16` 的随机测试加对应的随机参数（`make_case(rq=True)` 那样），逐字节过了再接真实权重。
3. `python -m npu.rvm.encoder_npu`：NPU 和模拟必须逐字节一致；不一致就逐层读回比（`read_tensor`），找第一个错的层。

## 怎么测性能（细节见 `references/perf.md`）

- 单层调用里固定成本（XRT 派发 + 流水线启停，0.25–0.4 ms）占大头，**会误导**。用 deep/mnv3 链按前缀逐层加，看边际。
- 拆"纯计算 vs 额外开销"：内核每行重复调用 r = 1/2/3 次（`--compute`），对 r 的斜率是纯计算。
- 中位数受机器负载影响很大，同时看最快值；判据"每层额外开销 ≤ 0.1 ms"。
- 现在的数字：真实编码器 70 层 **9.2 ms 最快 / 9.6 中位**（纯计算 2.7，压开销前 11.4）；以下是压开销前、负载 ≈ 3 时：mnv3 27 层 ≈ 5.55 ms，mnv3r 39 层 ≈ 6.8 ms，deep 13 层 ≈ 2.84 ms；
  每层额外 ≈ 0.12 ms。重新量化路径（bias + 逐通道 M）比只移位慢约 9%。

## 调试挂死 / 算错

- `ERT_CMD_STATE_TIMEOUT` = 某个 core 在等永远不来的数据（或 DMA 在等没人收的数据）。NPU 下次调用会自动恢复，不用重启。
  先核对：每步 fill 的字节数 = core 消费的元素数 × 元素大小；drain 的字节数 = core 产出的块数 × 640。
- 生成物在 `~/.npu/cache/<hash>/`：`aie.mlir`（core 的结构化代码，看循环和 acquire/release）、
  `input_with_addresses.mlir`（ObjectFifo 下降后的锁操作，查 `AcquireGreaterEqual` 的数量对不对）、`main_core_*.elf`（用 `.venv-iron/.../llvm-aie/bin/llvm-size -A` 看 `.text`）。
- 结果错但不挂：先怀疑栈（规则 5）、常量偏移（规则 6）、陈旧缓存（规则 8），再怀疑 TAP。`chain_c16.check` 会把"边/补宽列不是零"报成 999。
- 二分：把链缩短（`--spec a,b,c` 逐个加层）、单独跑某一层（`--spec Conv_xxx`）、临时改 `chain_c16` 顶部的常量
  （`W_DEPTH`、`ACC_CHUNKS`、`FILLS_PER_TG`…，改完清缓存）。在 `choose_tiling` 里临时关掉 OC 并行 / 换模式也能定位是哪种模式的问题。

## 现状和待办（详见 `references/chain-framework.md` 末尾）

- **真实 RVM 编码器（70 层）在 NPU 上和 numpy 整数模拟逐字节一致**（`python -m npu.rvm.encoder_npu`）；反量化后对 onnxruntime 的余弦
  f1 0.998 / f2 0.966 / f3 0.943 / f4 0.879（比 onnxruntime 自己的 int8 静态量化好：f4 0.72–0.84）。量化方案见 `references/quant.md`。
- **但它还比 CPU 慢**：NPU 9.2 ms（压开销后，原 11.4），onnxruntime CPU 单线程 8.4 / 4 线程 3.1 ms。纯计算只有 2.7 ms，
  非计算开销 6.6 ms（层间屏障、每步固定代价）。权重预取 + task group 流水 + 层间权重预取已经做了（见 overhead spike）；
  下一步是把倒残差块融合成一层（少屏障、少步），不是再加算子。
- **功耗和画质都过关**：30 fps 下 NPU 编码器比 CPU 每秒省约 2.9 W（`powerbench_enc.py`）；用真实人像画面校准后，最终 alpha 和浮点只差边缘带 2.4/255、IoU 0.997（`npu/rvm/alpha_check.py`，见 quant.md）。下一步是解码器上 NPU。
- **和核显比**：780M 跑 RVM 整网 +3.3 W；NPU 编码器 + 解码器的卷积（`npu/rvm/decoder_probe.py`，高分辨率 3×3 250–300 GMAC/s）+1.2 W → 整网上 NPU 预计每秒省约 2 W。NPU 只有在计算密集时才比核显省（编码器那样开销占大头时每 MMAC 不如核显）。
- **解码器进行中**（`docs/npu-spikes/rvm-npu-decoder-spike.md`）：浮点参考 `npu/rvm/decoder_ref.py`、量化方案 `npu/rvm/decoder_q.py` 定了；
  NPU 侧已有：双线性 ×2 上采样（`up` 链）、通用存储分组（`Layer.nx`，多路拼接、跨层往后放）、窗口读、就地写（`Layer.al`）、
  外部输入（`op=3`）、分段调用（`parts`，**每多一段 +1.5 ms**，要重新加载硬件配置）。还缺 sigmoid / tanh 查表、GRU 逐元素、
  LRASPP 逐通道乘、DGF 细化，然后接入 studio。
- 已知性能债：12×20 的 IC 分块层（输入按 OC 份数重读）、12×20 的 5×5 dilation 2 depthwise（步数多）、通用 3×3 只有专用版 61% 的速度、宽 20 补成 40。
- RVM 真实形状表：`docs/npu-spikes/rvm_convs_180x320.tsv`（onnxruntime 在工作分辨率 320×180 下跑出来的）。
