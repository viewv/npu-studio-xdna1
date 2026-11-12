# Phoenix NPU 后续功能分析 + AIE 编译器 bug 定位

日期 2026-11-11，分支 `phase2-npu`（未改仓库任何文件，未提交）。
实验文件都在本目录 `aie-compiler-bug/`（探针内核、汇编对比、两个内核的绕过版副本）。

---

## 0. 先说最重要的发现：编译器 bug 已定位，一行绕过，两个内核都修好了

README 里「x%4∈{0,1} 的 lane 出错」的根因**不是** acc32→uint8 规约，也**不是**「跨迭代存活向量上限 ~4」。
是 **llvm-aie（Peano）在 AIE2 上把超出立即数范围的常量地址偏移折成 post-increment 时丢掉了偏移**：

- `up4_i16(bq, j0)` 里的第一个非对齐加载 `load_unaligned_v<16>(bq + j0, 1)`（v[j-1]），
  在 `bq = aq + 常量` 且常量 ≥ 1024 字节时，实际读的是 **`aq + j0`**（另一个平面！）。
  v[j]、v[j+1] 两个加载是对的。v[j-1] 只有 `wp={3,1,0,0}` 的 lane 0、1 用到 → 就是 `x%4∈{0,1}`。
  （README 里的 +95/+31 正好是 3:1，就是 wp 的比例。）
- 门槛非常锐利（探针，upB 单独输出）：

  | 平面偏移 | 结果 |
  |---|---|
  | 352/448/480/496 元素（≤ 992 B） | ✅ 0% 错 |
  | 512/528/544/704 元素（≥ 1024 B） | ❌ lane0/1 100% 错 |

  这解释了所有现象：fused 1280 的 `LOW_STRIDE=352`（Bq 偏移 704 B）过；1920 的 512/544/576（≥1024 B）全挂；
  rvm_tail 在 1280 上 P2 以后的平面（≥1408 B）挂。之前的「软件流水修好了」只是碰巧改了地址计算。
- 汇编（`aie-compiler-bug/v11_496.s` vs `v11_512.s`）：
  - 992 B：`vlda wl6, [p2, #992]` —— 立即数偏移形式，正确。
  - 1024 B：`vlda wl6, [p2], #1056` —— **post-increment 形式，先用 p2 加载、后加偏移**；
    而 `vshift` 的对齐量却是从正确地址 `padda [p3], #1024` 算的。低半个窗口读错了地址。
- 与迭代次数无关（4/8/20/40/50/60/80 次迭代错误率一样），与 `AIE_PREPARE_FOR_PIPELINING` 无关，常量数据复现不了（因为两个平面一样）。

**绕过**：让编译器看不到平面偏移是常量：

```cpp
volatile int32_t ls_v = LOW_STRIDE;   // llvm-aie: 常量平面偏移 >=1024B 会被丢
const int32_t ls = ls_v;
const int16_t *bq = aq + ls;
```

实测（7840HS，绕过版副本 `aie-compiler-bug/*.workaround.cc`，用仓库自带 selftest）：

| 内核 | 尺寸 | 结果 | 时间 |
|---|---|---|---|
| fused_composite | 1280×720 | 最大误差 0 ✅ | 1.38 ms |
| fused_composite | **1920×1080** | **最大误差 0 ✅**（原来 23% 错） | 2.92 ms |
| rvm_tail | **1280×720** | **最大差 0 ✅**（原来 27.7% 错） | NPU 2.80 ms，总 5.84 ms |
| rvm_tail | 1920×1080 | 最大差 0 ✅ | NPU 5.53 ms，总 13.2 ms |

不需要 README 里计划的「拆成多遍」。

### 顺带发现的第二个编译器 bug（ICE）

`widen_u8(acc.to_vector<uint8_t>(6))` 再 `store_v` 一个 int16 向量（探针 `-DPV=8`）会让 clang 段错误：
`AIE2InstructionSelector::canCombineUNPACKLoad` → `llvm::getLoadStoreSize`（selectG_AIE_LOAD_UNPACK）。
它把一个**不是来自 load** 的 unpack 当成 load+unpack 来合并。确定性复现，纯编译即可：

```bash
source npu/env.sh
$PEANO_INSTALL_DIR/bin/clang++ probe.cc -c -o /dev/null -O2 -std=c++20 --target=aie2-none-unknown-elf \
  -I$MLIR_AIE_INSTALL_DIR/include -I$MLIR_AIE_INSTALL_DIR/include/aie_kernels -Inpu/lib \
  -D__AIE_API_AIE_ADF_HPP__ -DNDEBUG -DPV=8 -DPLS=704
```

### 最小复现方案 / 是否提 issue

已经做完了（`aie-compiler-bug/probe.cc` + `probe.py`）：

- 一个内核、一路输入 `[A int16×LS][B int16×LS][I int16×W]`、一路输出，`RowPipeline(n_cols=1, height=4)`；
  `-DPV=` 选变体，`-DPLS=` 选平面偏移，宽度（=迭代次数）作为运行时标量，同一份内核目标码跑 4…80 次迭代。
- 变体结论：`V0` 只放大 A ✅；`V11` 只放大 B ❌；`V5` upA+I ✅；`V6` upB+I ❌；`V7` upA·I ✅；`V12` = V11 + volatile 偏移 ✅。
  → 与 acc 规约、MAC 链长度、uint8 转换都无关，只跟「哪个平面 + 偏移多大」有关。
- 宿主侧还给出了「错误值 = 用 A 平面的 pad[j] 代替 B 平面 pad[j]」的逐元素解释（`✔ 解释` 那段）。

**值得提，而且应该提到 llvm-aie（Xilinx/llvm-aie），不是 mlir-aie**：静默算错、确定性、只要编译就能从 `.s` 里看出来
（`vlda wl6, [p2], #1056` 读错址）。issue 里只需要：一个 ~30 行的独立 .cc（去掉 IRON，只留函数）、编译命令、
PLS=496 vs 512 的两段汇编、版本 `llvm-aie 21.0.0.2026062301 (cb664e8c)`。ICE 另开一个。
提之前先在仓库里搜 `post-increment` / `PostInc` / `canCombineUNPACKLoad` 去重，并在最新 nightly 上复测一次。
（对外发帖需要用户自己确认/操作。）

---

## 1. 现状：CPU 时间还花在哪里（720p，RVM@780M + NPU 合成，13 ms CPU/帧）

单独微基准（`.venv`，OpenCV 默认线程；多线程会让 CPU 时间略偏高，只看量级）：

| 步骤（CPU） | CPU ms/帧 | 能否上 NPU |
|---|---|---|
| MJPG 解码 720p（合成图，真实画面更多） | ~2–4 | ❌ 霍夫曼串行；可走 780M VCN/VA-API |
| 720p→320×180 INTER_AREA（RVM 输入、LowResPrep **各做一次**） | 0.74 ×2 | 可以，但先在 CPU 上**合并成一次**就省 0.74 |
| LowResPrep（导向滤波系数 + 无光晕模糊，320×180） | ~2.7 | 可以（二维模板，中等难度） |
| a/b 定点化 | ~1.0 | 随 LowResPrep 一起走 |
| aq/bq/bg 垂直插值到 720 行 | ~2.5 | ✅ **很适合**：DMA 行重复 + 内核里做垂直权重 |
| BGR→BGRA | ~0.2 | ✅ 并进内核（读 BGR 三字节） |
| BGRA→YU12 + tobytes（虚拟摄像头输出） | ~0.9 | ✅ 并进内核（直接输出 I420） |

结论：**目前 CPU 上还剩 ~3.5 ms/帧的「全分辨率或接近全分辨率的逐像素搬运」**，形状正好是 NPU 已经证明能赢的那一类
（融合、整数、流式、不回 CPU）。这是比任何新模型都更确定的 NPU 收益。

---

## 2. 候选功能清单

计算量都按 720p30 估。「骨架」指 `RowPipeline` + `row_ops.h` + `fused_composite.cc`。

### A. 合成内核吃掉剩下的逐像素活（垂直插值 + I420 输出 + BGR 输入）——**新提出，最推荐**

- 做什么：低分辨率行不再由 CPU 垂直插值到 720 行；由 DMA 用 stride-0 重复把第 j、j+1 两行低分辨率数据送给 4 个输出行，
  内核里做垂直 ×4 权重（与水平 up4 同一套 /8 权重，可分离，结果与现在逐字节一致）。
  输出直接写 YU12（Y 每行，U/V 每两行一次 → 内核改成一次处理两行），输入直接吃 BGR 三字节（32 像素 = 96 B）。
- 为什么适合：纯逐像素、整数、能完全融合进已经 1.4 ms 的那一遍；数据量反而下降
  （低分辨率输入 ÷4；输出 4 B/px → 1.5 B/px，720p 输出 3.7 MB → 1.4 MB/帧）。
- 计算量：每像素多 ~6 次 MAC（垂直）+ ~9 次 MAC（RGB→YUV），720p ≈ 0.4 GMAC/s，远低于 4 列能力；NPU 时间预计 1.4 → ≤2 ms。
- 复用：骨架几乎全复用；需要给 `RowPipeline` 加「输入行可重复 / 每次取 2 行」的 tap（`TensorAccessPattern` 本身支持 stride 0），
  和「一次调用处理 2 行」的模式。
- 风险：I420 的 U/V 需要 2×2 平均 → 内核一次两行，栈/寄存器压力上升（记得 `stack_size`）；
  又碰上同一个编译器 bug（**所有平面指针都用 volatile 偏移**）；DMA 重复模式要在 IRON 1.3.4 上先试通。
- 预期收益：CPU −3 ~ −3.5 ms/帧（13 → ~9.5 ms，约 −25%）。

### B. 离开时关闭画面（在场检测）

- 做什么：用**已经有的** 320×180 遮罩算前景占比/重心，连续 N 秒低于阈值 → 输出黑屏或静帧；
  **同时把 RVM 降到 1–2 fps、跳过 NPU 合成**；回来后恢复。
- 为什么（不）适合 NPU：检测本身零成本（遮罩求和，0.01 ms），不需要新模型，更不需要 NPU。
  但它是**最大的实际省电手段**：人不在时 GPU/NPU/CPU 都能停下来。
- 风险：阈值/滞回调参（背光、人只露半身）；RVM 有循环状态，降频后回来第一帧要重置状态或多跑几帧。
- 结论：功能价值高、工作量 ~1 天，但它不是「NPU 功能」。`FEATURES` 里可以标为可用，描述别写「在 NPU 上」。

### C. 自动构图（人脸检测 + 裁剪）

- 做什么：人位置**同样直接用遮罩的外接框/重心**（不需要人脸检测模型），平滑后决定裁剪窗口。
- 最省的实现：摄像头开 1080p MJPG，NPU 合成内核从 1080p 帧里**按 DMA 偏移取一个 1280×720 窗口**
  （裁剪 = 改 `rt.fill` 的 tap offset/stride，零计算），输出仍是 720p。等效 1.5× 数码变焦 + 平移。
  任意缩放比需要逐像素小数重采样（gather），AIE 不擅长，不建议放 NPU。
- 风险：1080p MJPG 解码 CPU 约 ×2.25（+2–4 ms/帧），可能把 A 省下来的又吃回去 → 应和 VA-API 解码一起考虑；
  遮罩/低分辨率准备也得按裁剪窗口算；平移要做阻尼，不然晕。
- 结论：产品价值中等，NPU 部分很小（DMA 窗口），主要成本在解码。排在 A、B 之后。

### D. 通话降噪（RNNoise 类）

- 计算量：RNNoise 网络每 10 ms 帧 ≈ 90k MAC（≈ 9 MMAC/s），加上 FFT/基音分析；CPU 上约 1% 单核。
- 为什么**不适合** NPU：每 10 ms 一次调用 = 100 次/秒同步，每次 0.3–1 ms 的同步开销本身就比计算贵；
  驱动保持唤醒 → NPU 永不休眠；还要另开一条 PipeWire 音频链路。
  只有换成大得多的模型（DeepFilterNet 级，~0.3–0.5 GMAC/s）并且能常驻流式（NPU 侧循环 + 环形缓冲，不每帧同步）才有意义，
  那是新一套基础设施。
- 结论：功能值得做，但**用 CPU**：PipeWire filter-chain + 现成的 RNNoise LADSPA 插件（noise-suppression-for-voice），
  GUI 里加开关即可。NPU 版不排期。

### E. RVM 尾部融合内核（`rvm_tail.cc`）

- 现状：**已被上面的绕过修好**（1280/1920 逐字节一致），不需要多遍拆分。
- 但它的价值是**画质**（RVM 自己的 refiner：前景去色溢 + 学习到的边缘），不是省 CPU：
  - 需要换一份带 refiner 的 RVM 部署（全分辨率 src 进模型、ratio=0.25），GPU 侧要输出 2×4 通道的 1/4 分辨率张量；
  - 现在 tail 的主机侧要垂直插值 **8 个平面**（selftest 里主机 ~3 ms/帧），CPU 比现在的导向滤波路径更多。
- 结论：等 A 的「垂直插值上 NPU」做完再接；那时 tail 的主机侧只剩定点化，才划算。

### F. 给 NPU 写一个高效 3×3 卷积（替代上游 `conv2dk3`）

- 目标是把 RVM 解码器（占 80% 时间）从 GPU 挪到 NPU，让 780M 能睡。
- 门槛：Conv_290（0.29 GMAC）现在 10.8 ms（~27 GMAC/s，≈ 峰值 0.5%）；能量打平线 6.7 ms；
  要让整个解码器（估计 ~1–1.5 GMAC/帧）在 33 ms 内跑完并且比 GPU 省电，单层效率得到 **≥ 4–10×**。
- 难点：权重 + 激活在 4×4×64 KB 里放不下要分块流式；int8 量化 RVM（ConvGRU 循环状态的量化漂移是真风险）；
  还有 upsample、concat、GRU 的整套图要手写调度。工作量以月计。
- 结论：研究项目。若做，先只做一个独立基准：自写 conv3x3（宽度不写死、按输出通道 16/32 分组、权重常驻、行流式），
  **验收线 Conv_290 ≤ 2.5 ms 且逐字节对 numpy**；过不了就停。另外在决定之前先用 PPT 量清楚「780M 跑 RVM 的边际瓦数」——
  如果只有 1–2 W，这条路的上限本来就不大。

### G. 其他可选方向（低成本加在合成内核上）

- **画面增强**（自动白平衡 / 补光 / 轻度美颜）：每像素 3×3 颜色矩阵 + 增益，或用遮罩只提亮人像，
  直接并进 A 的那一遍，NPU 增量 < 0.2 ms，CPU 零增量。适合作为「NPU 免费送的效果」。
- **1080p 支持**：绕过已证明 fused 1920×1080 逐字节一致（2.9 ms）；但 1080p 下 CPU 侧解码/准备也变大，
  产品上仍建议 720p 默认，1080p 作为可选。
- **CPU 小修**（非 NPU，顺手）：RVM 输入和 LowResPrep 的 720p→320×180 缩放合并成一次（−0.74 ms）；MJPG 改 VA-API 解码。

---

## 3. 优先级

| # | 项 | 类型 | 工作量 | 收益 |
|---|---|---|---|---|
| 0 | 把 volatile 偏移绕过落进 `fused_composite.cc` / `rvm_tail.cc`，更新 README 结论，1920×1080 进 `VERIFIED_SIZES`；向 llvm-aie 提 issue（2 个） | 修复 | 0.5 天 | 解锁 1080p、解锁 tail |
| 1 | **A：合成内核吃掉垂直插值 + I420 输出 + BGR 输入** | NPU | 3–5 天 | CPU −25% |
| 2 | B：在场检测 + 离开时整条流水线降频 | 产品逻辑 | 1 天 | 离开时省电最多 |
| 3 | G：画面增强并进合成内核 | NPU | 1–2 天（A 之后） | 新效果、零 CPU |
| 4 | C：自动构图（遮罩框 + DMA 裁剪窗口） | 产品 + DMA | 2–3 天 | 功能；需配合硬件解码 |
| 5 | LowResPrep（导向滤波系数/模糊）上 NPU | NPU | 1 周 | CPU 再 −2.7 ms |
| 6 | E：接入 RVM refiner + tail | NPU + 模型 | 1 周（A、5 之后） | 画质 |
| 7 | F：自写 conv3x3 基准 | 研究 | 数周起 | 不确定 |
| — | D：降噪用 CPU RNNoise 做，NPU 版不做 | 产品 | 1 天 | 功能 |

## 4. 推荐的下一步与成功标准

**第一步（#0，很小，建议立刻做）**
- 改动：两个内核的平面指针改成 volatile 偏移（并在 `row_ops.h` 头注释写清楚规则：所有「基址 + 常量平面偏移」都要这样）；
  README「分辨率」「RVM 尾部融合」两节改成真实根因；`VERIFIED_SIZES` 加 `(1920, 1080)`。
- 成功标准：`rm -rf ~/.npu/cache/*` 后，`python -m npu.fused --selftest` 在 1280×720 与 1920×1080 最大误差 0；
  `python -m npu.rvm.tail` 在两种尺寸最大差 0；`studio --npu` 720p 实机跑 2000+ 帧耗时不回退（NPU ~1.4 ms）。
- issue：用户确认后发到 Xilinx/llvm-aie（材料在 `aie-compiler-bug/`）。

**第二步（#1，A）**
- 成功标准：
  - 正确性：新的 numpy 参考（垂直+水平可分离 ×4、BT.601 I420）与 NPU 逐字节一致，1280×720 和 1920×1080 都跑 selftest；
    再与现有 CPU 路径的 YU12 输出比 PSNR ≥ 45 dB（垂直插值的取整与 cv2 不同，允许 ±1）。
  - 速度：NPU 每帧 ≤ 2.0 ms（含同步）；端到端仍 30 fps。
  - CPU：引擎统计（主进程 + worker）13 ms/帧 → **≤ 10 ms/帧**，同一场景 ≥ 2000 帧。
  - 功耗：`python powerbench.py --backends npu --secs 60` 改动前后各 ≥ 2 轮，看整包 PPT；
    预期只是小幅下降（~0.3–0.8 W 量级），**不能比改动前高**；电池读数不用。

---

## 更正（同日，单线程复测）

引擎实际跑在 `--cv-threads 1`，上面第 1 节的微基准用了 OpenCV 默认 16 线程，CPU 时间被放大了。单线程复测：

| 步骤 | CPU ms/帧 |
|---|---|
| aq/bq/bg 垂直插值到 720 行 | 1.03 |
| a/b 定点化（numpy） | 0.78 |
| BGRA→I420 | 0.25（+ tobytes 0.02） |
| BGR→BGRA | 0.10 |

所以项 A（垂直插值 + I420 + BGR 输入）只能省 **~1.4 ms/帧，约 13%**（实测总 CPU 10.8 ms/帧），不是 25%。
建议拆小：只做「垂直插值上 NPU」（ObjectFifo 3 行滑动窗口，输出仍是 BGRA，预览不受影响），~1 ms；
BGR 输入（0.1 ms，三路解交织在 AIE 上别扭）不做；I420 输出（0.25 ms）要连带改预览和输出 DMA 布局，不划算。
更大的一块是 LowResPrep（~2.7 ms）+ 定点化（0.8 ms）：和垂直插值一起上 NPU 才是 ~4.5 ms/帧量级的收益。

## 准备上 NPU 的实测（同日晚）

- 上限实验（NPU_PREP_ORACLE，同进程 15 s 交替、16 对）：去掉全部 CPU 准备，CPU 11.3→6.8 ms/帧，PPT −1.5 W（中位 −1.9，SE 0.5）。机器是 performance governor/EPP/platform_profile。
- 已实现：垂直 ×4 上 NPU（fusedCompositeLineV，3 行滑窗 ObjectFifo），720p/1080p(3 列) 逐字节一致；但只省 ~0.3 ms CPU（cv2 垂直 resize 实际 0.27 ms，之前 1.03 ms 的微基准偏高），NPU +0.4 ms。
- 桌面负载下微基准很不稳（同一函数前后差 3×），只信同进程 A/B。
- 剩下的大头：blur_bg（同进程对比可用 4 通道合并 + uint8 plain 模糊减半，差 ≤1/255）、720p→320×180 缩放 0.73 ms（RVM 也做一次，可共享）、导向滤波 0.37 ms。定点化实际只有 ~0.05 ms。
- 建议：先做 CPU 精简（共享缩放 + 精简模糊），用交替法量 PPT；剩下的若还 >0.5 W 再考虑导向滤波+模糊整段上 NPU（~1 周）。

## CPU 精简（已实现，未提交）
- 共享 RVM 的 320×180 缩放 + 背景模糊在 160×90 上做：准备 4.8→3.0 ms/帧；同进程交替 16 对 PPT −2.7 W (SE 0.16)，A/A 对照 −0.57±0.29 W。
- "4 通道合并的精简模糊"实测反而更慢 (2.27 vs 1.96 ms)，已放弃。
- 剩余 CPU 准备 ~3 ms：导向滤波 0.37、定点化 ~0.05、BGR→BGRA 0.1、其余是 numpy 杂项/同步。再往 NPU 搬的边际收益已经不大。

## 自写 conv3x3（已实现，未提交）
- npu/kernels/conv3x3.cc + npu/conv3x3.py：Conv_290 逐字节一致，4 core 0.95–1.2 ms（旧 10.8 ms，CPU ORT 2 线程 4.1 ms）。
- powerbench_conv：NPU 17 mJ/次 vs CPU 127 mJ/次，NPU 省 86%。2.5 ms 验收线以 2× 余量通过。
- 8 core 撞 shim DMA 通道上限；1×4 切分超 64 KB。扩展需 memtile 分发。
- 限制：OC=16 写死；IC%8==0；bias 走常数通道。

## RVM 整网 int8 精度（CPU 模拟，未改仓库）
- 780M 跑 RVM 的边际功耗：同进程交替 16 对，+3.18 W（中位 +2.98，SE 0.52），CPU +1.8 ms/帧。
- ORT QDQ 静态量化（opset 升到 13，per-channel 权重，对称 int8 激活，MinMax，用前 300 帧的偶数帧 + 真实循环状态校准），闭环 597 帧：
  - 全部算子 int8：有人帧 MAE 0.0062、IoU 0.969、>0.05 的像素 3.4%；空场景 MAE ~0.0013，600 帧内无漂移。
  - 只量化 Conv：几乎一样（MAE 0.0059、IoU 0.968）→ 误差主要来自卷积本身量化，非卷积算子用 int8 不额外伤精度。
  - 对比：现有 ncnn fp16 部署的 pha 平均差 ~5e-4，int8 约差一个数量级，但 IoU 0.97。
- 局限：录像很暗（亮度 27/255），人只在前 ~80 帧；校准帧也在测试序列里。Percentile 校准内存不足被杀。
- 结论：没有循环漂移这个致命风险；单帧误差集中在边缘，需要在正常光照、人持续在画面里的录像上再验一次并肉眼看边缘。
