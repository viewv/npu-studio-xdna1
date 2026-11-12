# RVM 整网上 NPU：描述符驱动的 core 层循环（desc-loop spike）

日期 2026-11-11，分支 `phase2-npu`，未提交。接 `rvm-npu-hetero-spike.md`。
代码：`npu/chain_hetero.py` 加了 `MODE` / `--mode {class,static,dyn}`，默认 `class`；内核不变。
附 `rvm_convs_180x320.tsv`：RVM 85 个卷积在工作分辨率 320×180 下的实测形状表（onnxruntime 跑出来的）。

## 结论（一句话）

**程序内存问题解决了。** 用"按类别分派"：每个（每 core 行数, stride）类别在 core 上只有一段静态行循环，
每层从权重头读一个 0/1 选择字节，当作这段 `range_` 的运行时次数。这样 **core 代码大小只取决于类别数，与层数和层的顺序无关**：
3 个类别 `.text` = 6672 B，3 层到 85 层都不变。85 层链逐字节一致，每层额外开销 **0.043（最快）– 0.064（中位）ms**，
跟按层组静态展开的写法没有区别（分派本身看不出代价）。

## 三种写法的对比

| MODE | core 循环 | 结果 |
|---|---|---|
| static | 相邻同（行数, stride）的层合成一组，每组一段 | 能跑；每组多 0.7–1.5 KB，约 10 组就满 16 KB |
| **class** | 每个类别一段静态行循环，每层用描述符字节选段（`range_(0/1)`） | **能跑，代码大小恒定**（≈ 3.3 KB 内核 + 每类约 0.75 KB） |
| dyn | 行数也从描述符读（`range_(rows)`），只按 stride 分两段 | **编译器拒绝**：`cannot statically analyze cyclostatic acquire pattern: acquire/release is inside a conditional` |

→ ObjectFifo 的滑窗 acquire/release 只能放在**编译期次数**的循环里（下降要静态推算持有数）；
外层用运行时 0/1 次数包起来没问题，因为段内进出持有数都是 0（s1 段末尾 release 2，s2 段每行 release 3）。
**所以"每段行数是编译期常量、类别在运行时选"是这条路上唯一能走的写法。**

core 里读描述符：`memref.load(wv, [const])` → `arith.index_cast` → 直接给 `range_`
（`aie.helpers.dialects.scf._for` 接受 `Value`）。

## 数字（7840HS，class 模式）

| 链 | 层数 | 中位 ms | 最快 ms | 主机 CPU ms/次 |
|---|---|---|---|---|
| A,A,B | 3 | 1.76 | 1.44 | 0.44 |
| A,A,B,D,C,D | 6 | 2.25–2.53 | 2.08 | 0.5 |
| A,B,(C,D)×16 | 34 | 7.75 | 7.03 | 0.78 |
| A,B,(C,D)×32 | 66 | 13.64 | 12.68 | 0.74 |
| A×20,B,(C,D)×32 | 85 | 20.21 | 19.59 | 0.69 |

- 拟合 A,B,(C,D)×K（K=0..32）：T = 0.93 + 0.194·N（中位），0.64 + 0.184·N（最快）；static 模式是 0.201·N，一样。
- 85 层 reps 拆分：纯计算 15.5 ms，截距 5.69 / 3.92 ms → 每层额外 (5.69−0.25)/85 = **0.064**，最快 (3.92−0.2)/85 = **0.044**。
- 主机 CPU 每次调用 0.4–0.8 ms，基本不随层数涨（比同形状实验的 0.15–0.35 高，这几轮机器负载 ~3，可能偏高）。

## RVM 真实形状表（320×180 输入，`rvm_convs_180x320.tsv`）

总计 1.84 GMAC。按核类型：3×3 普通卷积 14 层、73% 计算；1×1 52 层、25%；depthwise 3×3 13 层 + 5×5 6 层（含 stride 2、dilation 2）共 2%。

按（输出分辨率, stride）：

| 输出 | s1 层数 | s2 层数 |
|---|---|---|
| 180×320 | 10 | – |
| 90×160 | 6 | 1 |
| 45×80 | 8 | 1 |
| 23×40 | 11 | 1 |
| 12×20 | 29 | 1 |
| 1×1（SE 全连接） | 17 | – |

→ 类别数 ≈ 9（不算 1×1 的 SE 层）→ 内核 + 9×0.75 KB ≈ 10 KB，**放得下**；但内核本身还要加 depthwise / 5×5 / 空洞的分支，会变大，
16 KB 要精打细算（可以考虑 depthwise 用单独的 core 或单独的调用）。

## 这次暴露出来、下一步要解决的问题

1. **奇数行数和小宽度切不开。** 45、23、12 行没法均分成 2 段；宽 40、20 没法按 2 条 × 8 的倍数切。
   要么不同 core 走不同行数（每个 Worker 本来就是独立程序，可以给 4 个 core 传不同的类别表），
   要么低分辨率层只用 1–2 个 core（它们计算量很小）。
2. **输出块大小必须所有层相等**（hetero spike 的限制 2）还没解决：OC 4–960、宽 20–320，OCB·WT 不可能凑成一个数。
   需要"一行输出拆成若干个固定大小的块"（内核每次产出一块，core 每行 acquire 多次，块数是类别的一部分）。
3. **17 个 SE 全连接层（1×1 分辨率）+ GlobalAveragePool + HardSigmoid·Mul**：整张图的归约，不适合行流水线，
   需要单独的算子形态（或者把 SE 的缩放融进下一个 1×1 卷积的权重读取）。
4. **其他算子**：HardSigmoid/Mul（hardswish）27 处、Add 16、Concat 23、Resize 7、ConvGRU（Sigmoid/Tanh）。
5. 通用内核只有专用版 61% 的速度（hetero spike），3×3 占 73% 计算 → 优先给 3×3 做特化。

建议顺序：先做 **2（固定块输出）+ 1（每个 core 自己的类别表）**，因为它们决定 fifo / 描述符格式，
之后拿 RVM 编码器前几层（Conv_8 … Conv_36：3×3 s2、dw3×3、1×1、dw5×5 s2）做第一个"真实层"的逐字节验证。

## 复现

```bash
source npu/env.sh
python -m npu.chain_hetero --mode class --spec A,A,B,D,C,D       # 默认就是 class
python -m npu.chain_hetero --mode dyn --spec A,B,C,D             # 看编译器拒绝的报错
python -m npu.chain_hetero --compute A,B,C,D --mode class
```
