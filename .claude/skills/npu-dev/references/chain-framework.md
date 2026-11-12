# 整网串联框架 `npu/chain_c16.py` 设计细节

对应内核 `npu/kernels/conv_c16.cc`（入口 `convC16Chunk`）。下面的常量都在 `chain_c16.py` 顶部。

## 1. 为什么是这个形状（每条都是被逼出来的）

| 约束 | 后果 |
|---|---|
| ObjectFifo 元素大小是编译期的；每个 compute tile 只有 2 进 2 出 DMA 通道 | 不能给每种层形状各开 fifo → **所有层共用固定大小的元素** |
| core 程序内存 16 KB | 不能按层或按形状展开代码 → **core 循环由描述符驱动，代码与层数和形状无关** |
| 运行时次数的循环里 acquire/release 必须平衡 | 循环结构只能有少数几种固定形态（s1 / s2 / 累加 / 收尾） |
| 输出块会被 drain 原样写进 DDR | **输出不能带垃圾** → 输出块必须是 DDR 布局里连续、整齐的一段 → 16 通道像素布局 |
| 输入可以多读 | 输入元素统一 7680 B，层自己用多少是多少 |

## 2. 激活布局

`[H + 2·BR][CP][PAD + Ws + PAD][16]` int8/uint8（CP = C/16，BR = 4，PAD = 4）。

- 每个像素 16 个通道连续，一个"通道组平面"一行 = `cplane = (Ws + 2·PAD)·16` 字节，一行 = `frow = CP·cplane`。
- 上下 BR 行、左右 PAD 像素是零，**从不被写**。每个中间张量在 scratch 里有自己的槽（`Plan.loc`；以前按形状乒乓，
  有了分支以后改掉了）。不同形状不能共用槽（上一层写的内部像素会落到下一种形状的"边"上）。
- 图：张量 0 是输入，层 k 的输出是张量 k+1；层 k 读张量 `k+1+src`（默认 src=-1 = 上一层），
  SE 的投影层另外读缩放张量 `k+1+sc`（C×1×1）。参考实现 `make_case` 按同样的规则连。
- **通道拼接**（`Layer.cat`）：层 k 读 `[张量 k+1+src ‖ 张量 k+1+cat]`。Plan 把 B 放进 A 的存储（`Tensor.ctot` = 合计通道，
  `Tensor.g0` = B 的起始组；`frow` 按 ctot 算），写 B 的层直接写到位，读的层用一个 `ctot` 宽的视图（`Plan.tin[k]`）。
  限制：两个张量 (h, w, ws) 相同；一个张量只能在一个拼接里；输入 / 输出张量不能参与。
  残差加就是拼接 + 投影层里残差那部分的权重设成 `RES_W·I`（64），**内核不用改**；SE 只乘前 `h[48]` 个组（后面是残差）。
- 存储宽度 Ws = 生产者的 `n_tw · WT`（WT 是 40 的倍数），可能大于真实宽度：12×20 存成 40 宽。
  内核把 ≥ 有效宽度的输出像素写 0，所以补出来的列对下一层就是零填充。
- 层间约定：内核输出 uint8（饱和 = ReLU），下一层把同一字节当 int8 读（实验约定；接真实模型时换成 per-tensor 零点）。

## 3. 三个 fifo 和每 core 的内存

| fifo / 缓冲区 | 大小 | 深度 | 说明 |
|---|---|---|---|
| 输入 `c16_{c}i` | `E_IN` = 7680 B = `[ICP_pad][PLANE]` 或 dw 的 `[k][m][PLANE]` | 4 | 每块 PLANE 字节里前面是真实像素，后面是 DMA 顺带读的垃圾 |
| 权重 `c16_{c}w` | `e_w`（= 128 B 头 + 最大一步的权重，补到 256 的倍数，目前约 18.3 KB） | 1 | 一个元素 = 一步 |
| 输出 `c16_{c}o` | 640 B = 1 个通道组 × CW(40) 像素 | 2 | 一行输出 = 若干个块 |
| `c16_{c}acc`（Buffer） | `ACC_CHUNKS`(3) × 640 个 int32 = 7.5 KB | – | IC 分块的部分和 |
| `c16_{c}dummy`（Buffer） | 640 B | – | 累加步没有输出时给内核的占位 |
| `c16_{c}sc`（Buffer） | 1024 B | – | SE 缩放向量（≤ 64 个通道组），层前面的"装缩放"槽装、本层各步用 |

合计约 58.5 KB / 64 KB。权重 fifo 改深度 2 实测**没有收益**（瓶颈不在权重预取），还会让 3×3 128 通道的层放不下。

## 4. 权重头（256 B）

```
int32[0..7]   ICP  OCP  WT  stride  k  shift  NXC(=WT/40)  PLANE
bytes 32+16c  core c 的一段, 8 个 u16:
              [0] rows  [1] sel_s1  [2] sel_s2  [3] wvalid  [4] 每行块数 nch  [5] sel_acc  [6] sel_fin  [7] 0
int32[24]     first   (IC 分块: 本步从零开始累加)
int32[25]     last    (IC 分块: 本步结束量化输出)
int32[26]     dilation
int32[27]     depthwise 标志
int32[28]     depthwise 输入元素的行步长 (= m·PLANE)
int32[29]     输出激活 (0 ReLU, 1 HardSigmoid, 2 线性 int8, 3 hardswish)
int32[30]     SE: 这一步的 1×1 权重要乘 sc (内核做完把它清 0, 所以 --compute 的重复调用也只乘一次)
int32[31]     SE: 这一步输入通道组的起点 (sc 里从这组开始)
int32[33]     池化 (1 = 累加步, 2 = 收尾步)
int32[34..39] 池化: H  kr(每元素行数)  X(元素里的行步长)  nvec(每行 64 B 向量数)  M  S
int32[40..41] 缩放张量: 每个元素的通道组数 (= E_IN / 768)、组步长 (768)
int32[42]     本步是"装 SE 缩放向量" (走累加段, r = 第几个元素)
int32[43]     池化每组元素数 ne
int32[44..46] hardswish 的 q3 = 3·HS_Q, q6 = 6·HS_Q, s2 = HS_S
int32[47]     池化输入是 int8 (前一层是线性 / hardswish)
int32[48]     SE 只乘前几个输入通道组 (拼进来的残差不乘)
int32[49..50] 重新量化 s1 s2: y = sat8((sat16((acc + 2^(s1-1)) >> s1) · M[c] + 2^(s2-1)) >> s2)
int32[52]     重新量化表在权重区里的偏移 (= 这一步的权重字节数)
int32[53..54] hardswish 第二次重新量化的 t1 t2
```

**重新量化表**（每个输出通道组 `RQB` = 576 B，跟在这一步的权重后面，主机侧 `rq_table` 已排成内核要的布局）：

| 字节 | 内容 |
|---|---|
| [0, 256) | bias int32：mmul（3×3 / 1×1）= 低 8 通道 `[4 像素][8]` + 高 8 通道 `[4 像素][8]`；dw = `[4 像素][16]` |
| [256, 384) | M int16，同样的布局 |
| [384, 576) | hardswish 的 q3 / q6 / M2，各 int16 `[2 像素][16]` |

`W_MAX` 包括重新量化表（18432 + 576）；`_step_wbytes` 和各模式选 nocp 时都把 `RQB` 算进每个输出通道组。
层参数是 `LParam`（W、s1、s2、bias、M、q3、q6、M2、t1、t2）；随机权重的测试用默认值（bias 0、M 1、s1 = shift、s2 0）=
"只移位"，`--rq` / `selftest(rq=True)` 用随机 bias 和 M 测完整路径。

`ICP`/`OCP` 是**这一步**的输入/输出通道组数，不是整层的。core 只读自己那一段（`hc = 32 + 16·c`）；空闲槽是全零头（所有 sel 都是 0）。
权重缓冲区本来就是每 core 一份（`[槽][core][e_w]`），所以"全局"字段其实也可以按 core 不同（n_pre 就是）。
HDR 从 128 涨到 256 没多花内存：e_w 本来就补到 256 的倍数（18432 + 128 → 18688）。

## 5. core 程序（`make_core`，所有 core 一样，只有 core id 不同）

```
for _ in range_(n_steps):                       # 编译期常量 = 整条链的槽数
    wv = f_w.acquire(1); rows = u16(hc); nch = u16(hc+8)
    for _ in range_(sel_s1):                    # stride 1 滑窗 (3×3 和 full/oc 的 1×1)
        f_in.acquire(2)
        for r in range_(rows):  win = acquire(3); for j in range_(nch): out acquire/kernel/release; release(1)
        f_in.release(2)
    for _ in range_(sel_s2):                    # stride 2: DMA 每个输出行发 3 行
        for r in range_(rows):  win = acquire(3); 块循环; release(3)
    for _ in range_(sel_acc):                   # IC 分块中间步: 只累加, 输出给 dummy
        for r in range_(rows):  win = acquire(1); for j: kernel(..., dummy, acc, j, r); release(1)
    for _ in range_(sel_fin):                   # IC 分块最后一步 / depthwise: 出输出
        for r in range_(rows):  win = acquire(1); for j: out acquire/kernel/release; release(1)
    f_w.release(1)
```

不是卷积的步也走这 4 段，由权重头的标志让内核分派：
- 装 SE 缩放向量 = 一个累加段步（rows = 元素数，nch = 1，`h[42]`）；
- 池化 = 一个累加段步（每组前 ne−1 个元素，rows = 组数·(ne−1)）+ 一个收尾段步（每组最后一个元素，rows = 组数），`h[33]` = 1 / 2。
上一版给这两样各加了一段循环，core 程序 3.2 → 4.4 KB；改成这样又回到 3.2 KB，代价是每处多 1 个槽（约 5 µs）。

每段进出时输入 fifo 的持有数都是 0，所以外面套 `range_(0 或 1)` 选段没问题。
内核签名：`convC16Chunk(r0, r1, r2, wts, out, acc, sc, j, r, core)`，j 是行内块号（`ocp = j / NXC`, `xc = j % NXC`）。

## 6. 层模式与切分（`choose_tiling`）

先按空间选切分：宽 t 条 × 高 4/t 段（t ∈ {4,2,1}，取补宽最少的），行数余数分给前几段（23 → 6,6,6,5）。
有效宽度 `we = min(WT, ceil8(min(wo, WT)))`，每块需要 `need = (s·we + 2·PAD)·16` 字节。

| 模式 | 条件 | 步 | 输入 fill（每 core） | 输出 drain（每 core） |
|---|---|---|---|---|
| `full` | 一行输入放得下（存在 `ICP_pad ≥ ICP` 使 `E_IN/ICP_pad ≥ need`），权重 ≤ W_MAX | 1 | s1: `[1, rows+2, ICP_pad, PLANE]` 步长 `[0, frow, cplane, 1]`；s2: `[≤64, 3, ICP_pad, PLANE]` 步长 `[2frow, frow, cplane, 1]` | `[n_ob, rows, nocp, WT·16]` 步长 `[nocp·cplane, frow, cplane, 1]` |
| `oc` | 一行放得下，权重 > W_MAX（18 KB） | n_ob 块，每块 nocp 个输出组 | s1 最外维 `n_ob`、**步长 0**（同一份输入重读，仍是一个 BD）；s2 每块单独 fill | 同上，一个 BD |
| `oc` + **OC 并行** | 上面 + 宽只有一条带（n_tw = 1）+ s1 | 每槽每 core 一块，core c 拿块 c, c+4, … | `[块数_c, rows+2, ICP_pad, PLANE]`，全部行 | `[块数_c, rows, nocp, WT·16]` 步长 `[4·nocp·cplane, …]`，偏移 `c·nocp·cplane` |
| `ict` | 1×1 且一行放不下 | n_ob 份 × n_icb 步，每步 m 个输入组 | 每份一个 BD：`[n_icb, rows, m, PLANE]` 步长 `[m·cplane, frow, cplane, 1]`（不带上下边行） | `[n_ob, rows, nocp, WT·16]` |
| `dw` | depthwise | 每步 m 个通道组（`k·m·PLANE = E_IN`） | 每步每 64 行一个 BD：`[行, k, m, PLANE]` 步长 `[s·frow, d·frow, cplane, 1]`，起点 `BR + s·y0 − pad` | 每步一个 BD |
| `pool` | `Layer.op = 1` | 2 槽（累加 / 收尾）；core c 拿组 c, c+4, …；每组 ne = ⌈H/kr⌉ 个元素 | 累加步 `[组数_c, ne−1, kr, X]` 步长 `[4·cplane, kr·frow, frow, 1]`；收尾步 `[1, 组数_c, kr, X]`（X = E_IN/kr ≥ cplane；最后一个元素多读的行内核不看） | `[1, 1, 组数_c, 640]` 步长 `[0, 0, 4·cplane, 1]` → C×1×1 张量 |
| FC | 1×1 图（ho = 1） | 按行只能切出 1 段 → oc / ict 都改成 OC 并行 | ict 且 1 行：所有份合成一个 BD `[份数_c, n_icb, m, PLANE]`，最外维步长 0 | 同 OC 并行 |

SE 投影层（`Layer.sc`）：层前面多一个"装缩放"槽（`Tiling.pre` 个元素）；层的第一个 task group 里，每个 core 先排一个缩放向量的
fill（C×1×1 张量从第 BR 行起连续读 `pre·E_IN` 字节，每 768 B 一个通道组），再排正常的输入 fill；`ob_groups` 给第一组少留一个 fill 的位置。
内核每步第一次调用时 `se_wscale` 把 sc 折进这一步的权重：`W' = (W·s + 128) >> 8`。先试过"每行原地乘输入"，
IC 分块的层每份 OC 都要重读、重乘输入，Conv_144 多 0.35 ms；改成乘权重以后只多 0.07 ms。

- `ict` 的 OC 份大小 nocp 受 acc 限制：`rows · nocp · NXC ≤ ACC_CHUNKS`。12×20 每 core 3 行 → nocp = 1 → 输入要按 OC 份数重读（目前最大的性能债）。
- `ict` **切行**：rows · NXC 都放不下（23×40 每 core 6 行）时，每 core 的行再切成 `n_rb` 段（每段 ≤ `rcap` 行），
  份号 o → (OC 块 o // n_rb, 行段 o % n_rb)；每份一个 fill BD、一个 drain BD，只读写自己那段行（`Tiling.sub_rows`）。
  步骤元组多一个第 7 项 = 行段号。
- 同一层按 OC 块/份分成几个 task group，保证每组每个 shim 通道 ≤ `FILLS_PER_TG`(4) 个 fill（`Plan.ob_groups`）。
- 权重缓冲区按 `[槽][core][e_w]` 存（每个 core 一份，非 OC 并行时 4 份相同）；每 core 的权重 fill 是
  `[1, 槽数, e_w/256, 256]` 步长 `[0, 4·e_w, 256, 1]`（跳着读，最内维要 ≤ 1023 字）。

## 7. 内核分支（`conv_c16.cc`）

- 输出：累加器初值 = bias（`bias_init` / dw 的 `b0`）；`store4` / `cvt` → `fin`：`to_vector<int16>(s1)` × M → `to_vector<int8>(s2)`
  → `max(·, lo)`（ReLU 的 lo = 0，其余 −128）。宽度外的像素显式写 0（有 bias 以后不会自然是 0；4 像素组里部分有效时走 `zero_tail`）。
  HardSigmoid（+128、宽度外清零）和逐通道 hardswish 由入口函数最后的 `post_act` 对整个 640 B 块做。
  act 0 现在也是 int8 的 ReLU（0..127）：原来"输出 uint8、下一层当 int8 读"的实验约定已经去掉。
- `conv3s1` / `conv3s2` / `conv1`：`mmul<4,8,8>`。4 像素 × 16 通道的 64 B 向量用 `interleave_unzip(·, 8)` 拆成低/高 8 通道两个 A，
  各喂两个输出通道块；kx 偏移用 `shuffle_up/down_fill(·, 16)`；s2 用 `filter_even/odd(·, 16)` 拆奇偶像素。
  每 8 像素 4 个累加器。整组在有效宽度外直接写 0、不算。`conv1` 支持 IC 分块（first/last、acc）。
- `dwconv<K, D>`（stride 是运行时参数，省程序内存）：逐元素 `aie::mac`（64 路 = 两条 32 路指令）；权重主机侧复制成
  `[组][ky][kx][4 像素][16 通道]`；水平偏移 o = kx·D − pad ∈ [−4, 4] 用常量 shuffle；S = 2 先按 stride 1 算再 `filter_even(·, 16)`。
  `noinline`。实例化了 (3,1) (5,1) (5,2)，加组合要改分派。
- 权重布局（主机侧 `pack_layer`）：3×3 `[ocp][ky][icp][kx][h][ob][ic8][oc8]`，1×1 `[ocp][icp][h][ob][ic8][oc8]`（h/ob = 低/高 8 个通道）。

## 8. 测试入口

```bash
python -m npu.chain_c16                         # e1 / head / enc / low / deep 全部
python -m npu.chain_c16 --spec mnv3 --iters 50  # 一条链, 打印每层切分
python -m npu.chain_c16 --spec Conv_135         # 一层 (R 字典里的名字), 或 --spec Conv_67,Conv_68
python -m npu.chain_c16 --compute deep          # 纯计算 / 额外开销拆分
```

链定义在 `R` / `CHAINS`：`enc`（编码器形状 13 层，旧版）、`low`、`deep`（12×20 高通道段 13 个真实形状层）、`mnv3`（MobileNetV3 主干 27 层，含全部 depthwise）、
`mnv3se`（主干 + 4 个 SE，39 层）、`se39` / `se95` / `se135`（单个 SE 块：投影层分别是 full、IC 分块、960 通道 + FC1 IC 分块）、
`mnv3a`（mnv3se 换成真实激活：第一层和后半段 hardswish、投影层线性）、`mnv3r`（mnv3a + 4 个残差加）、
`se135a`（hardswish 的 dw → 有符号池化）、`res24`（45×80 残差，full 模式）、`res144`（SE + 残差 + IC 分块）。
链名后缀 `~` = 真实激活的版本（`_real`），`r` = 带残差的投影层（`_resid`，残差对在 `_RES` 里）。
SE 块在 `R` 里由 `_se(dw, fc1, fc2, proj, mid, oc)` 生成（`<dw>.pool`、FC1、FC2、`<proj>`）。
`check()` 同时检查真实区域与参考一致、边和补宽列全是零（否则返回 999）。

## 9. RVM 覆盖情况（工作分辨率 320×180，`docs/npu-spikes/rvm_convs_180x320.tsv`）

85 个卷积，1.84 GMAC：3×3 普通 14 层（73% 计算）、1×1 52 层（25%）、depthwise 19 层（2.3%）、其中 SE 的 1×1@1×1 17 层（0.1%）。

| 已能跑 | 还没有 |
|---|---|
| 普通 3×3 s1/s2、1×1（含 OC/IC 分块到 960 通道）、depthwise 全部 5 种、SE（池化 / FC / HardSigmoid / 逐通道乘，8 处）、hardswish、残差加（拼接 + 单位权重） | Concat 23（拼接视图已有，解码器还没接）、Resize 7、ConvGRU（Sigmoid/Tanh）；LR-ASPP 的池化分支（Conv_166，池化已有） |
| 不均分行、补宽、最多 152 步的链 | 真实量化权重 / bias / per-tensor 零点、与 onnxruntime 比精度、接入 studio |

建议顺序：解码器（Concat / Resize / ConvGRU）→ 3×3 特化提速 / IC 分块的输入重读 → 真实权重 → studio。
加东西时盯住程序内存（**15.25/16 KB**）和数据内存（59/64 KB）。
已知小坑：只有一层、而且这层用不满 4 个 core 的链（例如单独跑 OC 并行 3 块的 Conv_92）会报 `Prod endpoint not set`
（某个 core 的输入 fifo 整条链都没有 fill）。和别的层串起来就没事。
