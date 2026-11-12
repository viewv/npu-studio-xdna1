# 踩过的坑（症状 → 原因 → 怎么办）

都是在这台机器、IRON 1.3.4 + llvm-aie 21.0.0.2026062301 上实测出来的。升级工具链后可能变化——
`npu.check` 里的编译器探针和 `chain_c16` 的几条链就是用来发现这些变化的。

## 编译器 / 工具链

### llvm-aie：基址 + 编译期常量偏移 ≥ 1024 B 读错地址
- **症状**：不报错，部分 lane 错（合成内核里 x%4 ∈ {0,1} 的像素错；1280 宽对、1920 宽错）。
- **原因**：非对齐向量加载 `base + CONST`，CONST ≥ 1024 B 时被折成 post-increment 形式（`vlda wl, [p], #off`），先用基址加载、后加偏移。992 B 还是对的。
- **办法**：一行里多个平面的偏移一律用运行时值：`rowops::plane_stride(N)`（`npu/lib/row_ops.h`，volatile 读一次）。`python -m npu.offset_probe` 会报告 bug 还在不在。
- 详细定位过程（汇编对比、探针）在 `docs/npu-spikes/npu-roadmap-analysis.md` 和 `docs/npu-spikes/aie-compiler-bug/`。

### IRON：运行时次数循环里的 acquire/release 不平衡 → 编译失败
- **症状**：`'aie.objectfifo.acquire' op cannot statically analyze cyclostatic acquire pattern: acquire/release is inside a conditional`。
- **原因**：ObjectFifo 下降要静态推算每次 acquire 时的持有数。循环次数是运行时值时，循环体必须"进入时持有 n、出来时还持有 n"。
- **办法**：stride 1 滑窗在循环前 `acquire(2)`，循环体 `acquire(3) … release(1)`，循环后 `release(2)`。
  这样行数、块数都能从描述符读（`range_` 接受 MLIR Value：`memref.load` → `arith.index_cast`，见 `chain_c16._u16`）。

### IRON：`acquire(3)/release(2)` 的 stride 2 滑窗 → 挂死
- **症状**：`ERT_CMD_STATE_TIMEOUT`，只在 stride 2 层前面有 `range_(≥2)` 的层循环时出现（如 A,A,B）。
- **原因**：下降后的行循环没剥第一次迭代，每次迭代都按"再取 3 个"上锁（应为 2）→ 多等 1 个对象。`input_with_addresses.mlir` 里能看到循环体内的 `AcquireGreaterEqual, 3`。
- **办法**：s2 不共享行：DMA 给每个输出行发完整 3 行（TAP `[R, 3, ICP, PLANE]`，步长 `[2·行, 行, 块, 1]`），core `acquire(3)/release(3)`。代价 DMA ×1.5。

### `acquire(1)` 返回元素本身
- **症状**：`'func.call' op operand type mismatch: expected 'memref<7680xi8>', but provided 'i8'`。
- **办法**：`win = f.acquire(1)` 直接传 `win`；`acquire(n ≥ 2)` 才是可下标的列表。

### 栈溢出（core 默认栈 1 KB）
- **症状**：不报错，结果错乱；或者连普通层也挂死。dw spike 里一个只在 stride 2 分支用的 `ACC64 a[4]` + `V64 v[6]`，
  因为函数被内联进入口，让**所有**调用都分了 1.4 KB 的栈帧。
- **办法**：内核里不要放向量/累加器数组，按需重算或分批；大函数 `__attribute__((noinline))`；
  确实要大栈就设 `stack_size`（`RowPipeline(..., stack_size=8192)` / Worker 参数），上限约 16 KB。

### 按引用把累加器传给 noinline 函数 → 栈溢出挂死
- **症状**：`ERT_CMD_STATE_TIMEOUT`，连本来就对的层也挂。SE 时把 HardSigmoid 的存储写成 `store4h(uint8_t*, MM &c0, MM &c1, …)` 并标 noinline。
- **原因**：被引用的累加器必须有内存地址 → 调用方（conv1，被内联进入口函数）把 4 个 1 KB 级的累加器放到栈上，超过默认 1 KB 栈。
- **办法**：在调用方先 `to_vector<int8>(shift)`，把向量按值传进去（`store4h(o, V32 v0, V32 v1, nv)`）。

### 激活在 0 处不为 0 → 补宽列被写脏
- **症状**：`check` 返回 999（边/补宽列不是零），真实区域是对的。SE 的 FC2（HardSigmoid）在 1×1 图上把像素 1..3 写成 128。
- **原因**：内核按 4 像素一组输出，组里超出有效宽度的像素 ReLU 版靠"零输入 → 0"自然是零；HardSigmoid(0) = 128。
- **办法**：这类激活按像素数 nv 把组内宽度外的字节清零（`store4h` 的 nv 参数）。以后加 bias 也一样。

### Peano 不做软件流水 → 逐元素后处理慢
- **症状**：hardswish 的后处理（640 B 一块）每层多 0.015–0.065 ms，折算约 22 周期 / 32 B。
- **原因**：`llvm-objdump -d` 看到循环体里 load（7 拍）→ add / max / min → mul（5 拍）→ srs 串成一条，中间全是 nop。
- **办法**：每轮手动做 4 个互不依赖的向量（定长小数组 + 完全展开的内层 for，SROA 后全在寄存器里；
  反汇编确认没有 `[sp, …]` 访问，否则就违反规则 5）。快约 2 倍。

### 程序内存溢出
- **症状**：编译到最后 `[AIE ERROR] _XAie_LoadProgMemSection(): Overflow of program memory` / `XAie_LoadElf failed with XAIE_INVALID_ELF`，
  JIT 缓存目录是空的（没有 ELF 可看）。
- **办法**：`python -m npu.ksize` 单独编译内核、按函数列大小（链接后 ≈ 合计 + 3.5 KB）。上一次超（17.8 KB）是新的重新量化路径在
  6 个存储点各内联一份、还带着 uint8 / int8 两条分支：去掉 uint8 ReLU、下限改成无分支的 `max(y, lo)`、罕见分支（部分像素清零）
  挪成 noinline、dw 的 stride 改成运行时参数（5 个模板实例 → 3 个），回到 15.25 KB。

### JIT 缓存不认头文件和 Python 模块常量
- **症状**：改了代码结果不变（或者"修好了"其实没跑新代码）。
- **原因**：缓存键看内核 `.cc` 的 mtime 和 `@iron.jit` 的编译参数；`row_ops.h`、`harness.py`、`chain_c16.py` 的模块级常量（`E_IN`、`W_MAX`、`ACC_CHUNKS`…）和 `Plan` 逻辑都不在里面。
- **办法**：改完 `rm -rf ~/.npu/cache/*`（`npu.check` 默认会清）。做对照实验时每换一次常量都要清。

### 资源分配失败
- **症状**：`Error: Resource allocation pipeline failed`。
- **原因**：通常是 core 数据内存超了 64 KB（比如权重 fifo 深度 1 → 2，18 KB × 2）。
- **办法**：先算账（见 chain-framework.md §3），再减深度 / 减元素大小 / 把大权重拆步。

## DMA / runtime sequence

### shim BD 维度上限
- 最高维（迭代维 / d3）≤ 64：`'aie.dma_bd' op Size 3 exceeds the [1:64] range` → 大的行数拆成多个 fill（每 64 行一个）。
- 跳着读（不能当连续段）时最内维 ≤ 1023 个 32 位字：`Size 0 exceeds the [0:1023] range` → 把大块拆成 `[…, n/256, 256]`。
- 纯线性的一段可以很长（几十 KB 的权重元素没问题），因为会被合并成一维。

### 一个 task group 里同一 shim 通道排太多任务 → 挂死
- **症状**：`ERT_CMD_STATE_TIMEOUT`。实测 5 个 fill 没事，7 个以上挂死（像是通道任务队列深度）。
- **办法**：一层拆成多个 task group，每组每通道 ≤ 4 个 fill（`FILLS_PER_TG`），组之间是普通屏障。

### fill/drain 字节数必须和 core 的消费/产出严格对上
- 输入 fifo 元素固定大小 → 每个 fill 的总字节必须是元素大小的整数倍，且等于 core 要 acquire 的元素数 × 元素大小；
  drain 同理（块数 × 640）。少一点就挂死，多一点下一层读到错位数据。
- 输入可以多读（TAP 读超出真实条带的字节），但缓冲区尾部要留余量（`Plan.size` 按所有 TAP 的最远地址算）。

## 量化 / 数值

- 真实权重的量化坑（死通道、hardswish 的 q 溢出 int16、SE 折两次）见 `quant.md`。

- 舍入 `(acc + 2^(s−1)) >> s` = `rounding_mode::positive_inf` + `saturation_mode::saturate`，`to_vector<uint8>(shift)` 一次完成舍入 + ReLU。参考实现必须一模一样。
- 随机权重 ±12 时 shift 要按层自动选（输出均值 ~24），否则 10 层内就衰减成全零，逐字节比较变成平凡的 0 == 0。

## 测量

- 单层计时会骗人（固定成本盖住一切），见 perf.md。
- 机器负载高时中位数能抖 30%，看最快值，或者多跑几遍。
