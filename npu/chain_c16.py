"""RVM 形状的整链实验: 固定大小输出块 + 每 core 自己的行数 + 完全由描述符驱动的 core 循环 + OC / IC 分块

  source npu/env.sh && python -m npu.chain_c16                 # 正确性 + 计时
  python -m npu.chain_c16 --spec enc --iters 50                # 只跑一条链 (见 CHAINS), 打印每层切分
  python -m npu.chain_c16 --compute enc                        # 纯计算 / 额外开销拆分

接 chain_hetero.py (desc-loop spike)。那边留下的问题: (1) 输出 fifo 元素大小要求所有层 OCB·WT 相等;
(2) 45/23/12 行切不成两段相等、宽 40/20 切不成 2 条 8 的倍数; (3) core 代码随 (行数, stride) 类别数增长。
这里的做法:
  · 激活布局改成"16 通道像素" [H+2·BR][CP][PAD + Ws + PAD][16] (上下各 BR=4 行零边) (内核 npu/kernels/conv_c16.cc)。
    一个输出块 = 1 个通道组 × CW (=40) 像素 = 640 字节, 是 DDR 里连续的一段 → 任何层的一行输出都是整数个块,
    drain 的 TAP 不用变 ([行][通道组][WT·16])。
  · 存储宽度 Ws 补到 CW 的倍数 (12×20 存成 40 宽), 内核把超出真实宽度的输出写 0 → 补的列就是下一层的零填充。
  · 每层自己选切分 (宽 t 条 × 高 4/t 段), 行数不能均分时各 core 行数不同 (23 = 6+6+6+5); 用不满 4 个 core 的层,
    空闲 core 这一层什么都不做。每 core 的行数 / 有效宽度 / 选择字节都在权重头里按 core 给。
  · core 循环完全由描述符驱动: 行数、每行块数都是运行时 range_ 边界。
    规则 (desc-loop spike 后续试出来的): ObjectFifo 的 acquire/release 可以放进运行时次数的循环,
    只要循环体相对于进入时的持有数是平衡的。s1 滑窗先 acquire(2), 循环体 acquire(3)/release(1) 就是平衡的;
    s2 每行 acquire(3)/release(3); 块循环 acquire(1)/release(1)。
  · 输入元素固定 E_IN 字节 = [ICP_pad][PLANE], 每层 PLANE = E_IN / ICP_pad (≥ 真实条带宽度), 多读的是垃圾。

"步骤" (一个权重元素 = 一步, core 的外层循环按步走, 每步的权重头说这一步干什么):
  · full  整层一步。
  · oc    OC 分块: 权重放不进一个元素 (W_MAX) 时, 按输出通道组切成 n_wb 块, 每块把同一份输入再喂一遍
          (fill 的 TAP 最外维步长 0 重复 n_wb 次, 还是一个 BD), 输出写到对应的通道组 (drain 一个 BD)。
  · ict   IC 分块 (只用于 1×1): 一行输入放不进一个元素时, 每步只带 ≤ 8 个输入通道组, 部分和存在 core 本地的
          int32 acc 缓冲区里 (Buffer); 中间步 (sel_acc) 不出输出, 每个 OC 子块的最后一步 (sel_fin) 量化输出。
          acc 只有 ACC_CHUNKS 个块那么大, 所以 OC 也要切成 k 个通道组一份 (每份把输入再读一遍)。
  同一层的步按 OC 块/份分成几个 task group (每组每个 shim 通道 FILLS_PER_TG = 2 个 fill)。组之间流水: 先发下一组
  再等上一组 (TG_INFLIGHT = 2, 每个通道在途的任务合计 ≤ 4, 多了会挂死); 层之间才是全屏障, 下一层第一组的权重在屏障之前
  就发出 (W_PREFETCH)。权重元素: 链里没有大 3×3 时用小元素 W_SMALL + fifo 深度 2 (算这一步时预取下一步),
  否则 W_BIG + 深度 1 (Plan 自动选)。
  · OC 并行: 一条带 (宽 ≤ 40) 的 OC 分块层不再按行切给 4 个 core, 而是每个 core 算不同的 OC 块 (c, c+4, …)、
    处理全部行 → 每 core 只拿 1/4 的权重。权重缓冲区按 [槽][core] 存, 每个 core 读自己那份
    (所以 e_w 补到 256 的倍数, fill 写成 [槽][e_w/256][256]: 跳着读时最内维不能超过 1023 个字)。
  · 补宽列不算也不读: 整组 8 像素都在有效宽度之外时内核只写零; PLANE 按有效宽度 (向上取 8) 定,
    12×20 的层每块只要 512 B → 一个输入元素放 15 个通道组, 少了很多 IC 分块。

  · dw    depthwise k×k (k = 3/5, stride 1/2, dilation 1/2): 输入元素 = [k 行][m 个通道组][PLANE], DMA 按
          (s·y + ky·d - pad) 把每个输出行要的 k 行放进一个元素 (TAP [行][ky][组][PLANE], 步长 [s·行, d·行, 组, 1]),
          core 每行 acquire 1 / release 1 (复用 IC 分块的收尾段, core 代码不变); 每步 m 个通道组, 输入不重读。
          张量上下各留 BR=4 行零边 (5×5 dilation 2 要 4 行)。内核按 (k, d, s) 模板展开 (运行时版慢 2–3 倍)。

  · SE  (squeeze-excitation) = 三个普通"层" + 一个带逐通道乘的投影层, 层之间可以有分支 (Layer.src / Layer.sc):
          - 池化 (Layer.op = 1): 按通道组分给 4 个 core (组 c, c+4, …), 每 core 读自己组的全部行, 不用跨 core 归约;
            输入元素 = 一个组的 kr 行, 分两步: 累加步收每组前 ne-1 个元素 (int32 和放 acc[组]), 收尾步收每组最后一个
            并出块; 均值 = (Σ·M + 2^21) >> 22 → C×1×1 张量。装缩放和池化都复用累加 / 收尾段, core 程序不为它们加段。
          - FC1 / FC2 = 1×1 图上的 1×1 卷积 (只有 1 行 → OC 并行; 960 通道输入走 IC 分块 + OC 并行);
            FC2 的输出激活是 HardSigmoid (Layer.act = 1: int8 饱和后 +128)。
          - 投影层 (Layer.sc): 层前面多一个"装缩放"槽, 从输入 fifo 收 pre 个元素的缩放向量放进 core 的 sc 缓冲区;
            之后每步第一次调用时把缩放折进这一步的权重 (W' = (W·s + 128) >> 8, 原地), 和行数、输入重读次数无关。
  · 每个存储张量一个自己的 scratch 槽 (有分支后不能再按形状乒乓)。
  · 输出激活 (Layer.act): 0 ReLU (uint8 饱和), 1 HardSigmoid, 2 线性 (int8 饱和), 3 hardswish。卷积内核只分
    "uint8 / int8 饱和"两种存法; HardSigmoid / hardswish 在入口函数最后对整个 640 B 输出块做一遍 (post_act),
    这样只有一个调用点, 不用在每个存储点加代码。hardswish 的整数定义见 requant (HS_Q, HS_S)。
  · 通道拼接 (Layer.cat): 层读 [张量 src ‖ 张量 cat], Plan 把后者放进前者的存储里 (每行 [组][像素][16] 里接着往后排),
    写的人直接写到位, 不用搬。残差加 = 拼接 + 投影层里残差那部分用单位权重 RES_W (内核不用改)。
  · ict 切行: 一个 core 的行数 × 块数放不进 acc 时, 行再切成几段 (n_rb), 每段一份 (输入只读自己那段)。

  · 重新量化 (LParam): 累加器初值 = bias, 输出 y = sat8((sat16((acc + 2^(s1-1)) >> s1) · M[c] + 2^(s2-1)) >> s2),
    M 是逐通道 int16。bias / M / hardswish 的逐通道参数放在每步权重后面的重新量化表里 (每个输出通道组 RQB 字节,
    已经排成内核要的向量布局)。随机权重的测试用 bias 0、M 1、s1 = shift、s2 0 (= 只移位); --rq 用随机 bias 和 M。
    真实 RVM 编码器的权重和参数由 npu/rvm/encoder_q.py (量化模拟) 冻结, npu/rvm/encoder_npu.py 转成这里的层。

还没做: Concat 层 (拼接视图有了, 解码器还没接)、Resize、ConvGRU; 3×3 的 IC 分块 (RVM 里没有需要的层)。
层间激活约定: 全部是对称 int8 (零点 0); ReLU 输出 0..127, HardSigmoid 输出 uint8 (比例 1/256, 只给 SE 的缩放用)。
"""
from __future__ import annotations

import os
import resource
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction

from .fused import KERNEL_INC
from .harness import SelfTestResult

SRC = str(Path(__file__).resolve().parent / "kernels" / "conv_c16.cc")
PAD, HDR, CW = 4, 256, 40
BR = 4                         # 张量上下各 BR 行零边 (5×5 dilation 2 的 depthwise 要 4 行)
E_IN = 7680                    # 输入元素字节数 (所有层相同; = 10 组 × (40+8)·16)
RQB = 576                      # 每个输出通道组的重新量化表 (bias / M / hardswish 参数), 跟在这一步的权重后面
W_BIG = 18432 + RQB            # 一个权重元素里最多的权重 + 重新量化表字节 (= 3×3 128→16; 权重 fifo 深度 1)
W_SMALL = 9216 + RQB           # 链里没有大 3×3 时用小元素 + 深度 2 (算这一步时预取下一步, 实测快 ~11%)
W_BUDGET = 24576               # 每 core 权重 fifo 的内存 (深度 × e_w); 太小会让大权重层退化成深度 1
W_BD_MAX = 16384               # 单个 DMA BD 的字节上限 (实测 ≥ 16384 会挂死); 权重 fill 组按它切
ACC_CHUNKS = 3                 # acc 缓冲区能放几个输出块的部分和 (每块 CW·16 个 int32 = 2.5 KB)
N_CORES = 4
FILLS_PER_TG = 2               # 一个 task group 里同一个 shim 通道排几个 fill (在途的组合计 ≤ 4, 多了会挂死)
W_PREFETCH = True              # 下一层第一组的权重在层间屏障之前发出
W_BCAST = os.environ.get("NPU_W_BCAST", "1") != "0"   # 行并行的链权重广播给 4 个 core (见 Plan.w_bcast)
TG_INFLIGHT = 2                # 同一层里同时在途的 task group 数 (2 = 发下一组之后再等前一组)
MAX_PLANE = 4064               # shim BD 最内维上限之内
SC_BYTES = 1024                # 每 core 的 SE 缩放向量缓冲区 (≤ 64 个通道组)


def cdiv(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class Layer:
    ic: int
    oc: int
    k: int
    s: int
    h: int
    w: int
    dw: int = 0     # 1 = depthwise (逐通道, ic == oc)
    d: int = 1      # dilation
    op: int = 0     # 0 = 卷积, 1 = 全局平均池化 (输出 C×1×1, ic == oc), 2 = 双线性 ×2 上采样, 3 = 外部输入 (主机写, 不算)
    act: int = 0    # 输出激活: 0 ReLU (uint8 饱和), 1 HardSigmoid (int8 饱和 + 128 → uint8, 128 = 0.5),
                    #           2 线性 (int8 饱和), 3 hardswish (int8, 见 requant)
    src: int = -1   # 输入张量 (相对本层输出张量的偏移; -1 = 上一层的输出)
    sc: int = 0     # ≠ 0: 输入逐通道乘这个 C×1×1 张量 (SE 的 Mul), 折进权重: W' = (W·s[ic] + 128) >> 8
    cat: int = 0    # ≠ 0: 输入 = 通道拼接 [张量 src, 张量 cat] (两者放进同一块存储, 不用搬); 残差加 = 拼接 + 单位权重
    oh: int = 0     # op = 2 (双线性 ×2 上采样) 的输出高度 (0 = 2h; 2h − 1 = 裁掉最后一行, 如 12 → 23)
    nx: int = 0     # ≠ 0: 本层输出放在张量 (k+1+nx) 的存储里、紧跟在它后面 (可以指向后面的层; 拼接的通用写法)
    al: int = 0     # ≠ 0: 本层输出就地写进张量 (k+1+al) 的位置 (GRU 状态更新); 之后读那个张量读到的是新值
    sy: int = 0     # ≠ 0: 垂直步长 (非对称 stride=(sy, s), 如 OCR 的 (2,1)/(1,2)); 0 = 与 s 相同

    @property
    def stride_y(self): return self.sy or self.s
    @property
    def stride_x(self): return self.s
    @property
    def pad(self): return 0 if self.op else self.d * (self.k - 1) // 2
    @property
    def ho(self):
        return 1 if self.op == 1 else (self.oh or 2 * self.h) if self.op == 2 else self.h if self.op == 3 \
            else cdiv(self.h, self.stride_y)
    @property
    def wo(self): return 1 if self.op == 1 else 2 * self.w if self.op == 2 else self.w if self.op == 3 else cdiv(self.w, self.stride_x)
    @property
    def icp(self): return self.ic // 16
    @property
    def ocp(self): return self.oc // 16
    @property
    def n_w(self): return 0 if self.op else self.k * self.k * (self.oc if self.dw else self.ic * self.oc)
    @property
    def macs(self):
        if self.op == 1:
            return self.h * self.w * self.ic
        if self.op == 2:
            return self.ho * self.wo * self.ic
        if self.op == 3:
            return 0
        if self.op == 6:
            return 2 * (self.h * self.w) ** 2 * self.oc
        return self.ho * self.wo * self.oc * (1 if self.dw else self.ic) * self.k * self.k


# RVM 编码器形状 (通道补到 16 的倍数; dw 卷积暂时用普通 3×3 代替), 输入 16 通道 180×320
R = {
    "e1": Layer(16, 16, 3, 2, 180, 320),   # → 90×160            (≈ Conv_8 / Conv_17)
    "e2": Layer(16, 64, 1, 1, 90, 160),    # 1×1 升维            (≈ Conv_15)
    "e3": Layer(64, 32, 3, 2, 90, 160),    # → 45×80, 45 行奇数  (≈ Conv_17 + Conv_19)
    "e4": Layer(32, 80, 1, 1, 45, 80),     # 1×1 32→80          (≈ Conv_20, 72→80)
    "e5": Layer(80, 32, 1, 1, 45, 80),     # 1×1 80→32, ICP 5   (≈ Conv_24)
    "e6": Layer(32, 32, 3, 1, 45, 80),     # 3×3 s1 @45×80, 行 23+22
    "e7": Layer(32, 48, 3, 2, 45, 80),     # → 23×40             (≈ Conv_28)
    "e8": Layer(48, 128, 1, 1, 23, 40),    # 1×1 48→128 @23×40  (≈ Conv_37)
    "e9": Layer(128, 48, 1, 1, 23, 40),    # 1×1 128→48         (≈ Conv_47)
    "e10": Layer(48, 32, 3, 2, 23, 40),    # → 12×20, 宽 20 存成 40
    "e11": Layer(32, 128, 1, 1, 12, 20),   # 1×1 @12×20
    "e12": Layer(128, 32, 1, 1, 12, 20),
    "e13": Layer(32, 32, 3, 1, 12, 20),    # 3×3 s1 @12×20 (补宽列必须是零)
    "gelu1": Layer(64, 128, 1, 1, 12, 20, act=5),   # GELU 查表 (act 5) 的测试层
    # LayerNorm (op 5) 的测试层: 4 行 × 35 像素、512 通道 (语音模型真实形状, 4 个 IC 块) —— 与参考逐字节一致。
    "lm1": Layer(512, 512, 1, 1, 4, 35, op=5),
    # 多头自注意力 (op 6) 的测试层: [Q ‖ K ‖ V] 1536→512, 4×35 (140 token, 8 头 × 64 维)
    "attn1": Layer(1536, 512, 1, 1, 4, 35, op=6),
    # 短窗口 (按说话长度分档): 每行 W 个 token, 4 行共 4W —— W=15 (60 token, 3 s)、W=10 (40 token, 2 s)、W=13 (不是 4 的倍数)
    "attn15": Layer(1536, 512, 1, 1, 4, 15, op=6),
    "attn10": Layer(1536, 512, 1, 1, 4, 10, op=6),
    "attn13": Layer(1536, 512, 1, 1, 4, 13, op=6),
    "lm15": Layer(512, 512, 1, 1, 4, 15, op=5),
    "dw_asym1": Layer(32, 32, 3, 1, 24, 160, dw=1, sy=2),
    "dw_asym2": Layer(32, 32, 5, 2, 12, 160, dw=1, sy=1),
}
# RVM 编码器 12×20 段的真实形状 (跳过 depthwise 和 SE; 200/184 补到 208/192), 输入 240 通道
_T = [(240, 80, 1, "Conv_67"), (80, 208, 1, "Conv_68"), (208, 80, 1, "Conv_74"), (80, 480, 1, "Conv_92"),
      (480, 112, 1, "Conv_104"), (112, 672, 1, "Conv_105"), (672, 160, 1, "Conv_131"),
      (160, 960, 1, "Conv_132"), (960, 160, 1, "Conv_144"), (160, 960, 1, "Conv_146"),
      (960, 128, 1, "Conv_163"), (128, 128, 3, "Conv_176"), (128, 64, 3, "Conv_181")]
for _ic, _oc, _k, _n in _T:
    R[_n] = Layer(_ic, _oc, _k, 1, 12, 20)
# RVM 编码器 (MobileNetV3) 主干的真实形状, 跳过 SE 和 hardswish; 通道补到 16 的倍数 (3→16, 24→32, 72→80, 40→48,
# 120→128, 200→208), Conv_105 的输入用 80 (原 112, 中间几层省掉了)
_M = [("Conv_8", 16, 16, 3, 2, 180, 320, 0, 1), ("Conv_11", 16, 16, 3, 1, 90, 160, 1, 1),
      ("Conv_13", 16, 16, 1, 1, 90, 160, 0, 1), ("Conv_15", 16, 64, 1, 1, 90, 160, 0, 1),
      ("Conv_17", 64, 64, 3, 2, 90, 160, 1, 1), ("Conv_19", 64, 32, 1, 1, 45, 80, 0, 1),
      ("Conv_20", 32, 80, 1, 1, 45, 80, 0, 1), ("Conv_22", 80, 80, 3, 1, 45, 80, 1, 1),
      ("Conv_24", 80, 32, 1, 1, 45, 80, 0, 1), ("Conv_26", 32, 80, 1, 1, 45, 80, 0, 1),
      ("Conv_28", 80, 80, 5, 2, 45, 80, 1, 1), ("Conv_36", 80, 48, 1, 1, 23, 40, 0, 1),
      ("Conv_37", 48, 128, 1, 1, 23, 40, 0, 1), ("Conv_39", 128, 128, 5, 1, 23, 40, 1, 1),
      ("Conv_47", 128, 48, 1, 1, 23, 40, 0, 1), ("Conv_61", 48, 240, 1, 1, 23, 40, 0, 1),
      ("Conv_64", 240, 240, 3, 2, 23, 40, 1, 1), ("Conv_67m", 240, 80, 1, 1, 12, 20, 0, 1),
      ("Conv_68m", 80, 208, 1, 1, 12, 20, 0, 1), ("Conv_71", 208, 208, 3, 1, 12, 20, 1, 1),
      ("Conv_74m", 208, 80, 1, 1, 12, 20, 0, 1), ("Conv_105m", 80, 672, 1, 1, 12, 20, 0, 1),
      ("Conv_122", 672, 672, 5, 1, 12, 20, 1, 2), ("Conv_131m", 672, 160, 1, 1, 12, 20, 0, 1),
      ("Conv_132m", 160, 960, 1, 1, 12, 20, 0, 1), ("Conv_135", 960, 960, 5, 1, 12, 20, 1, 2),
      ("Conv_144m", 960, 160, 1, 1, 12, 20, 0, 1)]
for _n, *_a in _M:
    R[_n] = Layer(*_a)
MNV3 = [n for n, *_ in _M]


# SE: dw 输出 → 全局平均池化 → FC1 (1×1@1×1, ReLU) → FC2 (HardSigmoid) → 投影 1×1 (输入 = dw 输出, 逐通道乘 FC2 输出)
def _se(dw: str, fc1: str, fc2: str, proj: str, mid: int, oc: int):
    l = R[dw]
    c, h, w = l.oc, l.ho, l.wo
    R[dw + ".pool"] = Layer(c, c, 1, 1, h, w, op=1)
    R[fc1] = Layer(c, mid, 1, 1, 1, 1)
    R[fc2] = Layer(mid, c, 1, 1, 1, 1, act=1)
    R[proj] = Layer(c, oc, 1, 1, h, w, src=-4, sc=-1)
    return [dw + ".pool", fc1, fc2, proj]


SE = {   # dw 层 → SE 三层 + 带逐通道乘的投影层 (通道补到 16 的倍数: 24→32, 120→128)
    "Conv_28": _se("Conv_28", "Conv_31", "Conv_33", "Conv_36se", 32, 48),
    "Conv_39": _se("Conv_39", "Conv_42", "Conv_44", "Conv_47se", 32, 48),
    "Conv_122": _se("Conv_122", "Conv_126", "Conv_128", "Conv_131se", 176, 160),
    "Conv_135": _se("Conv_135", "Conv_139", "Conv_141", "Conv_144se", 240, 160),
}
R["Conv_95"] = Layer(480, 480, 3, 1, 12, 20, dw=1)
SE["Conv_95"] = _se("Conv_95", "Conv_99", "Conv_101", "Conv_104se", 128, 112)
MNV3SE = []
for _n in MNV3:
    if _n in ("Conv_36", "Conv_47", "Conv_131m", "Conv_144m"):
        continue                                   # 换成 SE 版的投影层
    MNV3SE.append(_n)
    MNV3SE += SE.get(_n, [])
ENC = ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "e11", "e12", "e13"]
DEEP = [n for *_, n in _T]
# 真实激活 (MobileNetV3-Large): 第一层 hardswish; 投影层线性; 240 通道 (Conv_61) 起升维 / dw 用 hardswish; 其余 ReLU
_LINEAR = {"Conv_13", "Conv_19", "Conv_24", "Conv_36", "Conv_36se", "Conv_47", "Conv_47se", "Conv_67m", "Conv_74m",
           "Conv_131m", "Conv_131se", "Conv_144m", "Conv_144se", "Conv_104se"}
_HSWISH = {"Conv_8", "Conv_61", "Conv_64", "Conv_68m", "Conv_71", "Conv_105m", "Conv_122", "Conv_132m", "Conv_135",
           "Conv_92", "Conv_95"}


def _real(names):
    out = []
    for n in names:
        act = 2 if n in _LINEAR else 3 if n in _HSWISH else R[n].act
        if act != R[n].act:
            R[n + "~"] = replace(R[n], act=act)
            n += "~"
        out.append(n)
    return out


CHAINS = {
    "gelu": ["gelu1"],                                   # GELU 查表 (act 5)
    "lm": ["lm1"],                                       # LayerNorm (op 5)
    "attn": ["attn1"],                                   # 多头自注意力 (op 6)
    "attn15": ["attn15"], "attn10": ["attn10"], "attn13": ["attn13"],   # 短窗口注意力
    "lm15": ["lm15"],
    "dw_asym": ["dw_asym1", "dw_asym2"],
    "enc": ENC,
    "head": ENC[:3],
    "low": ENC + ["e11", "e12", "e13"] * 4,           # 12×20 的三层重复
    "deep": DEEP,                                     # 12×20 高通道段 (OC / IC 分块)
    "mnv3": MNV3,                                     # 编码器主干, 含全部几种 depthwise
    "mnv3se": MNV3SE,                                 # 主干 + 4 个 SE
    "se39": ["Conv_37", "Conv_39"] + SE["Conv_39"],   # 一个 SE 块 (23×40, full 模式投影)
    "se95": ["Conv_92", "Conv_95"] + SE["Conv_95"],   # 12×20, 投影层是 IC 分块
    "se135": ["Conv_132m", "Conv_135"] + SE["Conv_135"],  # 960 通道, FC1 是 IC 分块
}
# 残差块: 投影层 → 块的输入 (上一个块的投影层输出)
_RES = {"Conv_24": ("Conv_19",), "Conv_47se": ("Conv_36se",), "Conv_74m": ("Conv_67m",),
        "Conv_144se": ("Conv_131se", "Conv_131m")}


def _resid(names):
    """把链里的残差块投影层换成"读 dw 输出 ‖ 块输入、残差部分单位权重"的版本"""
    base = [n.rstrip("~") for n in names]
    out = list(names)
    for k, n in enumerate(base):
        js = [base.index(m) for m in _RES.get(n, ()) if m in base]
        if js:
            j = js[0]                        # 块输入 = 张量 j+1; 本层输出 = 张量 k+1
            l = R[names[k]]
            R[names[k] + "r"] = replace(l, ic=l.ic + l.oc, cat=j - k)
            out[k] = names[k] + "r"
    return out


CHAINS["mnv3a"] = _real(MNV3SE)                      # 主干 + SE, 真实激活 (hardswish / 线性 / ReLU)
CHAINS["mnv3r"] = _resid(CHAINS["mnv3a"])            # + 4 个残差加
CHAINS["res24"] = _resid(_real(["Conv_17", "Conv_19", "Conv_20", "Conv_22", "Conv_24"]))   # 45×80 一个残差块
CHAINS["res144"] = _resid(_real(["Conv_131m", "Conv_132m", "Conv_135"] + SE["Conv_135"]))   # 有 SE + IC 分块
CHAINS["se135a"] = _real(CHAINS["se135"])            # hardswish 的 dw → 有符号池化


# 直接用 Layer 写的链 (形状不在 RVM 编码器的卷积表里的)
LAYER_CHAINS = {
    # 解码器的上采样 + 卷积骨架: 4 个真实形状的双线性 ×2 (含 12→23、23→45 的裁行), 中间用 3×3 接起来
    "up": [Layer(128, 128, 1, 1, 12, 20, op=2, oh=23), Layer(128, 80, 3, 1, 23, 40),
           Layer(80, 80, 1, 1, 23, 40, op=2, oh=45), Layer(80, 48, 3, 1, 45, 80),
           Layer(48, 48, 1, 1, 45, 80, op=2), Layer(48, 32, 3, 1, 90, 160),
           Layer(32, 32, 1, 1, 90, 160, op=2), Layer(32, 16, 3, 1, 180, 320)],
    # 三路拼接 (解码器一级的样子): 存储 [上采样 ‖ f ‖ s], f 由前面的卷积写进来 (nx 指向后面的张量), s 是主机输入
    "cat3": [Layer(16, 48, 3, 1, 23, 40, nx=2),              # 张量 1 = f, 放在张量 3 (上采样输出) 后面
             Layer(64, 64, 1, 1, 12, 20, op=3),              # 张量 2 = 低分辨率输入 (主机)
             Layer(64, 64, 1, 1, 12, 20, op=2, oh=23),       # 张量 3 = 上采样
             Layer(16, 16, 1, 1, 23, 40, op=3, nx=-3),       # 张量 4 = s (主机), 放在 f 后面
             Layer(128, 64, 3, 1, 23, 40, src=-2),           # 读窗口 [张量 3 ‖ 1 ‖ 4] = 64 + 48 + 16
             Layer(64, 32, 1, 1, 23, 40)],
    # 就地覆盖: 层 2 把结果写回张量 1 的位置, 层 3 读张量 1 读到的是新值
    "alias": [Layer(32, 32, 1, 1, 23, 40), Layer(32, 32, 1, 1, 23, 40),
              Layer(32, 32, 1, 1, 23, 40, src=-2, al=-2), Layer(32, 32, 1, 1, 23, 40, src=-3)],
}


def _tblock(W):
    """语音模型的一个 Transformer 块 (9 个算子, 和 npu/asr/encoder_npu.make_block_layers 一样), 每行 W 个 token"""
    return [Layer(512, 512, 1, 1, 4, W, op=5), Layer(512, 512, 1, 1, 4, W, act=2),
            Layer(512, 512, 1, 1, 4, W, act=2, src=-2, nx=-1), Layer(512, 512, 1, 1, 4, W, act=2, src=-3, nx=-1),
            Layer(1536, 512, 1, 1, 4, W, op=6, src=-3), Layer(1024, 512, 1, 1, 4, W, act=2, src=-1, cat=-6),
            Layer(512, 512, 1, 1, 4, W, op=5), Layer(512, 2048, 1, 1, 4, W, act=5),
            Layer(2560, 512, 1, 1, 4, W, act=2, src=-1, cat=-3)]


LAYER_CHAINS.update({"tblk": _tblock(35), "tblk15": _tblock(15)})


def chain_layers(spec: str) -> list[Layer]:
    if spec in LAYER_CHAINS:
        return list(LAYER_CHAINS[spec])
    if spec in CHAINS:
        return [R[k] for k in CHAINS[spec]]
    return [R[k] for k in spec.split(",")]


def spec_str(layers) -> str:
    return ";".join(f"{l.ic},{l.oc},{l.k},{l.s},{l.h},{l.w},{l.dw},{l.d},{l.op},{l.act},{l.src},{l.sc},{l.cat},{l.oh},"
                    f"{l.nx},{l.al},{l.sy}" for l in layers)


def from_spec_str(s: str) -> list[Layer]:
    return [Layer(*map(int, p.split(","))) for p in s.split(";")]


@dataclass
class Tensor:
    c: int
    h: int
    w: int        # 真实宽度
    ws: int       # 存储宽度 (≥ w)
    ctot: int = 0  # 所在存储张量的总通道数 (通道拼接: 几个张量共用一块存储, 每行 [组][像素][16] 里各占一段组)
    g0: int = 0    # 本张量在存储张量里的起始通道组

    @property
    def cplane(self): return (self.ws + 2 * PAD) * 16
    @property
    def frow(self): return ((self.ctot or self.c) // 16) * self.cplane
    @property
    def size(self): return (self.h + 2 * BR) * self.frow
    @property
    def key(self): return (self.c, self.h, self.ws)


@dataclass
class Tiling:
    n_tw: int
    n_bands: int
    wt: int
    band_y0: list
    band_rows: list
    mode: str = "full"          # full / oc / ict
    icp_pad: int = 0            # 输入元素里的通道组数 (full/oc) 或每步通道组数 m (ict)
    plane: int = 0
    nocp: int = 0               # 每个 OC 块 (oc) / 每份 (ict) 的输出通道组数
    n_ob: int = 1               # OC 块数 / 份数
    n_icb: int = 1              # IC 步数 (ict)
    steps: list = field(default_factory=list)   # [(ocp0, nocp, icp0, nicp, first, last)]
    ocpar: bool = False         # OC 并行: 4 个 core 各算不同的 OC 块、都算全部行 (低分辨率的 OC 分块层)
    pre: int = 0                # SE 投影层: 层前面加一个"装缩放向量"槽, 收 pre 个输入元素
    n_rb: int = 1               # ict: 每 core 的行再切成 n_rb 段 (每段 ≤ rcap 行, acc 放得下); 份号 o → 段 o % n_rb
    rcap: int = 0

    def sub_rows(self, b, o):
        """ict 切行时, 段 b 的份 o 这一段的 (起始行, 行数)"""
        if self.n_rb == 1:
            return 0, self.band_rows[b]
        r0 = (o % self.n_rb) * self.rcap
        return r0, max(0, min(self.rcap, self.band_rows[b] - r0))

    def cores(self):
        """[(core id, 条 t, 段 b)]"""
        if self.mode == "pool":
            return [(c, 0, 0) for c in range(min(N_CORES, self.nocp))]
        if self.ocpar:
            return [(c, 0, 0) for c in range(min(N_CORES, self.n_ob))]
        return [(t * self.n_bands + b, t, b) for t in range(self.n_tw) for b in range(self.n_bands)]

    def blocks_of(self, c):
        """OC 并行时 core c 负责的 OC 块数 (块 c, c+4, c+8, …)"""
        return len(range(c, self.n_ob, N_CORES))

    def slots(self):
        """core 外层循环的每一步 = 一个槽, 槽里每个 core 一个步骤 (None = 这个 core 这一步空闲)"""
        if self.mode == "input":
            return []
        act = {c for c, _, _ in self.cores()}
        load = [[LOAD if c in act else None for c in range(N_CORES)]] if self.pre else []
        if self.mode == "pool":                     # 两步: 累加 (每组前 ne-1 个元素), 收尾 (每组最后一个元素)
            return [[st if c in act else None for c in range(N_CORES)] for st in self.steps]
        if not self.ocpar:
            return load + [[st] * N_CORES for st in self.steps]
        spo = len(self.steps) // self.n_ob          # 每个 OC 块/份的步数 (ict 时 = IC 步数)
        return load + [[self.steps[(g * N_CORES + c) * spo + i] if g * N_CORES + c < self.n_ob else None
                        for c in range(N_CORES)]
                       for g in range(cdiv(self.n_ob, N_CORES)) for i in range(spo)]

    @property
    def nxc(self): return self.wt // CW


LOAD = (0, 0, 0, 0, 0, 0)       # "装 SE 缩放向量"槽里的步骤 (没有权重)


def _divisor_le(n, cap):
    return max(d for d in range(1, n + 1) if n % d == 0 and d <= cap)


def _pool_rows(cplane):
    """池化: 一个输入元素放 kr 行 (每行 X = E_IN / kr 字节 ≥ 一个通道组平面)"""
    return max(d for d in range(1, 64) if E_IN % d == 0 and cplane <= E_IN // d <= MAX_PLANE
               and (E_IN // d) % 64 == 0)


def _up_tiling(l: Layer) -> Tiling:
    """双线性 ×2 上采样: 按输出宽度切条 (wt 是 CW 的倍数, 每条读 wt/2 个输入像素), 按**输入**行切段
    (band_y0 / band_rows 记输出行 = 2 × 输入行; 最后一段可能多出裁掉的那一行, 内核写 0, 落在下边的零边里)。
    输入元素 = 一个输入行的 m 个通道组 [m][PLANE], s1 滑窗 (每个输入行窗口出 2 个输出行); 每步 m 个通道组。"""
    assert l.ic == l.oc and l.oh in (0, 2 * l.h, 2 * l.h - 1) and l.wo % CW == 0, l
    best = None
    for t in (4, 2, 1):
        wt = cdiv(cdiv(l.wo, t), CW) * CW
        bands = N_CORES // t
        if l.h < bands or (t - 1) * wt >= l.wo:
            continue
        score = (t * wt / l.wo, -t)
        if best is None or score < best[0]:
            best = (score, t, wt, bands)
    _, t, wt, bands = best
    q, r = divmod(l.h, bands)
    rows = [q + (1 if b < r else 0) for b in range(bands)]
    tl = Tiling(t, bands, wt, [2 * sum(rows[:b]) for b in range(bands)], [2 * x for x in rows])
    need = (wt // 2 + 2 * PAD) * 16
    m = max(p for p in range(1, l.icp + 1) if E_IN % p == 0 and need <= E_IN // p <= MAX_PLANE and (E_IN // p) % 64 == 0)
    tl.mode, tl.icp_pad, tl.plane, tl.nocp = "up", m, E_IN // m, m
    tl.n_ob = cdiv(l.icp, m)
    tl.steps = [(o * m, min(m, l.icp - o * m), o * m, min(m, l.icp - o * m), 1, 1) for o in range(tl.n_ob)]
    return tl


def choose_tiling(l: Layer, ws_in: int | None = None, w_max: int = 0) -> Tiling:
    W_MAX = w_max or W_BIG
    if l.op == 1:
        # 全局平均池化: 按通道组分给 4 个 core (组 c, c+4, …), 每 core 读自己的组的全部行, 不用跨 core 归约。
        # 输入元素 = 一个通道组的 kr 行 (每行 X 字节); 每组 ne 个元素, 最后一个元素多出来的行内核不看。
        cpl = ((ws_in or l.w) + 2 * PAD) * 16
        kr = _pool_rows(cpl)
        tl = Tiling(1, 1, CW, [0], [1], mode="pool", icp_pad=kr, plane=E_IN // kr, nocp=l.icp,
                    n_ob=1, n_icb=cdiv(l.h, kr))
        tl.steps = [(0, l.icp, 0, l.icp, 1, 0), (0, l.icp, 0, l.icp, 0, 1)]
        return tl
    if l.op == 5:                # LayerNorm: 一行一个 core、读全部通道组 (跨组归约不能按通道分核)
        wt = cdiv(l.wo, CW) * CW
        bands = min(N_CORES, l.ho)
        q, r = divmod(l.ho, bands)
        rows = [q + (1 if b < r else 0) for b in range(bands)]
        tl = Tiling(1, bands, wt, [sum(rows[:b]) for b in range(bands)], rows)
        we = min(wt, cdiv(min(l.wo, wt), 8) * 8)
        need = (we + 2 * PAD) * 16
        m = max(p for p in range(1, 65) if E_IN % p == 0 and need <= E_IN // p <= MAX_PLANE
                and (E_IN // p) % 64 == 0 and l.icp % p == 0)
        tl.mode, tl.icp_pad, tl.plane = "lm", m, E_IN // m
        tl.nocp, tl.n_icb = 1, cdiv(l.icp, m)
        # 步: (ocp0, nocp, icp0, nicp, first, last, 行段, pass, stats)
        tl.steps = ([(0, 1, i * m, min(m, l.icp - i * m), 0, 0, 0, 0, int(i == tl.n_icb - 1))
                     for i in range(tl.n_icb)]                       # pass 0: 累加
                    + [(0, 1, i * m, min(m, l.icp - i * m), 0, 0, 0, 1, 0)
                       for i in range(tl.n_icb)])                     # pass 1: 归一化
        tl.n_ob = len(tl.steps)          # 每步一个 fill 单元 (照 ict: 大 BD 会撞 DMA 限制)
        return tl
    if l.op == 6:                # 多头自注意力 (Q‖K‖V → O_in): ic = 3·oc, oc = 64·n_heads, H=4, W=35
        assert l.ic == 3 * l.oc and l.oc % 64 == 0 and l.ho == N_CORES and l.wo <= CW, f"attn 形状不支持: {l}"
        n_heads = l.oc // 64
        tl = Tiling(1, N_CORES, CW, [0, 1, 2, 3], [1, 1, 1, 1])
        tl.mode, tl.icp_pad, tl.plane, tl.nocp = "attn", 8, 960, 4
        tl.steps = [(4 * h, 4, 4 * h, 4, 0, sub, 0, sub, h) for h in range(n_heads) for sub in (0, 1)]
        tl.n_ob = len(tl.steps)
        return tl
    if l.op == 2:
        return _up_tiling(l)
    if l.op == 3:                # 外部输入: 不算, 没有步
        return Tiling(1, 1, CW, [0], [l.h], mode="input")
    best = None
    for t in (4, 2, 1):
        wt = cdiv(cdiv(l.wo, t), CW) * CW
        bands = N_CORES // t
        if l.ho < bands or (t - 1) * wt >= l.wo:
            continue
        score = (t * wt / l.wo, -t)
        if best is None or score < best[0]:
            best = (score, t, wt, bands)
    if best is None:                                 # 行数不够 4 段 (FC: 1×1 图)
        t, wt, bands = 1, cdiv(l.wo, CW) * CW, l.ho
    else:
        _, t, wt, bands = best
    q, r = divmod(l.ho, bands)
    rows = [q + (1 if b < r else 0) for b in range(bands)]
    tl = Tiling(t, bands, wt, [sum(rows[:b]) for b in range(bands)], rows)
    we = min(wt, cdiv(min(l.wo, wt), 8) * 8)          # 内核实际会读到的输出宽度 (整组在 wvalid 外的不读)
    need = (l.s * we + 2 * PAD) * 16
    if l.dw:
        # depthwise: 输入元素 = [k 行][m 个通道组][PLANE], 每步 m 个通道组, 输入不重读
        assert l.ic == l.oc and l.pad <= min(BR, PAD)
        m = max(p for p in range(1, l.icp + 1) if E_IN % (l.k * p) == 0 and need <= E_IN // (l.k * p) <= MAX_PLANE
                and (E_IN // (l.k * p)) % 64 == 0)
        tl.mode, tl.icp_pad, tl.plane, tl.nocp = "dw", m, E_IN // (l.k * m), m
        tl.n_ob = cdiv(l.icp, m)
        tl.steps = [(o * m, min(m, l.icp - o * m), o * m, min(m, l.icp - o * m), 1, 1) for o in range(tl.n_ob)]
        return tl
    per_ocp = l.k * l.k * l.ic * 16 + RQB
    for p in range(l.icp, 64):
        if E_IN % p == 0 and need <= E_IN // p <= MAX_PLANE and (E_IN // p) % 64 == 0:
            tl.icp_pad, tl.plane = p, E_IN // p
            break
    if tl.plane and (per_ocp <= W_MAX or l.k != 1):  # 一行输入放得下 (1×1 一个输出组的权重放不下时走 IC 分块)
        if per_ocp > W_MAX:
            raise ValueError(f"{l}: 一个输出通道组的权重 {per_ocp} B > {W_MAX} (要 3×3 的 IC 分块)")
        tl.nocp = _divisor_le(l.ocp, W_MAX // per_ocp)
        tl.n_ob = l.ocp // tl.nocp
        tl.mode = "full" if tl.n_ob == 1 else "oc"
        if tl.mode == "oc" and l.s == 1 and tl.n_tw == 1:
            # 低分辨率 (一条带) 的 OC 分块层: 改成 OC 并行, 每 core 只拿自己那 1/4 的权重, 处理全部行。
            # 块大小取 "每 core 最多要算的通道组数 + 每槽固定开销" 最小的
            cap = W_MAX // per_ocp
            d = min((dd for dd in range(1, l.ocp + 1) if l.ocp % dd == 0 and dd <= cap),
                    key=lambda dd: (cdiv(l.ocp // dd, N_CORES) * (dd + 1), -dd))
            tl.nocp, tl.n_ob, tl.ocpar = d, l.ocp // d, True
            tl.n_bands, tl.band_y0, tl.band_rows = 1, [0], [l.ho]
        tl.steps = [(o * tl.nocp, tl.nocp, 0, l.icp, 1, 1) for o in range(tl.n_ob)]
        if tl.mode == "oc" and l.s == 2 and tl.n_ob > 8:
            raise ValueError(f"{l}: s2 的 OC 分块要每块一个 fill BD, 块太多")
        return tl
    if l.k != 1:
        raise ValueError(f"{l}: 3×3 的一行输入放不下 (要 3×3 的 IC 分块)")
    # IC 分块: 每步的通道组数 m 取放得下的最大值 (一个输出组 m·256 B 的权重也要放进一个权重元素)
    m = max(p for p in range(1, min(64, (W_MAX - RQB) // 256 + 1))
            if E_IN % p == 0 and need <= E_IN // p <= MAX_PLANE and (E_IN // p) % 64 == 0)
    tl.mode, tl.icp_pad, tl.plane = "ict", m, E_IN // m
    tl.n_icb = cdiv(l.icp, m)
    # (试过: 行切得更细、每份多几个通道组 → 输入重读少一半, 但步数 / 份数变多, 整体反而慢; 见 overhead spike)
    k_cap = ACC_CHUNKS // (max(rows) * tl.nxc)
    if k_cap < 1:                # acc 放不下一个 core 的全部行: 行再切段, 每段一份 (输入按段读, 不多读)
        tl.rcap = ACC_CHUNKS // tl.nxc
        if tl.rcap < 1:
            raise ValueError(f"{l}: acc 放不下一行 ({tl.nxc} 块 > {ACC_CHUNKS})")
        tl.n_rb, k_cap = cdiv(max(rows), tl.rcap), 1
    tl.nocp = _divisor_le(l.ocp, min(k_cap, W_MAX // (m * 256 + RQB)))
    tl.n_ob = l.ocp // tl.nocp * tl.n_rb
    tl.steps = [((o // tl.n_rb) * tl.nocp, tl.nocp, i * m, min(m, l.icp - i * m), int(i == 0), int(i == tl.n_icb - 1),
                 o % tl.n_rb)
                for o in range(tl.n_ob) for i in range(tl.n_icb)]
    if l.ho == 1 and tl.n_tw == 1 and tl.n_ob > 1:
        # FC (1×1 图): 只有 1 行, 按行切只能用 1 个 core → OC 并行, 每 core 算不同的份
        tl.ocpar, tl.n_bands, tl.band_y0, tl.band_rows = True, 1, [0], [1]
    return tl


class Plan:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
        l0 = layers[0]
        # 权重元素: 先试小元素 + 深度 2 (预取); 有放不进的层 (大 3×3) 就用大元素 + 深度 1
        for w_max in (W_SMALL, W_BIG):
            try:
                self.tensors = [Tensor(l0.ic, l0.h, l0.w, l0.w)]
                self.tilings, self.inp = [], []          # inp[k] = 层 k 的输入张量号 (输出是 k+1)
                links, alias = [], {}                    # (A, B): B 紧跟在 A 后面 (同一块存储); 输出 → 就地覆盖的张量
                for k, l in enumerate(layers):
                    if l.op == 3:                        # 外部输入: 存储宽度 = 真实宽度 (和张量 0 一样)
                        self.tilings.append(choose_tiling(l))
                        self.inp.append(0)
                        self.tensors.append(Tensor(l.oc, l.h, l.w, l.w))
                        if l.nx:
                            links.append((k + 1 + l.nx, k + 1))
                        continue
                    i = k + 1 + l.src
                    assert 0 <= i <= k, f"层 {k} 的输入 {l.src} 越界"
                    cur = self.tensors[i]
                    if l.cat:
                        j = k + 1 + l.cat
                        assert 0 <= j <= k and 0 < i, f"层 {k} 的拼接越界"
                        if j == 0:
                            self.tensors[0].ws = cdiv(l0.w, CW) * CW
                        links.append((i, j))
                    assert l.h == cur.h and l.w == cur.w, f"接不上: {cur} → {l}"
                    assert l.ic % 16 == 0 and l.oc % 16 == 0
                    if l.sc:
                        ts = self.tensors[k + 1 + l.sc]
                        assert l.k == 1 and l.s == 1 and not l.dw and ts.h == ts.w == 1, f"层 {k} 的缩放张量 {ts} 不对"
                    tl = choose_tiling(l, cur.ws, w_max)
                    if l.sc:
                        ts = self.tensors[k + 1 + l.sc]
                        tl.pre = cdiv(ts.c // 16, E_IN // ts.cplane)
                    self.tilings.append(tl)
                    self.inp.append(i)
                    self.tensors.append(Tensor(l.oc, l.ho, l.wo, tl.n_tw * tl.wt))
                    if l.nx:
                        links.append((k + 1 + l.nx, k + 1))
                    if l.al:            # 就地写只对逐行的层安全 (3×3 会把还要读的上一行盖掉)
                        assert l.k == 1 and not l.dw and l.op == 0, f"层 {k}: 就地写只支持 1×1 / 逐元素层"
                        alias[k + 1] = k + 1 + l.al
                break
            except ValueError:
                if w_max == W_BIG:
                    raise
        self._place(links, alias)
        self.n_steps = sum(len(tl.slots()) for tl in self.tilings)       # core 外层循环次数
        # 补到 256 的倍数: 权重按 [槽][core] 交错存, fill 是跳着读的, 最内维不能超过 1023 个字
        self.e_w = cdiv(HDR + max(self._step_wbytes(l, st) for l, tl in zip(layers, self.tilings)
                                  for st in tl.steps), 256) * 256
        self.w_depth = 2 if 2 * self.e_w <= W_BUDGET else 1
        # 权重广播: 所有层都是行并行 (每个 core 同一槽同一步, 没有 OC 并行 / 池化 / SE 缩放装载) 时, 4 个 core 的权重元素
        # 逐字节相同 —— 只从 DDR 读一份 (core 0 那份), 一个 fifo 广播给 4 个 core, 权重流量 ÷4。C16Chain 打包时核对。
        self.w_bcast = W_BCAST and all(not tl.ocpar and not tl.pre and tl.mode not in ("pool", "input")
                                       for l, tl in zip(layers, self.tilings) if l.op != 3)
        self.step0 = np.cumsum([0] + [len(tl.slots()) for tl in self.tilings]).tolist()
        self.io = [(self.loc[self.inp[k]], self.loc[k + 1]) for k in range(len(layers))]
        ends = dict(self._ends)
        for k, l in enumerate(layers):
            if l.op == 3:
                continue
            (src, soff), _ = self.io[k]
            tl = self.tilings[k]
            taps = [tp for c, t, b in tl.cores() for tp in self.fill_taps(k, soff, c, t, b)]
            if l.sc:
                ssrc, sbase = self.loc[k + 1 + l.sc]
                o_, sz, st = self.scale_tap(k, sbase)
                ends[ssrc] = max(ends[ssrc], o_ + sz[-1])
            for o, sz, st in taps:
                end = o + sum((n - 1) * s_ for n, s_ in zip(sz, st)) + 1
                ends[src] = max(ends[src], end)
        self.size = {k: cdiv(v, 64) * 64 for k, v in ends.items()}

    def _place(self, links, alias):
        """存储分组 + 放置: links 连成一条条存储 (每个张量前后最多各一个邻居), 组里按顺序排通道组;
        含张量 0 / 外部输入的组放 act 缓冲区, 最后一个张量单独放 out, 其余每组一个 scratch 槽。
        就地覆盖的张量 (alias) 和被覆盖的共用位置。"""
        T, layers = self.tensors, self.layers
        last = len(T) - 1
        nxt, prv = {}, {}
        for a, b in links:
            assert nxt.get(a, b) == b and prv.get(b, a) == a, f"张量 {a} / {b} 的邻居冲突 (一个张量前后各只能挨一个)"
            nxt[a], prv[b] = b, a
        groups = []
        for t in range(len(T)):
            if t in prv or t in alias:
                continue
            g = [t]
            while g[-1] in nxt:
                g.append(nxt[g[-1]])
                assert len(g) <= len(T), "存储分组成环了"
            groups.append(g)
        assert sum(len(g) for g in groups) + len(alias) == len(T), "有张量既被拼接又被别名"
        self.groups, self.nxt = groups, nxt
        for g in groups:
            if len(g) == 1:
                continue
            t0 = T[g[0]]
            ctot, g0 = sum(T[m].c for m in g), 0
            for m in g:
                tm = T[m]
                assert (tm.h, tm.w, tm.ws) == (t0.h, t0.w, t0.ws), f"同一块存储里形状不一样: {t0} / {tm}"
                tm.ctot, tm.g0 = ctot, g0
                g0 += tm.c // 16
        for t, a in alias.items():
            ta, tt = T[a], T[t]
            assert (ta.c, ta.h, ta.ws) == (tt.c, tt.h, tt.ws), f"别名形状不一样: {ta} / {tt}"
            tt.ctot, tt.g0 = ta.ctot, ta.g0
        inputs = {0} | {k + 1 for k, l in enumerate(layers) if l.op == 3}
        self.loc = [None] * len(T)
        off = {"act": 0, "s": 0, "out": 0}
        for g in sorted(groups, key=lambda g: 0 not in g):          # 张量 0 的组排在 act 最前面
            if last in g:
                assert g == [last] and last not in alias.values(), "最后一个张量要单独一块 (结果从 out 读)"
                buf = "out"
            else:
                buf = "act" if inputs & set(g) else "s"
            base = off[buf]
            for m in g:
                self.loc[m] = (buf, base + T[m].g0 * T[m].cplane)
            off[buf] += T[g[0]].size
        for t, a in alias.items():
            self.loc[t] = self.loc[a]
        self._ends = {k: max(v, 1) for k, v in off.items()}
        # 每层实际读的张量: 输入通道数比源张量多时是"窗口" (按存储顺序连着读后面的张量)
        self.tin = []
        for k, l in enumerate(layers):
            t = T[self.inp[k]]
            if l.op == 3 or l.ic == t.c:
                self.tin.append(t)
                continue
            cs = sum(T[m].c for m in self.window(k))
            assert cs == l.ic, f"层 {k} 读 {l.ic} 个通道, 窗口只能是整张量 ({cs})"
            self.tin.append(Tensor(l.ic, t.h, t.w, t.ws, t.ctot))

    def window(self, k):
        """层 k 读的张量号 (窗口里按存储顺序)"""
        l, m = self.layers[k], self.inp[k]
        out, cs = [m], self.tensors[m].c
        while cs < l.ic:
            assert m in self.nxt, f"层 {k} 读 {l.ic} 个通道, 但张量 {self.inp[k]} 后面接不够"
            m = self.nxt[m]
            out.append(m)
            cs += self.tensors[m].c
        return out

    def scale_tap(self, k, base):
        """SE 缩放向量 (C×1×1 张量) → 输入 fifo 的 n_pre 个元素, 每元素 E_IN // cplane 个通道组 (连续读)"""
        ts = self.tensors[k + 1 + self.layers[k].sc]
        return base + BR * ts.frow, [1, 1, 1, self.n_pre(k) * E_IN], [0, 0, 0, 1]

    def n_pre(self, k):
        return self.tilings[k].pre

    @staticmethod
    def _step_wbytes(l, st):
        _, nocp, _, nicp = st[:4]
        if l.op == 5:                # LN: 本步 nicp·16 通道的 m1/m2/g10/b12/gq (int32) + 2 个 fp32 常量
            return 5 * 4 * (nicp * 16) + 8
        if l.op == 6:                # ATTN: sub=0 放 256 项 uint16 EXP_LUT (512 B); sub=1 放 64 项 int16 mul_v (128 B)
            return 512
        if l.op:
            return 0
        if l.dw:
            return nocp * (l.k * l.k * 64 + RQB) + (GELU_TBL if l.act == 5 and st[5] else 0)
        return l.k * l.k * nicp * 16 * nocp * 16 + nocp * RQB + (GELU_TBL if l.act == 5 and st[5] else 0)

    @staticmethod
    def ob_groups(tl, l, wmax=99):
        """一层按 OC 块/份分成几个 task group, 每组每个 shim 通道最多 FILLS_PER_TG 个 fill
        (实测一个 task group 里同一通道排 7 个以上 DMA 任务就挂死, 5 个没事)。
        wmax = 权重 fifo 深度: 一组里的步数不能超过它, 否则一次 fill 塞进去的元素比 fifo 深
        (后面的元素直接覆盖前面的, fifo 锁的计数也错位 → 核心读到错步的权重)"""
        if tl.mode == "input":
            return []
        if tl.ocpar or tl.mode == "pool":
            return [(0, tl.n_ob)]
        if tl.mode == "dw":
            per = cdiv(max(tl.band_rows), 64)
        elif tl.mode in ("up", "lm"):
            per = 1
        elif tl.mode == "attn":
            per = 2
        else:
            per = 1 if tl.mode == "ict" else (0 if l.stride_y == 1 else cdiv(max(tl.band_rows), 64))
        if per == 0:
            return [(0, tl.n_ob)]
        g = min(max(1, FILLS_PER_TG // per), wmax)
        g0 = max(1, (FILLS_PER_TG - (1 if l.sc else 0)) // per)     # 第一组还要放 SE 缩放向量的 fill
        g0 = min(g0, wmax)
        return [(0, min(g0, tl.n_ob))] + [(o, min(o + g, tl.n_ob)) for o in range(g0, tl.n_ob, g)]

    def kernel_flags(self, k0=0, k1=None) -> list[str]:
        """层 k0..k1 用不到的内核分支 → -DNO_xxx (程序内存只有 16 KB, 装不下所有分支; 编码器 / 解码器各编各的)"""
        L = self.layers[k0:k1]
        use = {
            "SE": any(l.op == 1 or l.sc for l in L),
            "UP": any(l.op == 2 for l in L),
            "DW": any(l.dw for l in L),
            "DW3": any(l.dw and l.k == 3 for l in L),
            "DW5": any(l.dw and l.k == 5 and l.d == 1 for l in L),
            "DW5D2": any(l.dw and l.k == 5 and l.d == 2 for l in L),
            "S2": any(not l.op and not l.dw and l.k == 3 and l.s == 2 for l in L),
            "K3": any(not l.op and not l.dw and l.k == 3 for l in L),
            "PA": any(l.act in (1, 3) for l in L),
            "GELU": any(l.act == 5 for l in L),
            "LM": any(l.op == 5 for l in L),
            "ATTN": any(l.op == 6 for l in L),
        }
        return [f"-DNO_{k}" for k, v in use.items() if not v]

    def chunks(self, k):
        """层 k 的 task group 们: [(ob0, ob1, 权重槽数 ns, 第一个槽 s0)]"""
        l, tl = self.layers[k], self.tilings[k]
        spo = len(tl.steps) // tl.n_ob                   # 每个 OC 块/份的步数
        out = []
        wmax = max(1, min(self.w_depth, (W_BD_MAX - 1) // self.e_w))   # 一次 fill 别超过 BD 上限
        for ob0, ob1 in self.ob_groups(tl, l, wmax):
            if tl.ocpar:
                ns, s0 = len(tl.slots()), self.step0[k]
            else:                                        # 有"装缩放"槽时它在第一组最前面
                pre = 1 if tl.pre else 0
                ns = (ob1 - ob0) * spo + (pre if ob0 == 0 else 0)
                s0 = self.step0[k] + ob0 * spo + (pre if ob0 > 0 else 0)
            out.append((ob0, ob1, ns, s0))
        return out

    def fill_taps(self, k, base, c, t, b, ob0=0, ob1=None):
        """core c 这一层 (OC 块/份 ob0..ob1) 的输入 fill: [(偏移, sizes, strides)]"""
        l, tl, ti = self.layers[k], self.tilings[k], self.tin[k]
        ob1 = tl.n_ob if ob1 is None else ob1
        nob = tl.blocks_of(c) if tl.ocpar else ob1 - ob0
        rows = tl.band_rows[b]
        if tl.mode == "pool":    # core c: 通道组 c, c+4, …; 每组 ne 个元素 × kr 行 × X 字节 (从真实第 0 行起)
            # 先给累加步: 每组前 ne-1 个元素; 再给收尾步: 每组最后一个元素
            ng, ne, kr = len(range(c, l.icp, N_CORES)), tl.n_icb, tl.icp_pad
            o0 = base + BR * ti.frow + c * ti.cplane
            taps = [(o0, [ng, ne - 1, kr, tl.plane], [N_CORES * ti.cplane, kr * ti.frow, ti.frow, 1])] if ne > 1 else []
            return taps + [(o0 + (ne - 1) * kr * ti.frow, [1, ng, kr, tl.plane], [0, N_CORES * ti.cplane, ti.frow, 1])]
        if tl.mode == "up":      # 每步 (m 个通道组) 一个 BD: 输入行 y0-1 .. y0+rows, 每行 [m 组][PLANE]; 条 t 从输入像素 t·wt/2 起
            y0, rr = tl.band_y0[b] // 2, tl.band_rows[b] // 2
            o0 = base + (BR - 1 + y0) * ti.frow + t * (tl.wt // 2) * 16
            return [(o0 + tl.steps[o][0] * ti.cplane, [1, rr + 2, tl.icp_pad, tl.plane], [0, ti.frow, ti.cplane, 1])
                    for o in range(ob0, ob1)]
        off = base + (BR - 1 + l.stride_y * tl.band_y0[b]) * ti.frow + t * l.stride_x * tl.wt * 16   # 窗口第一行 = 真实行 sy·y0-1
        if tl.mode == "dw":      # 每步 (m 个通道组) 每 64 个输出行一个 BD: [输出行][ky][m 组][PLANE]
            o0 = base + (BR + l.stride_y * tl.band_y0[b] - l.pad) * ti.frow + t * l.stride_x * tl.wt * 16
            return [(o0 + tl.steps[o][0] * ti.cplane + l.stride_y * r0 * ti.frow,
                     [min(64, rows - r0), l.k, tl.icp_pad, tl.plane], [l.stride_y * ti.frow, l.d * ti.frow, ti.cplane, 1])
                    for o in range(ob0, ob1) for r0 in range(0, rows, 64)]
        if tl.mode == "lm":      # 每步一个 BD: [行][m 组][PLANE]; 第二遍的步重读同一段
            o0 = base + (BR + tl.band_y0[b]) * ti.frow
            return [(o0 + (tl.steps[o][2] // tl.icp_pad) * tl.icp_pad * ti.cplane,
                     [rows, tl.icp_pad, tl.plane], [ti.frow, ti.cplane, 1]) for o in range(ob0, ob1)]
        if tl.mode == "attn":    # 步 2h (sub=0): r=0 读本 core 的 Q 行, r=1..2 读全部 4 行 K; 步 2h+1 (sub=1): r=0..1 读全部 4 行 V
            q_row = base + (BR + tl.band_y0[b]) * ti.frow
            k_row0 = base + BR * ti.frow + (l.oc // 16) * ti.cplane
            v_row0 = base + BR * ti.frow + 2 * (l.oc // 16) * ti.cplane
            taps = []
            for o in range(ob0, ob1):
                sub, h = tl.steps[o][7], tl.steps[o][8]
                if sub == 0:
                    taps.append((q_row + 4 * h * ti.cplane, [1, 1, 8, 960], [0, ti.frow, ti.cplane, 1]))
                    taps.append((k_row0 + 4 * h * ti.cplane, [1, 4, 4, 960], [0, ti.frow, ti.cplane, 1]))
                else:
                    taps.append((v_row0 + 4 * h * ti.cplane, [1, 4, 2, 960], [0, ti.frow, ti.cplane, 1]))
                    taps.append((v_row0 + (4 * h + 2) * ti.cplane, [1, 4, 2, 960], [0, ti.frow, ti.cplane, 1]))
            return taps
        if tl.mode == "ict":     # 不要上下边行; 每份 OC 一个 BD: [IC 步][行][m 组][PLANE]
            o1 = off + ti.frow
            if rows == 1:        # FC: 一行, 所有份合成一个 BD (最外维步长 0)
                return [(o1, [nob, tl.n_icb, tl.icp_pad, tl.plane], [0, tl.icp_pad * ti.cplane, ti.cplane, 1])]
            if tl.n_rb > 1:      # 切行: 每份只读自己那段行
                return [(o1 + r0 * ti.frow, [tl.n_icb, rs, tl.icp_pad, tl.plane],
                         [tl.icp_pad * ti.cplane, ti.frow, ti.cplane, 1])
                        for o in range(ob0, ob1) for r0, rs in [tl.sub_rows(b, o)] if rs]
            return [(o1, [tl.n_icb, rows, tl.icp_pad, tl.plane], [tl.icp_pad * ti.cplane, ti.frow, ti.cplane, 1])
                    for _ in range(nob)]
        if l.s == 1:             # OC 块: 最外维步长 0, 同一份输入重复 nob 次
            return [(off, [nob, rows + 2, tl.icp_pad, tl.plane], [0, ti.frow, ti.cplane, 1])]
        return [(off + 2 * r0 * ti.frow, [min(64, rows - r0), 3, tl.icp_pad, tl.plane],
                 [2 * ti.frow, ti.frow, ti.cplane, 1])
                for _ in range(nob) for r0 in range(0, rows, 64)]

    def drain_taps(self, k, base, c, t, b, ob0=0, ob1=None):
        tl, to = self.tilings[k], self.tensors[k + 1]
        if tl.mode == "lm":     # 只有第二遍的步出输出, 每个通道组一个 640 B 块
            o0 = base + (BR + tl.band_y0[b]) * to.frow + (PAD + t * tl.wt) * 16
            return [(o0 + (tl.steps[o][2] // tl.icp_pad) * tl.icp_pad * to.cplane,
                     [tl.band_rows[b], tl.icp_pad, CW * 16], [to.frow, to.cplane, 1])
                    for o in range(ob0, ob1) if tl.steps[o][7]]
        if tl.mode == "attn":   # 只有 sub=1 的步出输出, 每头 4 个通道组 × 640 B
            o0 = base + (BR + tl.band_y0[b]) * to.frow + (PAD + t * tl.wt) * 16
            return [(o0 + 4 * tl.steps[o][8] * to.cplane, [1, 1, 4, CW * 16], [0, to.frow, to.cplane, 1])
                    for o in range(ob0, ob1) if tl.steps[o][7]]
        tl, to = self.tilings[k], self.tensors[k + 1]
        ob1 = tl.n_ob if ob1 is None else ob1
        if tl.mode == "pool":    # core c 的组 c, c+4, … 各一个 640 B 块 (像素 0 = 均值, 其余 0)
            ng = len(range(c, self.layers[k].icp, N_CORES))
            return [(base + BR * to.frow + PAD * 16 + c * to.cplane, [1, 1, ng, CW * 16],
                     [0, 0, N_CORES * to.cplane, 1])]
        if tl.n_rb > 1:          # ict 切行: 每份一个 BD
            o0 = base + (BR + tl.band_y0[b]) * to.frow + (PAD + t * tl.wt) * 16
            return [(o0 + r0 * to.frow + tl.steps[o * len(tl.steps) // tl.n_ob][0] * to.cplane,
                     [1, rs, tl.nocp, tl.wt * 16], [0, to.frow, to.cplane, 1])
                    for o in range(ob0, ob1) for r0, rs in [tl.sub_rows(b, o)] if rs]
        if tl.mode in ("dw", "up"):      # 每步一个 BD (最后一步可能少几个通道组)
            o0 = base + (BR + tl.band_y0[b]) * to.frow + (PAD + t * tl.wt) * 16
            return [(o0 + tl.steps[o][0] * to.cplane, [1, tl.band_rows[b], tl.steps[o][1], tl.wt * 16],
                     [0, to.frow, to.cplane, 1]) for o in range(ob0, ob1)]
        return [self.drain_tap(k, base, c, t, b, ob0, ob1)]

    def drain_tap(self, k, base, c, t, b, ob0=0, ob1=None):
        tl, to = self.tilings[k], self.tensors[k + 1]
        if tl.ocpar:             # core c 的块 c, c+4, … → 通道组 (c + 4i)·nocp
            off = base + BR * to.frow + PAD * 16 + c * tl.nocp * to.cplane
            return off, [tl.blocks_of(c), tl.band_rows[0], tl.nocp, tl.wt * 16], \
                [N_CORES * tl.nocp * to.cplane, to.frow, to.cplane, 1]
        ob1 = tl.n_ob if ob1 is None else ob1
        off = base + (BR + tl.band_y0[b]) * to.frow + (PAD + t * tl.wt) * 16 + ob0 * tl.nocp * to.cplane
        return off, [ob1 - ob0, tl.band_rows[b], tl.nocp, tl.wt * 16], [tl.nocp * to.cplane, to.frow, to.cplane, 1]

    def describe(self):
        out = []
        for k, (l, tl) in enumerate(zip(self.layers, self.tilings)):
            kind = ("dw " if l.dw else "") + ("1×1" if l.k == 1 else f"{l.k}×{l.k} s{l.s}") + \
                (f" d{l.d}" if l.d > 1 else "")
            if tl.mode == "up":
                out.append(f"{l.ic} 上采样 ×2 {l.h}×{l.w}→{l.ho}×{l.wo}: {tl.n_tw}条×{tl.n_bands}段 WT {tl.wt} "
                           f"输出行 {tl.band_rows} 每步 {tl.nocp} 组 × {tl.n_ob} 步 PLANE {tl.plane}")
                continue
            if tl.mode == "pool":
                out.append(f"{l.ic} 全局平均池化 {l.h}×{l.w}: 每 core 组 c, c+4, …, 每组 {tl.n_icb} 个元素 × "
                           f"{tl.icp_pad} 行 × {tl.plane} B")
                continue
            if l.sc:
                kind += f" ×SE(张量{k + 1 + l.sc})"
            if l.src != -1:
                kind += f" 输入=张量{k + 1 + l.src}"
            if l.act == 1:
                kind += " HardSigmoid"
            extra = {"dw": f" 每步 {tl.nocp} 组 × {tl.n_ob} 步", "full": "", "oc": f" OC 分 {tl.n_ob} 块×{tl.nocp} 组" + (" (OC 并行)" if tl.ocpar else ""),
                     "ict": f" IC 分 {tl.n_icb} 步×{tl.icp_pad} 组, OC 分 {tl.n_ob} 份×{tl.nocp} 组",
             "lm": f" LayerNorm {tl.n_icb} 块×2 遍, 每块 {tl.icp_pad} 组",
             "attn": f" 多头自注意力 {l.oc // 64} 头×2 步"}[tl.mode]
            out.append(f"{l.ic}→{l.oc} {kind} {l.h}×{l.w}→{l.ho}×{l.wo}: {tl.n_tw}条×{tl.n_bands}段 "
                       f"WT {tl.wt} 行 {tl.band_rows} ICPpad {tl.icp_pad} PLANE {tl.plane}{extra}")
        return out


def _tap(n_total, off, sizes, strides):
    from aie.helpers.taplib import TensorAccessPattern
    return TensorAccessPattern((1, n_total), off, sizes, strides)


def _u16(buf, off):
    """core 里读权重头的一个 u16 → index"""
    from aie.dialects import arith, memref
    from aie.ir import IndexType, IntegerType
    i32 = IntegerType.get_signless(32)
    ld = lambda o: arith.extui(i32, memref.load(buf, [arith.constant(IndexType.get(), o)]))
    v = arith.addi(ld(off), arith.shli(ld(off + 1), arith.constant(i32, 8)))
    return arith.index_cast(IndexType.get(), v)


def _i32(iv):
    from aie.dialects import arith
    from aie.ir import IntegerType
    return arith.index_cast(IntegerType.get_signless(32), iv)


@iron.jit
def conv_c16(act: In, wts: In, scratch: InOut, out: Out, *, spec: CompileTime[str],
             reps: CompileTime[int] = 1, part: CompileTime[str] = ""):
    layers = from_spec_str(spec)
    P = Plan(layers)
    k0, k1 = map(int, part.split(",")) if part else (0, len(layers))
    n_steps = P.step0[k1] - P.step0[k0]                   # 这一段的槽数 (权重缓冲区按全链的槽号存, 偏移不变)
    i8, u8, i32 = np.dtype[np.int8], np.dtype[np.uint8], np.dtype[np.int32]
    in_ty = np.ndarray[(E_IN,), i8]
    out_ty = np.ndarray[(CW * 16,), u8]
    w_ty = np.ndarray[(P.e_w,), i8]
    acc_ty = np.ndarray[(ACC_CHUNKS * CW * 16,), i32]
    sc_bytes = 6656 if any(l.op == 6 for l in layers[k0:k1]) else SC_BYTES
    sc_ty = np.ndarray[(sc_bytes,), u8]
    kern = ExternalFunction("convC16Chunk", source_file=SRC,
                            arg_types=[in_ty, in_ty, in_ty, w_ty, out_ty, acc_ty, sc_ty, np.int32, np.int32,
                                       np.int32],
                            include_dirs=[KERNEL_INC], compile_flags=[f"-DCW={CW}", *P.kernel_flags(k0, k1)])

    def make_core(cid):
        hc = 32 + 16 * cid          # 本 core 在权重头里的那段

        def call(k, win, wv, o, acc, sc, j, r):
            for _ in range(reps):
                k(win[0], win[1], win[2], wv, o, acc, sc, _i32(j), _i32(r), cid)

        def core(f_in, f_w, f_out, k, acc, dummy, sc):
            for _ in range_(n_steps):
                wv = f_w.acquire(1)
                rows, nch = _u16(wv, hc), _u16(wv, hc + 8)
                for _ in range_(_u16(wv, hc + 2)):          # stride 1 (含 1×1): 滑窗
                    f_in.acquire(2)
                    for r in range_(rows):
                        win = f_in.acquire(3)
                        for j in range_(nch):
                            o = f_out.acquire(1)
                            call(k, win, wv, o, acc, sc, j, r)
                            f_out.release(1)
                        f_in.release(1)
                    f_in.release(2)
                for _ in range_(_u16(wv, hc + 4)):          # stride 2: 每行 3 行
                    for r in range_(rows):
                        win = f_in.acquire(3)
                        for j in range_(nch):
                            o = f_out.acquire(1)
                            call(k, win, wv, o, acc, sc, j, r)
                            f_out.release(1)
                        f_in.release(3)
                for _ in range_(_u16(wv, hc + 10)):         # IC 分块中间步 / 池化累加 / 装 SE 缩放: 不出输出
                    for r in range_(rows):
                        win = f_in.acquire(1)
                        for j in range_(nch):
                            k(win, win, win, wv, dummy, acc, sc, _i32(j), _i32(r), cid)
                        f_in.release(1)
                for _ in range_(_u16(wv, hc + 12)):         # IC 分块最后一步: 出输出
                    for r in range_(rows):
                        win = f_in.acquire(1)
                        for j in range_(nch):
                            o = f_out.acquire(1)
                            k(win, win, win, wv, o, acc, sc, _i32(j), _i32(r), cid)
                            f_out.release(1)
                        f_in.release(1)
                f_w.release(1)
        return core

    fifos, workers = [], []
    f_wb = ObjectFifo(w_ty, depth=P.w_depth, name="c16_wb") if P.w_bcast else None
    # LN / ATTN 内核调用软浮点或软除法，栈帧较大，超过默认 1 KB 栈会覆写 0x70400 处的首个缓冲区
    st_sz = 2048 if any(l.op in (5, 6) for l in layers[k0:k1]) else 1024
    for c in range(N_CORES):
        f_in = ObjectFifo(in_ty, depth=4, name=f"c16_{c}i")
        f_w = f_wb if f_wb is not None else ObjectFifo(w_ty, depth=P.w_depth, name=f"c16_{c}w")
        f_out = ObjectFifo(out_ty, depth=2, name=f"c16_{c}o")
        acc = Buffer(acc_ty, name=f"c16_{c}acc")
        dummy = Buffer(out_ty, name=f"c16_{c}dummy")
        sc = Buffer(sc_ty, name=f"c16_{c}sc")
        fifos.append((f_in, f_w, f_out))
        workers.append(Worker(make_core(c), [f_in.cons(), f_w.cons(), f_out.prod(), kern, acc, dummy, sc],
                              stack_size=st_sz))

    nw = P.n_steps * N_CORES * P.e_w                # 权重缓冲区 [槽][core][e_w]
    in_h = [f_in.prod() for f_in, _, _ in fifos]
    out_h = [f_out.cons() for _, _, f_out in fifos]
    w_h = [f_wb.prod()] if f_wb is not None else [f_w.prod() for _, f_w, _ in fifos]

    def seq(a, w, s, o, *handles):
        bufs = {"act": a, "s": s, "out": o}
        h_in = handles[:N_CORES]
        h_out = handles[N_CORES:2 * N_CORES]
        h_w = handles[2 * N_CORES:]
        pending, pre_tg = [], None

        def wfill(k, ci, tg, wait=False):           # 层 k 第 ci 组的权重 (每个 core 一个 BD, 连续 ns 个槽)
            _, _, ns, s0 = P.chunks(k)[ci]
            if f_wb is not None:                    # 广播: 只读 core 0 那份
                h_w[0].fill(w, tap=_tap(nw, s0 * N_CORES * P.e_w, [1, ns, P.e_w // 256, 256],
                                        [0, N_CORES * P.e_w, 256, 1]), wait=wait, group=tg)
                return
            for c in range(N_CORES):
                h_w[c].fill(w, tap=_tap(nw, (s0 * N_CORES + c) * P.e_w, [1, ns, P.e_w // 256, 256],
                                        [0, N_CORES * P.e_w, 256, 1]), wait=wait, group=tg)

        for k in range(k0, k1):
            l = layers[k]
            if l.op == 3:                                     # 外部输入: 不算
                continue
            (src, soff), (dst, doff) = P.io[k]
            tl = P.tilings[k]
            for ci, (ob0, ob1, _, _) in enumerate(P.chunks(k)):
                # 没有 drain 的 task group (如 LN 的 pass 0) 必须在 fill 上设 wait=True, 否则 finish
                # 只发 dma_free_task 不发 dma_await_task, 后续组会直接覆写尚在传输中的 shim DMA BD
                has_drain = any(P.drain_taps(k, doff, c, t, b, ob0, ob1) for c, t, b in tl.cores())
                if ci == 0 and pre_tg is not None:          # 权重在上一层屏障之前就发了
                    tg, pre_tg = pre_tg, None
                else:
                    tg = TaskGroup()
                    wfill(k, ci, tg)
                pending.append(tg)
                if l.sc and ob0 == 0:                        # SE 缩放向量: 每个 core 层的第一步之前
                    ssrc, sbase = P.loc[k + 1 + l.sc]
                    for c, t, b in tl.cores():
                        h_in[c].fill(bufs[ssrc], tap=_tap(P.size[ssrc], *P.scale_tap(k, sbase)),
                                     group=tg)
                for c, t, b in tl.cores():
                    ftaps = P.fill_taps(k, soff, c, t, b, ob0, ob1)
                    for ti_idx, (o_, sz, st) in enumerate(ftaps):
                        h_in[c].fill(bufs[src], tap=_tap(P.size[src], o_, sz, st),
                                     wait=(not has_drain and ti_idx == len(ftaps) - 1), group=tg)
                for c, t, b in tl.cores():
                    for o_, sz, st in P.drain_taps(k, doff, c, t, b, ob0, ob1):
                        h_out[c].drain(bufs[dst], tap=_tap(P.size[dst], o_, sz, st), wait=True,
                                       group=tg)
                while len(pending) >= TG_INFLIGHT:          # 下一组已经排上了, 再等前一组 (队列不空)
                    pending.pop(0).finish()
            if W_PREFETCH and k + 1 < k1 and P.chunks(k + 1):   # 下一层第一组的权重不依赖这一层, 先发
                pre_tg = TaskGroup()
                wfill(k + 1, 0, pre_tg)
            while pending:                                   # 层间屏障: 下一层要读这一层的输出
                pending.pop(0).finish()

    rt = Runtime(seq, [np.ndarray[(P.size["act"],), i8], np.ndarray[(nw,), i8],
                       np.ndarray[(P.size["s"],), i8], np.ndarray[(P.size["out"],), u8],
                       *in_h, *out_h, *w_h])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


# ---------------------------------------------------------------- 主机端

def to_layout(x: np.ndarray, t: Tensor, size: int) -> np.ndarray:
    c, h, w = x.shape
    p = np.zeros((c, h + 2 * BR, t.ws + 2 * PAD), x.dtype)
    p[:, BR:BR + h, PAD:PAD + w] = x
    flat = p.reshape(c // 16, 16, h + 2 * BR, t.ws + 2 * PAD).transpose(2, 0, 3, 1).reshape(-1)
    buf = np.zeros(size, x.dtype)
    buf[:flat.size] = flat
    return buf


def from_layout(buf: np.ndarray, t: Tensor) -> np.ndarray:
    """返回 (C, H+2·BR, Ws+2PAD) 的整幅 (含边), 方便同时检查真实区域和补宽列是否为零"""
    wp = t.ws + 2 * PAD
    hp = t.h + 2 * BR
    return buf[:t.size].reshape(hp, t.c // 16, wp, 16).transpose(1, 3, 0, 2).reshape(t.c, hp, wp)


@dataclass
class LParam:
    """一层的整数参数。y = act(sat8((sat16((acc + bias + 2^(s1-1)) >> s1) · M[c] + 2^(s2-1)) >> s2))
    hardswish (act 3): yt = y · clamp(y + q3[c], 0, q6[c]), out = sat8((sat16((yt + 2^(t1-1)) >> t1) · M2[c] + …) >> t2)
    默认值 (bias 0, M 1, s2 0, q3 = 3·HS_Q …) 就是随机权重老测试的"只移位"量化。"""
    W: np.ndarray | None
    s1: int = 0
    s2: int = 0
    bias: np.ndarray | None = None
    M: np.ndarray | None = None
    q3: np.ndarray | None = None
    q6: np.ndarray | None = None
    M2: np.ndarray | None = None
    t1: int = 0
    t2: int = 0
    inp: np.ndarray | None = None   # 外部输入层 (op 3) 的数据 (C, H, W), 构造 C16Chain 时写进 act 缓冲区
    gelu: np.ndarray | None = None  # act 5 用: 256 项 int8 查表（int8 值 → int8 值）
    ln: tuple | None = None         # op 5 用: (gamma, beta, s_in, g_out)，各 ic 个 fp32
    attn: tuple | None = None       # op 6 用: (alpha_q20, mul_v[, valid_t])，alpha_q20 为 n_heads 个 int32，mul_v 为 oc 个 int16

    def arr(self, name, n, default):
        v = getattr(self, name)
        return np.full(n, default, np.int64) if v is None else np.asarray(v, np.int64)


def rq_table(l: Layer, p: LParam) -> np.ndarray:
    """[ocp][RQB] 字节: 每个输出通道组的 bias / M (mmul 或 dw 的布局) / hardswish 参数"""
    oc = l.oc
    bias, M = p.arr("bias", oc, 0).reshape(-1, 16), p.arr("M", oc, 1).reshape(-1, 16)
    q3 = p.arr("q3", oc, 3 * HS_Q).reshape(-1, 16)
    q6 = p.arr("q6", oc, 6 * HS_Q).reshape(-1, 16)
    M2 = p.arr("M2", oc, 1).reshape(-1, 16)
    n = oc // 16
    out = np.zeros((n, RQB), np.int8)
    if l.dw:     # [4 像素][16 通道]
        b4 = np.repeat(bias[:, None, :], 4, 1).reshape(n, 64)
        m4 = np.repeat(M[:, None, :], 4, 1).reshape(n, 64)
    else:        # mmul: 低 8 通道 [4 像素][8], 高 8 通道 [4 像素][8]
        b4 = np.concatenate([np.repeat(bias[:, None, :8], 4, 1).reshape(n, 32),
                             np.repeat(bias[:, None, 8:], 4, 1).reshape(n, 32)], 1)
        m4 = np.concatenate([np.repeat(M[:, None, :8], 4, 1).reshape(n, 32),
                             np.repeat(M[:, None, 8:], 4, 1).reshape(n, 32)], 1)
    out[:, :256] = b4.astype(np.int32).view(np.int8).reshape(n, 256)
    out[:, 256:384] = m4.astype(np.int16).view(np.int8).reshape(n, 128)
    for i, v in enumerate((q3, q6, M2)):     # [2 像素][16]
        out[:, 384 + 64 * i: 448 + 64 * i] = np.tile(v, 2).astype(np.int16).view(np.int8).reshape(n, 64)
    return out


def pack_layer(l: Layer, tl: Tiling, p: LParam, e_w: int, ws_in: int = 0, sgn_in=False,
               sc_cp=0) -> list[np.ndarray]:
    """一层的所有槽 × 每个 core 一个 e_w 字节的权重元素 [头][权重][重新量化表], 顺序 [槽][core]"""
    ocp, icp = l.ocp, l.icp
    Wq = p.W
    if l.op:                 # 池化 / 上采样: 没有权重
        full = None
    elif l.dw:     # [cp][ky][kx][4 像素][16 通道] (16 个通道的权重复制 4 份, 对上 4 像素 × 16 通道的向量)
        w4 = Wq.reshape(icp, 16, l.k, l.k).transpose(0, 2, 3, 1)
        full = np.ascontiguousarray(np.broadcast_to(w4[:, :, :, None, :], (icp, l.k, l.k, 4, 16)))
    elif l.k == 3:   # [ocp][ky][icp][kx][h][ob][ic8][oc8]
        full = Wq.reshape(ocp, 2, 8, icp, 2, 8, 3, 3).transpose(0, 6, 3, 7, 4, 1, 5, 2)
    else:          # [ocp][icp][h][ob][ic8][oc8]
        full = Wq.reshape(ocp, 2, 8, icp, 2, 8).transpose(0, 3, 4, 1, 5, 2)
    rq = None if l.op else rq_table(l, p)
    return [_pack_step(l, tl, full, p, rq, e_w, st, c_own, ws_in, sgn_in, sc_cp) for slot in tl.slots()
            for c_own, st in enumerate(slot)]


POOL_S = 22      # 池化均值 = (Σ·M + 2^(S-1)) >> S, M = round(2^S / (H·W)); 255·H·W·M < 2^31


def _pack_step(l, tl, full, p, rq, e_w, st, c_own, ws_in=0, sgn_in=False, sc_cp=0):
    buf = np.zeros(e_w, np.int8)
    if st is None:               # 这个 core 这一槽空闲: 全零头 (所有选择字节为 0)
        return buf
    ocp0, nocp, icp0, nicp, first, last = st[:6]
    hdr = np.zeros(HDR // 4, np.int32)
    hdr[:8] = [0, 0, CW, 1, 1, 0, 1, tl.plane]
    if st is LOAD:               # 装 SE 缩放向量: 走累加段, 每个元素一"行"
        hdr[40:43] = [E_IN // ((CW + 2 * PAD) * 16), (CW + 2 * PAD) * 16, 1]
        per = np.array([tl.pre, 0, 0, 0, 1, 1, 0, 0], np.uint16)
        buf[:HDR] = hdr.view(np.int8)
        buf[32 + 16 * c_own: 48 + 16 * c_own] = per.view(np.int8)
        return buf
    if l.op == 5:                # LayerNorm: 本步 nicp·16 通道的 gamma/beta/s_in/g_out + inv_b/eps 在权重区
        assert p.ln is not None, "op 5 需要 LParam.ln = (gamma, beta, s_in, g_out)"
        c0, c1 = icp0 * 16, (icp0 + nicp) * 16
        gamma, beta, s_in, g_out = (np.asarray(a, np.float32)[c0:c1] for a in p.ln)
        hdr[56] = int(round(np.log2(l.ic)))            # log2(D): mean 用移位 (h[58] 被 nxc 占了)
        hdr[59] = 1                                   # LN 标记
        hdr[60], hdr[61], hdr[62], hdr[63] = st[7], st[8], 0, nicp * 16
        hdr[24] = int(icp0 == 0)                       # 第一块: acc 从零开始
        hdr[:8] = [nicp, 1, tl.wt, 1, 1, 0, tl.nxc, tl.plane]
        buf[:HDR] = hdr.view(np.int8)
        for c, t, b in tl.cores():
            rows = tl.band_rows[b]
            wvalid = min(max(l.wo - t * tl.wt, 0), tl.wt)
            nch = 1 if st[7] == 0 else nicp             # 第一遍每行一次调用；第二遍每个通道组一次
            per = np.array([rows, 0, 0, wvalid, nch, int(st[7] == 0), int(st[7] == 1), 0], np.uint16)
            buf[32 + 16 * c: 48 + 16 * c] = per.view(np.int8)
        ic32 = lambda v: np.rint(np.asarray(v, np.float64) * 4096.0).astype(np.int32)
        assert l.ic & (l.ic - 1) == 0, f"LN 的通道数必须是 2 的幂 (mean 用移位; 实测其他值 AIE2 会挂): {l.ic}"
        inv_b = np.array([1.0 / (l.ic * 524288.0), 1e-6], np.float32)  # 1/(D·2^19) + eps 备用
        par = np.concatenate([ic32(s_in),                                      # m1  Q12
                              np.rint(np.asarray(s_in, np.float64) ** 2 * 524288.0).astype(np.int32),  # m2 Q19
                              np.rint(np.asarray(gamma, np.float64) * 1024.0).astype(np.int32),        # g10 Q10
                              ic32(beta),                                      # b12 Q12
                              np.rint(np.asarray(g_out, np.float64)).astype(np.int32)])  # gq: 整数比例
        buf[HDR: HDR + par.nbytes] = par.view(np.int8)
        buf[HDR + par.nbytes: HDR + par.nbytes + 8] = inv_b.view(np.int8)
        return buf
    if l.op == 6:                # 多头自注意力: 步 2h (sub=0: QKᵀ+softmax) / 步 2h+1 (sub=1: W@V)
        assert p.attn is not None, "op 6 需要 LParam.attn = (alpha_q20, mul_v[, valid_t])"
        alpha_q20, mul_v = p.attn[0], p.attn[1]
        valid_t = int(p.attn[2]) if len(p.attn) > 2 else l.h * l.w
        sub, h = st[7], st[8]
        hdr[59] = 2              # ATTN 标记
        hdr[60] = sub
        hdr[61] = int(alpha_q20[h])
        hdr[62] = valid_t
        hdr[:8] = [4, 4, tl.wt, 1, 1, 0, tl.nxc, tl.plane]
        buf[:HDR] = hdr.view(np.int8)
        for c, t, b in tl.cores():
            wvalid = min(max(l.wo - t * tl.wt, 0), tl.wt)
            per = np.array([3, 0, 0, wvalid, 1, 1, 0, 0] if sub == 0 else [2, 0, 0, wvalid, 2, 0, 1, 0], np.uint16)
            buf[32 + 16 * c: 48 + 16 * c] = per.view(np.int8)
        if sub == 0:
            exp_lut = np.rint(np.exp(-np.arange(256) * (16.0 / 255.0)) * (1 << 15)).astype(np.uint16)
            buf[HDR: HDR + 512] = exp_lut.view(np.int8)
        else:
            mv = np.asarray(mul_v[h * 64: (h + 1) * 64], np.int16)
            buf[HDR: HDR + 128] = mv.view(np.int8)
        return buf
    if l.op == 1:                # 池化: 累加步 (每组前 ne-1 个元素) + 收尾步 (每组最后一个), 参数在 int32[33..39, 43]
        m = round((1 << POOL_S) / (l.h * l.w))
        assert m < 1 << 15, f"池化 {l.h}×{l.w} 太小: M = {m} 放不进 int16"
        hdr[33:40] = [1 if first else 2, l.h, tl.icp_pad, tl.plane, (ws_in + 2 * PAD) // 4, m, POOL_S]
        hdr[43] = tl.n_icb
        hdr[47] = int(sgn_in)
        buf[:HDR] = hdr.view(np.int8)
        ng = len(range(c_own, l.icp, N_CORES))
        per = np.array([ng * (tl.n_icb - 1), 0, 0, 1, 1, 1, 0, 0] if first else [ng, 0, 0, 1, 1, 0, 1, 0], np.uint16)
        buf[32 + 16 * c_own: 48 + 16 * c_own] = per.view(np.int8)
        return buf
    if l.op == 2:                # 上采样: 没有权重; 每 core [行数, s1 滑窗, 0, 有效输出宽, 每窗口块数 = 2 × 组 × 块, 0, 0, 标志]
        hdr[:8] = [nicp, nocp, tl.wt, 1, 1, 0, tl.nxc, tl.plane]
        hdr[57] = 1
        assert tl.nxc & (tl.nxc - 1) == 0
        hdr[58] = tl.nxc.bit_length() - 1
        buf[:HDR] = hdr.view(np.int8)
        for c, t, b in tl.cores():
            rows = tl.band_rows[b] // 2
            wvalid = min(max(l.wo - t * tl.wt, 0), tl.wt)
            last_b = b == tl.n_bands - 1
            flags = (t == 0) | (t == tl.n_tw - 1) << 1 | (b == 0) << 2 | last_b << 3 | (last_b and l.ho == 2 * l.h - 1) << 4
            per = np.array([rows, 1, 0, wvalid, 2 * nocp * tl.nxc, 0, 0, flags], np.uint16)
            buf[32 + 16 * c: 48 + 16 * c] = per.view(np.int8)
        return buf
    if l.dw:
        w = full[ocp0:ocp0 + nocp]
    elif l.k == 3:
        w = full[ocp0:ocp0 + nocp, :, icp0:icp0 + nicp]
    else:
        w = full[ocp0:ocp0 + nocp, icp0:icp0 + nicp]
    hdr[:8] = [nicp, nocp, tl.wt, l.s, l.k, p.s1, tl.nxc, tl.plane]
    hdr[24:26] = [first, last]
    if l.dw:
        hdr[26:29] = [l.d, 1, tl.icp_pad * tl.plane]
    hdr[29:32] = [l.act, int(bool(l.sc)), icp0]
    hdr[48] = sc_cp                  # SE: 只有前 sc_cp 个输入通道组乘缩放 (后面是拼进来的残差)
    flat = np.ascontiguousarray(w).reshape(-1)
    hdr[49:51] = [p.s1, p.s2]
    hdr[52:55] = [flat.size, p.t1, p.t2]     # 重新量化表紧跟在权重后面 (权重字节数都是 64 的倍数)
    buf[:HDR] = hdr.view(np.int8)
    for c, t, b in tl.cores():
        if tl.ocpar and c != c_own:
            continue
        rows = tl.sub_rows(b, st[6])[1] if len(st) > 6 else tl.band_rows[b]
        wvalid = min(max(l.wo - t * tl.wt, 0), tl.wt)
        ict = tl.mode in ("ict", "dw")       # depthwise 也走"每行 acquire 1 + 出输出"那一段
        per = np.array([rows, int(not ict and l.s == 1), int(not ict and l.s == 2), wvalid, nocp * tl.nxc,
                        int(ict and not last), int(ict and last), 0], np.uint16)
        buf[32 + 16 * c: 48 + 16 * c] = per.view(np.int8)
    buf[HDR:HDR + flat.size] = flat
    t = rq[ocp0:ocp0 + nocp].reshape(-1)
    buf[HDR + flat.size: HDR + flat.size + t.size] = t
    if l.act == 5 and last:      # GELU 查表: 只在最后一步 (last=1) 用, 省掉中间 IC 步的 2048 B 使 e_w ≤ 6656 B
        blob = gelu_lut_blob(p.gelu)
        off = HDR + flat.size + t.size
        buf[off: off + blob.size] = blob
    return buf


def conv_acc(l: Layer, x_i8: np.ndarray, Wq: np.ndarray) -> np.ndarray:
    if l.k == 1 and l.s == 1 and not l.dw:
        return (Wq[:, :, 0, 0].astype(np.float64) @ x_i8.reshape(l.ic, -1).astype(np.float64)).astype(np.int32).reshape(l.oc, l.ho, l.wo)
    p = l.pad
    xp = np.zeros((l.ic, l.h + 2 * p, l.w + 2 * p), np.int32)
    xp[:, p:p + l.h, p:p + l.w] = x_i8
    acc = np.zeros((l.oc, l.ho * l.wo), np.int32)
    Wi = Wq.astype(np.int32)
    for ky in range(l.k):
        for kx in range(l.k):
            y0, x0 = ky * l.d, kx * l.d
            patch = xp[:, y0:y0 + l.stride_y * (l.ho - 1) + 1:l.stride_y, x0:x0 + l.stride_x * (l.wo - 1) + 1:l.stride_x]
            if l.dw:
                acc += Wi[:, 0, ky, kx][:, None] * patch.reshape(l.ic, -1)
            else:
                acc += Wi[:, :, ky, kx] @ patch.reshape(l.ic, -1)
    return acc.reshape(l.oc, l.ho, l.wo)


GELU_TBL = 2048   # act 5 (GELU): 每步在重新量化表后面挂 2 KB（256 项 int16 × 2 份 × 2 组，

HS_Q, HS_S = 20, 7   # hardswish: y 的 1.0 = HS_Q; 输出 = (y · clamp(y + 3q, 0, 6q) + 2^(s-1)) >> s


def _rs(v, s):
    return (v + (1 << (s - 1))) >> s if s > 0 else v


def gelu_lut_blob(tbl: np.ndarray) -> np.ndarray:
    """256 项 int8 查表 → AIE2 parallel_lookup 的 2 KB 布局。

    值按 int16 存（8bit 值的查表也走 16 位访问），每 8 个一组（128b 行）重复两次；
    lut_ab / lut_cd 各一份（4 路并行需要 4 份数据）。"""
    v = np.asarray(tbl, np.int16).reshape(32, 8)
    pair = np.repeat(v, 2, axis=0).reshape(-1)          # 512 个 int16
    return np.concatenate([pair, pair]).view(np.int8)     # 2048 字节


def requant(acc, p: LParam | int, act=0):
    """acc (C, H, W) int (已含 bias) → 输出字节 (uint8; act 2 / 3 / 4 是 int8 的位模式, 下一层按 int8 读)。
    p 是 int 时就是老测试的"只移位" (LParam(None, s1=p))"""
    if not isinstance(p, LParam):
        p = LParam(None, s1=p)
    c = acc.shape[0]
    col = lambda v: v.reshape(-1, *([1] * (acc.ndim - 1)))
    t = np.clip(_rs(acc.astype(np.int64), p.s1), -32768, 32767)
    r = _rs(t * col(p.arr("M", c, 1)), p.s2)
    if act in (0, 4):    # ReLU: int8 的 0..127
        return np.clip(r, 0, 127).astype(np.uint8)
    y = np.clip(r, -128, 127)
    if act == 1:     # HardSigmoid: int8 饱和后 +128 (0 → 0.5)
        return (y + 128).astype(np.uint8)
    if act == 3:     # hardswish: 逐通道 q3 / q6, 再重新量化一次
        yt = y * np.clip(y + col(p.arr("q3", c, 3 * HS_Q)), 0, col(p.arr("q6", c, 6 * HS_Q)))
        t2 = np.clip(_rs(yt, p.t1), -32768, 32767)
        y = np.clip(_rs(t2 * col(p.arr("M2", c, 1)), p.t2), -128, 127)
    if act == 5:     # GELU: 逐元素查表（表也是 int8 → int8）
        assert p.gelu is not None, "act 5 需要 LParam.gelu"
        y = np.asarray(p.gelu, np.int8)[y + 128]
    return y.astype(np.int8).view(np.uint8)


def ln_ref(x_u8: np.ndarray, p: LParam) -> np.ndarray:
    """LayerNorm 的定点参考 —— 和内核**逐位同一套**运算 (float32 序列也一样):

      s1 = Σx·m1 (Q12)   s2 = Σx²·m2 (Q19)   mean_q = s1 >> log2(D)
      var = float32(s2)·inv_b − float32(mean_q)²·2^-16      (inv_b = 1/(D·2^19) 主机算好)
      r12 = int32(rsqrt_q12(var + eps)·2^12 + 0.5)          (位技巧 + 3 次 Newton, 向零截断)
      dev = x·m1 − mean_q (Q12) → (dev·r12)>>12 → (·γ10)>>10 → 加 β12 → 乘 gq, >>12
    """
    gamma, beta, s_in, g_out = (np.asarray(a, np.float64) for a in p.ln)
    c, h, w = x_u8.shape
    f32 = np.float32
    m1 = np.rint(s_in * 4096).astype(np.int64)
    m2 = np.rint(s_in ** 2 * 524288).astype(np.int64)
    g10 = np.rint(gamma * 1024).astype(np.int64)
    b12 = np.rint(beta * 4096).astype(np.int64)
    gq = np.rint(g_out).astype(np.int64)                        # 整数比例 (不再乘 2^12)
    x = x_u8.view(np.int8).astype(np.int64)
    s1 = (x * m1.reshape(-1, 1, 1)).sum(0)                     # Q12
    s2 = (x * x * m2.reshape(-1, 1, 1)).sum(0)                 # Q19
    sh = int(np.log2(c))
    mq = s1 >> sh
    inv_b = f32(1.0 / (c * 524288.0))
    # var: 逐元素按 float32 算 (和 C 的表达式求值顺序一致)
    t1 = f32(s2) * inv_b
    mf = mq.astype(f32)
    t3 = (mf * mf) * f32(5.9604644775390625e-08)     # mean_q 是 Q12 → mean² 要乘 2^-24
    var = t1 - t3
    # rsqrt_q12: 位技巧 + 3 次 Newton, 全部 float32
    v = var + f32(1e-06)
    v = np.where(v > f32(0.0), v, f32(1e-06)).astype(f32)
    hh = f32(0.5) * v
    bits = v.view(np.int32).astype(np.int64)                    # 位模式
    bits = (0x5F3759DF - (bits >> 1)).astype(np.int64)
    bits = np.where(bits >= (1 << 31), bits - (1 << 32), bits)   # 回 int32
    rs = bits.astype(np.int32).view(np.float32)
    for _ in range(3):
        rs = rs * (f32(1.5) - hh * rs * rs)
    rr = rs * f32(4096.0) + f32(0.5)
    rr = np.where((rr > f32(-2.0e9)) & (rr < f32(2.0e9)), rr, f32(0.0))
    r12 = np.trunc(rr).astype(np.int64)                          # C 的 f2i 向零截断
    dev = x * m1.reshape(-1, 1, 1) - mq.reshape(1, h, w)
    n12 = (dev * r12.reshape(1, h, w)) >> 12
    y12 = (n12 * g10.reshape(-1, 1, 1)) >> 10
    o = ((y12 + b12.reshape(-1, 1, 1)) * gq.reshape(-1, 1, 1)) >> 12
    return np.clip(o, -128, 127).astype(np.int8).view(np.uint8)


_ATTN_EXP_U16 = np.rint(np.exp(-np.arange(256) * (16.0 / 255.0)) * (1 << 15)).astype(np.int64)


def attn_ref(qkv_u8: np.ndarray, p: LParam) -> np.ndarray:
    """多头自注意力 (op 6) 的定点参考 —— 与内核 attn_step 逐位一致:
    输入 qkv_u8 形状 (3·oc, 4, 35)，按通道拆成 Q, K, V 各 (oc, 4, 35)。"""
    alpha_q20, mul_v = np.asarray(p.attn[0], np.int64), np.asarray(p.attn[1], np.int64)
    c3, h_rows, w_cols = qkv_u8.shape
    oc = c3 // 3
    n_heads = oc // 64
    T = h_rows * w_cols
    valid_t = int(p.attn[2]) if len(p.attn) > 2 else T
    qkv = qkv_u8.view(np.int8).reshape(3, n_heads, 64, T).transpose(0, 1, 3, 2)  # [3, n_heads, T, 64]
    qi, ki, vi = qkv[0], qkv[1], qkv[2]
    out = np.zeros((n_heads, T, 64), np.int8)
    for h in range(n_heads):
        raw = qi[h].astype(np.int64) @ ki[h].T.astype(np.int64)                  # [T, T]
        s16 = np.clip((raw * int(alpha_q20[h]) + 2048) >> 12, -32768, 32767)
        if valid_t < T:
            s16[:, valid_t:] = -32768
        m = s16.max(axis=1, keepdims=True)
        idx = np.clip((m - s16 + 8) >> 4, 0, 255)
        e = _ATTN_EXP_U16[idx]                                                   # [T, T], uint16 Q15
        den = e.sum(axis=1, keepdims=True)
        inv_den = ((255 << 23) + (den >> 1)) // den
        w_u8 = np.clip((e * inv_den + (1 << 22)) >> 23, 0, 255)
        acc_v = w_u8 @ vi[h].astype(np.int64)                                    # [T, 64]
        s_v = np.clip((acc_v + 1) >> 1, -32768, 32767)
        mv = mul_v[h * 64: (h + 1) * 64].reshape(1, 64)
        out[h] = np.clip((s_v * mv + (1 << 14)) >> 15, -128, 127).astype(np.int8)
    return out.transpose(0, 2, 1).reshape(oc, h_rows, w_cols).view(np.uint8)


def pool_ref(x_u8: np.ndarray, sgn=False) -> np.ndarray:
    c, h, w = x_u8.shape
    m = round((1 << POOL_S) / (h * w))
    s = (x_u8.view(np.int8) if sgn else x_u8).astype(np.int64).reshape(c, -1).sum(1)
    r = (s * m + (1 << (POOL_S - 1))) >> POOL_S
    return (np.clip(r, -128, 127).astype(np.int8).view(np.uint8) if sgn else np.minimum(r, 255).astype(np.uint8)) \
        .reshape(c, 1, 1)


def up_ref(x_u8: np.ndarray, ho: int) -> np.ndarray:
    """双线性 ×2 (pytorch_half_pixel) 的整数版: 输出 (2i+a, 2j+b) = (9·x[i,j] + 3·x[i',j] + 3·x[i,j'] + x[i',j'] + 8) >> 4,
    i' = i−1 (a = 0) / i+1 (a = 1), j' 同理, 出界就夹在边上; 再裁到 ho 行。按 int8 算, 返回 uint8 位模式"""
    x = x_u8.view(np.int8).astype(np.int64)

    def nb(a, ax, d):            # 沿 ax 取前 (d=-1) / 后 (d=1) 一个, 边上夹住
        n = a.shape[ax]
        idx = np.clip(np.arange(n) + d, 0, n - 1)
        return np.take(a, idx, ax)

    def axis(a, ax):             # 返回 (3·a + 前, 3·a + 后) 交错 → 长度翻倍
        e, o = 3 * a + nb(a, ax, -1), 3 * a + nb(a, ax, 1)
        return np.stack([e, o], ax + 1).reshape(*a.shape[:ax], 2 * a.shape[ax], *a.shape[ax + 1:])
    y = (axis(axis(x, 1), 2) + 8) >> 4
    return y[:, :ho].astype(np.int8).view(np.uint8)


def signed_in(layers, k) -> bool:
    """层 k 的输入是不是 int8 (输入张量由 act 2 / 3 的层产生; 池化输出跟着输入走)"""
    i = k + layers[k].src
    while i >= 0 and layers[i].op == 1:
        i += layers[i].src
    return i >= 0 and layers[i].act in (2, 3)


RES_W = 64           # 残差的单位权重 (= 2^6; 输出 shift 约 6 时残差和主路同比例相加)


def se_weights(Wq: np.ndarray, s_u8: np.ndarray) -> np.ndarray:
    """SE 的逐通道乘折进下一层 1×1 的权重: W'[o, i] = (W[o, i]·s[i] + 128) >> 8 (内核每步原地做一次);
    只乘前 len(s) 个输入通道 (后面是拼接进来的残差)"""
    n = s_u8.size
    w = Wq.copy()
    w[:, :n] = ((Wq[:, :n].astype(np.int32) * s_u8.reshape(1, -1, 1, 1).astype(np.int32) + 128) >> 8).astype(np.int8)
    return w


def make_case(layers, seed=0, rq=False):
    """随机输入和权重, 按层的 src / sc 连成图, 返回 (输入, 每层 LParam, 末层参考)。
    rq: 随机 bias 和逐通道 M (测完整的重新量化路径); 否则只移位 (bias 0, M 1)"""
    rng = np.random.default_rng(seed)
    x = rng.integers(-100, 101, (layers[0].ic, layers[0].h, layers[0].w), dtype=np.int64).astype(np.int8)
    params, ts = [], [x]
    P = Plan(layers)                 # 只用来拿存储分组 (窗口读) —— 不编译
    for k, l in enumerate(layers):
        if l.op == 3:                # 外部输入: 随机, 放进 LParam.inp (C16Chain 构造时写进 act 缓冲区)
            xi = rng.integers(-100, 101, (l.oc, l.h, l.w), dtype=np.int64).astype(np.int8).view(np.uint8)
            params.append(LParam(None, inp=xi))
            ts.append(xi)
            continue
        cur = np.concatenate([ts[m] for m in P.window(k)]) if l.ic != ts[k + 1 + l.src].shape[0] else ts[k + 1 + l.src]
        if l.op == 1:
            params.append(LParam(None))
            ts.append(pool_ref(cur.view(np.uint8), signed_in(layers, k)))
            continue
        if l.op == 2:
            params.append(LParam(None))
            ts.append(up_ref(cur, l.ho))
            continue
        Wq = rng.integers(-12, 13, (l.oc, 1 if l.dw else l.ic, l.k, l.k)).astype(np.int8)
        if l.op == 5:                # LN: 随机参数 (gamma≈1, beta≈0, s_in≈0.02, g_out≈40)
            p_ = LParam(None, ln=(rng.normal(1, 0.1, l.ic).astype(np.float32),
                                  rng.normal(0, 0.1, l.ic).astype(np.float32),
                                  np.full(l.ic, 0.02, np.float32),
                                  np.full(l.ic, 40.0, np.float32)))
            params.append(p_)
            ts.append(ln_ref(cur.view(np.uint8), p_))
            continue
        if l.op == 6:                # ATTN: 随机每头分数比例 alpha_q16 和逐通道 V 重新量化比例 mul_v
            n_heads = l.oc // 64
            alpha_q16 = rng.integers(50, 100, n_heads, dtype=np.int32)
            mul_v = rng.integers(220, 300, l.oc, dtype=np.int16)
            p_ = LParam(None, attn=(alpha_q16, mul_v, l.h * l.w))
            params.append(p_)
            ts.append(attn_ref(cur.view(np.uint8), p_))
            continue
        gelu = None
        if l.act == 5:               # GELU 查表: 随机表（任何一个 bank 摆错都会露出来）；
            gelu = rng.integers(-120, 121, 256).astype(np.int8)
            gelu[128] = 0            # 真表的中心项必须为 0（gelu(0)=0），否则补宽列会被写成非零
        if l.cat:                    # 残差加: 拼进来的 B (= 块的输入) 用单位权重 RES_W 累加进同一个输出通道
            b = ts[k + 1 + l.cat]
            assert b.shape[0] == l.oc and l.k == 1
            Wq[:, -l.oc:, 0, 0] = np.eye(l.oc, dtype=np.int8) * RES_W
        acc = conv_acc(l, cur.view(np.int8), se_weights(Wq, ts[k + 1 + l.sc]) if l.sc else Wq)
        if l.act in (1, 2, 3):   # 有符号: 让 y = acc >> s 铺开 (标准差 ~40), 饱和的不太多
            shift = min(range(3, 18), key=lambda s: abs(np.clip((acc + (1 << (s - 1))) >> s, -128, 127).std() - 40))
        else:
            shift = min(range(3, 16), key=lambda s: abs(requant(acc, s).mean() - 24))
        p = LParam(Wq, s1=shift)
        p.gelu = gelu
        if rq:                   # bias ±(2^shift·16); M ∈ [2^13, 2^14) 配 s1 = shift-6, s2 = 7: 整体 ≈ 原来的 1–2 倍
            p.bias = rng.integers(-(16 << shift), 16 << shift, l.oc)
            p.s1, p.s2 = max(0, shift - 6), 7 + min(6, shift)
            p.M = rng.integers(1 << 13, 1 << 14, l.oc)
            acc = acc + p.bias.reshape(-1, 1, 1)
            if l.act == 3:       # 逐通道 q: 10..30; yt 大约到 127·6q → t1 = 3, M2 ~ 2^12, t2 = 13
                q = rng.integers(10, 31, l.oc)
                p.q3, p.q6, p.M2, p.t1, p.t2 = 3 * q, 6 * q, rng.integers(1 << 11, 1 << 12, l.oc), 3, 13
        params.append(p)
        ts.append(requant(acc, p, l.act))
        if l.al:                     # 就地覆盖: 之后读那个张量读到的是新值
            ts[k + 1 + l.al] = ts[-1]
    return x, params, ts[-1]


class C16Chain:
    """parts: 把链按层号切成几次 NPU 调用 [(k0, k1), …] (每段单独编译, 只编自己用到的内核分支; 共用所有缓冲区)"""

    def __init__(self, layers, params, reps=1, parts=None, *, w_arr=None):
        self.layers, self.reps = layers, reps
        self.P = P = Plan(layers)
        self.spec = spec_str(layers)
        self.parts = parts or [(0, len(layers))]
        self.t_act = iron.zeros(P.size["act"], dtype=np.int8, device="npu")
        if params is not None:
            for k, (l, p) in enumerate(zip(layers, params)):
                if l.op == 3 and p.inp is not None:
                    self.write_tensor(k + 1, p.inp, sync=False)
        self.t_act._sync_to_device()
        if w_arr is not None:
            self.t_w = iron.tensor(w_arr, dtype=np.int8, device="npu")
        else:
            steps = [b for k, (l, tl, p) in enumerate(zip(layers, P.tilings, params))
                     for b in pack_layer(l, tl, p, P.e_w, P.tensors[P.inp[k]].ws, signed_in(layers, k),
                                         P.tensors[k + 1 + l.sc].c // 16 if l.sc else 0)]
            w_arr = np.concatenate(steps)
            self.t_w = iron.tensor(w_arr, dtype=np.int8, device="npu")
        if P.w_bcast:                                # 广播要求每槽 4 个 core 的权重元素相同
            w4 = np.asarray(w_arr).reshape(-1, N_CORES, P.e_w)
            assert (w4 == w4[:, :1]).all(), "w_bcast: 各 core 的权重元素不同"
        self.t_s = iron.zeros(P.size["s"], dtype=np.int8, device="npu")
        self.t_out = iron.zeros(P.size["out"], dtype=np.uint8, device="npu")

    def set_input(self, x):
        self.write_tensor(0, x)

    def write_tensor(self, i, x, sync=True):
        """主机写张量 i (外部输入, 在 act 缓冲区里; 可能和别的张量共用一块存储, 只写自己的通道组)"""
        P, t = self.P, self.P.tensors[i]
        buf, off = P.loc[i]
        assert buf == "act", f"张量 {i} 不在 act 缓冲区, 主机写不了"
        ctot = t.ctot or t.c
        root = off - t.g0 * t.cplane
        a = self.t_act.data.view(np.uint8)
        full = a[root: root + (t.h + 2 * BR) * (ctot // 16) * t.cplane].reshape(t.h + 2 * BR, ctot // 16, -1, 16)
        g = np.zeros((t.c, t.h, t.ws), np.uint8)
        g[:, :, :t.w] = x.view(np.uint8)[:, :t.h, :t.w]
        full[BR:BR + t.h, t.g0:t.g0 + t.c // 16, PAD:PAD + t.ws] = g.reshape(t.c // 16, 16, t.h, t.ws).transpose(2, 0, 3, 1)
        if sync:
            self.t_act._sync_to_device()

    def call(self):
        for k0, k1 in self.parts:
            conv_c16(self.t_act, self.t_w, self.t_s, self.t_out, spec=self.spec, reps=self.reps,
                     part=f"{k0},{k1}")

    def result_full(self):
        return from_layout(self.t_out.numpy(), self.P.tensors[-1])


def check(full: np.ndarray, y: np.ndarray, t: Tensor) -> int:
    """真实区域与参考的最大差; 边和补宽列必须是 0 (否则返回 999)"""
    real = full[:, BR:BR + t.h, PAD:PAD + t.w]
    d = int(np.abs(real.astype(int) - y.astype(int)).max())
    mask = np.ones(full.shape, bool)
    mask[:, BR:BR + t.h, PAD:PAD + t.w] = False
    return d if not full[mask].any() else 999


def _cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def run_one(spec, iters=50, reps=1, seed=0, verbose=False, rq=False, parts=None):
    layers = chain_layers(spec)
    x, params, y = make_case(layers, seed, rq)
    ch = C16Chain(layers, params, reps, parts)
    if verbose:
        for line in ch.P.describe():
            print(f"    {line}", file=sys.stderr)
    ch.set_input(x)
    t0 = time.perf_counter()
    ch.call()
    jit_s = time.perf_counter() - t0
    d = check(ch.result_full(), y, ch.P.tensors[-1])
    walls, cpus = [], []
    for _ in range(iters):
        c0, t0 = _cpu(), time.perf_counter()
        ch.call()
        walls.append(time.perf_counter() - t0)
        cpus.append(_cpu() - c0)
    d = max(d, check(ch.result_full(), y, ch.P.tensors[-1]))
    walls.sort()
    return dict(spec=spec, n=len(layers), steps=ch.P.n_steps, max_diff=d, med=walls[len(walls) // 2] * 1e3,
                best=walls[0] * 1e3, cpu=float(np.mean(cpus)) * 1e3, jit=jit_s,
                gmac=sum(l.macs for l in layers) / 1e9,
                stats=f"末层 均值 {y.mean():.1f} 零 {(y == 0).mean() * 100:.0f}%")


def report(r):
    print(f"[c16] {r['spec']:<14s} N={r['n']:<3d} 步 {r['steps']:<3d} 最大差 {r['max_diff']}  中位 {r['med']:.2f} ms  "
          f"最快 {r['best']:.2f} ms  CPU {r['cpu']:.3f} ms/次  ({r['gmac']:.3f} GMAC, 首次 {r['jit']:.1f} s; "
          f"{r['stats']})", file=sys.stderr, flush=True)


def compute_split(spec, reps=(1, 2, 3), iters=50):
    res = [run_one(spec, iters, r) for r in reps]
    for r, x in zip(reps, res):
        print(f"[c16] {spec} reps={r} 最大差 {x['max_diff']} 中位 {x['med']:.2f} ms 最快 {x['best']:.2f} ms",
              file=sys.stderr, flush=True)
    for key in ("med", "best"):
        slope, icpt = np.polyfit(np.array(reps, float), np.array([x[key] for x in res]), 1)
        n = res[0]["n"]
        print(f"[c16] ({key}) {spec}: 纯计算 {slope:.3f} ms; 截距 {icpt:.3f} ms → 每层额外 ≈ "
              f"{(icpt - 0.25) / n:.3f} ms (固定按 0.25)", file=sys.stderr)


def selftest(spec="enc", iters=10, rq=False, parts=None) -> SelfTestResult:
    """npu.check 用: 一条链逐字节一致 (含补宽列为零), 返回中位耗时; rq: 随机 bias / 逐通道 M;
    parts: 分几次调用 [(k0, k1), …] (共用缓冲区, 每段只编自己的内核分支)"""
    r = run_one(spec, iters, rq=rq, parts=parts)
    report(r)
    return SelfTestResult(r["max_diff"] == 0, r["max_diff"], r["med"])


def main(iters=50):
    ok = True
    for spec in ["e1", "e1,e2", "head", "enc", "low", "deep"]:
        r = run_one(spec, iters, verbose=(spec in ("enc", "deep")))
        report(r)
        ok &= r["max_diff"] == 0
    return SelfTestResult(ok, 0 if ok else 1, r["med"])


if __name__ == "__main__":
    def arg(name, default):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
    iters = int(arg("--iters", 50))
    if "--compute" in sys.argv:
        compute_split(arg("--compute", "enc"), iters=iters)
        sys.exit(0)
    if "--spec" in sys.argv:
        r = run_one(arg("--spec", "enc"), iters, verbose=True, rq="--rq" in sys.argv)
        report(r)
        sys.exit(0 if r["max_diff"] == 0 else 1)
    sys.exit(0 if main(iters) else 1)
