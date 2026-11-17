// 通用 int8 卷积 + ReLU, "16 通道像素"布局, 每次调用只算一个固定大小的输出块 (AIE2 / Phoenix)
//
// 用于 npu/chain_c16.py。和 conv_generic.cc 的区别:
//   · 激活布局 [H+2][CP][PAD + Ws + PAD][16]: 每个像素 16 个通道连续存 (CP = C/16)。
//     这样一个输出块 = 1 个通道组 (16 ch) × CW 个像素 = CW·16 字节, 是 DDR 布局里连续的一段,
//     任何层 (OC、宽度不同) 的一行输出都能拆成整数个同样大小的块 → 输出 ObjectFifo 元素大小对所有层相同。
//   · 每次调用算一个块: 块号 j → (ocp = j / NXC, xc = j % NXC), 像素 [xc·CW, xc·CW + CW)。
//   · 宽度可以补齐: 像素 ≥ wvalid (本 core 条带的真实宽度, 描述符里按 core 给) 的输出写 0,
//     所以存储宽度 Ws 可以大于真实宽度, 补出来的列对下一层就是零填充。整组 (8 像素) 都在 wvalid 之外时
//     不做乘加也不读输入, 所以输入元素每块只需要放到 wvalid 向上取 8 的倍数为止 (主机侧据此定 PLANE)。
//
// mmul<4,8,8>: A = 4 像素 × 8 输入通道。16 通道像素的 64 字节向量 (4 像素) 用 interleave_unzip(·, 8)
// 拆成低 8 通道 / 高 8 通道两个 A, 各喂 2 个输出通道块。
//
// 输入元素: [ICP_pad][PLANE] 字节 (PLANE = 描述符给的块步长, 运行时); 每块前 (s·WT + 2·PAD)·16 字节是真实像素。
// 权重:     [HDR][权重], 每个输出通道组 ocp:
//             3×3: [ky][icp][kx][h][ob][ic8][oc8]   (h = 输入通道的低/高 8 个, ob = 输出的低/高 8 个)
//             1×1: [icp][h][ob][ic8][oc8]
// 输出块:   [CW][16] uint8 = sat_u8((Σ x·w + 2^(s-1)) >> s)
//
// IC 分块 (只用于 1×1, 输入通道太多、一行放不进一个输入元素时): 一层拆成若干步, 每步只带一部分输入通道,
// 部分和存在 core 本地的 acc 缓冲区 (int32, 每块 CW·16 个, 按 [行][块] 排), 描述符里的 first/last 决定
// 这一步是从零开始还是接着累加、结束时写回 acc 还是量化成输出。
#define NOCPP
#include <stdint.h>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef CW
#define CW 40 // 每个输出块的像素数 (8 的倍数)
#endif
#define HDR 256
#define PAD 4

using MM = ::aie::mmul<4, 8, 8, int8, int8>;
using V32 = ::aie::vector<int8, 32>;
using V64 = ::aie::vector<int8, 64>;
using V128 = ::aie::vector<int8, 128>;

namespace {

struct Desc {
  int icp, ocp, wt, stride, k, shift, nxc, plane;
  int act;    // 0 / 4 ReLU (int8, 0..127)  1 HardSigmoid  2 线性  3 hardswish  5 GELU(查表)
  int s1, s2; // 重新量化: y = sat8((sat16((acc + 2^(s1-1)) >> s1) · M[c] + 2^(s2-1)) >> s2)
  int lo;     // y 的下限: ReLU 0, 其余 -128
};

// 每个输出通道组的重新量化表 (RQB 字节, 在这一步的权重后面, 偏移 h[52]):
//   [0, 256)   bias int32: mmul 布局 = 低 8 通道 [4 像素][8] + 高 8 通道 [4 像素][8]; dw = [4 像素][16]
//   [256, 384) M    int16: 同样的布局 (mmul: 低 8 / 高 8 各 32 个; dw: 64 个)
//   [384, 576) hardswish: q3 / q6 / M2 各 int16[2 像素][16]
#define RQB 576
using V32U = ::aie::vector<uint8, 32>;
using ACC32 = ::aie::accum<acc32, 32>;
using V32S = ::aie::vector<int16, 32>;

// 重新量化 32 个 int16 → int8, 再按下限 (ReLU) 截
inline V32 fin(const V32S &v, const int8_t *M, const Desc &d) {
  const ACC32 a = ::aie::mul(v, ::aie::load_v<32>((const int16_t *)M));
  return ::aie::max(a.template to_vector<int8>(d.s2), (int8_t)d.lo);
}

// 4 像素组里只有前 nv 个像素在真实宽度内 (只有 1×1 图的 FC 会遇到): 后面的清零
__attribute__((noinline)) void zero_tail(uint8_t *o, int nv) {
  for (int b = nv * 16; b < 64; b++) o[b] = 0;
}

inline ::aie::accum<acc32, 32> ld_acc(const int32_t *p) {
  ::aie::accum<acc32, 32> a;
  a.from_vector(::aie::load_v<32>(p), 0);
  return a;
}

// 4 像素 × 16 通道 → (低 8 通道 A, 高 8 通道 A)
inline void split(const V64 &a, V32 &lo, V32 &hi) {
  auto [e, o] = ::aie::interleave_unzip(a.extract<32>(0), a.extract<32>(1), 8);
  lo = e;
  hi = o;
}

// 一个像素组 (4 像素) 的 A 和一个 kx 的权重 (256 B: [h][ob][8][8]) 累加进 (c0, c1)
inline void mac2(MM &c0, MM &c1, const V64 &a, const int8_t *w) {
  V32 lo, hi;
  split(a, lo, hi);
  c0.mac(lo, ::aie::load_v<64>(w));
  c1.mac(lo, ::aie::load_v<64>(w + 64));
  c0.mac(hi, ::aie::load_v<64>(w + 128));
  c1.mac(hi, ::aie::load_v<64>(w + 192));
}

// 输出 4 像素 × 16 通道 (64 B); nv = 其中几个像素在真实宽度内 (有 bias 以后宽度外不会自然是 0, 要清)
inline void store4(uint8_t *o, MM &c0, MM &c1, const int8_t *rq, const Desc &d, int nv) {
  if (nv <= 0) {
    ::aie::store_v(o, ::aie::zeros<uint8, 64>());
    return;
  }
  auto [p, q] = ::aie::interleave_zip(fin(c0.template to_vector<int16>(d.s1), rq + 256, d),
                                      fin(c1.template to_vector<int16>(d.s1), rq + 320, d), 8);
  ::aie::store_v((int8_t *)o, p);
  ::aie::store_v((int8_t *)o + 32, q);
  if (nv < 4) zero_tail(o, nv);
}

// 累加器初值 = bias (mmul 布局: c0 = 低 8 通道, c1 = 高 8 通道)
inline void bias_init(MM &c00, MM &c01, MM &c10, MM &c11, const int8_t *rq) {
  c00 = MM(ld_acc((const int32_t *)rq));
  c01 = MM(ld_acc((const int32_t *)rq + 32));
  c10 = c00;
  c11 = c01;
}

void conv3s1(int8_t **rows, const int8_t *w0, uint8_t *out, const Desc &d, int px0, int wvalid, const int8_t *rq) {
  for (int x = 0; x < CW; x += 8) {
    if (px0 + x >= wvalid) { // 整组都是补出来的列: 只写零
      ::aie::store_v(out + x * 16, ::aie::zeros<uint8, 64>());
      ::aie::store_v(out + x * 16 + 64, ::aie::zeros<uint8, 64>());
      continue;
    }
    MM c00, c01, c10, c11;
    bias_init(c00, c01, c10, c11, rq);
    const int8_t *w = w0;
    for (int ky = 0; ky < 3; ky++) {
      const int8_t *p = rows[ky] + (PAD + px0 + x) * 16;
      for (int icp = 0; icp < d.icp; icp++) {
        const V64 L = ::aie::load_v<64>(p - 64);
        const V64 C0 = ::aie::load_v<64>(p);
        const V64 C1 = ::aie::load_v<64>(p + 64);
        const V64 R = ::aie::load_v<64>(p + 128);
        p += d.plane;
        mac2(c00, c01, ::aie::shuffle_up_fill(C0, L, 16), w);
        mac2(c10, c11, ::aie::shuffle_up_fill(C1, C0, 16), w);
        mac2(c00, c01, C0, w + 256);
        mac2(c10, c11, C1, w + 256);
        mac2(c00, c01, ::aie::shuffle_down_fill(C0, C1, 16), w + 512);
        mac2(c10, c11, ::aie::shuffle_down_fill(C1, R, 16), w + 512);
        w += 768;
      }
    }
    store4(out + x * 16, c00, c01, rq, d, wvalid - (px0 + x));
    store4(out + (x + 4) * 16, c10, c11, rq, d, wvalid - (px0 + x + 4));
  }
}

// stride 2: 输出像素 x ← 输入 2x-1, 2x, 2x+1; 16 字节 = 1 像素, filter_even/odd(·, 16) 拆奇偶像素
void conv3s2(int8_t **rows, const int8_t *w0, uint8_t *out, const Desc &d, int px0, int wvalid, const int8_t *rq) {
  for (int x = 0; x < CW; x += 8) {
    if (px0 + x >= wvalid) { // 整组都是补出来的列: 只写零
      ::aie::store_v(out + x * 16, ::aie::zeros<uint8, 64>());
      ::aie::store_v(out + x * 16 + 64, ::aie::zeros<uint8, 64>());
      continue;
    }
    MM c00, c01, c10, c11;
    bias_init(c00, c01, c10, c11, rq);
    const int8_t *w = w0;
    for (int ky = 0; ky < 3; ky++) {
      const int8_t *p = rows[ky] + (PAD + 2 * (px0 + x)) * 16;
      for (int icp = 0; icp < d.icp; icp++) {
        const V64 L = ::aie::load_v<64>(p - 64);   // 2x0-4 .. 2x0-1
        const V64 V0 = ::aie::load_v<64>(p);       // 2x0   .. 2x0+3
        const V64 V1 = ::aie::load_v<64>(p + 64);  // 2x0+4 .. 2x0+7
        const V64 V2 = ::aie::load_v<64>(p + 128);
        const V64 V3 = ::aie::load_v<64>(p + 192); // .. 2x0+15
        p += d.plane;
        const V128 g0 = ::aie::concat(V0, V1), g1 = ::aie::concat(V2, V3);
        const V64 E0 = ::aie::filter_even(g0, 16), O0 = ::aie::filter_odd(g0, 16);
        const V64 E1 = ::aie::filter_even(g1, 16), O1 = ::aie::filter_odd(g1, 16);
        mac2(c00, c01, ::aie::shuffle_up_fill(O0, L, 16), w);
        mac2(c10, c11, ::aie::shuffle_up_fill(O1, V1, 16), w);
        mac2(c00, c01, E0, w + 256);
        mac2(c10, c11, E1, w + 256);
        mac2(c00, c01, O0, w + 512);
        mac2(c10, c11, O1, w + 512);
        w += 768;
      }
    }
    store4(out + x * 16, c00, c01, rq, d, wvalid - (px0 + x));
    store4(out + (x + 4) * 16, c10, c11, rq, d, wvalid - (px0 + x + 4));
  }
}

void conv1(const int8_t *r1, const int8_t *w0, uint8_t *out, const Desc &d, int px0, int wvalid,
           int32_t *acc, int first, int last, const int8_t *rq) {
  for (int x = 0; x < CW; x += 8) {
    if (px0 + x >= wvalid) { // 整组都是补出来的列 (acc 也不用管: 收尾步会写零)
      if (last) {
        ::aie::store_v(out + x * 16, ::aie::zeros<uint8, 64>());
        ::aie::store_v(out + x * 16 + 64, ::aie::zeros<uint8, 64>());
      }
      continue;
    }
    int32_t *ap = acc + (x / 8) * 128; // 每 8 像素 4 个累加器 × 32 = 128 个 int32
    MM c00, c01, c10, c11;
    if (first) {
      bias_init(c00, c01, c10, c11, rq);
    } else {
      c00 = MM(ld_acc(ap));
      c01 = MM(ld_acc(ap + 32));
      c10 = MM(ld_acc(ap + 64));
      c11 = MM(ld_acc(ap + 96));
    }
    const int8_t *w = w0;
    const int8_t *p = r1 + (PAD + px0 + x) * 16;
    for (int icp = 0; icp < d.icp; icp++) {
      const V64 C0 = ::aie::load_v<64>(p);
      const V64 C1 = ::aie::load_v<64>(p + 64);
      p += d.plane;
      mac2(c00, c01, C0, w);
      mac2(c10, c11, C1, w);
      w += 256;
    }
    if (last) {
      store4(out + x * 16, c00, c01, rq, d, wvalid - (px0 + x));
      store4(out + (x + 4) * 16, c10, c11, rq, d, wvalid - (px0 + x + 4));
    } else {
      ::aie::store_v(ap, c00.template to_vector<int32>(0));
      ::aie::store_v(ap + 32, c01.template to_vector<int32>(0));
      ::aie::store_v(ap + 64, c10.template to_vector<int32>(0));
      ::aie::store_v(ap + 96, c11.template to_vector<int32>(0));
    }
  }
}


// ---------------------------------------------------------------- depthwise (逐通道) k×k, stride s, dilation dil
// 输入元素 [k 行][m 个通道组][PLANE]: DMA 已经按 (s·y + ky·dil - pad) 把每个输出行要的 k 行放好 (隔行/重复都由 TAP 做),
// 行步长 = m·PLANE (描述符 int32[28])。权重 [通道组][ky][kx][4 像素][16 通道] (主机侧把 16 个通道的权重复制 4 份,
// 正好对上 4 像素 × 16 通道的 64 字节向量)。水平偏移 o = kx·dil - pad ∈ [-4, 4] 像素, 用 shuffle 从相邻向量拼。
// stride 2: 先按 stride 1 算 2 倍的位置, 再取偶数像素 (depthwise 计算量小, 多算一倍换简单)。
using V64U = ::aie::vector<uint8, 64>;
using ACC64 = ::aie::accum<acc32, 64>;

template <int O> inline V64 hshift(const V64 &L, const V64 &C, const V64 &R) {
  if constexpr (O == 0) return C;
  else if constexpr (O <= -4) return L;
  else if constexpr (O >= 4) return R;
  else if constexpr (O < 0) return ::aie::shuffle_up_fill(C, L, (unsigned)(-O * 16));
  else return ::aie::shuffle_down_fill(C, R, (unsigned)(O * 16));
}

// 一行 (ky) 对一个 4 像素组的累加, kx 全展开、移位量是常量
template <int K, int D, int KX = 0>
inline void dw_row(ACC64 &a, const V64 &L, const V64 &C, const V64 &R, const int8_t *w) {
  if constexpr (KX < K) {
    constexpr int O = KX * D - D * (K - 1) / 2;
    a = ::aie::mac(a, hshift<O>(L, C, R), ::aie::load_v<64>(w + KX * 64));
    dw_row<K, D, KX + 1>(a, L, C, R, w);
  }
}

// 4 像素 × 16 通道的累加器 → 重新量化 (M 是 [4 像素][16] 的 int16)
inline V64U cvt(const ACC64 &a, const int8_t *rq, const Desc &d) {
  const ::aie::vector<int16, 64> v = a.template to_vector<int16>(d.s1);
  return ::aie::concat(fin(v.extract<32>(0), rq + 256, d), fin(v.extract<32>(1), rq + 320, d)).template cast_to<uint8>();
}

// stride 2: 先按 stride 1 算 2 倍的位置, 再取偶数像素; S 是运行时的 (省一份模板实例的程序内存)
template <int K, int D>
__attribute__((noinline)) void dwconv(const int8_t *elem, const int8_t *w0, uint8_t *out, int rstride, int S,
                                      int px0, int wvalid, const int8_t *rq, const Desc &d) {
  const ACC64 b0 = ::aie::concat(ld_acc((const int32_t *)rq), ld_acc((const int32_t *)rq + 32));
  for (int x = 0; x < CW; x += 8) {
    if (px0 + x >= wvalid) {
      ::aie::store_v(out + x * 16, ::aie::zeros<uint8, 64>());
      ::aie::store_v(out + x * 16 + 64, ::aie::zeros<uint8, 64>());
      continue;
    }
    // S=1: 输出 x0..x0+7 = 两组 (x0, x0+4); S=2: 每半 (h2) 算 stride 1 的 8 个位置再取偶数
    for (int h2 = 0; h2 < S; h2++) {
      ACC64 a0 = b0, a1 = b0;
      const int8_t *p = elem + (PAD + S * (px0 + x) + 8 * h2) * 16;
      const int8_t *w = w0;
      for (int ky = 0; ky < K; ky++) {
        const V64 Lm = ::aie::load_v<64>(p - 64), C0 = ::aie::load_v<64>(p);
        const V64 C1 = ::aie::load_v<64>(p + 64), Rp = ::aie::load_v<64>(p + 128);
        dw_row<K, D>(a0, Lm, C0, C1, w);
        dw_row<K, D>(a1, C0, C1, Rp, w);
        p += rstride;
        w += K * 64;
      }
      if (S == 1) {
        ::aie::store_v(out + x * 16, px0 + x < wvalid ? cvt(a0, rq, d) : ::aie::zeros<uint8, 64>());
        ::aie::store_v(out + (x + 4) * 16, px0 + x + 4 < wvalid ? cvt(a1, rq, d) : ::aie::zeros<uint8, 64>());
      } else {
        const V64U e = ::aie::filter_even(::aie::concat(cvt(a0, rq, d), cvt(a1, rq, d)), 16);
        ::aie::store_v(out + (x + 4 * h2) * 16, px0 + x + 4 * h2 < wvalid ? e : ::aie::zeros<uint8, 64>());
      }
    }
  }
}

// ---------------------------------------------------------------- 双线性 ×2 上采样 (pytorch_half_pixel)
// 一个输出块 = CW 个输出像素 = CW/2 个输入像素。c / n = 中心行 / 相邻行里这一块的第一个输入像素 (16 通道):
// 输出 (2j, 2j+1) = (9·c[j] + 3·n[j] + 3·c[j∓1] + n[j∓1] + 8) >> 4 (rounding positive_inf 就是 +8)。
// 边上夹住由调用方处理 (相邻行指针换成中心行、左右把边像素抄进 PAD 列)。
inline void upsample2(const int8_t *c, const int8_t *n, uint8_t *out, int px0, int wvalid) {
  const V64 k9 = ::aie::broadcast<int8, 64>(9), k3 = ::aie::broadcast<int8, 64>(3),
            k1 = ::aie::broadcast<int8, 64>(1);
  for (int x = 0; x < CW; x += 8, c += 64, n += 64) { // 4 个输入像素 → 8 个输出像素
    if (px0 + x >= wvalid) {
      ::aie::store_v(out + x * 16, ::aie::zeros<uint8, 64>());
      ::aie::store_v(out + x * 16 + 64, ::aie::zeros<uint8, 64>());
      continue;
    }
    const V64 cl = ::aie::load_v<64>(c - 64), cc = ::aie::load_v<64>(c), cr = ::aie::load_v<64>(c + 64);
    const V64 nl = ::aie::load_v<64>(n - 64), nc = ::aie::load_v<64>(n), nr = ::aie::load_v<64>(n + 64);
    ACC64 e = ::aie::mul(cc, k9), o = ::aie::mul(cc, k9);
    e = ::aie::mac(e, nc, k3);
    o = ::aie::mac(o, nc, k3);
    e = ::aie::mac(e, hshift<-1>(cl, cc, cr), k3);
    o = ::aie::mac(o, hshift<1>(cl, cc, cr), k3);
    e = ::aie::mac(e, hshift<-1>(nl, nc, nr), k1);
    o = ::aie::mac(o, hshift<1>(nl, nc, nr), k1);
    const auto [lo, hi] = ::aie::interleave_zip(e.template to_vector<int8>(4), o.template to_vector<int8>(4), 16);
    ::aie::store_v(out + x * 16, lo.template cast_to<uint8>());
    ::aie::store_v(out + x * 16 + 64, hi.template cast_to<uint8>());
  }
}

// 一次调用: 滑窗 (r0, r1, r2) = 输入行 i-1, i, i+1; j = (子行 a, 通道组 g, 块 xc); 每 core 段 [7] = 边界标志
// (1 左边 2 右边 4 顶段 8 底段 16 最后一行要裁掉)
__attribute__((noinline)) void upsample_step(int8_t *r0, int8_t *r1, int8_t *r2, const int8_t *wts, uint8_t *out,
                                             int j, int r, int core) {
  const int32_t *h = (const int32_t *)wts;
  const uint16_t *pc = (const uint16_t *)(wts + 32 + 16 * core);
  const int per = h[1] * h[6], a = j >= per, rest = a ? j - per : j; // 不用除法 (AIE2 没有除法指令, 软件除法很占程序内存)
  const int g = rest >> h[58], xc = rest & (h[6] - 1);               // h[58] = log2(NXC), NXC 是 2 的幂
  const int f = pc[7], last = r == pc[0] - 1;
  int8_t *ce = r1 + g * h[7] + PAD * 16, *ne = (a ? r2 : r0) + g * h[7] + PAD * 16;
  if ((a == 0 && (f & 4) && r == 0) || (a && (f & 8) && last))
    ne = ce;
  if (f & 1) { // 左边夹住: 像素 0 抄进像素 -1 (PAD 列; 元素在本地, 可以写)
    ::aie::store_v(ce - 16, ::aie::load_v<16>(ce));
    ::aie::store_v(ne - 16, ::aie::load_v<16>(ne));
  }
  if (f & 2) { // 右边夹住: 最后一个有效输入像素抄到它后面一列
    const int wl = pc[3] / 2;
    ::aie::store_v(ce + wl * 16, ::aie::load_v<16>(ce + (wl - 1) * 16));
    ::aie::store_v(ne + wl * 16, ::aie::load_v<16>(ne + (wl - 1) * 16));
  }
  upsample2(ce + xc * (CW / 2) * 16, ne + xc * (CW / 2) * 16, out, xc * CW, (a && (f & 16) && last) ? 0 : pc[3]);
}

// ---------------------------------------------------------------- SE (squeeze-excitation)
using V16U = ::aie::vector<uint8, 16>;
using V32U = ::aie::vector<uint8, 32>;
using ACC32 = ::aie::accum<acc32, 32>;

// 缩放向量装载: 输入元素 e 是 C×1×1 张量的连续 E_IN 字节, 每 cpl 字节一个通道组, 像素 0 在 PAD·16 处
__attribute__((noinline)) void se_load(const uint8_t *e, uint8_t *sc, int r, int gpe, int cpl) {
  for (int i = 0; i < gpe; i++) {
    const int g = r * gpe + i;
    if (g >= 64) break;
    ::aie::store_v(sc + g * 16, ::aie::load_v<16>(e + i * cpl + PAD * 16));
  }
}

// SE 的逐通道乘折进这一步的 1×1 权重 (原地, 每步一次): W'[o][i] = (W[o][i]·s[i] + 128) >> 8。
// 权重 [ocp][icp][h][ob][ic8][oc8]: 一个 64 B 块的第 i 个字节对应输入通道 h·8 + i/8 → 缩放模式 = 8 个 s 各重复 8 次
__attribute__((noinline)) void se_wscale(int8_t *w, const uint8_t *sc, int n, int nicp, int nocp) {
  for (int icp = 0; icp < n; icp++) {
    const V16U s16 = ::aie::load_v<16>(sc + icp * 16);
    const auto [a0, a1] = ::aie::interleave_zip(s16, s16, 1);       // s0 s0 s1 s1 … (每个 2 次)
    for (int h = 0; h < 2; h++) {
      const V16U a = h ? a1 : a0;
      const auto [b0, b1] = ::aie::interleave_zip(a, a, 2);         // 每个 4 次
      const V32U b = ::aie::concat(b0, b1);
      const auto [c0, c1] = ::aie::interleave_zip(b, b, 4);         // 每个 8 次
      const V64U pat = ::aie::concat(c0, c1);
      int8_t *p = w + icp * 256 + h * 128;
      for (int o = 0; o < nocp; o++) {
        for (int ob = 0; ob < 2; ob++) {
          const ACC64 m = ::aie::mul(::aie::load_v<64>(p + ob * 64), pat);
          ::aie::store_v(p + ob * 64, m.template to_vector<int8>(8));
        }
        p += nicp * 256;
      }
    }
  }
}

// 全局平均池化: 元素 = 一个通道组的 kr 行 (每行 X 字节, 前 nvec·64 字节是整行含零边), 是这个组的第 e 个元素;
// 16 个通道的和在 acc[16·g ..] (int32) 里跨调用累加 (e == 0 时从零开始), last 时出 640 B 块
__attribute__((noinline)) void se_pool(const uint8_t *e, int32_t *acc, uint8_t *out, int ei, int last,
                                       const int32_t *h) {
  const int H = h[34], kr = h[35], X = h[36], nvec = h[37], M = h[38], S = h[39], sg = h[47];
  ACC32 a0 = ::aie::zeros<acc32, 32>(), a1 = ::aie::zeros<acc32, 32>();
  int kv = H - ei * kr;
  if (kv > kr) kv = kr;
  // 输入是 int8 (sg) 时异或 0x80 变成 uint8 (x + 128), 收尾时减掉 128 × 像素数 (零边也算在内, 0 → 128 → 0)
  const V32U ones = ::aie::broadcast<uint8, 32>(1);
  const V64U flip = ::aie::broadcast<uint8, 64>(sg ? 0x80 : 0);
  for (int i = 0; i < kv; i++) {
    const uint8_t *p = e + i * X;
    for (int v = 0; v < nvec; v++) {
      const V64U x = ::aie::bit_xor(::aie::load_v<64>(p), flip);
      a0 = ::aie::mac(a0, x.extract<32>(0), ones);
      a1 = ::aie::mac(a1, x.extract<32>(1), ones);
      p += 64;
    }
  }
  const ::aie::vector<int32, 32> t = ::aie::add(a0.template to_vector<int32>(0), a1.template to_vector<int32>(0));
  ::aie::vector<int32, 16> sum = ::aie::add(t.extract<16>(0), t.extract<16>(1));
  if (ei > 0) sum = ::aie::add(sum, ::aie::load_v<16>(acc));
  if (!last) {
    ::aie::store_v(acc, sum);
    return;
  }
  if (sg) sum = ::aie::sub(sum, 128 * 4 * H * nvec);
  for (int i = 64; i < CW * 16; i += 64) ::aie::store_v(out + i, ::aie::zeros<uint8, 64>());
  // (Σ·M + 2^(S-1)) >> S, 饱和到 uint8 / int8 (M < 2^15)
  const ::aie::accum<acc64, 16> m = ::aie::mul(sum, ::aie::broadcast<int16, 16>((int16_t)M));
  const V16U r = sg ? m.template to_vector<int8>(S).template cast_to<uint8>() : m.template to_vector<uint8>(S);
  const V32U lo = ::aie::concat(r, ::aie::zeros<uint8, 16>());
  ::aie::store_v(out, ::aie::concat(lo, ::aie::zeros<uint8, 32>()));
}

// 输出块的后处理 (块里已经是 int8 的 y): act 1 HardSigmoid = y + 128 (宽度外的像素清零, HardSigmoid(0) ≠ 0);
// act 3 hardswish (逐通道, 参数在重新量化表 [384, 576)): yt = y · clamp(y + q3, 0, q6) (int32),
//       out = sat8((sat16((yt + 2^(t1-1)) >> t1) · M2 + 2^(t2-1)) >> t2)
__attribute__((noinline)) void post_act(uint8_t *o, int act, int nv, const int8_t *rq, int t1, int t2,
                                        const int8_t *tbl) {
  if (act == 1) {
    const auto m = ::aie::broadcast<uint8, 64>(0x80);
#pragma clang loop unroll(disable)
    for (int i = 0; i < CW * 16; i += 64) ::aie::store_v(o + i, ::aie::bit_xor(::aie::load_v<64>(o + i), m));
#pragma clang loop unroll(disable)
    for (int p = nv < 0 ? 0 : nv; p < CW; p++) ::aie::store_v(o + p * 16, ::aie::zeros<uint8, 16>());
    return;
  }
#ifndef NO_GELU
  if (act == 5) {
    // GELU: AIE2 硬件并行查表 (4 路)。表是 256 项 int8（值按 16 位存，每组 8 个重复两次，
    // lut_ab / lut_cd 各 1 KB，见 chain_c16.gelu_lut_blob）。
    // 有符号索引 + bias=128：正好覆盖 int8 全空间，越界检查被优化掉。
    const ::aie::lut<4, int8> L(256, tbl, tbl + 1024);
    ::aie::parallel_lookup<int8, ::aie::lut<4, int8>, ::aie::lut_oor_policy::saturate> pl(L, 0, 128);
    int8_t *q = (int8_t *)o;
    // fetch 的输入向量最多 32 个元素 (AIE2 的并行查表一次 32 个 int8 索引)
#pragma clang loop unroll(disable)
    for (int i = 0; i < CW * 16; i += 32) ::aie::store_v(q + i, pl.fetch(::aie::load_v<32>(q + i)));
    return;   // 宽度外像素本来就是 0，表中心项 = 0 → 仍是 0
  }
#endif
  // 每轮 4 个互不依赖的 32 B (编译器不做软件流水; 一条链 load → … → mul → srs 约 22 个周期, 4 条交错着走)
  int8_t *q = (int8_t *)o;
  const V32S c3 = ::aie::load_v<32>((const int16_t *)(rq + 384));
  const V32S c6 = ::aie::load_v<32>((const int16_t *)(rq + 448));
  const V32S m2 = ::aie::load_v<32>((const int16_t *)(rq + 512));
#pragma clang loop unroll(disable)
  for (int i = 0; i < CW * 16; i += 128) {
    V32S y[4];
    ACC32 m[4];
    for (int u = 0; u < 4; u++) y[u] = ::aie::unpack(::aie::load_v<32>(q + i + 32 * u));
    for (int u = 0; u < 4; u++)
      m[u] = ::aie::mul(y[u], ::aie::min(::aie::max(::aie::add(y[u], c3), (int16_t)0), c6));
    for (int u = 0; u < 4; u++)
      m[u] = ::aie::mul(m[u].template to_vector<int16>(t1), m2);
    for (int u = 0; u < 4; u++) ::aie::store_v(q + i + 32 * u, m[u].template to_vector<int8>(t2));
  }
}

#ifndef NO_LM
// ---- LayerNorm (op 5, 模式 lm) ------------------------------------------------------------
// 每个 core 负责一行; 输入分 n_icb 块送。两遍:
//   pass 0: 逐像素累加 Σx·m1 (Q12) / Σx²·m2 (Q19) 到 acc
//   pass 1: 逐像素算 mean/rstd, 再逐 16 通道组归一化输出
// 参数在权重区 (每个 ic 个 int32): m1(Q12) m2(Q19) γ10(Q10) β12(Q12) gq(整数) + 2 个 fp32:
//   1/(D·2^19) 和 eps。AIE2 没有 sqrtf / 整数除法 / double, 也不支持 float 向量的 reduce_add。

// 1/sqrt(v) 的 Q12 定点: 位技巧初值 + 3 次 Newton。单独一个 noinline 函数 —— 和 var 的计算
// 挤在同一个循环里时, 编译器生成出来的代码会让核心挂死 (实测)。
__attribute__((noinline, minsize)) static int32_t rsqrt_q12(float v) {
  if (!(v > 0.0f)) v = 1e-6f;                 // var 可能被定点误差压成负数
  const float hh = 0.5f * v;
  int32_t bits = *(int32_t *)&v;
  bits = 0x5f3759df - (bits >> 1);
  float rs = *(float *)&bits;
  rs = rs * (1.5f - hh * rs * rs);
  rs = rs * (1.5f - hh * rs * rs);
  rs = rs * (1.5f - hh * rs * rs);
  float rr = rs * 4096.0f + 0.5f;
  if (!(rr > -2.0e9f && rr < 2.0e9f)) rr = 0.0f;      // NaN / 越界 → 0
  return (int32_t)rr;                                 // C 的 f2i 是向零截断
}

// (Σx·m1, Σx²·m2) → (mean_q, r12)。mean_q 是 Q12, 所以 mean² 要乘 2^-24。
__attribute__((noinline, minsize)) static void mean_rstd_q12(int32_t s1, int32_t s2, float inv_b, int sh,
                                                    int32_t *mean_q, int32_t *r12) {
  const int32_t mq = s1 >> sh;                        // / D (D 是 2 的幂)
  const float mf = (float)mq;
  const float var = (float)s2 * inv_b - mf * mf * 5.9604644775390625e-08f;
  *mean_q = mq;
  *r12 = rsqrt_q12(var + 1e-6f);
}

// 向量版: 一次 16 个通道 (一个通道组)。所有乘法用 64 位累加器、移位按向下取整 —— 和整数参考 (int64) 逐位一致,
// 不依赖 int32 不溢出。
using V16B = ::aie::vector<int8, 16>;
using V16H = ::aie::vector<int16, 16>;
using V16I = ::aie::vector<int32, 16>;
using A64 = ::aie::accum<acc64, 16>;

inline V16I ld_x32(const int8_t *p) { return ::aie::unpack(::aie::unpack(::aie::load_v<16>(p))); }

__attribute__((noinline)) void ln_step(const int8_t *in, const int32_t *h, const int8_t *w, uint8_t *out,
                                       int32_t *acc, int j, int wvalid) {
  const int pass = h[60], g0 = h[62], icp = h[0], plane = h[7], ic = h[63];
  const int sh = h[56];                               // log2(D), 主机给出
  const int32_t *q = (const int32_t *)w;
  const int32_t *m1 = q, *m2 = q + ic, *g10 = q + 2 * ic, *b12 = q + 3 * ic, *gq = q + 4 * ic;
  const float inv_b = *(const float *)(q + 5 * ic);
  int32_t *af = acc;
  ::aie::set_rounding(::aie::rounding_mode::floor);
  if (pass == 0) {
    #pragma clang loop unroll(disable)
    for (int p = 0; p < wvalid; p++) {
      A64 a1 = ::aie::zeros<acc64, 16>(), a2 = ::aie::zeros<acc64, 16>();
      #pragma clang loop unroll(disable)
      for (int g = 0; g < icp; g++) {
        const V16I x = ld_x32(in + g * plane + (PAD + p) * 16);
        const int ch = (g0 + g) * 16;
        a1 = ::aie::mac(a1, x, ::aie::load_v<16>(m1 + ch));
        const V16I xx = ::aie::mul(x, x).template to_vector<int32>(0);
        a2 = ::aie::mac(a2, xx, ::aie::load_v<16>(m2 + ch));
      }
      // 每个通道 (lane) 的部分和都在 int32 范围内 (s1 ≤ 127·Σm1, s2 各项非负且 ≤ 总和 < 2^31, 主机按此选 s_in)
      const int32_t s1 = ::aie::reduce_add(a1.template to_vector<int32>(0));
      const int32_t s2 = ::aie::reduce_add(a2.template to_vector<int32>(0));
      if (h[24]) { af[2 * p] = s1; af[2 * p + 1] = s2; }     // 第一块从零开始
      else { af[2 * p] += s1; af[2 * p + 1] += s2; }
    }
    ::aie::set_rounding(::aie::rounding_mode::positive_inf);
    if (h[61]) {
      for (int p = 0; p < wvalid; p++)
        mean_rstd_q12(af[2 * p], af[2 * p + 1], inv_b, sh, &af[2 * p], &af[2 * p + 1]);
    }
    return;
  }
  const int ch = (g0 + j) * 16;
  const V16I vm1 = ::aie::load_v<16>(m1 + ch), vg10 = ::aie::load_v<16>(g10 + ch);
  const V16I vb12 = ::aie::load_v<16>(b12 + ch), vgq = ::aie::load_v<16>(gq + ch);
  #pragma clang loop unroll(disable)
  for (int p = 0; p < wvalid; p++) {
    const int32_t mean_q = af[2 * p], r12 = af[2 * p + 1];
    const V16I x = ld_x32(in + j * plane + (PAD + p) * 16);
    const V16I dev = ::aie::sub(::aie::mul(x, vm1).template to_vector<int32>(0), mean_q);       // Q12
    const V16I n12 = ::aie::mul(dev, ::aie::broadcast<int32, 16>(r12)).template to_vector<int32>(12);
    const V16I y12 = ::aie::mul(n12, vg10).template to_vector<int32>(10);
    const V16I o32 = ::aie::mul(::aie::add(y12, vb12), vgq).template to_vector<int32>(12);
    const V16I oc = ::aie::max(::aie::min(o32, (int32_t)127), (int32_t)-128);
    const V16H o16 = ::aie::accum<acc32, 16>(oc).template to_vector<int16>(0);
    ::aie::store_v((int8_t *)out + p * 16, ::aie::pack(o16));
  }
  ::aie::set_rounding(::aie::rounding_mode::positive_inf);
  for (int p = wvalid < 0 ? 0 : wvalid; p < CW; p++) ::aie::store_v(out + p * 16, ::aie::zeros<uint8, 16>());
}
#endif

#ifndef NO_ATTN
using MM_U8 = ::aie::mmul<4, 8, 8, uint8, int8>;

// 把分段行缓冲 (每行 W 像素, 行距 row_stride*960) 连续拷到 dst[n_rows*W*16] 并补零到 pad_px
__attribute__((noinline, minsize)) void gather_rows_16(const int8_t *in, int n_rows, int row_stride, int g_off, int pad_px,
                                              int8_t *dst, int W) {
  #pragma clang loop unroll(disable)
  for (int rr = 0; rr < n_rows; rr++) {
    const int8_t *src = in + (rr * row_stride + g_off) * 960 + 64;
    int8_t *d = dst + rr * (W * 16);
    #pragma clang loop unroll(disable)
    for (int kt = 0; kt < W; kt++) {
      ::aie::store_v(d + kt * 16, ::aie::load_v<16>(src + kt * 16));
    }
  }
  ::aie::vector<int8, 16> z16 = ::aie::zeros<int8, 16>();
  #pragma clang loop unroll(disable)
  for (int k = n_rows * W; k < pad_px; k++) {
    ::aie::store_v(dst + k * 16, z16);
  }
}

// 多头自注意力 (op 6): 步 2h (h[60]=0: QKᵀ + softmax → W_u8 存 acc) / 步 2h+1 (h[60]=1: W@V → out[40][16])
// W = 每行 token 数 (= 本 core 的 wvalid, 1..35), 4 行共 T = 4W 个 key。缓冲区按 W=35 的最大值排 (行距 70 / 140 / 18 块),
// 循环次数按 W 算 —— 短窗口 (W<35) 时 QKᵀ / softmax / W@V 都只算有效部分。
__attribute__((noinline)) void attn_step(const int8_t *__restrict in, const int32_t *__restrict h,
                                         const int8_t *__restrict w, uint8_t *__restrict out,
                                         int32_t *__restrict acc, uint8_t *__restrict sc_buf, int r, int j, int core,
                                         int W) {
  // acc = 7680 B: [192..5091] S01 (r=2 按 qb 依次被 [0..5183] w_blk 覆写), [5376..7679] q_buf
  // sc_buf = 6144 B: [0..4607] kt_buf, [4608..5183] s23_qb / w_tmp, [5184..5503] s_row, [5504..5823] e_row
  int8_t *q_buf = ((int8_t *)acc) + 5376;                 // 9 * 8 * 32 = 2304 B
  int16_t *s01_all = (int16_t *)(((int8_t *)acc) + 192);   // 35 * 70 * 2 = 4900 B
  uint8_t *w_blk = (uint8_t *)acc;                         // 9 * 18 * 32 = 5184 B
  int8_t *kt_buf = (int8_t *)sc_buf;                       // 9 * 8 * 64 = 4608 B
  int16_t *s23_qb = (int16_t *)(sc_buf + 4608);            // 4 * 70 * 2 = 560 B
  const int nqb = (W + 3) >> 2;                            // query 块 (每块 4 个)
  const int nq_last = W - 4 * (nqb - 1);
  const int K2 = 2 * W;                                    // 每个 r 的 key 数 (2 行)
  const int nkb = (K2 + 7) >> 3;                           // key 块 (每块 8 个)
  const int nk_last = K2 - 8 * (nkb - 1);
  const int T = 4 * W;
  const int nkb2 = (T + 7) >> 3;                           // W@V 的 key 块

  if (h[60] == 0) {
    if (r == 0) {
      #pragma clang loop unroll(disable)
      for (int qb = 0; qb < nqb; qb++) {
        const int q0 = qb * 4;
        #pragma clang loop unroll(disable)
        for (int g = 0; g < 4; g++) {
          V32 lo0, hi0;
          alignas(32) int8_t tmp[64];
          ::aie::vector<int8, 16> z16 = ::aie::zeros<int8, 16>();
          #pragma clang loop unroll(disable)
          for (int dq = 0; dq < 4; dq++) {
            const int q = q0 + dq;
            ::aie::store_v(tmp + dq * 16, (q < W) ? ::aie::load_v<16>(in + g * 960 + 64 + q * 16) : z16);
          }
          split(::aie::load_v<64>(tmp), lo0, hi0);
          ::aie::store_v(q_buf + qb * 256 + (2 * g) * 32, lo0);
          ::aie::store_v(q_buf + qb * 256 + (2 * g + 1) * 32, hi0);
        }
      }
      return;
    }

    int8_t *k_lin = (int8_t *)(sc_buf + 4608);
    #pragma clang loop unroll(disable)
    for (int g = 0; g < 4; g++) {
      gather_rows_16(in, 2, 4, g, nkb * 8, k_lin, W);
      #pragma clang loop unroll(disable)
      for (int kb = 0; kb < nkb; kb++) {
        V32 lo0, hi0, lo1, hi1;
        split(::aie::load_v<64>(k_lin + kb * 128), lo0, hi0);
        split(::aie::load_v<64>(k_lin + kb * 128 + 64), lo1, hi1);
        ::aie::store_v(kt_buf + (kb * 8 + 2 * g) * 64, ::aie::transpose(::aie::concat(lo0, lo1), 8, 8));
        ::aie::store_v(kt_buf + (kb * 8 + 2 * g + 1) * 64, ::aie::transpose(::aie::concat(hi0, hi1), 8, 8));
      }
    }

    const int32_t alpha = h[61];
    const int32_t valid_t = h[62];
    const uint16_t *exp_lut = (const uint16_t *)w;
    int16_t *s_row = (int16_t *)(sc_buf + 5184);
    uint16_t *e_row = (uint16_t *)(sc_buf + 5504);

    #pragma clang loop unroll(disable)
    for (int qb = 0; qb < nqb; qb++) {
      const int8_t *qp = q_buf + qb * 256;
      V32 qa0 = ::aie::load_v<32>(qp + 0 * 32), qa1 = ::aie::load_v<32>(qp + 1 * 32);
      V32 qa2 = ::aie::load_v<32>(qp + 2 * 32), qa3 = ::aie::load_v<32>(qp + 3 * 32);
      V32 qa4 = ::aie::load_v<32>(qp + 4 * 32), qa5 = ::aie::load_v<32>(qp + 5 * 32);
      V32 qa6 = ::aie::load_v<32>(qp + 6 * 32), qa7 = ::aie::load_v<32>(qp + 7 * 32);
      const int nq = (qb < nqb - 1) ? 4 : nq_last;
      int16_t *s_dst_qb = (r == 1) ? (s01_all + qb * 4 * 70) : s23_qb;
      #pragma clang loop unroll(disable)
      for (int kb = 0; kb < nkb; kb++) {
        const int8_t *kp = kt_buf + kb * 512;
        MM c;
        c.mul(qa0, ::aie::load_v<64>(kp + 0 * 64));
        c.mac(qa1, ::aie::load_v<64>(kp + 1 * 64));
        c.mac(qa2, ::aie::load_v<64>(kp + 2 * 64));
        c.mac(qa3, ::aie::load_v<64>(kp + 3 * 64));
        c.mac(qa4, ::aie::load_v<64>(kp + 4 * 64));
        c.mac(qa5, ::aie::load_v<64>(kp + 5 * 64));
        c.mac(qa6, ::aie::load_v<64>(kp + 6 * 64));
        c.mac(qa7, ::aie::load_v<64>(kp + 7 * 64));
        alignas(32) int32_t tmp_s[32];
        ::aie::store_v(tmp_s, c.template to_vector<int32>(0));
        const int k0 = kb * 8;
        const int nk = (kb < nkb - 1) ? 8 : nk_last;
        #pragma clang loop unroll(disable)
        for (int dq = 0; dq < nq; dq++) {
          int16_t *row_dst = s_dst_qb + dq * 70 + k0;
          #pragma clang loop unroll(disable)
          for (int dk = 0; dk < nk; dk++) {
            int32_t sc = (tmp_s[dq * 8 + dk] * alpha + 2048) >> 12;
            row_dst[dk] = (int16_t)(sc < -32768 ? -32768 : (sc > 32767 ? 32767 : sc));
          }
        }
      }
      if (r == 2) {
        int16_t *s01_qb = (int16_t *)(sc_buf + 5824);
        const int16_t *s01_src = s01_all + qb * 4 * 70;
        #pragma clang loop unroll(disable)
        for (int k = 0; k < nq * 70; k++) s01_qb[k] = s01_src[k];
        uint8_t *w_qblk = w_blk + qb * (18 * 32);
        #pragma clang loop unroll(disable)
        for (int dq = 0; dq < nq; dq++) {
          const int16_t *s01_q = s01_qb + dq * 70;
          const int16_t *s23_q = s23_qb + dq * 70;
          #pragma clang loop unroll(disable)
          for (int k = 0; k < K2; k++) {
            s_row[k] = s01_q[k];
            s_row[K2 + k] = s23_q[k];
          }
          #pragma clang loop unroll(disable)
          for (int k = valid_t; k < T; k++) s_row[k] = -32768;
          int32_t m = s_row[0];
          #pragma clang loop unroll(disable)
          for (int k = 1; k < T; k++) {
            if (s_row[k] > m) m = s_row[k];
          }
          uint32_t den = 0;
          #pragma clang loop unroll(disable)
          for (int k = 0; k < T; k++) {
            int32_t idx = (m - (int32_t)s_row[k] + 8) >> 4;
            if (idx > 255) idx = 255;
            uint16_t ek = exp_lut[idx];
            e_row[k] = ek;
            den += ek;
          }
          uint32_t inv_den = ((255U << 23) + (den >> 1)) / den;
          uint8_t *w_qbase = w_qblk + dq * 8;
          #pragma clang loop unroll(disable)
          for (int k = 0; k < T; k++) {
            uint32_t wv = ((uint32_t)e_row[k] * inv_den + (1U << 22)) >> 23;
            w_qbase[(k >> 3) * 32 + (k & 7)] = (uint8_t)(wv > 255U ? 255U : wv);
          }
        }
        // 最后一块里不存在的 query 行 (dq ≥ nq) 全写 0
        #pragma clang loop unroll(disable)
        for (int dq = nq; dq < 4; dq++) {
          #pragma clang loop unroll(disable)
          for (int kb = 0; kb < nkb2; kb++) {
            uint8_t *p = w_qblk + kb * 32 + dq * 8;
            for (int dk = 0; dk < 8; dk++) p[dk] = 0;
          }
        }
        // 最后一个 key 块里 T 之后的 key 写 0
        const int k_tail = T - 8 * (nkb2 - 1);
        #pragma clang loop unroll(disable)
        for (int dq = 0; dq < 4; dq++) {
          uint8_t *p = w_qblk + (nkb2 - 1) * 32 + dq * 8;
          for (int dk = k_tail; dk < 8; dk++) p[dk] = 0;
        }
      }
    }
    return;
  }

  // sub=1: W_u8[W][T] @ V_i8[T][16] -> out[40][16]
  int8_t *v_lo = (int8_t *)sc_buf;
  int8_t *v_hi = (int8_t *)(sc_buf + 1280);
  int8_t *v_lin = (int8_t *)(sc_buf + 2560); // 144 * 16 = 2304 B (< 6656)
  gather_rows_16(in, 4, 2, j, nkb2 * 8, v_lin, W);
  #pragma clang loop unroll(disable)
  for (int kb = 0; kb < nkb2; kb++) {
    V32 lo0, hi0, lo1, hi1;
    split(::aie::load_v<64>(v_lin + kb * 128), lo0, hi0);
    split(::aie::load_v<64>(v_lin + kb * 128 + 64), lo1, hi1);
    ::aie::store_v(v_lo + kb * 64, ::aie::concat(lo0, lo1));
    ::aie::store_v(v_hi + kb * 64, ::aie::concat(hi0, hi1));
  }

  const int16_t *vm_src = ((const int16_t *)w) + (h[63] ? core * 64 : 0) + (2 * r + j) * 16;
  ::aie::vector<int16, 16> vm16 = ::aie::load_v<16>(vm_src);
  ::aie::vector<int16, 8> vm_lo8 = vm16.extract<8>(0), vm_hi8 = vm16.extract<8>(1);
  ::aie::vector<int16, 32> vm_lo = ::aie::concat(::aie::concat(vm_lo8, vm_lo8), ::aie::concat(vm_lo8, vm_lo8));
  ::aie::vector<int16, 32> vm_hi = ::aie::concat(::aie::concat(vm_hi8, vm_hi8), ::aie::concat(vm_hi8, vm_hi8));

  #pragma clang loop unroll(disable)
  for (int qb = 0; qb < nqb; qb++) {
    const uint8_t *wp = w_blk + qb * (18 * 32);
    MM_U8 c0, c1;
    V32U a0 = ::aie::load_v<32>(wp);
    c0.mul(a0, ::aie::load_v<64>(v_lo));
    c1.mul(a0, ::aie::load_v<64>(v_hi));
    #pragma clang loop unroll(disable)
    for (int kb = 1; kb < nkb2; kb++) {
      V32U ak = ::aie::load_v<32>(wp + kb * 32);
      c0.mac(ak, ::aie::load_v<64>(v_lo + kb * 64));
      c1.mac(ak, ::aie::load_v<64>(v_hi + kb * 64));
    }
    V32 o0 = ::aie::mul(c0.template to_vector<int16>(1), vm_lo).template to_vector<int8>(15);
    V32 o1 = ::aie::mul(c1.template to_vector<int16>(1), vm_hi).template to_vector<int8>(15);
    auto [p, q] = ::aie::interleave_zip(o0, o1, 8);
    int8_t *o_ptr = (int8_t *)out + qb * 4 * 16;
    ::aie::store_v(o_ptr, p);
    ::aie::store_v(o_ptr + 32, q);
  }
  ::aie::vector<int8, 16> z16 = ::aie::zeros<int8, 16>();
  #pragma clang loop unroll(disable)
  for (int q = W; q < 40; q++) {
    ::aie::store_v((int8_t *)out + q * 16, z16);
  }
}
#endif

} // namespace

extern "C" {

// 描述符 int32[0..7]: ICP OCP WT stride k shift NXC PLANE;  int32[24] first  int32[25] last
//        int32[26] dilation  int32[27] depthwise  int32[28] depthwise 输入元素的行步长 (= m·PLANE)
// 每个 core c 一段 (字节 32 + 16c 起, u16): [0] rows [1] sel_s1 [2] sel_s2 [3] wvalid [4] 每行块数
//                                          [5] sel_acc [6] sel_fin
//        int32[29] 输出激活 (0 ReLU uint8, 1 HardSigmoid, 2 线性 int8, 3 hardswish, 4 ReLU int8, 5 GELU 查表)
//        [49] [50] 重新量化的 s1 s2  [52] 重新量化表在权重区里的偏移  [53] [54] hardswish 第二次重新量化的 t1 t2  [30] 输入逐通道乘 sc  [31] 这一步的输入通道组起点
//        [33] 池化 (1 累加步, 2 收尾步)  [34..39] 池化参数  [43] 池化每组元素数 ne  [47] 池化输入是 int8
//        [57] 双线性 ×2 上采样 (每 core 段的 [7] 是边界标志)  [58] 上采样的 log2(NXC)
//        [40] 缩放张量每元素的组数  [41] 缩放张量的组步长  [42] 本步是"装 SE 缩放向量"  [48] SE 只乘前几个输入通道组
// r = 本 core 这一步里的行号 (只有 IC 分块用来定位 acc)
void convC16Chunk(int8_t *r0, int8_t *r1, int8_t *r2, int8_t *wts, uint8_t *out, int32_t *acc, uint8_t *sc,
                  int32_t j, int32_t r, int32_t core) {
  ::aie::set_saturation(::aie::saturation_mode::saturate);
  ::aie::set_rounding(::aie::rounding_mode::positive_inf);
  const int32_t *h = (const int32_t *)wts;
  // 按链编译: chain_c16.Plan.kernel_flags() 给链里用不到的分支加 -DNO_xxx (程序内存只有 16 KB, 装不下所有分支)
#ifndef NO_SE
  if (h[42]) { // 装 SE 缩放向量 (累加段, r = 第几个元素)
    se_load((const uint8_t *)r0, sc, r, h[40], h[41]);
    return;
  }
  if (h[33]) { // 池化: 累加步 r = 组 · (ne-1) + 元素号; 收尾步 r = 组 (元素号 ne-1)
    const int ne1 = h[43] - 1;
    if (h[33] == 1) {
      const int g = r / ne1;
      se_pool((const uint8_t *)r0, acc + g * 16, out, r - g * ne1, 0, h);
    } else {
      se_pool((const uint8_t *)r0, acc + r * 16, out, ne1, 1, h);
    }
    return;
  }
#endif
#ifndef NO_UP
  if (h[57]) { // 双线性 ×2 上采样 (整段逻辑在 noinline 函数里, 入口函数保持小 → conv1 仍被内联)
    upsample_step(r0, r1, r2, wts, out, j, r, core);
    return;
  }
#endif
  const Desc d{h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[29], h[49], h[50],
               h[29] == 0 || h[29] == 4 ? 0 : -128};
  const uint16_t *pc = (const uint16_t *)(wts + 32 + 16 * core);
  const int wvalid = (pc[7] && r >= (int)pc[7]) ? 0 : pc[3];
  const int ocp = j / d.nxc, xc = j - ocp * d.nxc;
  const int px0 = xc * CW;
#ifndef NO_LM
  if (h[59] == 1) { ln_step(r1, h, wts + HDR, out, acc, j, wvalid - px0); return; }
#endif
#ifndef NO_ATTN
  if (h[59] == 2) { attn_step(r1, h, wts + HDR, out, acc, sc, r, j, core, wvalid); return; }
#endif
  int8_t *rows[3] = {r0, r1, r2};
  const int8_t *w = wts + HDR;
  const int8_t *rq = w + h[52] + ocp * RQB; // 这个输出通道组的重新量化表
#ifndef NO_DW
  if (h[27]) { // depthwise: j 里的通道组号 = 这一步里的第几个通道组
    const int8_t *e = r0 + ocp * d.plane;
    const int8_t *wd = w + ocp * (d.k * d.k * 64);
#ifndef NO_DW3
    if (d.k == 3) dwconv<3, 1>(e, wd, out, h[28], d.stride, px0, wvalid, rq, d); else // 3×3 s1/s2
#endif
#ifndef NO_DW5
    if (h[26] == 1) dwconv<5, 1>(e, wd, out, h[28], d.stride, px0, wvalid, rq, d); else // 5×5 s1/s2
#endif
#ifndef NO_DW5D2
    dwconv<5, 2>(e, wd, out, h[28], d.stride, px0, wvalid, rq, d); // 5×5 dilation 2
#else
    {}
#endif
  } else
#endif
  if (d.k == 1) {
    const int first = h[24], last = h[25];
    const int nch = pc[4];
#ifndef NO_SE
    if (h[30]) { // SE: 这一步第一次调用时把缩放折进权重 (组 h[31].. 是这一步的输入通道组), 然后清掉标志
      const int n = h[48] - h[31] < d.icp ? h[48] - h[31] : d.icp; // 拼进来的残差组不乘
      if (n > 0) se_wscale(wts + HDR, sc + h[31] * 16, n, d.icp, d.ocp);
      ((int32_t *)wts)[30] = 0;
    }
#endif
    conv1(r1, w + ocp * (d.icp * 256), out, d, px0, wvalid, acc + (r * nch + j) * (CW * 16), first,
          last, rq);
  }
#ifndef NO_S2
  else if (d.stride == 2) {
    conv3s2(rows, w + ocp * (9 * d.icp * 256), out, d, px0, wvalid, rq);
  }
#endif
#ifndef NO_K3
  else {
    conv3s1(rows, w + ocp * (9 * d.icp * 256), out, d, px0, wvalid, rq);
  }
#endif
#if !defined(NO_PA) || !defined(NO_GELU)
  if ((h[29] == 1 || h[29] == 3
#ifndef NO_GELU
       || h[29] == 5
#endif
       ) && h[25]) // IC 分块的中间步 (last = 0) 没有输出
    post_act(out, h[29], wvalid - px0, rq, h[53], h[54], w + h[52] + d.ocp * RQB);
#endif
}

} // extern "C"
