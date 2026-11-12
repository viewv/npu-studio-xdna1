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
  int act;    // 0 / 4 ReLU (int8, 0..127)  1 HardSigmoid  2 线性  3 hardswish
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
__attribute__((noinline)) void post_act(uint8_t *o, int act, int nv, const int8_t *rq, int t1, int t2) {
  if (act == 1) {
    const auto m = ::aie::broadcast<uint8, 64>(0x80);
#pragma clang loop unroll(disable)
    for (int i = 0; i < CW * 16; i += 64) ::aie::store_v(o + i, ::aie::bit_xor(::aie::load_v<64>(o + i), m));
#pragma clang loop unroll(disable)
    for (int p = nv < 0 ? 0 : nv; p < CW; p++) ::aie::store_v(o + p * 16, ::aie::zeros<uint8, 16>());
    return;
  }
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

} // namespace

extern "C" {

// 描述符 int32[0..7]: ICP OCP WT stride k shift NXC PLANE;  int32[24] first  int32[25] last
//        int32[26] dilation  int32[27] depthwise  int32[28] depthwise 输入元素的行步长 (= m·PLANE)
// 每个 core c 一段 (字节 32 + 16c 起, u16): [0] rows [1] sel_s1 [2] sel_s2 [3] wvalid [4] 每行块数
//                                          [5] sel_acc [6] sel_fin
//        int32[29] 输出激活 (0 ReLU uint8, 1 HardSigmoid, 2 线性 int8, 3 hardswish, 4 ReLU int8)
//        [49] [50] 重新量化的 s1 s2  [52] 重新量化表在权重区里的偏移  [53] [54] hardswish 第二次重新量化的 t1 t2  [30] 输入逐通道乘 sc  [31] 这一步的输入通道组起点
//        [33] 池化 (1 累加步, 2 收尾步)  [34..39] 池化参数  [43] 池化每组元素数 ne  [47] 池化输入是 int8
//        [40] 缩放张量每元素的组数  [41] 缩放张量的组步长  [42] 本步是"装 SE 缩放向量"  [48] SE 只乘前几个输入通道组
// r = 本 core 这一步里的行号 (只有 IC 分块用来定位 acc)
void convC16Chunk(int8_t *r0, int8_t *r1, int8_t *r2, int8_t *wts, uint8_t *out, int32_t *acc, uint8_t *sc,
                  int32_t j, int32_t r, int32_t core) {
  ::aie::set_saturation(::aie::saturation_mode::saturate);
  ::aie::set_rounding(::aie::rounding_mode::positive_inf);
  const int32_t *h = (const int32_t *)wts;
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
  const Desc d{h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[29], h[49], h[50],
               h[29] == 0 || h[29] == 4 ? 0 : -128};
  const uint16_t *pc = (const uint16_t *)(wts + 32 + 16 * core);
  const int wvalid = pc[3];
  const int ocp = j / d.nxc, xc = j - ocp * d.nxc;
  const int px0 = xc * CW;
  int8_t *rows[3] = {r0, r1, r2};
  const int8_t *w = wts + HDR;
  const int8_t *rq = w + h[52] + ocp * RQB; // 这个输出通道组的重新量化表
  if (h[27]) { // depthwise: j 里的通道组号 = 这一步里的第几个通道组
    const int8_t *e = r0 + ocp * d.plane;
    const int8_t *wd = w + ocp * (d.k * d.k * 64);
    if (d.k == 3) dwconv<3, 1>(e, wd, out, h[28], d.stride, px0, wvalid, rq, d);   // RVM 用到的: 3×3 s1/s2,
    else if (h[26] == 1) dwconv<5, 1>(e, wd, out, h[28], d.stride, px0, wvalid, rq, d); // 5×5 s1/s2,
    else dwconv<5, 2>(e, wd, out, h[28], d.stride, px0, wvalid, rq, d);                // 5×5 dilation 2
  } else if (d.k == 1) {
    const int first = h[24], last = h[25];
    const int nch = pc[4];
    if (h[30]) { // SE: 这一步第一次调用时把缩放折进权重 (组 h[31].. 是这一步的输入通道组), 然后清掉标志
      const int n = h[48] - h[31] < d.icp ? h[48] - h[31] : d.icp; // 拼进来的残差组不乘
      if (n > 0) se_wscale(wts + HDR, sc + h[31] * 16, n, d.icp, d.ocp);
      ((int32_t *)wts)[30] = 0;
    }
    conv1(r1, w + ocp * (d.icp * 256), out, d, px0, wvalid, acc + (r * nch + j) * (CW * 16), first,
          last, rq);
  } else if (d.stride == 2) {
    conv3s2(rows, w + ocp * (9 * d.icp * 256), out, d, px0, wvalid, rq);
  } else {
    conv3s1(rows, w + ocp * (9 * d.icp * 256), out, d, px0, wvalid, rq);
  }
  if ((h[29] == 1 || h[29] == 3) && h[25]) // IC 分块的中间步 (last = 0) 没有输出
    post_act(out, h[29], wvalid - px0, rq, h[53], h[54]);
}

} // extern "C"
