// 通用 int8 卷积 + ReLU, "16 通道像素"布局, 每次调用只算一个固定大小的输出块 (AIE2 / Phoenix)
//
// 用于 npu/chain_c16.py。和 conv_generic.cc 的区别:
//   · 激活布局 [H+2][CP][PAD + Ws + PAD][16]: 每个像素 16 个通道连续存 (CP = C/16)。
//     这样一个输出块 = 1 个通道组 (16 ch) × CW 个像素 = CW·16 字节, 是 DDR 布局里连续的一段,
//     任何层 (OC、宽度不同) 的一行输出都能拆成整数个同样大小的块 → 输出 ObjectFifo 元素大小对所有层相同。
//   · 每次调用算一个块: 块号 j → (ocp = j / NXC, xc = j % NXC), 像素 [xc·CW, xc·CW + CW)。
//   · 宽度可以补齐: 像素 ≥ wvalid (本 core 条带的真实宽度, 描述符里按 core 给) 的输出写 0,
//     所以存储宽度 Ws 可以大于真实宽度, 补出来的列对下一层就是零填充。
//
// mmul<4,8,8>: A = 4 像素 × 8 输入通道。16 通道像素的 64 字节向量 (4 像素) 用 interleave_unzip(·, 8)
// 拆成低 8 通道 / 高 8 通道两个 A, 各喂 2 个输出通道块。
//
// 输入元素: [ICP_pad][PLANE] 字节 (PLANE = 描述符给的块步长, 运行时); 每块前 (s·WT + 2·PAD)·16 字节是真实像素。
// 权重:     [HDR][权重], 每个输出通道组 ocp:
//             3×3: [ky][icp][kx][h][ob][ic8][oc8]   (h = 输入通道的低/高 8 个, ob = 输出的低/高 8 个)
//             1×1: [icp][h][ob][ic8][oc8]
// 输出块:   [CW][16] uint8 = sat_u8((Σ x·w + 2^(s-1)) >> s)
#define NOCPP
#include <stdint.h>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef CW
#define CW 40 // 每个输出块的像素数 (8 的倍数)
#endif
#define HDR 128
#define PAD 4

using MM = ::aie::mmul<4, 8, 8, int8, int8>;
using V32 = ::aie::vector<int8, 32>;
using V64 = ::aie::vector<int8, 64>;
using V128 = ::aie::vector<int8, 128>;

namespace {

struct Desc {
  int icp, ocp, wt, stride, k, shift, nxc, plane;
};

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

// 输出 4 像素 × 16 通道 (64 B); 超出真实宽度写 0
inline void store4(uint8_t *o, MM &c0, MM &c1, int shift, bool valid) {
  if (valid) {
    auto [p, q] = ::aie::interleave_zip(c0.template to_vector<uint8>(shift),
                                        c1.template to_vector<uint8>(shift), 8);
    ::aie::store_v(o, p);
    ::aie::store_v(o + 32, q);
  } else {
    ::aie::store_v(o, ::aie::zeros<uint8, 32>());
    ::aie::store_v(o + 32, ::aie::zeros<uint8, 32>());
  }
}

void conv3s1(int8_t **rows, const int8_t *w0, uint8_t *out, const Desc &d, int px0, int wvalid) {
  for (int x = 0; x < CW; x += 8) {
    MM c00(::aie::zeros<acc32, 32>()), c01(::aie::zeros<acc32, 32>());
    MM c10(::aie::zeros<acc32, 32>()), c11(::aie::zeros<acc32, 32>());
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
    store4(out + x * 16, c00, c01, d.shift, px0 + x < wvalid);
    store4(out + (x + 4) * 16, c10, c11, d.shift, px0 + x + 4 < wvalid);
  }
}

// stride 2: 输出像素 x ← 输入 2x-1, 2x, 2x+1; 16 字节 = 1 像素, filter_even/odd(·, 16) 拆奇偶像素
void conv3s2(int8_t **rows, const int8_t *w0, uint8_t *out, const Desc &d, int px0, int wvalid) {
  for (int x = 0; x < CW; x += 8) {
    MM c00(::aie::zeros<acc32, 32>()), c01(::aie::zeros<acc32, 32>());
    MM c10(::aie::zeros<acc32, 32>()), c11(::aie::zeros<acc32, 32>());
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
    store4(out + x * 16, c00, c01, d.shift, px0 + x < wvalid);
    store4(out + (x + 4) * 16, c10, c11, d.shift, px0 + x + 4 < wvalid);
  }
}

void conv1(const int8_t *r1, const int8_t *w0, uint8_t *out, const Desc &d, int px0, int wvalid) {
  for (int x = 0; x < CW; x += 8) {
    MM c00(::aie::zeros<acc32, 32>()), c01(::aie::zeros<acc32, 32>());
    MM c10(::aie::zeros<acc32, 32>()), c11(::aie::zeros<acc32, 32>());
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
    store4(out + x * 16, c00, c01, d.shift, px0 + x < wvalid);
    store4(out + (x + 4) * 16, c10, c11, d.shift, px0 + x + 4 < wvalid);
  }
}

} // namespace

extern "C" {

// 描述符 int32[0..7]: ICP OCP WT stride k shift NXC PLANE
// 每个 core c 一段 (字节 32 + 16c 起, u16): [0] rows [1] sel_s1 [2] sel_s2 [3] wvalid [4] 每行块数
void convC16Chunk(int8_t *r0, int8_t *r1, int8_t *r2, int8_t *wts, uint8_t *out, int32_t j,
                  int32_t core) {
  ::aie::set_saturation(::aie::saturation_mode::saturate);
  ::aie::set_rounding(::aie::rounding_mode::positive_inf);
  const int32_t *h = (const int32_t *)wts;
  const Desc d{h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]};
  const uint16_t *pc = (const uint16_t *)(wts + 32 + 16 * core);
  const int wvalid = pc[3];
  const int ocp = j / d.nxc, xc = j - ocp * d.nxc;
  const int px0 = xc * CW;
  int8_t *rows[3] = {r0, r1, r2};
  const int8_t *w = wts + HDR;
  if (d.k == 1) {
    conv1(r1, w + ocp * (d.icp * 256), out, d, px0, wvalid);
  } else if (d.stride == 2) {
    conv3s2(rows, w + ocp * (9 * d.icp * 256), out, d, px0, wvalid);
  } else {
    conv3s1(rows, w + ocp * (9 * d.icp * 256), out, d, px0, wvalid);
  }
}

} // extern "C"
