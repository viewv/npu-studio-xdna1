// 通用 int8 卷积 + ReLU, 一次算一行输出; 层形状由权重块头部的描述符在运行时给出 (AIE2 / Phoenix)
//
// 用于"异构层串联"实验 (npu/chain_hetero.py): 整条链只编一个内核, ObjectFifo 元素大小固定,
// 每层的 IC / OC / 输出宽度 / stride / 核大小 / shift 都写在权重块的头 64 字节里。
//
// 支持三种层: 3×3 stride 1 (pad 1)、3×3 stride 2 (pad 1)、1×1 stride 1。
//
// 布局 (8 通道一块, 块内像素优先, 与 mmul<4,8,8> 的 A/C 矩阵行优先一致):
//   输入行 r0/r1/r2 : 固定 [ICB_E][PLANE_E] 字节; 第 icb 块从 icb·PLANE_E 开始,
//                     前 (PAD + WIN + PAD)·8 字节是真实像素 (WIN = s·WT), 后面是 DMA 顺带读进来的垃圾,
//                     只用前 icb < 描述符 ICB 个块。1×1 层只用中间一行 r1。
//   权重 wts        : [HDR 字节描述符][权重], 权重按每两个输出通道块 (16 个输出通道) 一组:
//                     3×3: [ocp][ky][icb][kx][2][ic8][oc8]   1×1: [ocp][icb][2][ic8][oc8]
//   输出行 out      : [OCB][WT][8] uint8 = sat_u8((Σ x·w + 2^(s-1)) >> s)  (饱和即 ReLU)
//                     OCB·WT·8 必须等于输出元素大小 (主机侧保证)。
#define NOCPP
#include <stdint.h>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef PLANE_E
#define PLANE_E 1344 // 输入元素里每个通道块的字节数 (编译期, 所有层共用)
#endif
#define HDR 64 // 权重块头部描述符字节数 (保持 64 字节对齐)
#define PAD 4  // 输入行左右各 4 个像素

using MM = ::aie::mmul<4, 8, 8, int8, int8>;
using V32 = ::aie::vector<int8, 32>;
using V64 = ::aie::vector<int8, 64>;

namespace {

inline void store_px8(uint8_t *out, int wt, int ocp, int x0, int shift, MM &c00, MM &c01, MM &c10,
                      MM &c11) {
  uint8_t *o0 = out + (2 * ocp) * wt * 8;
  uint8_t *o1 = o0 + wt * 8;
  ::aie::store_v(o0 + x0 * 8, c00.template to_vector<uint8>(shift));
  ::aie::store_v(o0 + (x0 + 4) * 8, c10.template to_vector<uint8>(shift));
  ::aie::store_v(o1 + x0 * 8, c01.template to_vector<uint8>(shift));
  ::aie::store_v(o1 + (x0 + 4) * 8, c11.template to_vector<uint8>(shift));
}

// 3×3 stride 1: 与 conv3x3.cc 相同的算法, ICB / OCB / WT 为运行时值
void conv3s1(int8_t **rows, const int8_t *wts, uint8_t *out, int icb_n, int ocp_n, int wt,
             int shift) {
  for (int ocp = 0; ocp < ocp_n; ocp++) {
    const int8_t *wbase = wts + ocp * (3 * icb_n * 384);
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int x0 = 0; x0 < wt; x0 += 8) {
        MM c00(::aie::zeros<acc32, 32>()), c01(::aie::zeros<acc32, 32>());
        MM c10(::aie::zeros<acc32, 32>()), c11(::aie::zeros<acc32, 32>());
        const int8_t *w = wbase;
        for (int ky = 0; ky < 3; ky++) {
          const int8_t *p = rows[ky] + (PAD + x0) * 8;
          AIE_LOOP_MIN_ITERATION_COUNT(2)
          for (int icb = 0; icb < icb_n; icb++) {
              const V32 L = ::aie::load_v<32>(p - 32);
              const V32 C0 = ::aie::load_v<32>(p);
              const V32 C1 = ::aie::load_v<32>(p + 32);
              const V32 R = ::aie::load_v<32>(p + 64);
              p += PLANE_E;
              V32 a0 = ::aie::shuffle_up_fill(C0, L, 8);
              V32 a1 = ::aie::shuffle_up_fill(C1, C0, 8);
              V64 b0 = ::aie::load_v<64>(w), b1 = ::aie::load_v<64>(w + 64);
              c00.mac(a0, b0); c01.mac(a0, b1); c10.mac(a1, b0); c11.mac(a1, b1);
              b0 = ::aie::load_v<64>(w + 128); b1 = ::aie::load_v<64>(w + 192);
              c00.mac(C0, b0); c01.mac(C0, b1); c10.mac(C1, b0); c11.mac(C1, b1);
              a0 = ::aie::shuffle_down_fill(C0, C1, 8);
              a1 = ::aie::shuffle_down_fill(C1, R, 8);
              b0 = ::aie::load_v<64>(w + 256); b1 = ::aie::load_v<64>(w + 320);
              c00.mac(a0, b0); c01.mac(a0, b1); c10.mac(a1, b0); c11.mac(a1, b1);
              w += 384;
            }
        }
        store_px8(out, wt, ocp, x0, shift, c00, c01, c10, c11);
      }
  }
}

// 3×3 stride 2: 输出像素 x 用输入像素 2x-1, 2x, 2x+1。
// 每步 8 个输出像素 = 输入 2x0-1 .. 2x0+15; 用 filter_even/odd (8 字节一组 = 1 个像素) 拆奇偶。
void conv3s2(int8_t **rows, const int8_t *wts, uint8_t *out, int icb_n, int ocp_n, int wt,
             int shift) {
  for (int ocp = 0; ocp < ocp_n; ocp++) {
    const int8_t *wbase = wts + ocp * (3 * icb_n * 384);
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int x0 = 0; x0 < wt; x0 += 8) {
        MM c00(::aie::zeros<acc32, 32>()), c01(::aie::zeros<acc32, 32>());
        MM c10(::aie::zeros<acc32, 32>()), c11(::aie::zeros<acc32, 32>());
        const int8_t *w = wbase;
        for (int ky = 0; ky < 3; ky++) {
          const int8_t *p = rows[ky] + (PAD + 2 * x0) * 8;
          AIE_LOOP_MIN_ITERATION_COUNT(2)
          for (int icb = 0; icb < icb_n; icb++) {
              const V32 L = ::aie::load_v<32>(p - 32);  // 2x0-4 .. 2x0-1
              const V32 V0 = ::aie::load_v<32>(p);      // 2x0   .. 2x0+3
              const V32 V1 = ::aie::load_v<32>(p + 32); // 2x0+4 .. 2x0+7
              const V32 V2 = ::aie::load_v<32>(p + 64);
              const V32 V3 = ::aie::load_v<32>(p + 96); // .. 2x0+15
              p += PLANE_E;
              const V64 g0 = ::aie::concat(V0, V1), g1 = ::aie::concat(V2, V3);
              const V32 E0 = ::aie::filter_even(g0, 8), O0 = ::aie::filter_odd(g0, 8);
              const V32 E1 = ::aie::filter_even(g1, 8), O1 = ::aie::filter_odd(g1, 8);
              // kx = 0: 2x-1 = 奇数像素右移一格, 最低一格来自前一组的最后一个像素
              V32 a0 = ::aie::shuffle_up_fill(O0, L, 8);
              V32 a1 = ::aie::shuffle_up_fill(O1, V1, 8);
              V64 b0 = ::aie::load_v<64>(w), b1 = ::aie::load_v<64>(w + 64);
              c00.mac(a0, b0); c01.mac(a0, b1); c10.mac(a1, b0); c11.mac(a1, b1);
              b0 = ::aie::load_v<64>(w + 128); b1 = ::aie::load_v<64>(w + 192);
              c00.mac(E0, b0); c01.mac(E0, b1); c10.mac(E1, b0); c11.mac(E1, b1);
              b0 = ::aie::load_v<64>(w + 256); b1 = ::aie::load_v<64>(w + 320);
              c00.mac(O0, b0); c01.mac(O0, b1); c10.mac(O1, b0); c11.mac(O1, b1);
              w += 384;
            }
        }
        store_px8(out, wt, ocp, x0, shift, c00, c01, c10, c11);
      }
  }
}

// 1×1 stride 1: 只用中间一行
void conv1(const int8_t *r1, const int8_t *wts, uint8_t *out, int icb_n, int ocp_n, int wt,
           int shift) {
  for (int ocp = 0; ocp < ocp_n; ocp++) {
    const int8_t *wbase = wts + ocp * (icb_n * 128);
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int x0 = 0; x0 < wt; x0 += 8) {
        MM c00(::aie::zeros<acc32, 32>()), c01(::aie::zeros<acc32, 32>());
        MM c10(::aie::zeros<acc32, 32>()), c11(::aie::zeros<acc32, 32>());
        const int8_t *w = wbase;
        const int8_t *p = r1 + (PAD + x0) * 8;
        AIE_LOOP_MIN_ITERATION_COUNT(2)
        for (int icb = 0; icb < icb_n; icb++) {
            const V32 C0 = ::aie::load_v<32>(p);
            const V32 C1 = ::aie::load_v<32>(p + 32);
            p += PLANE_E;
            V64 b0 = ::aie::load_v<64>(w), b1 = ::aie::load_v<64>(w + 64);
            c00.mac(C0, b0); c01.mac(C0, b1); c10.mac(C1, b0); c11.mac(C1, b1);
            w += 128;
          }
        store_px8(out, wt, ocp, x0, shift, c00, c01, c10, c11);
      }
  }
}

} // namespace

extern "C" {

// 描述符 (int32): [0] ICB  [1] OCB  [2] WT (输出像素)  [3] stride  [4] 核大小 (1/3)  [5] shift
void convGenericLine(int8_t *r0, int8_t *r1, int8_t *r2, int8_t *wts, uint8_t *out) {
  ::aie::set_saturation(::aie::saturation_mode::saturate);
  ::aie::set_rounding(::aie::rounding_mode::positive_inf);
  const int32_t *d = (const int32_t *)wts;
  const int icb_n = d[0], ocp_n = d[1] / 2, wt = d[2], stride = d[3], ks = d[4], shift = d[5];
  const int8_t *w = wts + HDR;
  int8_t *rows[3] = {r0, r1, r2};
  if (ks == 1)
    conv1(r1, w, out, icb_n, ocp_n, wt, shift);
  else if (stride == 2)
    conv3s2(rows, w, out, icb_n, ocp_n, wt, shift);
  else
    conv3s1(rows, w, out, icb_n, ocp_n, wt, shift);
}

} // extern "C"
