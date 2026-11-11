// int8 3×3 卷积 + ReLU (stride 1, pad 1), 一次算一行输出 (AIE2 / Phoenix)
//
// 布局 (都按 8 通道一块, 块内像素优先, 与 mmul<4,8,8> 的 A/C 矩阵行优先一致):
//   输入行  r0/r1/r2 : [ICB][PAD + WT + PAD][8] int8   (上/中/下三行; 左右各 PAD 像素,
//                      由主机放好邻居像素或零)
//   权重    wts      : [ky][ICB][kx][OCB][ic8][oc8] int8   (每块 = mmul 的 B 矩阵, 8×8)
//   输出行  out      : [OCB][WT][8] uint8 = sat_u8(round((Σ x·w) >> shift))  (饱和即 ReLU)
//
// 内层: 每步 8 个输出像素 × 16 个输出通道 = 4 个 4×8 累加器。对每个 (ky, icb):
//   读 4 个对齐的 32 字节向量 (像素 x0-4 .. x0+11), 用 shuffle_up/down_fill 移 1 个像素
//   得到 kx=0/2 的 A 矩阵; 每块权重喂 2 个像素组, 每个 A 喂 2 个输出通道块 → 12 次 mmul。
// 舍入: (acc + 2^(s-1)) >> s, 即 rounding_mode::positive_inf; 与 npu/rvm/conv_proto.py 的
// QConv.reference 逐位一致。
#define NOCPP
#include <stdint.h>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef WT
#define WT 160 // 每次调用的输出宽度 (像素), 8 的倍数
#endif
#ifndef ICB
#define ICB 5 // 输入通道块数 (IC = 8·ICB, 不足的通道补零 / 放 bias 常数)
#endif
#define OCB 2 // 输出通道块数 (OC = 16)
#define PAD 4 // 左右各 4 个像素 (保持 32 字节对齐; 实际只用到 1 个)
#define PLANE ((WT + 2 * PAD) * 8) // 一个输入通道块一行的字节数

using MM = ::aie::mmul<4, 8, 8, int8, int8>;

extern "C" {

void conv3x3I8Line(int8_t *r0, int8_t *r1, int8_t *r2, int8_t *wts, uint8_t *out,
                   int32_t shift) {
  ::aie::set_saturation(::aie::saturation_mode::saturate);
  ::aie::set_rounding(::aie::rounding_mode::positive_inf);
  int8_t *rows[3] = {r0, r1, r2};

  for (int x0 = 0; x0 < WT; x0 += 8) {
    MM c00(::aie::zeros<acc32, 32>()), c01(::aie::zeros<acc32, 32>());
    MM c10(::aie::zeros<acc32, 32>()), c11(::aie::zeros<acc32, 32>());
    const int8_t *w = wts;
    for (int ky = 0; ky < 3; ky++) {
      const int8_t *p = rows[ky] + (PAD + x0) * 8;
      AIE_LOOP_UNROLL_FULL
      for (int icb = 0; icb < ICB; icb++) {
        const ::aie::vector<int8, 32> L = ::aie::load_v<32>(p - 32);  // x0-4 .. x0-1
        const ::aie::vector<int8, 32> C0 = ::aie::load_v<32>(p);      // x0   .. x0+3
        const ::aie::vector<int8, 32> C1 = ::aie::load_v<32>(p + 32); // x0+4 .. x0+7
        const ::aie::vector<int8, 32> R = ::aie::load_v<32>(p + 64);  // x0+8 .. x0+11
        p += PLANE;
        // kx = 0: 像素 x-1
        ::aie::vector<int8, 32> a0 = ::aie::shuffle_up_fill(C0, L, 8);
        ::aie::vector<int8, 32> a1 = ::aie::shuffle_up_fill(C1, C0, 8);
        ::aie::vector<int8, 64> b0 = ::aie::load_v<64>(w), b1 = ::aie::load_v<64>(w + 64);
        c00.mac(a0, b0); c01.mac(a0, b1); c10.mac(a1, b0); c11.mac(a1, b1);
        // kx = 1: 像素 x
        b0 = ::aie::load_v<64>(w + 128); b1 = ::aie::load_v<64>(w + 192);
        c00.mac(C0, b0); c01.mac(C0, b1); c10.mac(C1, b0); c11.mac(C1, b1);
        // kx = 2: 像素 x+1
        a0 = ::aie::shuffle_down_fill(C0, C1, 8);
        a1 = ::aie::shuffle_down_fill(C1, R, 8);
        b0 = ::aie::load_v<64>(w + 256); b1 = ::aie::load_v<64>(w + 320);
        c00.mac(a0, b0); c01.mac(a0, b1); c10.mac(a1, b0); c11.mac(a1, b1);
        w += 384;
      }
    }
    ::aie::store_v(out + x0 * 8, c00.template to_vector<uint8>(shift));
    ::aie::store_v(out + (x0 + 4) * 8, c10.template to_vector<uint8>(shift));
    ::aie::store_v(out + WT * 8 + x0 * 8, c01.template to_vector<uint8>(shift));
    ::aie::store_v(out + WT * 8 + (x0 + 4) * 8, c11.template to_vector<uint8>(shift));
  }
}

} // extern "C"
