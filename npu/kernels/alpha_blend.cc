// 逐像素 alpha 混合 (AIE2 / Phoenix NPU)
//
//   out = (fg * m + bg * (255 - m)) >> 8
//
// fg / bg / m 都是同长度的 uint8 行 (BGR 交错, m 已按通道展开)。
// 每次处理 32 个字节: 两次 8bit×8bit 乘累加到 acc32, 右移 8 位饱和回 uint8。
// (>>8 相当于 /256, 比 /255 暗约 0.4%, 肉眼不可见, 省掉一次除法)

#define NOCPP

#include <stdint.h>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

template <int N>
static inline void alpha_blend_aie(const uint8_t *__restrict fg,
                                   const uint8_t *__restrict bg,
                                   const uint8_t *__restrict m,
                                   uint8_t *__restrict out, const int32_t n) {
  ::aie::set_saturation(aie::saturation_mode::saturate);
  const ::aie::vector<uint8_t, N> full = ::aie::broadcast<uint8_t, N>(255);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int i = 0; i < n; i += N) {
    ::aie::vector<uint8_t, N> vf = ::aie::load_v<N>(fg);
    fg += N;
    ::aie::vector<uint8_t, N> vb = ::aie::load_v<N>(bg);
    bg += N;
    ::aie::vector<uint8_t, N> vm = ::aie::load_v<N>(m);
    m += N;
    ::aie::vector<uint8_t, N> vinv = ::aie::sub(full, vm);
    ::aie::accum<acc32, N> acc = ::aie::mul(vf, vm);
    acc = ::aie::mac(acc, vb, vinv);
    ::aie::store_v(out, acc.template to_vector<uint8_t>(8));
    out += N;
  }
}

extern "C" {

// packed = [fg 行 | bg 行 | mask 行], 每段 lineBytes 字节
// (Phoenix 每个 Shim 只有 2 路输入 DMA, 三路数据打包成一路传输)
void alphaBlendPackedLine(uint8_t *packed, uint8_t *out, int32_t lineBytes) {
  alpha_blend_aie<32>(packed, packed + lineBytes, packed + 2 * lineBytes, out,
                      lineBytes);
}

} // extern "C"
