// 最小复现: up4 (运行时偏移非对齐加载 ×3 + mul/mac) → 再一次 mac → acc32 规约
#define NOCPP
#include <stdint.h>
#include "row_ops.h"
using namespace rowops;
#ifndef PLS
#define PLS 704
#endif
#ifndef PV
#define PV 0
#endif
extern "C" {
void probeLine(uint8_t *in, uint8_t *out8, int32_t width) {
  init();
  const int16_t *aq = (const int16_t *)in;
  const int16_t *bq = aq + PLS;
  volatile int32_t ls_v = PLS;
  const int32_t ls_rt = ls_v;   // 编译器看不到常量, 不会把偏移折进立即数
  const int16_t *ip = bq + PLS;
  int16_t *out = (int16_t *)out8;
  const Up4 w16 = Up4::make();
  const v32i16 one = ::aie::broadcast<int16_t, 32>(1);
#if PV != 9
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
#endif
  for (int x = 0; x < width; x += 32) {
    const int j0 = x >> 2;
    v32i16 I = ::aie::load_v<32>(ip + x);
#if PV == 0        // 只放大 A
    v32i16 r = up4_i16(aq, j0, w16);
#elif PV == 1      // A·I + B → int16 (>>6)
    ::aie::accum<acc32, 32> q = ::aie::mul(up4_i16(aq, j0, w16), I);
    q = ::aie::mac(q, up4_i16(bq, j0, w16), one);
    v32i16 r = q.template to_vector<int16_t>(6);
#elif PV == 2 || PV == 9 // 与 fused_composite 相同: → uint8 (>>6) 再宽化存出  (9 = 去掉流水线 pragma)
    ::aie::accum<acc32, 32> q = ::aie::mul(up4_i16(aq, j0, w16), I);
    q = ::aie::mac(q, up4_i16(bq, j0, w16), one);
    ::aie::store_v(out8 + x, q.template to_vector<uint8_t>(6));
#elif PV == 3      // 只有 B 的放大, 再 mac 一次 (A 不参与)
    ::aie::accum<acc32, 32> q = ::aie::mul(I, one);
    q = ::aie::mac(q, up4_i16(bq, j0, w16), one);
    ::aie::store_v(out8 + x, q.template to_vector<uint8_t>(6));
#elif PV == 5     // upA + I  (误差直接可见)
    ::aie::accum<acc32, 32> q = ::aie::mul(up4_i16(aq, j0, w16), one);
    q = ::aie::mac(q, I, one);
    v32i16 r = q.template to_vector<int16_t>(0);
#elif PV == 6     // upB + I
    ::aie::accum<acc32, 32> q = ::aie::mul(up4_i16(bq, j0, w16), one);
    q = ::aie::mac(q, I, one);
    v32i16 r = q.template to_vector<int16_t>(0);
#elif PV == 7     // upA * I  (单次 mul)
    ::aie::accum<acc32, 32> q = ::aie::mul(up4_i16(aq, j0, w16), I);
    v32i16 r = q.template to_vector<int16_t>(6);
#elif PV == 10    // upA + upB (两个放大, 不碰 I)
    ::aie::accum<acc32, 32> q = ::aie::mul(up4_i16(aq, j0, w16), one);
    q = ::aie::mac(q, up4_i16(bq, j0, w16), one);
    v32i16 r = q.template to_vector<int16_t>(0);
#elif PV == 11    // 只放大 B
    v32i16 r = up4_i16(bq, j0, w16);
#elif PV == 12    // 只放大 B, 指针洗白 (编译器看不到常量偏移)
    const int16_t *bl = aq + ls_rt;
    v32i16 r = up4_i16(bl, j0, w16);
#elif PV == 8     // ICE: widen_u8(to_vector<uint8>) → store int16
    ::aie::accum<acc32, 32> q = ::aie::mul(up4_i16(aq, j0, w16), I);
    q = ::aie::mac(q, up4_i16(bq, j0, w16), one);
    v32i16 r = widen_u8(q.template to_vector<uint8_t>(6));
#endif
#if PV == 0 || PV == 1 || PV == 5 || PV == 6 || PV == 7 || PV == 10 || PV == 11 || PV == 12 || PV == 8
    ::aie::store_v(out + x, r);
#endif
  }
}
}
