// LN 定点阶梯的可行性探针: 一个 token (512 通道) 的 LayerNorm，全 int32 定点。
//   x: 512 int8 ([组32][16通道])
//   参数 (逐通道, int32): m1(Q12) m2(Q19) g10(Q10) b12(Q12) gq(Q12) —— 5*512 int32
//   y: 512 int8
#include <stdint.h>
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/operators.hpp>

using namespace aie::operators;
using ACC = ::aie::accum<acc32, 16>;
using V16S = ::aie::vector<int16, 16>;
using V16I8 = ::aie::vector<int8, 16>;

extern "C" {

#ifndef D
#define D 512
#endif
void lnProbe(const int8_t *x, const int32_t *p, int8_t *y) {
  const int32_t *m1 = p, *m2 = p + D, *g10 = p + 2 * D, *b12 = p + 3 * D, *gq = p + 4 * D;
  // 1) Σx·m1（Q12）与 Σx²·m2（Q19）: 逐通道乘加, int32 累加
  int32_t s1 = 0, s2 = 0;
  for (int c = 0; c < D; c++) {
    int32_t xi = (int32_t)x[c] * m1[c];      // Q12
    s1 += xi;
    s2 += (int32_t)x[c] * x[c] * m2[c];      // Q19
  }
  const int32_t mean_q = s1 >> 9;   // D = 512, 用移位避开 AIE2 的整数除法
  const float var = (float)s2 / (D * 524288.0) - (float)(mean_q) * mean_q / (4096.0 * 4096.0);
  // 1/sqrt: AIE2 没有 sqrtf —— 位技巧初值 + 3 次 Newton
  float v = var + 1e-6f;
  float h = 0.5f * v;
  int32_t i = *(int32_t *)&v;
  i = 0x5f3759df - (i >> 1);
  float rs = *(float *)&i;
  rs = rs * (1.5f - h * rs * rs);
  rs = rs * (1.5f - h * rs * rs);
  rs = rs * (1.5f - h * rs * rs);
  const float rstd = rs;
  const int32_t r12 = (int32_t)(rstd * 4096.0 + 0.5);
  for (int c = 0; c < D; c++) {
    int32_t dev = (int32_t)x[c] * m1[c] - mean_q;        // Q12
    int32_t n12 = (dev * r12) >> 12;                     // Q12
    int32_t y12 = (n12 * g10[c]) >> 10;                  // Q12
    int32_t o = ((y12 + b12[c]) * gq[c]) >> 12;
    y[c] = (int8_t)(o < -128 ? -128 : (o > 127 ? 127 : o));
  }
}

}
