// 编译器回归探针: "基址 + 编译期常量偏移" 的非对齐向量加载 (见 row_ops.h 文件头)
//
// 输入一行 [A int16 × PLANE][B int16 × PLANE], 输出 up4(B) (int16 × W)。
// llvm-aie 21.0.0.2026062301 上, PLANE*2 >= 1024 B 且偏移是常量时, up4 的 v[j-1]
// 会读成 A 平面 → x%4∈{0,1} 的 lane 出错。LAUNDER=1 用 plane_stride() 绕过。
#define NOCPP
#include <stdint.h>
#include "row_ops.h"
using namespace rowops;

#ifndef PLANE
#define PLANE 512
#endif
#ifndef LAUNDER
#define LAUNDER 0
#endif

extern "C" {
void offsetProbeLine(uint8_t *in, uint8_t *out8, int32_t width) {
  init();
  const int16_t *a = (const int16_t *)in;
#if LAUNDER
  const int16_t *b = a + plane_stride(PLANE);
#else
  const int16_t *b = a + PLANE;
#endif
  int16_t *out = (int16_t *)out8;
  const Up4 w16 = Up4::make();
  for (int x = 0; x < width; x += 32)
    ::aie::store_v(out + x, up4_i16(b, x >> 2, w16));
}
}
