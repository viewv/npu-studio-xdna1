// 融合内核: 水平×4双线性放大 + 导向滤波遮罩 + 背景放大 + alpha 合成 (AIE2 / Phoenix)
//
// 每次调用处理一行全分辨率图像 (宽 W 像素, BGRA):
//
//   px  : 原图一行, BGRA uint8, W*4 字节
//   low : 低分辨率数据一行 (宽 LW = W/4, 已由 CPU 做好垂直插值), 打包为
//           [Aq  : int16 × LOW_STRIDE]   Aq = round(64·a),   pad 后 a[-1], a[0..LW-1], a[LW]
//           [Bq  : int16 × LOW_STRIDE]   Bq = round(64·255·b'), b' = b + a·128/255
//           [bg  : BGRA  × LOW_STRIDE]   模糊后的背景
//         每段第 0 个元素是左侧复制填充, LW+1 是右侧填充
//   out : 合成结果一行, BGRA uint8
//
// 计算 (对每个输出像素 x = 4j + r):
//   up(v)[x] = (wp[r]·v[j-1] + wc[r]·v[j] + wn[r]·v[j+1]) / 8
//              wp = {3,1,0,0}, wc = {5,7,7,5}, wn = {0,0,1,3}   (与 cv2 INTER_LINEAR ×4 一致)
//   I     = (29·B + 150·G + 77·R) >> 8                           (灰度, 导向图)
//   m     = sat_u8( (up(Aq)·(I-128) + up(Bq)) >> 6 )              (导向滤波输出 ×255)
//   out   = (px·m + up(bg)·(255-m)) >> 8
//
// 公共算子见 npu/lib/row_ops.h (其他逐行内核可复用同一套)。

#define NOCPP

#include <stdint.h>

#include "row_ops.h"

#ifndef LOW_STRIDE
#define LOW_STRIDE 352 // 每段元素数 (>= LW+2, 并让各段按 64 字节对齐)
#endif

using namespace rowops;

extern "C" {

void fusedCompositeLine(uint8_t *px, uint8_t *low, uint8_t *out,
                        int32_t width) {
  init();

  const int16_t *aq = (const int16_t *)low;
  volatile int32_t ls_v = LOW_STRIDE;   // 绕开 llvm-aie: 常量平面偏移 >=1024B 被丢掉
  const int32_t ls = ls_v;
  const int16_t *bq = aq + ls;
  const uint8_t *bgl = (const uint8_t *)(bq + ls);

  const Up4 w16 = Up4::make();
  const W8x4 w8 = W8x4::make();
  const v32i16 one = ::aie::broadcast<int16_t, 32>(1);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int x = 0; x < width; x += 32) {
    const int j0 = x >> 2; // 低分辨率索引 (未含 pad)

    // ---- 原图 32 像素, 拆成 B/G/R 平面 ----
    v32u8 p0 = ::aie::load_v<32>(px);
    v32u8 p1 = ::aie::load_v<32>(px + 32);
    v32u8 p2 = ::aie::load_v<32>(px + 64);
    v32u8 p3 = ::aie::load_v<32>(px + 96);
    px += 128;
    v32u8 vB, vG, vR;
    split_bgr(p0, p1, p2, p3, vB, vG, vR);
    v32i16 I = gray_m128(vB, vG, vR);

    // ---- 导向滤波系数放大并应用 ----
    v32i16 A = up4_i16(aq, j0, w16);
    v32i16 Bv = up4_i16(bq, j0, w16);
    ::aie::accum<acc32, 32> q = ::aie::mul(A, I);
    q = ::aie::mac(q, Bv, one);
    v32u8 m = q.template to_vector<uint8_t>(6); // 饱和到 0..255

    v32u8 mq0, mq1, mq2, mq3;
    expand4_u8(m, mq0, mq1, mq2, mq3);

    // ---- 背景水平 ×4 放大 ----
    const uint8_t *bb = bgl + 4 * j0;
    v32u8 sp0, sp1, sp2, sp3, sc0, sc1, sc2, sc3, sn0, sn1, sn2, sn3;
    rep4_px(::aie::load_unaligned_v<32>(bb, 4), sp0, sp1, sp2, sp3);
    rep4_px(::aie::load_unaligned_v<32>(bb + 4, 4), sc0, sc1, sc2, sc3);
    rep4_px(::aie::load_unaligned_v<32>(bb + 8, 4), sn0, sn1, sn2, sn3);

    // ---- 合成 ----
    ::aie::store_v(out, blend_u8(p0, up4_px(sp0, sc0, sn0, w8), mq0));
    ::aie::store_v(out + 32, blend_u8(p1, up4_px(sp1, sc1, sn1, w8), mq1));
    ::aie::store_v(out + 64, blend_u8(p2, up4_px(sp2, sc2, sn2, w8), mq2));
    ::aie::store_v(out + 96, blend_u8(p3, up4_px(sp3, sc3, sn3, w8), mq3));
    out += 128;
  }
}

} // extern "C"
