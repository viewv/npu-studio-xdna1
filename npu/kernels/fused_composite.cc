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
// 所有通道拆分/扩展都用 2 路 interleave_zip / unzip 完成, 每次迭代 32 个像素。

#define NOCPP

#include <stdint.h>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef LOW_STRIDE
#define LOW_STRIDE 352 // 每段元素数 (>= LW+2, 并让各段按 64 字节对齐)
#endif

using v32u8 = ::aie::vector<uint8_t, 32>;
using v32i16 = ::aie::vector<int16_t, 32>;
using v16i16 = ::aie::vector<int16_t, 16>;

// 8 个 int16 → 每个重复 4 次 → 32 lanes
static inline v32i16 rep4_i16(v16i16 v16) {
  // 只用低 8 个元素
  auto [a, b] = ::aie::interleave_zip(v16, v16, 1); // a: v0 v0 v1 v1 ... v7 v7
  auto [c, d] = ::aie::interleave_zip(a, a, 2);     // c: v0×4 .. v3×4, d: v4×4 .. v7×4
  return ::aie::concat(c, d);
}

// 8 个 int16 (三个相邻窗口) 水平 ×4 插值, 结果 32 lanes
static inline v32i16 upsample_i16(const int16_t *base, int j0, v32i16 wp,
                                  v32i16 wc, v32i16 wn) {
  v16i16 vp = ::aie::load_unaligned_v<16>(base + j0, 1);     // v[j-1]
  v16i16 vc = ::aie::load_unaligned_v<16>(base + j0 + 1, 1); // v[j]
  v16i16 vn = ::aie::load_unaligned_v<16>(base + j0 + 2, 1); // v[j+1]
  ::aie::accum<acc32, 32> acc = ::aie::mul(rep4_i16(vp), wp);
  acc = ::aie::mac(acc, rep4_i16(vc), wc);
  acc = ::aie::mac(acc, rep4_i16(vn), wn);
  return acc.template to_vector<int16_t>(3);
}

// 8 个 BGRA 像素 (32 字节) → 每个像素重复 4 次 → 4 个 32 字节向量 (32 像素)
static inline void rep4_px(v32u8 v, v32u8 &o0, v32u8 &o1, v32u8 &o2,
                           v32u8 &o3) {
  auto [h0, h1] = ::aie::interleave_zip(v, v, 4);  // 每像素 ×2
  auto [a0, a1] = ::aie::interleave_zip(h0, h0, 8); // 每像素 ×4
  auto [b0, b1] = ::aie::interleave_zip(h1, h1, 8);
  o0 = a0;
  o1 = a1;
  o2 = b0;
  o3 = b1;
}

static inline v32u8 blend_u8(v32u8 fg, v32u8 bg, v32u8 m, v32u8 full) {
  ::aie::accum<acc32, 32> acc = ::aie::mul(fg, m);
  acc = ::aie::mac(acc, bg, ::aie::sub(full, m));
  return acc.template to_vector<uint8_t>(8);
}

extern "C" {

void fusedCompositeLine(uint8_t *px, uint8_t *low, uint8_t *out,
                        int32_t width) {
  ::aie::set_saturation(aie::saturation_mode::saturate);

  const int16_t *aq = (const int16_t *)low;
  const int16_t *bq = aq + LOW_STRIDE;
  const uint8_t *bgl = (const uint8_t *)(bq + LOW_STRIDE);

  // 每 4 个输出像素一组的插值权重 (/8)
  const int16_t wp_t[32] = {3, 1, 0, 0, 3, 1, 0, 0, 3, 1, 0, 0, 3, 1, 0, 0,
                            3, 1, 0, 0, 3, 1, 0, 0, 3, 1, 0, 0, 3, 1, 0, 0};
  const int16_t wc_t[32] = {5, 7, 7, 5, 5, 7, 7, 5, 5, 7, 7, 5, 5, 7, 7, 5,
                            5, 7, 7, 5, 5, 7, 7, 5, 5, 7, 7, 5, 5, 7, 7, 5};
  const int16_t wn_t[32] = {0, 0, 1, 3, 0, 0, 1, 3, 0, 0, 1, 3, 0, 0, 1, 3,
                            0, 0, 1, 3, 0, 0, 1, 3, 0, 0, 1, 3, 0, 0, 1, 3};
  // 背景按字节: 每个输出像素 4 字节同一权重
  const uint8_t bp_t[32] = {3, 3, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                            3, 3, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
  const uint8_t bc_t[32] = {5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5,
                            5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5};
  const uint8_t bn_t[32] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3,
                            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3};
  const v32i16 wp = ::aie::load_v<32>(wp_t);
  const v32i16 wc = ::aie::load_v<32>(wc_t);
  const v32i16 wn = ::aie::load_v<32>(wn_t);
  const v32u8 bpw = ::aie::load_v<32>(bp_t);
  const v32u8 bcw = ::aie::load_v<32>(bc_t);
  const v32u8 bnw = ::aie::load_v<32>(bn_t);

  const v32u8 full = ::aie::broadcast<uint8_t, 32>(255);
  const v32u8 wB = ::aie::broadcast<uint8_t, 32>(29);
  const v32u8 wG = ::aie::broadcast<uint8_t, 32>(150);
  const v32u8 wR = ::aie::broadcast<uint8_t, 32>(77);
  const v32i16 c128 = ::aie::broadcast<int16_t, 32>(128);
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
    auto [e01, o01] = ::aie::interleave_unzip(p0, p1, 1); // B R ..  / G A ..
    auto [e23, o23] = ::aie::interleave_unzip(p2, p3, 1);
    auto [vB, vR] = ::aie::interleave_unzip(e01, e23, 1);
    auto [vG, vA] = ::aie::interleave_unzip(o01, o23, 1);
    (void)vA;

    // ---- 灰度 (int16) 并减 128 ----
    ::aie::accum<acc32, 32> g = ::aie::mul(vB, wB);
    g = ::aie::mac(g, vG, wG);
    g = ::aie::mac(g, vR, wR);
    v32i16 I = ::aie::sub(g.template to_vector<int16_t>(8), c128);

    // ---- 导向滤波系数放大并应用 ----
    v32i16 A = upsample_i16(aq, j0, wp, wc, wn);
    v32i16 Bv = upsample_i16(bq, j0, wp, wc, wn);
    ::aie::accum<acc32, 32> q = ::aie::mul(A, I);
    q = ::aie::mac(q, Bv, one);
    v32u8 m = q.template to_vector<uint8_t>(6); // 饱和到 0..255

    // m 每像素扩展到 4 字节
    auto [m2a, m2b] = ::aie::interleave_zip(m, m, 1);
    auto [mq0, mq1] = ::aie::interleave_zip(m2a, m2a, 2);
    auto [mq2, mq3] = ::aie::interleave_zip(m2b, m2b, 2);

    // ---- 背景水平 ×4 放大 ----
    const uint8_t *bb = bgl + 4 * j0;
    v32u8 sp0, sp1, sp2, sp3, sc0, sc1, sc2, sc3, sn0, sn1, sn2, sn3;
    rep4_px(::aie::load_unaligned_v<32>(bb, 4), sp0, sp1, sp2, sp3);
    rep4_px(::aie::load_unaligned_v<32>(bb + 4, 4), sc0, sc1, sc2, sc3);
    rep4_px(::aie::load_unaligned_v<32>(bb + 8, 4), sn0, sn1, sn2, sn3);

#define BG_UP(k)                                                               \
  ({                                                                           \
    ::aie::accum<acc32, 32> t = ::aie::mul(sp##k, bpw);                        \
    t = ::aie::mac(t, sc##k, bcw);                                             \
    t = ::aie::mac(t, sn##k, bnw);                                             \
    t.template to_vector<uint8_t>(3);                                          \
  })
    v32u8 bg0 = BG_UP(0), bg1 = BG_UP(1), bg2 = BG_UP(2), bg3 = BG_UP(3);
#undef BG_UP

    // ---- 合成 ----
    ::aie::store_v(out, blend_u8(p0, bg0, mq0, full));
    ::aie::store_v(out + 32, blend_u8(p1, bg1, mq1, full));
    ::aie::store_v(out + 64, blend_u8(p2, bg2, mq2, full));
    ::aie::store_v(out + 96, blend_u8(p3, bg3, mq3, full));
    out += 128;
  }
}

} // extern "C"
