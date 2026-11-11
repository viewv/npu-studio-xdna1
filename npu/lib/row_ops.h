// 可复用的 AIE2 逐行像素算子 (Phoenix / XDNA1)
//
// 约定: 所有函数一次处理 32 个输出像素; BGRA 用 32 字节向量 (8 像素),
// 因此一个输出块 = 4 个 v32u8。定点运算默认 向下取整 (floor), 与 numpy 的 >> 一致。
//
// 目前提供:
//   Up4 / W8x4      水平 ×4 双线性权重 (与 cv2 INTER_LINEAR ×4 相同, /8)
//   up4_i16         单通道 int16: 8 个低分辨率元素 → 32 个输出
//   up4_px          4 通道 uint8 (BGRA): 8 个低分辨率像素 → 32 个输出像素
//   rep4_px         8 个 BGRA 像素 → 每个重复 4 次 (配合 up4_* 的权重排布)
//   split_bgr       32 个 BGRA 像素 → B / G / R 三个平面
//   gray_m128       灰度 (29B+150G+77R)>>8 - 128, int16
//   expand4_u8      1 个 32 lane 的 mask → 每像素 4 字节 (4 个向量)
//   blend_u8        (fg·m + bg·(255-m)) >> 8
//
// 新增逐行内核时: #include "row_ops.h", 开头 rowops::init(), 然后按上面的约定写循环。
//
// ⚠ 编译器坑 (llvm-aie 21.0.0.2026062301, AIE2): "基址 + 编译期常量偏移" 的非对齐向量加载,
//   偏移 >= 1024 字节时会被折成 post-increment 形式 (`vlda wl, [p], #off+32`), 先用基址加载、
//   后加偏移, 读到的是错误平面 (实测 992 B 正确, 1024 B 起错)。表现为 up4_* 的 v[j-1]
//   来自另一个平面, 即 x%4∈{0,1} 的 lane 出错。
//   规则: 一行里多个平面的偏移一律用 plane_stride() 取成运行时值, 不要写 base + CONST。
#ifndef NPU_ROW_OPS_H
#define NPU_ROW_OPS_H

#include <stdint.h>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

namespace rowops {

using v32u8 = ::aie::vector<uint8_t, 32>;
using v32i16 = ::aie::vector<int16_t, 32>;
using v16i16 = ::aie::vector<int16_t, 16>;

/// 把编译期的平面步长变成编译器看不穿的运行时值 (绕开上面的 post-increment 错折)
inline int32_t plane_stride(int32_t n) {
  volatile int32_t v = n;
  return v;
}

/// 每个行内核开头调用一次
inline void init() {
  ::aie::set_saturation(::aie::saturation_mode::saturate);
  ::aie::set_rounding(::aie::rounding_mode::floor);
}

// ---------------------------------------------------------------- 水平 ×4 权重

/// 单通道 (int16) 的 ×4 双线性权重
struct Up4 {
  v32i16 wp, wc, wn;
  static inline Up4 make() {
    const int16_t p[32] = {3, 1, 0, 0, 3, 1, 0, 0, 3, 1, 0, 0, 3, 1, 0, 0,
                                  3, 1, 0, 0, 3, 1, 0, 0, 3, 1, 0, 0, 3, 1, 0, 0};
    const int16_t c[32] = {5, 7, 7, 5, 5, 7, 7, 5, 5, 7, 7, 5, 5, 7, 7, 5,
                                  5, 7, 7, 5, 5, 7, 7, 5, 5, 7, 7, 5, 5, 7, 7, 5};
    const int16_t n[32] = {0, 0, 1, 3, 0, 0, 1, 3, 0, 0, 1, 3, 0, 0, 1, 3,
                                  0, 0, 1, 3, 0, 0, 1, 3, 0, 0, 1, 3, 0, 0, 1, 3};
    return {::aie::load_v<32>(p), ::aie::load_v<32>(c), ::aie::load_v<32>(n)};
  }
};

/// 4 通道 (uint8, 每像素 4 字节共用权重) 的 ×4 双线性权重
struct W8x4 {
  v32u8 pb, pc, pn;
  static inline W8x4 make() {
    const uint8_t p[32] = {3, 3, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                  3, 3, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint8_t c[32] = {5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5,
                                  5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5};
    const uint8_t n[32] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3,
                                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3};
    return {::aie::load_v<32>(p), ::aie::load_v<32>(c), ::aie::load_v<32>(n)};
  }
};

// ---------------------------------------------------------------- 展开 / 升采样

/// 8 个 int16 → 每个重复 4 次 → 32 lanes
inline v32i16 rep4_i16(v16i16 v) {
  auto [a, b] = ::aie::interleave_zip(v, v, 1);   // v0 v0 v1 v1 ... v7 v7
  auto [c, d] = ::aie::interleave_zip(a, a, 2);   // v0×4 .. v3×4 / v4×4 .. v7×4
  return ::aie::concat(c, d);
}

/// 8 个 BGRA 像素 (32 字节) → 每个像素重复 4 次 → 4 个 32 字节向量 (32 像素)
inline void rep4_px(v32u8 v, v32u8 &o0, v32u8 &o1, v32u8 &o2, v32u8 &o3) {
  auto [h0, h1] = ::aie::interleave_zip(v, v, 4);   // 每像素 ×2
  auto [a0, a1] = ::aie::interleave_zip(h0, h0, 8); // 每像素 ×4
  auto [b0, b1] = ::aie::interleave_zip(h1, h1, 8);
  o0 = a0;
  o1 = a1;
  o2 = b0;
  o3 = b1;
}

/// 单通道 int16: 8 个低分辨率元素 (三个相邻窗口) → 32 个输出
inline v32i16 up4_i16(const int16_t *base, int j0, const Up4 &w) {
  v16i16 vp = ::aie::load_unaligned_v<16>(base + j0, 1);     // v[j-1]
  v16i16 vc = ::aie::load_unaligned_v<16>(base + j0 + 1, 1); // v[j]
  v16i16 vn = ::aie::load_unaligned_v<16>(base + j0 + 2, 1); // v[j+1]
  ::aie::accum<acc32, 32> acc = ::aie::mul(rep4_i16(vp), w.wp);
  acc = ::aie::mac(acc, rep4_i16(vc), w.wc);
  acc = ::aie::mac(acc, rep4_i16(vn), w.wn);
  return acc.template to_vector<int16_t>(3);
}

/// 4 通道 uint8: 三个相邻的低分辨率像素 (各 4 字节) → 3 个输出像素的 12 字节
inline v32u8 up4_px(v32u8 p, v32u8 c, v32u8 n, const W8x4 &w) {
  ::aie::accum<acc32, 32> t = ::aie::mul(p, w.pb);
  t = ::aie::mac(t, c, w.pc);
  t = ::aie::mac(t, n, w.pn);
  return t.template to_vector<uint8_t>(3);
}

// ---------------------------------------------------------------- 通道拆分 / 合成

/// 32 个 BGRA 像素 (4 个向量) → B / G / R 三个平面 (A 丢弃)
inline void split_bgr(v32u8 p0, v32u8 p1, v32u8 p2, v32u8 p3,
                      v32u8 &B, v32u8 &G, v32u8 &R) {
  auto [e01, o01] = ::aie::interleave_unzip(p0, p1, 1);   // B R .. / G A ..
  auto [e23, o23] = ::aie::interleave_unzip(p2, p3, 1);
  auto [vB, vR] = ::aie::interleave_unzip(e01, e23, 1);
  auto [vG, vA] = ::aie::interleave_unzip(o01, o23, 1);
  (void)vA;
  B = vB;
  G = vG;
  R = vR;
}

/// 灰度 (29B + 150G + 77R) >> 8 - 128, int16
inline v32i16 gray_m128(v32u8 B, v32u8 G, v32u8 R) {
  const v32u8 wB = ::aie::broadcast<uint8_t, 32>(29);
  const v32u8 wG = ::aie::broadcast<uint8_t, 32>(150);
  const v32u8 wR = ::aie::broadcast<uint8_t, 32>(77);
  const v32i16 c128 = ::aie::broadcast<int16_t, 32>(128);
  ::aie::accum<acc32, 32> g = ::aie::mul(B, wB);
  g = ::aie::mac(g, G, wG);
  g = ::aie::mac(g, R, wR);
  return ::aie::sub(g.template to_vector<int16_t>(8), c128);
}

/// 32 lane 的 mask → 每像素 4 字节 (4 个向量)
inline void expand4_u8(v32u8 m, v32u8 &q0, v32u8 &q1, v32u8 &q2, v32u8 &q3) {
  auto [m2a, m2b] = ::aie::interleave_zip(m, m, 1);
  auto [a0, a1] = ::aie::interleave_zip(m2a, m2a, 2);
  auto [b0, b1] = ::aie::interleave_zip(m2b, m2b, 2);
  q0 = a0;
  q1 = a1;
  q2 = b0;
  q3 = b1;
}

/// (fg·m + bg·(255-m)) >> 8
inline v32u8 blend_u8(v32u8 fg, v32u8 bg, v32u8 m) {
  const v32u8 full = ::aie::broadcast<uint8_t, 32>(255);
  ::aie::accum<acc32, 32> acc = ::aie::mul(fg, m);
  acc = ::aie::mac(acc, bg, ::aie::sub(full, m));
  return acc.template to_vector<uint8_t>(8);
}

// ---------------------------------------------------------------- 平面 ↔ BGRA

/// uint8 → int16 (零扩展)
inline v32i16 widen_u8(v32u8 v) { return ::aie::unpack(v).template cast_to<int16_t>(); }

/// 三个通道的算术平均 (0..255 → int16), 与 fp32 的 (B+G+R)/3 相差 <1
inline v32i16 mean3_u8(v32u8 B, v32u8 G, v32u8 R) {
  v32i16 s = ::aie::add(::aie::add(widen_u8(B), widen_u8(G)), widen_u8(R));
  ::aie::accum<acc32, 32> a = ::aie::mul(s, ::aie::broadcast<int16_t, 32>(21846));
  return a.template to_vector<int16_t>(16);   // 21846/65536 ≈ 1/3
}

/// 4 个平面 (B/G/R/A 各 32 lane) → 32 个 BGRA 像素 (4 个向量), split_bgr 的逆
inline void bgra_from_planes(v32u8 B, v32u8 G, v32u8 R, v32u8 A,
                             v32u8 &o0, v32u8 &o1, v32u8 &o2, v32u8 &o3) {
  auto [bg_lo, bg_hi] = ::aie::interleave_zip(B, G, 1);   // B0 G0 B1 G1 ...
  auto [ra_lo, ra_hi] = ::aie::interleave_zip(R, A, 1);   // R0 A0 R1 A1 ...
  auto [p0, p1] = ::aie::interleave_zip(bg_lo, ra_lo, 2);
  auto [p2, p3] = ::aie::interleave_zip(bg_hi, ra_hi, 2);
  o0 = p0; o1 = p1; o2 = p2; o3 = p3;
}

} // namespace rowops

#endif // NPU_ROW_OPS_H
