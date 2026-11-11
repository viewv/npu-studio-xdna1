// RVM 尾部融合: 1/4 分辨率 → 全分辨率 (上采样 + refiner 残差 + 合成)
//
// 对应 ONNX 里 Resize_332 / Resize_346 之后的那一段 (见 npu/rvm/tail.py 的注释):
// 原来要在全分辨率上写两个 4 通道 fp32 张量 (720p 每帧 ~24 MB), 再算乘加、Split、
// Clip, 最后才合成。这里把整段压成一遍行扫描, 输入只有 1/4 分辨率的两个 4 通道
// 张量 (每帧 ~1.8 MB)。
//
// 每次调用处理一行 (宽 W):
//   src : 原图一行 BGRA uint8
//   low : 低分辨率一行 (宽 LW = W/4, 已由 CPU 做垂直插值), 打包为
//           [P0..P3  int16 × LOW_STRIDE]   refiner.conv.6 输出 (4 通道), 定点 /256
//           [Q0..Q3  int16 × LOW_STRIDE]   box_filter 残差 (4 通道), 定点 /256
//           [bg      BGRA  × LOW_STRIDE]   模糊后的背景
//         每段首尾各 1 个复制填充 (与 npu/fused.py 的低分辨率行布局一致)
//   out : 合成结果一行 BGRA
//
// 计算 (x = 4j + r):
//   u  = src 通道 (0..255),  g = (B+G+R)/3
//   fgr_c = clamp_u8( (P_c·u + Q_c·255) >> 8 + u )        c = 0,1,2
//   m     = clamp_u8( (P_3·g + Q_3·255) >> 8 )
//   out   = (fgr·m + up(bg)·(255-m)) >> 8
//
// ⚠ 一个必须绕开的编译器坑: 如果在本轮里"算完放大结果 → 立刻送进 16 位 MAC",
//   第 0、1 (mod 4) lane 的累加器会被多加一个常数 (实测 +95 / +31, 与数据无关),
//   相当于同一个 up4 结果在"转 uint8"和"送 MAC"两条路上拿到了不同的值。
//   已排除: 权重按值传参 / 权重改局部数组 / 结果过一遍本地内存 / 拆 16 lane 累加器 /
//           换 u 的宽化方式 / mac 换 add / 折项 / 流水线 pragma。
//   解法(已验证到一半): 手工软件流水 —— 把放大结果做成跨迭代存活的变量, 下一轮的
//   提前算好。单通道/两通道(<=4 个跨迭代向量)能做到逐字节正确;
//   8 个(本内核需要 4 个 P + 4 个 Q)仍然错 —— 16 lane 也一样, 所以不是寄存器宽度问题。
//   下一步: 把本内核拆成每遍 <=4 个跨迭代向量的多遍实现 (中间结果存一行 scratch)。
//
// 公共算子见 npu/lib/row_ops.h。

#define NOCPP

#include <stdint.h>

#include "row_ops.h"

#ifndef LOW_STRIDE
#define LOW_STRIDE 352
#endif

using namespace rowops;

// fgr_c = clamp((P·u + Q·255) >> 8 + u, 0, 255);  +u*256 与 >>8 合并不丢精度
static inline v32u8 fgr_ch(v32i16 P, v32i16 Q, v32i16 u) {
  const v32i16 c255 = ::aie::broadcast<int16_t, 32>(255);
  const v32i16 c256 = ::aie::broadcast<int16_t, 32>(256);
  ::aie::accum<acc32, 32> a = ::aie::mul(P, u);
  a = ::aie::mac(a, Q, c255);
  a = ::aie::mac(a, u, c256);
  return a.template to_vector<uint8_t>(8);
}

// m = clamp((P·g + Q·255) >> 8, 0, 255)
static inline v32u8 alpha_ch(v32i16 P, v32i16 Q, v32i16 g) {
  const v32i16 c255 = ::aie::broadcast<int16_t, 32>(255);
  ::aie::accum<acc32, 32> a = ::aie::mul(P, g);
  a = ::aie::mac(a, Q, c255);
  return a.template to_vector<uint8_t>(8);
}

extern "C" {

void rvmTailLine(uint8_t *src, uint8_t *low, uint8_t *out, int32_t width) {
  init();

  const int16_t *P = (const int16_t *)low;            // 4 个平面
  const int16_t *Q = P + 4 * LOW_STRIDE;              // 4 个平面
  const uint8_t *bgl = low + 16 * LOW_STRIDE;         // BGRA
  const Up4 w16 = Up4::make();
  const W8x4 w8 = W8x4::make();
  const v32u8 a255 = ::aie::broadcast<uint8_t, 32>(255);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int x = 0; x < width; x += 32) {
    const int j0 = x >> 2;
    v32i16 p0 = up4_i16(P, j0, w16), p1 = up4_i16(P + LOW_STRIDE, j0, w16);
    v32i16 p2 = up4_i16(P + 2 * LOW_STRIDE, j0, w16), p3 = up4_i16(P + 3 * LOW_STRIDE, j0, w16);
    v32i16 q0 = up4_i16(Q, j0, w16), q1 = up4_i16(Q + LOW_STRIDE, j0, w16);
    v32i16 q2 = up4_i16(Q + 2 * LOW_STRIDE, j0, w16), q3 = up4_i16(Q + 3 * LOW_STRIDE, j0, w16);

    // ---- 原图 32 像素 → B/G/R 平面 ----
    v32u8 s0 = ::aie::load_v<32>(src);
    v32u8 s1 = ::aie::load_v<32>(src + 32);
    v32u8 s2 = ::aie::load_v<32>(src + 64);
    v32u8 s3 = ::aie::load_v<32>(src + 96);
    src += 128;
    v32u8 vB, vG, vR;
    split_bgr(s0, s1, s2, s3, vB, vG, vR);

    v32u8 fB = fgr_ch(p0, q0, widen_u8(vB));
    v32u8 fG = fgr_ch(p1, q1, widen_u8(vG));
    v32u8 fR = fgr_ch(p2, q2, widen_u8(vR));
    v32u8 m = alpha_ch(p3, q3, mean3_u8(vB, vG, vR));

    v32u8 f0, f1, f2, f3;
    bgra_from_planes(fB, fG, fR, a255, f0, f1, f2, f3);
    v32u8 mq0, mq1, mq2, mq3;
    expand4_u8(m, mq0, mq1, mq2, mq3);

    // ---- 背景水平 ×4 放大 ----
    const uint8_t *bb = bgl + 4 * j0;
    v32u8 sp0, sp1, sp2, sp3, sc0, sc1, sc2, sc3, sn0, sn1, sn2, sn3;
    rep4_px(::aie::load_unaligned_v<32>(bb, 4), sp0, sp1, sp2, sp3);
    rep4_px(::aie::load_unaligned_v<32>(bb + 4, 4), sc0, sc1, sc2, sc3);
    rep4_px(::aie::load_unaligned_v<32>(bb + 8, 4), sn0, sn1, sn2, sn3);

    ::aie::store_v(out, blend_u8(f0, up4_px(sp0, sc0, sn0, w8), mq0));
    ::aie::store_v(out + 32, blend_u8(f1, up4_px(sp1, sc1, sn1, w8), mq1));
    ::aie::store_v(out + 64, blend_u8(f2, up4_px(sp2, sc2, sn2, w8), mq2));
    ::aie::store_v(out + 96, blend_u8(f3, up4_px(sp3, sc3, sn3, w8), mq3));
    out += 128;
  }
}

} // extern "C"
