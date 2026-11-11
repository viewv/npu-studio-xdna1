// RVM on GPU (ncnn + Vulkan): 常驻 worker, 通过 stdin/stdout 和 Python 通信
//
//   ./rvm_worker <model.param> <model.bin>
//   ./rvm_worker <model.param> --info
//
// 协议 (每帧):
//   stdin : [u32 width][u32 height][width*height*3 BGR]
//   stdout: [f32 alpha × width*height]      (0..1, 行优先)
//
// 循环状态 r1..r4 常驻进程: 每帧从网络输出取回来, 下一帧再喂回去。
//
// 模型是 pnnx 转出来的固定形状版本 (见 studio/native/README.md):
//   输入  in0=src  in1=r1i in2=r2i in3=r3i in4=r4i
//   输出  out0=fgr(3ch) out1=pha(1ch) out2..out5=r1o..r4o
// 只取 pha —— 前景仍用原始画面。

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "net.h"

static const char *IN_NAMES[5] = {"in0", "in1", "in2", "in3", "in4"};
static const char *OUT_STATE[4] = {"out2", "out3", "out4", "out5"};
static const char *OUT_ALPHA = "out1";

static bool read_all(void *p, size_t n) { return std::fread(p, 1, n, stdin) == n; }
static bool write_all(const void *p, size_t n) { return std::fwrite(p, 1, n, stdout) == n; }

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.param model.bin|--info\n", argv[0]);
    return 1;
  }
  const bool info = std::strcmp(argv[2], "--info") == 0;

  ncnn::Net net;
  net.opt.use_vulkan_compute = true;
  net.opt.num_threads = 1;
  // 780M 的 Vulkan 支持 fp16 存储/算术 (fp16-p/s/u/a=1/1/1/1)。实测比纯 fp32
  // 快约 7%, 且输出与 fp32 完全一致; 不支持 fp16 的设备 ncnn 会自动退化。
  net.opt.use_fp16_packed = true;
  net.opt.use_fp16_storage = true;
  net.opt.use_fp16_arithmetic = true;
  if (net.load_param(argv[1]) != 0) {
    std::fprintf(stderr, "load_param failed: %s\n", argv[1]);
    return 1;
  }
  if (info) {
    for (const char *n : net.input_names()) std::printf("in  %s\n", n);
    for (const char *n : net.output_names()) std::printf("out %s\n", n);
    return 0;
  }
  if (net.load_model(argv[2]) != 0) {
    std::fprintf(stderr, "load_model failed: %s\n", argv[2]);
    return 1;
  }

  const int state_ch[4] = {16, 20, 40, 64};
  std::vector<ncnn::Mat> state(4);
  std::uint32_t w = 0, h = 0;
  std::vector<unsigned char> frame;
  double t_net = 0;
  int nf = 0;

  while (read_all(&w, 4) && read_all(&h, 4)) {
    if (w == 0 || h == 0 || w > 8192 || h > 8192) break;
    frame.resize(static_cast<size_t>(w) * h * 3);
    if (!read_all(frame.data(), frame.size())) break;

    if (state[0].empty()) {
      for (int i = 0; i < 4; i++) {
        int sw = (static_cast<int>(w) + (1 << (i + 1)) - 1) >> (i + 1);
        int sh = (static_cast<int>(h) + (1 << (i + 1)) - 1) >> (i + 1);
        state[i] = ncnn::Mat(sw, sh, state_ch[i]);
        state[i].fill(0.f);
      }
    }

    ncnn::Mat in = ncnn::Mat::from_pixels(frame.data(), ncnn::Mat::PIXEL_BGR2RGB, w, h);
    const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(nullptr, norm);

    ncnn::Extractor ex = net.create_extractor();
    for (int i = 0; i < 4; i++) ex.input(IN_NAMES[i + 1], state[i]);
    ex.input(IN_NAMES[0], in);

    ncnn::Mat pha;
    if (ex.extract(OUT_ALPHA, pha) != 0) {
      std::fprintf(stderr, "extract %s failed\n", OUT_ALPHA);
      return 1;
    }
    for (int i = 0; i < 4; i++) {
      ncnn::Mat s;
      if (ex.extract(OUT_STATE[i], s) != 0) {
        std::fprintf(stderr, "extract %s failed\n", OUT_STATE[i]);
        return 1;
      }
      state[i] = s;
    }

    std::vector<float> out(static_cast<size_t>(w) * h);
    for (std::uint32_t y = 0; y < h; y++) {
      const float *p = pha.row(y);
      std::memcpy(out.data() + static_cast<size_t>(y) * w, p, w * sizeof(float));
    }
    if (!write_all(out.data(), out.size() * sizeof(float))) break;
    std::fflush(stdout);
    nf++;
    (void)t_net;
  }
  if (nf) std::fprintf(stderr, "frames=%d\n", nf);
  return 0;
}
