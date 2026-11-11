// RVM on GPU (ncnn + Vulkan): 一个极小的常驻 worker, 通过 stdin/stdout 和 Python 通信
//
//   ./rvm_ncnn_worker <model.param> <model.bin>
//   ./rvm_ncnn_worker <model.param> --info      # 只打印输入/输出名和形状
//
// 协议 (host 侧每帧):
//   stdin : [u32 width][u32 height][width*height*3 BGR]
//   stdout: [f32 alpha × width*height]      (值域 0..1, 行优先)
//
// 循环状态 r1..r4 常驻在进程里: 每帧从网络输出取回来, 下一帧再喂回去。
// 这样 Python 侧只需要搬一帧图和一张遮罩, 不用管模型状态。

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "net.h"

static bool read_all(void *p, size_t n) { return std::fread(p, 1, n, stdin) == n; }
static bool write_all(const void *p, size_t n) { return std::fwrite(p, 1, n, stdout) == n; }

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.param model.bin|--info\n", argv[0]);
    return 1;
  }
  const bool info_mode = std::strcmp(argv[2], "--info") == 0;

  ncnn::Net net;
  net.opt.use_vulkan_compute = true;   // 关键: 走 780M
  net.opt.num_threads = 1;
  if (net.load_param(argv[1]) != 0) {
    std::fprintf(stderr, "load_param failed: %s\n", argv[1]);
    return 1;
  }

  if (info_mode) {
    for (const char *n : net.input_names()) std::printf("in  %s\n", n);
    for (const char *n : net.output_names()) std::printf("out %s\n", n);
    std::printf("vulkan=%d\n", net.opt.use_vulkan_compute);
    return 0;
  }

  if (net.load_model(argv[2]) != 0) {
    std::fprintf(stderr, "load_model failed: %s\n", argv[2]);
    return 1;
  }

  std::vector<ncnn::Mat> state(4);          // r1..r4
  std::vector<int> state_ch = {16, 20, 40, 64};
  std::uint32_t w = 0, h = 0;
  std::vector<unsigned char> frame;

  while (read_all(&w, 4) && read_all(&h, 4)) {
    if (w == 0 || h == 0 || w > 8192 || h > 8192) break;
    frame.resize(static_cast<size_t>(w) * h * 3);
    if (!read_all(frame.data(), frame.size())) break;

    // 状态尺寸: 工作分辨率的 1/2,1/4,1/8,1/16 (向上取整), 第一帧建好
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
    ex.input("src", in);
    ex.input("r1i", state[0]);
    ex.input("r2i", state[1]);
    ex.input("r3i", state[2]);
    ex.input("r4i", state[3]);
    ncnn::Mat ratio(1);
    ratio[0] = 1.f;                       // 我们自己已经缩到工作分辨率
    ex.input("downsample_ratio", ratio);

    ncnn::Mat pha;
    if (ex.extract("pha", pha) != 0) {
      std::fprintf(stderr, "extract pha failed\n");
      return 1;
    }
    for (int i = 0; i < 4; i++) {
      ncnn::Mat s;
      char name[8];
      std::snprintf(name, sizeof(name), "r%do", i + 1);
      if (ex.extract(name, s) != 0) {
        std::fprintf(stderr, "extract %s failed\n", name);
        return 1;
      }
      state[i] = s;
    }

    // pha: 1 通道, w*h float
    std::vector<float> out(static_cast<size_t>(w) * h);
    for (std::uint32_t y = 0; y < h; y++) {
      const float *p = pha.row(y);
      std::memcpy(out.data() + static_cast<size_t>(y) * w, p, w * sizeof(float));
    }
    if (!write_all(out.data(), out.size() * sizeof(float))) break;
    std::fflush(stdout);
  }
  return 0;
}
