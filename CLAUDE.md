# npu-studio-effects

Linux 桌面 NPU 实验工具箱（目标硬件：AMD Ryzen 7 7840HS / Phoenix XDNA1 NPU）。项目结构与硬规则见 [`AGENTS.md`](AGENTS.md)。

## 快速命令

```bash
# 1. NPU 内核与 20 项回归测试 (每次改 npu/ 后必跑)
source npu/env.sh
.venv-iron/bin/python -m npu.check --quick

# 2. ASR 40 层四档基准测试
.venv-iron/bin/python -m npu.asr.encoder_npu --buckets --iters 5

# 3. 导出 C++ 四档包并编译常驻后端
.venv-iron/bin/python -m npu.asr.export_c_bundle
L=$PWD/.venv-iron/lib/python3.14/site-packages/opencv_python_headless.libs
g++ -O3 -std=c++17 -I/usr/include/xrt npu/asr/asr_npu.cpp -lxrt_coreutil \
    $L/libopenblasp-r0-59ffcd50.3.15.so -Wl,-rpath,$L -lm -o ~/.npu/asr/asr_npu_cpp

# 4. 构建并安装 NPU Studio 桌面端
./app/install.sh
```

- 改 `npu/` 前请先阅读 `.claude/skills/npu-dev/SKILL.md`。
- 提交信息请使用中文，风格参考 `git log`。
