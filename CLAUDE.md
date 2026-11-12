# npu-studio-effects

Linux 上的摄像头背景虚化 / 替换（虚拟摄像头），目标硬件 Ryzen 7040（Phoenix，XDNA1 NPU）。概况见 `README.md`。

## NPU 开发

动 `npu/` 下任何东西之前，先读 skill **`.claude/skills/npu-dev/SKILL.md`**（硬规则、代码地图、加算子流程、性能测量、调试挂死）。
实验的原始记录在 `docs/npu-spikes/`（历史，结论以 skill 为准）。

```bash
source npu/env.sh          # 每个 shell 都要
python -m npu.check        # 改完 NPU 代码必须全部 ✅ (默认清 JIT 缓存)
```

- NPU 出错的典型样子是"不报错、只算错"——所有内核都和 numpy 整数参考逐字节比对，新功能要加进 `npu/check.py`。
- 改了 `row_ops.h`、`harness.py` 或 Python 模块常量后要 `rm -rf ~/.npu/cache/*`（JIT 缓存不认它们）。

## 其他

- studio（摄像头、合成、RVM 后端）：`python -m studio --help`；RVM 在 780M 上跑：`studio/rvm_gpu.py`。
- 功耗测量：`powerbench.py`、`powerbench_conv.py`（以整包 PPT 为准，见 README）。
- 提交信息用中文，风格参考 `git log`。
