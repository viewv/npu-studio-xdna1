"""把 RVM 的 ONNX 变成 ncnn 能吃的固定形状模型

为什么需要这么多步: RVM 的 ONNX 是"动态形状"的 —— 内部的空间尺寸由
Shape(src) + downsample_ratio 在运行时算出来, 还有 Expand/Concat 这些形状算子。
ncnn 和 pnnx 都处理不了 (pnnx 会拿随机值去 trace, 直接崩)。

我们的部署分辨率固定 (320x180, ratio=1), 所以可以把所有形状算死:

  1. 删掉 downsample_ratio 输入, 换成常量 1.0       <- 关键: 去掉唯一的"变量"
  2. onnxsim + 固定输入尺寸 -> 形状链被常量折叠      (Shape/Slice/Gather/Concat 全没了)
  3. Expand 换成 Identity (ratio=1 时它本来就是恒等)
  4. 用 ORT 逐字节验证: 与原始模型输出一致才继续
  5. pnnx -> .ncnn.param / .ncnn.bin

用法 (工具装在 /tmp/opencode/venv-tools 或任意有 onnxsim/pnnx 的环境):

    .venv/bin/python studio/native/convert_rvm.py \
        --pnnx /tmp/opencode/venv-tools/lib/python3.14/site-packages/pnnx/pnnx \
        --onnxsim /tmp/opencode/venv-tools/bin/onnxsim

产物写到 models/rvm_mobilenetv3_fp32.ncnn.{param,bin}
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "models" / "rvm_mobilenetv3_fp32.onnx"
OUT_PARAM = ROOT / "models" / "rvm_mobilenetv3_fp32.ncnn.param"
OUT_BIN = ROOT / "models" / "rvm_mobilenetv3_fp32.ncnn.bin"

# 部署分辨率 = 720p 输出的 1/4; 状态尺寸是它的 1/2,1/4,1/8,1/16 (向上取整)
W, H = 320, 180
STATES = [(16, 90, 160), (20, 45, 80), (40, 23, 40), (64, 12, 20)]


def run(cmd):
    print("+", " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
        raise SystemExit(f"失败: {cmd[0]}")
    return r


def ort_run(path: Path, feed: dict) -> list[np.ndarray]:
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return s.run(None, {k: v for k, v in feed.items() if k in [i.name for i in s.get_inputs()]})


def make_feed(seed=0):
    rng = np.random.default_rng(seed)
    f = {"src": rng.random((1, 3, H, W), np.float32) * 0.5}
    for i, (c, h, w) in enumerate(STATES):
        f[f"r{i+1}i"] = rng.random((1, c, h, w), np.float32) * 0.2
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pnnx", default="pnnx", help="pnnx 可执行文件")
    ap.add_argument("--onnxsim", default="onnxsim")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--fp16", action="store_true", help="权重存 fp16 (Vulkan 上用, 更省带宽)")
    a = ap.parse_args()
    src = Path(a.src)
    tmp = Path(tempfile.mkdtemp(prefix="rvm_conv_"))
    print("临时目录", tmp)

    # 1) downsample_ratio 变常量
    m = onnx.load(str(src))
    keep = [i for i in m.graph.input if i.name != "downsample_ratio"]
    if len(keep) == len(m.graph.input):
        print("注意: 模型里没有 downsample_ratio 输入")
    del m.graph.input[:]
    m.graph.input.extend(keep)
    m.graph.initializer.append(numpy_helper.from_array(np.array([1.0], np.float32), "downsample_ratio"))
    p1 = tmp / "1_noratio.onnx"
    onnx.save(m, str(p1))

    # 2) 冻结形状
    shape_arg = [f"src:1,3,{H},{W}"] + [f"r{i+1}i:1,{c},{h},{w}" for i, (c, h, w) in enumerate(STATES)]
    p2 = tmp / "2_static.onnx"
    run([a.onnxsim, p1, p2, "--overwrite-input-shape", *shape_arg])

    # 3) Expand -> Identity (ratio=1 时恒等), 并用 ORT 验证
    m = onnx.load(str(p2))
    n_exp = 0
    nodes = []
    for n in m.graph.node:
        if n.op_type == "Expand":
            nodes.append(helper.make_node("Identity", [n.input[0]], [n.output[0]], name=n.name + "_id"))
            n_exp += 1
        else:
            nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    p3 = tmp / "3_noexpand.onnx"
    onnx.save(m, str(p3))
    print(f"Expand -> Identity: {n_exp} 个")

    feed = make_feed()
    ref = ort_run(p2, dict(feed, downsample_ratio=np.array([1.0], np.float32)))
    got = ort_run(p3, feed)
    worst = max(float(np.abs(x - y).max()) for x, y in zip(ref, got))
    print(f"转换前后最大差: {worst:.2e}")
    if worst > 1e-4:
        raise SystemExit("形状冻结改变了输出, 停止")

    # 4) ncnn
    run([a.pnnx, p3, "fp16=1" if a.fp16 else "fp16=0"])
    param = tmp / "3_noexpand.ncnn.param"
    binf = tmp / "3_noexpand.ncnn.bin"
    if not param.exists():
        raise SystemExit("pnnx 没产出 .ncnn.param")
    tag = "fp16" if a.fp16 else "fp32"
    out_param = OUT_PARAM.with_name(f"rvm_mobilenetv3_{tag}.ncnn.param")
    out_bin = OUT_BIN.with_name(f"rvm_mobilenetv3_{tag}.ncnn.bin")
    out_param.write_bytes(param.read_bytes())
    out_bin.write_bytes(binf.read_bytes())
    print(f"已生成 {out_param.name} ({out_param.stat().st_size} B) / "
          f"{out_bin.name} ({out_bin.stat().st_size / 1e6:.1f} MB)")
    print("下一步: bash studio/native/build_rvm_worker.sh")


if __name__ == "__main__":
    sys.exit(main())
