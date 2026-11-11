#!/usr/bin/env bash
# 编译 RVM 的 Vulkan worker (ncnn 后端)
#
#   bash studio/native/build_rvm_worker.sh
#
# 需要一个带 Vulkan 的 ncnn。Arch/Manjaro 上不需要 sudo、也不需要编译:
#   cd third_party && curl -sL -o ncnn.pkg.tar.zst \
#       "https://archlinux.org/packages/extra/x86_64/ncnn/download/" \
#     && mkdir -p ncnn-pkg && tar --zstd -xf ncnn.pkg.tar.zst -C ncnn-pkg
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NCNN="${NCNN_PREFIX:-$ROOT/third_party/ncnn-pkg/usr}"
OUT="$ROOT/third_party/rvm_worker"

if [[ ! -f "$NCNN/lib/libncnn.so" ]]; then
    echo "找不到 ncnn: $NCNN/lib/libncnn.so" >&2
    echo "见本脚本开头的注释 (下载 Arch 包解压, 不需要 sudo)" >&2
    exit 1
fi
if ! ldd "$NCNN/lib/libncnn.so" | grep -q glslang; then
    echo "警告: 这个 ncnn 可能没编译 Vulkan 支持" >&2
fi

g++ -O2 -std=c++17 \
    -I "$NCNN/include/ncnn" \
    "$ROOT/studio/native/rvm_worker.cpp" \
    -o "$OUT" \
    -L "$NCNN/lib" -lncnn \
    -Wl,-rpath,"$NCNN/lib"

echo "已生成 $OUT"
