# source npu/env.sh  —— 设置 IRON (mlir-aie) + Peano 的环境变量
# 在 Arch/Manjaro 上 XRT 装在 /usr, pyxrt 来自系统 Python 3.14
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)"
_SITE="$_ROOT/.venv-iron/lib/python3.14/site-packages"
export MLIR_AIE_INSTALL_DIR="$_SITE/mlir_aie"
export PEANO_INSTALL_DIR="$_SITE/llvm-aie"
export PATH="$MLIR_AIE_INSTALL_DIR/bin:$_ROOT/.venv-iron/bin:$PATH"
export PYTHONPATH="$MLIR_AIE_INSTALL_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$MLIR_AIE_INSTALL_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XILINX_XRT=/usr
export NPU2=0
unset _ROOT _SITE
