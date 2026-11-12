"""单独编译一个内核 .cc, 打印各函数占多少程序内存 (每个 core 只有 16 KB)

  source npu/env.sh && python -m npu.ksize [npu/kernels/conv_c16.cc] [-DCW=40 …]

链接后的 core ELF = 这里的 .text 合计 + core 程序 (IRON 生成的 core_x_y, conv_c16 约 3.2 KB) + 启动代码 ≈ 合计 + 3.5 KB
(实测: 合计 11.76 KB → ELF .text 15.25 KB)。
超了的时候 aiecc 报 "Overflow of program memory" / "XAie_LoadElf failed with XAIE_INVALID_ELF"。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from aie.utils import config

from .fused import KERNEL_INC


def main(argv):
    src = next((a for a in argv if not a.startswith("-")), str(Path(__file__).parent / "kernels" / "conv_c16.cc"))
    defs = [a for a in argv if a.startswith("-D")] or ["-DCW=40"]
    cxx = config.peano_cxx_path()
    bindir = Path(cxx).parent
    with tempfile.TemporaryDirectory() as d:
        obj = f"{d}/k.o"
        subprocess.run([cxx, src, "-c", "-o", obj, f"-I{config.cxx_header_path()}", "-std=c++20", "-Wno-parentheses",
                        "-Wno-attributes", "-Wno-macro-redefined", "-Wno-empty-body", "-O2", "-DNDEBUG",
                        "-D__AIE_API_AIE_ADF_HPP__", "--target=aie2-none-unknown-elf", "-I", KERNEL_INC, *defs],
                       check=True)
        nm = subprocess.run([str(bindir / "llvm-nm"), "--size-sort", "-S", "-C", obj], capture_output=True, text=True,
                            check=True).stdout
    rows = []
    for line in nm.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) == 4 and parts[2].lower() == "t" and int(parts[1], 16):
            rows.append((int(parts[1], 16), parts[3]))
    for n, name in rows:
        print(f"{n:6d}  {name[:110]}")
    tot = sum(n for n, _ in rows)
    print(f"{tot:6d}  合计 (+ core 程序和启动代码 ≈ 3.5 KB → ≈ {(tot + 3500) / 1024:.1f} / 16 KB)")


if __name__ == "__main__":
    main(sys.argv[1:])
