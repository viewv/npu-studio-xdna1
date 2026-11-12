"""NPU 一键回归: 清 JIT 缓存 → 跑全部内核 (合成 / RVM 尾部 / 3×3 卷积 / 整网串联框架) 的逐字节自检 + 编译器探针 → 汇总

  source npu/env.sh && python -m npu.check [--quick] [--keep-cache] [-v]

    --quick       只跑 1280×720 (外加探针), 约快一半
    --keep-cache  不清 ~/.npu/cache (只在确定没改过内核/头文件时用)
    -v            打印每一项的详细输出 (默认只在失败时打印)

改内核、改 npu/lib/*.h、改 harness、换编译器 (mlir-aie / llvm-aie) 之后都要跑一遍,
全部 ✅ 才算数 —— NPU 内核出错的典型样子是"不报错、只算错"。退出码: 0 全过, 1 有失败。
"""
from __future__ import annotations

import contextlib
import io
import shutil
import sys
import time
import traceback
from pathlib import Path

CACHE = Path.home() / ".npu" / "cache"


def _cases(quick: bool):
    from . import fused
    from .rvm import tail
    sizes = [(1280, 720)] if quick else [(1280, 720), (1920, 1080)]
    for w, h in sizes:
        yield f"fused     {f"{w}×{h}":<9}", lambda w=w, h=h: fused.selftest(w, h, 4, 10)
    for w, h in sizes:
        cols = 4 if h % 16 == 0 else 3          # 垂直版每列行数要是 4 的倍数: 1080p 用 3 列
        yield f"fused-v   {f"{w}×{h}":<9}", lambda w=w, h=h, c=cols: fused.selftest_v(w, h, c, 10)
    for w, h in sizes:
        yield f"rvm_tail  {f"{w}×{h}":<9}", lambda w=w, h=h: tail.selftest(w, h, 4, 10)
    from . import conv3x3
    yield "conv3x3   Conv_290 ", lambda: conv3x3.selftest(2, 2, 10)
    # RVM 整网串联框架 (一次调用多层; 覆盖 s2 滑窗绕法、描述符驱动循环、不均分行、补宽写零)
    from . import chain_c16
    yield "chain_c16 enc 13 层", lambda: chain_c16.selftest("enc", 10)
    if not quick:
        yield "chain_c16 low 25 层", lambda: chain_c16.selftest("low", 10)
        yield "chain_c16 deep 13 层", lambda: chain_c16.selftest("deep", 10)   # OC / IC 分块


def _probe():
    from .offset_probe import wrong_fraction
    raw, fixed = wrong_fraction(512, False), wrong_fraction(512, True)
    if fixed:
        return False, f"plane_stride 绕过失效 (错 {fixed * 100:.1f}%), 所有多平面内核都要重查"
    if raw:
        return True, f"bug 仍在 (原样写法错 {raw * 100:.0f}%), plane_stride 绕过有效"
    return True, "原样写法也对了: 编译器已修复, plane_stride 可保留"


def main() -> int:
    quick, keep, verbose = "--quick" in sys.argv, "--keep-cache" in sys.argv, "-v" in sys.argv
    if not keep:
        n = sum(1 for _ in CACHE.glob("*")) if CACHE.exists() else 0
        for p in CACHE.glob("*") if CACHE.exists() else []:
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        print(f"清空 JIT 缓存 ({n} 项) ……… ok")

    results = []
    t_all = time.perf_counter()
    for name, fn in _cases(quick):
        log = io.StringIO()
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stderr(log):
                r = fn()
            ok, line = bool(r) and r.max_diff == 0, f"最大误差 {r.max_diff:<3d} {r.ms:6.2f} ms"
        except Exception:
            ok, line = False, "异常"
            log.write(traceback.format_exc())
        print(f"{name}  {line}  {'✅' if ok else '❌'}   ({time.perf_counter() - t0:.0f} s)", flush=True)
        if verbose or not ok:
            print("    " + log.getvalue().strip().replace("\n", "\n    "))
        results.append(ok)

    log = io.StringIO()
    try:
        with contextlib.redirect_stderr(log):
            ok, msg = _probe()
    except Exception:
        ok, msg = False, "异常"
        log.write(traceback.format_exc())
    print(f"编译器探针 (常量偏移 1024 B)  {msg}  {'✅' if ok else '❌'}", flush=True)
    if verbose or not ok:
        print("    " + log.getvalue().strip().replace("\n", "\n    "))
    results.append(ok)

    n_ok = sum(results)
    print(f"—— {n_ok}/{len(results)} 通过  ({time.perf_counter() - t_all:.0f} s)")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
