"""整机功耗对比 (交替测量空闲 / 各后端), 单位瓦特

  source npu/env.sh && python powerbench.py [--secs 60] [--warmup 15] [--backends cpu,npu,rusticl,rocm]

功率来源有两个, 同时记录:

  PPT  /sys/class/hwmon/hwmon*/power1_average   整包 (CPU+iGPU+NPU) 平均功率, 插电也能读, 首选
  BAT  /sys/class/power_supply/BAT*             电池放电功率, 整机口径 (含屏幕/硬盘/WiFi)

⚠ 这台 TUXEDO 的 BAT0 没有 power_now, 只能拿 current_now×voltage_now 估算, 而它给出的
  12.1V×98mA≈1.2W 与同一时刻的 PPT 19W 差 16 倍 —— 量纲是坏的。所以脚本以 PPT 为准,
  BAT 只做量级交叉验证并在差异过大时给出警告。

顺序: 空闲 → 各后端 → 各后端 (第二轮) → 空闲
每段丢弃开头 warmup 秒 (模型加载 / NPU 首次调用)。测量期间请勿操作电脑、不要调亮度。
"""
from __future__ import annotations

import glob
import os
import signal
import statistics
import subprocess
import sys
import time


def _sensors() -> tuple[str | None, str | None]:
    """自动找 PPT (amdgpu) 和电池, 不写死 hwmonN"""
    ppt = None
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            if open(f"{h}/name").read().strip() == "amdgpu":
                p = f"{h}/power1_average"
                if os.path.exists(p):
                    ppt = p
        except OSError:
            pass
    bat = None
    for b in sorted(glob.glob("/sys/class/power_supply/BAT*")):
        if os.path.exists(f"{b}/power_now") or os.path.exists(f"{b}/current_now"):
            bat = b
    return ppt, bat


PPT, BAT = _sensors()


def read_ppt() -> float | None:
    if not PPT:
        return None
    try:
        return int(open(PPT).read()) / 1e6
    except (OSError, ValueError):
        return None


def read_bat() -> float | None:
    if not BAT:
        return None
    try:
        if os.path.exists(f"{BAT}/power_now"):
            return int(open(f"{BAT}/power_now").read()) / 1e6
        i = int(open(f"{BAT}/current_now").read())
        v = int(open(f"{BAT}/voltage_now").read())
        return i * v / 1e12
    except (OSError, ValueError):
        return None


def on_battery(bat: str) -> bool:
    try:
        return open(f"{bat}/status").read().strip() == "Discharging"
    except OSError:
        return False


SECS = int(sys.argv[sys.argv.index("--secs") + 1]) if "--secs" in sys.argv else 60
WARM = int(sys.argv[sys.argv.index("--warmup") + 1]) if "--warmup" in sys.argv else 15
_RVM = ["--matting", "rvm", "--rvm-gpu", "--high-fps"]
BACKENDS = {"cpu": [], "npu": ["--npu"], "rusticl": ["--gpu", "rusticl"], "rocm": ["--gpu", "rocm"],
            # RVM@780M + NPU 合成 (产品默认配置)
            "rvm-npu": _RVM + ["--npu"],
            # 同上, 但 CPU 侧的低分辨率准备只做第一帧 (之后复用): "准备全部搬走"的功耗上限实验
            "rvm-npu-noprep": _RVM + ["--npu"]}
ENVS = {"rvm-npu-noprep": {"NPU_PREP_ORACLE": "1"}}
_order = sys.argv[sys.argv.index("--backends") + 1].split(",") if "--backends" in sys.argv else ["cpu", "npu"]
for _b in _order:
    if _b not in BACKENDS:
        sys.exit(f"未知后端 {_b} (可选: {', '.join(BACKENDS)})")
PHASES = ["idle"] + _order * 2 + ["idle"]


def phase(name: str) -> tuple[list[float], list[float], list[float]]:
    proc = None
    if name != "idle":
        cmd = [sys.executable, "-m", "studio", "--output", "none"] + BACKENDS[name]
        env = {**os.environ, **ENVS.get(name, {})}
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    ppt, bat, fps = [], [], []
    t0 = time.monotonic()
    try:
        while (el := time.monotonic() - t0) < SECS:
            if el >= WARM:
                p, b = read_ppt(), read_bat()
                if p is not None:
                    ppt.append(p)
                if b is not None:
                    bat.append(b)
            time.sleep(0.5)
    finally:
        if proc:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    if proc and proc.returncode not in (0, -2, 130):
        print(f"  ⚠ {name} 进程返回码 {proc.returncode}", file=sys.stderr)
    return ppt, bat, fps


def main():
    if not PPT and not BAT:
        sys.exit("找不到任何功率传感器")
    print(f"功率来源: PPT={PPT or '无'}  BAT={BAT or '无'}")
    if BAT and not os.path.exists(f"{BAT}/power_now"):
        print(f"  ⚠ {BAT} 没有 power_now, 电池读数只能估算, 以 PPT 为准", file=sys.stderr)
    if BAT and not on_battery(BAT):
        print("  提示: 当前插着电, PPT 仍有效; 电池读数会接近 0")
    print(f"每段 {SECS}s (丢弃前 {WARM}s), 顺序: {' → '.join(PHASES)}, 共约 {SECS*len(PHASES)//60+1} 分钟\n")

    res: dict[str, dict[str, list[float]]] = {}
    for i, name in enumerate(PHASES, 1):
        ppt, bat, _ = phase(name)
        res.setdefault(name, {"ppt": [], "bat": []})["ppt"].extend(ppt)
        res[name]["bat"].extend(bat)
        p = statistics.mean(ppt) if ppt else float("nan")
        b = statistics.mean(bat) if bat else float("nan")
        print(f"[{i}/{len(PHASES)}] {name:7} PPT {p:6.2f} W   电池 {b:6.2f} W   "
              f"(样本 {len(ppt)}/{len(bat)})", flush=True)
        time.sleep(3)    # 让摄像头/NPU 回到空闲

    def agg(name):
        d = res[name]
        return (statistics.mean(d["ppt"]) if d["ppt"] else None,
                statistics.mean(d["bat"]) if d["bat"] else None)

    idle_p, idle_b = agg("idle")
    print("\n汇总 (整机):")
    print(f"  {'空闲':8} PPT {idle_p:6.2f} W" if idle_p else "  空闲     PPT 无读数")
    for name in _order:
        p, b = agg(name)
        if p is None:
            continue
        extra = p - idle_p if idle_p else 0.0
        rel = ""
        if "cpu" in res and name != "cpu":
            base = (agg("cpu")[0] or p) - (idle_p or 0)
            if base > 0:
                rel = f"  (比 CPU {(extra / base - 1) * 100:+.0f}%)"
        print(f"  {name:8} PPT {p:6.2f} W   比空闲多 {extra:5.2f} W{rel}")
    # 电池/PPT 一致性检查
    for name in dict.fromkeys(PHASES):
        p, b = agg(name)
        if p and b and b > 0 and abs(p - b) > 2.5 and b < 0.5 * p:
            print(f"  ⚠ {name}: 电池 {b:.2f} W 与 PPT {p:.2f} W 差太多, 电池通道不可信 "
                  f"(这台的 current_now 量纲有问题)", file=sys.stderr)


if __name__ == "__main__":
    main()
