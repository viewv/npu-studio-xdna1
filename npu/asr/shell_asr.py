"""极简 Shell 语音输入前端 (NPU 16核 40层 ASR 交互测试)。

用法:
  1. 交互式麦克风录音测试 (按回车开始录音, 再按回车结束并识别, 输入 q 回车退出):
     cd /home/viewv/develop/npu-studio-effects && source npu/env.sh
     .venv-iron/bin/python -m npu.asr.shell_asr

  2. 直接测试本地 WAV 文件:
     .venv-iron/bin/python -m npu.asr.shell_asr --wav /path/to/audio.wav
"""
from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
import numpy as np

from npu.asr.encoder_npu import D, NPZ, MODEL, NPUTransformer40, fbank_window, load_wav_window, SAMPLES_DIR
from npu.asr.encoder_16c import NPUTransformer40_16c
from npu.asr.reference import text


def _load_pcm_16k(path: Path) -> np.ndarray:
    """读取 WAV 并自动转成 16kHz 单声道 int16 PCM。"""
    with wave.open(str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sr == 16000 and ch == 1 and sw == 2:
        return np.frombuffer(raw, np.int16)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(f"WAV 格式为 {sr}Hz/{ch}ch/{sw*8}bit，且未找到 ffmpeg 进行重采样")
    res = subprocess.run(
        [ffmpeg, "-y", "-i", str(path), "-ar", "16000", "-ac", "1", "-f", "s16le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(res.stdout, np.int16)


def _record_cmd(wav_path: Path) -> list[str]:
    if shutil.which("pw-record"):
        return ["pw-record", "--rate=16000", "--channels=1", "--format=s16", str(wav_path)]
    if shutil.which("arecord"):
        return ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav", str(wav_path)]
    raise RuntimeError("未找到 pw-record 或 arecord，无法录音")


def recognize_pcm(runner: NPUTransformer40_16c, vocab: list[str], pcm: np.ndarray) -> tuple[str, float, float]:
    t0 = time.perf_counter()
    win, vt = fbank_window(pcm)
    t_fe = time.perf_counter()
    logits = runner.forward_window(win, vt)
    got = text(logits.argmax(1).astype(np.uint32), vocab)
    t_end = time.perf_counter()
    return got, (t_end - t_fe) * 1e3, (t_end - t0) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser(description="NPU ASR 极简 Shell 前端")
    ap.add_argument("--wav", type=Path, help="直接识别指定 WAV 文件并退出")
    ap.add_argument("--16c", dest="use_16c", action="store_true", help="使用 16核实验执行器 (默认使用稳定 4核)")
    ap.add_argument("--cpp", action="store_true", help="使用 C++ 四档常驻后端 (~/.npu/asr/asr_npu_cpp, 0.7s 启动, 137-320ms)")
    ap.add_argument("--cpu-daemon", action="store_true", help="CPU 参考模式常驻守护 (stdin PCM 协议)")
    args = ap.parse_args()

    if args.cpu_daemon:
        import sys
        from npu.asr.reference import fbank as cpu_fbank, forward as cpu_forward
        z = dict(np.load(NPZ))
        vocab = [l.split()[1] if len(l.split()) >= 2 else "" for l in (MODEL / "dict.decoder.utf8.txt").read_text().splitlines()]
        print("READY", flush=True)
        while True:
            hdr = sys.stdin.buffer.readline()
            if not hdr:
                break
            line = hdr.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if line in ("quit", "exit"):
                break
            if line.startswith("PCM "):
                n_samples = int(line[4:])
                raw = sys.stdin.buffer.read(n_samples * 2)
                if len(raw) < n_samples * 2:
                    break
                pcm = np.frombuffer(raw, dtype=np.int16)
            else:
                pcm = _load_pcm_16k(Path(line))
            t0 = time.perf_counter()
            frames = cpu_fbank(pcm)
            ids = cpu_forward(frames, z, int_path=False)
            got = text(ids.astype(np.uint32), vocab)
            ms = (time.perf_counter() - t0) * 1e3
            print(f"OK W=CPU {ms:.1f}ms | {got}", flush=True)
        return 0

    if args.cpp:
        cpp_bin = Path.home() / ".npu" / "asr" / "asr_npu_cpp"
        if not cpp_bin.exists():
            raise RuntimeError(f"未找到 C++ 后端: {cpp_bin}")
        print("[shell-asr] 正在启动 C++ 四档常驻后端...", flush=True)
        cpp = subprocess.Popen(
            [str(cpp_bin)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        assert cpp.stdout is not None and cpp.stdin is not None
        while True:
            line = cpp.stdout.readline()
            if not line:
                raise RuntimeError("C++ 后端启动失败")
            line = line.strip()
            if line == "READY":
                break
            print(line, flush=True)
        if args.wav:
            cpp.stdin.write(f"{args.wav}\nquit\n")
            cpp.stdin.flush()
            print(cpp.stdout.readline().strip())
            cpp.wait()
            return 0
        print("\n=== NPU C++ 四档语音输入测试 (16kHz, 2/3/5/7s 自动分档) ===")
        print("操作: 直接按 [Enter] 开始录音 → 再按 [Enter] 停止并识别 | 输入 q 回车退出\n")
        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "rec.wav"
            while True:
                try:
                    cmd = input("▶ 按 [Enter] 开始录音 (q 退出): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if cmd == "q":
                    break
                proc = subprocess.Popen(_record_cmd(wav_path), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                t_rec = time.perf_counter()
                try:
                    input("🎙  录音中... 按 [Enter] 结束录音: ")
                except (EOFError, KeyboardInterrupt):
                    print()
                finally:
                    proc.send_signal(signal.SIGINT)
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                dur = time.perf_counter() - t_rec
                if not wav_path.exists() or wav_path.stat().st_size < 1000:
                    print("⚠️  录音过短或失败，请重试。\n")
                    continue
                cpp.stdin.write(f"{wav_path}\n")
                cpp.stdin.flush()
                resp = cpp.stdout.readline().strip()
                print(f"📝 [按键 {dur:.2f}s] {resp}\n")
        cpp.stdin.write("quit\n")
        cpp.stdin.flush()
        cpp.wait()
        return 0

    mode = "16核" if args.use_16c else "4核"
    print(f"[shell-asr] 正在加载缓存权重与预热 {mode} NPU...", flush=True)
    t_init = time.perf_counter()
    z = dict(np.load(NPZ))
    vocab = [l.split()[1] if len(l.split()) >= 2 else "" for l in (MODEL / "dict.decoder.utf8.txt").read_text().splitlines()]
    runner = NPUTransformer40_16c.from_cache(z) if args.use_16c else NPUTransformer40.from_cache(z)
    # 预热一次 XCLBIN 编译与加载 (用真实校准窗)
    runner.forward_window(*load_wav_window(SAMPLES_DIR / "funasr_asr_example.wav"))
    print(f"[shell-asr] 就绪! (初始化+预热 {time.perf_counter() - t_init:.1f}s)", flush=True)

    if args.wav:
        pcm = _load_pcm_16k(args.wav)
        got, npu_ms, tot_ms = recognize_pcm(runner, vocab, pcm)
        print(f"音频时长: {len(pcm)/16000:.2f}s | NPU+后端: {npu_ms:.1f}ms | 端到端(含FBank): {tot_ms:.1f}ms")
        print(f"识别结果: {got}")
        return 0

    print("\n=== NPU 语音输入测试 (16kHz, 最长 7.0s/句) ===")
    print("操作: 直接按 [Enter] 开始录音 → 再按 [Enter] 停止并识别 | 输入 q 回车退出\n")
    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "rec.wav"
        while True:
            try:
                cmd = input("▶ 按 [Enter] 开始录音 (q 退出): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if cmd == "q":
                break
            rec_cmd = _record_cmd(wav_path)
            proc = subprocess.Popen(rec_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            t_rec = time.perf_counter()
            try:
                input("🎙  录音中... 按 [Enter] 结束录音: ")
            except (EOFError, KeyboardInterrupt):
                print()
            finally:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            dur = time.perf_counter() - t_rec
            if not wav_path.exists() or wav_path.stat().st_size < 1000:
                print("⚠️  录音过短或失败，请重试。\n")
                continue
            pcm = _load_pcm_16k(wav_path)
            got, npu_ms, tot_ms = recognize_pcm(runner, vocab, pcm)
            print(f"📝 [{len(pcm)/16000:.2f}s (按键 {dur:.2f}s) | NPU {npu_ms:.1f}ms | 端到端 {tot_ms:.1f}ms]")
            print(f"   {got}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
