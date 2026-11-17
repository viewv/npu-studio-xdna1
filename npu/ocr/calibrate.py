"""多配色/多字号图文 + 真实 UI 截图联合 INT8 校准 (PTQ), 保存校准表到 ~/.npu/ocr/:

  source npu/env.sh && .venv-iron/bin/python -m npu.ocr.calibrate
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import det_npu, rec_npu

CALIB_DIR = "/home/viewv/.npu/ocr"
DET_CALIB_NPZ = os.path.join(CALIB_DIR, "det_480x640_calib.npz")
REC_CALIB_NPZ = os.path.join(CALIB_DIR, "rec_48x320_calib.npz")
V6_DET_CALIB_NPZ = os.path.join(CALIB_DIR, "v6_det_480x640_calib.npz")
V6_REC_CALIB_NPZ = os.path.join(CALIB_DIR, "v6_rec_48x320_calib.npz")

EXTRA_UI_SCREENSHOT = "/home/viewv/.gemini/antigravity/brain/ebe891b4-07e3-4c4e-8574-e2373d5f2b8d/.user_uploaded/media_1794928363070_edc7782c.png"
FONTS = [p for p in rec_npu.FONT_CANDIDATES if os.path.exists(p)] + [
    "/usr/share/fonts/MapleMono-NF-CN/MapleMono-NF-CN-BoldItalic.ttf",
    "/usr/share/fonts/TTF/LXGWWenKaiScreen.ttf",
]
FONTS = [p for p in FONTS if os.path.exists(p)]


def make_det_calib_images(h: int = 480, w: int = 640) -> list[np.ndarray]:
    """生成覆盖多种背景/按钮配色/字号的 480×640 校准图 + 1 张非测试 UI 截图"""
    rng = np.random.default_rng(42)
    themes = [
        ((248, 249, 250), (20, 20, 20), (124, 58, 237), (255, 255, 255)),   # 浅灰+紫按钮
        ((255, 255, 255), (15, 23, 42), (0, 102, 204), (255, 255, 255)),    # 纯白+蓝按钮
        ((30, 32, 36), (235, 238, 242), (59, 130, 246), (255, 255, 255)),   # 深色模式
        ((240, 242, 245), (55, 65, 81), (22, 163, 74), (255, 255, 255)),    # 卡片灰+绿按钮
    ]
    sample_lines = [
        "NPU Studio · Ryzen AI NPU Toolbox",
        "Start Live ASR · Stop Listening · Clear",
        "Compute Backend: NPU (Recommended) / CPU / GPU",
        "Live Transcript · Idle · Active Bucket 3s",
        "中英文混合 OCR 2026 · 7840HS XDNA1",
        "OpenBLAS float32 · No NPU · ~1.1 s",
        "Video Blur · Speech ASR · 中文 / English",
        "Hardware Accel · Hair-level edges · 30 fps",
    ]
    imgs = []
    for t_idx in range(12):
        bg, fg, btn_bg, btn_fg = themes[t_idx % len(themes)]
        im = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(im)
        font_path = FONTS[t_idx % len(FONTS)] if FONTS else None
        y = 24
        for row in range(6):
            sz = int(rng.choice([14, 18, 22, 28, 34]))
            font = ImageFont.truetype(font_path, sz) if font_path else ImageFont.load_default()
            txt = sample_lines[(t_idx + row) % len(sample_lines)]
            if row % 3 == 1:
                # 模拟彩色圆角按钮反白字
                draw.rounded_rectangle([32, y, 280, y + sz + 18], radius=8, fill=btn_bg)
                draw.text((46, y + 8), txt[:22], fill=btn_fg, font=font)
            else:
                draw.text((32 + int(rng.integers(0, 40)), y), txt, fill=fg, font=font)
            y += sz + 38
            if y > h - 50:
                break
        imgs.append(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))

    if os.path.exists(EXTRA_UI_SCREENSHOT):
        raw = cv2.imread(EXTRA_UI_SCREENSHOT)
        ih, iw = raw.shape[:2]
        sc = min(w / iw, h / ih)
        nw, nh_i = int(iw * sc), int(ih * sc)
        canvas = np.full((h, w, 3), 245, np.uint8)
        canvas[:nh_i, :nw] = cv2.resize(raw, (nw, nh_i), interpolation=cv2.INTER_AREA)
        imgs.append(canvas)
    return imgs


def make_rec_calib_strips() -> list[np.ndarray]:
    """生成覆盖多种背景/字号/语言的 48×320 识别校准条"""
    rng = np.random.default_rng(123)
    styles = [
        ((245, 245, 245), (15, 15, 15)),   # 白底黑字
        ((255, 255, 255), (50, 55, 65)),   # 白底灰字
        ((0, 102, 178), (255, 255, 255)),  # 蓝按钮白字
        ((124, 58, 237), (255, 255, 255)), # 紫按钮白字
        ((235, 238, 243), (30, 35, 45)),   # 卡片灰底黑字
        ((32, 34, 38), (235, 238, 242)),   # 深底白字
    ]
    phrases = [
        "NPU Studio", "Ryzen AI NPU Toolbox", "Start Live ASR", "Live Transcript",
        "ASR Backend", "Video Blur", "Speech ASR", "中文", "English",
        "4-Core Buckets (C++)", "CPU · Python Reference", "OpenBLAS float32",
        "Ryzen 7840HS NPU 测试", "中英文混合 OCR 2026", "XDNA1 神经网络加速",
        "Hello World! 你好世界", "PP-OCRv4 Mobile INT8", "算子库性能与精度验证",
        "Virtual Camera", "Ready /dev/video2", "Compute Backend", "Matting Model",
        "Recommended", "Zero CPU load", "137-320 ms", "Active Bucket",
    ]
    strips = []
    for idx, txt in enumerate(phrases):
        bg, fg = styles[idx % len(styles)]
        font_path = FONTS[idx % len(FONTS)] if FONTS else None
        sz = int(rng.choice([16, 20, 26, 30]))
        font = ImageFont.truetype(font_path, sz) if font_path else ImageFont.load_default()
        tw = max(80, min(310, len(txt) * sz))
        th = max(24, sz + 14)
        im = Image.new("RGB", (tw, th), bg)
        draw = ImageDraw.Draw(im)
        draw.text((6, 5), txt, fill=fg, font=font)
        strips.append(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
    return strips


def save_calib(path: str, amax: dict[str, np.ndarray], accmax: dict[str, int]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {f"amax::{k}": v for k, v in amax.items()}
    payload.update({f"acc::{k}": np.array([v], np.int64) for k, v in accmax.items()})
    np.savez(path, **payload)


def load_calib(path: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    d = np.load(path)
    amax, accmax = {}, {}
    for k in d.files:
        if k.startswith("amax::"):
            amax[k[6:]] = d[k]
        elif k.startswith("acc::"):
            accmax[k[5:]] = int(d[k][0])
    return amax, accmax


def save_v6_calib(path: str, ranges: dict[str, np.ndarray]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{f"rng::{k}": v for k, v in ranges.items()})


def load_v6_calib(path: str) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k[5:]: d[k] for k in d.files if k.startswith("rng::")}


def calibrate_v6():
    from . import v6_det_npu, v6_rec_npu

    t0 = time.perf_counter()
    det_imgs = make_det_calib_images()
    sess_d_stem = v6_det_npu.build_sub_ort(
        v6_det_npu.V6_DET_ONNX, ["x"], ["p2o.pd_op.relu.3.0"], [[1, 3, 480, 640]], [[1, 64, 120, 160]]
    )
    d_ops = v6_det_npu.load_v6_det_backbone()
    d_fl = [v6_det_npu.run_float_v6(d_ops, sess_d_stem.run(None, {"x": det_npu.preprocess_det(im)[None]})[0][0])
            for im in det_imgs]
    qd = v6_det_npu.QV6Det(d_ops, d_fl, pct=99.95)
    save_v6_calib(V6_DET_CALIB_NPZ, qd.ranges)
    print(f"v6 Det 校准完成 ({len(det_imgs)} 图) -> {V6_DET_CALIB_NPZ} ({time.perf_counter()-t0:.1f} s)")

    t1 = time.perf_counter()
    rec_strips = make_rec_calib_strips()
    sess_r_stem = v6_det_npu.build_sub_ort(
        v6_rec_npu.V6_REC_ONNX, ["x"], [v6_rec_npu.REC_STEM_OUT], [[1, 3, 48, 320]], [[1, 64, 12, 80]]
    )
    r_ops = v6_rec_npu.load_v6_rec_backbone()
    r_fl = [v6_rec_npu.run_float_v6_rec(r_ops, sess_r_stem.run(None, {"x": rec_npu.preprocess_rec(st)[None]})[0][0])
            for st in rec_strips]
    qr = v6_rec_npu.QV6Rec(r_ops, r_fl, pct=99.95)
    save_v6_calib(V6_REC_CALIB_NPZ, qr.ranges)
    print(f"v6 Rec 校准完成 ({len(rec_strips)} 图) -> {V6_REC_CALIB_NPZ} ({time.perf_counter()-t1:.1f} s)")


def main():
    import sys
    if "--v6" in sys.argv:
        calibrate_v6()
        return
    t0 = time.perf_counter()
    # 1. 校准 Det 主干 (49 层)
    det_imgs = make_det_calib_images()
    d_ops, _ = det_npu.load_det_backbone()
    cut = next(i for i, o in enumerate(d_ops) if o.out == "conv2d_469.tmp_0")
    ops_bb = d_ops[:cut]
    xs_det = [det_npu.preprocess_det(im) for im in det_imgs]
    calib_det = [det_npu.run_float(ops_bb, x) for x in xs_det]
    qd = det_npu.QDet(ops_bb, calib_det, pct=99.99)
    for x in xs_det:
        qd.run(x)
    save_calib(DET_CALIB_NPZ, qd.amax, qd.accmax)
    print(f"Det 校准完成 ({len(det_imgs)} 图) -> {DET_CALIB_NPZ} ({time.perf_counter()-t0:.1f} s)")

    # 2. 校准 Rec 主干 (50 层)
    t1 = time.perf_counter()
    rec_strips = make_rec_calib_strips()
    r_ops, _, _, _ = rec_npu.load_rec_backbone()
    xs_rec = [rec_npu.preprocess_rec(st) for st in rec_strips]
    calib_rec = [rec_npu.run_float(r_ops, x) for x in xs_rec]
    qr = rec_npu.QRec(r_ops, calib_rec, pct=99.99)
    for x in xs_rec:
        qr.run(x)
    save_calib(REC_CALIB_NPZ, qr.amax, qr.accmax)
    print(f"Rec 校准完成 ({len(rec_strips)} 图) -> {REC_CALIB_NPZ} ({time.perf_counter()-t1:.1f} s)")


if __name__ == "__main__":
    main()
