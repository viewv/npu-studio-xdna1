"""常驻 OCR 推理子进程 (stdin/stdout JSON 行协议), 供 NPU Studio 桌面端调用:

  source npu/env.sh && .venv-iron/bin/python -u -m npu.ocr.worker

协议:
  请求: {"cmd": "ocr", "path": "/tmp/snip.png", "backend": "npu"|"cpu"}
  响应: {"type": "ready"}
       {"type": "result", "text": "...", "boxes": N, "det_ms": 48.2, "rec_ms": 62.5, "backend": "npu"}
       {"type": "error", "message": "..."}
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np
import onnx
from onnx import numpy_helper as nh
import onnxruntime as ort

from .. import chain_c16 as c
from . import calibrate, det_npu, rec_npu, v6_det_npu, v6_rec_npu
from .fast_xrt import Fast16cChain, FastXrtChain


BUNDLE_PKL = os.path.join(calibrate.CALIB_DIR, "ocr_npu_bundle.pkl")
V6_BUNDLE_PKL = os.path.join(calibrate.CALIB_DIR, "ocr_v6_bundle.pkl")


def _emit(obj: dict):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _build_bundle():
    d_ops, _ = det_npu.load_det_backbone()
    cut = next(i for i, o in enumerate(d_ops) if o.out == "conv2d_469.tmp_0")
    ops_bb = d_ops[:cut]
    d_amax, d_accmax = calibrate.load_calib(calibrate.DET_CALIB_NPZ)
    qd = det_npu.QDet.__new__(det_npu.QDet)
    qd.ops = ops_bb
    qd.byname = {o.out: o for o in ops_bb}
    qd.users = {}
    for o in ops_bb:
        for inp in o.ins:
            qd.users.setdefault(inp, []).append(o)
    qd.amax, qd.accmax, qd.params, qd.export, qd.frozen = d_amax, d_accmax, {}, {}, True
    dummy_d = np.zeros((3, 480, 640), np.float32)
    fl_d = det_npu.run_float(ops_bb, dummy_d)
    shapes_d = {k: v.shape for k, v in fl_d.items()}
    qd.run(dummy_d)

    stage_names = ["p2o.Add.43", "p2o.Add.71", "p2o.Add.147", "p2o.Add.215"]
    m_raw = onnx.load(det_npu.DET_ONNX)
    consts_raw = {i.name: nh.to_array(i).astype(np.float32) for i in m_raw.graph.initializer}
    for n in m_raw.graph.node:
        if n.op_type == "Constant":
            consts_raw[n.output[0]] = nh.to_array(n.attribute[0].t).astype(np.float32)
    stage_map = {}
    for sn in stage_names:
        if sn in fl_d:
            stage_map[sn] = (sn, 1.0, 0.0)
        else:
            add_n = next(n for n in m_raw.graph.node if sn in n.output)
            mul_out = [i for i in add_n.input if i not in consts_raw][0]
            b_nm = [i for i in add_n.input if i in consts_raw][0]
            mul_n = next(n for n in m_raw.graph.node if mul_out in n.output)
            h_out = [i for i in mul_n.input if i not in consts_raw][0]
            w_nm = [i for i in mul_n.input if i in consts_raw][0]
            stage_map[sn] = (h_out, float(consts_raw[w_nm].reshape(-1)[0]), float(consts_raw[b_nm].reshape(-1)[0]))

    tid_d, chans_d = {"x": 0}, {"x": 3}
    d_layers, d_params = [], []
    for op in ops_bb:
        k = len(d_layers)
        if op.kind == "gap":
            src = op.ins[0]
            ch, hh, ww = shapes_d[src]
            d_layers.append(c.Layer(det_npu.pad16(ch), det_npu.pad16(ch), 1, 1, hh, ww, op=1, src=tid_d[src] - (k + 1)))
            d_params.append(c.LParam(None))
            tid_d[op.out], chans_d[op.out] = k + 1, ch
        elif op.kind == "conv":
            ex = qd.export[op.name]
            src, se = ex["src"], ex["se"]
            ci, (hh, ww) = chans_d[src], shapes_d[src][1:]
            co, kk = op.W.shape[0], op.W.shape[2]
            dw = op.g > 1
            ica, oc = det_npu.pad16(ci), det_npu.pad16(co)
            d_layers.append(c.Layer(ica, oc, kk, op.s, hh, ww, dw=int(dw), d=op.d, act=det_npu.ACT[ex["act"]],
                                    src=tid_d[src] - (k + 1), sc=tid_d[se] - (k + 1) if se else 0))
            W = np.zeros((oc, 1 if dw else ica, kk, kk), np.int8)
            W[:co, :ex["W"].shape[1]] = ex["W"]
            s1, M, s2 = qd.params[op.name]
            p = c.LParam(W, s1=s1, s2=s2, bias=det_npu._pad(ex["b"], oc), M=det_npu._pad(M, oc))
            if ex["act"] == "hswish":
                t1, M2, t2 = qd.params[op.name + ".hs"]
                q = ex["q"]
                p.q3, p.q6, p.M2, p.t1, p.t2 = det_npu._pad(3 * q, oc), det_npu._pad(6 * q, oc), det_npu._pad(M2, oc), t1, t2
            d_params.append(p)
            tid_d[ex["out"]], chans_d[ex["out"]] = k + 1, co
            tid_d[op.out] = k + 1

    stage_info = []
    for sn in stage_names:
        base_nm, wa, ba = stage_map[sn]
        sc = qd.scale(base_nm).reshape(-1, 1, 1).astype(np.float32)
        stage_info.append((sn, tid_d[base_nm], sc, wa, ba))
    det_parts = [(0, stage_info[0][1]), (stage_info[0][1], stage_info[1][1]),
                 (stage_info[1][1], stage_info[2][1]), (stage_info[2][1], len(d_layers))]
    det_bundle = (d_layers, d_params, qd.scale("x").reshape(-1, 1, 1), stage_info, det_parts)

    r_ops, r_tail, (w_tail, b_tail), chars = rec_npu.load_rec_backbone()
    r_amax, r_accmax = calibrate.load_calib(calibrate.REC_CALIB_NPZ)
    qr = rec_npu.QRec.__new__(rec_npu.QRec)
    qr.ops = r_ops
    qr.byname = {o.out: o for o in r_ops}
    qr.users = {}
    for o in r_ops:
        for inp in o.ins:
            qr.users.setdefault(inp, []).append(o)
    qr.amax, qr.accmax, qr.params, qr.export, qr.frozen = r_amax, r_accmax, {}, {}, True
    dummy_r = np.zeros((3, 48, 320), np.float32)
    shapes_r = {k: v.shape for k, v in rec_npu.run_float(r_ops, dummy_r).items()}
    qr.run(dummy_r)
    r_layers, r_params, r_tid = rec_npu.build_chain(qr, shapes_r)
    rec_bundle = (
        r_layers, r_params, qr.scale("x").reshape(-1, 1, 1),
        r_tid[r_tail], qr.scale(r_tail).reshape(-1, 1, 1).astype(np.float32),
        w_tail, b_tail, chars,
    )

    with open(BUNDLE_PKL, "wb") as f:
        pickle.dump((det_bundle, rec_bundle), f, protocol=pickle.HIGHEST_PROTOCOL)
    return det_bundle, rec_bundle


def _build_v6_bundle():
    d_ops = v6_det_npu.load_v6_det_backbone()
    d_rng = calibrate.load_v6_calib(calibrate.V6_DET_CALIB_NPZ)
    qd = v6_det_npu.QV6Det.__new__(v6_det_npu.QV6Det)
    qd.ops, qd.in_name = d_ops, "p2o.pd_op.relu.3.0"
    qd.ranges, qd.luts, qd.params, qd.export, qd.frozen = d_rng, {}, {}, {}, False
    qd.users = {}
    for op in d_ops:
        for inp in op.ins:
            qd.users.setdefault(inp, []).append(op)
    dummy_d = np.zeros((64, 120, 160), np.float32)
    shapes_d = {k: v.shape for k, v in v6_det_npu.run_float_v6(d_ops, dummy_d).items()}
    qd.run(dummy_d)
    qd.frozen = True
    qd.run(dummy_d)
    d_layers, d_params, d_parts, d_stage_tid = v6_det_npu.build_v6_det_chain(qd, d_ops, shapes_d)
    d_stage_info = [
        (nm, d_stage_tid[nm], qd.last_int[nm][1].reshape(-1, 1, 1).astype(np.float32))
        for nm in v6_det_npu.STAGE_PROJ_OUTS
    ]
    v6_det_bundle = (d_layers, d_params, qd.scale(qd.in_name).reshape(-1, 1, 1).astype(np.float32), d_parts, d_stage_info)

    r_ops = v6_rec_npu.load_v6_rec_backbone()
    r_rng = calibrate.load_v6_calib(calibrate.V6_REC_CALIB_NPZ)
    qr = v6_rec_npu.QV6Rec.__new__(v6_rec_npu.QV6Rec)
    qr.ops, qr.in_name = r_ops, v6_rec_npu.REC_STEM_OUT
    qr.ranges, qr.luts, qr.params, qr.export, qr.frozen = r_rng, {}, {}, {}, False
    qr.users = {}
    for op in r_ops:
        for inp in op.ins:
            qr.users.setdefault(inp, []).append(op)
    dummy_r = np.zeros((64, 12, 80), np.float32)
    shapes_r = {k: v.shape for k, v in v6_rec_npu.run_float_v6_rec(r_ops, dummy_r).items()}
    qr.run(dummy_r)
    qr.frozen = True
    qr.run(dummy_r)
    r_layers, r_params, r_tid = v6_rec_npu.build_v6_rec_chain(qr, r_ops, shapes_r)
    chars_v6 = v6_rec_npu.load_v6_dict()
    v6_rec_bundle = (
        r_layers, r_params, qr.scale(qr.in_name).reshape(-1, 1, 1).astype(np.float32),
        r_tid[v6_rec_npu.REC_BB_OUT], qr.last_int[v6_rec_npu.REC_BB_OUT][1].reshape(-1, 1, 1).astype(np.float32),
        chars_v6,
    )
    with open(V6_BUNDLE_PKL, "wb") as f:
        pickle.dump((v6_det_bundle, v6_rec_bundle), f, protocol=pickle.HIGHEST_PROTOCOL)
    return v6_det_bundle, v6_rec_bundle


class OcrEngine:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.sess_det_cpu = ort.InferenceSession(det_npu.DET_ONNX, opts, providers=["CPUExecutionProvider"])
        self.sess_rec_cpu = ort.InferenceSession(rec_npu.REC_ONNX, opts, providers=["CPUExecutionProvider"])
        self.sess_fpn = det_npu.build_fpn_head()
        self.sess_rhead = rec_npu.build_rec_head()

        if os.path.exists(BUNDLE_PKL):
            with open(BUNDLE_PKL, "rb") as f:
                self._v4_bundle = pickle.load(f)
        else:
            self._v4_bundle = _build_bundle()

        self.xd16 = np.zeros((16, 480, 640), np.int8)
        self.xr16 = np.zeros((16, 48, 320), np.int8)
        self._v4_loaded = False
        self._v6_loaded = False
        self._ensure_v4()

    def _release_v4(self):
        if self._v4_loaded:
            self.ch_det = None
            self.ch_rec = None
            gc.collect()
            self._v4_loaded = False

    def _release_v6(self):
        if self._v6_loaded:
            self.ch_v6_det = None
            gc.collect()
            self._v6_loaded = False

    def _ensure_v4(self):
        if self._v4_loaded:
            return
        self._release_v6()
        det_bundle, rec_bundle = self._v4_bundle
        d_layers, d_params, self.s_in_d, self.stage_info, self.det_parts = det_bundle
        self.ch_det = FastXrtChain(d_layers, d_params, parts=self.det_parts, tag="v4_det")
        (r_layers, r_params, self.s_in_r, self.r_tail_tid,
         self.s_tail_r, self.w_tail, self.b_tail, self.chars) = rec_bundle
        self.ch_rec = FastXrtChain(r_layers, r_params, tag="v4_rec")
        self._v4_loaded = True

    def _ensure_v6(self):
        if self._v6_loaded:
            return
        self._release_v4()
        if not hasattr(self, "sess_v6_det_cpu"):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            self.sess_v6_det_cpu = ort.InferenceSession(v6_det_npu.V6_DET_ONNX, opts, providers=["CPUExecutionProvider"])
            self.sess_v6_rec_cpu = ort.InferenceSession(v6_rec_npu.V6_REC_ONNX, opts, providers=["CPUExecutionProvider"])
            self.sess_v6_det_stem = v6_det_npu.build_sub_ort(
                v6_det_npu.V6_DET_ONNX, ["x"], ["p2o.pd_op.relu.3.0"], [[1, 3, 480, 640]], [[1, 64, 120, 160]]
            )
            self.sess_v6_det_tail = v6_det_npu.build_sub_ort(
                v6_det_npu.V6_DET_ONNX, v6_det_npu.STAGE_PROJ_OUTS, ["fetch_name_0"],
                [[1, 256, 120, 160], [1, 256, 60, 80], [1, 256, 30, 40], [1, 256, 15, 20]],
                [[1, 1, 480, 640]],
            )

        if not hasattr(self, "_v6_bundle"):
            if os.path.exists(V6_BUNDLE_PKL):
                with open(V6_BUNDLE_PKL, "rb") as f:
                    self._v6_bundle = pickle.load(f)
            else:
                self._v6_bundle = _build_v6_bundle()

        v6_det_bundle, v6_rec_bundle = self._v6_bundle
        d_layers, d_params, self.v6_s_in_d, self.v6_det_parts, self.v6_stage_info = v6_det_bundle
        self.ch_v6_det = Fast16cChain(d_layers, d_params, tag="v6_det")

        (_, _, _, _, _, self.chars_v6) = v6_rec_bundle
        self._v6_loaded = True

    def _det_tile(self, tile_bgr: np.ndarray, backend: str, engine: str = "v4") -> np.ndarray:
        """对单个 <=480x640 图块运行 Det, 返回 (nh, nw) 概率图"""
        th, tw = tile_bgr.shape[:2]
        canvas = np.full((480, 640, 3), 245, np.uint8)
        canvas[:th, :tw] = tile_bgr
        x_det = det_npu.preprocess_det(canvas)
        if engine == "v6":
            self._ensure_v6()
            if backend == "cpu":
                in_nm = self.sess_v6_det_cpu.get_inputs()[0].name
                prob = self.sess_v6_det_cpu.run(None, {in_nm: x_det[None]})[0][0, 0]
            else:
                sf = self.sess_v6_det_stem.run(None, {"x": x_det[None]})[0][0]
                xq = np.clip(np.round(sf / self.v6_s_in_d), 0, 127).astype(np.uint8)
                self.ch_v6_det.write_tensor(1, xq)
                self.ch_v6_det.call_part(0)
                feats = {
                    nm: (self.ch_v6_det.read_tensor(tid_s).astype(np.float32) * sc)[None]
                    for nm, tid_s, sc in self.v6_stage_info
                }
                prob = self.sess_v6_det_tail.run(None, feats)[0][0, 0]
            return prob[:th, :tw]
        if backend == "cpu":
            prob = self.sess_det_cpu.run(None, {"x": x_det[None]})[0][0, 0]
        else:
            self._ensure_v4()
            self.xd16[:3] = np.clip(np.round(x_det / self.s_in_d), -127, 127).astype(np.int8)
            self.ch_det.set_input(self.xd16)
            feats = {}
            for p_idx, (sn, tid_s, sc, wa, ba) in enumerate(self.stage_info):
                self.ch_det.call_part(p_idx)
                arr = self.ch_det.read_tensor(tid_s).astype(np.float32) * sc
                if wa != 1.0 or ba != 0.0:
                    arr = arr * wa + ba
                feats[sn] = arr[None]
            prob = self.sess_fpn.run(None, feats)[0][0, 0]
        return prob[:th, :tw]

    def _rec_chunk(self, bgr_48: np.ndarray, backend: str, engine: str = "v4") -> str:
        """bgr_48: 高 48, 宽 <=320 的裁剪条 -> 识别文本"""
        rw = bgr_48.shape[1]
        x = (bgr_48.astype(np.float32) / 255.0 - 0.5) / 0.5
        xr = np.zeros((3, 48, 320), np.float32)
        xr[:, :, :rw] = x.transpose(2, 0, 1)
        if engine == "v6":
            self._ensure_v6()
            # v6_rec 输入极小且全在 Zen4 缓存内，CPU ORT (24 ms) 快于 NPU 16核 (28 ms)，混合路由直接走 CPU ORT
            in_nm = self.sess_v6_rec_cpu.get_inputs()[0].name
            probs = self.sess_v6_rec_cpu.run(None, {in_nm: xr[None]})[0]
            txt, _ = v6_rec_npu.ctc_decode_v6(probs, self.chars_v6)
            return txt.strip()
        if backend == "cpu":
            probs = self.sess_rec_cpu.run(None, {"x": xr[None]})[0][0]
        else:
            self._ensure_v4()
            self.xr16[:3] = np.clip(np.round(xr / self.s_in_r), -127, 127).astype(np.int8)
            self.ch_rec.set_input(self.xr16)
            self.ch_rec.call_part(0)
            feat = self.ch_rec.read_tensor(self.r_tail_tid).astype(np.float32) * self.s_tail_r
            feat = feat * self.w_tail.reshape(-1, 1, 1) + self.b_tail.reshape(-1, 1, 1)
            probs = self.sess_rhead.run(None, {"p2o.Add.231": feat[None]})[0][0]
        txt, _ = rec_npu.ctc_decode(probs, self.chars)
        return txt.strip()

    def _rec_line(self, crop: np.ndarray, backend: str, engine: str = "v4") -> str:
        """支持超宽文本框: 等比缩放到高 48, 若宽 >320 则在空白列处切分后逐段识别"""
        ih, iw = crop.shape[:2]
        rw_full = max(1, int(np.ceil(48.0 * iw / ih)))
        line_48 = cv2.resize(crop, (rw_full, 48), interpolation=cv2.INTER_LINEAR)
        if rw_full <= 320:
            return self._rec_chunk(line_48, backend, engine)

        # 超宽行: 在 [x0+200 .. x0+318] 找梯度最小的空白列切分, 避免切到字中间
        gray = cv2.cvtColor(line_48, cv2.COLOR_BGR2GRAY).astype(np.float32)
        col_energy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :])).sum(axis=0)
        parts = []
        x0 = 0
        while x0 < rw_full:
            if rw_full - x0 <= 320:
                parts.append(self._rec_chunk(line_48[:, x0:rw_full], backend, engine))
                break
            lo, hi = x0 + 200, x0 + 318
            split_x = lo + int(np.argmin(col_energy[lo:hi]))
            parts.append(self._rec_chunk(line_48[:, x0:split_x], backend, engine))
            x0 = split_x
        return " ".join(p for p in parts if p)

    def run_ocr(self, path: str, backend: str = "npu", engine: str = "v4") -> dict:
        raw = cv2.imread(path)
        if raw is None:
            raise ValueError(f"无法读取图片: {path}")
        ih, iw = raw.shape[:2]
        # 若过大 (>1280 宽或 >960 高) 先等比缩放到 1280x960 以内, 再按 480x640 分块 1:1 检测
        if iw > 1280 or ih > 960:
            sc = min(1280.0 / iw, 960.0 / ih)
            raw = cv2.resize(raw, (max(1, int(iw * sc)), max(1, int(ih * sc))), interpolation=cv2.INTER_AREA)
            ih, iw = raw.shape[:2]

        t0 = time.perf_counter()
        prob_full = np.zeros((ih, iw), np.float32)
        stride_x, stride_y = 576, 416
        for y0 in range(0, ih, stride_y):
            y1 = min(ih, y0 + 480)
            for x0 in range(0, iw, stride_x):
                x1 = min(iw, x0 + 640)
                p = self._det_tile(raw[y0:y1, x0:x1], backend, engine)
                prob_full[y0:y1, x0:x1] = np.maximum(prob_full[y0:y1, x0:x1], p)
                if x1 == iw:
                    break
            if y1 == ih:
                break
        boxes = det_npu.extract_boxes(prob_full)
        det_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        lines = []
        for x0, y0, x1, y1 in boxes:
            crop = raw[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            txt = self._rec_line(crop, backend, engine)
            if txt:
                lines.append(txt)
        rec_ms = (time.perf_counter() - t1) * 1000.0

        return {
            "type": "result",
            "text": "\n".join(lines),
            "boxes": len(boxes),
            "det_ms": round(det_ms, 1),
            "rec_ms": round(rec_ms, 1),
            "backend": backend,
            "engine": engine,
        }


def main():
    _emit({"type": "status", "status": "loading"})
    engine = OcrEngine()
    _emit({"type": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "ocr":
                res = engine.run_ocr(req["path"], req.get("backend", "npu"), req.get("engine", "v4"))
                _emit(res)
        except Exception as e:
            _emit({"type": "error", "message": str(e)})


if __name__ == "__main__":
    main()
