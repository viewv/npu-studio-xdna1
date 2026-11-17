"""OCR NPU 极速冷启动执行器: 直接通过 pyxrt 加载预导出的 final.xclbin + insts.bin，跳过 Python MLIR-AIE 构图。

若 ~/.npu/ocr/xclbins/ 下尚无对应 xclbin，会自动通过 chain_c16.conv_c16 编译一次并导出保存，
后续启动直接走 pyxrt (< 1s)。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyxrt

from .. import chain_c16 as c

XCLBIN_DIR = Path.home() / ".npu" / "ocr" / "xclbins"
_SHARED_DEV: pyxrt.device | None = None


def _get_device() -> pyxrt.device:
    global _SHARED_DEV
    if _SHARED_DEV is None:
        _SHARED_DEV = pyxrt.device(0)
    return _SHARED_DEV


def _part_key(spec: str, k0: int, k1: int) -> str:
    return hashlib.sha256(f"{spec}|{k0},{k1}|reps=1".encode()).hexdigest()[:16]


def _ensure_xclbin(layers, params, spec: str, k0: int, k1: int, tag: str) -> tuple[Path, Path]:
    XCLBIN_DIR.mkdir(parents=True, exist_ok=True)
    key = _part_key(spec, k0, k1)
    xcl_dst = XCLBIN_DIR / f"{tag}_{k0}_{k1}_{key}.xclbin"
    ins_dst = XCLBIN_DIR / f"{tag}_{k0}_{k1}_{key}.insts.bin"
    if xcl_dst.exists() and ins_dst.exists():
        return xcl_dst, ins_dst

    # 首次导出: 在独立子进程 + 临时缓存目录中用 chain_c16 编译一次 (确保 NPU_CACHE_HOME 在 import aie 之前生效)
    import subprocess
    import sys

    tmp_cache = tempfile.mkdtemp(prefix=f"ocr_xcl_{tag}_")
    env = os.environ.copy()
    env["NPU_CACHE_HOME"] = tmp_cache
    code = (
        "import pickle; from npu import chain_c16 as c; "
        "layers, params = pickle.load(open(r'" + tmp_cache + "/in.pkl', 'rb')); "
        f"ch = c.C16Chain(layers, params, reps=1, parts=[({k0}, {k1})]); "
        f"c.conv_c16(ch.t_act, ch.t_w, ch.t_s, ch.t_out, spec=ch.spec, reps=1, part='{k0},{k1}')"
    )
    try:
        import pickle
        with open(Path(tmp_cache) / "in.pkl", "wb") as f:
            pickle.dump((layers, params), f, protocol=pickle.HIGHEST_PROTOCOL)
        subprocess.run([sys.executable, "-c", code], env=env, check=True)
        xcls = list(Path(tmp_cache).rglob("final.xclbin"))
        assert len(xcls) == 1, f"期望 1 个 final.xclbin, 实际 {xcls}"
        src_dir = xcls[0].parent
        shutil.copy2(src_dir / "final.xclbin", xcl_dst)
        shutil.copy2(src_dir / "insts.bin", ins_dst)
    finally:
        shutil.rmtree(tmp_cache, ignore_errors=True)
    return xcl_dst, ins_dst


class FastXrtChain:
    """与 C16Chain 等价的轻量 pyxrt 运行时 (无 MLIR 构图开销)。"""

    def __init__(self, layers: list[c.Layer], params: list[c.LParam],
                 parts: list[tuple[int, int]] | None = None, tag: str = "ocr"):
        self.layers = layers
        self.P = P = c.Plan(layers)
        self.spec = c.spec_str(layers)
        self.parts = parts or [(0, len(layers))]
        self.dev = _get_device()

        self._runners = []
        k0_ref = None
        for k0, k1 in self.parts:
            xcl_p, ins_p = _ensure_xclbin(layers, params, self.spec, k0, k1, tag)
            xclbin = pyxrt.xclbin(str(xcl_p))
            uuid = self.dev.register_xclbin(xclbin)
            ctx = pyxrt.hw_context(self.dev, uuid)
            kern = pyxrt.kernel(ctx, "MLIR_AIE")
            insts = np.fromfile(ins_p, dtype=np.uint32)
            bo_ins = pyxrt.bo(self.dev, insts.nbytes, pyxrt.bo.flags.cacheable, kern.group_id(1))
            bo_ins.write(insts.view(np.uint8), 0)
            bo_ins.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._runners.append((ctx, kern, bo_ins, len(insts)))
            if k0_ref is None:
                k0_ref = kern

        nw = P.n_steps * c.N_CORES * P.e_w
        flags = pyxrt.bo.flags.host_only
        self.bo_act = pyxrt.bo(self.dev, max(P.size["act"], 64), flags, k0_ref.group_id(3))
        self.bo_w = pyxrt.bo(self.dev, max(nw, 64), flags, k0_ref.group_id(4))
        self.bo_s = pyxrt.bo(self.dev, max(P.size["s"], 64), flags, k0_ref.group_id(5))
        self.bo_out = pyxrt.bo(self.dev, max(P.size["out"], 64), flags, k0_ref.group_id(6))

        self._act_mv = np.asarray(self.bo_act.map())
        self._w_mv = np.asarray(self.bo_w.map()).view(np.int8)
        self._s_mv = np.asarray(self.bo_s.map())
        self._out_mv = np.asarray(self.bo_out.map())

        self._act_mv.fill(0)
        self._s_mv.fill(0)
        self._out_mv.fill(0)
        self.load_weights(params)

    def load_weights(self, params: list[c.LParam]):
        P = self.P
        nw = P.n_steps * c.N_CORES * P.e_w
        for k, (l, p) in enumerate(zip(self.layers, params)):
            if l.op == 3 and p.inp is not None:
                self.write_tensor(k + 1, p.inp, sync=False)
        steps = [
            b for k, (l, tl, p) in enumerate(zip(self.layers, P.tilings, params))
            for b in c.pack_layer(
                l, tl, p, P.e_w, P.tensors[P.inp[k]].ws, c.signed_in(self.layers, k),
                P.tensors[k + 1 + l.sc].c // 16 if l.sc else 0
            )
        ]
        w_arr = np.concatenate(steps)
        assert w_arr.size == nw, f"w_arr size {w_arr.size} != {nw}"
        self._w_mv[:nw] = w_arr
        self.bo_w.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.bo_act.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.bo_s.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def write_tensor(self, i: int, x: np.ndarray, sync: bool = True):
        P, t = self.P, self.P.tensors[i]
        buf, off = P.loc[i]
        assert buf == "act", f"张量 {i} 不在 act 缓冲区"
        ctot = t.ctot or t.c
        root = off - t.g0 * t.cplane
        full = self._act_mv[root: root + (t.h + 2 * c.BR) * (ctot // 16) * t.cplane].reshape(
            t.h + 2 * c.BR, ctot // 16, -1, 16
        )
        g = np.zeros((t.c, t.h, t.ws), np.uint8)
        g[:, :, :t.w] = x.view(np.uint8)[:, :t.h, :t.w]
        full[c.BR:c.BR + t.h, t.g0:t.g0 + t.c // 16, c.PAD:c.PAD + t.ws] = (
            g.reshape(t.c // 16, 16, t.h, t.ws).transpose(2, 0, 3, 1)
        )
        if sync:
            self.bo_act.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def set_input(self, x: np.ndarray):
        self.write_tensor(0, x)

    def call_part(self, p_idx: int = 0):
        _, kern, bo_ins, n_insts = self._runners[p_idx]
        run = kern(3, bo_ins, n_insts, self.bo_act, self.bo_w, self.bo_s, self.bo_out)
        run.wait()
        self.bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def read_tensor(self, i: int) -> np.ndarray:
        P, t = self.P, self.P.tensors[i]
        where, off = P.loc[i]
        if where == "s":
            self.bo_s.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            buf = self._s_mv.view(np.int8)
        elif where == "act":
            buf = self._act_mv.view(np.int8)
        else:
            buf = self._out_mv.view(np.int8)
        root = off - t.g0 * t.cplane
        ctot = t.ctot or t.c
        full = buf[root: root + (t.h + 2 * c.BR) * (ctot // 16) * t.cplane].reshape(
            t.h + 2 * c.BR, ctot // 16, -1, 16
        )
        g = full[c.BR:c.BR + t.h, t.g0:t.g0 + t.c // 16, c.PAD:c.PAD + t.w]
        return g.transpose(1, 3, 0, 2).reshape(t.c, t.h, t.w)


def _ensure_16c_xclbin(layers, params, spec: str, tag: str) -> tuple[Path, Path]:
    XCLBIN_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{spec}|16c_v1".encode()).hexdigest()[:16]
    xcl_dst = XCLBIN_DIR / f"{tag}_16c_{key}.xclbin"
    ins_dst = XCLBIN_DIR / f"{tag}_16c_{key}.insts.bin"
    if xcl_dst.exists() and ins_dst.exists():
        return xcl_dst, ins_dst

    import pickle
    import subprocess
    import sys

    tmp_cache = tempfile.mkdtemp(prefix=f"ocr_xcl16_{tag}_")
    env = os.environ.copy()
    env["NPU_CACHE_HOME"] = tmp_cache
    code = (
        "import pickle, numpy as np, aie.iron as iron; "
        "from npu import chain_c16 as c; "
        "from npu.ocr import v6_rec_16c; "
        "layers, params = pickle.load(open(r'" + tmp_cache + "/in.pkl', 'rb')); "
        "P, _, _, w_packed = v6_rec_16c.build_v6_rec_16c_schedule(layers, params); "
        "t_act = iron.zeros(P.size['act'], dtype=np.int8, device='npu'); "
        "t_w = iron.tensor(w_packed, dtype=np.int8, device='npu'); "
        "t_s = iron.zeros(P.size['s'], dtype=np.int8, device='npu'); "
        "t_out = iron.zeros(P.size['out'], dtype=np.uint8, device='npu'); "
        "v6_rec_16c.v6_rec_16c_kernel(t_act, t_w, t_s, t_out, spec=c.spec_str(layers))"
    )
    try:
        with open(Path(tmp_cache) / "in.pkl", "wb") as f:
            pickle.dump((layers, params), f, protocol=pickle.HIGHEST_PROTOCOL)
        subprocess.run([sys.executable, "-c", code], env=env, check=True)
        xcls = list(Path(tmp_cache).rglob("final.xclbin"))
        assert len(xcls) == 1, f"期望 1 个 final.xclbin, 实际 {xcls}"
        src_dir = xcls[0].parent
        shutil.copy2(src_dir / "final.xclbin", xcl_dst)
        shutil.copy2(src_dir / "insts.bin", ins_dst)
    finally:
        shutil.rmtree(tmp_cache, ignore_errors=True)
    return xcl_dst, ins_dst


class Fast16cChain:
    """16 核 (4×4) 链的轻量 pyxrt 运行时 (单 XCLBIN，无 MLIR 构图开销)。"""

    def __init__(self, layers: list[c.Layer], params: list[c.LParam], tag: str = "ocr16"):
        from .v6_rec_16c import build_v6_rec_16c_schedule

        self.layers = layers
        self.spec = c.spec_str(layers)
        self.P, _, self.n_col_steps, w_packed = build_v6_rec_16c_schedule(layers, params)
        P = self.P
        self.dev = _get_device()

        xcl_p, ins_p = _ensure_16c_xclbin(layers, params, self.spec, tag)
        xclbin = pyxrt.xclbin(str(xcl_p))
        uuid = self.dev.register_xclbin(xclbin)
        self._ctx = pyxrt.hw_context(self.dev, uuid)
        self._kern = pyxrt.kernel(self._ctx, "MLIR_AIE")
        insts = np.fromfile(ins_p, dtype=np.uint32)
        self._bo_ins = pyxrt.bo(self.dev, insts.nbytes, pyxrt.bo.flags.cacheable, self._kern.group_id(1))
        self._bo_ins.write(insts.view(np.uint8), 0)
        self._bo_ins.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self._n_insts = len(insts)

        flags = pyxrt.bo.flags.host_only
        self.bo_act = pyxrt.bo(self.dev, max(P.size["act"], 64), flags, self._kern.group_id(3))
        self.bo_w = pyxrt.bo(self.dev, max(w_packed.size, 64), flags, self._kern.group_id(4))
        self.bo_s = pyxrt.bo(self.dev, max(P.size["s"], 64), flags, self._kern.group_id(5))
        self.bo_out = pyxrt.bo(self.dev, max(P.size["out"], 64), flags, self._kern.group_id(6))

        self._act_mv = np.asarray(self.bo_act.map())
        self._w_mv = np.asarray(self.bo_w.map()).view(np.int8)
        self._s_mv = np.asarray(self.bo_s.map())
        self._out_mv = np.asarray(self.bo_out.map())

        self._act_mv.fill(0)
        self._s_mv.fill(0)
        self._out_mv.fill(0)
        self._load_packed(params, w_packed)

    def _load_packed(self, params: list[c.LParam], w_packed: np.ndarray):
        for k, (l, p) in enumerate(zip(self.layers, params)):
            if l.op == 3 and p.inp is not None:
                self.write_tensor(k + 1, p.inp, sync=False)
        nw = w_packed.size
        self._w_mv[:nw] = w_packed
        self.bo_w.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.bo_act.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.bo_s.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def write_tensor(self, i: int, x: np.ndarray, sync: bool = True):
        P, t = self.P, self.P.tensors[i]
        buf, off = P.loc[i]
        assert buf == "act", f"张量 {i} 不在 act 缓冲区"
        ctot = t.ctot or t.c
        root = off - t.g0 * t.cplane
        full = self._act_mv[root: root + (t.h + 2 * c.BR) * (ctot // 16) * t.cplane].reshape(
            t.h + 2 * c.BR, ctot // 16, -1, 16
        )
        g = np.zeros((t.c, t.h, t.ws), np.uint8)
        g[:, :, :t.w] = x.view(np.uint8)[:, :t.h, :t.w]
        full[c.BR:c.BR + t.h, t.g0:t.g0 + t.c // 16, c.PAD:c.PAD + t.ws] = (
            g.reshape(t.c // 16, 16, t.h, t.ws).transpose(2, 0, 3, 1)
        )
        if sync:
            self.bo_act.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def call_part(self, p_idx: int = 0):
        run = self._kern(3, self._bo_ins, self._n_insts, self.bo_act, self.bo_w, self.bo_s, self.bo_out)
        run.wait()
        self.bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def read_tensor(self, i: int) -> np.ndarray:
        P, t = self.P, self.P.tensors[i]
        where, off = P.loc[i]
        if where == "s":
            self.bo_s.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            buf = self._s_mv.view(np.int8)
        elif where == "act":
            buf = self._act_mv.view(np.int8)
        else:
            buf = self._out_mv.view(np.int8)
        root = off - t.g0 * t.cplane
        ctot = t.ctot or t.c
        full = buf[root: root + (t.h + 2 * c.BR) * (ctot // 16) * t.cplane].reshape(
            t.h + 2 * c.BR, ctot // 16, -1, 16
        )
        g = full[c.BR:c.BR + t.h, t.g0:t.g0 + t.c // 16, c.PAD:c.PAD + t.w]
        return g.transpose(1, 3, 0, 2).reshape(t.c, t.h, t.w)
