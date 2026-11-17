"""40 层 ASR Transformer 的 16 核 (4×4 列广播) NPU 执行器。

布局 (Ryzen 7840HS Phoenix XDNA1, 4 列 × 4 计算行 = 16 核):
  - f_w[col]      (col=0..3): 每列 1 路 Shim MM2S 垂直广播给 Tile(col, 2..5)
  - f_in_shim[r]  (r=0..3):   每行 1 路 Shim MM2S 经 MemTile(r, 1) 水平广播给 Tile(0..3, 2+r)
  - f_out_shim[c] (c=0..3):   每列 4 行输出经 MemTile(c, 1) join 成 1 路 Shim S2MM

算子映射:
  - 1×1 线性层 (Q/K/V/Out/FFN1/FFN2): 16 核全开, 4 列切 OC (权重 DMA 降 4×), 4 行切 H (输入广播降 4×)
  - LayerNorm (op=5): col=0 的 4 核处理 4 行, col=1..3 消费广播但不输出
  - Attention (op=6): col=0 (Head 0..3) + col=1 (Head 4..7) 共 8 核 (1 head/核), col=2..3 消费广播不输出
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import pyxrt
import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction, Buffer, CompileTime
from aie.iron.device.tile import Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern

import npu.chain_c16 as c
from npu.asr.encoder_npu import (
    D, HEADS, HD, FF, T_WIN, LAYERS, BLOCKS_PER_CALL, N_CALLS,
    NPZ, MODEL, SAMPLES_DIR, FastFrontend, make_block_layers, calibrate_and_freeze, sim_blocks, load_wav_window,
)
from npu.asr.reference import frontend, text


def _build_16c_schedule(layers: list[c.Layer], params: list[c.LParam]):
    """为 5 块 (45 层) 构造 16 核的 Plan、每列权重序列和主机调度信息。"""
    P = c.Plan(layers)
    col_steps = [[] for _ in range(4)]   # 每列打包好的 uint8[e_w] 列表
    ops = []                             # 每层调度描述 (供 @iron.jit 展开和主机填 tap)
    vt_idx = [[] for _ in range(4)]      # 每列 op=6 sub=0 槽的 hdr[62] (valid_t) int32 偏移

    for k, (l, tl, p) in enumerate(zip(layers, P.tilings, params)):
        if l.op == 3:
            continue
        raw_w = c.pack_layer(l, tl, p, P.e_w, P.tin[k].ws, c.signed_in(layers, k),
                             P.tensors[k + 1 + l.src].c // 16)
        # raw_w 长度 = len(tl.steps) * 4, 按 [step][core] 排 (每 core 一个 e_w B 字节块)
        assert len(raw_w) == len(tl.steps) * c.N_CORES

        if l.op == 0:
            # 1x1: tl.n_ob 为 4 的倍数; spo = 每 ob 的 IC 步数
            assert tl.n_ob % 4 == 0, f"L{k}: n_ob={tl.n_ob} not mult of 4"
            spo = len(tl.steps) // tl.n_ob
            for g in range(tl.n_ob // 4):
                for s_i in range(spo):
                    for col in range(4):
                        ob = g * 4 + col
                        slot = (ob * spo + s_i) * c.N_CORES
                        # 4 行在 1x1 卷积里权重与 header 完全相同 (只复制 core=0 的槽, 再把 pc[0..3] 填齐)
                        w_buf = raw_w[slot].copy()
                        # 确保 4 个 row (core=0..3) 的 per-core u16 表都等于 row 0 的表 (4 行等宽 w=35)
                        w_buf[32:96] = np.tile(w_buf[32:48], 4)
                        col_steps[col].append(w_buf)
            ops.append(("conv1", k, tl.n_ob // 4, spo, tl.steps[0][1]))

        elif l.op == 5:
            # LN: 8 步 (4 步 pass 0 + 4 步 pass 1), col=0 算 4 行, col=1..3 跳过 (nch=0)
            for s_i in range(len(tl.steps)):
                w0 = raw_w[s_i * c.N_CORES].copy()
                w0[32:96] = np.tile(w0[32:48], 4)
                col_steps[0].append(w0)
                w_skip = w0.copy()
                # col 1..3: pc[4] (nch)=0, pc[5] (sel_acc)=1, pc[6] (sel_fin)=0, pc[0] (rows)=1
                for rr in range(4):
                    hc = 32 + 16 * rr
                    w_skip[hc:hc + 16].view(np.uint16)[:7] = np.array([1, 0, 0, 0, 0, 1, 0], dtype=np.uint16)
                for col in range(1, 4):
                    col_steps[col].append(w_skip.copy())
            ops.append(("ln", k, len(tl.steps), tl.steps[0][3]))

        elif l.op == 6:
            # Attn: 8 heads (h=0..7), col = h // 2 负责该 head 的 4 行 (row=0..3), 其余 3 列跳过
            for h_idx in range(HEADS):
                active_col = h_idx // 2
                for sub in range(2):
                    step_idx = h_idx * 2 + sub
                    w_act = raw_w[step_idx * c.N_CORES].copy()
                    w_act[32:96] = np.tile(w_act[32:48], 4)
                    if sub == 0:
                        vt_idx[active_col].append(len(col_steps[active_col]) * (P.e_w // 4) + 62)

                    w_skip = w_act.copy()
                    n_rows = 3 if sub == 0 else 2
                    for rr in range(4):
                        hc = 32 + 16 * rr
                        w_skip[hc:hc + 16].view(np.uint16)[:7] = np.array([n_rows, 0, 0, 0, 0, 1, 0], dtype=np.uint16)

                    for col in range(4):
                        col_steps[col].append(w_act.copy() if col == active_col else w_skip.copy())
            ops.append(("attn", k))

    n_col_steps = len(col_steps[0])
    assert all(len(cs) == n_col_steps for cs in col_steps)
    # 打包成 [n_col_steps, 4_cols, e_w]
    w_packed = np.stack([np.stack(col_steps[col], axis=0) for col in range(4)], axis=1).reshape(-1).copy()
    vt_flat = []
    for col in range(4):
        for idx in vt_idx[col]:
            step_i, word_in_step = divmod(idx, P.e_w // 4)
            vt_flat.append((step_i * 4 + col) * (P.e_w // 4) + word_in_step)
    return P, ops, n_col_steps, w_packed, np.array(vt_flat, dtype=np.int64)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def transformer_16c(act: c.In, wts: c.In, scratch: c.InOut, out: c.Out, *,
                    spec: CompileTime[str], part: CompileTime[str] = ""):
    layers = c.from_spec_str(spec)
    k0, k1 = map(int, part.split(",")) if part else (0, len(layers))
    layers_sub = layers[k0:k1]
    P = c.Plan(layers)
    # 计算每列步数与 op 列表 (仅需形状与模式, 用占位参数即可)
    n_col_steps = 0
    ops = []
    for k in range(k0, k1):
        l, tl = layers[k], P.tilings[k]
        if l.op == 3:
            continue
        if l.op == 0:
            spo = len(tl.steps) // tl.n_ob
            n_col_steps += (tl.n_ob // 4) * spo
            ops.append(("conv1", k, tl.n_ob // 4, spo, tl.steps[0][1]))
        elif l.op == 5:
            n_col_steps += len(tl.steps)
            ops.append(("ln", k, len(tl.steps), tl.steps[0][3]))
        elif l.op == 6:
            n_col_steps += 16
            ops.append(("attn", k))

    i8, u8, i32 = np.dtype[np.int8], np.dtype[np.uint8], np.dtype[np.int32]
    in_ty = np.ndarray[(c.E_IN,), i8]
    out_ty = np.ndarray[(c.CW * 16,), u8]
    out4_ty = np.ndarray[(4 * c.CW * 16,), u8]
    w_ty = np.ndarray[(P.e_w,), i8]
    acc_ty = np.ndarray[(c.ACC_CHUNKS * c.CW * 16,), i32]
    sc_bytes = 6656 if any(l.op == 6 for l in layers_sub) else c.SC_BYTES
    sc_ty = np.ndarray[(sc_bytes,), u8]
    st_sz = 2048 if any(l.op in (5, 6) for l in layers_sub) else 1024

    kern = ExternalFunction(
        "convC16Chunk", source_file=c.SRC,
        arg_types=[in_ty, in_ty, in_ty, w_ty, out_ty, acc_ty, sc_ty, np.int32, np.int32, np.int32],
        include_dirs=[c.KERNEL_INC],
        compile_flags=[f"-DCW={c.CW}", *P.kernel_flags(k0, k1)],
    )

    f_w = [ObjectFifo(w_ty, depth=2, name=f"w_c{col}") for col in range(4)]
    f_in_shim = [ObjectFifo(in_ty, depth=2, name=f"in_shim_r{row}") for row in range(4)]
    f_in = [f_in_shim[row].cons().forward(tile=Tile(row, 1), depth=4, name=f"in_r{row}") for row in range(4)]
    f_out_shim = [ObjectFifo(out4_ty, depth=2, name=f"out_shim_c{col}") for col in range(4)]
    f_out = [
        f_out_shim[col].prod().join(
            offsets=[r * c.CW * 16 for r in range(4)],
            tile=Tile(col, 1),
            depths=[2, 2, 2, 2],
            obj_types=[out_ty, out_ty, out_ty, out_ty],
            names=[f"out_c{col}_r{r}" for r in range(4)],
        )
        for col in range(4)
    ]

    workers = []
    def make_core(row):
        hc = 32 + 16 * row
        def core(fi, fw, fo, k, acc, dummy, sc):
            for _ in range_(n_col_steps):
                wv = fw.acquire(1)
                rows, nch = c._u16(wv, hc), c._u16(wv, hc + 8)
                for _ in range_(c._u16(wv, hc + 10)):
                    for r in range_(rows):
                        win = fi.acquire(1)
                        for j in range_(nch):
                            k(win, win, win, wv, dummy, acc, sc, c._i32(j), c._i32(r), row)
                        fi.release(1)
                for _ in range_(c._u16(wv, hc + 12)):
                    for r in range_(rows):
                        win = fi.acquire(1)
                        for j in range_(nch):
                            o = fo.acquire(1)
                            k(win, win, win, wv, o, acc, sc, c._i32(j), c._i32(r), row)
                            fo.release(1)
                        fi.release(1)
                fw.release(1)
        return core

    for col in range(4):
        for row in range(4):
            acc = Buffer(acc_ty, name=f"acc_{col}_{row}")
            dummy = Buffer(out_ty, name=f"dummy_{col}_{row}")
            sc = Buffer(sc_ty, name=f"sc_{col}_{row}")
            workers.append(Worker(
                make_core(row),
                [f_in[row].cons(), f_w[col].cons(), f_out[col][row].prod(), kern, acc, dummy, sc],
                tile=Tile(col, 2 + row),
                stack_size=st_sz,
            ))

    rt = Runtime()
    nw = n_col_steps * 4 * P.e_w
    tap = lambda n, o_, sz, st: TensorAccessPattern((1, n), o_, sz, st)
    with rt.sequence(np.ndarray[(P.size["act"],), i8], np.ndarray[(nw,), i8],
                     np.ndarray[(P.size["s"],), i8], np.ndarray[(P.size["out"],), u8]) as (a, w, s, o):
        bufs = {"act": a, "s": s, "out": o}
        rt.start(*workers)
        pending = []
        s_cur = 0

        for op_desc in ops:
            kind = op_desc[0]
            k = op_desc[1]
            (src, soff), (dst, doff) = P.io[k]
            tout = P.tensors[k + 1]
            tout_ctot = tout.ctot or tout.c
            out_row_stride = (tout_ctot // 16) * (tout.ws + 2 * c.PAD) * 16
            out_plane_stride = (tout.ws + 2 * c.PAD) * 16

            if kind == "conv1":
                _, _, ng, spo, nocp = op_desc
                for g in range(ng):
                    tg = rt.task_group()
                    pending.append(tg)
                    for col in range(4):
                        rt.fill(f_w[col].prod(), w,
                                tap=tap(nw, (s_cur * 4 + col) * P.e_w, [1, spo, P.e_w // 256, 256], [0, 4 * P.e_w, 256, 1]),
                                task_group=tg)
                    for row in range(4):
                        for o_, sz, st in P.fill_taps(k, soff, row, 0, row, g * 4, g * 4 + 1):
                            rt.fill(f_in_shim[row].prod(), bufs[src], tap=tap(P.size[src], o_, sz, st), task_group=tg)
                    for col in range(4):
                        ob = g * 4 + col
                        off0 = doff + (c.BR * (tout_ctot // 16) + ob * nocp) * out_plane_stride + c.PAD * 16
                        rt.drain(f_out_shim[col].cons(), bufs[dst],
                                 tap=tap(P.size[dst], off0, [1, nocp, 4, c.CW * 16], [0, out_plane_stride, out_row_stride, 1]),
                                 wait=True, task_group=tg)
                    s_cur += spo
                    while pending:
                        rt.finish_task_group(pending.pop(0))

            elif kind == "ln":
                _, _, n_ln, nicp = op_desc
                # 4 步 pass 0 (无 drain), 4 步 pass 1 (有 drain)
                for si in range(n_ln):
                    tg = rt.task_group()
                    pending.append(tg)
                    has_drain = (si >= 4)
                    for col in range(4):
                        rt.fill(f_w[col].prod(), w,
                                tap=tap(nw, (s_cur * 4 + col) * P.e_w, [1, 1, P.e_w // 256, 256], [0, 4 * P.e_w, 256, 1]),
                                task_group=tg)
                    for row in range(4):
                        ftaps = P.fill_taps(k, soff, row, 0, row, si, si + 1)
                        for ti_i, (o_, sz, st) in enumerate(ftaps):
                            rt.fill(f_in_shim[row].prod(), bufs[src], tap=tap(P.size[src], o_, sz, st),
                                    wait=(not has_drain and ti_i == len(ftaps) - 1), task_group=tg)
                    if has_drain:
                        g_start = (si - 4) * nicp
                        # 分 2 个 BD (每 BD 4×2560=10240 B < 16384 B)
                        for half in range(2):
                            off0 = doff + (c.BR * (tout_ctot // 16) + g_start + half * 4) * out_plane_stride + c.PAD * 16
                            rt.drain(f_out_shim[0].cons(), bufs[dst],
                                     tap=tap(P.size[dst], off0, [1, 4, 4, c.CW * 16], [0, out_plane_stride, out_row_stride, 1]),
                                     wait=(half == 1), task_group=tg)
                    s_cur += 1
                    while pending:
                        rt.finish_task_group(pending.pop(0))

            elif kind == "attn":
                tin = P.tensors[k + 1 + layers[k].src]
                tin_ctot = tin.ctot or tin.c
                in_plane_stride = (tin.ws + 2 * c.PAD) * 16
                in_row_stride = (tin_ctot // 16) * in_plane_stride
                for h_idx in range(HEADS):
                    active_col = h_idx // 2
                    # sub=0: 无 drain
                    tg0 = rt.task_group()
                    pending.append(tg0)
                    for col in range(4):
                        rt.fill(f_w[col].prod(), w,
                                tap=tap(nw, (s_cur * 4 + col) * P.e_w, [1, 1, P.e_w // 256, 256], [0, 4 * P.e_w, 256, 1]),
                                task_group=tg0)
                    for row in range(4):
                        q_off = soff + ((c.BR + row) * (tin_ctot // 16) + h_idx * 4) * in_plane_stride + (c.PAD - 4) * 16
                        k_off = soff + (c.BR * (tin_ctot // 16) + 32 + h_idx * 4) * in_plane_stride + (c.PAD - 4) * 16
                        rt.fill(f_in_shim[row].prod(), bufs[src],
                                tap=tap(P.size[src], q_off, [1, 1, 8, 960], [0, in_row_stride, in_plane_stride, 1]),
                                task_group=tg0)
                        rt.fill(f_in_shim[row].prod(), bufs[src],
                                tap=tap(P.size[src], k_off, [1, 4, 4, 960], [0, in_row_stride, in_plane_stride, 1]),
                                wait=True, task_group=tg0)
                    s_cur += 1
                    while pending:
                        rt.finish_task_group(pending.pop(0))

                    # sub=1: active_col 产出 4 个 row 的该 head (4 个通道组)
                    tg1 = rt.task_group()
                    pending.append(tg1)
                    for col in range(4):
                        rt.fill(f_w[col].prod(), w,
                                tap=tap(nw, (s_cur * 4 + col) * P.e_w, [1, 1, P.e_w // 256, 256], [0, 4 * P.e_w, 256, 1]),
                                task_group=tg1)
                    for row in range(4):
                        v0_off = soff + (c.BR * (tin_ctot // 16) + 64 + h_idx * 4) * in_plane_stride + (c.PAD - 4) * 16
                        v1_off = v0_off + 2 * in_plane_stride
                        rt.fill(f_in_shim[row].prod(), bufs[src],
                                tap=tap(P.size[src], v0_off, [1, 4, 2, 960], [0, in_row_stride, in_plane_stride, 1]),
                                task_group=tg1)
                        rt.fill(f_in_shim[row].prod(), bufs[src],
                                tap=tap(P.size[src], v1_off, [1, 4, 2, 960], [0, in_row_stride, in_plane_stride, 1]),
                                task_group=tg1)
                    off0 = doff + (c.BR * (tout_ctot // 16) + h_idx * 4) * out_plane_stride + c.PAD * 16
                    rt.drain(f_out_shim[active_col].cons(), bufs[dst],
                             tap=tap(P.size[dst], off0, [1, 4, 4, c.CW * 16], [0, out_plane_stride, out_row_stride, 1]),
                             wait=True, task_group=tg1)
                    s_cur += 1
                    while pending:
                        rt.finish_task_group(pending.pop(0))

            while pending:
                rt.finish_task_group(pending.pop(0))

    return Program(iron.get_current_device(), rt).resolve_program()


class NPUTransformer40_16c:
    """40 层 ASR Transformer 的 16 核 NPU 执行器 (5 块 × 8 段)。"""

    def __init__(self, z: dict, sh0: np.ndarray, sh_final: np.ndarray, all_params: list[c.LParam] | None = None,
                 *, prepacked: tuple[list[np.ndarray], list[np.ndarray]] | None = None):
        self.z = z
        self.head = FastFrontend(z)
        self.sh0 = sh0
        self.sh_final = sh_final
        self.all_params = all_params
        self.layers = make_block_layers(BLOCKS_PER_CALL)
        self.spec = c.spec_str(self.layers)
        self.P = c.Plan(self.layers)

        self.w_bos = []
        self.vt_idxs = []
        if prepacked is not None:
            w_arrs, vt_idxs = prepacked
            for w_arr, vt_idx in zip(w_arrs, vt_idxs):
                self.w_bos.append(iron.tensor(w_arr, dtype=np.int8, device="npu"))
                self.vt_idxs.append(vt_idx)
        else:
            assert all_params is not None
            for g in range(N_CALLS):
                p_grp = all_params[g * BLOCKS_PER_CALL * 9 : (g + 1) * BLOCKS_PER_CALL * 9]
                _, _, _, w_arr, vt_idx = _build_16c_schedule(self.layers, p_grp)
                self.w_bos.append(iron.tensor(w_arr, dtype=np.int8, device="npu"))
                self.vt_idxs.append(vt_idx)

        self.t_act = iron.zeros(self.P.size["act"], dtype=np.int8, device="npu")
        self.t_s = iron.zeros(self.P.size["s"], dtype=np.int8, device="npu")
        self.t_out = iron.zeros(self.P.size["out"], dtype=np.uint8, device="npu")
        self.n_col_steps = self.w_bos[0].data.size // (4 * self.P.e_w)
        self._last_vt = T_WIN

    @classmethod
    def from_cache(cls, z: dict, cache_path: Path = Path.home() / ".npu" / "asr" / "asr_16c_weights.npz") -> "NPUTransformer40_16c":
        """优先从磁盘缓存加载已校准打包的 16 核权重 (启动 <1s、不吃 CPU); 缓存不存在时自动校准并落盘。"""
        if cache_path.exists():
            d = np.load(cache_path)
            w_arrs = [d[f"w{g}"] for g in range(N_CALLS)]
            vt_idxs = [d[f"vt{g}"] for g in range(N_CALLS)]
            return cls(z, d["sh0"], d["sh_final"], prepacked=(w_arrs, vt_idxs))
        cal_wins = [load_wav_window(SAMPLES_DIR / fn) for fn in ("funasr_asr_example.wav", "ming16k.wav")]
        sh0, sh_final, all_params = calibrate_and_freeze(cal_wins, z)
        layers = make_block_layers(BLOCKS_PER_CALL)
        w_arrs, vt_idxs = [], []
        for g in range(N_CALLS):
            p_grp = all_params[g * BLOCKS_PER_CALL * 9 : (g + 1) * BLOCKS_PER_CALL * 9]
            _, _, _, w_arr, vt_idx = _build_16c_schedule(layers, p_grp)
            w_arrs.append(w_arr)
            vt_idxs.append(vt_idx)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_kw = {"sh0": sh0, "sh_final": sh_final}
        for g in range(N_CALLS):
            save_kw[f"w{g}"] = w_arrs[g]
            save_kw[f"vt{g}"] = vt_idxs[g]
        np.savez(cache_path, **save_kw)
        return cls(z, sh0, sh_final, all_params, prepacked=(w_arrs, vt_idxs))

    def set_valid_t(self, valid_t: int):
        if valid_t == self._last_vt:
            return
        for bo, vt_idx in zip(self.w_bos, self.vt_idxs):
            bo.data.reshape(-1).view(np.int32)[vt_idx] = int(valid_t)
            bo._sync_to_device()
        self._last_vt = valid_t

    def run_hidden(self, h0_u8: np.ndarray, valid_t: int = T_WIN) -> np.ndarray:
        self.set_valid_t(valid_t)
        cur = h0_u8.view(np.int8)
        P = self.P
        t_in, t_out = P.tensors[0], P.tensors[-1]
        _, off = P.loc[0]
        ctot = t_in.ctot or t_in.c
        root = off - t_in.g0 * t_in.cplane
        a = self.t_act.data.view(np.uint8)
        in_view = a[root: root + (t_in.h + 2 * c.BR) * (ctot // 16) * t_in.cplane].reshape(t_in.h + 2 * c.BR, ctot // 16, -1, 16)
        g_pad = np.zeros((t_in.c, t_in.h, t_in.ws), np.uint8)
        for g in range(N_CALLS):
            g_pad[:, :, :t_in.w] = cur.view(np.uint8)[:, :t_in.h, :t_in.w]
            in_view[c.BR:c.BR + t_in.h, t_in.g0:t_in.g0 + t_in.c // 16, c.PAD:c.PAD + t_in.ws] = g_pad.reshape(
                t_in.c // 16, 16, t_in.h, t_in.ws).transpose(2, 0, 3, 1)
            self.t_act._sync_to_device()
            transformer_16c(self.t_act, self.w_bos[g], self.t_s, self.t_out, spec=self.spec)
            full = c.from_layout(self.t_out.numpy(), t_out)
            cur = full[:, c.BR:c.BR + t_out.h, c.PAD:c.PAD + t_out.w].view(np.int8)
        return cur.view(np.uint8)

    def forward_window(self, win: np.ndarray, valid_t: int) -> np.ndarray:
        """700 帧 fbank 窗 → 前端 (CPU) → 40 层 Transformer (16核 NPU) → out_w (CPU) → logits [valid_t, VOCAB]。"""
        h_real = self.head.forward_in(win, valid_t)
        hq = np.clip(np.rint(h_real / self.sh0), -127, 127).astype(np.int8)
        h0_u8 = hq.T.reshape(D, 4, 35).view(np.uint8)
        h40_u8 = self.run_hidden(h0_u8, valid_t)
        return self.head.forward_out(h40_u8, self.sh_final, valid_t)


def main(iters: int = 5) -> int:
    z = dict(np.load(NPZ))
    vocab = [l.split()[1] if len(l.split()) >= 2 else "" for l in (MODEL / "dict.decoder.utf8.txt").read_text().splitlines()]
    samples = [
        ("funasr_asr_example.wav", "欢迎大家来体验达摩院推出的语音识别模型。"),
        ("ming16k.wav", "丽江和湘江有秦代开凿的灵渠沟通。"),
    ]
    wins = [load_wav_window(SAMPLES_DIR / fn) for fn, _ in samples]
    sh0, sh_final, all_params = calibrate_and_freeze(wins, z)
    runner = NPUTransformer40_16c(z, sh0, sh_final, all_params)
    print(f"[asr-16c] 16核 (4×4) 打包完成: {runner.n_col_steps} 步/列/段 (原 4240 步 → {runner.n_col_steps} 步, 权重流量降 {(4240/runner.n_col_steps):.2f}×)")

    worst_40 = 0
    for (fn, expect), (win, vt) in zip(samples, wins):
        _, y3 = frontend(win, z)
        h_real = y3 @ (z["lin_w"].T * float(z["lin_s"])) + z["lin_b"]
        h_real[vt:] = 0.0
        h0_u8 = np.clip(np.rint(h_real / sh0), -127, 127).astype(np.int8).T.reshape(D, 4, 35).view(np.uint8)
        t0 = time.perf_counter()
        npu_u8 = runner.run_hidden(h0_u8, vt)
        el = time.perf_counter() - t0
        ref_u8 = sim_blocks(h0_u8, all_params, vt)
        diff = int(np.abs(npu_u8.view(np.int8).astype(np.int32) - ref_u8.view(np.int8).astype(np.int32)).max())
        worst_40 = max(worst_40, diff)
        h_final = (npu_u8.view(np.int8).reshape(D, T_WIN).T.astype(np.float64)) * sh_final
        logits = h_final[:vt] @ (z["out_w"].T * float(z["out_s"])) + z["out_b"]
        got = text(logits.argmax(1).astype(np.uint32), vocab)
        ok = (got == expect) and (diff == 0)
        print(f"[asr-16c] {fn:<24s} 40层最大差={diff}  耗时={el*1e3:.1f}ms  结果: {got}  {'✅' if ok else '❌'}")

    win0, vt0 = wins[0]
    _, y3 = frontend(win0, z)
    h_real = y3 @ (z["lin_w"].T * float(z["lin_s"])) + z["lin_b"]
    h_real[vt0:] = 0.0
    h0_u8 = np.clip(np.rint(h_real / sh0), -127, 127).astype(np.int8).T.reshape(D, 4, 35).view(np.uint8)
    walls = []
    for _ in range(iters):
        t0 = time.perf_counter()
        runner.run_hidden(h0_u8, vt0)
        walls.append((time.perf_counter() - t0) * 1e3)
    walls.sort()
    med, best = walls[len(walls) // 2], walls[0]
    print(f"[asr-16c] 40 层 Transformer (16核) NPU 耗时: 中位 {med:.1f} ms  最快 {best:.1f} ms ({best / LAYERS:.2f} ms/层, 7.0s 窗口 → {7000.0 / best:.1f}× 实时)")
    return 0 if worst_40 == 0 else 1


if __name__ == "__main__":
    it = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 5
    sys.exit(main(it))
