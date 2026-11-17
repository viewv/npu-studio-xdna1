"""PP-OCRv6_medium Rec 16 核 (4×4 全阵列) NPU 加速验证:

  source npu/env.sh && .venv-iron/bin/python -m npu.ocr.v6_rec_16c [--iters 10]
"""
from __future__ import annotations

import sys
import time
import numpy as np

import aie.iron as iron
from aie.iron import Buffer, CompileTime, ExternalFunction, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device.tile import Tile
from aie.helpers.taplib import TensorAccessPattern

from .. import chain_c16 as c
from .det_npu import read_tensor
from .rec_npu import col, cos, preprocess_rec, render_text_strip
from . import v6_det_npu, v6_rec_npu


def _pad_h4(layers: list[c.Layer]) -> list[c.Layer]:
    """将层高度向上补齐到 4 的倍数 (方便 4 行×4 列 join 统一写出)，h=1 (SE FC) 保持 1。"""
    out = []
    for l in layers:
        if l.h == 1:
            out.append(l)
            continue
        hp = -(-l.h // 4) * 4 if l.h > 1 else 1
        # 对于 sy=2 的 DW (12->6): 输入 h=12, 输出 ho=6 -> 保持 l.h=12, 写出补到 8
        out.append(c.Layer(l.ic, l.oc, l.k, l.s, hp, l.w, dw=l.dw, d=l.d, op=l.op,
                           act=l.act, src=l.src, sc=l.sc, cat=l.cat, oh=l.oh, nx=l.nx, al=l.al, sy=l.sy))
    return out


ACC_CHUNKS_16 = 4


def _plan_16c(layers: list[c.Layer]) -> c.Plan:
    """基于 16 核 (rows_16 = ceil(ho/4)) 重新优化 ict 层的 nocp 与 n_rb。"""
    P = c.Plan(layers)
    for l, tl in zip(layers, P.tilings):
        if tl.mode != "ict" or l.h == 1:
            continue
        rows_16 = max(1, -(-l.ho // 4))
        k_cap = ACC_CHUNKS_16 // (rows_16 * tl.nxc)
        if k_cap < 1:
            continue
        m = tl.icp_pad
        nocp = c._divisor_le(l.ocp, min(k_cap, (P.e_w - c.HDR) // (m * 256 + c.RQB)))
        if nocp > tl.nocp or tl.n_rb > 1:
            tl.nocp = nocp
            tl.n_ob = l.ocp // nocp
            tl.n_rb = 1
            tl.steps = [(o * nocp, nocp, i * m, min(m, l.icp - i * m), int(i == 0), int(i == tl.n_icb - 1), 0)
                        for o in range(tl.n_ob) for i in range(tl.n_icb)]
    return P


def build_v6_rec_16c_schedule(layers: list[c.Layer], params: list[c.LParam]):
    P = _plan_16c(layers)
    col_steps = [[] for _ in range(4)]
    ops = []
    for k, (l, tl, p) in enumerate(zip(layers, P.tilings, params)):
        if l.op == 3:
            continue
        raw_w = c.pack_layer(l, tl, p, P.e_w, P.tensors[P.inp[k]].ws, c.signed_in(layers, k),
                             P.tensors[k + 1 + l.sc].c // 16 if l.sc else 0)
        if l.op == 1:
            # pool: col=0 跑全部通道组 (row=0..3 跑同一份, 只用 row=0), col=1..3 跳过 (<0.3ms)
            # 这里用 4-core pack_layer? 注意 4-core pool 把通道按 c%4 分给了 4 个 core!
            # 直接用 Layer(ic, oc, ..., op=1) 在 1 个核上算全部通道: 自己构造 2 步 (累加 + 收尾) 或让 col=0..3 各自算?
            # 更简单: pool 的输入是 all rows, 在 build_v6_rec_16c_schedule 里直接让 col=0 算全部通道组:
            for s_i in range(len(tl.steps)):
                w0 = raw_w[s_i * c.N_CORES].copy()
                w0[32:96] = np.tile(w0[32:48], 4)
                for rr in range(4):
                    hc = 32 + 16 * rr
                    hdr_rr = w0[hc:hc + 16].view(np.uint16)
                    hdr_rr[0] = l.icp if s_i == len(tl.steps) - 1 else l.icp * max(0, tl.n_icb - 1)
                col_steps[0].append(w0)
                w_skip = w0.copy()
                for rr in range(4):
                    hc = 32 + 16 * rr
                    hdr_rr = w_skip[hc:hc + 16].view(np.uint16)
                    # 保持 rows 不变 (消耗广播的 f_in[row]), 设 nch=0, sel_acc=1, sel_fin=0
                    hdr_rr[4], hdr_rr[5], hdr_rr[6] = 0, 1, 0
                for col_i in range(1, 4):
                    col_steps[col_i].append(w_skip.copy())
            ops.append(("pool", k, len(tl.steps)))
        elif l.dw:
            # DW 3x3: col=0 跑全部通道组, col=1..3 跳过; 每个 ti 跑全量 len(tl.steps) 步
            rows_16 = max(1, -(-l.ho // 4))
            for ti in range(tl.n_tw):
                wvalid = min(max(l.wo - ti * c.CW, 0), c.CW)
                for s_i in range(len(tl.steps)):
                    w0 = raw_w[s_i * c.N_CORES].copy()
                    w0[32:96] = np.tile(w0[32:48], 4)
                    for rr in range(4):
                        hdr_rr = w0[32 + 16 * rr: 48 + 16 * rr].view(np.uint16)
                        rem = l.ho - rr * rows_16
                        hdr_rr[0] = rows_16
                        hdr_rr[3] = 0 if rem <= 0 else wvalid
                        hdr_rr[7] = rem if 0 < rem < rows_16 else 0
                    col_steps[0].append(w0)
                    w_skip = w0.copy()
                    for rr in range(4):
                        hc = 32 + 16 * rr
                        w_skip[hc:hc + 16].view(np.uint16)[:7] = np.array([rows_16, 0, 0, wvalid, 0, 1, 0], dtype=np.uint16)
                    for col_i in range(1, 4):
                        col_steps[col_i].append(w_skip.copy())
            ops.append(("dw", k, tl.n_tw, len(tl.steps), [st[1] for st in tl.steps]))
        elif l.h == 1:
            # SE fc1 / fc2 (1x1 @ 1x1): 4 个 row 跑同一份, 按 col 分担 OC
            n_ob = tl.n_ob
            n_g = -(-n_ob // 4)
            spo = len(tl.steps) // n_ob
            for g in range(n_g):
                for s_i in range(spo):
                    for col_i in range(4):
                        ob = g * 4 + col_i
                        if ob < n_ob:
                            slot = ((g * spo + s_i) * c.N_CORES + col_i) if tl.ocpar else ((ob * spo + s_i) * c.N_CORES)
                            w_buf = raw_w[slot].copy()
                            w_buf[32:96] = np.tile(w_buf[32 + 16 * col_i: 48 + 16 * col_i] if tl.ocpar else w_buf[32:48], 4)
                        else:
                            w_buf = raw_w[0].copy()
                            skip_hdr = [1, 0, 0, 1, 0, 1, 0] if tl.mode == "ict" else [1, 1, 0, 1, 0, 0, 0]
                            for rr in range(4):
                                hc = 32 + 16 * rr
                                w_buf[hc:hc + 16].view(np.uint16)[:7] = np.array(skip_hdr, dtype=np.uint16)
                        col_steps[col_i].append(w_buf)
            ops.append(("se_fc", k, n_g, n_ob, spo, tl.steps[0][1]))
        else:
            # 1x1 (ict / oc / full): 16 核二维并行 (4 行 × 4 列), 不足 4 块时多余列跳过
            if tl.pre:
                w_load = raw_w[0].copy()
                w_load[32:96] = np.tile(w_load[32:48], 4)
                for col_i in range(4):
                    col_steps[col_i].append(w_load.copy())
                ops.append(("se_load", k, tl.pre))
            n_ob_4 = tl.n_ob
            n_ob = n_ob_4 // tl.n_rb
            n_g = -(-n_ob // 4)
            spo = len(tl.steps) // n_ob_4
            rows_16 = max(1, -(-l.ho // 4))
            is_ict = tl.mode == "ict"
            rcap_16 = ACC_CHUNKS_16 // tl.nxc
            n_rb_16 = -(-rows_16 // rcap_16) if is_ict else 1
            assert n_rb_16 == 1 or is_ict, f"层 {k}: non-ict rows_16={rows_16} > rcap={rcap_16}"
            pre_off = 1 if tl.pre else 0
            for ti in range(tl.n_tw):
                wvalid = min(max(l.wo - ti * c.CW, 0), c.CW)
                for g in range(n_g):
                    for rb in range(n_rb_16):
                        r0 = rb * rcap_16
                        rs = min(rcap_16, rows_16 - r0) if n_rb_16 > 1 else rows_16
                        for s_i in range(spo):
                            for col_i in range(4):
                                oc_b = g * 4 + col_i
                                if oc_b < n_ob:
                                    ob = oc_b * tl.n_rb
                                    slot = (pre_off + ob * spo + s_i) * c.N_CORES
                                    w_buf = raw_w[slot].copy()
                                    w_buf[32:96] = np.tile(w_buf[32:48], 4)
                                    for rr in range(4):
                                        hdr_rr = w_buf[32 + 16 * rr: 48 + 16 * rr].view(np.uint16)
                                        rem = l.ho - (rr * rows_16 + r0)
                                        hdr_rr[0] = rs
                                        hdr_rr[3] = 0 if rem <= 0 else wvalid
                                        hdr_rr[7] = rem if 0 < rem < rs else 0
                                else:
                                    w_buf = raw_w[pre_off * c.N_CORES].copy()
                                    skip_hdr = ([rs, 0, 0, wvalid, 0, 1, 0] if is_ict
                                                else [rs, 1, 0, wvalid, 0, 0, 0])
                                    for rr in range(4):
                                        hc = 32 + 16 * rr
                                        w_buf[hc:hc + 16].view(np.uint16)[:7] = np.array(skip_hdr, dtype=np.uint16)
                                col_steps[col_i].append(w_buf)
                ops.append(("conv1", k, ti, n_g, n_ob, spo, tl.steps[0][1], n_rb_16, rcap_16))

    n_col_steps = len(col_steps[0])
    assert all(len(cs) == n_col_steps for cs in col_steps)
    w_packed = np.stack([np.stack(col_steps[col_i], axis=0) for col_i in range(4)], axis=1).reshape(-1).copy()
    return P, ops, n_col_steps, w_packed


@iron.jit
def v6_rec_16c_kernel(act: c.In, wts: c.In, scratch: c.InOut, out: c.Out, *, spec: CompileTime[str]):
    layers = c.from_spec_str(spec)
    P = _plan_16c(layers)
    n_col_steps = 0
    ops = []
    for k, (l, tl) in enumerate(zip(layers, P.tilings)):
        if l.op == 3:
            continue
        if l.op == 1:
            n_col_steps += len(tl.steps)
            ops.append(("pool", k, len(tl.steps)))
        elif l.dw:
            n_col_steps += tl.n_tw * len(tl.steps)
            ops.append(("dw", k, tl.n_tw, len(tl.steps), [st[1] for st in tl.steps]))
        elif l.h == 1:
            n_ob = tl.n_ob
            n_g = -(-n_ob // 4)
            spo = len(tl.steps) // n_ob
            n_col_steps += n_g * spo
            ops.append(("se_fc", k, n_g, n_ob, spo, tl.steps[0][1]))
        else:
            if tl.pre:
                n_col_steps += 1
                ops.append(("se_load", k, tl.pre))
            n_ob_4 = tl.n_ob
            n_ob = n_ob_4 // tl.n_rb
            n_g = -(-n_ob // 4)
            spo = len(tl.steps) // n_ob_4
            rows_16 = max(1, -(-l.ho // 4))
            rcap_16 = ACC_CHUNKS_16 // tl.nxc
            n_rb_16 = -(-rows_16 // rcap_16) if tl.mode == "ict" else 1
            n_col_steps += tl.n_tw * n_g * n_rb_16 * spo
            for ti in range(tl.n_tw):
                ops.append(("conv1", k, ti, n_g, n_ob, spo, tl.steps[0][1], n_rb_16, rcap_16))

    i8, u8, i32 = np.dtype[np.int8], np.dtype[np.uint8], np.dtype[np.int32]
    in_ty = np.ndarray[(c.E_IN,), i8]
    out_ty = np.ndarray[(c.CW * 16,), u8]
    out4_ty = np.ndarray[(4 * c.CW * 16,), u8]
    w_ty = np.ndarray[(P.e_w,), i8]
    acc_ty = np.ndarray[(ACC_CHUNKS_16 * c.CW * 16,), i32]
    sc_ty = np.ndarray[(c.SC_BYTES,), u8]

    kern = ExternalFunction(
        "convC16Chunk", source_file=c.SRC,
        arg_types=[in_ty, in_ty, in_ty, w_ty, out_ty, acc_ty, sc_ty, np.int32, np.int32, np.int32],
        include_dirs=[c.KERNEL_INC],
        compile_flags=[f"-DCW={c.CW}", *P.kernel_flags()],
    )

    f_w = [ObjectFifo(w_ty, depth=2, name=f"w_c{col_i}") for col_i in range(4)]
    f_in_shim = [ObjectFifo(in_ty, depth=2, name=f"in_shim_r{row}") for row in range(4)]
    f_in = [f_in_shim[row].cons().forward(tile=Tile(row, 1), depth=4, name=f"in_r{row}") for row in range(4)]
    f_out_shim = [ObjectFifo(out4_ty, depth=2, name=f"out_shim_c{col_i}") for col_i in range(4)]
    f_out = [
        f_out_shim[col_i].prod().join(
            offsets=[r * c.CW * 16 for r in range(4)],
            tile=Tile(col_i, 1),
            depths=[2, 2, 2, 2],
            obj_types=[out_ty, out_ty, out_ty, out_ty],
            names=[f"out_c{col_i}_r{r}" for r in range(4)],
        )
        for col_i in range(4)
    ]

    workers = []
    def make_core(row):
        hc = 32 + 16 * row
        def core(fi, fw, fo, k, acc, dummy, sc):
            for _ in range_(n_col_steps):
                wv = fw.acquire(1)
                rows, nch = c._u16(wv, hc), c._u16(wv, hc + 8)
                for _ in range_(c._u16(wv, hc + 2)):
                    fi.acquire(2)
                    for r in range_(rows):
                        win = fi.acquire(3)
                        for j in range_(nch):
                            o = fo.acquire(1)
                            k(win[0], win[1], win[2], wv, o, acc, sc, c._i32(j), c._i32(r), row)
                            fo.release(1)
                        fi.release(1)
                    fi.release(2)
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

    for col_i in range(4):
        for row in range(4):
            workers.append(Worker(
                make_core(row),
                [f_in[row].cons(), f_w[col_i].cons(), f_out[col_i][row].prod(), kern,
                 Buffer(acc_ty, name=f"acc_{col_i}_{row}"),
                 Buffer(out_ty, name=f"dummy_{col_i}_{row}"),
                 Buffer(sc_ty, name=f"sc_{col_i}_{row}")],
                tile=Tile(col_i, 2 + row),
            ))

    nw = n_col_steps * 4 * P.e_w
    tap = lambda n, o_, sz, st: TensorAccessPattern((1, n), o_, sz, st)
    in_h = [f.prod() for f in f_in_shim]
    w_h = [f.prod() for f in f_w]
    out_h = [f.cons() for f in f_out_shim]

    def seq(a, w, s, o, *handles):
        bufs = {"act": a, "s": s, "out": o}
        h_in, h_w, h_out = handles[:4], handles[4:8], handles[8:12]
        s_cur = 0
        pending = []

        def wfill(n_sp, tg):
            nonlocal s_cur
            for col_i in range(4):
                h_w[col_i].fill(w, tap=tap(nw, (s_cur * 4 + col_i) * P.e_w,
                                           [1, n_sp, P.e_w // 256, 256], [0, 4 * P.e_w, 256, 1]), group=tg)
            s_cur += n_sp

        for op_desc in ops:
            kind, k = op_desc[0], op_desc[1]
            (src, soff), (dst, doff) = P.io[k]
            tl = P.tilings[k]
            l = layers[k]
            tin = P.tensors[P.inp[k]]
            tout = P.tensors[k + 1]
            tout_ctot = tout.ctot or tout.c
            out_row_stride = (tout_ctot // 16) * (tout.ws + 2 * c.PAD) * 16
            out_plane_stride = (tout.ws + 2 * c.PAD) * 16

            if kind == "se_load":
                _, _, pre = op_desc
                tg = TaskGroup()
                wfill(1, tg)
                ssrc, sbase = P.loc[k + 1 + l.sc]
                for row in range(4):
                    h_in[row].fill(bufs[ssrc], tap=tap(P.size[ssrc], *P.scale_tap(k, sbase)),
                                   wait=True, group=tg)
                tg.finish()
            elif kind == "conv1":
                _, _, ti, ng, n_ob, spo, nocp, n_rb_16, rcap_16 = op_desc
                rows_16 = max(1, -(-l.ho // 4))
                for g in range(ng):
                    for rb in range(n_rb_16):
                        r0 = rb * rcap_16
                        rs = min(rcap_16, rows_16 - r0) if n_rb_16 > 1 else rows_16
                        tg = TaskGroup()
                        pending.append(tg)
                        wfill(spo, tg)
                        for row in range(4):
                            if tl.mode == "ict":
                                off_r = soff + (c.BR + row * rows_16 + r0) * tin.frow + ti * c.CW * 16
                                h_in[row].fill(bufs[src], tap=tap(P.size[src], off_r,
                                                                  [tl.n_icb, rs, tl.icp_pad, tl.plane],
                                                                  [tl.icp_pad * tin.cplane, tin.frow, tin.cplane, 1]), group=tg)
                            else:
                                off_r = soff + (c.BR - 1 + row * rows_16) * tin.frow + ti * c.CW * 16
                                h_in[row].fill(bufs[src], tap=tap(P.size[src], off_r,
                                                                  [1, rs + 2, tl.icp_pad, tl.plane],
                                                                  [0, tin.frow, tin.cplane, 1]), group=tg)
                        for col_i in range(4):
                            oc_ob = g * 4 + col_i
                            if oc_ob < n_ob:
                                off0 = (doff + (c.BR * (tout_ctot // 16) + oc_ob * nocp) * out_plane_stride
                                        + r0 * out_row_stride + (c.PAD + ti * c.CW) * 16)
                                h_out[col_i].drain(bufs[dst], tap=tap(P.size[dst], off0, [rs, nocp, 4, c.CW * 16],
                                                                      [out_row_stride, out_plane_stride, rows_16 * out_row_stride, 1]),
                                                   wait=True, group=tg)
                        while len(pending) >= 2:
                            pending.pop(0).finish()
            elif kind == "dw":
                _, _, n_tw, n_steps, nocp_list = op_desc
                rows_16 = max(1, -(-l.ho // 4))
                for ti in range(n_tw):
                    for s_i in range(n_steps):
                        nocp_i = nocp_list[s_i]
                        tg = TaskGroup()
                        pending.append(tg)
                        wfill(1, tg)
                        icp0 = tl.steps[s_i][0]
                        for row in range(4):
                            o0 = soff + (c.BR + l.stride_y * row * rows_16 - l.pad) * tin.frow + ti * l.stride_x * c.CW * 16
                            h_in[row].fill(bufs[src], tap=tap(P.size[src], o0 + icp0 * tin.cplane,
                                                              [rows_16, l.k, tl.icp_pad, tl.plane],
                                                              [l.stride_y * tin.frow, l.d * tin.frow, tin.cplane, 1]), group=tg)
                        off0 = doff + (c.BR * (tout_ctot // 16) + tl.steps[s_i][2]) * out_plane_stride + (c.PAD + ti * c.CW) * 16
                        h_out[0].drain(bufs[dst], tap=tap(P.size[dst], off0, [rows_16, nocp_i, 4, c.CW * 16],
                                                          [out_row_stride, out_plane_stride, rows_16 * out_row_stride, 1]),
                                       wait=True, group=tg)
                        while len(pending) >= 2:
                            pending.pop(0).finish()
            elif kind == "pool":
                _, _, n_ps = op_desc
                # pool: col=0 跑全部通道组 (row=0..3 跑同一份)
                ne, kr = tl.n_icb, tl.icp_pad
                if ne > 1:
                    tg1 = TaskGroup()
                    wfill(1, tg1)
                    for row in range(4):
                        o0 = soff + c.BR * tin.frow
                        h_in[row].fill(bufs[src], tap=tap(P.size[src], o0,
                                                          [l.icp, ne - 1, kr, tl.plane],
                                                          [tin.cplane, kr * tin.frow, tin.frow, 1]),
                                       wait=True, group=tg1)
                    tg1.finish()
                tg2 = TaskGroup()
                wfill(1 if ne > 1 else n_ps, tg2)
                for row in range(4):
                    o0 = soff + c.BR * tin.frow
                    h_in[row].fill(bufs[src], tap=tap(P.size[src], o0 + (ne - 1) * kr * tin.frow,
                                                      [1, l.icp, kr, tl.plane],
                                                      [0, tin.cplane, tin.frow, 1]), group=tg2)
                off0 = doff + c.BR * (tout_ctot // 16) * out_plane_stride + c.PAD * 16
                h_out[0].drain(bufs[dst], tap=tap(P.size[dst], off0, [1, l.ocp, 4, c.CW * 16],
                                                  [0, out_plane_stride, out_row_stride, 1]),
                               wait=True, group=tg2)
                tg2.finish()
            elif kind == "se_fc":
                _, _, n_g, n_ob, spo, nocp = op_desc
                for g in range(n_g):
                    tg = TaskGroup()
                    pending.append(tg)
                    wfill(spo, tg)
                    for row in range(4):
                        if tl.mode == "ict":
                            o0 = soff + c.BR * tin.frow
                            h_in[row].fill(bufs[src], tap=tap(P.size[src], o0,
                                                              [tl.n_icb, 1, tl.icp_pad, tl.plane],
                                                              [tl.icp_pad * tin.cplane, tin.frow, tin.cplane, 1]), group=tg)
                        else:
                            o0 = soff + (c.BR - 1) * tin.frow
                            h_in[row].fill(bufs[src], tap=tap(P.size[src], o0,
                                                              [1, 3, tl.icp_pad, tl.plane],
                                                              [0, tin.frow, tin.cplane, 1]), group=tg)
                    for col_i in range(4):
                        ob = g * 4 + col_i
                        if ob < n_ob:
                            off0 = doff + (c.BR * (tout_ctot // 16) + ob * nocp) * out_plane_stride + c.PAD * 16
                            h_out[col_i].drain(bufs[dst], tap=tap(P.size[dst], off0, [1, nocp, 4, c.CW * 16],
                                                                  [0, out_plane_stride, out_row_stride, 1]),
                                               wait=True, group=tg)
                    while len(pending) >= 2:
                        pending.pop(0).finish()
            while pending:
                pending.pop(0).finish()

    rt = Runtime(seq, [np.ndarray[(P.size["act"],), i8], np.ndarray[(nw,), i8],
                       np.ndarray[(P.size["s"],), i8], np.ndarray[(P.size["out"],), u8],
                       *in_h, *w_h, *out_h])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    iters = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 10
    chars = v6_rec_npu.load_v6_dict()
    text_gt = "Ryzen NPU 中文测试 2026"
    img = render_text_strip(text_gt, 48, 320)
    x = preprocess_rec(img)

    sess_stem = v6_det_npu.build_sub_ort(v6_rec_npu.V6_REC_ONNX, ["x"], [v6_rec_npu.REC_STEM_OUT], [[1, 3, 48, 320]], [[1, 64, 12, 80]])
    stem_feat = sess_stem.run(None, {"x": x[None]})[0][0]
    ops = v6_rec_npu.load_v6_rec_backbone()
    fl = v6_rec_npu.run_float_v6_rec(ops, stem_feat)
    shapes = {k: v.shape for k, v in fl.items()}
    qr = v6_rec_npu.QV6Rec(ops, [fl], pct=99.95)
    qr.run(stem_feat)
    qr.frozen = True
    qr.run(stem_feat)
    layers, params, tid = v6_rec_npu.build_v6_rec_chain(qr, ops, shapes)

    P, ops_16, n_col_steps, w_packed = build_v6_rec_16c_schedule(layers, params)
    spec = c.spec_str(layers)
    print(f"PP-OCRv6_medium Rec 16 核: {len(layers)-1} 层, n_col_steps={n_col_steps} (原 4 核 8288 步)")

    t_act = iron.zeros(P.size["act"], dtype=np.int8, device="npu")
    t_w = iron.tensor(w_packed, dtype=np.int8, device="npu")
    t_s = iron.zeros(P.size["s"], dtype=np.int8, device="npu")
    t_out = iron.zeros(P.size["out"], dtype=np.uint8, device="npu")
    t = P.tensors[1]
    _, root = P.loc[1]
    a = t_act.data.view(np.uint8)
    full = a[root: root + (t.h + 2 * c.BR) * (t.c // 16) * t.cplane].reshape(t.h + 2 * c.BR, t.c // 16, -1, 16)
    g = np.zeros((t.c, t.h, t.ws), np.uint8)
    x0 = qr.last_int[v6_rec_npu.REC_STEM_OUT][0]
    g[:, :, :t.w] = x0[:, :t.h, :t.w]
    full[c.BR:c.BR + t.h, :, c.PAD:c.PAD + t.ws] = g.reshape(t.c // 16, 16, t.h, t.ws).transpose(2, 0, 3, 1)
    t_act._sync_to_device()

    v6_rec_16c_kernel(t_act, t_w, t_s, t_out, spec=spec)
    t0 = time.perf_counter()
    for _ in range(iters):
        v6_rec_16c_kernel(t_act, t_w, t_s, t_out, spec=spec)
    npu_ms = (time.perf_counter() - t0) * 1000 / iters
    print(f"16 核 NPU 主干耗时 ({iters} 次平均): {npu_ms:.2f} ms")


if __name__ == "__main__":
    main()
