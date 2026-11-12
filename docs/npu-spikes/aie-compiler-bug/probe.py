import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser('~/develop/npu-studio-effects'))
import aie.iron as iron
from aie.iron import CompileTime, In, Out, Program, Runtime
from npu.harness import RowPipeline, external_row_kernel
from npu.fused import KERNEL_INC, LIB, _up_h
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'probe.cc')
WMAX, LS, H = int(os.environ.get('WMAX', 2560)), 704, 4

@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def probe(inp: In, out: Out, *, variant: CompileTime[int], ls: CompileTime[int], width: CompileTime[int]):
    in_b, out_b = ls * 4 + WMAX * 2, WMAX * 2
    k = external_row_kernel("probeLine", SRC, [np.ndarray[(in_b,), np.dtype[np.uint8]]],
                            np.ndarray[(out_b,), np.dtype[np.uint8]], scalars=[np.int32],
                            include_dirs=[KERNEL_INC, LIB], compile_flags=[f"-DPV={variant}", f"-DPLS={ls}"])
    pl = RowPipeline(k, [in_b], out_b, height=H, n_cols=1, extra_args=[width], prefix=f"p{variant}")
    pl.build()
    rt = Runtime()
    with rt.sequence(*pl.tensor_types()) as bufs:
        pl.stream(rt, bufs)
    return Program(iron.get_current_device(), rt).resolve_program()

def ref(v, A, B, I):
    Au, Bu = _up_h(A[:, None, :]).squeeze(1) if False else None, None
    up = lambda X: _up_h(X)  # X (H, LW+?) 已含 pad? 这里传入未 pad 的 (H, LW)
    a, b = up(A), up(B)
    if v == 0: return a
    if v == 1: return np.clip((a * I + b) >> 6, -32768, 32767)
    if v in (2, 9): return np.clip((a * I + b) >> 6, 0, 255)
    if v == 5: return a + I
    if v == 6: return b + I
    if v == 7: return np.clip((a * I) >> 6, -32768, 32767)
    if v == 10: return a + b
    if v in (11, 12): return b
    if v == 3: return np.clip((I + b) >> 6, 0, 255)

variant = int(sys.argv[1]); ls = int(sys.argv[2]) if len(sys.argv) > 2 else LS
for W in [int(w) for w in os.environ.get("WIDTHS", "128,256,640,1280,1600,1920,2560").split(",")]:
    WIDTH = W   # 运行时标量 (编进 sequence 的常量参数)
    lw = W // 4
    rng = np.random.default_rng(W)
    A = rng.integers(-400, 400, (H, lw)).astype(np.int16)
    B = rng.integers(-8000, 24000, (H, lw)).astype(np.int16)
    I = rng.integers(-128, 128, (H, W)).astype(np.int16)
    buf = np.zeros((H, ls * 4 + WMAX * 2), np.uint8)
    va, vb = buf[:, :ls * 2].view(np.int16), buf[:, ls * 2:ls * 4].view(np.int16)
    for dst, s in ((va, A), (vb, B)):
        dst[:, 1:lw + 1] = s; dst[:, 0] = s[:, 0]; dst[:, lw + 1] = s[:, -1]
    buf[:, ls * 4:ls * 4 + W * 2].view(np.int16)[:] = I
    t_in = iron.tensor(buf.reshape(-1), dtype=np.uint8, device="npu")
    t_out = iron.zeros(H * WMAX * 2, dtype=np.uint8, device="npu")
    probe(t_in, t_out, variant=variant, ls=ls, width=W)
    raw = t_out.numpy().reshape(H, WMAX * 2)
    got = (raw[:, :W] if variant in (2, 3, 9) else raw[:, :W * 2].view(np.int16)).astype(int)
    exp = ref(variant, A, B, I.astype(int))
    d = got != exp
    lane = [d[:, r::4].mean() * 100 for r in range(4)]
    blk = np.flatnonzero(d.any(0)) // 32
    print(f"V={variant} LS={ls} W={W:5d} iters={W//32:3d}: wrong {d.mean()*100:6.2f}%  by x%4 "
          + " ".join(f"{l:5.1f}" for l in lane)
          + (f"  first bad block {blk.min()} of {W//32}" if d.any() else ""), flush=True)

    if variant in (5, 6) and d.any():
        A0 = A
        if variant == 6: A = B
        e = got - exp
        pad = np.concatenate([A[:, :1], A, A[:, -1:]], 1).astype(int)
        for r in (0, 1):
            er = e[:, r::4]
            print(f"   lane{r}: err sample {er[0,:8].tolist()}")
        # 假设: vp 取成了别的偏移 k (v[j-1+k]), 看哪个 k 能解释
        WPv=[3,1,0,0]; WCv=[5,7,7,5]; WNv=[0,0,1,3]
        for k in range(-3, 10):
            ok = True
            for r in (0,1):
                jj = np.arange(lw)
                vpk = pad[:, np.clip(jj + k, 0, lw + 1)]
                pred = (WPv[r]*vpk + WCv[r]*pad[:, jj+1] + WNv[r]*pad[:, jj+2]) >> 3
                ok &= np.array_equal(pred + I[:, r::4], got[:, r::4])
                if r == 0 and k == 0: print('   lane0 pred-vs-got sample', (got[:, r::4]-pred-I[:, r::4])[0,:8].tolist())
            if ok: print(f"   ✔ 解释: v[j-1] 被换成了 v[j-1+{k}]")

    if False:
        WPv=[3,1,0,0]; WCv=[5,7,7,5]; WNv=[0,0,1,3]
        pa = np.concatenate([A0[:, :1], A0, A0[:, -1:]], 1).astype(int) if variant == 6 else np.concatenate([A[:, :1], A, A[:, -1:]], 1).astype(int)
        pb = np.concatenate([B[:, :1], B, B[:, -1:]], 1).astype(int)
        jj = np.arange(lw); extra = I if variant == 6 else 0
        for k in range(-2, 4):
            ok = all(np.array_equal(((WPv[r]*pa[:, jj+k] + WCv[r]*pb[:, jj+1] + WNv[r]*pb[:, jj+2]) >> 3) + (extra[:, r::4] if variant == 6 else 0), got[:, r::4]) for r in (0, 1))
            if ok: print(f"   ✔ 解释: B 的 v[j-1] 被换成了 A 平面的 pad[j+{k}] (即 aq + j0 + {k})")
