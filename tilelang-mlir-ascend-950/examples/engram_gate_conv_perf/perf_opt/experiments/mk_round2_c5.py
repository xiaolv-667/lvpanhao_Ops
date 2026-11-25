"""Build round-2 branch c5: hoist the row loads into bulk 2-D window copies.

Measured floor probe (perf_opt/experiments/floor_probe.py, grid=48, d=768, bf16,
msopprof Task Duration, median of 3 captures):

    kind=0  1 row load + 2 row stores                       2.72 us
    kind=1  27 row loads (baseline) + 2 row stores          6.07 us
    kind=4  3 bulk (rc+3, d) loads + 2 row stores           3.38 us
    kind=5  6 bulk loads (owned rc rows + halo 3 rows)      3.59 us

i.e. the baseline's 27 per-row DMA descriptors cost ~3.35 us of a 9.3 us block
while moving the same 41.5 KB that 6 bulk copies move in ~0.9 us.  The row-level
software pipeline (r2_c4) made this worse (12.9 us: MTE2 busy ratio 0.31 -> 0.72),
so the descriptor count, not the overlap, is the cost driver here.

This branch therefore keeps the plain `T.serial` row loop and only changes where
the data comes from:
  * the whole (rc + 3, d) H/K/V window is copied once per block (6 bulk copies:
    the rc owned rows plus the 3 causal halo rows, split because the halo origin
    is t0-3 and must fall back to a zero fill at the sequence start),
  * every per-row DMA inside the row loop disappears,
  * the row loop loses its `if r >= 0` guard (a zero-filled head row produces
    g = fl(0.5 * 0) * r_v = 0 exactly, which is what the causal zero fill means),
  * the owned-row outputs are written with one 2-D copy each (branch r1_c3),
  * r_v is used straight from its register (branch r1_c2).

    python mk_round2_c5.py
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(os.path.dirname(HERE), "engram_gate_conv_round2_c5.py")

OLD_ALLOC = '''            h_ub = T.alloc_shared((dw,), dtype)
            k_ub = T.alloc_shared((dw,), dtype)
            v_ub = T.alloc_shared((dw,), dtype)
            y_ub = T.alloc_shared((dw,), dtype)
            vh_ub = T.alloc_shared((dw,), dtype)
'''

NEW_ALLOC = '''            # (rc + 3, dw) windows: the whole per-block row window (owned rows plus
            # the 3 causal halo rows) in one buffer per input tensor.
            h_ub = T.alloc_shared((rc + 3, dw), dtype)
            k_ub = T.alloc_shared((rc + 3, dw), dtype)
            v_ub = T.alloc_shared((rc + 3, dw), dtype)
            y_ub = T.alloc_shared((rc + 3, dw), dtype)
            vh_ub = T.alloc_shared((rc + 3, dw), dtype)
'''

OLD_HEAD = '''            # -------- row pipeline: RC owned rows + 3 halo rows ------------------
            for i in T.serial(rc + 3):
                r = t0 - 3 + i
                if r >= 0:  # block level causal guard: a missing halo row stays zero
                    T.copy(H[b, r, :], h_ub[:d])
                    T.copy(K[b, r, :], k_ub[:d])
                    T.copy(V[b, r, :], v_ub[:d])
                with T.SimdVF():
'''

NEW_HEAD = '''            # -------- window load: 6 bulk copies, no DMA inside the row loop ------
            # Only the block that owns the start of a sequence (t0 < 3) can have
            # halo rows that do not exist; those slots are zero filled, which is
            # the causal zero boundary of the reference.  Every other block loads
            # its 3 halo rows with one 3-row copy per tensor.
            if t0 < 3:
                with T.SimdVF():
                    zfull = T.simd.pset(32)
                    zb = T.simd.vcvt(
                        T.simd.vdup(T.float32(0), fp32, zfull),
                        dtype,
                        round="ROUND_R",
                        sat=False,
                        part=0,
                    )
                    for hz in T.serial(3):
                        for ch in T.serial(nchunks):
                            T.simd.vsts(h_ub[hz, ch * VL], zb, zfull, dist="PK_B32")
                            T.simd.vsts(k_ub[hz, ch * VL], zb, zfull, dist="PK_B32")
                            T.simd.vsts(v_ub[hz, ch * VL], zb, zfull, dist="PK_B32")
                for hz in range(3):  # build-time unroll over the 3 halo slots
                    rr = t0 - 3 + hz
                    if rr >= 0:
                        T.copy(H[b, rr, :], h_ub[hz, :d])
                        T.copy(K[b, rr, :], k_ub[hz, :d])
                        T.copy(V[b, rr, :], v_ub[hz, :d])
            else:
                T.copy(H[b, t0 - 3 : t0, :], h_ub[0:3, :d])
                T.copy(K[b, t0 - 3 : t0, :], k_ub[0:3, :d])
                T.copy(V[b, t0 - 3 : t0, :], v_ub[0:3, :d])
            T.copy(H[b, t0 : t0 + rc, :], h_ub[3 : 3 + rc, :d])
            T.copy(K[b, t0 : t0 + rc, :], k_ub[3 : 3 + rc, :d])
            T.copy(V[b, t0 : t0 + rc, :], v_ub[3 : 3 + rc, :d])

            # -------- row pipeline: RC owned rows + 3 halo rows ------------------
            for i in T.serial(rc + 3):
                r = t0 - 3 + i
                with T.SimdVF():
'''

OLD_GUARD = '''                    if r >= 0:
                        # Three row statistics in one scan'''
NEW_GUARD = '''                    if True:  # the head rows are zero filled, so every row is computed
                        # Three row statistics in one scan'''

OLD_RVB = '''                        T.simd.vsts(st[3, i, 0], rv_v[0], one, extent=1)
                        # r_v is re-read from the statistics tile as a broadcast
                        # vector (the same idiom the 950DT port uses for its gate
                        # scalar).  That read is a vector store -> vector load
                        # dependency, which needs the explicit barrier below.
                        T.simd.mem_bar("VST_VLD")
                        rvb = T.simd.vld(st[3, i, 0], dist="BRC_B32")
'''

NEW_RVB = '''                        T.simd.vsts(st[3, i, 0], rv_v[0], one, extent=1)
                        # r_v is already a lane-broadcast fp32 vector: use the
                        # register directly instead of the store -> BRC_B32 reload
                        # round trip and its VST_VLD barrier (branch r1_c2).
                        rvb = rv_v[0]
'''

OLD_TAIL = '''                if i >= 3:  # owned row only: the halo rows are not written back
                    T.copy(vh_ub[:d], VHAT[b, r, :])
                    T.copy(y_ub[:d], Y[b, r, :])

            # -------- four fp32 row scalars: one strided copy per member --------
'''

NEW_TAIL = '''
            # -------- owned rows: one 2-D store per output, not one per row ------
            # The owned rows are contiguous in GM, so a single (rc, d) region copy
            # replaces rc per-row copies, which each drained PIPE_MTE3 before the
            # next store (TRAP-950-rowstore-mte3-drain; branch r1_c3 measured the
            # MTE3 busy ratio drop 0.131 -> 0.033).
            T.copy(y_ub[3 : 3 + rc, :d], Y[b, t0 : t0 + rc, :])
            T.copy(vh_ub[3 : 3 + rc, :d], VHAT[b, t0 : t0 + rc, :])

            # -------- four fp32 row scalars: one strided copy per member --------
'''


def main():
    with open(PATH, encoding="utf-8") as f:
        text = f.read()

    def sub(old, new, count=1):
        nonlocal text
        if text.count(old) != count:
            raise SystemExit(f"anchor {text.count(old)}x (want {count}): {old[:120]!r}")
        text = text.replace(old, new)

    import re

    def sub_re(pat, new):
        nonlocal text
        text, n = re.subn(pat, new, text)
        if n == 0:
            raise SystemExit(f"no match: {pat}")
        print(f"  {pat} -> {new} x{n}")

    sub(OLD_ALLOC, NEW_ALLOC)
    sub(OLD_HEAD, NEW_HEAD)
    sub(OLD_GUARD, NEW_GUARD)
    sub(OLD_RVB, NEW_RVB)
    sub(OLD_TAIL, NEW_TAIL)
    sub_re(r"vh_ub\[ch \* VL\]", "vh_ub[i, ch * VL]")
    sub_re(r"y_ub\[ch \* VL\]", "y_ub[i, ch * VL]")
    for buf in ("h_ub", "k_ub", "v_ub"):
        sub_re(rf"(?<![A-Za-z0-9_]){buf}\[ch \* VL\]", f"{buf}[i, ch * VL]")

    with open(PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print("patched", PATH, len(text.splitlines()), "lines")


if __name__ == "__main__":
    main()
