"""Build round-2 branch c4: row-level software pipeline by build-time unrolling.

The baseline row loop is `for i in T.serial(rc+3)` with one row buffer set, so the
compiler emits "DMA row -> wait -> compute row" per iteration (measured: 27
serialised DMA waits per block at d=768, ~3.3 us of a 9.3 us block on top of a
2.7 us launch floor).  `T.Pipelined(num_stages=2)` does not rotate the buffers
here (experiment r1_c1: byte-identical emitted code, no version offsets), so the
pipeline is expressed explicitly instead: two row-buffer sets, the DMA of row i+1
issued before the vector compute of row i, and a build-time unrolled row loop so
every iteration statically selects its buffer set.

Also folds in the two wins measured independently in round 1:
  * r_v used straight from its register (no VST_VLD barrier + BRC reload)  [r1_c2]
  * the owned-row outputs staged in (rc+3, d) tiles and written with one 2-D
    copy each (no per-row PIPE_MTE3 drain)                               [r1_c3]

    python mk_round2_c4.py
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(os.path.dirname(HERE), "engram_gate_conv_round2_c4.py")

OLD_HEAD = '''            # -------- row pipeline: RC owned rows + 3 halo rows ------------------
            for i in T.serial(rc + 3):
                r = t0 - 3 + i
                if r >= 0:  # block level causal guard: a missing halo row stays zero
                    T.copy(H[b, r, :], h_ub[:d])
                    T.copy(K[b, r, :], k_ub[:d])
                    T.copy(V[b, r, :], v_ub[:d])
                with T.SimdVF():
'''

NEW_HEAD = '''            # -------- row pipeline: RC owned rows + 3 halo rows ------------------
            # Row-level software pipeline, expressed explicitly: two row-buffer
            # sets, and the DMA of row i+1 is issued before the vector compute of
            # row i, so MTE2 streams the next row while the vector core works on
            # the current one.  The row loop is unrolled at build time (rc + 3 is
            # a build-time constant) so each iteration's buffer set is static --
            # `T.Pipelined(num_stages=2)` does not rotate these buffers (measured
            # as branch r1_c1: identical emitted code, no version offsets).
            #
            # A SIMD_VF frame may not read GM, so every T.copy stays outside the
            # frame; that is exactly the interleaving point of the two stages.
            r0 = t0 - 3
            if r0 >= 0:  # block level causal guard: a missing halo row stays zero
                T.copy(H[b, r0, :], h_ub[0, :d])
                T.copy(K[b, r0, :], k_ub[0, :d])
                T.copy(V[b, r0, :], v_ub[0, :d])
            for i in range(rc + 3):  # build-time unroll (static buffer set per row)
                r = t0 - 3 + i
                cur = i % 2
                if i + 1 < rc + 3:
                    rn = r + 1
                    if rn >= 0:  # prefetch of the next row, issued before this row
                        T.copy(H[b, rn, :], h_ub[(i + 1) % 2, :d])
                        T.copy(K[b, rn, :], k_ub[(i + 1) % 2, :d])
                        T.copy(V[b, rn, :], v_ub[(i + 1) % 2, :d])
                with T.SimdVF():
'''

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

    sub(OLD_HEAD, NEW_HEAD)
    sub(OLD_RVB, NEW_RVB)
    sub(OLD_TAIL, NEW_TAIL)
    # per-row buffer set selection in the compute body (identifier-boundary
    # matches: `h_ub[` is a substring of `vh_ub[`)
    import re

    def sub_re(pat, new):
        nonlocal text
        text, n = re.subn(pat, new, text)
        if n == 0:
            raise SystemExit(f"no match: {pat}")
        print(f"  {pat} -> {new} x{n}")

    sub_re(r"vh_ub\[ch \* VL\]", "vh_ub[i, ch * VL]")
    sub_re(r"y_ub\[ch \* VL\]", "y_ub[i, ch * VL]")
    for buf in ("h_ub", "k_ub", "v_ub"):
        sub_re(rf"(?<![A-Za-z0-9_]){buf}\[ch \* VL\]", f"{buf}[cur, ch * VL]")

    with open(PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print("patched", PATH, len(text.splitlines()), "lines")


if __name__ == "__main__":
    main()
