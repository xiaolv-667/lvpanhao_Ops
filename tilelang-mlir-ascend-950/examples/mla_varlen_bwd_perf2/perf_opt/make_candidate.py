"""Build a candidate of MLAVarlenBwdOp.py with a **query-chunked GM fill**.

The Stage 4 round-1 winner is `num_stages=3` on a loop that fills exactly one
64-row query block per iteration.  This round attacks the residual that round left
open: the per-iteration price grows with the request length (3.14 us/iteration at
len<=5376 vs 4.26 at len=16384, same AIC pipe work) because a long request
re-reads its whole q/do suffix once per key block and every fill pays a DRAM round
trip that the pipeline cannot hide.  Filling `_QCHUNK` query blocks per descriptor
keeps the GEMM tile (block_n x block_m), the task mapping, the grid and the slab
exactly as they are; it only makes each fill cover more query rows, so the number
of fill round trips per query row drops by `_QCHUNK`.

Usage:
    python make_candidate.py perf_opt/MLAVarlenBwdOp.py <dst> <variant>
"""

import sys
from pathlib import Path

REGION_START = "                    loop_ed = T.max(T.ceildiv(len_q, _BM), loop_st + 1)"
REGION_END = "                        T.set_atomic_none()"

# ---------------------------------------------------------------------------
# The query loop, rewritten for a chunked fill.  `{chunk}` is the number of
# 64-row query blocks per fill, `{stages}` the pipeline depth of the chunk loop,
# `{nt}` selects the natural (K,N) B operand for dV/dK/dQ (drops qT/doT/kT and
# halves the fill bytes) instead of the transposed L1 copies.
# ---------------------------------------------------------------------------
LOOP = '''                    loop_ed = T.max(T.ceildiv(len_q, _BM), loop_st + 1)
                    # ---- query-chunk loop: one GM fill per {chunk} 64-row blocks ----
                    # The fill granularity is what this round changes: a long request
                    # re-reads its q/do suffix once per key block, and the per-fill
                    # DRAM round trip is what makes the per-iteration price grow with
                    # the request length.  `_QCHUNK` query blocks share one descriptor
                    # (and one lse/delta run), so the round-trip count per query row
                    # drops by `_QCHUNK`; the GEMM tile, the task mapping, the grid and
                    # the slab are untouched.
                    cst = loop_st // _QCHUNK
                    ced = T.max(T.ceildiv(loop_ed, _QCHUNK), cst + 1)
                    for kqc in T.Pipelined(cst, ced, num_stages=_STAGES):
                        m0 = kqc * _MF
                        qoff = q_start + m0
                        T.copy(q[qoff : qoff + _MF, h, :], q_s)
                        T.copy(dout[qoff : qoff + _MF, h, :], do_s)
{transposed_fill}                        T.copy(lse[h, qoff : qoff + _MF], lse_u)  # head-major run
                        T.copy(delta[h, qoff : qoff + _MF], dlt_u)
                        for sub in range(_QCHUNK):
                            mb = sub * _BM
                            m1 = m0 + mb
                            qo = qoff + mb
                            clr = (kqc == cst) if sub == 0 else False
{body}'''

BODY_TRANSPOSED = '''                            # ---- s^T and dP^T: two L0 GEMMs, cleared every sub-step ----
                            T.copy(k_s, a_k)
                            T.copy(q_s[mb : mb + _BM, :], b_lo)
                            T.gemm(a_k, b_lo, st_c, transpose_B=True, clear_accum=True)
                            T.copy(v_s, a_k[0:_BN, 0:dim_v])
                            T.copy(do_s[mb : mb + _BM, :], b_lo[0:_BM, 0:dim_v])
                            T.gemm(a_k[0:_BN, 0:dim_v], b_lo[0:_BM, 0:dim_v], dpt_c,
                                   transpose_B=True, clear_accum=True)
                            # L0C -> UB: each AIV receives its own 48 key rows.
                            T.dual_copy(st_c, st_u)
                            T.dual_copy(dpt_c, dp_u)

                            fast = (n0 + _BN <= len_kv) & (m1 + _BM <= len_q)
                            if is_causal:
                                fast = fast & (n0 + _BN - 1 <= m1 + shift)
                            if a2_fast:
                                if fast:
                                    # Fast arm: the whole tile is visible and in range.
                                    with T.SimtVF(threads=128):
                                        for i, j in T.Parallel(_HN, _BM):
                                            p = T.exp(st_u[i, j] * scale - lse_u[mb + j])
                                            p16_u[i, j] = T.cast(p, dt)
                                            dsT_u[i, j] = T.cast(
                                                p * (dp_u[i, j] - dlt_u[mb + j]), dt)
                                else:
                                    with T.SimtVF(threads=128):
                                        for i, j in T.Parallel(_HN, _BM):
                                            gi = n0 + T.cast(sid, "int32") * _HN + i
                                            gj = m1 + j
                                            vis = (gi < len_kv) & (gj < len_q)
                                            if is_causal:
                                                vis = vis & (gi <= gj + shift)
                                            p = T.exp(st_u[i, j] * scale - lse_u[mb + j])
                                            pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                                            p16_u[i, j] = T.cast(pv, dt)
                                            dsT_u[i, j] = T.cast(
                                                pv * (dp_u[i, j] - dlt_u[mb + j]), dt)
                            else:
                                with T.SimtVF(threads=128):
                                    for i, j in T.Parallel(_HN, _BM):
                                        gi = n0 + T.cast(sid, "int32") * _HN + i
                                        gj = m1 + j
                                        vis = (gi < len_kv) & (gj < len_q)
                                        if is_causal:
                                            vis = vis & (gi <= gj + shift)
                                        p = T.exp(st_u[i, j] * scale - lse_u[mb + j])
                                        pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                                        p16_u[i, j] = T.cast(pv, dt)
                                        dsT_u[i, j] = T.cast(
                                            pv * (dp_u[i, j] - dlt_u[mb + j]), dt)

                            # UB -> L1: M-split of the two produced half tiles.
                            T.dual_copy(p16_u, pT_s)
                            T.dual_copy(dsT_u, dsT_s)
                            # ---- dV / dK (accumulate) and dQ (transpose_A reuse of dsT) ---
                            T.copy(pT_s, a_k[0:_BN, 0:_BM])
                            T.copy(doT_s[0:dim_v, mb : mb + _BM], b_hi[0:dim_v, 0:_BM])
                            T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:dim_v, 0:_BM], dv_c,
                                   transpose_B=True, clear_accum=clr)
                            T.copy(dsT_s, a_k[0:_BN, 0:_BM])
                            T.copy(qT_s[0:dim_qk, mb : mb + _BM], b_hi[0:dim_qk, 0:_BM])
                            T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:dim_qk, 0:_BM], dk_c,
                                   transpose_B=True, clear_accum=clr)
                            # Same dsT tile as dK's A operand; the transpose is done by
                            # the L1 -> L0A transposing copy (MTE1), not by a vector
                            # scatter (v3's 63% hotspot, DESIGN.md 0.6 R3).
                            T.copy(dsT_s, a_t)
                            # kT must be reloaded into b_hi AFTER qT/doT (DESIGN.md 9.2).
                            T.copy(kT_s, b_hi)
                            T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=True,
                                   clear_accum=True)
                            # dQ accumulates across cores -> hardware store-atomic into
                            # fp32.  Rows/columns past this request carry exactly 0
                            # (DESIGN.md 1.6.1 A3), so the neighbour request's dq_acc
                            # rows are left untouched; the tensor end is clamped by the
                            # platform.
                            T.set_atomic("add", "float32")
                            T.copy(dq_c, dq_acc[qo : qo + _BM, h, :])
                            T.set_atomic_none()'''

BODY_NATURAL = '''                            # ---- s^T and dP^T: two L0 GEMMs, cleared every sub-step ----
                            T.copy(k_s, a_k)
                            T.copy(q_s[mb : mb + _BM, :], b_lo)
                            T.gemm(a_k, b_lo, st_c, transpose_B=True, clear_accum=True)
                            T.copy(v_s, a_k[0:_BN, 0:dim_v])
                            T.copy(do_s[mb : mb + _BM, :], b_lo[0:_BM, 0:dim_v])
                            T.gemm(a_k[0:_BN, 0:dim_v], b_lo[0:_BM, 0:dim_v], dpt_c,
                                   transpose_B=True, clear_accum=True)
                            # L0C -> UB: each AIV receives its own 48 key rows.
                            T.dual_copy(st_c, st_u)
                            T.dual_copy(dpt_c, dp_u)

                            fast = (n0 + _BN <= len_kv) & (m1 + _BM <= len_q)
                            if is_causal:
                                fast = fast & (n0 + _BN - 1 <= m1 + shift)
                            if a2_fast:
                                if fast:
                                    # Fast arm: the whole tile is visible and in range.
                                    with T.SimtVF(threads=128):
                                        for i, j in T.Parallel(_HN, _BM):
                                            p = T.exp(st_u[i, j] * scale - lse_u[mb + j])
                                            p16_u[i, j] = T.cast(p, dt)
                                            dsT_u[i, j] = T.cast(
                                                p * (dp_u[i, j] - dlt_u[mb + j]), dt)
                                else:
                                    with T.SimtVF(threads=128):
                                        for i, j in T.Parallel(_HN, _BM):
                                            gi = n0 + T.cast(sid, "int32") * _HN + i
                                            gj = m1 + j
                                            vis = (gi < len_kv) & (gj < len_q)
                                            if is_causal:
                                                vis = vis & (gi <= gj + shift)
                                            p = T.exp(st_u[i, j] * scale - lse_u[mb + j])
                                            pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                                            p16_u[i, j] = T.cast(pv, dt)
                                            dsT_u[i, j] = T.cast(
                                                pv * (dp_u[i, j] - dlt_u[mb + j]), dt)
                            else:
                                with T.SimtVF(threads=128):
                                    for i, j in T.Parallel(_HN, _BM):
                                        gi = n0 + T.cast(sid, "int32") * _HN + i
                                        gj = m1 + j
                                        vis = (gi < len_kv) & (gj < len_q)
                                        if is_causal:
                                            vis = vis & (gi <= gj + shift)
                                        p = T.exp(st_u[i, j] * scale - lse_u[mb + j])
                                        pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                                        p16_u[i, j] = T.cast(pv, dt)
                                        dsT_u[i, j] = T.cast(
                                            pv * (dp_u[i, j] - dlt_u[mb + j]), dt)

                            # UB -> L1: M-split of the two produced half tiles.
                            T.dual_copy(p16_u, pT_s)
                            T.dual_copy(dsT_u, dsT_s)
                            # ---- dV / dK (accumulate) and dQ (transpose_A reuse of dsT) ---
                            T.copy(pT_s, a_k[0:_BN, 0:_BM])
                            T.copy(do_s[mb : mb + _BM, 0:dim_v], b_hi[0:_BM, 0:dim_v])
                            T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:_BM, 0:dim_v], dv_c,
                                   transpose_B=False, clear_accum=clr)
                            T.copy(dsT_s, a_k[0:_BN, 0:_BM])
                            T.copy(q_s[mb : mb + _BM, :], b_hi[0:_BM, 0:dim_qk])
                            T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:_BM, 0:dim_qk], dk_c,
                                   transpose_B=False, clear_accum=clr)
                            # Same dsT tile as dK's A operand; the transpose is done by
                            # the L1 -> L0A transposing copy (MTE1), not by a vector
                            # scatter (v3's 63% hotspot, DESIGN.md 0.6 R3).
                            T.copy(dsT_s, a_t)
                            T.copy(k_s, b_hi)
                            T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=False,
                                   clear_accum=True)
                            # dQ accumulates across cores -> hardware store-atomic into
                            # fp32.  Rows/columns past this request carry exactly 0
                            # (DESIGN.md 1.6.1 A3), so the neighbour request's dq_acc
                            # rows are left untouched; the tensor end is clamped by the
                            # platform.
                            T.set_atomic("add", "float32")
                            T.copy(dq_c, dq_acc[qo : qo + _BM, h, :])
                            T.set_atomic_none()'''

TR_FILL = ("                        T.copy(q[qoff : qoff + _MF, h, :], qT_s, transpose=True)\n"
           "                        T.copy(dout[qoff : qoff + _MF, h, :], doT_s, transpose=True)\n")

ALLOC_NEW = """            k_s = T.alloc_l1((_BN, dim_qk), dt)
            v_s = T.alloc_l1((_BN, dim_v), dt)
            kT_s = T.alloc_l1((dim_qk, _BN), dt)
            q_s = T.alloc_l1((_MF, dim_qk), dt)
            qT_s = T.alloc_l1((dim_qk, _MF), dt)
            do_s = T.alloc_l1((_MF, dim_v), dt)
            doT_s = T.alloc_l1((dim_v, _MF), dt)"""
ALLOC_OLD = """            k_s = T.alloc_l1((_BN, dim_qk), dt)
            v_s = T.alloc_l1((_BN, dim_v), dt)
            kT_s = T.alloc_l1((dim_qk, _BN), dt)
            q_s = T.alloc_l1((_BM, dim_qk), dt)
            qT_s = T.alloc_l1((dim_qk, _BM), dt)
            do_s = T.alloc_l1((_BM, dim_v), dt)
            doT_s = T.alloc_l1((dim_v, _BM), dt)"""

ALLOC_NATURAL_OLD = """            b_hi = T.alloc_l0b((dim_qk, _BN), dt)  # qT / doT / kT (kT reloaded last)"""
ALLOC_NATURAL_NEW = """            b_hi = T.alloc_l0b((_BN, dim_qk), dt)  # dV/dK/dQ (K, N) operand"""


def build(text, chunk, stages, natural):
    if chunk == 1:
        raise SystemExit("chunk=1 is the frozen design; nothing to build")
    # 1. module constants
    old = f"_STAGES = {stages if not natural else 3}"
    anchor = "_STAGES = 3  # DESIGN.md 3.3 / 6.3: num_stages of the query loop"
    assert anchor in text, "STAGES anchor"
    text = text.replace(
        anchor,
        f"_STAGES = {stages}  # query-chunk loop pipeline depth (Round 2 sweep)\n"
        f"_QCHUNK = {chunk}  # 64-row query blocks per GM fill (1 = frozen design)\n"
        f"_MF = _BM * _QCHUNK  # query rows per fill",
    )
    # 2. L1 buffers / lse / delta sized by the fill
    assert ALLOC_OLD in text, "alloc anchor"
    text = text.replace(ALLOC_OLD, ALLOC_NEW)
    text = text.replace('lse_u = T.alloc_shared((_BM,), "float32")',
                        'lse_u = T.alloc_shared((_MF,), "float32")')
    text = text.replace('dlt_u = T.alloc_shared((_BM,), "float32")',
                        'dlt_u = T.alloc_shared((_MF,), "float32")')
    if natural:
        assert ALLOC_NATURAL_OLD in text
        text = text.replace(ALLOC_NATURAL_OLD, ALLOC_NATURAL_NEW)
        # kT is no longer materialised
        text = text.replace(
            "                    T.copy(k[koff : koff + _BN, h, :], kT_s, transpose=True)\n",
            "")
        text = text.replace("            kT_s = T.alloc_l1((dim_qk, _BN), dt)\n", "")
    # 3. the loop region
    start = text.index(REGION_START)
    end = text.index(REGION_END, start) + len(REGION_END)
    body = BODY_NATURAL if natural else BODY_TRANSPOSED
    loop = LOOP.format(chunk=chunk, body=body,
                       transposed_fill="" if natural else TR_FILL)
    text = text[:start] + loop + text[end:]
    return text


def build_slab(text, slab):
    """Only the launch plan's preferred head slab changes (DESIGN.md 5.6 b)."""
    anchor = "HEAD_SLAB_DEFAULT = 32  # DESIGN.md 5.6 (b): at most 4 launches for the manifest"
    assert anchor in text, "slab anchor"
    return text.replace(
        anchor,
        f"HEAD_SLAB_DEFAULT = {slab}  # Round 2: bounded by the L2 working set, see opt_log")


VARIANTS = {
    "s4": ("slab", 4),
    "s8": ("slab", 8),
    "s16": ("slab", 16),
    # chunk, stages, natural-operands
    "q2s2": (2, 2, False),
    "q2s3nt": (2, 3, True),
    "q2s2nt": (2, 2, True),
    "q4s2nt": (4, 2, True),
}


def main():
    src, dst, variant = sys.argv[1], sys.argv[2], sys.argv[3]
    spec = VARIANTS[variant]
    if spec[0] == "slab":
        text = build_slab(Path(src).read_text(), spec[1])
        Path(dst).write_text(text)
        print(f"[{variant}] {src} -> {dst}: HEAD_SLAB_DEFAULT={spec[1]}")
        return
    chunk, stages, natural = spec
    text = build(Path(src).read_text(), chunk, stages, natural)
    Path(dst).write_text(text)
    print(f"[{variant}] {src} -> {dst}: chunk={chunk} stages={stages} natural={natural}")


if __name__ == "__main__":
    main()
