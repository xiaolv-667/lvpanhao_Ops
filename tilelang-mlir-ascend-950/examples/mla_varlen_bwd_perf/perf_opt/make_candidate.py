"""Build one candidate variant of MLAVarlenBwdOp.py from an explicit edit list.

Each candidate applies **one** main optimisation point to the current best file,
so a measured difference is attributable.  Usage:

    python perf_opt/make_candidate.py <src> <dst> <variant>
"""

import sys
from pathlib import Path

# --- c1: drop the transposed L1 operands (qT/doT/kT) --------------------------
# The three transposed GM loads exist only because dV/dK/dQ were written with
# `transpose_B=True` (B stored as (N,K)).  The same GEMMs accept the natural
# (K,N) operand (`transpose_B=False`) from q_s/do_s/k_s, which removes
# 40,960 B/iteration of GM->L1 fill (50% of the per-iteration fill) and
# 36,864 B/block of the k/v prologue, at an unchanged L1->L0 byte count.
C1_EDITS = [
    # L1 buffers: qT_s / doT_s / kT_s are gone; b_hi becomes the (K,N) dQ operand.
    ("""            kT_s = T.alloc_l1((dim_qk, _BN), dt)
            q_s = T.alloc_l1((_BM, dim_qk), dt)
            qT_s = T.alloc_l1((dim_qk, _BM), dt)
            do_s = T.alloc_l1((_BM, dim_v), dt)
            doT_s = T.alloc_l1((dim_v, _BM), dt)""",
     """            q_s = T.alloc_l1((_BM, dim_qk), dt)
            do_s = T.alloc_l1((_BM, dim_v), dt)"""),
    ("""            b_hi = T.alloc_l0b((dim_qk, _BN), dt)  # qT / doT / kT (kT reloaded last)""",
     """            b_hi = T.alloc_l0b((_BN, dim_qk), dt)  # dV/dK/dQ (K, N) operand"""),
    # Block prologue: kT is not materialised any more.
    ("""                    T.copy(k[koff : koff + _BN, h, :], k_s)
                    T.copy(v[koff : koff + _BN, h, :], v_s)
                    T.copy(k[koff : koff + _BN, h, :], kT_s, transpose=True)""",
     """                    T.copy(k[koff : koff + _BN, h, :], k_s)
                    T.copy(v[koff : koff + _BN, h, :], v_s)"""),
    # Per-iteration fill: only the natural-layout q/do copies remain.
    ("""                        T.copy(q[qoff : qoff + _BM, h, :], q_s)
                        T.copy(dout[qoff : qoff + _BM, h, :], do_s)
                        T.copy(q[qoff : qoff + _BM, h, :], qT_s, transpose=True)
                        T.copy(dout[qoff : qoff + _BM, h, :], doT_s, transpose=True)""",
     """                        T.copy(q[qoff : qoff + _BM, h, :], q_s)
                        T.copy(dout[qoff : qoff + _BM, h, :], do_s)"""),
    # dV: B = do (K=64, N=D_v) from b_lo, natural layout.
    ("""                        T.copy(pT_s, a_k[0:_BN, 0:_BM])
                        T.copy(doT_s, b_hi[0:dim_v, 0:_BM])
                        T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:dim_v, 0:_BM], dv_c,
                               transpose_B=True, clear_accum=(kq == loop_st))""",
     """                        T.copy(pT_s, a_k[0:_BN, 0:_BM])
                        T.copy(do_s, b_hi[0:_BM, 0:dim_v])
                        T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:_BM, 0:dim_v], dv_c,
                               transpose_B=False, clear_accum=(kq == loop_st))"""),
    # dK: B = q (K=64, N=D_qk) from b_lo, natural layout.
    ("""                        T.copy(dsT_s, a_k[0:_BN, 0:_BM])
                        T.copy(qT_s, b_hi[0:dim_qk, 0:_BM])
                        T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:dim_qk, 0:_BM], dk_c,
                               transpose_B=True, clear_accum=(kq == loop_st))""",
     """                        T.copy(dsT_s, a_k[0:_BN, 0:_BM])
                        T.copy(q_s, b_hi[0:_BM, 0:dim_qk])
                        T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:_BM, 0:dim_qk], dk_c,
                               transpose_B=False, clear_accum=(kq == loop_st))"""),
    # dQ: A stays the transposed half tile, B = k (K=_BN, N=D_qk) from k_s.
    ("""                        T.copy(dsT_s, a_t)
                        # kT must be reloaded into b_hi AFTER qT/doT (DESIGN.md 9.2).
                        T.copy(kT_s, b_hi)
                        T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=True,
                               clear_accum=True)""",
     """                        T.copy(dsT_s, a_t)
                        T.copy(k_s, b_hi)
                        T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=False,
                               clear_accum=True)"""),
]

# --- generic: num_stages sweep ------------------------------------------------
def stages_edits(stages):
    return [("""_STAGES = 2  # DESIGN.md 3.3 / 6.3: num_stages of the query loop""",
             f"""_STAGES = {stages}  # DESIGN.md 3.3 / 6.3: num_stages of the query loop""")]


VARIANTS = {
    "c1_nt_operands": C1_EDITS,
    "c2_stages1": stages_edits(1),
    "c3_stages3": stages_edits(3),
    "c4_stages4": stages_edits(4),
    "c5_stages3_nt": C1_EDITS + stages_edits(3),
    "c6_stages4_nt": C1_EDITS + stages_edits(4),
    "c7_stages5_nt": C1_EDITS + stages_edits(5),
}


def main():
    src, dst, variant = sys.argv[1], sys.argv[2], sys.argv[3]
    edits = VARIANTS[variant]
    text = Path(src).read_text()
    for old, new in edits:
        if old not in text:
            raise SystemExit(f"[{variant}] anchor not found:\n{old[:200]}")
        if text.count(old) != 1:
            raise SystemExit(f"[{variant}] anchor not unique ({text.count(old)}):\n{old[:120]}")
        text = text.replace(old, new)
    Path(dst).write_text(text)
    print(f"[{variant}] {src} -> {dst}: {len(edits)} edit(s)")


if __name__ == "__main__":
    main()
