"""Build one candidate variant of MLADenseBwdOp.py from an explicit edit list.

Each candidate applies **one** main optimisation point to the current best file,
so a measured difference is attributable.  Usage:

    python perf_opt/make_candidate.py <src> <dst> <variant>

The epilogue-split branch (round-1 c5) is a larger multi-anchor rewrite and lives
in its own generator, ``perf_opt/make_c5.py`` (same src/dst convention); the
``block_n`` prototypes are a one-constant change on the baseline
(``_BN = 96`` -> 80 / 128, see opt_log.md 2.2).
"""

import sys
from pathlib import Path

# --- c1: drop the transposed L1 operands (qT/doT/kT) --------------------------
# The three transposed GM loads exist only because dV/dK/dQ were written with
# `transpose_B=True` (B stored as (N,K)).  The same GEMMs accept the natural
# (K,N) operand (`transpose_B=False`) from q_s/do_s/k_s, which removes
# 65,536 B/iteration of GM->L1 fill and 36,864 B/block of the k/v prologue.
NT_EDITS = [
    ("""            kT_s = T.alloc_l1((dim_qk, _BN), dt)
""", ""),
    ("""            qT_s = T.alloc_l1((dim_qk, _BM), dt)
""", ""),
    ("""            doT_s = T.alloc_l1((dim_v, _BM), dt)
""", ""),
    ("""            b_hi = T.alloc_l0b((dim_qk, _BN), dt)  # qT / doT / kT (kT reloaded last)""",
     """            b_hi = T.alloc_l0b((_BN, dim_qk), dt)  # dV/dK/dQ (K, N) operand"""),
    ("""            T.copy(k[b, n0 : n0 + _BN, h, :], k_s)
            T.copy(v[b, n0 : n0 + _BN, h, :], v_s)
            T.copy(k[b, n0 : n0 + _BN, h, :], kT_s, transpose=True)""",
     """            T.copy(k[b, n0 : n0 + _BN, h, :], k_s)
            T.copy(v[b, n0 : n0 + _BN, h, :], v_s)"""),
    ("""                T.copy(q[b, m0 : m0 + _BM, h, :], q_s)
                T.copy(dout[b, m0 : m0 + _BM, h, :], do_s)
                T.copy(q[b, m0 : m0 + _BM, h, :], qT_s, transpose=True)
                T.copy(dout[b, m0 : m0 + _BM, h, :], doT_s, transpose=True)""",
     """                T.copy(q[b, m0 : m0 + _BM, h, :], q_s)
                T.copy(dout[b, m0 : m0 + _BM, h, :], do_s)"""),
    ("""                T.copy(pT_s, a_k[0:_BN, 0:_BM])
                T.copy(doT_s, b_hi[0:dim_v, 0:_BM])
                T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:dim_v, 0:_BM], dv_c,
                       transpose_B=True, clear_accum=(kq == loop_st))""",
     """                T.copy(pT_s, a_k[0:_BN, 0:_BM])
                T.copy(do_s, b_hi[0:_BM, 0:dim_v])
                T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:_BM, 0:dim_v], dv_c,
                       transpose_B=False, clear_accum=(kq == loop_st))"""),
    ("""                T.copy(dsT_s, a_k[0:_BN, 0:_BM])
                T.copy(qT_s, b_hi[0:dim_qk, 0:_BM])
                T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:dim_qk, 0:_BM], dk_c,
                       transpose_B=True, clear_accum=(kq == loop_st))""",
     """                T.copy(dsT_s, a_k[0:_BN, 0:_BM])
                T.copy(q_s, b_hi[0:_BM, 0:dim_qk])
                T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:_BM, 0:dim_qk], dk_c,
                       transpose_B=False, clear_accum=(kq == loop_st))"""),
    ("""                T.copy(dsT_s, a_t)
                # kT must be reloaded into b_hi AFTER qT/doT (DESIGN.md 9.2).
                T.copy(kT_s, b_hi)
                T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=True, clear_accum=True)""",
     """                T.copy(dsT_s, a_t)
                T.copy(k_s, b_hi)
                T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=False,
                       clear_accum=True)"""),
]


# --- generic: num_stages sweep ------------------------------------------------
def stages_edits(stages):
    return [("""_STAGES = 2  # DESIGN.md 3.3 / 6.3: two versions per per-iteration buffer""",
             f"""_STAGES = {stages}  # DESIGN.md 3.3 / 6.3: num_stages of the query loop""")]


VARIANTS = {
    "c1_stages3": stages_edits(3),
    "c2_stages4": stages_edits(4),
    "c3_nt_operands": NT_EDITS,
    "c4_stages3_nt": NT_EDITS + stages_edits(3),
    "c5_stages4_nt": NT_EDITS + stages_edits(4),
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
