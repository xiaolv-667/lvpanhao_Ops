"""Build a candidate of the A5 (v5) MLAVarlenBwdOp.py.

Variants (each changes **one** main point relative to the v5 baseline):

* ``ubv2``  - double-version ``st_u``/``dp_u`` per half body (the UB half tiles
  are the only buffers still shared between the two half bodies of the A5 loop, so
  the epilogue of half B waits for half A's ``_scores`` WAR; +24,576 B UB).
* ``nt``    - dV/dK/dQ take the natural (K,N) B operand from ``q_s/do_s/k_s``
  instead of the transposed ``qT/doT/kT`` L1 copies: the per-half-body GM fill
  drops from 6 to 4 descriptors and from 82,432 to 41,472 B (MTE2 is the v5
  critical AIC pipe at 81-85% busy), and four L1 buffers plus ``kT_s`` go away
  (L1 -118,784 B).  ``dk``/``dv`` stay bitwise identical (verified in v4).
* ``ubv2_nt`` - both of the above.
"""

import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# ubv2: st_u / dp_u get one version per half body
# --------------------------------------------------------------------------- #
UBV2 = [
    ("""            st_u = T.alloc_shared((_HN, _BM), "float32")
            dp_u = T.alloc_shared((_HN, _BM), "float32")""",
     """            # One version per half body (Stage 4 round 3): a single ``st_u/dp_u``
            # makes `_scores(B)` wait for `_epilogue(A)`'s read (a WAR inside the
            # loop), which serialises the two half bodies this A5 loop exists to
            # overlap.
            st_u0 = T.alloc_shared((_HN, _BM), "float32")
            dp_u0 = T.alloc_shared((_HN, _BM), "float32")
            st_u1 = T.alloc_shared((_HN, _BM), "float32")
            dp_u1 = T.alloc_shared((_HN, _BM), "float32")"""),
    # half body A
    ("""                        _scores(k_s, v_s, q_s0, do_s0, a_k, b_lo, st_c, dpt_c,
                                st_u, dp_u)
                        _epilogue(sid, st_u, dp_u, p16_u, dsT_u, lse_u0, dlt_u0,
                                  pT_s0, dsT_s0, n0, m0_a, len_q, len_kv, shift)""",
     """                        _scores(k_s, v_s, q_s0, do_s0, a_k, b_lo, st_c, dpt_c,
                                st_u0, dp_u0)
                        _epilogue(sid, st_u0, dp_u0, p16_u, dsT_u, lse_u0, dlt_u0,
                                  pT_s0, dsT_s0, n0, m0_a, len_q, len_kv, shift)"""),
    # half body B
    ("""                            _scores(k_s, v_s, q_s1, do_s1, a_k, b_lo, st_c, dpt_c,
                                    st_u, dp_u)
                            _epilogue(sid, st_u, dp_u, p16_u, dsT_u, lse_u1, dlt_u1,
                                      pT_s1, dsT_s1, n0, m0_b, len_q, len_kv, shift)""",
     """                            _scores(k_s, v_s, q_s1, do_s1, a_k, b_lo, st_c, dpt_c,
                                    st_u1, dp_u1)
                            _epilogue(sid, st_u1, dp_u1, p16_u, dsT_u, lse_u1, dlt_u1,
                                      pT_s1, dsT_s1, n0, m0_b, len_q, len_kv, shift)"""),
]

# --------------------------------------------------------------------------- #
# nt: natural (K,N) B operands for dV / dK / dQ
# --------------------------------------------------------------------------- #
NT = [
    # dQ's B operand is k_s as (K=_BN, N=dim_qk); dV/dK use its (_BM, dim_qk) region.
    ("""            b_hi = T.alloc_l0b((dim_qk, _BN), dt)  # qT / doT / kT (kT reloaded last)""",
     """            b_hi = T.alloc_l0b((_BN, dim_qk), dt)  # dV/dK/dQ natural (K, N) operand"""),
    # the two transposed GM fills per half body are gone
    ("""        T.copy(q[qoff : qoff + _BM, h, :], q_s)
        T.copy(dout[qoff : qoff + _BM, h, :], do_s)
        T.copy(q[qoff : qoff + _BM, h, :], qT_s, transpose=True)
        T.copy(dout[qoff : qoff + _BM, h, :], doT_s, transpose=True)""",
     """        T.copy(q[qoff : qoff + _BM, h, :], q_s)
        T.copy(dout[qoff : qoff + _BM, h, :], do_s)"""),
    # no L1 transposed copies, no kT
    ("""            q_s0 = T.alloc_l1((_BM, dim_qk), dt)
            qT_s0 = T.alloc_l1((dim_qk, _BM), dt)
            do_s0 = T.alloc_l1((_BM, dim_v), dt)
            doT_s0 = T.alloc_l1((dim_v, _BM), dt)""",
     """            q_s0 = T.alloc_l1((_BM, dim_qk), dt)
            do_s0 = T.alloc_l1((_BM, dim_v), dt)"""),
    ("""            q_s1 = T.alloc_l1((_BM, dim_qk), dt)
            qT_s1 = T.alloc_l1((dim_qk, _BM), dt)
            do_s1 = T.alloc_l1((_BM, dim_v), dt)
            doT_s1 = T.alloc_l1((dim_v, _BM), dt)""",
     """            q_s1 = T.alloc_l1((_BM, dim_qk), dt)
            do_s1 = T.alloc_l1((_BM, dim_v), dt)"""),
    ("""            kT_s = T.alloc_l1((dim_qk, _BN), dt)
""", ""),
    ("""                    T.copy(k[koff : koff + _BN, h, :], kT_s, transpose=True)
""", ""),
    # _fill signature loses qT_s / doT_s
    ("""    def _fill(q, dout, lse, delta, q_s, do_s, qT_s, doT_s, lse_u, dlt_u, h, qoff):""",
     """    def _fill(q, dout, lse, delta, q_s, do_s, lse_u, dlt_u, h, qoff):"""),
    ("""                        _fill(q, dout, lse, delta, q_s0, do_s0, qT_s0, doT_s0,
                              lse_u0, dlt_u0, h, qoff_a)""",
     """                        _fill(q, dout, lse, delta, q_s0, do_s0,
                              lse_u0, dlt_u0, h, qoff_a)"""),
    ("""                            _fill(q, dout, lse, delta, q_s1, do_s1, qT_s1, doT_s1,
                                  lse_u1, dlt_u1, h, qoff_b)""",
     """                            _fill(q, dout, lse, delta, q_s1, do_s1,
                                  lse_u1, dlt_u1, h, qoff_b)"""),
    # _gradients: natural operands
    ("""    def _gradients(pT_p, dsT_p, doT_p, qT_p, kT_s, a_k, a_t, b_hi, dq_c, dk_c, dv_c,
                   dq_acc, qoff_p, h, clear_prev):""",
     """    def _gradients(pT_p, dsT_p, do_p, q_p, k_s, a_k, a_t, b_hi, dq_c, dk_c, dv_c,
                   dq_acc, qoff_p, h, clear_prev):"""),
    ("""        T.copy(pT_p, a_k[0:_BN, 0:_BM])
        T.copy(doT_p, b_hi[0:dim_v, 0:_BM])
        T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:dim_v, 0:_BM], dv_c, transpose_B=True,
               clear_accum=clear_prev)
        T.copy(dsT_p, a_k[0:_BN, 0:_BM])
        T.copy(qT_p, b_hi[0:dim_qk, 0:_BM])
        T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:dim_qk, 0:_BM], dk_c, transpose_B=True,
               clear_accum=clear_prev)""",
     """        T.copy(pT_p, a_k[0:_BN, 0:_BM])
        T.copy(do_p[0:_BM, 0:dim_v], b_hi[0:_BM, 0:dim_v])
        T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:_BM, 0:dim_v], dv_c, transpose_B=False,
               clear_accum=clear_prev)
        T.copy(dsT_p, a_k[0:_BN, 0:_BM])
        T.copy(q_p, b_hi[0:_BM, 0:dim_qk])
        T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:_BM, 0:dim_qk], dk_c, transpose_B=False,
               clear_accum=clear_prev)"""),
    ("""        T.copy(dsT_p, a_t)
        # kT must be reloaded into b_hi AFTER qT/doT (DESIGN.md 9.2).
        T.copy(kT_s, b_hi)
        T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=True, clear_accum=True)""",
     """        T.copy(dsT_p, a_t)
        T.copy(k_s, b_hi)
        T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=False, clear_accum=True)"""),
    # the four _gradients call sites pass do/q versions instead of doT/qT/kT
    ("""                            _gradients(pT_s1, dsT_s1, doT_s1, qT_s1, kT_s, a_k, a_t,
                                       b_hi, dq_c, dk_c, dv_c, dq_acc,
                                       qoff_a - _BM, h, kq_a - 1 == loop_st)""",
     """                            _gradients(pT_s1, dsT_s1, do_s1, q_s1, k_s, a_k, a_t,
                                       b_hi, dq_c, dk_c, dv_c, dq_acc,
                                       qoff_a - _BM, h, kq_a - 1 == loop_st)"""),
    ("""                            _gradients(pT_s0, dsT_s0, doT_s0, qT_s0, kT_s, a_k, a_t,
                                       b_hi, dq_c, dk_c, dv_c, dq_acc,
                                       qoff_b - _BM, h, kq_b - 1 == loop_st)""",
     """                            _gradients(pT_s0, dsT_s0, do_s0, q_s0, k_s, a_k, a_t,
                                       b_hi, dq_c, dk_c, dv_c, dq_acc,
                                       qoff_b - _BM, h, kq_b - 1 == loop_st)"""),
    ("""                        _gradients(pT_s0, dsT_s0, doT_s0, qT_s0, kT_s, a_k, a_t,
                                   b_hi, dq_c, dk_c, dv_c, dq_acc,
                                   q_start + last * _BM, h, last == loop_st)""",
     """                        _gradients(pT_s0, dsT_s0, do_s0, q_s0, k_s, a_k, a_t,
                                   b_hi, dq_c, dk_c, dv_c, dq_acc,
                                   q_start + last * _BM, h, last == loop_st)"""),
    ("""                        _gradients(pT_s1, dsT_s1, doT_s1, qT_s1, kT_s, a_k, a_t,
                                   b_hi, dq_c, dk_c, dv_c, dq_acc,
                                   q_start + last * _BM, h, last == loop_st)""",
     """                        _gradients(pT_s1, dsT_s1, do_s1, q_s1, k_s, a_k, a_t,
                                   b_hi, dq_c, dk_c, dv_c, dq_acc,
                                   q_start + last * _BM, h, last == loop_st)"""),
]

UBV2B = UBV2 + [
    ("""            p16_u = T.alloc_shared((_HN, _BM), dt)
            dsT_u = T.alloc_shared((_HN, _BM), dt)""",
     """            # p16/dsT are produced and consumed inside one `_epilogue`, so a
            # single copy only leaves a WAR against the *next* half body's
            # epilogue; a second version removes it (UB +12,288 B).
            p16_u0 = T.alloc_shared((_HN, _BM), dt)
            dsT_u0 = T.alloc_shared((_HN, _BM), dt)
            p16_u1 = T.alloc_shared((_HN, _BM), dt)
            dsT_u1 = T.alloc_shared((_HN, _BM), dt)"""),
    ("""                        _epilogue(sid, st_u0, dp_u0, p16_u, dsT_u, lse_u0, dlt_u0,
                                  pT_s0, dsT_s0, n0, m0_a, len_q, len_kv, shift)""",
     """                        _epilogue(sid, st_u0, dp_u0, p16_u0, dsT_u0, lse_u0, dlt_u0,
                                  pT_s0, dsT_s0, n0, m0_a, len_q, len_kv, shift)"""),
    ("""                            _epilogue(sid, st_u1, dp_u1, p16_u, dsT_u, lse_u1, dlt_u1,
                                      pT_s1, dsT_s1, n0, m0_b, len_q, len_kv, shift)""",
     """                            _epilogue(sid, st_u1, dp_u1, p16_u1, dsT_u1, lse_u1, dlt_u1,
                                      pT_s1, dsT_s1, n0, m0_b, len_q, len_kv, shift)"""),
]

VARIANTS = {
    "ubv2": UBV2,
    "ubv2b": UBV2B,
    "nt": NT,
    "ubv2_nt": UBV2 + NT,
}


def main():
    src, dst, variant = sys.argv[1], sys.argv[2], sys.argv[3]
    text = Path(src).read_text()
    for old, new in VARIANTS[variant]:
        if text.count(old) != 1:
            raise SystemExit(f"[{variant}] anchor count {text.count(old)}:\n{old[:180]}")
        text = text.replace(old, new)
    Path(dst).write_text(text)
    print(f"[{variant}] {src} -> {dst}: {len(VARIANTS[variant])} edit(s)")


if __name__ == "__main__":
    main()
