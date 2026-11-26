"""c5: split the P/dS epilogue so dV starts as soon as P is ready.

One main optimisation point: the single vector epilogue (which waits for both
L0C drains and blocks every GEMM) is split into a P pass (needs `st` only) and a
dS pass (needs `dpt`), and `dV` — whose only operand is P — is issued right
after the P pass.  The dS pass and the dP GEMM then overlap on the other core.
Numerics are unchanged: the same fp32 `p` is recomputed by the second pass
(identical inputs -> identical `exp`), and no accumulation order changes.
"""
from pathlib import Path

OLD_VEC = '''                # ---- s^T and dP^T: two L0 GEMMs, cleared every iteration ----
                T.copy(k_s, a_k)
                T.copy(q_s, b_lo)
                T.gemm(a_k, b_lo, st_c, transpose_B=True, clear_accum=True)
                T.copy(v_s, a_k[0:_BN, 0:dim_v])
                T.copy(do_s, b_lo[0:_BM, 0:dim_v])
                T.gemm(a_k[0:_BN, 0:dim_v], b_lo[0:_BM, 0:dim_v], dpt_c,
                       transpose_B=True, clear_accum=True)
                # L0C -> UB: each AIV receives its own 48 key rows of the tile.
                T.dual_copy(st_c, st_u)
                T.dual_copy(dpt_c, dp_u)

                fast = (n0 + _BN <= seq_kv) & (m0 + _BM <= seq_q)
                if is_causal:
                    fast = fast & (n0 + _BN - 1 <= m0 + shift)
                if a2_fast:
                    if fast:
                        # Fast arm: the whole tile is visible and fully in range.
                        with T.SimtVF(threads=128):
                            for i, j in T.Parallel(_HN, _BM):
                                p = T.exp(st_u[i, j] * scale - lse_u[j])
                                p16_u[i, j] = T.cast(p, dt)
                                dsT_u[i, j] = T.cast(p * (dp_u[i, j] - dlt_u[j]), dt)
                    else:
                        with T.SimtVF(threads=128):
                            for i, j in T.Parallel(_HN, _BM):
                                gi = n0 + T.cast(sid, "int32") * _HN + i
                                gj = m0 + j
                                vis = (gi < seq_kv) & (gj < seq_q)
                                if is_causal:
                                    vis = vis & (gi <= gj + shift)
                                p = T.exp(st_u[i, j] * scale - lse_u[j])
                                pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                                p16_u[i, j] = T.cast(pv, dt)
                                dsT_u[i, j] = T.cast(pv * (dp_u[i, j] - dlt_u[j]), dt)
                else:
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(_HN, _BM):
                            gi = n0 + T.cast(sid, "int32") * _HN + i
                            gj = m0 + j
                            vis = (gi < seq_kv) & (gj < seq_q)
                            if is_causal:
                                vis = vis & (gi <= gj + shift)
                            p = T.exp(st_u[i, j] * scale - lse_u[j])
                            pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                            p16_u[i, j] = T.cast(pv, dt)
                            dsT_u[i, j] = T.cast(pv * (dp_u[i, j] - dlt_u[j]), dt)

                # ---- dV / dK (accumulate) and dQ (transpose_A reuse of dsT) ---
                T.dual_copy(p16_u, pT_s)
                T.dual_copy(dsT_u, dsT_s)
                T.copy(pT_s, a_k[0:_BN, 0:_BM])
                T.copy(doT_s, b_hi[0:dim_v, 0:_BM])
                T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:dim_v, 0:_BM], dv_c,
                       transpose_B=True, clear_accum=(kq == loop_st))
                T.copy(dsT_s, a_k[0:_BN, 0:_BM])
'''

NEW_VEC = '''                # ---- s^T: scores, then P and dV as soon as P is ready --------
                T.copy(k_s, a_k)
                T.copy(q_s, b_lo)
                T.gemm(a_k, b_lo, st_c, transpose_B=True, clear_accum=True)
                # L0C -> UB: each AIV receives its own 48 key rows of the tile.
                T.dual_copy(st_c, st_u)

                fast = (n0 + _BN <= seq_kv) & (m0 + _BM <= seq_q)
                if is_causal:
                    fast = fast & (n0 + _BN - 1 <= m0 + shift)
                # Pass 1: P = exp(scale*q.k - lse) -- needs the score tile only, so
                # it runs while the dP GEMM is still in flight.
                if a2_fast:
                    if fast:
                        with T.SimtVF(threads=128):
                            for i, j in T.Parallel(_HN, _BM):
                                p16_u[i, j] = T.cast(
                                    T.exp(st_u[i, j] * scale - lse_u[j]), dt)
                    else:
                        with T.SimtVF(threads=128):
                            for i, j in T.Parallel(_HN, _BM):
                                gi = n0 + T.cast(sid, "int32") * _HN + i
                                gj = m0 + j
                                vis = (gi < seq_kv) & (gj < seq_q)
                                if is_causal:
                                    vis = vis & (gi <= gj + shift)
                                p = T.exp(st_u[i, j] * scale - lse_u[j])
                                p16_u[i, j] = T.cast(
                                    T.if_then_else(vis, p, T.cast(0, "float32")), dt)
                else:
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(_HN, _BM):
                            gi = n0 + T.cast(sid, "int32") * _HN + i
                            gj = m0 + j
                            vis = (gi < seq_kv) & (gj < seq_q)
                            if is_causal:
                                vis = vis & (gi <= gj + shift)
                            p = T.exp(st_u[i, j] * scale - lse_u[j])
                            p16_u[i, j] = T.cast(
                                T.if_then_else(vis, p, T.cast(0, "float32")), dt)

                # ---- dV: P^T dO, the only GEMM that does not need dS ---------
                T.dual_copy(p16_u, pT_s)
                T.copy(pT_s, a_k[0:_BN, 0:_BM])
                T.copy(doT_s, b_hi[0:dim_v, 0:_BM])
                T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:dim_v, 0:_BM], dv_c,
                       transpose_B=True, clear_accum=(kq == loop_st))

                # ---- dP^T GEMM, then pass 2 of the epilogue -------------------
                T.copy(v_s, a_k[0:_BN, 0:dim_v])
                T.copy(do_s, b_lo[0:_BM, 0:dim_v])
                T.gemm(a_k[0:_BN, 0:dim_v], b_lo[0:_BM, 0:dim_v], dpt_c,
                       transpose_B=True, clear_accum=True)
                T.dual_copy(dpt_c, dp_u)

                # Pass 2: dS = P (dP - delta); `p` is recomputed from the same
                # score tile, so the value is bit-identical to pass 1's.
                if a2_fast:
                    if fast:
                        with T.SimtVF(threads=128):
                            for i, j in T.Parallel(_HN, _BM):
                                p = T.exp(st_u[i, j] * scale - lse_u[j])
                                dsT_u[i, j] = T.cast(p * (dp_u[i, j] - dlt_u[j]), dt)
                    else:
                        with T.SimtVF(threads=128):
                            for i, j in T.Parallel(_HN, _BM):
                                gi = n0 + T.cast(sid, "int32") * _HN + i
                                gj = m0 + j
                                vis = (gi < seq_kv) & (gj < seq_q)
                                if is_causal:
                                    vis = vis & (gi <= gj + shift)
                                p = T.exp(st_u[i, j] * scale - lse_u[j])
                                pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                                dsT_u[i, j] = T.cast(pv * (dp_u[i, j] - dlt_u[j]), dt)
                else:
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(_HN, _BM):
                            gi = n0 + T.cast(sid, "int32") * _HN + i
                            gj = m0 + j
                            vis = (gi < seq_kv) & (gj < seq_q)
                            if is_causal:
                                vis = vis & (gi <= gj + shift)
                            p = T.exp(st_u[i, j] * scale - lse_u[j])
                            pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                            dsT_u[i, j] = T.cast(pv * (dp_u[i, j] - dlt_u[j]), dt)

                # ---- dK (accumulate) and dQ (transpose_A reuse of dsT) --------
                T.dual_copy(dsT_u, dsT_s)
                T.copy(dsT_s, a_k[0:_BN, 0:_BM])
'''

src = Path('MLADenseBwdOp_baseline_stage3.py').read_text()
assert src.count(OLD_VEC) == 1
Path('MLADenseBwdOp_round1_c5.py').write_text(src.replace(OLD_VEC, NEW_VEC))
print('c5 written')
