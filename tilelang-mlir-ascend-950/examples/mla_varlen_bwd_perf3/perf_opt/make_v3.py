"""V=3 variant of the A5 loop: three static half-body versions per iteration.

The A5 `T.serial` + static-name expansion rotates two half bodies per iteration.
This builds the 3-way rotation (`*_s0/_s1/_s2`) with the same one-half-body lag
(`GB(k-1)` inside half body k) and a 3-way drain, so `F(k+2)` can be in flight
while `EP(k)` runs.  Built on the `ubv2` variant (st_u/dp_u already split).

Usage: python make_v3.py <src=ubv2 candidate> <dst>
"""

import sys
from pathlib import Path

LOOP_OLD_START = "                    n_it = loop_ed - loop_st"
LOOP_OLD_END = "                    # ---- block tail: dk = scale * cast(dk_c), dv = cast(dv_c) ----"

NEW_ALLOC_ANCHOR = """            lse_u1 = T.alloc_shared((_BM,), "float32")
            dlt_u1 = T.alloc_shared((_BM,), "float32")"""
NEW_ALLOC = NEW_ALLOC_ANCHOR + """
            # third static version (V=3): lets F(k+2) be in flight while EP(k) runs
            q_s2 = T.alloc_l1((_BM, dim_qk), dt)
            qT_s2 = T.alloc_l1((dim_qk, _BM), dt)
            do_s2 = T.alloc_l1((_BM, dim_v), dt)
            doT_s2 = T.alloc_l1((dim_v, _BM), dt)
            pT_s2 = T.alloc_l1((_BN, _BM), dt)
            dsT_s2 = T.alloc_l1((_BN, _BM), dt)
            lse_u2 = T.alloc_shared((_BM,), "float32")
            dlt_u2 = T.alloc_shared((_BM,), "float32")"""

LOOP = '''                    n_it = loop_ed - loop_st
                    # V=3 (Stage 4 round 3): three half bodies per `T.serial`
                    # iteration, each with its own statically named version, and the
                    # same one-half-body lag for the gradient GEMMs.  The third
                    # version keeps one more fill in flight, which is the point: the
                    # AIC's MTE2 fill is the v5 critical pipe at 81-85% busy.
                    n_third = (n_it + 2) // 3
                    for i in T.serial(n_third):
                        # ---- half body A (version 0) ----
                        kq_a = loop_st + i * 3
                        m0_a = kq_a * _BM
                        qoff_a = q_start + m0_a
                        _fill(q, dout, lse, delta, q_s0, do_s0, qT_s0, doT_s0,
                              lse_u0, dlt_u0, h, qoff_a)
                        _scores(k_s, v_s, q_s0, do_s0, a_k, b_lo, st_c, dpt_c,
                                st_u0, dp_u0)
                        _epilogue(sid, st_u0, dp_u0, p16_u, dsT_u, lse_u0, dlt_u0,
                                  pT_s0, dsT_s0, n0, m0_a, len_q, len_kv, shift)
                        if kq_a > loop_st:
                            _gradients(pT_s2, dsT_s2, doT_s2, qT_s2, kT_s, a_k, a_t,
                                       b_hi, dq_c, dk_c, dv_c, dq_acc,
                                       qoff_a - _BM, h, kq_a - 1 == loop_st)
                        # ---- half body B (version 1) ----
                        kq_b = kq_a + 1
                        if kq_b < loop_ed:
                            m0_b = kq_b * _BM
                            qoff_b = q_start + m0_b
                            _fill(q, dout, lse, delta, q_s1, do_s1, qT_s1, doT_s1,
                                  lse_u1, dlt_u1, h, qoff_b)
                            _scores(k_s, v_s, q_s1, do_s1, a_k, b_lo, st_c, dpt_c,
                                    st_u1, dp_u1)
                            _epilogue(sid, st_u1, dp_u1, p16_u, dsT_u, lse_u1, dlt_u1,
                                      pT_s1, dsT_s1, n0, m0_b, len_q, len_kv, shift)
                            _gradients(pT_s0, dsT_s0, doT_s0, qT_s0, kT_s, a_k, a_t,
                                       b_hi, dq_c, dk_c, dv_c, dq_acc,
                                       qoff_b - _BM, h, kq_b - 1 == loop_st)
                        # ---- half body C (version 2) ----
                        kq_c = kq_a + 2
                        if kq_c < loop_ed:
                            m0_c = kq_c * _BM
                            qoff_c = q_start + m0_c
                            _fill(q, dout, lse, delta, q_s2, do_s2, qT_s2, doT_s2,
                                  lse_u2, dlt_u2, h, qoff_c)
                            _scores(k_s, v_s, q_s2, do_s2, a_k, b_lo, st_c, dpt_c,
                                    st_u0, dp_u0)
                            _epilogue(sid, st_u0, dp_u0, p16_u, dsT_u, lse_u2, dlt_u2,
                                      pT_s2, dsT_s2, n0, m0_c, len_q, len_kv, shift)
                            _gradients(pT_s1, dsT_s1, doT_s1, qT_s1, kT_s, a_k, a_t,
                                       b_hi, dq_c, dk_c, dv_c, dq_acc,
                                       qoff_c - _BM, h, kq_c - 1 == loop_st)
                    # ---- drain: GB(loop_ed-1) has no next half body ----------
                    last = loop_ed - 1
                    if (n_it % 3) == 1:
                        _gradients(pT_s0, dsT_s0, doT_s0, qT_s0, kT_s, a_k, a_t,
                                   b_hi, dq_c, dk_c, dv_c, dq_acc,
                                   q_start + last * _BM, h, last == loop_st)
                    else:
                        if (n_it % 3) == 2:
                            _gradients(pT_s1, dsT_s1, doT_s1, qT_s1, kT_s, a_k, a_t,
                                       b_hi, dq_c, dk_c, dv_c, dq_acc,
                                       q_start + last * _BM, h, last == loop_st)
                        else:
                            _gradients(pT_s2, dsT_s2, doT_s2, qT_s2, kT_s, a_k, a_t,
                                       b_hi, dq_c, dk_c, dv_c, dq_acc,
                                       q_start + last * _BM, h, last == loop_st)

'''


def main():
    src, dst = sys.argv[1], sys.argv[2]
    text = Path(src).read_text()
    if text.count(NEW_ALLOC_ANCHOR) != 1:
        raise SystemExit("alloc anchor not found")
    text = text.replace(NEW_ALLOC_ANCHOR, NEW_ALLOC)
    start = text.index(LOOP_OLD_START)
    end = text.index(LOOP_OLD_END)
    text = text[:start] + LOOP + text[end:]
    Path(dst).write_text(text)
    print(f"[v3] {src} -> {dst}")


if __name__ == "__main__":
    main()
