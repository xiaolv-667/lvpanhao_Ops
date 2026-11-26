"""Build one candidate variant of the v4 (A5-rotated) MLADenseBwdOp.py.

Each candidate applies **one** main optimisation point to the current best file,
so a measured difference is attributable (combinations are only built after the
single points showed evidence).  Usage:

    python perf_opt/make_candidate.py <src> <dst> <variant>
"""

import sys
from pathlib import Path

ALLOC_OLD = """            st_u = T.alloc_shared((_HN, _BM), "float32")
            dp_u = T.alloc_shared((_HN, _BM), "float32")
            p16_u = T.alloc_shared((_HN, _BM), dt)
            dsT_u = T.alloc_shared((_HN, _BM), dt)"""

# --- c1: one UB score pair per half body -------------------------------------
# ``st_u``/``dp_u`` are written by GA(k)'s fixpipe and read by EP(k).  With a
# single pair the fixpipe of GA(k+1) must wait for EP(k)'s vector reads (WAR),
# which is the 0.42 us/iteration exposure attributed in the v3 Stage 4 report.
# The two half bodies are statically expanded, so each can own its own pair:
# GA(B) writes st_u1/dp_u1 while EP(A) reads st_u0/dp_u0 (+24,576 B UB => 60.9%).
SCORE_VER_EDITS = [
    (ALLOC_OLD,
     """            st_u0 = T.alloc_shared((_HN, _BM), "float32")
            dp_u0 = T.alloc_shared((_HN, _BM), "float32")
            st_u1 = T.alloc_shared((_HN, _BM), "float32")
            dp_u1 = T.alloc_shared((_HN, _BM), "float32")
            p16_u = T.alloc_shared((_HN, _BM), dt)
            dsT_u = T.alloc_shared((_HN, _BM), dt)"""),
    # half body A
    ("""                _scores(k_s, v_s, q_s0, do_s0, a_k, b_lo, st_c, dpt_c, st_u, dp_u)
                _epilogue(sid, st_u, dp_u, p16_u, dsT_u, lse_u0, dlt_u0,
                          pT_s0, dsT_s0, n0, m0_a)""",
     """                _scores(k_s, v_s, q_s0, do_s0, a_k, b_lo, st_c, dpt_c, st_u0, dp_u0)
                _epilogue(sid, st_u0, dp_u0, p16_u, dsT_u, lse_u0, dlt_u0,
                          pT_s0, dsT_s0, n0, m0_a)"""),
    # half body B
    ("""                    _scores(k_s, v_s, q_s1, do_s1, a_k, b_lo, st_c, dpt_c, st_u, dp_u)
                    _epilogue(sid, st_u, dp_u, p16_u, dsT_u, lse_u1, dlt_u1,
                              pT_s1, dsT_s1, n0, m0_b)""",
     """                    _scores(k_s, v_s, q_s1, do_s1, a_k, b_lo, st_c, dpt_c, st_u1, dp_u1)
                    _epilogue(sid, st_u1, dp_u1, p16_u, dsT_u, lse_u1, dlt_u1,
                              pT_s1, dsT_s1, n0, m0_b)"""),
]

# --- c2: one epilogue output pair per half body -------------------------------
# ``p16_u``/``dsT_u`` are written by EP(k)'s VF and drained by the same body's
# UB->L1 copy; sharing them across half bodies makes EP(k+1)'s first vector store
# wait for EP(k)'s mte3 (+12,288 B UB => 65.7%).
OUT_VER_EDITS = [
    (ALLOC_OLD,
     """            st_u = T.alloc_shared((_HN, _BM), "float32")
            dp_u = T.alloc_shared((_HN, _BM), "float32")
            p16_u0 = T.alloc_shared((_HN, _BM), dt)
            dsT_u0 = T.alloc_shared((_HN, _BM), dt)
            p16_u1 = T.alloc_shared((_HN, _BM), dt)
            dsT_u1 = T.alloc_shared((_HN, _BM), dt)"""),
    ("""                _epilogue(sid, st_u, dp_u, p16_u, dsT_u, lse_u0, dlt_u0,
                          pT_s0, dsT_s0, n0, m0_a)""",
     """                _epilogue(sid, st_u, dp_u, p16_u0, dsT_u0, lse_u0, dlt_u0,
                          pT_s0, dsT_s0, n0, m0_a)"""),
    ("""                    _epilogue(sid, st_u, dp_u, p16_u, dsT_u, lse_u1, dlt_u1,
                              pT_s1, dsT_s1, n0, m0_b)""",
     """                    _epilogue(sid, st_u, dp_u, p16_u1, dsT_u1, lse_u1, dlt_u1,
                              pT_s1, dsT_s1, n0, m0_b)"""),
]

# --- c3: both pairs per half body (c1 + c2) ----------------------------------
BOTH_EDITS = [
    (ALLOC_OLD,
     """            st_u0 = T.alloc_shared((_HN, _BM), "float32")
            dp_u0 = T.alloc_shared((_HN, _BM), "float32")
            st_u1 = T.alloc_shared((_HN, _BM), "float32")
            dp_u1 = T.alloc_shared((_HN, _BM), "float32")
            p16_u0 = T.alloc_shared((_HN, _BM), dt)
            dsT_u0 = T.alloc_shared((_HN, _BM), dt)
            p16_u1 = T.alloc_shared((_HN, _BM), dt)
            dsT_u1 = T.alloc_shared((_HN, _BM), dt)"""),
    ("""                _scores(k_s, v_s, q_s0, do_s0, a_k, b_lo, st_c, dpt_c, st_u, dp_u)
                _epilogue(sid, st_u, dp_u, p16_u, dsT_u, lse_u0, dlt_u0,
                          pT_s0, dsT_s0, n0, m0_a)""",
     """                _scores(k_s, v_s, q_s0, do_s0, a_k, b_lo, st_c, dpt_c, st_u0, dp_u0)
                _epilogue(sid, st_u0, dp_u0, p16_u0, dsT_u0, lse_u0, dlt_u0,
                          pT_s0, dsT_s0, n0, m0_a)"""),
    ("""                    _scores(k_s, v_s, q_s1, do_s1, a_k, b_lo, st_c, dpt_c, st_u, dp_u)
                    _epilogue(sid, st_u, dp_u, p16_u, dsT_u, lse_u1, dlt_u1,
                              pT_s1, dsT_s1, n0, m0_b)""",
     """                    _scores(k_s, v_s, q_s1, do_s1, a_k, b_lo, st_c, dpt_c, st_u1, dp_u1)
                    _epilogue(sid, st_u1, dp_u1, p16_u1, dsT_u1, lse_u1, dlt_u1,
                              pT_s1, dsT_s1, n0, m0_b)"""),
]

VARIANTS = {
    "c1_scorever": SCORE_VER_EDITS,
    "c2_outver": OUT_VER_EDITS,
    "c3_bothver": BOTH_EDITS,
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
    sys.exit(main())
