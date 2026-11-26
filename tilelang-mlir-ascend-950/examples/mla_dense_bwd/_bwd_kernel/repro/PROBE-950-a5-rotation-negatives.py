"""PROBE-950: negative controls for the A5 rotation of MLA-dense ``_bwd_main``.

Why this exists
---------------
A5 moves the gradient GEMMs one iteration late (``F(k) -> GA(k) -> EP(k) ->
GB(k-1)``) and expresses the "other version" by static buffer names
(``*_s0``/``*_s1``).  Both facts are invisible to a numeric test that happens
to pass: if the compiler folded the two versions back into one, or if the
trailing ``GB(loop_ed-1)`` (the drain) were missing, a small shape can still
land inside the tolerance.  The only honest check is that a *deliberately
broken* body fails.  Measured on this toolchain (2026-11-26):

    P10  half body A's GB reads the current version (_s0 instead of _s1)
         -> 69.4 % of the elements outside the tolerance        (caught)
    P9   drain deleted (both parity branches)
         -> non-causal S=128 (all n_it=2, even): 93.1 % out     (caught)
         -> non-causal S=64  (all n_it=1, odd) : 97.0 % out     (caught)
         -> causal S=384/H=2 (n_it 1..6)       : 78.5 % out     (caught)

Block geometry note: ``BLOCK_M = 64``, so ``n_it = ceil(S_q/64) - loop_st``.
Causal ``S=96`` gives ``loop_ed = 2`` and ``n_it = 2`` (**even**), not 1 -- an
attribution error that DESIGN.md 8.3 L0-13 makes; the odd branch has to be
isolated with ``S=64`` or a non-causal geometry (``loop_st = 0`` makes the
parity uniform over blocks).

Run
---
    cd /tmp && source <repo>/.agents/platforms/toolchains/ascend/env-950pr.sh \
      && TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
         <repo>/examples/mla_dense_bwd/_bwd_kernel/repro/PROBE-950-a5-rotation-negatives.py

Exit 0 = both defects were caught (the gates have discriminating power).
"""
import importlib.util
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "_bwd_kernel.py")


def load(name, text):
    path = f"/tmp/{name}.py"
    with open(path, "w") as fh:
        fh.write(text)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    text = open(SRC).read()

    # ---- P10: half body A's GB reads version A (the current half) ----------
    p10 = text.replace(
        """                    _gradients(pT_s1, dsT_s1, doT_s1, qT_s1, kT_s, a_k, a_t, b_hi,
                               dq_c, dk_c, dv_c, dq_acc, b, m0_a - _BM, h,""",
        """                    _gradients(pT_s0, dsT_s0, doT_s0, qT_s0, kT_s, a_k, a_t, b_hi,
                               dq_c, dk_c, dv_c, dq_acc, b, m0_a - _BM, h,""")
    assert p10 != text, "P10 patch anchor not found (did the loop body change?)"

    # ---- P9: delete the drain (both parity branches) -----------------------
    p9 = text.replace("            if (n_it % 2) == 1:",
                      "            if (n_it % 2) == 9:")
    p9 = p9.replace(
        """            else:
                _gradients(pT_s1, dsT_s1, doT_s1, qT_s1, kT_s, a_k, a_t, b_hi,
                           dq_c, dk_c, dv_c, dq_acc, b, last * _BM, h,
                           last == loop_st)""",
        """            elif (n_it % 2) == 9:
                _gradients(pT_s1, dsT_s1, doT_s1, qT_s1, kT_s, a_k, a_t, b_hi,
                           dq_c, dk_c, dv_c, dq_acc, b, last * _BM, h,
                           last == loop_st)""")
    assert p9.count("(n_it % 2) == 9") == 2, "P9 patch anchors not found"

    print("=== P10: half A's GB reads the current version (must FAIL) ===", flush=True)
    m10 = load("neg_p10", p10)
    ok10, _ = m10.run_case("NEG", "P10 current-version read", 1, 384, 384, 2,
                           "bfloat16", seed=201)

    cases = [
        ("P9 even all-n_it=2", 1, 128, 1, "bfloat16", False, 202),
        ("P9 odd  all-n_it=1", 1, 64, 1, "bfloat16", False, 203),
        ("P9 causal multi-block", 1, 384, 2, "bfloat16", True, 204),
    ]
    print("=== P9: drain deleted (must FAIL on every geometry) ===", flush=True)
    m9 = load("neg_p9", p9)
    caught = []
    for tag, b, s, h, dt, causal, seed in cases:
        ok, _ = m9.run_case("NEG", tag, b, s, s, h, dt, seed=seed, is_causal=causal)
        caught.append(not ok)

    all_caught = (not ok10) and all(caught)
    print(f"[NEG-SUMMARY] p10_caught={not ok10} p9_caught={caught} "
          f"{'ALL_CAUGHT' if all_caught else 'CHECKER_BLIND'}", flush=True)
    assert all_caught, "a known-broken body passed: the version/drain gates are blind"
    return 0


if __name__ == "__main__":
    sys.exit(main())
