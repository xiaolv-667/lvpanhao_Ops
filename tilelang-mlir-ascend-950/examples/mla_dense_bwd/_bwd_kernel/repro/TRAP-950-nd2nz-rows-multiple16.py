"""TRAP-950: the UB->L1 `nd2nz` scatter rejects a non-16-multiple row count.

Claim (measured 2026-11-25, ascent-950pr-9579 / tilelang 0.1.15 / CANN 9.2.1):
the mixed Cube+Vector MLA backward kernel writes its two epilogue tiles back with
`T.dual_copy(p16_u, pT_s)` / `T.dual_copy(dsT_u, dsT_s)` (UB -> L1, lowered by
`InsertNd2Nz` into an `ascend_nd2nz_scatter`).  That path emits a template whose
per-AIV row count must be a multiple of 16:

    tl_templates/ascend/nd2nz_copy.h:59: static_assert(ROWS % 16 == 0, ...)

The kernel's per-AIV row count is `block_n / 2` (the 2:1 `dual_copy` split on the
key axis), so `block_n = 80` -- the DESIGN.md 1.6.3 pre-planned fallback for the
MTE1 budget -- *cannot be built at all*, while `block_n = 96` (48 rows) can.
Consequence: the fallback ladder for this kernel is `block_n in {96, 112, 128}`
upwards or `{64}` downwards (32 rows), not 80; `block_n <= 32` also fails because
`block_n/2 = 16` is the smallest legal value.

The script also pins the two correctness-critical properties of the built kernel
(DESIGN.md 8.3 P2b): the `dk`/`dv` store clamp is this request's `S_kv`, never the
packed `B*S_kv`, and the five GEMMs are explicit-L0 `asc_mmad`s with exactly one
`asc_copy_l12l0a_transpose` (the `transpose_A=True` reuse of `dS`).

Run (outside the repo root; the platform default interpreter has no torch):

    cd /tmp && source <repo>/.agents/platforms/toolchains/ascend/env-950pr.sh \
      && TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
         <repo>/examples/mla_dense_bwd/_bwd_kernel/repro/TRAP-950-nd2nz-rows-multiple16.py
"""

import importlib.util
import os
import sys

import torch  # noqa: F401  -- imported before the module under test
import torch_npu  # noqa: F401

_OP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_bwd_kernel.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("mla_bwd", _OP)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mla_bwd"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    kb = _load()
    scale = 192**-0.5

    # ---- 1. block_n=96 builds; its structure is the designed one ------------
    kb._BN, kb._HN = 96, 48
    kb._bwd_kernel.cache_clear()
    fn = kb._launch(2, 4, 192, 128, 512, 512, True, scale, "bfloat16")
    src = fn.get_kernel_source()
    assert src.count("asc_mmad") == 5, src.count("asc_mmad")
    assert src.count("asc_copy_l12l0a_transpose") == 1, "dQ must reuse dS via trans_A"
    assert src.count("asc_copy_l0c2ub") == 4, "2 per-iteration + 2 block-tail halves"
    assert src.count("asc_copy_ub2gm_align") == 2, "one dk + one dv store"
    dk_lines = [ln for ln in src.splitlines() if "asc_copy_ub2gm_align" in ln and "dk[" in ln]
    assert dk_lines, "no dk store found"
    store = dk_lines[0]
    assert "min(48, max(((512 - " in store, store[:160]
    assert "1024" not in store, "store clamp must use S_kv, not B*S_kv"
    print("[PASS] block_n=96: 5 L0 mmad, 1 transposing L0A copy, "
          "dk store clamped to S_kv=512 (no B*S_kv)")

    # ---- 2. block_n=80 (DESIGN 1.6.3 fallback) cannot be compiled -----------
    kb._BN, kb._HN = 80, 40
    kb._bwd_kernel.cache_clear()
    try:
        kb._launch(2, 4, 192, 128, 512, 512, True, scale, "bfloat16")
    except RuntimeError as exc:
        msg = str(exc)
        assert "ROWS must be a multiple of 16" in msg, msg[-600:]
        assert "nd2nz_copy.h" in msg, msg[-600:]
        print("[PASS] block_n=80 rejected: per-AIV rows 40 is not a multiple of 16 "
              "(nd2nz_copy.h static_assert) -> the Nk=80 fallback is infeasible")
        return 0
    raise AssertionError("block_n=80 unexpectedly compiled")


if __name__ == "__main__":
    sys.exit(main())
