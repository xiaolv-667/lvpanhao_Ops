"""Generate Stage-4 candidate kernel modules by patching the frozen Stage-3 source.

Every candidate is a *full standalone module* (same factory signatures as the
baseline, so `bench_one.py` and the module's own L0/L1 runner drive it unchanged)
obtained by an explicit, reviewable text patch of the read-only baseline.  One
candidate = one optimised point (skill Phase 2 discipline); nothing here touches
`examples/TileOPs` — candidates land in `perf_opt/`.

Usage (cwd = examples/TileOPs):

    python mkvariant.py            # writes/refreshes every variant below
    python mkvariant.py r1_c2      # one variant
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PERF_OPT = os.path.dirname(HERE)
BASE_DIR = os.path.abspath(os.path.join(
    PERF_OPT, "..", "..", "TileOPs", "tileops", "kernels", "sequence_modeling",
    "engram_decode", "engram_decode_kernel"))

PROJECT = "_engram_project_kernel.py"
STEP = "_engram_step_kernel.py"

PROJ_HEAD = '"""`_engram_project_kernel` on Ascend 950PR: both decode projections + cache shift.'
STEP_HEAD = '"""`_engram_step_kernel` on Ascend 950PR (SIMT): launch 2 of 2 of EngramDecodeFwdOp.'


def patch(src, edits, tag):
    for edit in edits:
        mode, old, new = edit if len(edit) == 3 else ("one", edit[0], edit[1])
        if old not in src:
            raise SystemExit(f"[{tag}] patch anchor not found:\n{old[:400]}")
        src = src.replace(old, new) if mode == "all" else src.replace(old, new, 1)
    return src


# ===========================================================================
# O-1 / O-1b: cache shift restructure (round 1)
# ===========================================================================
SHIFT_CONSTS_OLD = "# Frozen tiling (DESIGN.md 5.6)."
SHIFT_CONSTS_NEW = """# Stage 4 O-1 knobs: how many blocks run the cache shift (one batch row each) and
# the UB budget that keeps the full-width row form legal.
SHIFT_BLOCKS_MAX = 16
SHIFT_UB_BUDGET = 98304

# Frozen tiling (DESIGN.md 5.6)."""

SHIFT_GRID_OLD = """    nshift = max_conv_len - 1
    grid = m_tiles * 2 * n_tiles
"""

SHIFT_GRID_NEW_DEDICATED = """    nshift = max_conv_len - 1
    gemm_grid = m_tiles * 2 * n_tiles
    # Stage 4 O-1: the cache shift owns `nb_shift` trailing blocks, one batch row
    # each, as a single full-width copy pair (`conv_state[b, 1:]` is contiguous).
    nb_shift = min(batch, SHIFT_BLOCKS_MAX) if nshift > 0 else 0
    shift_reps = -(-batch // nb_shift) if nb_shift > 0 else 0
    shift_wide = nshift * d_pad * 2 <= SHIFT_UB_BUDGET
    shift_width = d_pad if shift_wide else block_n
    grid = gemm_grid + nb_shift
"""

SHIFT_GRID_NEW_INLINE = """    nshift = max_conv_len - 1
    grid = m_tiles * 2 * n_tiles
    # Stage 4 O-1b: the cache shift is spread over the first `nb_shift` blocks, one
    # batch row per block, as a single full-width copy pair (`conv_state[b, 1:]` is
    # contiguous).  The baseline ran it as `for b in T.serial(batch)` x
    # (copy + sync + copy) inside the `which == 0` blocks: serving-b8 paid 3.67 us
    # for 8 serialised pairs in 4 blocks.  Falls back to the per-n-tile slice form
    # when a full row does not fit the UB budget.
    nb_shift = min(batch, grid, SHIFT_BLOCKS_MAX) if nshift > 0 else 0
    shift_reps = -(-batch // nb_shift) if nb_shift > 0 else 0
    shift_wide = nshift * d_pad * 2 <= SHIFT_UB_BUDGET
    shift_width = d_pad if shift_wide else block_n
"""

SHIFT_SLICE_FALLBACK = """                            else:
                                for nt in T.serial(n_tiles):
                                    cn0 = nt * block_n
                                    can = T.min(d_pad - cn0, block_n)
                                    T.copy(
                                        conv_state[b, 1:max_conv_len, cn0 : cn0 + can],
                                        shift_ub[0:nshift, 0:can],
                                    )
                                    T.copy(
                                        shift_ub[0:nshift, 0:can],
                                        new_conv_state[b, 0:nshift, cn0 : cn0 + can],
                                    )
"""

SHIFT_BODY_OLD = """            if which == 0:
                _projection(e_t, W_K, kv, 0, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
                if nshift > 0:
                    if m0 == 0:
                        shift_ub = T.alloc_shared((nshift, block_n), dtype_str)
                        _shift_cache(conv_state, new_conv_state, shift_ub, n0, an)
            else:
                _projection(e_t, W_V, kv, 1, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
"""

SHIFT_BODY_NEW_DEDICATED = """            if bx < gemm_grid:
                if which == 0:
                    _projection(e_t, W_K, kv, 0, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
                else:
                    _projection(e_t, W_V, kv, 1, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
            if nshift > 0:
                if bx >= gemm_grid:
                    shift_ub = T.alloc_shared((nshift, shift_width), dtype_str)
                    sb = bx - gemm_grid
                    for r in T.serial(shift_reps):
                        b = sb + r * nb_shift
                        if b < batch:
                            if shift_wide:
                                T.copy(conv_state[b, 1:max_conv_len, :], shift_ub[0:nshift, :])
                                T.copy(shift_ub[0:nshift, :], new_conv_state[b, 0:nshift, :])
""" + SHIFT_SLICE_FALLBACK

SHIFT_BODY_NEW_INLINE = """            if which == 0:
                _projection(e_t, W_K, kv, 0, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
            else:
                _projection(e_t, W_V, kv, 1, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
            if nshift > 0:
                if bx < nb_shift:
                    shift_ub = T.alloc_shared((nshift, shift_width), dtype_str)
                    for r in T.serial(shift_reps):
                        b = bx + r * nb_shift
                        if b < batch:
                            if shift_wide:
                                T.copy(conv_state[b, 1:max_conv_len, :], shift_ub[0:nshift, :])
                                T.copy(shift_ub[0:nshift, :], new_conv_state[b, 0:nshift, :])
""" + SHIFT_SLICE_FALLBACK

# ===========================================================================
# O-4: contiguous tap window in the step launch (round 2)
#
# The w-1 taps are single history rows at stride `dilation`; the baseline issues
# one single-row GM->UB copy per tap, each with its own MTE2 wait before the
# compute pass.  Rows `max_conv_len - dilation*(w-1) .. max_conv_len-1` are
# contiguous, so one window copy covers every tap (tap p -> window row
# `dilation*p`).  `max_conv_len >= dilation*(w-1)` (upstream shape rule) keeps the
# window inside the cache.
# ===========================================================================
TAP_CONSTS_OLD = "# DESIGN.md 8.2 == examples/TileOPs/tests/ops/test_engram.py::_tol_decode."
TAP_CONSTS_NEW = """# Stage 4 O-4 knob: UB budget for the contiguous tap window.
TAP_UB_BUDGET = 98304

# DESIGN.md 8.2 == examples/TileOPs/tests/ops/test_engram.py::_tol_decode."""

TAP_SPAN_OLD = """    w = conv_kernel_size
    tile = _next_pow2(d_padded)
"""
TAP_SPAN_NEW = """    w = conv_kernel_size
    # Stage 4 O-4: the tap rows sit in one contiguous span of history rows; the
    # upstream shape rule max_conv_len >= dilation*(w-1) keeps `tap_span` legal.
    tap_span = dilation * (w - 1)
    tap_win = (w > 1) and (
        tap_span * d_padded * (2 if dtype_str in ("float16", "bfloat16") else 4)
        <= TAP_UB_BUDGET
    )
    tile = _next_pow2(d_padded)
"""

TAP_ALLOC_OLD = """            # Zero rows when conv_kernel_size == 1: legal here and never read.
            tap_ub = T.alloc_shared((w - 1, d_padded), dtype_str)
"""
TAP_ALLOC_NEW = """            # Zero rows when conv_kernel_size == 1: legal here and never read.
            if tap_win:
                tap_ub = T.alloc_shared((tap_span, d_padded), dtype_str)
            else:
                tap_ub = T.alloc_shared((w - 1, d_padded), dtype_str)
"""

TAP_LOAD_OLD = """            for p in range(w - 1):
                T.copy(
                    conv_state[bx, max_conv_len - (w - 1 - p) * dilation, :],
                    tap_ub[p, :],
                )
"""
TAP_LOAD_NEW = """            if tap_win:
                T.copy(conv_state[bx, max_conv_len - tap_span : max_conv_len, :], tap_ub)
            else:
                for p in range(w - 1):
                    T.copy(
                        conv_state[bx, max_conv_len - (w - 1 - p) * dilation, :],
                        tap_ub[p, :],
                    )
"""

# both use sites (exact-width and masked-tail branches) read the tap through the
# window row index when the window form is active
TAP_USE_OLD = "tap_ub[p, i]"
TAP_USE_NEW = "tap_ub[dilation * p if tap_win else p, i]"

STEP_HEAD_NEW = '"""Stage 4 O-4 (perf_opt): contiguous tap window in the decode step launch.'

# ---------------------------------------------------------------------------
# O-1c (round 1, candidate 3): per-batch-row shift, additionally split over the d
# columns so `batch * col_splits` blocks share it.  serving-b4 keeps the shift
# inside the projection blocks but one block still moves 19.5 KB + 19.5 KB on a
# single descriptor chain; splitting it 4 ways (grid 16 / batch 4) cuts that to
# ~1/4 of the per-block copy time.
# ---------------------------------------------------------------------------
SHIFT_CONSTS_NEW_C3 = """# Stage 4 O-1c knobs: column chunks per batch row for the cache shift, and the UB
# budget that keeps the multi-row copy form legal.
SHIFT_COL_SPLITS_MAX = 4
SHIFT_UB_BUDGET = 98304

# Frozen tiling (DESIGN.md 5.6)."""

SHIFT_GRID_NEW_COLSPLIT = """    nshift = max_conv_len - 1
    grid = m_tiles * 2 * n_tiles
    # Stage 4 O-1c: the cache shift is spread over the blocks, one (batch row,
    # d-column chunk) per block: `col_splits` column chunks per batch row, all
    # inside the same launch and overlapped with the Cube projection.  The
    # baseline ran it as `for b in T.serial(batch)` x (copy + sync + copy) of a
    # `(max_conv_len-1, block_n)` slice inside the `which == 0` blocks.
    col_splits = max(1, min(SHIFT_COL_SPLITS_MAX, grid // batch)) if nshift > 0 else 0
    nb_shift = batch * col_splits if nshift > 0 else 0
    shift_reps = -(-nb_shift // grid) if nb_shift > 0 else 0
    colw = -(-d_pad // col_splits) if nshift > 0 else 0
    shift_wide = nshift * colw * 2 <= SHIFT_UB_BUDGET
    shift_cols = colw if shift_wide else block_n
"""

SHIFT_BODY_NEW_COLSPLIT = """            if which == 0:
                _projection(e_t, W_K, kv, 0, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
            else:
                _projection(e_t, W_V, kv, 1, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
            if nshift > 0:
                shift_ub = T.alloc_shared((nshift, shift_cols), dtype_str)
                for r in T.serial(shift_reps):
                    sid = bx + r * grid
                    if sid < nb_shift:
                        sb = sid // col_splits
                        c0 = (sid % col_splits) * colw
                        cw = T.min(d_pad - c0, colw)
                        if shift_wide:
                            T.copy(
                                conv_state[sb, 1:max_conv_len, c0 : c0 + cw],
                                shift_ub[0:nshift, 0:cw],
                            )
                            T.copy(
                                shift_ub[0:nshift, 0:cw],
                                new_conv_state[sb, 0:nshift, c0 : c0 + cw],
                            )
                        else:
                            for nt in T.serial(n_tiles):
                                cn0 = c0 + nt * block_n
                                can = T.min(d_pad - cn0, block_n)
                                if 0 < can:
                                    T.copy(
                                        conv_state[sb, 1:max_conv_len, cn0 : cn0 + can],
                                        shift_ub[0:nshift, 0:can],
                                    )
                                    T.copy(
                                        shift_ub[0:nshift, 0:can],
                                        new_conv_state[sb, 0:nshift, cn0 : cn0 + can],
                                    )
"""

SHIFT_CONSTS_NEW_C4 = """# Stage 4 O-1d knobs: the wide per-batch-row shift form is used only when the
# baseline's per-block batch loop would serialise more than one copy pair, i.e.
# when `batch > n_tiles` (the baseline spreads the shift over `n_tiles` blocks).
SHIFT_BLOCKS_MAX = 16
SHIFT_UB_BUDGET = 98304

# Frozen tiling (DESIGN.md 5.6)."""

SHIFT_GRID_NEW_C4 = """    nshift = max_conv_len - 1
    grid = m_tiles * 2 * n_tiles
    # Stage 4 O-1d: `n_tiles` blocks carry the baseline shift, each looping over
    # all `batch` rows.  When batch > n_tiles that loop serialises >1 copy pair
    # per block: serving-b8 (batch 8, n_tiles 4) paid 3.67 us for it (project
    # 6.672 -> 2.999 us with the shift removed).  In that case each batch row
    # moves to its own block as one full-width copy pair instead.
    shift_wide_form = nshift > 0 and batch > n_tiles
    nb_shift = min(batch, grid, SHIFT_BLOCKS_MAX) if shift_wide_form else 0
    shift_reps = -(-batch // nb_shift) if shift_wide_form else 0
    shift_wide = nshift * d_pad * 2 <= SHIFT_UB_BUDGET
    shift_width = d_pad if shift_wide else block_n
"""

SHIFT_BODY_NEW_C4 = """            if which == 0:
                _projection(e_t, W_K, kv, 0, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
            else:
                _projection(e_t, W_V, kv, 1, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
            if nshift > 0:
                if shift_wide_form:
                    if bx < nb_shift:
                        shift_ub = T.alloc_shared((nshift, shift_width), dtype_str)
                        for r in T.serial(shift_reps):
                            b = bx + r * nb_shift
                            if b < batch:
                                if shift_wide:
                                    T.copy(
                                        conv_state[b, 1:max_conv_len, :], shift_ub[0:nshift, :]
                                    )
                                    T.copy(
                                        shift_ub[0:nshift, :],
                                        new_conv_state[b, 0:nshift, :],
                                    )
                                else:
                                    for nt in T.serial(n_tiles):
                                        cn0 = nt * block_n
                                        can = T.min(d_pad - cn0, block_n)
                                        T.copy(
                                            conv_state[b, 1:max_conv_len, cn0 : cn0 + can],
                                            shift_ub[0:nshift, 0:can],
                                        )
                                        T.copy(
                                            shift_ub[0:nshift, 0:can],
                                            new_conv_state[b, 0:nshift, cn0 : cn0 + can],
                                        )
                else:
                    if which == 0:
                        if m0 == 0:
                            shift_s = T.alloc_shared((nshift, block_n), dtype_str)
                            for b in T.serial(batch):
                                T.copy(
                                    conv_state[b, 1:max_conv_len, n0 : n0 + an],
                                    shift_s[0:nshift, 0:an],
                                )
                                T.copy(
                                    shift_s[0:nshift, 0:an],
                                    new_conv_state[b, 0:nshift, n0 : n0 + an],
                                )
"""

VARIANTS = {
    # ---- round 1: cache-shift restructure -------------------------------------
    "r1_c1_shift_blocks": (
        PROJECT,
        [
            (SHIFT_CONSTS_OLD, SHIFT_CONSTS_NEW),
            (SHIFT_GRID_OLD, SHIFT_GRID_NEW_DEDICATED),
            (SHIFT_BODY_OLD, SHIFT_BODY_NEW_DEDICATED),
            (PROJ_HEAD, '"""Stage 4 r1_c1 (perf_opt): cache shift on dedicated per-batch blocks.'),
        ],
    ),
    "r1_c2_shift_inline": (
        PROJECT,
        [
            (SHIFT_CONSTS_OLD, SHIFT_CONSTS_NEW),
            (SHIFT_GRID_OLD, SHIFT_GRID_NEW_INLINE),
            (SHIFT_BODY_OLD, SHIFT_BODY_NEW_INLINE),
            (PROJ_HEAD, '"""Stage 4 r1_c2 (perf_opt): cache shift folded into the first batch blocks.'),
        ],
    ),
    "r1_c3_shift_colsplit": (
        PROJECT,
        [
            (SHIFT_CONSTS_OLD, SHIFT_CONSTS_NEW_C3),
            (SHIFT_GRID_OLD, SHIFT_GRID_NEW_COLSPLIT),
            (SHIFT_BODY_OLD, SHIFT_BODY_NEW_COLSPLIT),
            (PROJ_HEAD, '"""Stage 4 r1_c3 (perf_opt): cache shift split per (batch row, d chunk).'),
        ],
    ),
    "r1_c4_shift_adaptive": (
        PROJECT,
        [
            (SHIFT_CONSTS_OLD, SHIFT_CONSTS_NEW_C4),
            (SHIFT_GRID_OLD, SHIFT_GRID_NEW_C4),
            (SHIFT_BODY_OLD, SHIFT_BODY_NEW_C4),
            (PROJ_HEAD,
             '"""Stage 4 r1_c4 (perf_opt): shape-adaptive cache shift (wide form when batch > n_tiles).'),
        ],
    ),
    # ---- round 2: step launch -------------------------------------------------
    "r2_c1_step_tapwindow": (
        STEP,
        [
            (TAP_CONSTS_OLD, TAP_CONSTS_NEW),
            (TAP_SPAN_OLD, TAP_SPAN_NEW),
            (TAP_ALLOC_OLD, TAP_ALLOC_NEW),
            (TAP_LOAD_OLD, TAP_LOAD_NEW),
            ("all", TAP_USE_OLD, TAP_USE_NEW),
            (STEP_HEAD, STEP_HEAD_NEW),
        ],
    ),
}


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, (base, edits) in VARIANTS.items():
        if only and not name.startswith(only):
            continue
        kind = "project" if base == PROJECT else "step"
        src = patch(read(base), edits, name)
        out = os.path.join(PERF_OPT, f"engram_decode_{name}_{kind}.py")
        with open(out, "w") as f:
            f.write(src)
        print(f"wrote {os.path.relpath(out, PERF_OPT)}")


def read(name):
    with open(os.path.join(BASE_DIR, name)) as f:
        return f.read()


if __name__ == "__main__":
    main()
