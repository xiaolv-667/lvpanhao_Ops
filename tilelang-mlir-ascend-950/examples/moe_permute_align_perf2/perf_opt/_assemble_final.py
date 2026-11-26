"""Assemble the final ``perf_opt/moe_permute_align.py`` from the validated parts.

The final module is self-contained (the frozen wrapper will import it as
``moe_permute_align_kernel.perf_opt._make_align_kernel``), so it carries:

* the Stage 4 tuned align stage (atomic histogram launch + merge/scan/metadata
  launch, ``expert_ids`` by expert expansion, bulk-merge copy above the measured
  core threshold, and -- for single-core geometries -- the placement itself);
* the Stage 4 tuned scatter stage (deterministic slot from the per-core cursor
  row; a no-op for single-core geometries);
* the Stage 3 fused small-batch kernel verbatim (the wrapper's own
  ``numel < 1024 and E <= 32`` path, never on a benchmark workload);
* the combined-pipeline precision harness (``--level L0|L1|boundary``).

Run from ``perf_opt/``:  python _assemble_final.py
"""

import pathlib

HERE = pathlib.Path(__file__).resolve().parent

HEAD = '''# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""MoE token-to-expert align + route scatter, Ascend 950PR (Stage 4 tuned).

Self-contained replacement for the three Stage 3 kernel modules the frozen
wrapper ``tileops/kernels/moe/moe_permute_align/moe_permute_align.py`` imports:

* ``_make_align_kernel(numel, num_experts, block_size)`` ->
  ``align(flat, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum)``
* ``_make_scatter_kernel(numel, num_experts, block_size)`` ->
  ``scatter(flat, sorted_token_ids, cumsum)``
* ``_make_small_batch_kernel(numel, num_experts, block_size)`` ->
  ``small(flat, sorted_token_ids, expert_ids, num_tokens_post_pad)``

Every launch keeps the Stage 3 output contract bit for bit: ``cumsum`` holds the
padded per-expert starts, ``expert_ids`` the block owners (``num_experts`` past
the padded length), ``num_tokens_post_pad`` the padded total, and
``sorted_token_ids`` holds the sentinel ``numel`` outside each expert's routed
slots and each route index exactly once inside its own expert's range (the order
*within* a range is the contract's documented free choice).

Tuned structure (each step measured, see ``perf_opt/opt_log.md``):

1. **Histogram launch** -- one on-chip ``atomic_add`` per route into the core's
   own ``int32`` histogram row, instead of ``numel x num_experts`` lane
   comparisons (29.1 us -> 3.8 us at 32768 routes).
2. **Align launch** -- merge the per-core rows (one bulk copy above
   ``_BULK_MERGE_MIN_CORES``, per-row copies below it), padded prefix sum,
   ``cumsum`` / ``num_tokens_post_pad``, ``expert_ids`` by expert-side expansion
   (no ``O(num_experts)`` rank count per block), the sentinel fill, and each
   core's slot-cursor row into the shared scratch.
3. **Scatter launch** -- every route's slot is *derived*, not allocated by a
   global atomic: ``cumsum[e]`` + the routes of ``e`` on earlier cores (from the
   histogram rows) + the routes of ``e`` already placed here (on-chip cursor).
   The Stage 3 per-route GM atomic cost 586 us at 32768 routes; this costs one
   on-chip bump plus the store the contract needs anyway.
4. Single-core geometries (``g == 1``) place inside the align launch, so the
   call is two launches instead of three.

Tuned defaults (module constants, referenced by the wrapper's perf_opt block):

* ``_THREADS = 128`` -- SIMT lane width of both the histogram and the placement;
* ``_UB_PARTIAL_BUDGET = 96 * 1024`` -- UB bytes the whole per-core histogram may
  occupy, which bounds the core count (it only binds for very wide expert
  tables);
* ``_BULK_MERGE_MIN_CORES = 16`` -- core count above which the merge copies the
  whole histogram window in one descriptor instead of per row (measured: 56
  cores -17.4 us, 8 cores +2.1 us).

Program entry: ``python moe_permute_align.py --level L0|L1|boundary`` runs the
combined-pipeline precision harness (align+scatter or the fused small-batch
path) against a torch CPU golden derived from the operator semantics.
"""

import argparse
import functools
import math
import time

import torch
import torch_npu  # noqa: F401 -- registers torch.npu
import tilelang
import tilelang.ascend.language as T

FILL_TILE = 2048  # sentinel tile, in int32 elements per bulk store
VECTOR_CORES = 56  # physical Vector cores of Ascend950PR_9579
_THREADS = 128  # SIMT lane width of the histogram and placement launches
_UB_PARTIAL_BUDGET = 96 * 1024  # UB bytes the whole per-core histogram may take
_BULK_MERGE_MIN_CORES = 16  # merge the histogram window in one descriptor


# ---------------------------------------------------------------------------
# Build-time geometry
# ---------------------------------------------------------------------------
def _align_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _scan_steps(lanes):
    return 0 if lanes <= 1 else int(math.ceil(math.log2(lanes)))


def _geometry(numel, num_experts, block_size):
    E = num_experts
    B = block_size
    p_total = numel + (E + 1) * (B - 1)
    return {
        "numel": numel,
        "E": E,
        "B": B,
        "E_pad": _align_up(max(E, 1), 64),
        "P": p_total,
        "NB": (p_total + B - 1) // B,
        "steps": _scan_steps(_align_up(max(E, 1), 64)),
        "shift": B.bit_length() - 1 if (B & (B - 1)) == 0 else None,
    }


def _fused_plan(numel, E, B):
    """Core count, padded expert count and routes per core (all launches)."""
    e_pad = _align_up(max(E, 1), 64)
    g = max(1, min(VECTOR_CORES, (numel + 511) // 512))
    while g > 1 and g * e_pad * 4 > _UB_PARTIAL_BUDGET:
        g = (g + 1) // 2
    return g, e_pad, (numel + g - 1) // g


_SCRATCH = {}


def _scratch(numel, E, B, device):
    """(per-core histogram rows, per-core slot cursors) for one geometry."""
    g, e_pad, _ = _fused_plan(numel, E, B)
    key = (numel, E, B, g, e_pad)
    bufs = _SCRATCH.get(key)
    if bufs is None or bufs[0].device != device:
        bufs = (
            torch.empty((g, e_pad), dtype=torch.int32, device=device),
            torch.empty((g, e_pad), dtype=torch.int32, device=device),
        )
        _SCRATCH[key] = bufs
    return bufs


'''

TAIL = '''

class _Place:
    """scatter callable: the wrapper's three args, plus this core's cursor row."""

    def __init__(self, numel, E, B, g):
        self._numel = numel
        self._E = E
        self._B = B
        self._prog = _compile_place(numel, E, B, g)

    def __call__(self, flat, sorted_token_ids, cumsum):
        _partial, cursor = _scratch(self._numel, self._E, self._B, flat.device)
        return self._prog(flat, sorted_token_ids, cursor)


class _Noop:
    """The scatter stage when the align stage already placed every route."""

    def __call__(self, *args, **kwargs):
        return None


class _AlignStage:
    """align callable: histogram launch, then merge/scan/metadata [+ placement]."""

    def __init__(self, numel, E, B, g):
        self._numel = numel
        self._E = E
        self._B = B
        self._g = g
        self._hist = _compile_hist_atomic(numel, E, B, g)
        self._align = _compile_align_stage(numel, E, B, g, g == 1)

    def __call__(self, flat, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum):
        partial, cursor = _scratch(self._numel, self._E, self._B, flat.device)
        self._hist(flat, partial)
        self._align(flat, partial, cursor, sorted_token_ids, expert_ids,
                    num_tokens_post_pad, cumsum)


def _make_align_kernel(numel: int, num_experts: int, block_size: int):
    """Build the align factory for one ``(numel, num_experts, block_size)``."""
    numel = int(numel)
    E = int(num_experts)
    B = int(block_size)
    g, _, _ = _fused_plan(numel, E, B)

    def _build():
        return _AlignStage(numel, E, B, g)

    _build.path = "L"
    _build.g_h = g
    _build.g_a = g
    return _build


@functools.lru_cache(maxsize=32)
def _make_scatter_kernel(numel: int, num_experts: int, block_size: int):
    """Build the scatter factory: deterministic placement, no global atomics."""
    if numel < 1:
        raise ValueError(f"the scatter needs at least one route, got numel={numel}")
    numel = int(numel)
    E = int(num_experts)
    B = int(block_size)
    g, _, _ = _fused_plan(numel, E, B)

    def _build():
        return _Noop() if g == 1 else _Place(numel, E, B, g)

    return _build


'''


def part(name):
    return (HERE / name).read_text()


def main():
    hist = part("_final_hist.part.py")
    align = part("_final_align.part.py")
    place = part("_final_place.part.py")
    base = (HERE / "_base_module.py").read_text()
    start = base.index("# ---------------------------------------------------------------------------\n"
                       "# Fused small-batch stage")
    end = base.index("# ---------------------------------------------------------------------------\n"
                     "# Combined-pipeline precision harness")
    sb = base[start:end]
    harness = part("_harness.part.py")
    text = HEAD + hist + "\n" + align + "\n" + place + "\n" + TAIL + sb + harness
    (HERE / "MoEPermuteAlignFwdOp.py").write_text(text)
    print(f"wrote MoEPermuteAlignFwdOp.py ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
