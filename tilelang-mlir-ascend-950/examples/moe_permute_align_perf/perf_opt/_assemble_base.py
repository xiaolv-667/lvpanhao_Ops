"""Assemble perf_opt/moe_permute_align.py from the three frozen kernel modules.

The Stage 4 deliverable keeps the exact factory contract the frozen wrapper
imports (``_make_align_kernel`` / ``_make_scatter_kernel`` /
``_make_small_batch_kernel``), so the three Stage 3 modules are copied verbatim
into one self-contained file and only the tuned parts are edited afterwards.

Run once from examples/TileOPs:  python ../moe_permute_align_perf/perf_opt/_assemble_base.py
"""

import pathlib
import re

KDIR = pathlib.Path(
    "tileops/kernels/moe/moe_permute_align/moe_permute_align_kernel"
)
OUT = pathlib.Path("../moe_permute_align_perf/perf_opt/moe_permute_align.py")

HEADER = '''# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""MoE token-to-expert alignment + route scatter, Ascend 950PR (Stage 4 tuned).

One self-contained module carrying the three kernel factories the frozen wrapper
``tileops/kernels/moe/moe_permute_align/moe_permute_align.py`` imports:

* ``_make_align_kernel(numel, num_experts, block_size)`` -> ``align(flat,
  sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum)``
* ``_make_scatter_kernel(numel, num_experts, block_size)`` -> ``scatter(flat,
  sorted_token_ids, cumsum)``
* ``_make_small_batch_kernel(numel, num_experts, block_size)`` -> ``small(flat,
  sorted_token_ids, expert_ids, num_tokens_post_pad)``

Stage 4 restructure (unchanged semantics, see ``perf_opt/opt_log.md``):

1. the align stage keeps only the per-core histogram (its launch 1) and hands
   the per-core partial counts to the scatter stage through a persistent
   scratch;
2. the scatter stage becomes the second, wider launch: it merges the partial
   counts, derives the padded prefix sum, writes ``cumsum`` /
   ``num_tokens_post_pad`` / ``expert_ids`` / the sentinel, and places every
   route with the *deterministic* slot
   ``cumsum[e] + (routes of e in earlier cores) + (routes of e earlier in this
   core)`` -- no global atomics;
3. ``expert_ids`` is produced by letting each expert expand its own block range
   instead of an ``O(num_experts)`` rank count per block.

Program entry: ``python moe_permute_align.py --level L0|L1|boundary`` runs the
combined-pipeline precision harness (align+scatter, or the small-batch path).
"""

import argparse
import functools
import math
import os
import time

import torch
import torch_npu  # noqa: F401 -- registers torch.npu
import tilelang
import tilelang.ascend.language as T


'''


def _trim_section(text):
    """Drop the trailing '# ---' + '# N. Launch' banner a section cut leaves."""
    lines = text.rstrip("\n").splitlines(keepends=True)
    while lines and (lines[-1].startswith("# ---") or re.match(r"^# \d+\. ", lines[-1])):
        lines.pop()
    return "".join(lines).rstrip("\n") + "\n"


def block(path, start_pat, end_pat=None):
    text = pathlib.Path(path).read_text().splitlines(keepends=True)
    start = next(i for i, line in enumerate(text) if re.match(start_pat, line))
    if end_pat is None:
        end = len(text)
    else:
        end = next(i for i, line in enumerate(text[start + 1:], start + 1)
                   if re.match(end_pat, line))
    return "".join(text[start:end])


def main():
    align = KDIR / "_make_align_kernel.py"
    scatter = KDIR / "_make_scatter_kernel.py"
    small = KDIR / "_make_small_batch_kernel.py"

    align_consts = block(align, r"^FILL_TILE = ", r"^# ---")
    align_body = block(align, r"^def _align_up", r"^MAX_DIFF = ")
    align_body = _trim_section(align_body)
    scatter_consts = block(scatter, r"^_THREADS = ", r"^# ---")
    scatter_body = block(scatter, r"^@functools\.lru_cache\(maxsize=32\)",
                         r"^_RESULTS = ")
    scatter_body = _trim_section(scatter_body)
    sb_consts = block(small, r"^W = 32", r"^# ---")
    sb_body = block(small, r"^def _align_up", r"^_RESULTS = ")
    sb_body = _trim_section(sb_body)

    # The small-batch module brings its own helpers with clashing names: give
    # everything it defines outside its two factories a SB_ prefix.
    sb_body = sb_body.replace("def _align_up(", "def _sb_align_up(")
    sb_body = sb_body.replace("_align_up(", "_sb_align_up(")
    sb_body = sb_body.replace("def _geometry(", "def _sb_geometry(")
    sb_body = sb_body.replace("_geometry(", "_sb_geometry(")
    sb_body = sb_body.replace("_sb_sb_", "_sb_")
    sb_body = sb_body.replace("def _compile_small(", "def _compile_sb_small(")
    sb_body = sb_body.replace("return _compile_small(", "return _compile_sb_small(")
    sb_body = re.sub(r"\b_compile_small\b", "_compile_sb_small", sb_body)
    for name in ("W", "UNROLL", "FILL_TILE", "IDS_LANES", "MAX_EXPERTS",
                 "MAX_CHUNK", "XFER_GUARD", "SCAN_STEPS"):
        sb_consts = re.sub(rf"^{name} = ", f"SB_{name} = ", sb_consts, flags=re.M)
        sb_body = re.sub(rf"\b{name}\b", f"SB_{name}", sb_body)
    # the docstrings must not claim the old names
    sb_body = sb_body.replace("SB_SB_", "SB_")

    out = [HEADER]
    out.append("# ---------------------------------------------------------------------------\n")
    out.append("# Align stage: build-time planning + both paths\n")
    out.append("# ---------------------------------------------------------------------------\n")
    out.append(align_consts + "\n" + align_body)
    out.append("\n# ---------------------------------------------------------------------------\n")
    out.append("# Scatter stage\n")
    out.append("# ---------------------------------------------------------------------------\n")
    out.append(scatter_consts + "\n" + scatter_body)
    out.append("\n# ---------------------------------------------------------------------------\n")
    out.append("# Fused small-batch stage (wrapper path: numel < 1024 and E <= 32)\n")
    out.append("# ---------------------------------------------------------------------------\n")
    out.append(sb_consts + "\n" + sb_body)
    out.append("\n" + (pathlib.Path(__file__).with_name("_harness.part.py")).read_text())
    OUT.write_text("".join(out))
    text = "".join(out)
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
