"""Assemble the Stage-4 deliverable `perf_opt/EngramDecodeFwdOp.py`.

One decode step is two TileLang launches, so the deliverable is one module holding
both factories under the names the wrapper's perf_opt switch block imports:

    from .perf_opt.EngramDecodeFwdOp import (
        ALIGNMENT, _engram_project_kernel, _engram_step_kernel,
        project_default_config, step_default_config,
    )

Sources (both read-only inputs to this script):
  * project launch : `perf_opt/engram_decode_round1_c4_shift_adaptive_project.py`
    (Stage-3 baseline + the round-1 winning patch, see mkvariant.py)
  * step launch    : the frozen Stage-3 `_engram_step_kernel.py` (unchanged — the
    measured differences of every round-2 candidate were inside noise)

Only the kernel bodies are copied; the per-module L0 runners stay in the candidate
files.  This module instead carries an **operator-level** L0/L1 harness that drives
both launches and compares against the repository torch reference
(`tileops.workloads.sequence_modeling.engram_decode_step_torch`) with the op
tolerances from `tests/ops/test_engram.py::_tol_decode`.

Usage (cwd = examples/TileOPs):

    python perf_opt/EngramDecodeFwdOp.py --level L0 | L1 | boundary | all
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PERF_OPT = os.path.dirname(HERE)
BASE_DIR = os.path.abspath(os.path.join(
    PERF_OPT, "..", "..", "TileOPs", "tileops", "kernels", "sequence_modeling",
    "engram_decode", "engram_decode_kernel"))
PROJECT_SRC = os.path.join(PERF_OPT, "engram_decode_round1_c4_shift_adaptive_project.py")
STEP_SRC = os.path.join(BASE_DIR, "_engram_step_kernel.py")
OUT = os.path.join(PERF_OPT, "EngramDecodeFwdOp.py")


def slice_between(src, start, end):
    i = src.index(start)
    j = src.index(end, i) + len(end)
    return src[i:j]


def main():
    with open(PROJECT_SRC) as f:
        proj = f.read()
    with open(STEP_SRC) as f:
        step = f.read()

    proj_kernel = slice_between(
        proj,
        "@functools.lru_cache(maxsize=None)\ndef _build_project",
        '    return tilelang.compile(engram_project, target="ascend", out_idx=[])',
    )
    proj_entry = slice_between(
        proj,
        "def _engram_project_kernel(batch, d_mem, d_padded, max_conv_len, dtype):",
        '        "num_stages": NUM_STAGES,\n    }',
    )
    step_kernel = slice_between(
        step,
        "def _next_pow2(n):",
        '    return {"threads": min(512, _next_pow2(d_padded))}',
    )
    # rename the two `default_config`s so one module can expose both
    proj_entry = proj_entry.replace("def default_config(d_pad):", "def project_default_config(d_pad):")
    step_kernel = step_kernel.replace("def default_config(d_padded):",
                                      "def step_default_config(d_padded):")
    step_kernel = step_kernel.replace('default_config(d_padded)["threads"]',
                                      'step_default_config(d_padded)["threads"]')
    proj_entry = proj_entry.replace(
        '    """Frozen Stage 3 tiling (DESIGN.md 5.6)."""',
        '    """Stage 4 tiling: baseline block_m/block_n/block_k/num_stages (unchanged —\n'
        '    the coarse sweep over {block_n, block_k, num_stages} moved serving-b4 by\n'
        '    <2%, i.e. the project launch is bandwidth/floor bound, not tiling bound)."""')

    constants = slice_between(
        proj,
        "# The host rounds the hidden width up",
        'BLOCK_K = 256\nNUM_STAGES = 2',
    )
    constants = constants + "\n\n" + slice_between(
        proj,
        "# Stage 4 O-1d knobs:",
        "SHIFT_UB_BUDGET = 98304",
    )
    imports = """import functools
import math

import torch
import torch_npu  # noqa: F401 -- registers torch.npu
import tilelang
import tilelang.ascend.language as T"""

    header = open(os.path.join(HERE, "_final_header.py")).read()
    harness = open(os.path.join(HERE, "_final_harness.py")).read()

    body = "\n\n".join([
        header.rstrip(),
        imports,
        constants.strip(),
        "# " + "-" * 73 + "\n# Launch 1 of 2: the two projections + the cache shift\n# " + "-" * 73,
        proj_kernel.strip(),
        proj_entry.strip(),
        "# " + "-" * 73 + "\n# Launch 2 of 2: norms / gate / dilated conv / SiLU (Stage-3 code, unchanged)\n# " + "-" * 73,
        step_kernel.strip(),
        harness.strip(),
    ])
    with open(OUT, "w") as f:
        f.write(body + "\n")
    print(f"wrote {OUT} ({len(body.splitlines())} lines)")


if __name__ == "__main__":
    main()
