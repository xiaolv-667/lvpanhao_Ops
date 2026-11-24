"""TRAP-950-grid-over-65535: a core grid of 65536 blocks aborts the device task.
First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-23; origin_task batch2-20261123.

Measured 2026-11-23 on ascend-950pr-9579 (dav-3510), TileLang ascend 0.1.15,
CANN 9.2.1.  Task: `_chain_speculative_sampling_kernel` Stage 3.

Claim under test
----------------
`T.Kernel(grid)` maps blocks onto the vector cores, and `grid` is carried in a
16-bit block index: a launch of 65535 blocks runs, a launch of 65536 (and
131072) blocks aborts with

    RuntimeError: npuSynchronizeDevice: ... device error type 3, error code is 507035

and the AIV error bitmap in the CANN device log has every vector core set.  The
abort is *not* a wrong-result or an out-of-range access inside the kernel: the
body only touches `Y[bx]` and `X[bx]`.

Why it matters here
-------------------
The operator folds ``B`` rows onto at most 56 vector cores
(``blocks = min(batch, 56)``, ``per_core = ceil(batch / blocks)`` rows a core).
Spelling the fold the other way round -- ``blocks = batch`` and no row loop --
is legal up to ``batch = 65535`` and aborts above it, so a 65536-row call (the
distribution probe of the correctness suite) cannot use that mapping.  The
per-core row loop is also what the design's sec 5.5 mapping chose; this repro
records the hard limit behind the choice.

Run
---
    cd examples/chain_speculative_sampling/_chain_speculative_sampling_kernel
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    /workspace/venv-ascend950/bin/python repro/TRAP-950-grid-over-65535.py

Exit code 0 means both arms behaved as this note claims (65535 runs and is
correct, 65536 aborts).
"""

import os
import subprocess
import sys

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T


def build(grid):
    @T.prim_func
    def main(X: T.Tensor((grid,), "float32"), Y: T.Tensor((grid,), "float32")) -> None:
        with T.Kernel(grid) as bx, T.SimtVF(threads=128):
            for i in T.Parallel(128):
                if i == 0:
                    Y[bx] = X[bx] + T.float32(1)

    return tilelang.compile(main, target="ascend", out_idx=[])


def run_one(grid):
    """One arm: launch and check the result; returns the process exit code."""
    fn = build(grid)
    x = torch.arange(grid, dtype=torch.float32)
    xd, yd = x.to("npu"), torch.zeros(grid, device="npu")
    fn(xd, yd)
    torch.npu.synchronize()
    if not torch.equal(yd.cpu(), x + 1):
        print(f"grid={grid}: WRONG result", flush=True)
        return 2
    print(f"grid={grid}: OK", flush=True)
    return 0


def main():
    grid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if grid is not None:
        sys.exit(run_one(grid))
    # Driver: one child a grid, so an abort cannot take the report down with it.
    codes = {}
    for g in (65535, 65536):
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), str(g)],
                              capture_output=True, text=True)
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:]
        codes[g] = proc.returncode
        print(f"grid={g}: exit {proc.returncode} {tail}", flush=True)
    assert codes[65535] == 0, f"a legal grid of 65535 blocks failed: {codes[65535]}"
    assert codes[65536] != 0, "a grid of 65536 blocks ran: the limit claim is stale"
    print("[TRAP-950-grid-over-65535] reproduced: 65535 runs, 65536 aborts", flush=True)


if __name__ == "__main__":
    main()
