"""TRAP-950-dynamic-base-tile-pass: a tile pass at a runtime base aborts the device.

Measured 2026-11-23 on ascend-950pr-9579 (dav-3510), TileLang ascend 0.1.15,
CANN 9.2.1.  Task: `_chain_speculative_sampling_kernel` Stage 3.

Claim under test
----------------
Recomputing the crossing tile of the residual row as **one** ``T.Parallel(tile)``
pass at the runtime base ``pick * tile`` -- with the predicate
``pick * tile + i < vocab`` and the (guarded) load ``target[at_t + pick*tile + i]``
-- aborts the device task on some vocabularies:

    RuntimeError: npuSynchronizeDevice: ... device error type 3, error code is 507035

Measured shapes (batch, num_draft, vocab), one process a shape:

    (2, 2, 16000)    abort        (2, 2, 8193)     ok
    (112, 1, 32000)  abort        (2, 2, 128256)   ok

The same pipeline with the crossing tile recomputed through a **serial loop over
tiles** (``if t == pick: T.Parallel(tile)``, i.e. the fold pass's own
compile-time base ``t * tile``) returns the reference's ``num_accepted`` on every
one of those shapes.  Nothing is numerically wrong before the abort: either the
row completes and matches the reference or the core raises.

Why it matters here
-------------------
The operator locates the drawn token in four levels (row -> tile -> lane ->
element) and re-reads only the crossing tile, which is the design's traffic
argument (``1 + 1/ntile`` passes over a row).  The tile index therefore has to
stay a serial loop variable -- the same access shape the fold pass already uses
-- instead of a single pass at a runtime base.  The same is true of the region
structure: one ``with T.SimtVF`` spanning the serial row loop (reducers allocated
once for all rows) aborts on ``(2, 2, 16000)`` as well, so the region is opened
and closed inside the row, as the source kernel does.

Run
---
    cd examples/chain_speculative_sampling/_chain_speculative_sampling_kernel
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    /workspace/venv-ascend950/bin/python repro/TRAP-950-dynamic-base-tile-pass.py

Exit code 0 means both arms behaved as this note claims (the shipped spelling is
exact on the shape, the flattened one aborts).
"""

import importlib.util
import os
import subprocess
import sys
import tempfile

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(TASK_DIR, "_chain_speculative_sampling_kernel.py")

# The shipped spelling: the crossing tile through a serial tile loop.
SHIPPED = '''                        for t in T.serial(ntile):
                            if t == pick[0]:
                                for i in T.Parallel(tile):
                                    r_ub[i] = _residual(
                                        draft_probs, target_probs, at_t, at_d, n_s,
                                        num_draft, vocab, t * tile, i,
                                    )
'''
# The flattened spelling: one pass at the runtime base.
FLAT = '''                        for i in T.Parallel(tile):
                            r_ub[i] = _residual(
                                draft_probs, target_probs, at_t, at_d, n_s,
                                num_draft, vocab, pick[0] * tile, i,
                            )
'''


def run_arm(arm, batch, num_draft, vocab):
    """One arm: run the pipeline on one shape and compare ``num_accepted``."""
    src = open(SRC).read()
    if arm == "flat":
        assert SHIPPED in src, "the shipped recompute changed; re-anchor the repro"
        src = src.replace(SHIPPED, FLAT)
    path = tempfile.NamedTemporaryFile(
        dir=TASK_DIR, prefix=f"_repro_{arm}_", suffix=".py", delete=False
    ).name
    try:
        with open(path, "w") as fh:
            fh.write(src)
        spec = importlib.util.spec_from_file_location(f"arm_{arm}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        draft, ids, target, seed, offset = mod._make_inputs(batch, num_draft, vocab, 27)
        _, ref_acc = mod.golden_chain_speculative_sampling(draft, ids, target, seed, offset)
        args = (draft.to("npu"), ids.to("npu"), target.to("npu"),
                seed.to("npu"), offset.to("npu"))
        _, got_acc = mod._run(*args)
        import torch

        torch.npu.synchronize()
        if not bool((got_acc.cpu() == ref_acc).all()):
            print(f"arm {arm}: WRONG num_accepted", flush=True)
            return 2
        print(f"arm {arm}: OK", flush=True)
        return 0
    finally:
        os.unlink(path)


def main():
    if len(sys.argv) > 1:
        arm = sys.argv[1]
        b, n, v = (int(x) for x in sys.argv[2:5])
        sys.exit(run_arm(arm, b, n, v))

    shape = (2, 2, 16000)
    codes = {}
    for arm in ("shipped", "flat"):
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), arm,
             str(shape[0]), str(shape[1]), str(shape[2])],
            capture_output=True, text=True,
        )
        codes[arm] = proc.returncode
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:]
        print(f"arm {arm} on {shape}: exit {proc.returncode} {tail}", flush=True)
    assert codes["shipped"] == 0, f"the shipped spelling failed: {codes['shipped']}"
    assert codes["flat"] != 0, "the flattened spelling ran; the trap note is stale"
    print("[TRAP-950-dynamic-base-tile-pass] reproduced: the shipped serial-tile "
          "recompute is exact, the runtime-base pass aborts", flush=True)


if __name__ == "__main__":
    main()
