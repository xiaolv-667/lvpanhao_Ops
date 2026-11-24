"""TRAP-950-grid-over-65535 (facet: a grid whose remainder is non-zero does NOT
abort -- it silently runs only ``grid % 65536`` blocks) and the head-slab fix.

First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510),
2026-11-24; origin_task ``mla_varlen_bwd`` / ``_bwd_kernel`` Stage 3.

Claim under test
----------------
``T.Kernel(grid)`` carries the block index in 16 bits.  65536 and 131072 abort
(``device error type 3, error code 507035``, see
``TRAP-950-grid-over-65535.py``), but those two are exact multiples of 65536.
With a non-zero remainder the device **does not report anything**: it executes
the first ``grid % 65536`` blocks and drops the rest, so the launch looks faster
and part of the output stays at its initial value.

Why it matters here
-------------------
``_bwd_kernel``'s logical block count is ``heads * (total_kv // 64 + batch)``,
i.e. it contains a **run-time** quantity, so no compile-time check can catch it:
DS-V3-8x4k (B=8, H=128, L=4096) needs 66,560 blocks and ran 1,024 of them
(requests 1..7 of ``dq/dk/dv`` were exactly 0, latency 20.75 ms instead of
~488 ms).  The fix splits the head axis into a compile-time **slab**
(``grid = tiles * slab``, ``tile = bid // slab``,
``h = h_off[0] + bid % slab``, ``if h < heads``) and launches one program per
slab, which keeps one launch <= 65,535 blocks and needs no dynamic divisor.

What this repro asserts (tiny, ~1 s of device time, no attention data)
---------------------------------------------------------------------
1. ``grid = 66,560`` (the pre-fix launch) marks **exactly 1,024** of the 66,560
   logical tasks, and they are the first 1,024: the silent-truncation facet.
2. ``grid = 520 * 32 = 16,640`` launched 4 times with ``h_off = 0, 32, 64, 96``
   (the fix) marks **all 66,560** tasks, each exactly once.

Run
---
    cd examples/mla_varlen_bwd/_bwd_kernel
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
        repro/TRAP-950-grid-over-65535-silent-truncation.py

Exit code 0 means both arms behaved as this note claims.
"""

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T

TILES = 520  # total_kv // 64 + batch of ds-v3-8x4k
HEADS = 128
SLAB = 32  # HEAD_SLAB_DEFAULT / plan.slab
GRID_OLD = TILES * HEADS  # 66,560 = 65,536 + 1,024
GRID_SLAB = TILES * SLAB  # 16,640, four launches


def build_old():
    """Pre-fix mapping: grid = tiles * heads, tile = bid // heads, h = bid % heads."""

    @T.prim_func
    def main(out: T.Tensor((GRID_OLD,), "float32")) -> None:
        with T.Kernel(GRID_OLD) as bid, T.SimtVF(threads=128):
            for i in T.Parallel(128):
                if i == 0:
                    out[bid] = T.float32(1)

    return tilelang.compile(main, target="ascend", out_idx=[])


def build_slab():
    """Fix: grid = tiles * slab, the head origin arrives as a run-time int32[1]."""

    @T.prim_func
    def main(out: T.Tensor((GRID_OLD,), "float32"),
             h_off: T.Tensor((1,), "int32")) -> None:
        with T.Kernel(GRID_SLAB) as bid, T.SimtVF(threads=128):
            tile = bid // SLAB  # static divisor (compile-time slab)
            h = h_off[0] + bid % SLAB  # run-time head origin + static slot
            for i in T.Parallel(128):
                if (i == 0) & (h < HEADS):  # tail-slab guard
                    out[tile * HEADS + h] = T.float32(1)

    return tilelang.compile(main, target="ascend", out_idx=[])


def main():
    problems = []

    # ---- arm 1: the pre-fix launch, 66,560 blocks -> 1,024 executed ----------
    fn_old = build_old()
    out_old = torch.zeros(GRID_OLD, device="npu")
    fn_old(out_old)
    torch.npu.synchronize()
    got_old = out_old.cpu()
    count_old = int((got_old != 0).sum())
    print(f"grid={GRID_OLD}: {count_old} of {GRID_OLD} blocks executed "
          f"(remainder {GRID_OLD % 65536})", flush=True)
    if count_old != GRID_OLD % 65536:
        problems.append(f"old form executed {count_old} blocks, expected "
                        f"{GRID_OLD % 65536}")
    if int((got_old[:GRID_OLD % 65536] != 0).sum()) != count_old:
        problems.append("the executed blocks are not the first grid % 65536 ones")

    # ---- arm 2: the head-slab launch, four times 16,640 blocks ---------------
    fn_slab = build_slab()
    out_new = torch.zeros(GRID_OLD, device="npu")
    h_off = torch.arange(0, HEADS, SLAB, dtype=torch.int32, device="npu")
    for index in range(h_off.numel()):
        fn_slab(out_new, h_off[index:index + 1])
    torch.npu.synchronize()
    got_new = out_new.cpu()
    count_new = int((got_new != 0).sum())
    slots = got_new.reshape(TILES, HEADS)
    print(f"slab grid={GRID_SLAB} x {h_off.numel()} launches: {count_new} of "
          f"{GRID_OLD} tasks covered, min/max per task {float(slots.min())}/"
          f"{float(slots.max())}", flush=True)
    if count_new != GRID_OLD:
        problems.append(f"slab form covered {count_new} tasks, expected {GRID_OLD}")
    if float(slots.min()) != 1.0 or float(slots.max()) != 1.0:
        problems.append("the (tile, head) -> block map is not a bijection")

    if problems:
        print("[TRAP_CONFIRMED_FAIL] " + "; ".join(problems), flush=True)
        return 1
    print(f"[TRAP_CONFIRMED] {GRID_OLD} blocks silently run {count_old}; the "
          f"{SLAB}-head slab covers all {GRID_OLD} tasks in {h_off.numel()} launches",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
