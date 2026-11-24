#!/usr/bin/env python3
"""Launch plan of the packed ``_bwd_kernel``: **per-launch core grid <= 65535**.

Platform fact this module exists for (DESIGN.md 5.6, v1):

``T.Kernel(grid)`` carries the block index in 16 bits on Ascend950PR.  ``grid <=
65535`` runs; ``grid = 65536`` (and ``131072``) aborts with ``device error type
3, error code 507035``; and -- the dangerous form found on 2026-11-24 -- a grid
whose remainder is non-zero does **not** abort, it silently executes only
``grid % 65536`` blocks and leaves the rest of the output untouched.

The packed kernel's logical block count is

    logical_blocks = heads * tiles,        tiles = total_kv // 64 + batch

which depends on the **run-time** token count, so no compile-time check can
catch it: the host side is the only place to plan and assert.  The fix keeps the
kernel body bit-identical and splits the head axis into slabs, one launch per
slab:

    grid(launch i) = tiles * slab          (``slab`` is a compile-time constant)
    tile = bid // slab,  h = h_off[i] + bid % slab

Static constants only (``bid // slab`` / ``bid % slab`` keep a compile-time
divisor, as DESIGN.md 1.6.3 candidate 7 requires); the per-launch head origin
``h_off[i]`` is passed as a one-element int32 tensor so one compiled program
serves every slab.

This file is pure Python (no torch / tilelang import) so it can be exercised on
a CPU-only host, which is exactly what ``verify_equiv.py`` item A4 does.  Stage 3
inlines ``plan_launch`` verbatim into ``_bwd_kernel.py`` (the kernel must stay
self-contained for the TileOPs port); ``verify_equiv.py`` A4 plus the L0 launch
plan case keep the two copies in agreement.
"""

from __future__ import annotations

from dataclasses import dataclass

NPU_GRID_MAX = 65535  # 16-bit block index, see TRAP-950-grid-over-65535
HEAD_SLAB_DEFAULT = 32  # DESIGN.md 5.6 (b): at most 4 launches for the manifest
BLOCK_N = 64  # key rows per block (DESIGN.md 5.2)


@dataclass(frozen=True)
class Launch:
    """One ``T.Kernel(...)`` invocation of the slabbed kernel."""

    index: int
    head_off: int  # value of the ``h_off`` tensor (head origin of this launch)
    active_heads: int  # heads this launch may touch (``h < heads`` guard)
    grid: int  # total_kv // 64 + batch) * slab
    idle_blocks: int  # blocks whose ``h`` lands past ``heads`` and return at once


@dataclass(frozen=True)
class Plan:
    tiles: int  # tile_upper_bound(total_kv) = total_kv // 64 + batch (per head)
    slab: int  # compile-time head slab fed to the kernel factory
    launches: tuple[Launch, ...]
    total_blocks: int  # sum of the per-launch grids (all launches)
    useful_blocks: int  # heads * tiles -- the task set the algorithm needs
    idle_blocks: int

    @property
    def max_grid(self) -> int:
        return max(launch.grid for launch in self.launches)


def tile_upper_bound(total_kv: int, batch: int, block_n: int = BLOCK_N) -> int:
    """`GroupTiling.tile_upper_bound`: sum_b ceil(len_kv_b / block_n) <= this."""
    return total_kv // block_n + batch


def plan_launch(
    batch: int,
    total_kv: int,
    heads: int,
    *,
    block_n: int = BLOCK_N,
    head_slab: int = HEAD_SLAB_DEFAULT,
) -> Plan:
    """Plan the launches of one operator call and assert the grid cap.

    ``head_slab`` is the *preferred* slab (a compile-time constant of the kernel);
    the returned ``slab`` is the effective one, shrunk to ``65535 // tiles`` when
    the token count is large enough to need it.  ``ValueError`` -- never a silent
    truncation -- when even ``slab == 1`` would exceed the cap.
    """
    if batch <= 0:
        raise ValueError(f"shape: batch must be positive, got {batch}")
    if heads <= 0:
        raise ValueError(f"shape: heads must be positive, got {heads}")
    if total_kv < 0:
        raise ValueError(f"shape: total_kv must be non-negative, got {total_kv}")
    if block_n <= 0:
        raise ValueError(f"shape: block_n must be positive, got {block_n}")
    if head_slab < 1:
        raise ValueError(f"grid: head_slab must be >= 1, got {head_slab}")

    tiles = tile_upper_bound(total_kv, batch, block_n)
    room = NPU_GRID_MAX // tiles  # max heads per launch at this token count
    slab = min(int(head_slab), heads, room)
    if slab < 1:
        raise ValueError(
            "grid: one head slab would need "
            f"tile_upper_bound={tiles} blocks > the {NPU_GRID_MAX}-block per-launch "
            f"grid cap (total_kv={total_kv}, batch={batch}, block_n={block_n}); "
            "split the token axis (smaller block_n is not allowed to be the fix) or "
            "run the operator in token chunks"
        )

    grid = tiles * slab
    if grid > NPU_GRID_MAX:  # defensive: unreachable by construction
        raise ValueError(f"grid: per-launch grid {grid} exceeds {NPU_GRID_MAX}")

    launches = []
    off = 0
    while off < heads:
        active = min(slab, heads - off)
        launches.append(
            Launch(
                index=len(launches),
                head_off=off,
                active_heads=active,
                grid=grid,
                idle_blocks=grid - tiles * active,
            )
        )
        off += slab

    plan = Plan(
        tiles=tiles,
        slab=slab,
        launches=tuple(launches),
        total_blocks=sum(launch.grid for launch in launches),
        useful_blocks=tiles * heads,
        idle_blocks=sum(launch.idle_blocks for launch in launches),
    )
    assert_launch_plan(plan, heads=heads)
    return plan


def assert_launch_plan(plan: Plan, *, heads: int) -> None:
    """Hard pre-launch assertion: cap honoured, heads covered exactly once."""
    for launch in plan.launches:
        if launch.grid > NPU_GRID_MAX:
            raise ValueError(
                f"grid: launch {launch.index} needs {launch.grid} blocks > {NPU_GRID_MAX}"
            )
        if launch.active_heads < 1 or launch.head_off + launch.active_heads > heads:
            raise ValueError(
                f"grid: launch {launch.index} covers heads "
                f"[{launch.head_off}, {launch.head_off + launch.active_heads}) "
                f"outside [0, {heads})"
            )
    covered = sum(launch.active_heads for launch in plan.launches)
    if covered != heads:
        raise ValueError(f"grid: head slabs cover {covered} heads, expected {heads}")


def describe(plan: Plan) -> str:
    """One-line summary for test logs."""
    per_launch = ", ".join(
        f"h[{launch.head_off}:{launch.head_off + launch.active_heads}) grid={launch.grid} "
        f"idle={launch.idle_blocks}"
        for launch in plan.launches
    )
    return (
        f"tiles={plan.tiles} slab={plan.slab} launches={len(plan.launches)} "
        f"max_grid={plan.max_grid} blocks={plan.total_blocks} "
        f"useful={plan.useful_blocks} idle={plan.idle_blocks} | {per_launch}"
    )


if __name__ == "__main__":  # tiny self-check of the manifest geometries
    cases = [
        ("ds-v3-8x4k", 8, 32768, 128),
        ("ds-v3-mixed", 6, 32256, 128),
        ("kimi-k2-4x8k", 4, 32768, 64),
        ("smoke", 2, 640, 8),
        ("H=126", 8, 32768, 126),
        ("H=127", 8, 32768, 127),
        ("auto-shrink 128k tokens", 8, 131072, 128),
    ]
    for name, batch, total_kv, heads in cases:
        print(f"{name:26s} {describe(plan_launch(batch, total_kv, heads))}")
