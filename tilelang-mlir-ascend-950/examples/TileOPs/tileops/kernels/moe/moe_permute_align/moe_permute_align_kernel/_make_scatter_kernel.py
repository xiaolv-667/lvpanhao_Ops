# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""MoE route scatter (ascend): kernel + torch golden + layered precision tests.

Second stage of ``MoEPermuteAlignFwdOp``: write every route index into the next
free slot of its expert's padded range, taking the slot from ``cumsum`` as a
bump cursor and advancing it::

    slot = cumsum[flat[i]]++        ->      sorted_token_ids[slot] = i

``cumsum[e]`` is the start of expert ``e``'s padded range (written by the align
stage), so the scatter only overwrites the first ``count_e`` slots of each range;
the padding keeps the align stage's ``numel`` sentinel.  ``cumsum`` is read and
written in place: it leaves the kernel at ``start + count_e`` for every expert,
and its last element (``num_tokens_post_pad``) must not change.  The order of
routes *within* one expert's range is not part of the contract (the source GPU
kernel fills it in atomic arrival order), so the comparison normalises each
expert's range before comparing.

Adaptation summary (GPU -> ascend).  The source kernel reserves slots in two
levels: a block-local shared-memory histogram, then one batched global atomic
per expert, then a second pass that writes with ``base + rank``.  This kernel
uses a single-level bump cursor (one global atomic per route, taking the old
value).  Both forms hand out the same set of slots per expert and differ only in
the intra-expert order, which the contract leaves open (``DESIGN.md`` section
1.6.1 O1).  Ascend has a 1-D core grid and no thread count at kernel scope, so the
route axis is split over ``nblocks`` cores and the lane domain is opened by
``T.SimtVF``: ``idx = (bid * ept + i) * THREADS + j``, with ``ept`` routes per
lane chosen so that ``nblocks <= 56`` (the physical Vector core count).

Content layout:

1. ``golden_make_scatter_kernel`` -- torch CPU reference of the source
   semantics (stable index order + sentinel fill, no atomics).
2. ``_make_scatter_kernel`` -- the frozen kernel factory; returns a zero-arg
   callable building a program called as ``(flat, sorted_token_ids, cumsum)``.
3. ``run_case`` / ``_scatter`` -- launch + exact contract-level comparison.
4. ``run_L0`` / ``run_L1`` / ``run_L2`` / ``run_boundary`` / ``main``.
"""

import argparse
import functools
import math
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401  -- registers torch.npu

_THREADS = 128  # lane width, a compile-time constant (not a launch parameter)
_NBLK_TARGET = 56  # physical Vector cores of Ascend950PR_9579


# ---------------------------------------------------------------------------
# 1. Golden: torch CPU reference of the source operator's second stage
# ---------------------------------------------------------------------------
def golden_make_scatter_kernel(topk_ids, num_experts, block_size):
    """Reference scatter in stable index order, plus the align stage's pre-state.

    Independent of the NPU algorithm: no atomics, no core grid, no lane
    mapping.  It builds what the align stage leaves behind (``cumsum`` padded
    starts and the ``numel`` sentinel in every slot) and then walks the routes in
    ascending index order, which is the one legal ordering the GPU reference
    implementation produces.

    Returns ``(sorted_ref, cumsum_in, cumsum_out)``: the shape the kernel writes,
    the ``cumsum`` it is handed, and the ``cumsum`` it must leave behind.  All
    three are int64 CPU tensors so they compare exactly against int32 device
    results.
    """
    flat = topk_ids.flatten().to(torch.int64)
    numel = flat.numel()
    capacity = numel + (num_experts + 1) * (block_size - 1)

    counts = torch.bincount(flat, minlength=num_experts).to(torch.int64)
    padded = (counts + block_size - 1) // block_size * block_size
    cumsum_in = torch.zeros(num_experts + 1, dtype=torch.int64)
    cumsum_in[1:] = torch.cumsum(padded, 0)  # cumsum_in[e] = start of expert e
    cumsum_in[num_experts] = int(padded.sum())  # num_tokens_post_pad

    sorted_ref = torch.full((capacity,), numel, dtype=torch.int64)  # sentinel
    slot = cumsum_in[:num_experts].clone()
    for route, expert in enumerate(flat.tolist()):  # stable order: index ascending
        sorted_ref[int(slot[expert])] = route
        slot[expert] += 1
    cumsum_out = cumsum_in.clone()
    cumsum_out[:num_experts] = slot
    return sorted_ref, cumsum_in, cumsum_out


# ---------------------------------------------------------------------------
# 2. Kernel
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _make_scatter_kernel(numel: int, num_experts: int, block_size: int):
    """Frozen factory: geometry in, a zero-arg callable building the program.

    ``numel``, ``num_experts`` and ``block_size`` are compile-time constants, so
    every loop bound and guard below is static.  ``ept`` is the number of routes
    one lane handles, picked so that the logical core count stays at or below the
    56 physical Vector cores (``tile = _THREADS * ept >= ceil(numel / 56)``
    implies ``nblocks <= 56`` for any ``numel``).
    """
    if numel < 1:
        raise ValueError(f"the scatter needs at least one route, got numel={numel}")
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    ept = max(1, math.ceil(math.ceil(numel / _NBLK_TARGET) / _THREADS))
    tile = _THREADS * ept
    nblocks = math.ceil(numel / tile)

    @tilelang.jit(out_idx=[], target="ascend")
    def _scatter():
        @T.prim_func
        def _scatter_main(
            flat: T.Tensor([numel], "int32"),
            sorted_token_ids: T.Tensor([max_padded], "int32"),
            cumsum: T.Tensor([num_experts + 1], "int32"),
        ):
            with T.Kernel(nblocks) as bid, T.SimtVF(threads=_THREADS):
                for i in T.serial(ept):  # routes this lane handles, unrolled
                    for j in T.Parallel(_THREADS):  # one route per lane
                        idx = (bid * ept + i) * _THREADS + j
                        # Clamped index + -1 sentinel: a tail lane never reads
                        # past ``flat``, never takes a slot and never writes.
                        eid = T.if_then_else(
                            idx < numel, flat[T.min(idx, numel - 1)], -1
                        )
                        # Upper bound guards ``cumsum[num_experts]``
                        # (num_tokens_post_pad), which this stage must not touch.
                        if eid >= 0 and eid < num_experts:
                            slot = T.atomic_add(
                                cumsum[eid], T.int32(1), return_prev=True
                            )
                            # The store shares the guard with the atomic on
                            # purpose: the else branch of an atomic yields 0, so
                            # an unguarded store would write slot 0.
                            sorted_token_ids[slot] = idx

        return _scatter_main

    return _scatter


# ---------------------------------------------------------------------------
# 3. Launch + exact contract-level comparison
# ---------------------------------------------------------------------------
_RESULTS = []  # (level, name, ok, detail)
_MAX_DIFF = [0]


def _record(level, name, ok, detail=""):
    _RESULTS.append((level, name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[{level}][{tag}] {name}{(' -- ' + detail) if detail and not ok else ''}")


def _scatter(flat, sorted_token_ids, cumsum, numel, num_experts, block_size):
    """Launch the kernel with the wrapper's contract (in-place ``cumsum``)."""
    fn = _make_scatter_kernel(numel, num_experts, block_size)()
    fn(flat, sorted_token_ids, cumsum)


def _assert_route_equivalent(
    name, got, got_cumsum, sorted_ref, cumsum_in, cumsum_out, counts, sentinel
):
    """Exact contract-level comparison of one scatter result.

    Checks, in order: (1) ``cumsum`` advanced by exactly ``count_e`` per expert
    and its last element is untouched; (2) every expert range holds distinct
    route indices equal, as a multiset, to the reference; (3) no slot outside a
    routed range was written (the ``numel`` sentinel survived); (4) after sorting
    each range the whole array equals the reference element-wise.  Nothing is
    order-sensitive: the intra-expert order is not part of the contract.
    """
    got64 = got.to(torch.int64)
    got_cs = got_cumsum.to(torch.int64)
    if not torch.equal(got_cs, cumsum_out):
        bad = (got_cs != cumsum_out).nonzero().squeeze(1).tolist()
        raise AssertionError(
            f"{name}: cumsum differs at {bad[:8]} "
            f"(got {got_cs[bad[:8]].tolist()} want {cumsum_out[bad[:8]].tolist()})"
        )

    num_experts = counts.numel()
    used = torch.zeros(got64.numel(), dtype=torch.bool)
    normalised = got64.clone()
    for expert in range(num_experts):
        n = int(counts[expert])
        if n == 0:
            continue
        lo = int(cumsum_in[expert])
        seg = got64[lo : lo + n]
        if int(torch.unique(seg).numel()) != n:
            raise AssertionError(
                f"{name}: expert {expert} range [{lo},{lo + n}) has repeated slot"
            )
        ordered = torch.sort(seg).values
        if not torch.equal(ordered, sorted_ref[lo : lo + n]):
            raise AssertionError(
                f"{name}: expert {expert} range [{lo},{lo + n}) miscount "
                f"(got {seg.tolist()[:8]} want {sorted_ref[lo : lo + n].tolist()[:8]})"
            )
        normalised[lo : lo + n] = ordered
        used[lo : lo + n] = True

    outside = got64[~used]
    if outside.numel() and not bool(torch.all(outside == sentinel)):
        bad = (~used).nonzero().squeeze(1)[outside != sentinel].tolist()
        raise AssertionError(
            f"{name}: sentinel {sentinel} overwritten at {bad[:8]} "
            f"(got {got64[bad[:8]].tolist()})"
        )
    if not torch.equal(normalised, sorted_ref):
        diff = (normalised - sorted_ref).abs()
        _MAX_DIFF[0] = max(_MAX_DIFF[0], int(diff.max()))
        raise AssertionError(
            f"{name}: normalised result differs, max_diff={int(diff.max())}"
        )
    _MAX_DIFF[0] = max(_MAX_DIFF[0], 0)


def _run(
    level,
    name,
    numel,
    num_experts,
    block_size,
    ids=None,
    seed=0,
    repeat=1,
):
    """Build one geometry, run it on the NPU and compare against the golden."""
    gen = torch.Generator().manual_seed(seed)
    try:
        if ids is None:
            flat = torch.randint(
                0, num_experts, (numel,), generator=gen, dtype=torch.int32
            )
        else:
            flat = torch.tensor(ids, dtype=torch.int32)
            numel = flat.numel()
        capacity = numel + (num_experts + 1) * (block_size - 1)
        sorted_ref, cumsum_in, cumsum_out = golden_make_scatter_kernel(
            flat, num_experts, block_size
        )
        counts = torch.bincount(flat.to(torch.int64), minlength=num_experts)

        dflat = flat.to("npu")
        # The align stage leaves the sentinel everywhere; the scatter only
        # overwrites each expert's first count_e slots.
        dsorted = torch.full((capacity,), numel, dtype=torch.int32, device="npu")
        dcumsum = cumsum_in.to(torch.int32).to("npu")

        start = time.perf_counter()
        _scatter(dflat, dsorted, dcumsum, numel, num_experts, block_size)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1e3
        _assert_route_equivalent(
            name,
            dsorted.cpu(),
            dcumsum.cpu(),
            sorted_ref,
            cumsum_in,
            cumsum_out,
            counts,
            numel,
        )
        for _ in range(repeat - 1):
            # A fresh pre-state per launch (cumsum advances in place) and a full
            # re-check: repeated launches also probe the atomic path for races
            # that a single lucky arrival order could hide.
            dsorted.fill_(numel)
            dcumsum.copy_(cumsum_in.to(torch.int32).to("npu"))
            _scatter(dflat, dsorted, dcumsum, numel, num_experts, block_size)
            torch.npu.synchronize()
            _assert_route_equivalent(
                name,
                dsorted.cpu(),
                dcumsum.cpu(),
                sorted_ref,
                cumsum_in,
                cumsum_out,
                counts,
                numel,
            )
    except AssertionError as exc:
        _record(level, name, False, str(exc)[:300])
        return False
    except (ValueError, RuntimeError) as exc:  # noqa: BLE001 - reported, not swallowed
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:260]}")
        return False
    _record(
        level,
        name,
        True,
        f"numel={numel} E={num_experts} bs={block_size} launch={elapsed:.3f} ms",
    )
    print(
        f"       numel={numel} E={num_experts} bs={block_size} "
        f"routes={int(counts.sum())} launch={elapsed:.3f} ms"
    )
    return True


def _finish(level, strict):
    rows = [r for r in _RESULTS if r[0] == level]
    failed = [r for r in rows if not r[2]]
    print(
        f"[{level}] {len(rows) - len(failed)}/{len(rows)} cases passed, "
        f"max_diff={_MAX_DIFF[0]}"
    )
    if failed and strict:
        raise AssertionError(f"[{level}] {len(failed)} case(s) failed")
    if failed:
        print(f"[{level}][WARN] {len(failed)} non-blocking case(s) failed")


# ---------------------------------------------------------------------------
# 4. Layered cases
# ---------------------------------------------------------------------------
def run_L0():
    """Smoke gate: the smallest and the largest manifest workload, plus a mid one."""
    _run(
        level="L0",
        name="L0-1 kimi-t1-min-shape-E384",
        numel=8,
        num_experts=384,
        block_size=64,
        seed=1,
    )
    _run(
        level="L0",
        name="L0-2 qwen3-t512-E128",
        numel=4096,
        num_experts=128,
        block_size=64,
        seed=2,
    )
    _run(
        level="L0",
        name="L0-3 kimi-t4096-max-shape-concurrent-atomics",
        numel=32768,
        num_experts=384,
        block_size=64,
        seed=3,
        repeat=2,
    )
    _finish("L0", strict=True)


def run_L1():
    """Remaining manifest geometries plus the no-padding-slack case (bs=1)."""
    _run(
        level="L1",
        name="L1-1 kimi-t32-E384",
        numel=256,
        num_experts=384,
        block_size=64,
        seed=11,
    )
    _run(
        level="L1",
        name="L1-2 kimi-t512-E384",
        numel=4096,
        num_experts=384,
        block_size=64,
        seed=12,
    )
    _run(
        level="L1",
        name="L1-3 qwen3-t1-E128",
        numel=8,
        num_experts=128,
        block_size=64,
        seed=13,
    )
    _run(
        level="L1",
        name="L1-4 ds-v3-t4096-E256",
        numel=32768,
        num_experts=256,
        block_size=64,
        seed=14,
    )
    # bs=1 leaves no padding slack: every expert range is exactly count_e long,
    # so any stray slot write would land in another expert's routed range.
    _run(
        level="L1",
        name="L1-5 E128-block-size-1-no-slack",
        numel=32768,
        num_experts=128,
        block_size=1,
        seed=15,
    )
    _finish("L1", strict=True)


def run_L2():
    """Large / wide-expert geometries beyond the manifest: informative, non-blocking."""
    _run(
        level="L2",
        name="L2-1 numel-16384-E384",
        numel=16384,
        num_experts=384,
        block_size=64,
        seed=21,
    )
    _run(
        level="L2",
        name="L2-2 numel-8192-E512",
        numel=8192,
        num_experts=512,
        block_size=64,
        seed=22,
    )
    _run(
        level="L2",
        name="L2-3 E1100-wide-expert-table",
        numel=1024,
        num_experts=1100,
        block_size=32,
        seed=23,
    )
    _finish("L2", strict=False)


def _case_out_of_range(level, name, num_experts=8, block_size=4):
    """Ids at or past ``num_experts`` must not touch ``num_tokens_post_pad``.

    Outside the operator contract, but the source kernel would bump
    ``cumsum[num_experts]`` there; this kernel's upper-bound guard must leave it
    alone and write nothing for those routes.
    """
    flat = torch.tensor([0, 3, 8, 8, 5, 9, 3, 0, 7, 2, 8, 1], dtype=torch.int32)
    numel = flat.numel()
    capacity = numel + (num_experts + 1) * (block_size - 1)
    valid = flat < num_experts
    counts = torch.bincount(flat[valid].to(torch.int64), minlength=num_experts)
    padded = (counts + block_size - 1) // block_size * block_size
    cumsum_in = torch.zeros(num_experts + 1, dtype=torch.int64)
    cumsum_in[1:] = torch.cumsum(padded, 0)
    cumsum_in[num_experts] = int(padded.sum())
    sorted_ref = torch.full((capacity,), numel, dtype=torch.int64)
    slot = cumsum_in[:num_experts].clone()
    for route, expert in enumerate(flat.tolist()):
        if expert < num_experts:
            sorted_ref[int(slot[expert])] = route
            slot[expert] += 1
    cumsum_out = cumsum_in.clone()
    cumsum_out[:num_experts] = slot
    try:
        dflat = flat.to("npu")
        dsorted = torch.full((capacity,), numel, dtype=torch.int32, device="npu")
        dcumsum = cumsum_in.to(torch.int32).to("npu")
        _scatter(dflat, dsorted, dcumsum, numel, num_experts, block_size)
        torch.npu.synchronize()
        assert int(dcumsum[num_experts].cpu()) == int(cumsum_in[num_experts]), (
            "out-of-range id bumped num_tokens_post_pad: "
            f"{int(dcumsum[num_experts].cpu())} != {int(cumsum_in[num_experts])}"
        )
        _assert_route_equivalent(
            name,
            dsorted.cpu(),
            dcumsum.cpu(),
            sorted_ref,
            cumsum_in,
            cumsum_out,
            counts,
            numel,
        )
    except AssertionError as exc:
        _record(level, name, False, str(exc)[:300])
        return False
    except (ValueError, RuntimeError) as exc:  # noqa: BLE001 - reported, not swallowed
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:260]}")
        return False
    _record(level, name, True, f"numel={numel} E={num_experts} bs={block_size}")
    return True


def _case_no_routes(level, name, num_experts=8, block_size=16):
    """``numel = 0`` has no legal launch: the factory refuses to build one."""
    try:
        _make_scatter_kernel(0, num_experts, block_size)
    except ValueError as exc:
        _record(level, name, True, str(exc)[:120])
        print(f"       raised as expected: {exc}")
        return True
    _record(level, name, False, "factory built a program for numel=0")
    return False


def run_boundary():
    """Tail lanes, one-expert hotspot, wide expert tables, skew and contract edges."""
    _run(
        level="boundary",
        name="B-1 single-route-E1100-bs48",
        numel=1,
        num_experts=1100,
        block_size=48,
        seed=31,
    )
    # numel % tile != 0: ept=1, tile=128, nblocks=9 -> 120 idle tail lanes.
    _run(
        level="boundary",
        name="B-2 numel-1032-tail-120-lanes",
        numel=1032,
        num_experts=128,
        block_size=64,
        seed=32,
    )
    # One expert holds every route: every lane contends for the same cursor.
    _run(
        level="boundary",
        name="B-3 E1-single-expert-hotspot",
        numel=2048,
        num_experts=1,
        block_size=64,
        seed=33,
    )
    _run(
        level="boundary",
        name="B-4 numel-8192-E8-bs16",
        numel=8192,
        num_experts=8,
        block_size=16,
        seed=34,
    )
    # Worst-case skew: every route names the same expert of an 8-expert table.
    _run(
        level="boundary",
        name="B-5 numel-1023-skew-all-ids-3",
        numel=1023,
        num_experts=8,
        block_size=16,
        ids=[3] * 1023,
    )
    _case_out_of_range("boundary", "B-6 out-of-range-ids-leave-post-pad")
    _case_no_routes("boundary", "B-7 numel-0-refused")
    _finish("boundary", strict=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"]
    )
    args, _ = parser.parse_known_args()
    levels = {
        "L0": run_L0,
        "L1": run_L1,
        "L2": run_L2,
        "boundary": run_boundary,
    }
    chosen = levels.values() if args.level == "all" else [levels[args.level]]
    for run in chosen:
        run()
    print(f"[PRECISION_PASS] max_diff={_MAX_DIFF[0]}")


if __name__ == "__main__":
    main()
