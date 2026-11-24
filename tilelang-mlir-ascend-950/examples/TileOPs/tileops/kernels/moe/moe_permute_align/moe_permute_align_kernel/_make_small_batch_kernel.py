# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""MoE token-to-expert alignment, fused small-batch path, on Ascend 950PR.

Port of the GPU TileLang ``_make_small_batch_kernel`` (``MoEPermuteAlignFwdOp``,
the fused third build path) to ``tilelang.ascend.language`` /
``target="ascend"``.

The path serves calls with ``numel < 1024`` routes and ``num_experts <= 32``.
In one launch -- and one core -- it writes the same three outputs as the
align + scatter pair:

* ``sorted_token_ids``     -- every one of the ``P`` slots set to the sentinel
  ``numel``; each route index then written into its expert's range
  ``[cs_e, cs_e + cnt_e)`` (distinct slots, intra-expert order undefined);
* ``expert_ids``           -- owner expert of every block of ``B`` slots,
  ``num_experts`` past the padded total ``total``;
* ``num_tokens_post_pad``  -- ``total = cs_E``, the padded route count.

with ``cnt_e = #{i : flat_i == e}``, ``p_e = ceil(cnt_e / B) * B``,
``cs_0 = 0, cs_{e+1} = cs_e + p_e``, ``P = numel + (E + 1)(B - 1)`` and
``NB = ceil(P / B)``.

Adaptation summary (GPU -> ascend).  The source kernel is a single CUDA block of
``256 + 32`` threads: 256 fill threads write the sentinel while 32 workers keep a
per-lane private histogram row, prefix-sum the rows, then take a rank from their
own row as the scatter cursor.  Ascend has a 1-D core grid with no thread count
at kernel scope, in-core thread domains are opened by ``T.SimtVF``, and any
data-dependent shared-memory (UB) index inside a loop makes the compiler insert
an in-core barrier per iteration (measured: 19 barriers for a 640-route tile,
plus a ``[ThreadSync] ... the race remains`` warning).  The port therefore:

* counts by ``lane <-> expert`` compare-and-accumulate over the route tile
  (no atomics, no private rows, no data-dependent UB index);
* pads and prefix-sums in UB with a 5-step Hillis-Steele scan over 32 lanes;
* derives ``expert_ids`` from the rank-count identity
  ``owner(b) = #{e < E : cs_e <= b*B} - 1`` instead of a data-dependent
  scatter loop;
* scatters ``lane <-> route`` with a per-expert UB cursor bump
  (``atomic_add(cs_e, 1, return_prev=True)``), reading the route id from GM so
  the hot loop keeps no data-dependent UB access;
* writes the sentinel as a bulk UB->GM store first, on the *same* core: the fill
  and the scatter write overlapping address ranges, and this toolchain has no
  certified cross-core barrier, so the two cannot be split over cores.

Content layout:

1. ``golden_make_small_batch_kernel`` -- torch CPU reference of the operator
   semantics (bincount + cumsum + binary search, routes in ascending index
   order); independent of every trick above.
2. ``_make_small_batch_kernel`` -- the frozen kernel factory; returns a
   zero-argument callable building a program called as
   ``(flat, sorted_token_ids, expert_ids, num_tokens_post_pad)``, matching
   ``MoEPermuteAlignKernel._small_batch_fn()``.
3. ``run_case`` -- launch plus exact contract-level comparison (intra-expert
   order normalised, sentinel coverage checked).
4. ``run_L0`` / ``run_L1`` / ``run_L2`` / ``run_boundary`` / ``main``.
"""

import argparse
import functools
import math
import time

import torch
import torch_npu  # noqa: F401 -- registers torch.npu
import tilelang
import tilelang.ascend.language as T

W = 32  # lane width of the expert and route axes (= the expert-count bound)
UNROLL = 4  # independent histogram accumulator chains per lane
FILL_TILE = 1024  # sentinel tile, in int32 elements per bulk store
IDS_LANES = 64  # lane width of the expert_ids pass
MAX_EXPERTS = 32  # the path's contract: at most one warp of experts
MAX_CHUNK = 1024  # route tile upper bound (one SIMT pass per chunk)
XFER_GUARD = 32  # spare slots past a partial tile (the GM->UB load rounds up)
SCAN_STEPS = 5  # ceil(log2(W)) -- Hillis-Steele steps over 32 lanes


# ---------------------------------------------------------------------------
# 1. Golden: torch CPU reference of the operator semantics
# ---------------------------------------------------------------------------
def golden_make_small_batch_kernel(topk_ids, num_experts, block_size):
    """The fused path's three outputs, straight from the operator semantics.

    ``bincount`` for the counts, ``cumsum`` for the padded prefix and
    ``searchsorted`` for the block owner -- a different derivation from the
    kernel's compare-and-accumulate histogram, Hillis-Steele scan, rank-count
    and cursor bump, so it is an independent check.

    Routes are placed in ascending index order within each expert.  The op
    contract leaves the intra-expert order undefined, so the caller normalises
    before comparing against a kernel that picks another order.
    """
    E, B = int(num_experts), int(block_size)
    flat = topk_ids.flatten().to(torch.int64)
    numel = int(flat.numel())

    counts = torch.bincount(flat, minlength=E)[:E] if numel else torch.zeros(E)
    padded = (counts + B - 1) // B * B
    cs = torch.zeros(E + 1, dtype=torch.int64)
    cs[1:] = torch.cumsum(padded, 0)
    total = int(cs[E].item())

    p_total = numel + (E + 1) * (B - 1)
    num_blocks = math.ceil(p_total / B) if p_total else 0

    sorted_ids = torch.full((p_total,), numel, dtype=torch.int32)
    for e in range(E):
        routed = (flat == e).nonzero().flatten()
        if routed.numel():
            lo = int(cs[e].item())
            sorted_ids[lo : lo + routed.numel()] = routed.to(torch.int32)

    expert_ids = torch.full((num_blocks,), E, dtype=torch.int32)
    owned = total // B
    if owned:
        pos = torch.arange(owned, dtype=torch.int64) * B
        # the e with cs[e] <= pos < cs[e+1]
        expert_ids[:owned] = torch.searchsorted(cs[1:], pos, right=True).to(
            torch.int32
        )

    npost = torch.tensor([total], dtype=torch.int32)
    return sorted_ids, expert_ids, npost


def _counts_and_starts(topk_ids, num_experts, block_size):
    """Per-expert route counts and padded range starts (test bookkeeping)."""
    E, B = int(num_experts), int(block_size)
    flat = topk_ids.flatten().to(torch.int64)
    numel = int(flat.numel())
    counts = torch.bincount(flat, minlength=E)[:E] if numel else torch.zeros(E)
    padded = (counts + B - 1) // B * B
    cs = torch.zeros(E + 1, dtype=torch.int64)
    cs[1:] = torch.cumsum(padded, 0)
    return counts.to(torch.int64), cs


# ---------------------------------------------------------------------------
# 2. Build-time planning
# ---------------------------------------------------------------------------
def _align_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _geometry(numel, num_experts, block_size):
    """Every build-time constant of one specialised program."""
    E, B = int(num_experts), int(block_size)
    p_total = numel + (E + 1) * (B - 1)
    nb = math.ceil(p_total / B) if p_total else 0
    chunk = min(MAX_CHUNK, max(64, _align_up(max(numel, 1), 64)))
    return {
        "numel": numel,
        "E": E,
        "B": B,
        "P": p_total,
        "NB": nb,
        "chunk": chunk,
        "n_chunks": (numel + chunk - 1) // chunk,
        "ids_blk": _align_up(max(nb, 1), IDS_LANES),
        "rounds": (numel + W - 1) // W,
        "n_fill": (p_total + FILL_TILE - 1) // FILL_TILE,
        "shift": B.bit_length() - 1 if (B & (B - 1)) == 0 else None,
    }


# ---------------------------------------------------------------------------
# 3. Kernel: one launch, one core, seven in-core stages
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _compile_small(numel, E, B):
    if W != 32 or SCAN_STEPS != 5:
        # the five doubling steps below are written out for a 32-lane domain
        raise ValueError(f"the scan is unrolled for 32 lanes, got {W}/{SCAN_STEPS}")
    geo = _geometry(numel, E, B)
    p_total = geo["P"]
    nb = geo["NB"]
    chunk = geo["chunk"]
    ids_blk = geo["ids_blk"]
    rounds = geo["rounds"]
    n_fill = geo["n_fill"]
    shift = geo["shift"]
    # Build-time split of the route tile into whole chunks plus the last one.
    # All four numbers are plain Python integers, so the histogram bounds and
    # the copy regions fold at build time.
    n_full = numel // chunk
    tail = numel - n_full * chunk
    tail_groups = tail // UNROLL
    tail_rem = tail - tail_groups * UNROLL

    @T.prim_func
    def _small_main(
        flat: T.Tensor((numel,), "int32"),
        sorted_token_ids: T.Tensor((p_total,), "int32"),
        expert_ids: T.Tensor((nb,), "int32"),
        num_tokens_post_pad: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            # The last GM->UB load carries a partial tile: the copy rounds its
            # length up to a 32-byte burst (repro/TRAP-950-partial-gm2ub-load),
            # so the slots past `tail` hold junk.  The guard band keeps that
            # spill inside s_x, and the histogram bound is the build-time
            # `tail`, so the junk is never read.
            s_x = T.alloc_shared((chunk + XFER_GUARD,), "int32")
            s_cnt = T.alloc_shared((W,), "int32")  # counts, then padded counts
            s_scan_a = T.alloc_shared((2 * W,), "int32")  # scan: [halo | data]
            s_scan_b = T.alloc_shared((2 * W,), "int32")  # alternate buffer
            s_cs = T.alloc_shared((W + 1,), "int32")  # prefix, then cursors
            s_ids = T.alloc_shared((ids_blk,), "int32")  # expert_ids staging
            s_fill = T.alloc_shared((FILL_TILE,), "int32")  # sentinel tile

            # ---- P0 sentinel fill: same core as the scatter, issued first ----
            T.fill(s_fill, numel)
            for t in T.serial(n_fill):  # tail block clipped to its live region
                off = t * FILL_TILE
                n = T.min(FILL_TILE, p_total - off)
                if n > 0:
                    T.copy(s_fill[0:n], sorted_token_ids[off : off + n])

            # ---- P2 histogram: lane <-> expert, UNROLL independent chains ----
            with T.SimtVF(threads=W):
                for e in T.Parallel(W):
                    s_cnt[e] = 0

            if numel > 0:  # build-time: a zero-route call folds the GM reads
                for c in range(n_full):
                    T.copy(flat[c * chunk : (c + 1) * chunk], s_x[0:chunk])
                    with T.SimtVF(threads=W):
                        for e in T.Parallel(W):
                            acc = T.alloc_local([UNROLL], "int32")
                            for j in range(UNROLL):
                                acc[j] = 0
                            for i in T.serial(chunk // UNROLL):
                                for j in range(UNROLL):
                                    acc[j] = acc[j] + T.if_then_else(
                                        s_x[i * UNROLL + j] == e, 1, 0
                                    )
                            s_cnt[e] = s_cnt[e] + acc[0] + acc[1] + acc[2] + acc[3]
                if tail > 0:
                    T.copy(flat[n_full * chunk : numel], s_x[0:tail])
                    # Static bound (tail is a build-time constant): the junk the
                    # load leaves in the guard band is never read, and the
                    # accumulator keeps its UNROLL chains over the whole tail.
                    with T.SimtVF(threads=W):
                        for e in T.Parallel(W):
                            acc = T.alloc_local([UNROLL], "int32")
                            for j in range(UNROLL):
                                acc[j] = 0
                            for i in T.serial(tail_groups):
                                for j in range(UNROLL):
                                    acc[j] = acc[j] + T.if_then_else(
                                        s_x[i * UNROLL + j] == e, 1, 0
                                    )
                            if tail_rem > 0:
                                for j in range(tail_rem):
                                    acc[j] = acc[j] + T.if_then_else(
                                        s_x[tail_groups * UNROLL + j] == e, 1, 0
                                    )
                            s_cnt[e] = s_cnt[e] + acc[0] + acc[1] + acc[2] + acc[3]

            # ---- P3 pad each count up to a whole block ----------------------
            with T.SimtVF(threads=W):
                for e in T.Parallel(W):
                    if shift is not None:
                        s_cnt[e] = ((s_cnt[e] + (B - 1)) >> shift) << shift
                    else:
                        s_cnt[e] = T.ceildiv(s_cnt[e], B) * B

            # ---- P4 exclusive prefix: Hillis-Steele over 32 lanes -----------
            # Data in the upper half of each scan buffer, a zero halo in the
            # lower half, so every shifted operand stays in bounds and no
            # shared -> shared copy is needed.  Lanes at e >= E keep a zero
            # count, a zero padded increment and therefore cs_e = total.
            with T.SimtVF(threads=W):
                for e in T.Parallel(W):
                    s_scan_a[W + e] = s_cnt[e]
            with T.SimtVF(threads=W):
                for e in T.Parallel(W):
                    s_scan_a[e] = 0
            with T.SimtVF(threads=2 * W):
                for e in T.Parallel(2 * W):
                    s_scan_b[e] = 0
            # The five doubling steps are written out: a traced `range` loop
            # cannot hold a build-time parity test.
            with T.SimtVF(threads=W):  # k = 1, a -> b
                for e in T.Parallel(W):
                    s_scan_b[W + e] = s_scan_a[W + e] + s_scan_a[W + e - 1]
            with T.SimtVF(threads=W):  # k = 2, b -> a
                for e in T.Parallel(W):
                    s_scan_a[W + e] = s_scan_b[W + e] + s_scan_b[W + e - 2]
            with T.SimtVF(threads=W):  # k = 4, a -> b
                for e in T.Parallel(W):
                    s_scan_b[W + e] = s_scan_a[W + e] + s_scan_a[W + e - 4]
            with T.SimtVF(threads=W):  # k = 8, b -> a
                for e in T.Parallel(W):
                    s_scan_a[W + e] = s_scan_b[W + e] + s_scan_b[W + e - 8]
            with T.SimtVF(threads=W):  # k = 16, a -> b (inclusive result in b)
                for e in T.Parallel(W):
                    s_scan_b[W + e] = s_scan_a[W + e] + s_scan_a[W + e - 16]

            # Shift by one lane, filled over [0, W] so every entry -- including
            # the lanes at e >= E -- is a defined value.
            with T.SimtVF(threads=W + W):
                for e in T.Parallel(W + W):
                    if e <= W:
                        s_cs[e] = T.if_then_else(e == 0, 0, s_scan_b[W + e - 1])

            total = s_cs[W]
            # contract value for a zero-route call too: the write is not guarded
            num_tokens_post_pad[0] = total

            # ---- P5 expert_ids: rank-count over [0, E) ----------------------
            for c in range(ids_blk // IDS_LANES):
                with T.SimtVF(threads=IDS_LANES):
                    for j in T.Parallel(IDS_LANES):
                        b = c * IDS_LANES + j
                        cnt = T.alloc_local([1], "int32")
                        cnt[0] = 0
                        for e in T.serial(E):
                            cnt[0] = cnt[0] + T.if_then_else(s_cs[e] <= b * B, 1, 0)
                        s_ids[b] = T.if_then_else(b * B < total, cnt[0] - 1, E)
            if nb > 0:
                T.copy(s_ids[0:nb], expert_ids[0:nb])

            # ---- P6 scatter: lane <-> route, per-expert UB cursor bump ------
            # The route id comes from GM, not from the UB tile: that keeps the
            # loop free of any data-dependent UB access, which would otherwise
            # make the compiler insert one in-core barrier per round.  The
            # cursors are cs_e themselves, so a bump walks [cs_e, cs_e + cnt_e)
            # and the slots stay distinct and inside the expert's range.
            # cs_W (= total) is already published and is never touched: a
            # contract-conforming id is < E <= W.
            if numel > 0:
                with T.SimtVF(threads=W):
                    for j in T.Parallel(W):
                        eid = T.alloc_local([1], "int32")
                        slot = T.alloc_local([1], "int32")
                        for r in T.serial(rounds):
                            idx = r * W + j
                            if idx < numel:
                                eid[0] = flat[idx]
                                slot[0] = T.atomic_add(
                                    s_cs[eid[0]], T.int32(1), return_prev=True
                                )
                                sorted_token_ids[slot[0]] = idx

    return tilelang.compile(_small_main, target="ascend", out_idx=[])


def _make_small_batch_kernel(numel: int, num_experts: int, block_size: int):
    """Build the fused small-batch factory for one ``(numel, E, block_size)``.

    Returns a no-argument callable whose result is invoked as
    ``small(flat, sorted_token_ids, expert_ids, num_tokens_post_pad)``, matching
    ``MoEPermuteAlignKernel._small_batch_fn()`` -- the path takes no launch
    parameter because ``T.Kernel`` has no thread count.

    Raises:
        ValueError: ``num_experts`` exceeds one warp of experts, matching the
            source factory's guard.
    """
    numel = int(numel)
    E = int(num_experts)
    B = int(block_size)
    if E > MAX_EXPERTS:
        raise ValueError(
            f"the small-batch kernel takes at most {MAX_EXPERTS} experts, got {E}"
        )
    geo = _geometry(numel, E, B)

    def _build():
        return _compile_small(numel, E, B)

    _build.geometry = geo
    return _build


# ---------------------------------------------------------------------------
# 4. Launch + exact contract-level comparison
# ---------------------------------------------------------------------------
_RESULTS = []  # (level, name, ok, detail)
_MAX_DIFF = [0]
_POISON = -12345  # neither a route index nor the sentinel, so a missed write shows


def _record(level, name, ok, detail=""):
    _RESULTS.append((level, name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[{level}][{tag}] {name}{(' -- ' + detail) if detail and not ok else ''}")


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


def _make_flat(numel, E, dist, seed, ids=None):
    """Routes for one case, honouring the input contract ``0 <= v < E``."""
    if ids is not None:
        return torch.tensor(ids, dtype=torch.int32)
    if numel == 0:
        return torch.zeros((0,), dtype=torch.int32)
    gen = torch.Generator().manual_seed(seed)
    if dist == "uniform":
        flat = torch.randint(0, E, (numel,), generator=gen, dtype=torch.int32)
    elif dist == "skewed":
        weights = torch.arange(1, E + 1, dtype=torch.float64) ** 2
        flat = torch.multinomial(weights, numel, replacement=True, generator=gen).to(
            torch.int32
        )
    elif dist == "zeros_expert":
        span = max(1, E // 4)
        flat = torch.randint(0, span, (numel,), generator=gen, dtype=torch.int32)
    elif dist == "alternating":
        flat = (torch.arange(numel) % max(1, min(E, 3))).to(torch.int32)
    elif dist == "single":
        flat = torch.full((numel,), E - 1, dtype=torch.int32)
    else:
        raise ValueError(f"unknown distribution {dist}")
    return torch.clamp(flat, 0, E - 1)


def _max_abs_diff(got, expected):
    if got.numel() == 0:
        return 0
    return int((got.to(torch.int64) - expected.to(torch.int64)).abs().max().item())


def _assert_route_equivalent(name, got, sorted_ref, counts, cs, sentinel):
    """Exact contract-level comparison of one fused-path result.

    Checks, in order: (1) every expert range holds distinct route indices whose
    sorted form equals the reference's, so the ranges hold exactly the routed
    slots; (2) every slot outside a routed range kept the ``numel`` sentinel --
    with a poisoned destination buffer this also proves the fill covers ``[0,P)``;
    (3) after normalising each range the whole array equals the reference.
    Nothing is order-sensitive: intra-expert order is not part of the contract.
    """
    got64 = got.to(torch.int64)
    used = torch.zeros(got64.numel(), dtype=torch.bool)
    normalised = got64.clone()
    for expert in range(int(counts.numel())):
        n = int(counts[expert])
        if n == 0:
            continue
        lo = int(cs[expert])
        seg = got64[lo : lo + n]
        if int(torch.unique(seg).numel()) != n:
            raise AssertionError(
                f"{name}: expert {expert} range [{lo},{lo + n}) has a repeated slot "
                f"(got {seg.tolist()[:8]})"
            )
        ordered = torch.sort(seg).values
        want = sorted_ref[lo : lo + n].to(torch.int64)
        if not torch.equal(ordered, want):
            raise AssertionError(
                f"{name}: expert {expert} range [{lo},{lo + n}) miscount "
                f"(got {seg.tolist()[:8]} want {want.tolist()[:8]})"
            )
        normalised[lo : lo + n] = ordered
        used[lo : lo + n] = True

    outside = got64[~used]
    if outside.numel() and not bool(torch.all(outside == sentinel)):
        bad = (~used).nonzero().flatten()[outside != sentinel].tolist()
        raise AssertionError(
            f"{name}: sentinel {sentinel} not found at {bad[:8]} "
            f"(got {got64[bad[:8]].tolist()})"
        )
    if not torch.equal(normalised, sorted_ref.to(torch.int64)):
        diff = int((normalised - sorted_ref.to(torch.int64)).abs().max())
        _MAX_DIFF[0] = max(_MAX_DIFF[0], diff)
        raise AssertionError(f"{name}: normalised result differs, max_diff={diff}")


def _run_case(
    level,
    name,
    numel,
    num_experts,
    block_size,
    dist="uniform",
    seed=0,
    repeat=1,
    ids=None,
):
    """Build one geometry, run it on the NPU and compare against the golden.

    ``repeat > 1`` relaunches the same routes and re-checks the contract every
    time: that is the acceptance run for the UB cursor bump (an arrival-order
    race would show up as a repeated slot or a missing route), and it also
    reports whether the outputs are bit-identical across launches.
    """
    E = int(num_experts)
    B = int(block_size)
    try:
        flat = _make_flat(numel, E, dist, seed, ids)
        numel = int(flat.numel())
        p_total = numel + (E + 1) * (B - 1)
        nb = math.ceil(p_total / B) if p_total else 0

        g_sorted, g_expert, g_npost = golden_make_small_batch_kernel(
            flat, E, block_size
        )
        counts, cs = _counts_and_starts(flat, E, block_size)

        factory = _make_small_batch_kernel(numel, E, block_size)
        small = factory()
        dflat = flat.npu()

        first = None
        bitwise_same = True
        elapsed = 0.0
        for rep in range(repeat):
            # Poisoned destination: every byte the kernel is responsible for
            # must be overwritten, no reliance on a pre-zeroed allocation.
            dsorted = torch.full((p_total,), _POISON, dtype=torch.int32, device="npu")
            dexpert = torch.full((nb,), _POISON, dtype=torch.int32, device="npu")
            dnpost = torch.full((1,), _POISON, dtype=torch.int32, device="npu")

            start = time.perf_counter()
            small(dflat, dsorted, dexpert, dnpost)
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) * 1e3

            sorted_h = dsorted.cpu()
            expert_h = dexpert.cpu()
            npost_h = dnpost.cpu()

            _assert_route_equivalent(
                f"{name} rep{rep}", sorted_h, g_sorted, counts, cs, numel
            )
            for out_name, got, expected in (
                ("expert_ids", expert_h, g_expert),
                ("num_tokens_post_pad", npost_h, g_npost),
            ):
                diff = _max_abs_diff(got, expected)
                _MAX_DIFF[0] = max(_MAX_DIFF[0], diff)
                if not torch.equal(got, expected):
                    bad = (got.flatten() != expected.flatten()).nonzero().flatten()
                    bad = bad[:6].tolist()
                    raise AssertionError(
                        f"{name} rep{rep} {out_name} mismatch, max|diff|={diff}, "
                        f"first idx={bad} got={got.flatten()[bad].tolist()} "
                        f"expected={expected.flatten()[bad].tolist()}"
                    )

            if first is None:
                first = (sorted_h.clone(), expert_h.clone(), npost_h.clone())
            else:
                bitwise_same = bitwise_same and (
                    torch.equal(first[0], sorted_h)
                    and torch.equal(first[1], expert_h)
                    and torch.equal(first[2], npost_h)
                )
    except AssertionError as exc:
        _record(level, name, False, str(exc)[:400])
        return False
    except (ValueError, RuntimeError) as exc:  # noqa: BLE001 - reported, not swallowed
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:300]}")
        return False
    _record(
        level,
        name,
        True,
        f"numel={numel} E={E} B={block_size} dist={dist} total={int(g_npost[0])} "
        f"launch={elapsed:.3f} ms x{repeat} bitwise_identical={bitwise_same}",
    )
    print(
        f"       numel={numel} E={E} B={block_size} dist={dist} "
        f"total={int(g_npost[0])} routes={numel} chunks={factory.geometry['n_chunks']} "
        f"rounds={factory.geometry['rounds']} launch={elapsed:.3f} ms "
        f"bitwise_identical={bitwise_same}"
    )
    return True


def _case_experts_out_of_range(level, name):
    """A factory call with more than one warp of experts must refuse to build."""
    try:
        _make_small_batch_kernel(8, MAX_EXPERTS + 1, 16)
    except ValueError as exc:
        _record(level, name, True, str(exc)[:120])
        print(f"       raised as expected: {exc}")
        return True
    _record(level, name, False, "factory built a program for num_experts=33")
    return False


def _case_ids_out_of_contract(level, name):
    """Ids at or past ``num_experts`` are outside the op contract.

    Neither the source kernel nor this one checks them (a conforming wrapper
    never produces them), so only "no crash, no host-side error" is asserted;
    the output values are unspecified.
    """
    try:
        flat = torch.tensor([0, 4, 5, 3, 7, 2, 6, 1], dtype=torch.int32)
        E, B = 4, 4
        p_total = int(flat.numel()) + (E + 1) * (B - 1)
        nb = math.ceil(p_total / B)
        dflat = flat.npu()
        dsorted = torch.full((p_total,), _POISON, dtype=torch.int32, device="npu")
        dexpert = torch.full((nb,), _POISON, dtype=torch.int32, device="npu")
        dnpost = torch.full((1,), _POISON, dtype=torch.int32, device="npu")
        _make_small_batch_kernel(int(flat.numel()), E, B)()(
            dflat, dsorted, dexpert, dnpost
        )
        torch.npu.synchronize()
    except (ValueError, RuntimeError) as exc:  # noqa: BLE001
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:260]}")
        return False
    _record(level, name, True, "undefined output, no error (as in the source)")
    return True


# ---------------------------------------------------------------------------
# 5. Layered cases
# ---------------------------------------------------------------------------
def run_L0(only=None):
    """Gate: every code path and every boundary of this build path.

    The manifest and bench workloads all have ``num_experts > 32``, so no
    workload-driven test ever reaches this path; these cases are the whole
    correctness evidence for it.
    """
    cases = (
        ("L0-1 tiny-bs4", (8, 4, 4, "uniform", 0)),
        ("L0-2 single-token", (2, 4, 4, "uniform", 1)),
        ("L0-3 numel-1-E32", (1, 32, 64, "uniform", 2)),
        ("L0-4 numel-eq-lanes", (32, 8, 16, "uniform", 3)),
        ("L0-5 sb-numel200", (200, 8, 16, "uniform", 4)),
        ("L0-6 numel600-maxexp", (600, 32, 64, "uniform", 5)),
        ("L0-7 numel1000-maxexp-skew", (1000, 32, 64, "skewed", 6)),
        ("L0-8 routes-1023-bs16", (1023, 8, 16, "zeros_expert", 7)),
        ("L0-9 max-workload", (1023, 32, 64, "uniform", 8)),
        ("L0-10 no-routes-few-experts", (0, 8, 16, "uniform", 9)),
        ("L0-11 bs-not-pow2-48", (100, 8, 48, "uniform", 10)),
        ("L0-12 bs1", (64, 8, 1, "uniform", 11)),
        ("L0-13 E1-B1-one-route", (1, 1, 1, "single", 12)),
        ("L0-14 E1-all-on-expert0", (8, 1, 16, "single", 13)),
        ("L0-15 numel1024-beyond-threshold", (1024, 8, 16, "uniform", 14)),
        ("L0-16 numel7-E3-B5", (7, 3, 5, "uniform", 15)),
        ("L0-17 numel8-E8-bs16", (8, 8, 16, "uniform", 16)),
    )
    for idx, (name, (numel, E, B, dist, seed)) in enumerate(cases):
        if only is not None and idx not in only:
            continue
        repeat = 3 if name == "L0-9 max-workload" else 1
        _run_case("L0", name, numel, E, B, dist, seed=seed, repeat=repeat)
    if only is None or len(cases) in only:
        _case_experts_out_of_range("L0", "L0-18 E33-must-raise")
    _finish("L0", strict=True)


def run_L1(only=None):
    """Representative geometries, distributions and shapes past the threshold."""
    cases = (
        ("L1-1 numel256-E16-bs64-skew", (256, 16, 64, "skewed", 21)),
        ("L1-2 numel512-E32-bs32", (512, 32, 32, "uniform", 22)),
        ("L1-3 numel777-E8-bs64-alt", (777, 8, 64, "alternating", 23)),
        ("L1-4 numel1000-E8-bs64-zeroexp", (1000, 8, 64, "zeros_expert", 24)),
        ("L1-5 numel1023-E8-bs64-skew", (1023, 8, 64, "skewed", 25)),
        ("L1-6 numel2048-multichunk", (2048, 8, 64, "uniform", 26)),
        ("L1-7 numel1024-E1-bs16", (1024, 1, 16, "single", 27)),
        ("L1-8 numel300-E32-bs1", (300, 32, 1, "uniform", 28)),
        ("L1-9 numel640-E8-bs128", (640, 8, 128, "uniform", 29)),
        ("L1-10 numel63-E32-bs16-tail", (63, 32, 16, "uniform", 30)),
    )
    for idx, (name, (numel, E, B, dist, seed)) in enumerate(cases):
        if only is not None and idx not in only:
            continue
        _run_case("L1", name, numel, E, B, dist, seed=seed)
    _finish("L1", strict=True)


def run_L2():
    """Larger / wider geometries beyond the path's design point: informative."""
    for name, (numel, E, B, dist, seed) in (
        ("L2-1 numel4096-E32-bs64", (4096, 32, 64, "uniform", 41)),
        ("L2-2 numel3000-E8-bs1", (3000, 8, 1, "uniform", 42)),
        ("L2-3 numel512-E32-bs256", (512, 32, 256, "skewed", 43)),
        ("L2-4 numel1-E1-bs1024", (1, 1, 1024, "single", 44)),
    ):
        _run_case("L2", name, numel, E, B, dist, seed=seed)
    _finish("L2", strict=False)


def run_boundary():
    """Degenerate shapes, extreme skew, tail rounds and out-of-contract input."""
    for name, (numel, E, B, dist, seed) in (
        ("B-1 numel0-E1-B1-zero-capacity", (0, 1, 1, "uniform", 51)),
        ("B-2 numel0-E32-bs64", (0, 32, 64, "uniform", 52)),
        ("B-3 numel1023-E1-bs1023", (1023, 1, 1023, "uniform", 53)),
        ("B-4 numel1-E32-B1", (1, 32, 1, "uniform", 54)),
        ("B-5 numel2047-E8-bs16-multichunk", (2047, 8, 16, "uniform", 55)),
        ("B-6 numel1023-E32-B1", (1023, 32, 1, "uniform", 56)),
    ):
        _run_case("boundary", name, numel, E, B, dist, seed=seed)
    _run_case(
        "boundary",
        "B-7 all-routes-on-one-expert",
        1023,
        8,
        16,
        "uniform",
        ids=[7] * 1023,
    )
    _case_ids_out_of_contract("boundary", "B-8 ids-out-of-contract-no-crash")
    _finish("boundary", strict=False)


def _parse_only(text):
    if text is None:
        return None
    return {int(v) for v in text.split(",") if v.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"]
    )
    parser.add_argument(
        "--case", default=None, help="comma separated case indices (debug)"
    )
    args, _ = parser.parse_known_args()
    only = _parse_only(args.case)
    started = time.perf_counter()
    levels = {
        "L0": lambda: run_L0(only),
        "L1": lambda: run_L1(only),
        "L2": run_L2,
        "boundary": run_boundary,
    }
    chosen = levels.values() if args.level == "all" else [levels[args.level]]
    for run in chosen:
        run()
    print(f"elapsed: {time.perf_counter() - started:.1f} s")
    print("\033[92m[PRECISION_PASS] All check passed!\033[0m")
    print(f"[PRECISION_PASS] max_diff={_MAX_DIFF[0]}")


if __name__ == "__main__":
    main()
