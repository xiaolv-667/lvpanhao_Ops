# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""MoE token-to-expert alignment (align stage) on Ascend 950PR.

Port of the GPU TileLang ``_make_align_kernel`` (``MoEPermuteAlignFwdOp``, stage
1 of 2) to ``tilelang.ascend.language`` / ``target="ascend"``.

For ``numel`` flattened routes (``topk_ids.flatten()``), ``E`` experts and
``block_size`` rows per GEMM block the align stage writes:

* ``sorted_token_ids``    -- every slot set to the sentinel ``numel``; the route
  indices themselves are written later by the scatter stage;
* ``cumsum``             -- exclusive prefix sum of the per-expert route counts
  rounded up to ``block_size`` (handed to the scatter stage as a cursor);
* ``expert_ids``         -- owner expert of every block of rows, ``E`` past the
  padded total;
* ``num_tokens_post_pad``-- the padded total ``cumsum[E]``.

Path selection is a build-time decision on ``(numel, E, block_size)``:

* small: one launch, ``grid=2`` -- core 0 runs histogram, padding, scan,
  ``expert_ids`` plus half of the sentinel fill, core 1 fills the other half;
* large: two launches -- launch 1 writes one partial histogram per core into a
  GM scratch, launch 2 has every core redundantly re-read that scratch and
  re-derive the same integer-exact prefix sum, so no cross-core barrier and no
  cross-core data dependency is needed.
"""

import argparse
import functools
import math
import time

import torch
import torch_npu  # noqa: F401 -- registers torch.npu
import tilelang
import tilelang.ascend.language as T

FILL_TILE = 2048
VECTOR_CORES = 56
PATH_S_INSTR_BUDGET = 8192


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference, from the operator semantics only)
# ---------------------------------------------------------------------------
def golden_make_align_kernel(topk_ids, num_experts, block_size):
    """The align stage's four outputs, straight from the operator semantics.

    Uses ``bincount`` + ``cumsum`` + ``searchsorted`` -- a different derivation
    from the kernel's "compare-and-accumulate histogram + doubling scan +
    rank-count" and therefore an independent check.
    """
    B = block_size
    E = num_experts
    flat = topk_ids.flatten()
    numel = int(flat.numel())

    counts = torch.bincount(flat.long(), minlength=E)[:E]
    padded = (counts + B - 1) // B * B
    cumsum = torch.zeros(E + 1, dtype=torch.int64)
    cumsum[1:] = torch.cumsum(padded, 0)
    total = int(cumsum[E].item())

    p_total = numel + (E + 1) * (B - 1)
    num_blocks = math.ceil(p_total / B)
    sorted_token_ids = torch.full((p_total,), numel, dtype=torch.int32)
    expert_ids = torch.full((num_blocks,), E, dtype=torch.int32)

    owned = total // B
    if owned:
        pos = torch.arange(owned, dtype=torch.int64) * B
        # the e with cumsum[e] <= pos < cumsum[e+1]
        expert_ids[:owned] = torch.searchsorted(cumsum[1:], pos, right=True).to(
            torch.int32
        )
    npost = torch.tensor([total], dtype=torch.int32)
    return sorted_token_ids, expert_ids, npost, cumsum.to(torch.int32)


# ---------------------------------------------------------------------------
# Build-time planning
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


def _use_small_path(geo):
    lanes_per_vec = (geo["E"] + 63) // 64
    hist = geo["numel"] * lanes_per_vec * 2
    ids = geo["NB"] * lanes_per_vec
    fill = (geo["P"] + 63) // 64
    return hist + ids + fill <= PATH_S_INSTR_BUDGET


def _hist_cores(numel):
    return max(1, min(VECTOR_CORES, (numel + 511) // 512))


def _align_cores(p_total, num_blocks):
    logical = max(
        (num_blocks + 15) // 16, ((p_total + FILL_TILE - 1) // FILL_TILE + 7) // 8
    )
    logical = max(1, min(VECTOR_CORES, logical))
    for cand in (1, 2, 4, 7, 8, 14, 28, 56):
        if cand >= logical:
            return cand
    return VECTOR_CORES


# ---------------------------------------------------------------------------
# Kernel: small path (single launch, grid=2)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _compile_single(numel, E, B):
    geo = _geometry(numel, E, B)
    e_pad = geo["E_pad"]
    p_total = geo["P"]
    nb = geo["NB"]
    blk = max(64, _align_up(nb, 64))
    steps = geo["steps"]
    shift = geo["shift"]
    chunk = max(64, min(1024, _align_up(max(numel, 1), 64)))
    half = (p_total + 1) // 2
    n_fill = (half + FILL_TILE - 1) // FILL_TILE
    n_full = numel // chunk
    tail = numel - n_full * chunk

    @T.prim_func
    def main(
        flat: T.Tensor((numel,), "int32"),
        sorted_token_ids: T.Tensor((p_total,), "int32"),
        expert_ids: T.Tensor((nb,), "int32"),
        num_tokens_post_pad: T.Tensor((1,), "int32"),
        cumsum: T.Tensor((E + 1,), "int32"),
    ):
        with T.Kernel(2) as bx:
            s_x = T.alloc_shared((chunk,), "int32")
            s_cnt = T.alloc_shared((e_pad,), "int32")
            s_pad = T.alloc_shared((e_pad,), "int32")
            s_scan_a = T.alloc_shared((2 * e_pad,), "int32")
            s_scan_b = T.alloc_shared((2 * e_pad,), "int32")
            s_pref = T.alloc_shared((e_pad + 64,), "int32")
            s_ids = T.alloc_shared((blk,), "int32")
            s_fill = T.alloc_shared((FILL_TILE,), "int32")

            # ---- sentinel fill: unconditional, one half per core -------------
            T.fill(s_fill, numel)
            if n_fill > 0:
                for t in T.serial(n_fill):
                    off = bx * half + t * FILL_TILE
                    n = T.min(FILL_TILE, p_total - off)
                    if n > 0:
                        T.copy(s_fill[0:n], sorted_token_ids[off : off + n])

            # ---- metadata and mapping: no `numel > 0` guard on the writes,
            #      so a zero-route call still emits the contract values
            #      (cumsum all 0, npost 0, expert_ids all E) ---------------
            if bx == 0:
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_cnt[e] = 0

                if numel > 0:  # build-time branch: only the GM read is guarded
                    for c in range(n_full):
                        T.copy(flat[c * chunk : (c + 1) * chunk], s_x[0:chunk])
                        with T.SimtVF(threads=e_pad):
                            for e in T.Parallel(e_pad):
                                acc = T.alloc_local([1], "int32")
                                acc[0] = 0
                                for i in T.serial(chunk):
                                    acc[0] = acc[0] + T.if_then_else(s_x[i] == e, 1, 0)
                                s_cnt[e] = s_cnt[e] + acc[0]
                    if tail > 0:
                        T.copy(flat[n_full * chunk : numel], s_x[0:tail])
                        with T.SimtVF(threads=e_pad):
                            for e in T.Parallel(e_pad):
                                acc = T.alloc_local([1], "int32")
                                acc[0] = 0
                                for i in T.serial(tail):
                                    acc[0] = acc[0] + T.if_then_else(s_x[i] == e, 1, 0)
                                s_cnt[e] = s_cnt[e] + acc[0]

                # padding lanes (e >= E) contribute nothing
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        pv = T.if_then_else(e < E, s_cnt[e], 0)
                        if shift is not None:
                            s_pad[e] = ((pv + (B - 1)) >> shift) << shift
                        else:
                            s_pad[e] = T.ceildiv(pv, B) * B

                # inclusive doubling scan; two alternating buffers, data in the upper
                # half and a zero halo in the lower half, so every shifted operand stays
                # in bounds and no shared->shared copy is needed
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_scan_a[e_pad + e] = s_pad[e]
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_scan_a[e] = 0
                with T.SimtVF(threads=2 * e_pad):
                    for e in T.Parallel(2 * e_pad):
                        s_scan_b[e] = 0
                for d in range(steps):
                    k = 1 << d
                    if d % 2 == 0:  # build-time branch
                        with T.SimtVF(threads=e_pad):
                            for e in T.Parallel(e_pad):
                                s_scan_b[e_pad + e] = (
                                    s_scan_a[e_pad + e] + s_scan_a[e_pad + e - k]
                                )
                    else:
                        with T.SimtVF(threads=e_pad):
                            for e in T.Parallel(e_pad):
                                s_scan_a[e_pad + e] = (
                                    s_scan_b[e_pad + e] + s_scan_b[e_pad + e - k]
                                )

                # shift by one lane, filled over [0, e_pad] so the padding lanes hold
                # `total`, a defined value, never an uninitialised one
                if steps % 2 == 0:
                    with T.SimtVF(threads=e_pad + 64):
                        for e in T.Parallel(e_pad + 64):
                            if e <= e_pad:
                                s_pref[e] = T.if_then_else(
                                    e == 0, 0, s_scan_a[e_pad + e - 1]
                                )
                else:
                    with T.SimtVF(threads=e_pad + 64):
                        for e in T.Parallel(e_pad + 64):
                            if e <= e_pad:
                                s_pref[e] = T.if_then_else(
                                    e == 0, 0, s_scan_b[e_pad + e - 1]
                                )

                total = s_pref[E]
                T.copy(s_pref[0 : E + 1], cumsum[0 : E + 1])
                num_tokens_post_pad[0] = total

                # rank-count over [0, E) only; the padding lanes are never read
                with T.SimtVF(threads=blk):
                    for b in T.Parallel(blk):
                        pos = b * B
                        cnt = T.alloc_local([1], "int32")
                        cnt[0] = 0
                        for e in T.serial(E):
                            cnt[0] = cnt[0] + T.if_then_else(s_pref[e] <= pos, 1, 0)
                        s_ids[b] = T.if_then_else(pos < total, cnt[0] - 1, E)
                if nb > 0:
                    T.copy(s_ids[0:nb], expert_ids[0:nb])

    return tilelang.compile(main, target="ascend", out_idx=[])


# ---------------------------------------------------------------------------
# Kernel: large path (launch 1: per-core partial histograms)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _compile_hist(numel, E, B, g_h):
    e_pad = _align_up(max(E, 1), 64)
    tpc = (numel + g_h - 1) // g_h
    chunk = max(64, min(1024, _align_up(max(tpc, 1), 64)))
    n_chunks = (tpc + chunk - 1) // chunk

    @T.prim_func
    def main(
        flat: T.Tensor((numel,), "int32"),
        partial: T.Tensor((g_h, e_pad), "int32"),
    ):
        with T.Kernel(g_h) as g:
            s_x = T.alloc_shared((chunk,), "int32")
            s_hist = T.alloc_shared((e_pad,), "int32")

            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_hist[e] = 0

            if n_chunks > 0:  # build-time: numel == 0 leaves an all-zero row
                # serial (not pipelined): the accumulator is loop-carried
                for c in T.serial(n_chunks):
                    base = g * tpc + c * chunk
                    # clamp to this core's own slice and to the tensor end
                    n = T.max(0, T.min(T.min(tpc - c * chunk, chunk), numel - base))
                    if n > 0:
                        T.copy(flat[base : base + n], s_x[0:n])
                        with T.SimtVF(threads=e_pad):
                            for e in T.Parallel(e_pad):
                                acc = T.alloc_local([1], "int32")
                                acc[0] = 0
                                for i in T.serial(n):
                                    acc[0] = acc[0] + T.if_then_else(s_x[i] == e, 1, 0)
                                s_hist[e] = s_hist[e] + acc[0]

            # plain store, no atomics, no initialisation contract
            T.copy(s_hist, partial[g, :])

    return tilelang.compile(main, target="ascend", out_idx=[])


# ---------------------------------------------------------------------------
# Kernel: large path (launch 2: redundant merge + scan + mapping + fill)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _compile_align_partial(numel, E, B, g_h, g_a):
    geo = _geometry(numel, E, B)
    e_pad = geo["E_pad"]
    p_total = geo["P"]
    nb = geo["NB"]
    steps = geo["steps"]
    shift = geo["shift"]
    blkpc = (nb + g_a - 1) // g_a
    blk = max(64, _align_up(max(blkpc, 1), 64))
    share = (p_total + g_a - 1) // g_a
    n_fill = (share + FILL_TILE - 1) // FILL_TILE

    @T.prim_func
    def main(
        partial: T.Tensor((g_h, e_pad), "int32"),
        sorted_token_ids: T.Tensor((p_total,), "int32"),
        expert_ids: T.Tensor((nb,), "int32"),
        num_tokens_post_pad: T.Tensor((1,), "int32"),
        cumsum: T.Tensor((E + 1,), "int32"),
    ):
        with T.Kernel(g_a) as a:
            s_part = T.alloc_shared((e_pad,), "int32")
            s_cnt = T.alloc_shared((e_pad,), "int32")
            s_pad = T.alloc_shared((e_pad,), "int32")
            s_scan_a = T.alloc_shared((2 * e_pad,), "int32")
            s_scan_b = T.alloc_shared((2 * e_pad,), "int32")
            s_pref = T.alloc_shared((e_pad + 64,), "int32")
            s_ids = T.alloc_shared((blk,), "int32")
            s_fill = T.alloc_shared((FILL_TILE,), "int32")

            # every core re-reads the whole scratch: integer addition is
            # associative, so all cores derive bit-identical counts
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_cnt[e] = 0
            for g in T.serial(g_h):
                T.copy(partial[g, :], s_part)
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_cnt[e] = s_cnt[e] + s_part[e]

            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    pv = T.if_then_else(e < E, s_cnt[e], 0)
                    if shift is not None:
                        s_pad[e] = ((pv + (B - 1)) >> shift) << shift
                    else:
                        s_pad[e] = T.ceildiv(pv, B) * B

            # inclusive doubling scan; two alternating buffers, data in the upper
            # half and a zero halo in the lower half, so every shifted operand stays
            # in bounds and no shared->shared copy is needed
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_scan_a[e_pad + e] = s_pad[e]
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_scan_a[e] = 0
            with T.SimtVF(threads=2 * e_pad):
                for e in T.Parallel(2 * e_pad):
                    s_scan_b[e] = 0
            for d in range(steps):
                k = 1 << d
                if d % 2 == 0:  # build-time branch
                    with T.SimtVF(threads=e_pad):
                        for e in T.Parallel(e_pad):
                            s_scan_b[e_pad + e] = (
                                s_scan_a[e_pad + e] + s_scan_a[e_pad + e - k]
                            )
                else:
                    with T.SimtVF(threads=e_pad):
                        for e in T.Parallel(e_pad):
                            s_scan_a[e_pad + e] = (
                                s_scan_b[e_pad + e] + s_scan_b[e_pad + e - k]
                            )

            # shift by one lane, filled over [0, e_pad] so the padding lanes hold
            # `total`, a defined value, never an uninitialised one
            if steps % 2 == 0:
                with T.SimtVF(threads=e_pad + 64):
                    for e in T.Parallel(e_pad + 64):
                        if e <= e_pad:
                            s_pref[e] = T.if_then_else(
                                e == 0, 0, s_scan_a[e_pad + e - 1]
                            )
            else:
                with T.SimtVF(threads=e_pad + 64):
                    for e in T.Parallel(e_pad + 64):
                        if e <= e_pad:
                            s_pref[e] = T.if_then_else(
                                e == 0, 0, s_scan_b[e_pad + e - 1]
                            )

            total = s_pref[E]
            if a == 0:
                T.copy(s_pref[0 : E + 1], cumsum[0 : E + 1])
                num_tokens_post_pad[0] = total

            blk0 = a * blkpc
            with T.SimtVF(threads=blk):
                for j in T.Parallel(blk):
                    b = blk0 + j
                    cnt = T.alloc_local([1], "int32")
                    cnt[0] = 0
                    for e in T.serial(E):
                        cnt[0] = cnt[0] + T.if_then_else(s_pref[e] <= b * B, 1, 0)
                    s_ids[j] = T.if_then_else(b * B < total, cnt[0] - 1, E)
            valid = T.max(0, T.min(blkpc, nb - blk0))
            if valid > 0:
                T.copy(s_ids[0:valid], expert_ids[blk0 : blk0 + valid])

            T.fill(s_fill, numel)
            if n_fill > 0:
                for t in T.serial(n_fill):
                    off = a * share + t * FILL_TILE
                    n = T.min(FILL_TILE, T.min((a + 1) * share, p_total) - off)
                    if n > 0:
                        T.copy(s_fill[0:n], sorted_token_ids[off : off + n])

    return tilelang.compile(main, target="ascend", out_idx=[])


class _AlignTwoPass:
    """The large path's align callable: two launches plus a persistent scratch.

    ``partial`` is allocated on first use from ``flat.device`` and reused by
    later calls of this instance.  Launch 1 overwrites every row, so the scratch
    needs no initialisation; concurrent streams sharing one instance are unsafe.
    """

    def __init__(self, numel, E, B, g_h, g_a, e_pad):
        self._numel = numel
        self._hist = _compile_hist(numel, E, B, g_h)
        self._align = _compile_align_partial(numel, E, B, g_h, g_a)
        self._shape = (g_h, e_pad)
        self._partial = None

    def __call__(self, flat, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum):
        if self._partial is None or self._partial.device != flat.device:
            self._partial = torch.empty(
                self._shape, dtype=torch.int32, device=flat.device
            )
        self._hist(flat, self._partial)
        self._align(
            self._partial, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum
        )


# ---------------------------------------------------------------------------
# Kernel factory (harness wrapper contract: factory() -> align(...) callable)
# ---------------------------------------------------------------------------
def _make_align_kernel(numel: int, num_experts: int, block_size: int):
    """Build the align factory for one ``(numel, num_experts, block_size)``.

    Returns a no-argument callable whose result is invoked as
    ``align(flat, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum)``,
    matching ``MoEPermuteAlignKernel._compiled_align``.
    """
    numel = int(numel)
    E = int(num_experts)
    B = int(block_size)
    geo = _geometry(numel, E, B)
    small = _use_small_path(geo)
    g_h = _hist_cores(numel)
    g_a = _align_cores(geo["P"], geo["NB"])
    e_pad = geo["E_pad"]

    def _build():
        if small:
            return _compile_single(numel, E, B)
        return _AlignTwoPass(numel, E, B, g_h, g_a, e_pad)

    _build.path = "S" if small else "L"
    _build.g_h = g_h
    _build.g_a = g_a
    return _build


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
MAX_DIFF = {"value": 0}


def _make_flat(numel, E, dist, seed):
    """Routes for one case, honouring the input contract ``0 <= v < E``."""
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


def _run_case(numel, E, B, dist, tag, seed=0):
    """Compare all four outputs against the golden twice through one callable.

    The second repetition uses different routes, so it also exercises the
    persistent GM scratch of the large path (launch 1 must overwrite every row)
    and any dependence on left-over device memory.
    """
    t0 = time.perf_counter()
    factory = _make_align_kernel(numel, E, B)
    align = factory()
    p_total = numel + (E + 1) * (B - 1)
    num_blocks = math.ceil(p_total / B) if B else 0
    device = "npu"
    elapsed_us = 0.0
    last_total = 0
    for rep in range(2):
        flat = _make_flat(numel, E, dist, seed + rep)
        g_sorted, g_expert, g_npost, g_cumsum = golden_make_align_kernel(flat, E, B)

        sorted_d = torch.empty(p_total, dtype=torch.int32, device=device)
        expert_d = torch.empty(num_blocks, dtype=torch.int32, device=device)
        npost_d = torch.empty(1, dtype=torch.int32, device=device)
        cumsum_d = torch.empty(E + 1, dtype=torch.int32, device=device)

        call_t0 = time.perf_counter()
        align(flat.npu(), sorted_d, expert_d, npost_d, cumsum_d)
        torch.npu.synchronize()
        elapsed_us = max(elapsed_us, (time.perf_counter() - call_t0) * 1e6)
        last_total = int(g_npost.item())

        checks = (
            ("sorted_token_ids", sorted_d.cpu(), g_sorted),
            ("expert_ids", expert_d.cpu(), g_expert),
            ("num_tokens_post_pad", npost_d.cpu(), g_npost),
            ("cumsum", cumsum_d.cpu(), g_cumsum),
        )
        for name, got, expected in checks:
            diff = _max_abs_diff(got, expected)
            MAX_DIFF["value"] = max(MAX_DIFF["value"], diff)
            if not torch.equal(got, expected):
                bad = (
                    (got.flatten() != expected.flatten())
                    .nonzero()
                    .flatten()[:6]
                    .tolist()
                )
                raise AssertionError(
                    f"[{tag}] ({numel},{E},{B},{dist}) rep={rep} {name} mismatch, "
                    f"max|diff|={diff}, first idx={bad} "
                    f"got={got.flatten()[bad].tolist()} "
                    f"expected={expected.flatten()[bad].tolist()} "
                    f"(path={factory.path}, g_h={factory.g_h}, g_a={factory.g_a})"
                )
    print(
        f"[{tag}] PASS (numel={numel}, E={E}, B={B}, dist={dist}, path={factory.path}, "
        f"total={last_total}, {elapsed_us / 1000.0:.1f} ms/call, "
        f"x2 reps, {(time.perf_counter() - t0) / 1000.0:.1f} ms incl. build)"
    )


def run_L0(only=None):
    cases = (
        (8, 384, 64, "uniform"),
        (256, 384, 64, "skewed"),
        (4096, 384, 64, "uniform"),
        (32768, 384, 64, "uniform"),
        (1024, 8, 16, "alternating"),
        (600, 33, 64, "skewed"),
        (0, 8, 16, "uniform"),
        (4096, 128, 64, "uniform"),
    )
    for idx, (numel, E, B, dist) in enumerate(cases):
        if only is not None and idx not in only:
            continue
        _run_case(numel, E, B, dist, f"L0-{idx + 1}")


def run_L1(only=None):
    cases = []
    for numel in (8, 256, 4096, 32768):
        for E in (384, 256, 128):
            dist = "skewed" if (numel + E) % 2 else "uniform"
            cases.append((numel, E, 64, dist))
    for idx, (numel, E, B, dist) in enumerate(cases):
        if only is not None and idx not in only:
            continue
        _run_case(numel, E, B, dist, f"L1-{idx + 1}")


def run_L2():
    for numel, E, B, dist in (
        (65536, 384, 64, "uniform"),
        (8192, 256, 64, "zeros_expert"),
        (1, 128, 64, "uniform"),
    ):
        try:
            _run_case(numel, E, B, dist, "L2")
        except Exception as exc:  # noqa: BLE001 - L2 warns without blocking
            print(f"[L2] WARN (recorded, non-blocking) ({numel},{E},{B},{dist}): {exc}")


def run_boundary():
    for numel, E, B, dist in (
        (7, 3, 5, "uniform"),
        (64, 8, 1, "uniform"),
        (15, 1, 1, "single"),
        (128, 64, 64, "skewed"),
        (4096, 65, 64, "uniform"),
        (32768, 128, 64, "single"),
    ):
        try:
            _run_case(numel, E, B, dist, "Boundary")
        except Exception as exc:  # noqa: BLE001 - boundary warns without blocking
            print(
                f"[Boundary] WARN (recorded, non-blocking) ({numel},{E},{B},{dist}): {exc}"
            )


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
    print(f"[PRECISION_PASS] max_diff={MAX_DIFF['value']}")


if __name__ == "__main__":
    main()
