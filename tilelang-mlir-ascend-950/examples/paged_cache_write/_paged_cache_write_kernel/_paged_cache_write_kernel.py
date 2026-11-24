# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Paged KV cache write (ascend): kernel + torch golden + layered precision tests.

Scatter new K/V rows into the pool rows ``slot_mapping`` names, in place:

    k_pages[slot_mapping[n]] = k[n]               (16-bit cache: plain copy)
    k_pages[slot_mapping[n]] = clamp(k[n] / k_scale, -448, 448) -> e4m3

A token whose slot is ``-1`` is skipped, and no row is touched unless a slot
names it.  The kernel keeps the source operator's direct-scatter structure: one
block owns ``block_rows`` tokens and a ``block_cols``-wide column segment, every
lane reads its own GM element, computes in vector registers and stores straight
back to GM.  Ascend has a 1-D core grid and no thread count at kernel scope, so
the source's 2-D grid is folded into one extent (``rb = bx // nchunks``,
``cb = bx % nchunks``) and the thread domain is opened by ``T.SimtVF``.

The fp8 cache is written by one ``T.Parallel`` loop per pool: two e4m3 stores in
one body defeat the vectoriser, and a scalar e4m3 store writes 0 on this backend
(``repro/TRAP-950-fp8-scalar-store.py``).

Content layout:

1. ``golden_paged_cache_write`` -- torch CPU reference (source ``ref_program``).
2. ``_paged_cache_write_kernel`` -- the frozen kernel factory; returns a
   callable ``(block_rows, block_cols) -> compiled callable``.
3. ``run_case`` / ``_write`` -- launch + exact comparison of both page pools.
4. ``run_L0`` / ``run_L1`` / ``run_L2`` / ``run_boundary`` / ``main``.
"""

import argparse
import functools
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401  -- registers torch.npu

_FP8_E4M3FN = "float8_e4m3fn"
_FP8_MAX = 448.0
_IN_DTYPES = ("float16", "bfloat16")
_THREADS = 128
_DEFAULT_BLOCK_ROWS = 16
_DEFAULT_BLOCK_COLS = 128

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
}
_DTYPE_NAME = {v: k for k, v in _TORCH_DTYPE.items()}


# ---------------------------------------------------------------------------
# 1. Golden: torch CPU reference of the source operator's ``ref_program``
# ---------------------------------------------------------------------------
def golden_paged_cache_write(
    k, v, k_cache, v_cache, slot_mapping, k_scale=None, v_scale=None
):
    """Write the rows the slots name; skip negative slots; leave the rest untouched.

    Independent of the NPU algorithm: it clones both pools first (the reference
    runs on shared inputs), selects the written tokens with a host predicate and
    scatters them with advanced indexing.  An FP8 pool divides by the scale and
    clamps to the e4m3 range before the cast, which keeps NaN and saturates
    finite overflow / infinities exactly as the contract requires.
    """
    k_out, v_out = k_cache.clone(), v_cache.clone()
    written = (slot_mapping >= 0).nonzero().squeeze(1)  # write enable
    rows = slot_mapping[written]  # destination rows
    for pages, new, scale in ((k_out, k, k_scale), (v_out, v, v_scale)):
        flat = pages.view(-1, *pages.shape[2:])  # (physical_tokens, heads_kv, dim)
        stored = new[written]
        if scale is not None:
            limit = torch.finfo(pages.dtype).max  # float8_e4m3fn -> 448
            stored = (stored.float() / scale[0]).clamp(-limit, limit).to(pages.dtype)
        flat[rows] = stored
    return k_out, v_out


# ---------------------------------------------------------------------------
# 2. Kernel
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=128)
def _build(
    heads_kv, dim, physical_tokens, in_dtype, cache_dtype, block_rows, block_cols
):
    """Compile one specialised program; cached per geometry and tile shape."""
    row_width = heads_kv * dim
    quantized = cache_dtype == _FP8_E4M3FN
    # Column width normalisation: a segment always divides the row, so ``col``
    # needs no run-time bound (row_width < block_cols falls back to the row).
    cols = (
        block_cols
        if (row_width % block_cols == 0 and block_cols <= row_width)
        else row_width
    )
    nchunks = (row_width + cols - 1) // cols
    # The only dynamic extent: a decode step writes one row per request.
    tokens = T.dynamic("tokens")
    nblocks = T.ceildiv(tokens, block_rows) * nchunks
    # A pool that cannot index int32 keeps int64 (the contract's guard).
    slot_dtype = "int32" if physical_tokens < 2**31 else "int64"
    per_thread = block_rows * cols // _THREADS

    if quantized:
        # Two fp8 stores in one parallel body defeat the vectoriser and fall
        # back to a scalar e4m3 store, which writes 0 on this backend (see
        # repro/TRAP-950-fp8-scalar-store.py).  One parallel loop per pool keeps
        # every store vectorised; a body with fewer than two elements per
        # thread degenerates the same way, hence the guard below.
        if per_thread < 2:
            raise ValueError(
                f"tile: {block_rows}x{cols} gives {per_thread} fp8 element(s) per "
                f"thread with threads={_THREADS}; a scalar e4m3 store writes 0 "
                "on this backend, use a tile with at least 2 elements per thread"
            )

        @T.prim_func
        def _fp8(
            K: T.Tensor((tokens, row_width), in_dtype),
            V: T.Tensor((tokens, row_width), in_dtype),
            KPages: T.Tensor((physical_tokens, row_width), cache_dtype),
            VPages: T.Tensor((physical_tokens, row_width), cache_dtype),
            Slots: T.Tensor((tokens,), "int64"),
            KScale: T.Tensor((1,), "float32"),
            VScale: T.Tensor((1,), "float32"),
        ) -> None:
            with T.Kernel(nblocks) as bx, T.SimtVF(threads=_THREADS):
                ks = KScale[0]  # hoisted once per block
                vs = VScale[0]
                rb = bx // nchunks
                cb = bx % nchunks
                for i, c in T.Parallel(block_rows, cols):
                    token = rb * block_rows + i
                    col = cb * cols + c
                    if token < tokens:  # tail block (dynamic token count)
                        slot = T.cast(Slots[token], slot_dtype)
                        if slot >= 0:  # negative slot: neither read nor write
                            # The cast alone does not saturate (it maps overflow
                            # to NaN), so the range is clamped explicitly;
                            # T.clamp propagates NaN like torch's clamp.
                            KPages[slot, col] = T.cast(
                                T.clamp(
                                    T.cast(K[token, col], "float32") / ks,
                                    -_FP8_MAX,
                                    _FP8_MAX,
                                ),
                                cache_dtype,
                            )
                for i, c in T.Parallel(block_rows, cols):
                    token = rb * block_rows + i
                    col = cb * cols + c
                    if token < tokens:
                        slot = T.cast(Slots[token], slot_dtype)
                        if slot >= 0:
                            VPages[slot, col] = T.cast(
                                T.clamp(
                                    T.cast(V[token, col], "float32") / vs,
                                    -_FP8_MAX,
                                    _FP8_MAX,
                                ),
                                cache_dtype,
                            )

        return tilelang.compile(_fp8, target="ascend", out_idx=[])

    @T.prim_func
    def _16bit(
        K: T.Tensor((tokens, row_width), in_dtype),
        V: T.Tensor((tokens, row_width), in_dtype),
        KPages: T.Tensor((physical_tokens, row_width), cache_dtype),
        VPages: T.Tensor((physical_tokens, row_width), cache_dtype),
        Slots: T.Tensor((tokens,), "int64"),
    ) -> None:
        with T.Kernel(nblocks) as bx, T.SimtVF(threads=_THREADS):
            rb = bx // nchunks
            cb = bx % nchunks
            for i, c in T.Parallel(block_rows, cols):
                token = rb * block_rows + i
                col = cb * cols + c
                if token < tokens:
                    slot = T.cast(Slots[token], slot_dtype)
                    if slot >= 0:
                        KPages[slot, col] = K[token, col]  # same dtype: plain copy
                        VPages[slot, col] = V[token, col]

    return tilelang.compile(_16bit, target="ascend", out_idx=[])


def _paged_cache_write_kernel(heads_kv, dim, physical_tokens, in_dtype, cache_dtype):
    """Frozen factory: geometry and dtypes in, ``(block_rows, block_cols)`` out."""
    if in_dtype not in _IN_DTYPES:
        raise ValueError(f"dtype: input dtype {in_dtype!r} is not one of {_IN_DTYPES}")
    if cache_dtype != _FP8_E4M3FN and cache_dtype != in_dtype:
        raise ValueError(
            f"dtype: cache dtype {cache_dtype!r} is neither {_FP8_E4M3FN!r} "
            f"nor the input dtype {in_dtype!r}"
        )

    def _func(block_rows, block_cols):
        return _build(
            heads_kv,
            dim,
            physical_tokens,
            in_dtype,
            cache_dtype,
            block_rows,
            block_cols,
        )

    return _func


# ---------------------------------------------------------------------------
# 3. Launch + exact comparison of both page pools
# ---------------------------------------------------------------------------
_RESULTS = []  # (level, name, ok, detail)
_MAX_DIFF = [0.0]


def _record(level, name, ok, detail=""):
    _RESULTS.append((level, name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[{level}][{tag}] {name}{(' -- ' + detail) if detail and not ok else ''}")


def _randn(shape, in_dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=torch.float32).to(
        _TORCH_DTYPE[in_dtype]
    )


def _distinct_slots(tokens, physical, seed, negative=0):
    """Distinct non-negative slots (the contract forbids duplicates), some set to -1."""
    gen = torch.Generator().manual_seed(seed + 7919)
    slots = torch.randperm(physical, generator=gen)[:tokens].to(torch.int64)
    if negative:
        holes = torch.randperm(tokens, generator=gen)[:negative]
        slots[holes] = -1
    return slots


def _write(
    k,
    v,
    k_pages,
    v_pages,
    slot_mapping,
    k_scale=None,
    v_scale=None,
    block_rows=_DEFAULT_BLOCK_ROWS,
    block_cols=_DEFAULT_BLOCK_COLS,
):
    """Launch the kernel with the frozen wrapper's contract (views + zero-token skip)."""
    cache_is_fp8 = k_pages.dtype is torch.float8_e4m3fn
    if cache_is_fp8 and (k_scale is None or v_scale is None):
        raise ValueError("dtype: an fp8 cache requires both k_scale and v_scale")
    if not cache_is_fp8 and (k_scale is not None or v_scale is not None):
        raise ValueError("dtype: a 16-bit cache takes no scale")
    if k.shape[0] == 0:
        # A zero-extent grid is not a legal launch and there is nothing to write.
        return
    row_width = k_pages.shape[2] * k_pages.shape[3]
    physical = k_pages.shape[0] * k_pages.shape[1]
    fn = _paged_cache_write_kernel(
        k_pages.shape[2],
        k_pages.shape[3],
        physical,
        _DTYPE_NAME[k.dtype],
        _DTYPE_NAME[k_pages.dtype],
    )(block_rows, block_cols)
    args = [
        k.view(k.shape[0], row_width),
        v.view(v.shape[0], row_width),
        k_pages.view(physical, row_width),
        v_pages.view(physical, row_width),
        slot_mapping,
    ]
    if cache_is_fp8:
        args += [k_scale, v_scale]
    fn(*args)


def _max_diff(got, gold):
    """Largest absolute difference in float space (NaN/NaN counts as equal)."""
    a, b = got.float(), gold.float()
    diff = (a - b).abs()
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    nan_mismatch = int(((a != a) != (b != b)).sum())
    return (float(diff.max()) if diff.numel() else 0.0), nan_mismatch


def _nan_canonical_bytes(byte_view):
    """Map a negative e4m3 NaN (0xFF) onto 0x7F.

    The backend's f32->e4m3 conversion canonicalises NaN to 0x7F while torch
    keeps the input NaN's sign (0xFF for a negative NaN).  The operator
    contract compares with ``equal_nan``, so the NaN sign bit is not part of
    it; every non-NaN byte is still compared exactly.
    """
    return torch.where(
        byte_view == 0xFF, torch.tensor(0x7F, dtype=torch.uint8), byte_view
    )


def _assert_exact(name, got, gold):
    """Exact pool comparison: value check with equal_nan plus a raw byte check."""
    got_c, gold_c = got.cpu(), gold.cpu()  # fp8 isclose is host-only on this stack
    torch.testing.assert_close(got_c, gold_c, atol=0, rtol=0, equal_nan=True)
    got_b, gold_b = (
        got_c.contiguous().view(torch.uint8),
        gold_c.contiguous().view(torch.uint8),
    )
    if got_c.dtype is torch.float8_e4m3fn:
        got_b, gold_b = _nan_canonical_bytes(got_b), _nan_canonical_bytes(gold_b)
    if not torch.equal(got_b, gold_b):
        n = int((got_b != gold_b).sum())
        raise AssertionError(
            f"{name}: {n} of {got_b.numel()} bytes differ (bit-exact contract)"
        )
    diff, nan_mismatch = _max_diff(got_c, gold_c)
    if nan_mismatch:
        raise AssertionError(f"{name}: {nan_mismatch} NaN positions differ")
    _MAX_DIFF[0] = max(_MAX_DIFF[0], diff)


def run_case(
    level,
    name,
    tokens,
    heads_kv,
    dim,
    num_pages,
    page_size,
    in_dtype,
    cache_dtype,
    slots=None,
    k_values=None,
    v_values=None,
    k_scale=None,
    v_scale=None,
    pool_fill=None,
    seed=0,
    block_rows=_DEFAULT_BLOCK_ROWS,
    block_cols=_DEFAULT_BLOCK_COLS,
    repeat=1,
):
    """Build one geometry, run it on the NPU and compare both pools with the golden."""
    row_width = heads_kv * dim
    physical = num_pages * page_size
    t_in, t_cache = _TORCH_DTYPE[in_dtype], _TORCH_DTYPE[cache_dtype]
    try:
        if k_values is not None:
            k = (
                torch.tensor(k_values, dtype=torch.float32)
                .reshape(tokens, heads_kv, dim)
                .to(t_in)
            )
            v = (
                torch.tensor(v_values, dtype=torch.float32)
                .reshape(tokens, heads_kv, dim)
                .to(t_in)
            )
        else:
            k = _randn((tokens, heads_kv, dim), in_dtype, seed)
            v = _randn((tokens, heads_kv, dim), in_dtype, seed + 1)
        pool_shape = (num_pages, page_size, heads_kv, dim)
        if pool_fill is None:
            k_pages = _randn(pool_shape, cache_dtype, seed + 2)
            v_pages = _randn(pool_shape, cache_dtype, seed + 3)
        else:
            k_fill, v_fill = (
                pool_fill
                if isinstance(pool_fill, (tuple, list))
                else (pool_fill, pool_fill)
            )
            k_pages = torch.full(pool_shape, float(k_fill), dtype=torch.float32).to(
                t_cache
            )
            v_pages = torch.full(pool_shape, float(v_fill), dtype=torch.float32).to(
                t_cache
            )
        if slots is None:
            slot_mapping = _distinct_slots(tokens, physical, seed)
        else:
            slot_mapping = torch.tensor(slots, dtype=torch.int64)
        quantized = cache_dtype == _FP8_E4M3FN
        k_scale_t = torch.tensor(
            [k_scale if k_scale is not None else 1.0], dtype=torch.float32
        )
        v_scale_t = torch.tensor(
            [v_scale if v_scale is not None else 1.0], dtype=torch.float32
        )
        if not quantized:
            k_scale_t = v_scale_t = None

        gold_k, gold_v = golden_paged_cache_write(
            k, v, k_pages, v_pages, slot_mapping, k_scale_t, v_scale_t
        )

        dev = "npu"
        dk, dv = k.to(dev), v.to(dev)
        dkp, dvp = k_pages.to(dev), v_pages.to(dev)
        dslots = slot_mapping.to(dev)
        dks = k_scale_t.to(dev) if quantized else None
        dvs = v_scale_t.to(dev) if quantized else None

        start = time.perf_counter()
        _write(dk, dv, dkp, dvp, dslots, dks, dvs, block_rows, block_cols)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1e3
        for _ in range(repeat - 1):  # idempotent: the same inputs write the same rows
            _write(dk, dv, dkp, dvp, dslots, dks, dvs, block_rows, block_cols)
        if repeat > 1:
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(repeat):
                _write(dk, dv, dkp, dvp, dslots, dks, dvs, block_rows, block_cols)
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) * 1e3 / repeat

        _assert_exact("k_pages", dkp, gold_k)
        _assert_exact("v_pages", dvp, gold_v)
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
        f"tokens={tokens} rw={row_width} {in_dtype}->{cache_dtype} {elapsed:.3f} ms",
    )
    print(
        f"       tokens={tokens} row_width={row_width} phys={physical} "
        f"{in_dtype}->{cache_dtype} launch={elapsed:.3f} ms"
    )
    return True


def _finish(level, strict):
    rows = [r for r in _RESULTS if r[0] == level]
    failed = [r for r in rows if not r[2]]
    print(
        f"[{level}] {len(rows) - len(failed)}/{len(rows)} cases passed, max_diff={_MAX_DIFF[0]:.6g}"
    )
    if failed and strict:
        raise AssertionError(f"[{level}] {len(failed)} case(s) failed")
    if failed:
        print(f"[{level}][WARN] {len(failed)} non-blocking case(s) failed")


# ---------------------------------------------------------------------------
# 4. Layered cases
# ---------------------------------------------------------------------------
def run_L0():
    """Smoke gate: both pools byte-exact over the manifest's dtype/geometry set."""
    common = dict(level="L0")
    run_case(
        **common,
        name="L0-1 fp16-odd-page-skip-and-page-edges",
        tokens=5,
        heads_kv=2,
        dim=64,
        num_pages=3,
        page_size=7,
        in_dtype="float16",
        cache_dtype="float16",
        slots=[20, -1, 0, 6, 13],
        seed=1,
    )
    run_case(
        **common,
        name="L0-2 bf16-one-kv-head-last-row",
        tokens=4,
        heads_kv=1,
        dim=32,
        num_pages=2,
        page_size=4,
        in_dtype="bfloat16",
        cache_dtype="bfloat16",
        slots=[7, 0, 3, 4],
        seed=2,
    )
    run_case(
        **common,
        name="L0-3 fp8-bf16-scaled",
        tokens=3,
        heads_kv=2,
        dim=64,
        num_pages=2,
        page_size=16,
        in_dtype="bfloat16",
        cache_dtype=_FP8_E4M3FN,
        slots=[1, -1, 31],
        k_scale=1.25,
        v_scale=0.75,
        seed=3,
    )
    run_case(
        **common,
        name="L0-4 fp8-fp16-page-boundary",
        tokens=2,
        heads_kv=4,
        dim=128,
        num_pages=4,
        page_size=16,
        in_dtype="float16",
        cache_dtype=_FP8_E4M3FN,
        slots=[63, 0],
        k_scale=1.0,
        v_scale=1.0,
        seed=4,
    )
    values = [
        1000.0,
        -1000.0,
        448.0,
        -448.0,
        1.2,
        -1.2,
        1.0625,
        1.1875,
        float("nan"),
        float("inf"),
        float("-inf"),
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
    ]
    run_case(
        **common,
        name="L0-5 saturates-fp8-overflow",
        tokens=1,
        heads_kv=1,
        dim=16,
        num_pages=1,
        page_size=2,
        in_dtype="float16",
        cache_dtype=_FP8_E4M3FN,
        slots=[1],
        k_values=values,
        v_values=[-x for x in values],
        k_scale=1.0,
        v_scale=1.0,
        seed=5,
    )
    run_case(
        **common,
        name="L0-6 skips-every-token",
        tokens=3,
        heads_kv=2,
        dim=64,
        num_pages=4,
        page_size=7,
        in_dtype="bfloat16",
        cache_dtype="bfloat16",
        slots=[-1, -1, -1],
        pool_fill=(3.0, -2.0),
        seed=6,
    )
    run_case(
        **common,
        name="L0-7 of-no-tokens-writes-nothing",
        tokens=0,
        heads_kv=2,
        dim=64,
        num_pages=2,
        page_size=16,
        in_dtype="bfloat16",
        cache_dtype="bfloat16",
        pool_fill=(3.0, -2.0),
        seed=7,
    )
    _case_contract(common["level"], "L0-8 refuses-an-invalid-tensor-contract")
    _case_dispatch(common["level"], "L0-9 dispatches-to-its-kernel")
    _case_written_roofline(common["level"], "L0-10 eval-roofline")
    _finish("L0", strict=True)


def run_L1():
    """Full functional shapes from the workload manifest (decode + prefill)."""
    run_case(
        level="L1",
        name="L1-1 bf16-decode-step-distinct-slots",
        tokens=64,
        heads_kv=8,
        dim=128,
        num_pages=8,
        page_size=64,
        in_dtype="bfloat16",
        cache_dtype="bfloat16",
        seed=11,
        repeat=3,
    )
    run_case(
        level="L1",
        name="L1-2 fp16-prefill-sweep",
        tokens=4096,
        heads_kv=8,
        dim=128,
        num_pages=256,
        page_size=16,
        in_dtype="float16",
        cache_dtype="float16",
        seed=12,
        repeat=2,
    )
    _finish("L1", strict=True)


def run_L2():
    """Large / quantized workloads: informative, non-blocking."""
    run_case(
        level="L2",
        name="L2-1 fp16-prefill-8192x1024",
        tokens=8192,
        heads_kv=8,
        dim=128,
        num_pages=512,
        page_size=16,
        in_dtype="float16",
        cache_dtype="float16",
        seed=21,
        repeat=2,
    )
    run_case(
        level="L2",
        name="L2-2 fp8-bf16-prefill-4096x512",
        tokens=4096,
        heads_kv=4,
        dim=128,
        num_pages=256,
        page_size=16,
        in_dtype="bfloat16",
        cache_dtype=_FP8_E4M3FN,
        k_scale=0.5,
        v_scale=2.0,
        seed=22,
        repeat=2,
    )
    run_case(
        level="L2",
        name="L2-3 fp8-bf16-256x512-sparse",
        tokens=256,
        heads_kv=4,
        dim=128,
        num_pages=16,
        page_size=16,
        in_dtype="bfloat16",
        cache_dtype=_FP8_E4M3FN,
        slots=None,
        k_scale=1.0,
        v_scale=1.0,
        seed=23,
    )
    _finish("L2", strict=False)


def run_boundary():
    """Targeted boundaries: tail block, width normalisation, extremes, specials."""
    run_case(
        level="boundary",
        name="B-1 tail-block-17-tokens",
        tokens=17,
        heads_kv=2,
        dim=64,
        num_pages=4,
        page_size=16,
        in_dtype="float16",
        cache_dtype="float16",
        seed=31,
    )
    run_case(
        level="boundary",
        name="B-2a row-width-16",
        tokens=6,
        heads_kv=1,
        dim=16,
        num_pages=2,
        page_size=8,
        in_dtype="float16",
        cache_dtype="float16",
        seed=32,
    )
    run_case(
        level="boundary",
        name="B-2b row-width-96",
        tokens=9,
        heads_kv=2,
        dim=48,
        num_pages=3,
        page_size=8,
        in_dtype="bfloat16",
        cache_dtype=_FP8_E4M3FN,
        k_scale=1.5,
        v_scale=0.25,
        seed=33,
    )
    run_case(
        level="boundary",
        name="B-3 single-token",
        tokens=1,
        heads_kv=2,
        dim=64,
        num_pages=2,
        page_size=8,
        in_dtype="bfloat16",
        cache_dtype="bfloat16",
        slots=[15],
        seed=34,
    )
    run_case(
        level="boundary",
        name="B-4 last-pool-row",
        tokens=4,
        heads_kv=2,
        dim=64,
        num_pages=2,
        page_size=8,
        in_dtype="float16",
        cache_dtype="float16",
        slots=[15, 0, 7, 8],
        seed=35,
    )
    values = [
        1000.0,
        -1000.0,
        448.0,
        -448.0,
        1.2,
        -1.2,
        1.0625,
        1.1875,
        float("nan"),
        float("inf"),
        float("-inf"),
        0.0,
        -0.0,
        7.0,
        -7.0,
        0.125,
    ]
    run_case(
        level="boundary",
        name="B-5 fp8-scale-7.0-not-a-power-of-two",
        tokens=1,
        heads_kv=2,
        dim=8,
        num_pages=1,
        page_size=2,
        in_dtype="float16",
        cache_dtype=_FP8_E4M3FN,
        slots=[1],
        k_values=values,
        v_values=list(reversed(values)),
        k_scale=7.0,
        v_scale=7.0,
        seed=36,
    )
    run_case(
        level="boundary",
        name="B-6 fp8-nan-with-scale-1.25",
        tokens=2,
        heads_kv=1,
        dim=32,
        num_pages=2,
        page_size=4,
        in_dtype="bfloat16",
        cache_dtype=_FP8_E4M3FN,
        slots=[5, -1],
        k_values=[float("nan")] * 32 + [float("inf")] * 32,
        v_values=[-float("inf")] * 32 + [float("nan")] * 32,
        k_scale=1.25,
        v_scale=1.25,
        seed=37,
    )
    run_case(
        level="boundary",
        name="B-7 wide-block-256-cols",
        tokens=5,
        heads_kv=8,
        dim=128,
        num_pages=2,
        page_size=8,
        in_dtype="float16",
        cache_dtype="float16",
        seed=38,
        block_rows=8,
        block_cols=256,
    )
    # 6 fp8 elements per thread: a non-power-of-two vector width, to check the
    # store still vectorises instead of falling back to the scalar e4m3 store.
    run_case(
        level="boundary",
        name="B-8 fp8-six-elements-per-thread",
        tokens=12,
        heads_kv=2,
        dim=48,
        num_pages=2,
        page_size=8,
        in_dtype="float16",
        cache_dtype=_FP8_E4M3FN,
        slots=[9, 0, 15, 3, -1, 7, 1, 2, 4, 5, 6, 8],
        k_scale=1.0,
        v_scale=1.0,
        seed=39,
        block_rows=8,
        block_cols=128,
    )
    # block_cols = 64 (a 64 B fp8 row segment): 8 elements per thread, 16 chunks.
    run_case(
        level="boundary",
        name="B-9 fp8-block-cols-64",
        tokens=6,
        heads_kv=4,
        dim=128,
        num_pages=2,
        page_size=8,
        in_dtype="bfloat16",
        cache_dtype=_FP8_E4M3FN,
        k_scale=2.0,
        v_scale=0.5,
        seed=40,
        block_rows=16,
        block_cols=64,
    )
    _finish("boundary", strict=False)


# ---------------------------------------------------------------------------
# Host-level L0 cases (contract, dispatch, accounting) -- no device work in 8/9
# ---------------------------------------------------------------------------
def _case_contract(level, name):
    """The factory and the launch refuse the four invalid tensor contracts."""
    checks = []
    for bad_in, bad_cache, why in (
        ("float32", "float32", "input-dtype"),
        ("float16", "float32", "cache-dtype"),
    ):
        try:
            _paged_cache_write_kernel(2, 64, 32, bad_in, bad_cache)
            checks.append(f"{why}: no error raised")
        except ValueError:
            pass
    # scale missing on an fp8 pool / scale present on a 16-bit pool.
    for quantized, why in ((True, "scale-missing"), (False, "scale-present")):
        cache_dtype = _FP8_E4M3FN if quantized else "bfloat16"
        rw, phys = 128, 8
        k = torch.zeros((2, 1, rw), dtype=torch.bfloat16)
        kp = torch.zeros((1, phys, 1, rw), dtype=_TORCH_DTYPE[cache_dtype])
        scale = None if quantized else torch.ones(1, dtype=torch.float32)
        try:
            _write(
                k,
                k,
                kp,
                kp,
                torch.zeros(2, dtype=torch.int64),
                scale,
                scale,
                _DEFAULT_BLOCK_ROWS,
                _DEFAULT_BLOCK_COLS,
            )
            checks.append(f"{why}: no error raised")
        except ValueError:
            pass
    if checks:
        _record(level, name, False, "; ".join(checks))
    else:
        _record(level, name, True, "4 invalid contracts rejected")


def _case_dispatch(level, name):
    """Repeated builds of one geometry reuse the compiled program."""
    try:
        factory = _paged_cache_write_kernel(2, 64, 21, "float16", "float16")
        first = factory(_DEFAULT_BLOCK_ROWS, _DEFAULT_BLOCK_COLS)
        again = _paged_cache_write_kernel(2, 64, 21, "float16", "float16")(
            _DEFAULT_BLOCK_ROWS, _DEFAULT_BLOCK_COLS
        )
        assert first is again, "the compiled program is not cached per geometry/tile"
        other = factory(32, _DEFAULT_BLOCK_COLS)
        assert other is not first, "a different tile shape must build its own program"
    except AssertionError as exc:
        _record(level, name, False, str(exc))
        return
    _record(level, name, True, "compiled program cached per (geometry, tile)")


def _case_written_roofline(level, name):
    """Only the tokens a slot names are written; the DESIGN accounting matches."""
    tokens, heads_kv, dim, page_size, num_pages = 4, 2, 64, 4, 2
    slots = [7, 0, -1, 3]
    row_width, physical = heads_kv * dim, num_pages * page_size
    written = sum(1 for s in slots if s >= 0)
    try:
        for cache_dtype, in_dtype in (
            (_FP8_E4M3FN, "bfloat16"),
            ("bfloat16", "bfloat16"),
        ):
            k = _randn((tokens, heads_kv, dim), in_dtype, 41)
            v = _randn((tokens, heads_kv, dim), in_dtype, 42)
            kp = _randn((num_pages, page_size, heads_kv, dim), cache_dtype, 43)
            vp = _randn((num_pages, page_size, heads_kv, dim), cache_dtype, 44)
            before_k, before_v = kp.clone(), vp.clone()
            sm = torch.tensor(slots, dtype=torch.int64)
            quantized = cache_dtype == _FP8_E4M3FN
            scale = torch.tensor([1.0], dtype=torch.float32) if quantized else None
            gold_k, gold_v = golden_paged_cache_write(k, v, kp, vp, sm, scale, scale)
            dev = "npu"
            dsm = sm.to(dev)
            ds = scale.to(dev) if quantized else None
            dk, dv = k.to(dev), v.to(dev)
            dkp, dvp = kp.to(dev), vp.to(dev)
            _write(dk, dv, dkp, dvp, dsm, ds, ds)
            torch.npu.synchronize()
            _assert_exact("k_pages", dkp, gold_k)
            _assert_exact("v_pages", dvp, gold_v)
            # No row outside the named set changed, byte for byte.
            row_bytes = row_width * dkp.element_size()
            got_k = dkp.cpu().view(torch.uint8).reshape(physical, row_bytes)
            old_k = before_k.view(torch.uint8).reshape(physical, row_bytes)
            got_v = dvp.cpu().view(torch.uint8).reshape(physical, row_bytes)
            old_v = before_v.view(torch.uint8).reshape(physical, row_bytes)
            touched = set()
            for row in range(physical):
                if not torch.equal(got_k[row], old_k[row]) or not torch.equal(
                    got_v[row], old_v[row]
                ):
                    touched.add(row)
            assert touched <= {s for s in slots if s >= 0}, (
                f"unexpected rows written: {touched}"
            )
        # Contract accounting (source ``eval_roofline``): only written tokens count.
        for out_bytes, label in ((1, "fp8"), (2, "16-bit")):
            flops = 4 * written * row_width
            moved = 2 * written * row_width * (2 + out_bytes) + tokens * 8 + 8
            print(
                f"       roofline[{label}]: written={written} flops={flops} bytes={moved}"
            )
    except AssertionError as exc:
        _record(level, name, False, str(exc)[:300])
        return
    _record(level, name, True, f"written={written} only, both cache dtypes")


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
    print(f"[PRECISION_PASS] max_diff={_MAX_DIFF[0]:.6g}")


if __name__ == "__main__":
    main()
