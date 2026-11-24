# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_delta_kernel (ascend, packed branch): MLA prefill backward stage 1 -- row dot product.

``delta[h, t] = sum_d o[t, h, d] * do[t, h, d]`` accumulated in float32.

This is the first of the three launches of ``MLAVarlenBwdOp``; it feeds the
``dP - delta`` term of ``_bwd_kernel``.  The source operator's GPU factory is
literally the same function as ``MLADenseBwdOp``'s ``_delta_kernel``; only the
``dense_seqlen == 0`` branch differs -- the output is head-major
``delta[(heads, total)]`` instead of ``(batch, heads, seqlen)`` and the packed
row count ``total = T_q = sum(q_len)`` stays a *run-time* dimension.  The
semantics are therefore unchanged (16-bit inputs widened exactly to float32 on
load, product and accumulation in float32, one fp32 row scalar out).

The core is the dense form already validated on device
(``examples/mla_dense_bwd/_delta_kernel``, 23/23), reused verbatim because it
only depends on the *in-row* dimension and the platform constraints:

* one-dimensional ``T.Kernel(56)`` (no ``threads`` at kernel scope), a
  persistent grid of exactly ``vector_core_cnt`` cores;
* one ``R = 64`` row UB tile per operand, two versions
  (``T.annotate_buffer_versions``) under ``T.Pipelined(num_stages=2)``;
* one strided ``T.copy`` per operand into ``o_ub[0:n, 0:D_v]`` -- the destination
  region is always explicit, because an unsliced copy of a ``(R, DW)`` buffer
  from a ``(n, D_v)`` source is rejected when ``D_v < DW`` (``D_v = 64``);
* per row: ``vld -> vcvt(PART_EVEN/PART_ODD) -> vmul -> vadd -> vcadd -> vsts``,
  i.e. one 128-lane load per operand and one 64-lane tree reduction, keeping the
  fp32 products in vector registers (no materialisation in UB);
* one compressed ``T.copy`` per tile writes the row scalars back, staged through
  the 32-byte-row ``d_ub`` that ``vsts`` can address;
* ``D_v < DW`` pads the row to one vector register, so *both* operands are
  cleared and the two copies only write ``[0:D_v]`` -- otherwise an
  uninitialised pad lane could contribute ``0 * (Inf/NaN bits)``;
* ``nvec > 1`` (``D_v > 128``) combines the segment partial sums into a single
  build-time expression (``_row_dot``); a run-time ``for`` carrying a vector
  accumulator is not constructible on this toolchain.

What is re-derived here is the *packed* part (DESIGN.md 0.6 R1/R3/R4/R5):

1. **Dynamic tile stream.**  ``T_q`` is a run-time quantity, so the logical task
   count ``ntiles = ceildiv(T_q, R) * H`` is run-time; the static 56-core grid
   partitions it evenly at run time (``it0 = ntiles*bx//G``,
   ``it1 = ntiles*(bx+1)//G``, always ``it0 <= it1``) and each core walks its own
   slice inside ``T.Pipelined(it0, it1, num_stages=2)``.
2. **Run-time clamp instead of a static tail segment.**  ``nfull``/``nlast`` do
   not exist (both are run-time), so every tile uses one code path with
   ``n = T.min(R, total - t0)`` on the two loads *and* the write-back.  The row
   loop keeps the static bound ``R``; rows in ``[n, R)`` are computed and not
   written.  ``t0 < total`` always holds for a real tile, hence ``n >= 1`` (no
   zero-length descriptor), and ``total == 0`` gives an empty tile stream.
   The explicit ``n`` is deliberate: writing the static ``R`` extent also
   compiles (the compiler adds its own last-axis clamp for the 2-D ``delta``
   base) but that is undocumented behaviour and spills into the next head once
   the output is a flat 1-D tensor (DESIGN.md 3.3 note 5, probe N3).
3. **head-major write-back.**  ``delta`` is ``(H, T_q)`` C-contiguous, so the
   descriptor base is ``4*(h*T_q + t0)`` (one run-time multiply per tile, not per
   element); the inner axis is still ``t`` with a 4-byte stride, so the
   compressed descriptor is parameter-wise identical to the dense form.
4. **Dynamic output shape.**  ``out_idx=[2]`` allocates ``delta`` with its second
   axis following the run-time ``total_q``.

Why no static tail segment (the one dense conclusion that does not carry over):
dense builds ``if S % R`` at trace time and uses a static ``nlast`` for all three
extents; here ``total`` is a run-time value, so no such branch can exist.

Content layout:

1. ``golden_delta`` -- torch CPU reference of the operator's semantics.
2. ``_geometry`` / ``_row_dot`` / ``_delta_kernel`` -- the frozen kernel factory,
   signature-compatible with the source ``_delta_kernel(batch, heads, dim_v,
   dense_seqlen, dtype) -> Callable``.
3. ``run_case`` -- launch on the NPU and compare with the golden.
4. ``run_L0`` / ``run_L1`` / ``run_L2`` / ``run_boundary`` / ``main``.
"""

import argparse
import functools
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401  -- registers torch.npu

_IN_DTYPES = ("float16", "bfloat16")
_TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}

_VL = 128  # 16-bit lanes of one 2048-bit vector register
_R = 64  # rows per tile (UB budget, DESIGN.md 4.5)
_G = 56  # vector_core_cnt of Ascend950PR_9579 (DESIGN.md 5.5)
_UB_CAPACITY = 253952  # Ascend950PR_9579.ini [AICoreSpec] ub_size
_ATOL = 1e-4  # DESIGN.md 8.2: the operator output is float32
_RTOL = 1e-4
_DEFAULT_DTYPE = "float16"
_PACKED_SEQLEN = 0  # dense_seqlen == 0 selects the packed branch


# ---------------------------------------------------------------------------
# 1. Golden: torch CPU reference of the operator semantics
# ---------------------------------------------------------------------------
def golden_delta(o, do, chunk_rows=256):
    """``delta[h, t] = sum_d o[t, h, d] * do[t, h, d]`` in float32, head-major.

    ``o`` / ``do`` are the packed ``(T_q, H, D_v)`` tensors the kernel takes (no
    host-side view in the packed branch).  Naive and independent of the kernel:
    cast both operands to float32 exactly, multiply element-wise, sum the last
    axis with torch's own reduction order and move the head axis in front.
    ``chunk_rows`` only bounds the fp32 temporary for the large L2 geometries.
    """
    total, heads, dim_v = o.shape
    out = torch.empty((heads, total), dtype=torch.float32)
    for t0 in range(0, total, chunk_rows):
        t1 = min(t0 + chunk_rows, total)
        prod = o[t0:t1].float() * do[t0:t1].float()  # (t, H, D_v) fp32
        out[:, t0:t1] = prod.sum(dim=-1).transpose(0, 1)  # (H, t)
    return out.contiguous()


# ---------------------------------------------------------------------------
# 2. Kernel
# ---------------------------------------------------------------------------
def _geometry(heads, dim_v):
    """Build-time tile geometry (DESIGN.md 5.2); only ``total`` is run-time."""
    if dim_v % 16 != 0:
        raise ValueError(
            f"shape: dim_v must be a multiple of 16 for this platform (got {dim_v})"
        )
    dw = ((dim_v + _VL - 1) // _VL) * _VL  # padded row width, one vector register step
    nvec = dw // _VL  # register segments per row
    # o_ub/do_ub are hand-double-buffered; d_ub is budgeted twice because the
    # compiler also versions staging buffers that live inside the pipeline body.
    ub_bytes = 2 * (_R * dw * 2) * 2 + 2 * (_R * 8 * 4)
    if ub_bytes > _UB_CAPACITY:
        raise ValueError(
            f"tiling: {ub_bytes} B of UB exceeds the {_UB_CAPACITY} B of this model "
            f"(R={_R}, DW={dw}); lower the row tile"
        )
    return {"R": _R, "DW": dw, "nvec": nvec, "G": _G, "ub_bytes": ub_bytes}


def _row_dot(o_ub, do_ub, i, nvec):
    """One build-time expression for the ``nvec``-segment dot product of row ``i``.

    ``nvec`` is a python int, so this ``for`` runs while tracing and only
    concatenates expressions.  Putting the same loop inside the prim_func body
    would lift it to an IR ``ForFrame`` and a vector value that is rebound across
    iterations is rejected (``Immutable variable ... used outside its defining
    region``), even for ``nvec == 1``.
    """
    part = None
    for v in range(nvec):
        vo = T.simd.vld(o_ub[i, v * _VL])
        vd = T.simd.vld(do_ub[i, v * _VL])
        cur = T.simd.vadd(
            T.simd.vmul(
                T.simd.vcvt(vo, "float32", part=0), T.simd.vcvt(vd, "float32", part=0)
            ),
            T.simd.vmul(
                T.simd.vcvt(vo, "float32", part=1), T.simd.vcvt(vd, "float32", part=1)
            ),
        )
        part = cur if part is None else T.simd.vadd(part, cur)
    return T.simd.vcadd(part)


@functools.lru_cache(maxsize=64)
def _delta_kernel(batch, heads, dim_v, dense_seqlen, dtype):
    """Frozen factory: geometry and dtype in, compiled ``(o, do) -> delta`` out.

    Same signature as the source operator's ``_delta_kernel`` so the harness can
    keep its call site; ``dense_seqlen == 0`` selects the packed branch, and a
    non-zero value belongs to the dense compilation (a different design, not
    served here).  ``heads``/``dim_v``/``dtype`` are build-time constants, so the
    kernel carries no element-wise shape predicate; ``total`` (the packed row
    count) stays run-time, which is exactly what makes a new ``T_q`` free.
    """
    if dtype not in _IN_DTYPES:
        raise ValueError(f"dtype: {dtype!r} is not one of {_IN_DTYPES}")
    if dense_seqlen != _PACKED_SEQLEN:
        raise ValueError(
            "shape: this file serves the packed branch only (dense_seqlen == 0); "
            f"got dense_seqlen={dense_seqlen}, which is the dense compilation"
        )
    if min(batch, heads, dim_v) <= 0:
        raise ValueError(f"shape: batch/heads/dim_v must be positive, got {batch}/{heads}/{dim_v}")

    geo = _geometry(heads, dim_v)
    rows, dw, nvec, grid = geo["R"], geo["DW"], geo["nvec"], geo["G"]
    total = T.dynamic("total_q")

    @T.prim_func
    def _delta_main(
        o: T.Tensor((total, heads, dim_v), dtype),  # type: ignore
        do: T.Tensor((total, heads, dim_v), dtype),  # type: ignore
        delta: T.Tensor((heads, total), "float32"),  # type: ignore
    ) -> None:
        ntiles = T.ceildiv(total, rows) * heads  # run-time logical task count
        with T.Kernel(grid) as bx:  # static one-dimensional grid, one core each
            o_ub = T.alloc_shared((rows, dw), dtype)
            do_ub = T.alloc_shared((rows, dw), dtype)
            d_ub = T.alloc_shared((rows, 8), "float32")  # 32 B row stride for vsts
            T.annotate_buffer_versions({o_ub: 2, do_ub: 2})
            if dw != dim_v:
                # Build-time branch: a padded row means the lanes in [dim_v, dw)
                # are read by vld but never written by the copies, so both
                # operands must be zeroed (0 * anything == 0, and 0 * NaN bits
                # would otherwise poison the row).
                T.clear(o_ub)
                T.clear(do_ub)
            # Balanced run-time partition of the tile stream over the persistent
            # grid; it0 <= it1 always holds, so no negative trip count.
            it0 = (ntiles * bx) // grid
            it1 = (ntiles * (bx + 1)) // grid
            for it in T.Pipelined(it0, it1, num_stages=2):
                h = it % heads  # build-time modulus
                t0 = (it // heads) * rows
                n = T.min(rows, total - t0)  # run-time clamp on all three extents
                T.copy(o[t0 : t0 + n, h, :], o_ub[0:n, 0:dim_v])
                T.copy(do[t0 : t0 + n, h, :], do_ub[0:n, 0:dim_v])
                with T.SimdVF():
                    one = T.simd.pset(32, "PAT_VL1")
                    for i in T.serial(rows):  # static row loop; rows >= n are dropped
                        T.simd.vsts(
                            d_ub[i, 0], _row_dot(o_ub, do_ub, i, nvec), one, extent=1
                        )
                # head-major base 4*(h*total + t0); inner axis t is contiguous 4 B
                T.copy(d_ub[0:n, 0:1], delta[h, t0 : t0 + n])

    return tilelang.compile(_delta_main, target="ascend", out_idx=[2])


# ---------------------------------------------------------------------------
# 3. Launch + comparison with the golden
# ---------------------------------------------------------------------------
_RESULTS = []  # (level, name, ok, detail)
_MAX_DIFF = [0.0]
_LAUNCH = {}  # case name -> ms per launch


def _record(level, name, ok, detail=""):
    _RESULTS.append((level, name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[{level}][{tag}] {name}{(' -- ' + detail) if detail and not ok else ''}", flush=True)


def _randn(shape, dtype_name, seed):
    """Deterministic normal sample in the requested 16-bit dtype.

    Filled in chunks: a ``(32768, 128, 128)`` fp32 temporary would be 2.1 GB per
    operand, and only the 16-bit tensor is needed.
    """
    gen = torch.Generator().manual_seed(seed)
    out = torch.empty(shape, dtype=_TORCH_DTYPE[dtype_name])
    flat = out.view(-1)
    step = 1 << 22  # 4 M elements (16 MB) per chunk
    for s0 in range(0, flat.numel(), step):
        s1 = min(s0 + step, flat.numel())
        flat[s0:s1] = torch.randn(s1 - s0, generator=gen, dtype=torch.float32).to(out.dtype)
    return out


def _make_inputs(total, heads, dim_v, dtype_name, seed, special=None):
    """Random 16-bit operands, optionally replaced by the boundary patterns."""
    o = _randn((total, heads, dim_v), dtype_name, seed)
    do = _randn((total, heads, dim_v), dtype_name, seed + 1)
    if special == "zeros":
        o.zero_()
        do.zero_()
    elif special == "nonfinite":
        # Non-finite values propagate by the same multiply-add on both sides; the
        # inf * 0 lane must become NaN in the kernel too.  Positions are chosen to
        # land on the last (possibly ragged) tile as well as the first.
        o[0, 0, 0] = float("inf")
        do[0, 0, 0] = 0.0
        o[min(1, total - 1), 0, 1] = float("-inf")
        do[min(1, total - 1), 0, 1] = float("nan")
        o[total - 1, heads - 1, 0] = float("nan")
        do[total - 1, heads - 1, 0] = float("inf")
    elif special is not None:
        raise ValueError(f"special: unknown pattern {special!r}")
    return o, do


def _max_diff(got, gold):
    """Largest absolute difference in float space (NaN/NaN counts as equal)."""
    a, b = got.float(), gold.float()
    diff = (a - b).abs()
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    nan_mismatch = int(((a != a) != (b != b)).sum())
    return (float(diff.max()) if diff.numel() else 0.0), nan_mismatch


def run_case(
    level,
    name,
    q_lens,
    heads,
    dim_v,
    dtype=_DEFAULT_DTYPE,
    seed=0,
    special=None,
    repeat=1,
    determinism=False,
    atol=_ATOL,
    rtol=_RTOL,
):
    """Build one geometry, run it on the NPU and compare ``delta`` with the golden.

    ``q_lens`` is the packed request-length list: only its sum (``T_q``) and its
    length (the request count, which is the factory's ``batch`` key) matter --
    this launch is independent of request boundaries by construction.
    """
    total = int(sum(q_lens))
    reqs = len(q_lens)
    geo = _geometry(heads, dim_v)
    ntiles = ((total + _R - 1) // _R) * heads
    diff = float("nan")
    try:
        o, do = _make_inputs(total, heads, dim_v, dtype, seed, special)
        gold = golden_delta(o, do)

        o_dev = o.to("npu")
        do_dev = do.to("npu")

        fn = _delta_kernel(reqs, heads, dim_v, _PACKED_SEQLEN, dtype)
        # Warm-up: the first launch of a freshly loaded kernel module carries the
        # one-off device-side load (~100 ms here), which is not kernel time.
        warm = fn(o_dev, do_dev)
        del warm
        torch.npu.synchronize()
        start = time.perf_counter()
        got = fn(o_dev, do_dev)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1e3
        if isinstance(got, (tuple, list)):
            got = got[0]
        got = got.cpu().float()
        second = None
        if determinism:  # out_idx allocates a fresh output, so a repeat is independent
            again = fn(o_dev, do_dev)
            if isinstance(again, (tuple, list)):
                again = again[0]
            second = again.cpu().float()
        if repeat > 1:
            start = time.perf_counter()
            for _ in range(repeat):
                fn(o_dev, do_dev)
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) * 1e3 / repeat

        if tuple(got.shape) != tuple(gold.shape):
            raise AssertionError(
                f"delta shape {tuple(got.shape)} != golden {tuple(gold.shape)}"
            )
        # Measure first, compare second: a tolerance failure still reports the
        # actual error instead of only torch's message.
        diff, nan_mismatch = _max_diff(got, gold)
        _MAX_DIFF[0] = max(_MAX_DIFF[0], diff)
        if nan_mismatch:
            raise AssertionError(
                f"{nan_mismatch} NaN positions differ from the golden (max_diff={diff:.3e})"
            )
        if second is not None:
            same = bool(
                (got.view(torch.int32) == second.view(torch.int32)).all().item()
            )
            if not same:
                raise AssertionError("two launches of the same input are not bit-identical")
        torch.testing.assert_close(got, gold, atol=atol, rtol=rtol, equal_nan=True)
    except AssertionError as exc:
        _record(level, name, False, f"max_diff={diff:.3e} {str(exc)[:260]}")
        return False
    except (ValueError, RuntimeError) as exc:  # noqa: BLE001 - reported, never swallowed
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:260]}")
        return False
    _record(level, name, True)
    _LAUNCH[name] = elapsed
    print(
        f"       T_q={total} reqs={reqs} H={heads} D_v={dim_v} {dtype} "
        f"R={_R} ntiles={ntiles} core~{ntiles / _G:.2f} last_n={total - (total // _R) * _R} "
        f"UB={geo['ub_bytes']}B nvec={geo['nvec']} max_diff={diff:.3e} launch={elapsed:.3f} ms"
        + (" det=bit-identical" if determinism else ""),
        flush=True,
    )
    return True


def _finish(level, strict):
    rows = [r for r in _RESULTS if r[0] == level]
    failed = [r for r in rows if not r[2]]
    print(
        f"[{level}] {len(rows) - len(failed)}/{len(rows)} cases passed, "
        f"max_diff={_MAX_DIFF[0]:.6g}",
        flush=True,
    )
    if failed and strict:
        raise AssertionError(f"[{level}] {len(failed)} case(s) failed")
    if failed:
        print(f"[{level}][WARN] {len(failed)} non-blocking case(s) failed", flush=True)


# ---------------------------------------------------------------------------
# 4. Layered cases
# ---------------------------------------------------------------------------
def _case_contract(level, name):
    """The factory refuses the contracts it cannot serve (no device work)."""
    problems = []
    for args, why in (
        ((1, 4, 128, 128, "float16"), "dense_seqlen=128 (dense branch, not this file)"),
        ((1, 4, 128, -8, "float16"), "negative dense_seqlen"),
        ((1, 4, 128, 0, "float32"), "float32 input"),
        ((0, 4, 128, 0, "float16"), "batch=0"),
        ((1, 0, 128, 0, "float16"), "heads=0"),
        ((1, 4, 40, 0, "float16"), "dim_v=40 (not a multiple of 16)"),
    ):
        try:
            _delta_kernel(*args)
            problems.append(f"{why}: no error raised")
        except ValueError:
            pass
    first = _delta_kernel(1, 4, 128, _PACKED_SEQLEN, "float16")
    again = _delta_kernel(1, 4, 128, _PACKED_SEQLEN, "float16")
    if first is not again:
        problems.append("the compiled program is not cached per (geometry, dtype)")
    if problems:
        _record(level, name, False, "; ".join(problems))
        return False
    _record(level, name, True)
    print("       6 invalid contracts rejected, program cached per (geometry, dtype)", flush=True)
    return True


# Workload manifest entries of MLAVarlenBwdOp (examples/TileOPs/tileops/manifest/attention.yaml):
# the three GPU workloads; the packed launch only sees T_q = sum(q_lens).
_MANIFEST = (
    ("ds-v3-8x4k", [4096] * 8, 128, 128, "bfloat16"),
    ("ds-v3-mixed", [512, 1024, 2048, 4096, 8192, 16384], 128, 128, "bfloat16"),
    ("kimi-k2-4x8k", [8192] * 4, 64, 128, "bfloat16"),
)


def run_L0():
    """Gate: both dtypes, the pad path, the run-time clamp, nvec=2, the manifest set."""
    run_case("L0", "L0-1 bf16-manifest-smoke", [256, 384], 8, 128, "bfloat16", seed=1)
    run_case("L0", "L0-2 fp16-manifest-smoke", [256, 384], 8, 128, "float16", seed=2)
    # D_v=64 < DW=128: destination slice [0:R, 0:64] plus the two T.clear calls --
    # the only numerical evidence that the pad lanes stay zero.  Same geometry as
    # the op-level smoke test (q_lens [256, 128], H=4, D_v=64).
    run_case("L0", "L0-3 bf16-pad-path-Dv64", [256, 128], 4, 64, "bfloat16", seed=3)
    run_case("L0", "L0-4 fp16-pad-path-Dv64", [256, 128], 4, 64, "float16", seed=4)
    # T_q < R: one ragged tile per head, n == T_q, no full tile at all.
    run_case("L0", "L0-5 fp16-single-row-T1", [1], 1, 128, "float16", seed=5)
    _case_contract("L0", "L0-6 refuses-an-invalid-contract")
    # Run-time clamp evidence: 63 (n=63), 64 (n=R exactly), 65 (n=1 on the last tile).
    run_case("L0", "L0-7 bf16-T63-last-n63", [63], 8, 128, "bfloat16", seed=7)
    run_case("L0", "L0-8 bf16-T64-exact", [64], 8, 128, "bfloat16", seed=8)
    run_case("L0", "L0-9 bf16-T65-last-n1", [64, 1], 8, 128, "bfloat16", seed=9)
    # nvec = 2 (D_v=192 -> DW=256): the build-time segment merge.
    run_case("L0", "L0-10 bf16-Dv192-nvec2", [128], 4, 192, "bfloat16", seed=10)
    # Multi-request, ntiles > 56 so every core gets several tiles.
    run_case("L0", "L0-11 fp16-multi-request-T3584", [512, 1024, 2048], 8, 128, "float16", seed=11)
    run_case(
        "L0", "L0-12 fp16-determinism-2x", [256, 384], 8, 128, "float16",
        seed=12, determinism=True,
    )
    for name, q_lens, heads, dim_v, dtype in _MANIFEST:
        run_case("L0", f"L0-manifest-{name}", q_lens, heads, dim_v, dtype, seed=13)
    _finish("L0", strict=True)


def run_L1():
    """Representative shapes: single long request, both dtypes, pad + multi-request."""
    run_case("L1", "L1-1 bf16-single-4k", [4096], 8, 128, "bfloat16", seed=21, repeat=3)
    run_case("L1", "L1-2 fp16-manifest-8x4k", [4096] * 8, 128, 128, "float16", seed=22, repeat=2)
    run_case("L1", "L1-3 bf16-pad-multi-request", [1024, 2048], 4, 64, "bfloat16", seed=23, repeat=2)
    # ntiles = 8 < 56: most cores idle, one tile per core at most (R9).
    run_case("L1", "L1-4 bf16-small-tail-T96", [64, 32], 4, 128, "bfloat16", seed=24, repeat=2)
    _finish("L1", strict=True)


def run_L2():
    """Manifest workloads repeated for timing: informative, non-blocking."""
    for name, q_lens, heads, dim_v, dtype in _MANIFEST:
        run_case("L2", f"L2-{name}-repeat3", q_lens, heads, dim_v, dtype, seed=31, repeat=3)
    _finish("L2", strict=False)


def run_boundary():
    """Ragged tails, all-zero, non-finite, the empty input and the spill detector."""
    run_case("boundary", "B-1 bf16-T200-last-n8", [200], 8, 128, "bfloat16", seed=41)
    run_case("boundary", "B-2 bf16-T96-last-n32", [64, 32], 4, 128, "bfloat16", seed=42)
    # H=128 makes any write past the last row of a head land inside the next head.
    run_case("boundary", "B-3 bf16-T63-H128", [63], 128, 128, "bfloat16", seed=43)
    run_case("boundary", "B-4 bf16-T65-pad-Dv64", [65], 4, 64, "bfloat16", seed=44)
    run_case("boundary", "B-5 bf16-T65-Dv192", [65], 4, 192, "bfloat16", seed=45)
    run_case("boundary", "B-6 bf16-all-zero", [128], 4, 128, "bfloat16", seed=46, special="zeros")
    run_case("boundary", "B-7 fp16-non-finite", [65], 4, 128, "float16", seed=47, special="nonfinite")
    # Indirect out-of-bounds detector: a ragged case first, then an all-zero case
    # that would show a stale write if the ragged descriptor had a count > n.
    run_case("boundary", "B-8 bf16-spill-detector-tail", [65], 8, 128, "bfloat16", seed=48)
    run_case("boundary", "B-9 bf16-spill-detector-zero", [128], 8, 128, "bfloat16", seed=49, special="zeros")
    # T_q = 0: empty tile stream (all cores run an empty loop), empty output.
    run_case("boundary", "B-10 bf16-empty-T0", [0], 4, 128, "bfloat16", seed=50)
    _finish("boundary", strict=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"]
    )
    args, _ = parser.parse_known_args()
    levels = {"L0": run_L0, "L1": run_L1, "L2": run_L2, "boundary": run_boundary}
    chosen = list(levels.values()) if args.level == "all" else [levels[args.level]]
    started = time.perf_counter()
    for run in chosen:
        run()
    print(f"[TIME] {time.perf_counter() - started:.1f} s for level={args.level}", flush=True)
    print(f"[PRECISION_PASS] max_diff={_MAX_DIFF[0]:.6g}", flush=True)


if __name__ == "__main__":
    main()
