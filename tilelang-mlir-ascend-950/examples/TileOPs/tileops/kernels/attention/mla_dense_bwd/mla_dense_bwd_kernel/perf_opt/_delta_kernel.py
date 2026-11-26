# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""_delta_kernel (ascend): MLA prefill backward stage 1 -- row dot product.

``delta[b, h, s] = sum_d o[s, h, d] * do[s, h, d]`` accumulated in float32.

This is the first of the three launches of ``MLADenseBwdOp``; it feeds the
``dP - delta`` term of ``_bwd_kernel``.  The kernel keeps the source
operator's semantics (16-bit inputs widened exactly to float32 on load,
product and accumulation in float32, one fp32 row scalar out) and replaces the
GPU implementation's 2-D grid + 128 threads + fragment reduction with the
validated ascend SIMD form of ``DESIGN.md`` v2 section 3.3:

* one-dimensional ``T.Kernel(grid)`` (no ``threads`` at kernel scope), each core
  owns one ``(batch, head)`` row slice and walks ``tpb`` row tiles through
  ``T.Pipelined`` with two UB versions (``T.annotate_buffer_versions``);
* one strided ``T.copy`` per operand into ``o_ub[0:R, 0:D_v]`` -- the destination
  region is always explicit, because an unsliced copy of a ``(R, DW)`` buffer
  from a ``(R, D_v)`` source is rejected when ``D_v < DW`` (``D_v = 64``);
* per row: ``vld -> vcvt(PART_EVEN/PART_ODD) -> vmul -> vadd -> vcadd -> vsts``,
  i.e. one 128-lane load per operand and one 64-lane tree reduction, keeping the
  fp32 products in vector registers (no materialisation in UB);
* one compressed ``T.copy`` per tile writes ``R`` fp32 row scalars back to GM,
  staged through the 32-byte-row ``d_ub`` that ``vsts`` can address;
* ``D_v < DW`` pads the row to one vector register, so *both* operands are
  cleared and the two copies only write ``[0:D_v]`` -- otherwise an
  uninitialised pad lane could contribute ``0 * (Inf/NaN bits)``;
* ``S % R != 0`` is a build-time branch: the tail segment uses the static
  ``nlast`` extent for both copies, the row loop and the write-back, so no row
  outside ``[0, S)`` is read or written.

``nvec > 1`` (``D_v > 128``) combines the segment partial sums into a single
build-time expression (``row_dot``); a run-time ``for`` carrying a vector
accumulator is not constructible on this toolchain.

Content layout:

1. ``golden_delta`` -- torch CPU reference of the operator's semantics.
2. ``_row_dot`` / ``_geometry`` / ``_delta_kernel`` -- the frozen kernel factory,
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
_R_MIN, _R_MAX = 16, 64  # row-tile rule of DESIGN.md 5.2
_CORE_TARGET = 224  # grid floor: 4 waves of 56 vector cores (DESIGN.md 5.5)
_UB_CAPACITY = 253952  # Ascend950PR_9579.ini [AICoreSpec] ub_size
_ATOL = 1e-4  # DESIGN.md 8.2: the operator output is float32
_RTOL = 1e-4
_DEFAULT_DTYPE = "float16"


# ---------------------------------------------------------------------------
# 1. Golden: torch CPU reference of the operator semantics
# ---------------------------------------------------------------------------
def golden_delta(o, do, chunk_rows=256):
    """``delta[b, h, s] = sum_d o[b, s, h, d] * do[b, s, h, d]`` in float32.

    ``o`` / ``do`` are the 4-D ``(B, S, H, D_v)`` tensors the caller holds; the
    kernel takes the ``(B*S, H, D_v)`` view, which is a host-side reshape and
    not a semantic difference.  Naive and independent of the kernel: cast both
    operands to float32 exactly, multiply element-wise, sum the last axis with
    torch's own reduction order and move the head axis in front.  ``chunk_rows``
    only bounds the fp32 temporary for the large L2 geometries.
    """
    batch, seqlen, heads, dim_v = o.shape
    out = torch.empty((batch, heads, seqlen), dtype=torch.float32)
    for s0 in range(0, seqlen, chunk_rows):
        s1 = min(s0 + chunk_rows, seqlen)
        prod = o[:, s0:s1].float() * do[:, s0:s1].float()  # (B, s, H, D_v) fp32
        out[:, :, s0:s1] = prod.sum(dim=-1).transpose(1, 2)  # (B, H, s)
    return out.contiguous()


# ---------------------------------------------------------------------------
# 2. Kernel
# ---------------------------------------------------------------------------
def _geometry(batch, heads, seqlen, dim_v):
    """Build-time tile geometry of DESIGN.md 5.2 (no run-time branch survives)."""
    row_tile = max(
        (d for d in range(_R_MIN, min(_R_MAX, seqlen) + 1) if seqlen % d == 0),
        default=_R_MAX,
    )
    dw = ((dim_v + _VL - 1) // _VL) * _VL  # padded row width, one vector register step
    nvec = dw // _VL  # register segments per row
    nfull = seqlen // row_tile  # full row tiles
    nlast = seqlen - nfull * row_tile  # tail rows (0 when row_tile divides seqlen)
    nt = nfull + bool(nlast)  # logical tiles per (batch, head)
    if nlast:
        # Static tail extents require one core per (b, h): the tail segment runs
        # outside the pipelined loop and must not be split over cores.
        nsp = 1
    else:
        nsp = next(
            (d for d in range(1, nt + 1) if nt % d == 0 and batch * heads * d >= _CORE_TARGET),
            nt,
        )
    return {
        "R": row_tile,
        "DW": dw,
        "nvec": nvec,
        "nfull": nfull,
        "nlast": nlast,
        "nt": nt,
        "nsp": nsp,
        "tpb": nfull // nsp,
        "grid": batch * heads * nsp,
        "tail": nlast,
    }


def _row_dot(o_ub, do_ub, i, nvec):
    """One build-time expression for the ``nvec``-segment dot product of row ``i``.

    ``nvec`` is a python int, so this ``for`` runs while tracing and only
    concatenates expressions.  Putting the same loop inside the prim_func body
    would lift it to an IR ``ForFrame`` and a vector value that is rebound across
    iterations is rejected (``Immutable variable ... used outside its defining
    region``), even for ``nvec == 1``.
    """
    acc = None
    for v in range(nvec):
        vo = T.simd.vld(o_ub[i, v * _VL])
        vd = T.simd.vld(do_ub[i, v * _VL])
        part = T.simd.vadd(
            T.simd.vmul(
                T.simd.vcvt(vo, "float32", part=0), T.simd.vcvt(vd, "float32", part=0)
            ),
            T.simd.vmul(
                T.simd.vcvt(vo, "float32", part=1), T.simd.vcvt(vd, "float32", part=1)
            ),
        )
        acc = part if acc is None else T.simd.vadd(acc, part)
    return acc


@functools.lru_cache(maxsize=64)
def _delta_kernel(batch, heads, dim_v, dense_seqlen, dtype):
    """Frozen factory: geometry and dtype in, compiled ``(o, do) -> delta`` out.

    Same signature as the source operator's ``_delta_kernel`` so the harness can
    keep its call site.  Every shape is a build-time constant (the source
    specialises through ``@target_cache`` the same way), so the kernel contains
    no run-time extent predicate; a new geometry is a new compilation.
    """
    if dtype not in _IN_DTYPES:
        raise ValueError(f"dtype: {dtype!r} is not one of {_IN_DTYPES}")
    if dense_seqlen <= 0:
        raise ValueError(
            "shape: dense_seqlen must be positive for MLADenseBwdOp "
            f"(got {dense_seqlen}); the packed layout is out of this kernel's scope"
        )
    if min(batch, heads, dim_v) <= 0:
        raise ValueError(f"shape: batch/heads/dim_v must be positive, got {batch}/{heads}/{dim_v}")

    geo = _geometry(batch, heads, dense_seqlen, dim_v)
    rows, dw, nvec = geo["R"], geo["DW"], geo["nvec"]
    nfull, nlast, nsp, tpb, grid = geo["nfull"], geo["nlast"], geo["nsp"], geo["tpb"], geo["grid"]
    if nsp * tpb != nfull:  # every full tile belongs to exactly one core
        raise ValueError(f"tiling: nsp={nsp} does not divide the {nfull} full tiles of this geometry")
    # o_ub/do_ub are double buffered by hand; d_ub is budgeted twice because the
    # compiler also versions staging buffers that live inside the pipeline body.
    ub_bytes = 2 * rows * dw * 2 * 2 + 2 * rows * 8 * 4
    if ub_bytes > _UB_CAPACITY:
        raise ValueError(
            f"tiling: {ub_bytes} B of UB exceeds the {_UB_CAPACITY} B of this model "
            f"(R={rows}, DW={dw}); lower the row tile"
        )
    total = batch * dense_seqlen

    @T.prim_func
    def _delta_main(
        o: T.Tensor((total, heads, dim_v), dtype),  # type: ignore
        do: T.Tensor((total, heads, dim_v), dtype),  # type: ignore
        delta: T.Tensor((batch, heads, dense_seqlen), "float32"),  # type: ignore
    ) -> None:
        with T.Kernel(grid) as bx:
            bh = bx // nsp
            sp = bx % nsp
            b = bh // heads
            h = bh % heads
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
            if tpb > 0:  # build-time: a geometry with fewer than R rows has no full tile
                for it in T.Pipelined(tpb, num_stages=2):
                    s0 = (sp * tpb + it) * rows
                    r0 = b * dense_seqlen + s0
                    T.copy(o[r0 : r0 + rows, h, :], o_ub[0:rows, 0:dim_v])
                    T.copy(do[r0 : r0 + rows, h, :], do_ub[0:rows, 0:dim_v])
                    with T.SimdVF():
                        one = T.simd.pset(32, "PAT_VL1")
                        for i in T.serial(rows):
                            T.simd.vsts(
                                d_ub[i, 0], T.simd.vcadd(_row_dot(o_ub, do_ub, i, nvec)), one, extent=1
                            )
                    T.copy(d_ub[0:rows, 0:1], delta[b, h, s0 : s0 + rows])
            if nlast:  # build-time tail segment; every extent is the static nlast
                s0 = nfull * rows
                r0 = b * dense_seqlen + s0
                T.copy(o[r0 : r0 + nlast, h, :], o_ub[0:nlast, 0:dim_v])
                T.copy(do[r0 : r0 + nlast, h, :], do_ub[0:nlast, 0:dim_v])
                with T.SimdVF():
                    one = T.simd.pset(32, "PAT_VL1")
                    for i in T.serial(nlast):
                        T.simd.vsts(
                            d_ub[i, 0], T.simd.vcadd(_row_dot(o_ub, do_ub, i, nvec)), one, extent=1
                        )
                T.copy(d_ub[0:nlast, 0:1], delta[b, h, s0 : s0 + nlast])

    return tilelang.compile(_delta_main, target="ascend", out_idx=[2])


# ---------------------------------------------------------------------------
# 3. Launch + comparison with the golden
# ---------------------------------------------------------------------------
_RESULTS = []  # (level, name, ok, detail)
_MAX_DIFF = [0.0]


def _record(level, name, ok, detail=""):
    _RESULTS.append((level, name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[{level}][{tag}] {name}{(' -- ' + detail) if detail and not ok else ''}", flush=True)


def _randn(shape, dtype_name, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=torch.float32).to(_TORCH_DTYPE[dtype_name])


def _make_inputs(batch, seqlen, heads, dim_v, dtype_name, seed, special=None):
    """Random 16-bit operands, optionally replaced by the boundary patterns."""
    o = _randn((batch, seqlen, heads, dim_v), dtype_name, seed)
    do = _randn((batch, seqlen, heads, dim_v), dtype_name, seed + 1)
    if special == "zeros":
        o.zero_()
        do.zero_()
    elif special == "nonfinite":
        # Non-finite values propagate by the same multiply-add on both sides; the
        # inf * 0 lane must become NaN in the kernel too (no padding involved:
        # this case runs with D_v == DW).
        o[0, 0, 0, 0] = float("inf")
        do[0, 0, 0, 0] = 0.0
        o[0, 1, 0, 1] = float("-inf")
        do[0, 1, 0, 1] = float("nan")
        o[0, min(2, seqlen - 1), -1, 0] = float("nan")
        do[0, min(2, seqlen - 1), -1, 0] = float("inf")
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
    batch,
    seqlen,
    heads,
    dim_v,
    dtype=_DEFAULT_DTYPE,
    seed=0,
    special=None,
    repeat=1,
    atol=_ATOL,
    rtol=_RTOL,
):
    """Build one geometry, run it on the NPU and compare ``delta`` with the golden."""
    geo = _geometry(batch, heads, seqlen, dim_v)
    diff = float("nan")
    try:
        o, do = _make_inputs(batch, seqlen, heads, dim_v, dtype, seed, special)
        gold = golden_delta(o, do)

        dev = "npu"
        o_dev = o.reshape(batch * seqlen, heads, dim_v).contiguous().to(dev)
        do_dev = do.reshape(batch * seqlen, heads, dim_v).contiguous().to(dev)

        fn = _delta_kernel(batch, heads, dim_v, seqlen, dtype)
        start = time.perf_counter()
        got = fn(o_dev, do_dev)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1e3
        if isinstance(got, (tuple, list)):
            got = got[0]
        if repeat > 1:  # out_idx allocates a fresh output, so repeats are independent
            start = time.perf_counter()
            for _ in range(repeat):
                fn(o_dev, do_dev)
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) * 1e3 / repeat
        got = got.cpu()

        # Measure first, compare second: a tolerance failure still reports the
        # actual error instead of only torch's message.
        diff, nan_mismatch = _max_diff(got, gold)
        _MAX_DIFF[0] = max(_MAX_DIFF[0], diff)
        if nan_mismatch:
            raise AssertionError(
                f"{nan_mismatch} NaN positions differ from the golden (max_diff={diff:.3e})"
            )
        torch.testing.assert_close(got, gold, atol=atol, rtol=rtol, equal_nan=True)
    except AssertionError as exc:
        _record(level, name, False, f"max_diff={diff:.3e} {str(exc)[:260]}")
        return False
    except (ValueError, RuntimeError) as exc:  # noqa: BLE001 - reported, never swallowed
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:260]}")
        return False
    _record(level, name, True)
    print(
        f"       B={batch} S={seqlen} H={heads} D_v={dim_v} {dtype} "
        f"R={geo['R']} nsp={geo['nsp']} tpb={geo['tpb']} grid={geo['grid']} "
        f"nlast={geo['nlast']} max_diff={diff:.3e} launch={elapsed:.3f} ms",
        flush=True,
    )
    return True


def _finish(level, strict):
    rows = [r for r in _RESULTS if r[0] == level]
    failed = [r for r in rows if not r[2]]
    print(
        f"[{level}] {len(rows) - len(failed)}/{len(rows)} cases passed, max_diff={_MAX_DIFF[0]:.6g}",
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
        ((1, 4, 128, 0, "float16"), "dense_seqlen=0 (packed layout)"),
        ((1, 4, 128, -8, "float16"), "negative dense_seqlen"),
        ((1, 4, 128, 64, "float32"), "float32 input"),
        ((0, 4, 128, 64, "float16"), "batch=0"),
    ):
        try:
            _delta_kernel(*args)
            problems.append(f"{why}: no error raised")
        except ValueError:
            pass
    first = _delta_kernel(1, 4, 128, 128, "float16")
    again = _delta_kernel(1, 4, 128, 128, "float16")
    if first is not again:
        problems.append("the compiled program is not cached per (geometry, dtype)")
    if problems:
        _record(level, name, False, "; ".join(problems))
        return False
    _record(level, name, True)
    print("       4 invalid contracts rejected, program cached per (geometry, dtype)", flush=True)
    return True


def run_L0():
    """Gate: main path, pad path (D_v=64, the op's own smoke geometry) and the tail-only path."""
    run_case("L0", "L0-1 fp16-main-path", 1, 128, 4, 128, "float16", seed=1)
    run_case("L0", "L0-2 bf16-main-path", 1, 128, 4, 128, "bfloat16", seed=2)
    # D_v=64 < DW=128: destination slice [0:R, 0:64] plus the two T.clear calls --
    # the only numerical evidence that the pad lanes stay zero.
    run_case("L0", "L0-3 bf16-pad-path-Dv64", 1, 256, 4, 64, "bfloat16", seed=3)
    run_case("L0", "L0-4 fp16-pad-path-Dv64", 1, 256, 4, 64, "float16", seed=4)
    run_case("L0", "L0-5 fp16-tail-only-S1", 1, 1, 1, 128, "float16", seed=5)
    _case_contract("L0", "L0-6 refuses-an-invalid-contract")
    _finish("L0", strict=True)


def run_L1():
    """Representative shapes from the workload manifest and the op-level smoke set."""
    run_case("L1", "L1-1 fp16-manifest-smoke", 1, 512, 8, 128, "float16", seed=11, repeat=3)
    run_case("L1", "L1-2 bf16-manifest-smoke", 1, 512, 8, 128, "bfloat16", seed=12, repeat=3)
    run_case("L1", "L1-3 fp16-two-batches", 2, 256, 8, 128, "float16", seed=13, repeat=3)
    run_case("L1", "L1-4 bf16-two-batches", 2, 256, 8, 128, "bfloat16", seed=14, repeat=3)
    _finish("L1", strict=True)


def run_L2():
    """Manifest workloads (large, informative): informative, non-blocking."""
    run_case("L2", "L2-1 bf16-ds-v3-4k", 1, 4096, 128, 128, "bfloat16", seed=21, repeat=2)
    run_case("L2", "L2-2 fp16-ds-v3-4k", 1, 4096, 128, 128, "float16", seed=22, repeat=2)
    run_case("L2", "L2-3 bf16-ds-v3-b2-8k", 2, 8192, 128, 128, "bfloat16", seed=23, repeat=2)
    run_case("L2", "L2-4 bf16-kimi-k2-8k", 1, 8192, 64, 128, "bfloat16", seed=24, repeat=2)
    _finish("L2", strict=False)


def run_boundary():
    """Non-divisible S, the tail segment, pad+tail, all-zero and non-finite inputs."""
    run_case("boundary", "B-1 S=200-R50", 1, 200, 8, 128, "bfloat16", seed=31)
    run_case("boundary", "B-2 S=320-five-tiles", 1, 320, 8, 128, "bfloat16", seed=32)
    # Tail segment: all three extents use the static nlast, so nothing outside
    # [0, S) may be read or written (a neighbour head/batch would show it).
    run_case("boundary", "B-3 S=65-tail-nlast1", 1, 65, 8, 128, "bfloat16", seed=33)
    # Pipeline iterations (2) plus a tail segment: the buffer-version interaction.
    run_case("boundary", "B-4 S=131-nfull2-nlast3", 1, 131, 8, 128, "bfloat16", seed=34)
    # Pad path plus tail segment combined.
    run_case("boundary", "B-5 S=65-Dv64-pad-and-tail", 1, 65, 4, 64, "bfloat16", seed=35)
    run_case("boundary", "B-6 S=1-bf16-single-row", 1, 1, 1, 128, "bfloat16", seed=36)
    # nvec = 2 (D_v = 192 > DW step): the build-time segment merge, which the
    # design could only confirm at compile time (DESIGN.md 9.1 R11).
    run_case("boundary", "B-7 bf16-Dv192-nvec2", 1, 128, 4, 192, "bfloat16", seed=37)
    run_case("boundary", "B-8 fp16-all-zero", 1, 128, 4, 128, "float16", seed=38, special="zeros")
    run_case("boundary", "B-9 fp16-non-finite", 1, 64, 4, 128, "float16", seed=39, special="nonfinite")
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
