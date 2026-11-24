# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""`splitk_reduce_kernel` on Ascend 950PR: split-K fp32 reduce + fused epilogue.

Launch 2 of ``SharedExpertMLPFwdOp`` (the companion of ``_gemm_basic_kernel``):
reduce the ``[split_k, m, n]`` float32 partial sums along the slice axis, add the
optional column bias in the float32 domain at the tail of the chain, optionally
fuse ``silu_and_mul`` (``gate * sigmoid(gate) * up``) and round exactly once into
``out_dtype or dtype``.  Migrated from the GPU TileLang source
``tileops/_kernels/gemm/dense.py::splitk_reduce_kernel``; follows the frozen
DESIGN.md / passed REVIEW.md of this directory.

* R1 -- one-dimensional ``T.Kernel(grid)`` with ``grid = m * ceil(out_n / BC)``;
  the row and the column block start are block-level scalars, so the source's
  per-element ``idx // out_n`` / ``idx % out_n`` guard disappears.
* R2 -- ``split_k`` single-descriptor column DMAs per stream (``BC * 4`` bytes
  each); the gated branch adds a second stream at the ``out_n`` offset.
* R3 -- a 64-lane chained ``vadd`` in slice-ascending order, which is bit
  identical to the source's ``((w0 + w1) + w2) + ...``; the bias joins at the
  tail of the chain, in float32, like the source.
* R4 -- gated epilogue as ``(g / (1 + exp(-g))) * u`` (DESIGN.md 1.6.1 A1,
  machine verified at 0 ulp on the fp16/bf16 output grid).
* R5 -- one rounding point: ``vcvt(ROUND_R, sat=False)`` + ``vsts(PK_B32)`` for
  16-bit outputs; a float32 output skips ``vcvt`` (fp32 -> fp32 does not exist)
  and stores with ``NORM_B32``.
* R6 -- ``split_k == 1`` / no bias / no activation are specialised away at build
  time, so the degenerate call is a ``vld -> vcvt -> vsts`` copy.

``C`` is a kernel parameter (``out_idx=[]``), not an allocated output: the caller
preallocates ``[m, out_n]`` before the mainloop, exactly like the source.

run: python splitk_reduce_kernel.py --level L0|L1|L2|boundary|all
"""

import argparse
import functools
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401 -- registers torch.npu

# ---------------------------------------------------------------------------
# Constants (DESIGN.md 4.5 / 5.2 / 5.5 / 8.2)
# ---------------------------------------------------------------------------
VL = 64  # fp32 vector register width; also the 16-bit chunk mapped by UNPK_B16/PK_B32
BLOCK_COL_CANDS = (256, 512, 1024)  # BC candidates, all multiples of VL
VECTOR_CORES = 56  # Ascend950PR_9579 vector cores (torch.npu.get_device_properties)
UB_BYTES = 253952  # Ascend950PR_9579.ini [AICoreSpec] ub_size
UB_INFLATION = 1.7  # TRAP-UB-multibuffer-inflation upper bound
ITEMSIZE = {"float16": 2, "bfloat16": 2, "float32": 4}
MANTISSA = {"float16": 10, "bfloat16": 7, "float32": 23}
MIN_EXP = {"float16": -14, "bfloat16": -126, "float32": -126}
# Op-level reference tolerances (TileOPs/workloads/numerics.py::reference_tolerance,
# rtol == atol; DESIGN.md 8.2).
TOLERANCE = {"float16": 1e-3, "bfloat16": 1.6e-2, "float32": 1e-5}
ACTIVATIONS = ("none", "silu_and_mul")


# ---------------------------------------------------------------------------
# 1. Golden (PyTorch CPU reference, from the operator semantics only)
# ---------------------------------------------------------------------------
def _torch_dtype(dtype):
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def golden_splitk_reduce(w, dtype="float16", bias=None, activation="none", out_dtype=None):
    """``c[i,j] = cast( sum_s w[s,i,j] [+ bias[j]] [| silu(gate)*up] )``.

    Deliberately free of kernel-side structure: no tiling, no UB buffers, no
    vector instructions, no block width.  The slice sum is ascending and left
    associative, the bias joins after the chain, and there is exactly one
    rounding, at the boundary (DESIGN.md 0.1 / 8.1).
    """
    sliced, m, n = w.shape
    acc = w[0].to(torch.float32).clone()
    for s in range(1, sliced):
        acc = acc + w[s].to(torch.float32)
    if bias is not None:
        acc = acc + bias.to(torch.float32).unsqueeze(0)  # column broadcast, fp32
    if activation == "silu_and_mul":
        if n % 2:
            raise ValueError("silu_and_mul requires an even output width")
        gate, up = acc[:, : n // 2], acc[:, n // 2 :]
        acc = F.silu(gate) * up
    elif activation != "none":
        raise ValueError(f"unsupported split-K epilogue: {activation}")
    return acc.to(_torch_dtype(out_dtype or dtype))


# ---------------------------------------------------------------------------
# 2. Build-time geometry (DESIGN.md 5.2) and UB budget (DESIGN.md 4.5)
# ---------------------------------------------------------------------------
def pick_block_cols(m, out_n):
    """DESIGN.md 5.2 rule: minimise ``waves * BC``, ties break to the wider block."""
    best_key, best_bc = None, BLOCK_COL_CANDS[0]
    for bc in BLOCK_COL_CANDS:
        nc = -(-out_n // bc)
        key = (-(-(m * nc) // VECTOR_CORES) * bc, -bc)
        if best_key is None or key < best_key:
            best_key, best_bc = key, bc
    return best_bc


def geometry(m, n, split_k, dtype, activation, has_bias, out_dtype):
    """Resolve the build-time geometry of one specialisation."""
    gated = activation == "silu_and_mul"
    out_n = n // 2 if gated else n
    c_dtype = out_dtype or dtype
    bc = pick_block_cols(m, out_n)
    nc = -(-out_n // bc)
    streams = 2 if gated else 1
    ub = (
        split_k * bc * 4 * streams  # g_ub (+ u_ub), fp32 partial sums
        + (ITEMSIZE[dtype] * bc * streams if has_bias else 0)  # b_ub
        + bc * ITEMSIZE[c_dtype]  # o_ub
    )
    return {
        "gated": gated,
        "out_n": out_n,
        "c_dtype": c_dtype,
        "BC": bc,
        "nc": nc,
        "grid": m * nc,
        "tail": (out_n % bc) != 0,
        "nch": bc // VL,
        "pk": "PK_B32" if ITEMSIZE[c_dtype] == 2 else "NORM_B32",
        "ub_bytes": ub,
    }


def _validate(m, n, split_k, dtype, activation, has_bias, out_dtype):
    """Build-time refusals shared by the factory and the compiled path."""
    if dtype not in ITEMSIZE:
        raise ValueError(f"dtype must be one of {sorted(ITEMSIZE)}, got {dtype!r}")
    if (out_dtype or dtype) not in ITEMSIZE:
        raise ValueError(f"out_dtype must be one of {sorted(ITEMSIZE)}, got {out_dtype!r}")
    if activation not in ACTIVATIONS:
        raise ValueError(f"unsupported split-K epilogue: {activation}")
    if activation == "silu_and_mul" and n % 2:
        raise ValueError("silu_and_mul requires an even output width")
    if split_k < 1 or m < 1 or n < 1:
        raise ValueError(f"degenerate geometry: split_k={split_k} m={m} n={n}")


# ---------------------------------------------------------------------------
# 3. Kernel
# ---------------------------------------------------------------------------
def _upconvert(addr):
    """16-bit UB lane -> fp32 vector (UNPK_B16 promote); the bias joins in fp32."""
    return T.simd.vcvt(T.simd.vld(addr, dist="UNPK_B16"), "float32", part=0)


@functools.lru_cache(maxsize=64)
def _compiled(split_k, m, n, dtype, activation, has_bias, out_dtype):
    """Compile one specialisation; the caller owns ``C`` (``out_idx=[]``)."""
    _validate(m, n, split_k, dtype, activation, has_bias, out_dtype)
    geo = geometry(m, n, split_k, dtype, activation, has_bias, out_dtype)
    gated = geo["gated"]
    out_n, c_dtype, BC = geo["out_n"], geo["c_dtype"], geo["BC"]
    nc, grid, tail, nch, pk = geo["nc"], geo["grid"], geo["tail"], geo["nch"], geo["pk"]
    if geo["ub_bytes"] * UB_INFLATION > UB_BYTES:
        raise ValueError(
            f"UB footprint {geo['ub_bytes']} B x{UB_INFLATION} exceeds {UB_BYTES} B "
            f"(split_k={split_k}, m={m}, n={n}, BC={BC}, gated={gated})"
        )

    if has_bias:

        @T.prim_func
        def main(
            W: T.Tensor((split_k, m, n), "float32"),
            BIAS: T.Tensor((n,), dtype),
            C: T.Tensor((m, out_n), c_dtype),
        ):
            with T.Kernel(grid) as bx:
                row = bx // nc
                c0 = (bx % nc) * BC
                bc = T.min(BC, out_n - c0)
                g_ub = T.alloc_shared((split_k, BC), "float32")
                u_ub = T.alloc_shared((split_k, BC), "float32") if gated else None
                b_ub = T.alloc_shared((2 if gated else 1, BC), dtype)
                o_ub = T.alloc_shared((BC,), c_dtype)
                versions = {g_ub: 1, b_ub: 1, o_ub: 1}
                if gated:
                    versions[u_ub] = 1
                T.annotate_buffer_versions(versions)

                # R2: one single-descriptor column DMA per slice (and per stream).
                for s in T.serial(split_k):
                    T.copy(W[s, row, c0 : c0 + bc], g_ub[s, :bc])
                    if gated:
                        T.copy(W[s, row, out_n + c0 : out_n + c0 + bc], u_ub[s, :bc])
                T.copy(BIAS[c0 : c0 + bc], b_ub[0, :bc])
                if gated:
                    T.copy(BIAS[out_n + c0 : out_n + c0 + bc], b_ub[1, :bc])

                # A SIMD_VF block may not touch GM: every input is staged above.
                with T.SimdVF():
                    full = T.simd.pset(32)
                    one = T.simd.vdup(T.float32(1), "float32", full)
                    for ch in T.serial(nch):
                        mask = (
                            T.simd.update_mask(T.min(VL, bc - ch * VL), 32)
                            if tail
                            else full
                        )
                        # R3: slice-ascending chained vadd, bias at the tail.
                        acc = T.simd.alloc_local((1,), "float32")
                        acc[0] = T.simd.vld(g_ub[0, ch * VL])
                        for s in T.serial(split_k - 1):
                            acc[0] = T.simd.vadd(
                                acc[0], T.simd.vld(g_ub[s + 1, ch * VL]), mask
                            )
                        acc[0] = T.simd.vadd(acc[0], _upconvert(b_ub[0, ch * VL]), mask)
                        if gated:
                            up = T.simd.alloc_local((1,), "float32")
                            up[0] = T.simd.vld(u_ub[0, ch * VL])
                            for s in T.serial(split_k - 1):
                                up[0] = T.simd.vadd(
                                    up[0], T.simd.vld(u_ub[s + 1, ch * VL]), mask
                                )
                            up[0] = T.simd.vadd(
                                up[0], _upconvert(b_ub[1, ch * VL]), mask
                            )
                            # R4 / A1: g * sigmoid(g) * u == (g / (1 + exp(-g))) * u.
                            den = T.simd.vadd(
                                one, T.simd.vexp(T.simd.vneg(acc[0], mask), mask), mask
                            )
                            y = T.simd.vmul(
                                T.simd.vdiv(acc[0], den, mask), up[0], mask
                            )
                        else:
                            y = acc[0]
                        # R5: one rounding point; fp32 output stores without vcvt.
                        q = (
                            y
                            if c_dtype == "float32"
                            else T.simd.vcvt(
                                y, c_dtype, round="ROUND_R", sat=False, part=0
                            )
                        )
                        T.simd.vsts(o_ub[ch * VL], q, mask, dist=pk)
                T.copy(o_ub[:bc], C[row, c0 : c0 + bc])

    else:

        @T.prim_func
        def main(
            W: T.Tensor((split_k, m, n), "float32"),
            C: T.Tensor((m, out_n), c_dtype),
        ):
            with T.Kernel(grid) as bx:
                row = bx // nc
                c0 = (bx % nc) * BC
                bc = T.min(BC, out_n - c0)
                g_ub = T.alloc_shared((split_k, BC), "float32")
                u_ub = T.alloc_shared((split_k, BC), "float32") if gated else None
                o_ub = T.alloc_shared((BC,), c_dtype)
                versions = {g_ub: 1, o_ub: 1}
                if gated:
                    versions[u_ub] = 1
                T.annotate_buffer_versions(versions)

                for s in T.serial(split_k):
                    T.copy(W[s, row, c0 : c0 + bc], g_ub[s, :bc])
                    if gated:
                        T.copy(W[s, row, out_n + c0 : out_n + c0 + bc], u_ub[s, :bc])

                with T.SimdVF():
                    full = T.simd.pset(32)
                    one = T.simd.vdup(T.float32(1), "float32", full)
                    for ch in T.serial(nch):
                        mask = (
                            T.simd.update_mask(T.min(VL, bc - ch * VL), 32)
                            if tail
                            else full
                        )
                        acc = T.simd.alloc_local((1,), "float32")
                        acc[0] = T.simd.vld(g_ub[0, ch * VL])
                        for s in T.serial(split_k - 1):
                            acc[0] = T.simd.vadd(
                                acc[0], T.simd.vld(g_ub[s + 1, ch * VL]), mask
                            )
                        if gated:
                            up = T.simd.alloc_local((1,), "float32")
                            up[0] = T.simd.vld(u_ub[0, ch * VL])
                            for s in T.serial(split_k - 1):
                                up[0] = T.simd.vadd(
                                    up[0], T.simd.vld(u_ub[s + 1, ch * VL]), mask
                                )
                            den = T.simd.vadd(
                                one, T.simd.vexp(T.simd.vneg(acc[0], mask), mask), mask
                            )
                            y = T.simd.vmul(
                                T.simd.vdiv(acc[0], den, mask), up[0], mask
                            )
                        else:
                            y = acc[0]
                        q = (
                            y
                            if c_dtype == "float32"
                            else T.simd.vcvt(
                                y, c_dtype, round="ROUND_R", sat=False, part=0
                            )
                        )
                        T.simd.vsts(o_ub[ch * VL], q, mask, dist=pk)
                T.copy(o_ub[:bc], C[row, c0 : c0 + bc])

    return tilelang.compile(main, target="ascend", out_idx=[])


def splitk_reduce_kernel(
    split_k, m, n, dtype="float16", activation="none", has_bias=False, out_dtype=None
):
    """Harness-shaped factory: same outer signature as the GPU source.

    The inner ``elems_per_cta`` is kept only for the wrapper call shape
    (``splitk_reduce_kernel(...)(_SPLIT_K_ELEMS_PER_CTA)``); on this backend the
    block width comes from the build-time ``BC`` rule, so it is a no-op
    (DESIGN.md 10.3).
    """

    def _func(elems_per_cta=1024):
        _ = elems_per_cta  # no-op on the NPU side (DESIGN.md 10.3)
        return _compiled(split_k, m, n, dtype, activation, has_bias, out_dtype)

    return _func


# ---------------------------------------------------------------------------
# 4. Precision comparing
# ---------------------------------------------------------------------------
_RESULTS = []  # (level, name, ok, detail)
_DIFFS = []  # (name, max_abs_diff, max_ulp)


def _record(level, name, ok, detail=""):
    _RESULTS.append((level, name, ok, detail))


def _max_ulp(got, want, c_dtype):
    """Largest distance in the *output dtype* grid (observed, never asserted).

    Measured on the grid of the rounded output and scaled from that dtype's
    minimum normal exponent, so the subnormal band cannot inflate the metric
    (DESIGN.md 8.2 / Stage 1 retrospective).
    """
    g, w = got.float(), want.float()
    keep = torch.isfinite(g) & torch.isfinite(w) & (w != 0)
    if not bool(keep.any()):
        return 0.0
    g, w = g[keep], w[keep]
    exponent = torch.floor(torch.log2(w.abs())).clamp(min=MIN_EXP[c_dtype])
    ulp = torch.pow(torch.full_like(exponent, 2.0), exponent - MANTISSA[c_dtype])
    return float(((g - w).abs() / ulp).max())


def _check(got, want, tag, c_dtype):
    """Elementwise tolerance + NaN-position + infinity-sign gate.

    Returns ``(max_abs_diff, max_ulp)`` over the finite positions.
    """
    assert got.shape == want.shape, f"[{tag}] shape {tuple(got.shape)} != {tuple(want.shape)}"
    assert got.dtype == want.dtype, f"[{tag}] dtype {got.dtype} != {want.dtype}"
    g, w = got.double(), want.double()
    nan_bad = int(((g != g) != (w != w)).sum())
    if nan_bad:
        raise AssertionError(f"[{tag}] {nan_bad} NaN positions differ")
    inf_bad = int((torch.isinf(g) ^ torch.isinf(w)).sum() + (
        torch.isinf(g) & torch.isinf(w) & (torch.sign(g) != torch.sign(w))
    ).sum())
    if inf_bad:
        raise AssertionError(f"[{tag}] {inf_bad} infinity positions/signs differ")
    finite = torch.isfinite(g) & torch.isfinite(w)
    diff = torch.where(finite, (g - w).abs(), torch.zeros_like(g))
    atol = rtol = TOLERANCE[c_dtype]
    bad = finite & (diff > atol + rtol * w.abs())
    n_bad = int(bad.sum())
    max_diff = float(diff.max()) if diff.numel() else 0.0
    if n_bad:
        first = tuple(bad.nonzero()[0].tolist())
        raise AssertionError(
            f"[{tag}] {n_bad}/{diff.numel()} elements out of tolerance "
            f"(atol={atol:g} rtol={rtol:g}); max_diff={max_diff:.3e}; "
            f"first_bad={first} got={float(g[first]):.6e} want={float(w[first]):.6e}"
        )
    return max_diff, _max_ulp(got, want, c_dtype)


def _make_inputs(split_k, m, n, dtype, has_bias, seed, special=None):
    """Partial sums as the upstream GEMM writes them: fp32, ``[split_k, m, n]``."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn((split_k, m, n), generator=gen, dtype=torch.float32) * 2.0
    bias = torch.randn(n, generator=gen, dtype=torch.float32) if has_bias else None
    if special == "zeros":
        w.zero_()
        if bias is not None:
            bias.zero_()
    elif special == "nonfinite":
        w[0, 0, 0] = float("inf")
        w[0, min(1, m - 1), min(1, n - 1)] = float("nan")
        w[0, min(2, m - 1), min(2, n - 1)] = float("-inf")
        if split_k > 1:
            w[1, min(2, m - 1), min(2, n - 1)] = float("inf")  # -inf + inf -> NaN
        w[0, min(3, m - 1), min(3, n - 1)] = -1e-40  # fp32 subnormal
        w[0, min(4, m - 1), min(4, n - 1)] = 0.0
        if split_k > 1:
            w[1, min(4, m - 1), min(4, n - 1)] = -0.0
    elif special == "subnormal":
        w.mul_(1e-42)  # every partial sum is an fp32 subnormal
    elif special == "overflow":
        w.fill_(1.0e5)  # finite fp32, outside the 16-bit range
    elif special is not None:
        raise ValueError(f"special: unknown pattern {special!r}")
    if bias is not None:
        bias = bias.to(_torch_dtype(dtype))
    return w, bias


def _launch_ms(fn, args, reps=5):
    """Wall-clock launch time of one kernel call (informational, Stage 4 baseline)."""
    fn(*args)
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    torch.npu.synchronize()
    return (time.perf_counter() - started) / reps * 1e3


def run_case(
    level,
    name,
    split_k,
    m,
    n,
    dtype,
    activation="none",
    has_bias=False,
    out_dtype=None,
    seed=0,
    special=None,
    repeat=1,
    timed=False,
):
    """Compile, launch on the NPU, and compare against the CPU golden."""
    c_dtype = out_dtype or dtype
    geo = geometry(m, n, split_k, dtype, activation, has_bias, out_dtype)
    elapsed = float("nan")
    try:
        w, bias = _make_inputs(split_k, m, n, dtype, has_bias, seed, special)
        want = golden_splitk_reduce(w, dtype, bias, activation, out_dtype)
        fn = splitk_reduce_kernel(
            split_k, m, n, dtype, activation, has_bias, out_dtype
        )()
        args = [w.to("npu")]
        if has_bias:
            args.append(bias.to("npu"))
        out = torch.empty(
            (m, geo["out_n"]), dtype=_torch_dtype(c_dtype), device="npu"
        )
        args.append(out)  # C is a parameter, preallocated by the caller
        fn(*args)
        torch.npu.synchronize()
        first = out.cpu().clone()
        for _ in range(repeat - 1):  # determinism gate: bitwise identical launches
            fn(*args)
        torch.npu.synchronize()
        if repeat > 1 and not torch.equal(out.cpu(), first):
            raise AssertionError(f"[{name}] not deterministic over {repeat} launches")
        max_diff, ulp = _check(first, want, name, c_dtype)
        if timed:
            elapsed = _launch_ms(fn, args)
    except AssertionError as exc:
        _record(level, name, False, str(exc)[:400])
        return False
    except (ValueError, RuntimeError) as exc:  # reported, never swallowed
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:400]}")
        return False
    _record(level, name, True)
    _DIFFS.append((name, max_diff, ulp))
    print(
        f"[{name}] PASS split_k={split_k} m={m} n={n} out_n={geo['out_n']} {dtype}"
        f"->{c_dtype} act={activation} bias={int(has_bias)} BC={geo['BC']} "
        f"grid={geo['grid']} nch={geo['nch']} tail={int(geo['tail'])} "
        f"max_diff={max_diff:.3e} ulp={ulp:.1f}"
        + (f" launch={elapsed:.3f}ms" if timed else ""),
        flush=True,
    )
    return True


def _finish(level, strict):
    rows = [r for r in _RESULTS if r[0] == level]
    failed = [r for r in rows if not r[2]]
    print(
        f"[{level}] {len(rows) - len(failed)}/{len(rows)} cases passed", flush=True
    )
    for row in failed:
        tag = "FAIL" if strict else "WARN"
        print(f"[{level}][{tag}] {row[1]}: {row[3]}", flush=True)
    if failed and strict:
        raise AssertionError(f"[{level}] {len(failed)} blocking case(s) failed")


def _warn_case(level, name, *args, **kwargs):
    """Non-blocking (L2 / boundary) case: failures are recorded, never raised."""
    try:
        return run_case(level, name, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - non-blocking by contract
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:400]}")
        print(f"[{level}][WARN] {name}: {type(exc).__name__}: {str(exc)[:200]}")
        return False


def _expect_value_error(fn, tag, level="boundary"):
    """A degraded contract must fail loudly at build time, never silently."""
    try:
        fn()
    except ValueError as exc:
        _record(level, tag, True)
        print(f"[{tag}] PASS ValueError: {str(exc).splitlines()[0][:140]}", flush=True)
        return
    _record(level, tag, False, "expected a ValueError")
    print(f"[{tag}][WARN] expected a ValueError", flush=True)


# ---------------------------------------------------------------------------
# 5. Layered cases
# ---------------------------------------------------------------------------
def run_L0():
    """Gate: split_k = 1/2/4, every manifest geometry, bias on/off, gated on/off,
    a non-64-multiple width, the split_k = 1 degradation and 3-launch determinism.
    """
    # L0-1..L0-4: the four manifest workloads (DESIGN.md 5.6).
    run_case("L0", "L0-1-ds-gate-up-bf16", 4, 32, 4096, "bfloat16", "silu_and_mul",
             repeat=3, timed=True)
    run_case("L0", "L0-2-ds-down-bf16", 4, 32, 7168, "bfloat16", repeat=3, timed=True)
    run_case("L0", "L0-3-glm-gate-up-bf16", 4, 32, 3072, "bfloat16", "silu_and_mul",
             timed=True)
    run_case("L0", "L0-4-glm-down-bf16", 4, 32, 5120, "bfloat16")
    # L0-5: the fp16 output path (vcvt<half>(ROUND_R) + PK_B32).
    run_case("L0", "L0-5-ds-gate-up-fp16", 4, 32, 4096, "float16", "silu_and_mul")
    # L0-6: split_k = 2 on a manifest geometry.
    run_case("L0", "L0-6-glm-gate-up-splitk2", 2, 32, 3072, "bfloat16",
             "silu_and_mul", repeat=3)
    # L0-7: split_k = 1 degenerate gated path (no vadd at all).
    run_case("L0", "L0-7-splitk1-gated", 1, 4, 4096, "bfloat16", "silu_and_mul",
             repeat=3)
    # L0-8: split_k = 1 + bias: copy + fp32 bias promote, no activation.
    run_case("L0", "L0-8-splitk1-bias", 1, 2, 1024, "float16", has_bias=True)
    # L0-9: partial column block and tail lanes (2000 % 256 = 208, 208 % 64 = 16).
    run_case("L0", "L0-9-tail-block-lane", 3, 3, 2000, "bfloat16")
    # L0-10: gated with a partial block (out_n = 500 -> tail) and two trailing lanes.
    run_case("L0", "L0-10-gated-tail", 2, 3, 1000, "bfloat16", "silu_and_mul")
    # L0-11: out_dtype = float32 (no vcvt, NORM_B32) with the gated + bias chain.
    run_case("L0", "L0-11-fp32-out-gated-bias", 2, 8, 512, "float16",
             "silu_and_mul", has_bias=True, out_dtype="float32")
    # L0-12: +-inf / NaN / fp32 subnormal / +-0 must follow IEEE through the chain.
    run_case("L0", "L0-12-nonfinite", 2, 4, 256, "bfloat16", special="nonfinite")
    _finish("L0", strict=True)


def run_L1():
    """Representative coverage beyond the gate: bias on every branch, split_k = 3,
    out_n < BC, m = 1, wider m, and longer determinism runs."""
    # L1-1/L1-2: the bias branch on both the gated and the plain manifest shapes.
    run_case("L1", "L1-1-ds-down-bias", 4, 32, 7168, "bfloat16", has_bias=True)
    run_case("L1", "L1-2-glm-gate-up-bias", 4, 32, 3072, "bfloat16",
             "silu_and_mul", has_bias=True, repeat=3)
    # L1-3: fp16 manifest down projection.
    run_case("L1", "L1-3-ds-down-fp16", 4, 32, 7168, "float16")
    # L1-4: the m = 1 decode fallback (split_k = 1, full width 18432).
    run_case("L1", "L1-4-m1-kimi-down", 1, 1, 18432, "bfloat16", repeat=3, timed=True)
    # L1-5: split_k = 3 (an odd chain length).
    run_case("L1", "L1-5-splitk3", 3, 8, 1536, "bfloat16")
    # L1-6: out_n below one block (gated: out_n = 32 < BC = 256).
    run_case("L1", "L1-6-outn-below-bc", 2, 4, 64, "bfloat16", "silu_and_mul")
    # L1-7: wider m (2048 blocks over 56 vector cores).
    run_case("L1", "L1-7-m64", 4, 64, 7168, "bfloat16")
    # L1-8: fp32 output with the plain chain.
    run_case("L1", "L1-8-fp32-out-plain", 2, 16, 512, "bfloat16", out_dtype="float32")
    # L1-9: all-zero partial sums (with bias only) must be exact.
    run_case("L1", "L1-9-zeros-bias", 2, 8, 1024, "float16", has_bias=True,
             special="zeros")
    # L1-10: fp16 gated + bias + split_k = 2, three identical launches.
    run_case("L1", "L1-10-fp16-gated-bias-splitk2", 2, 4, 512, "float16",
             "silu_and_mul", has_bias=True, repeat=3)
    _finish("L1", strict=True)


def run_L2():
    """Larger scale than the gate; recorded, non-blocking."""
    _warn_case("L2", "L2-1-m512-ds-down", 4, 512, 7168, "bfloat16", repeat=2)
    _warn_case("L2", "L2-2-m512-ds-gate-up", 4, 512, 4096, "bfloat16",
               "silu_and_mul", repeat=2)
    _warn_case("L2", "L2-3-m1024-glm-down", 1, 1024, 5120, "bfloat16", repeat=2)
    _warn_case("L2", "L2-4-m256-kimi-down", 4, 256, 18432, "bfloat16", repeat=2)
    _finish("L2", strict=False)


def run_boundary():
    """Boundary: single lane, 16-bit overflow, subnormals, zeros and the
    build-time refusals; non-blocking by contract."""
    # B-1/B-2: one output element (out_n = 1 < one vector register).
    _warn_case("boundary", "B-1-m1-n1", 2, 1, 1, "bfloat16")
    _warn_case("boundary", "B-2-gated-n2", 2, 1, 2, "bfloat16", "silu_and_mul")
    # B-3: finite fp32 input that overflows the fp16 output (sat=False keeps inf).
    _warn_case("boundary", "B-3-fp16-overflow", 1, 4, 256, "float16",
               special="overflow")
    # B-4: every partial sum is an fp32 subnormal, gated (vdiv FTZ probe).
    _warn_case("boundary", "B-4-subnormal-gated", 2, 4, 256, "bfloat16",
               "silu_and_mul", special="subnormal")
    # B-5: all-zero gated input.
    _warn_case("boundary", "B-5-zeros-gated", 2, 4, 256, "bfloat16",
               "silu_and_mul", special="zeros")
    # B-6: non-finite propagation through silu.
    _warn_case("boundary", "B-6-nonfinite-gated", 2, 4, 256, "bfloat16",
               "silu_and_mul", special="nonfinite")
    # B-7/B-8: a prime, non-divisible width and the split_k = 1 fp32 gated path.
    _warn_case("boundary", "B-7-n2003-splitk3", 3, 3, 2003, "bfloat16")
    _warn_case("boundary", "B-8-splitk1-fp32-gated", 1, 2, 512, "bfloat16",
               "silu_and_mul", out_dtype="float32")
    # B-9..B-11: the build-time refusals (odd gated width, unknown activation,
    # degenerate geometry).
    _expect_value_error(
        lambda: splitk_reduce_kernel(2, 4, 33, "bfloat16", "silu_and_mul")(),
        "B-9-err-odd-gated-width",
    )
    _expect_value_error(
        lambda: splitk_reduce_kernel(2, 4, 32, "bfloat16", "gelu")(),
        "B-10-err-unknown-activation",
    )
    _expect_value_error(
        lambda: splitk_reduce_kernel(0, 4, 32, "bfloat16")(),
        "B-11-err-degenerate-splitk",
    )
    _finish("boundary", strict=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"]
    )
    args, _ = parser.parse_known_args()

    started = time.perf_counter()
    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L1()
    elif args.level == "L2":
        run_L2()
    elif args.level == "boundary":
        run_boundary()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()

    worst = max((d for _, d, _ in _DIFFS), default=0.0)
    worst_ulp = max((u for _, _, u in _DIFFS), default=0.0)
    blocking = [r for r in _RESULTS if not r[2] and r[0] in ("L0", "L1")]
    print(
        f"cases={len(_DIFFS)} blocking_failures={len(blocking)} "
        f"max_diff={worst:.3e} max_ulp={worst_ulp:.1f} "
        f"total={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    if blocking:
        print(f"[PRECISION_FAIL] {len(blocking)} blocking case(s) failed", flush=True)
    else:
        print("\033[92mAll check passed!\033[0m")
        print("[PRECISION_PASS]", flush=True)


if __name__ == "__main__":
    main()
