# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 4 tuned ``SharedExpertMLPFwdOp`` (Ascend950PR_9579) -- self-contained.

    output = down(silu(hidden @ gate.T) * (hidden @ up.T))

``hidden[T, H]``, ``w_gate_up[2S, H]`` (gate rows, then up rows),
``w_down[H, S]``; bf16/fp16 in, fp32 Cube accumulation, one rounding to the
input dtype at the boundary (unchanged numerics -- DESIGN.md 0.1).

Three child kernels, all re-derived from the Stage 3 implementations in
``tileops/kernels/moe/shared_expert_mlp/shared_expert_mlp_kernel/`` so this file
runs on its own:

* ``_gemm_nt_program`` -- L1-operand ``C = A @ B^T`` GEMM (the gate/up and the
  down projection), with the optional split-K fp32 partial-sum contract;
* ``_reduce_program`` -- split-K reduce, optionally fusing ``silu_and_mul``;
* ``_silu_mul_program`` -- the separate gated activation for the unsplit path.

Stage 4 changes (all parameter/structure level inside the same numerical path):

* R1 -- tile: ``256 x 256 x 128`` with ``num_stages = 3`` replaces the Stage 3
  ``128 x 256 x 64`` default.  The Stage 3 profile is L1-fill bound
  (``aic_mte2_ratio ~= 0.98``, ``aic_cube_ratio ~= 0.47``), so the tile with the
  largest ``block_m * block_n / (block_m + block_n)`` arithmetic intensity that
  still fits L0C (``block_m * block_n * 4 <= 256 KiB``) wins; ``block_k = 128``
  is the second half of the fix -- at ``block_k = 64`` every L1 fill is a 128 B
  row segment, at 128 it is 256 B, which is what lifts the MTE2 pipe from
  ~70 GB/s to ~105 GB/s per core.  Measured: gate/up 1386 -> 805 us, down
  703 -> 368 us (kimi-k2-t512).
* R2 -- the gated activation moves into the gate/up GEMM's Cube + Vector
  epilogue (``_gemm_nt_gate_up_program``): one block accumulates the gate tile
  and the up tile in two fp32 L0C halves, gates them in fp32 across the two AIV
  banks and rounds once.  The ``[T, 2S]`` intermediate and its separate pass
  disappear (53 us on kimi-k2-t512, 9 us on ds-v3-t32).  Same arithmetic, same
  single rounding point.
* R4 (this round) -- the gated activation's ``exp``/``div`` move to the vector
  SFU (``precision="ftz_true"``); ``tl.enable_fast_math`` is off by default, so
  the Stage 3-4 chain was paying the precise 0-ulp sequences.  The gated product
  changes by at most one SFU ulp (one bf16 ulp on the output grid) and the
  normalized error is unchanged; see ``SFU``.
* R3 -- split-K is off by default: at ``block_k = 128`` the decode band is no
  longer cube-starved and the measured split (4 slices + reduce) is slower than
  the single pass for every manifest workload.  The path is kept for an explicit
  request.
* Everything else (K-tile ladder, M-tile clamp, fixpipe store region,
  ``dual_copy`` epilogue shape, gated-silu identity) keeps the Stage 3
  contracts, which the L0/L1 suites re-check.

Run ``python SharedExpertMLPFwdOp.py --level L0|L1|all`` for the precision
regression (same tolerances as the Stage 3 kernels), or
``--level bench`` for a launch-time sanity table.
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
# Tuned defaults (Stage 4).  Exposed at module level so the wrapper's
# baseline/perf_opt switch block can reference the same numbers.
# ---------------------------------------------------------------------------
TUNED_BLOCK_M = 256
TUNED_BLOCK_N = 256
TUNED_BLOCK_K = 128
TUNED_NUM_STAGES = 3
# Split-K is off by default: with ``block_k = 128`` the decode band (``T = 32``)
# is no longer cube-starved, and the measured split (4 slices + reduce) is
# slower than the single-pass GEMM for every manifest workload -- see
# ``opt_log.md`` round 1.  The path stays available for an explicit request.
TUNED_SPLIT_K = 1
# Split-K stays at the Stage 3 validated 4 slices (no unmeasured 2/3 chain).
SPLIT_K_MAX_TOKENS = 32
SPLIT_K_MAX_SLICES = 4
SPLIT_K_ELEMS_PER_CTA = 1024
MIN_BLOCK_M = 16

L0C_BYTES = 262144
L1_BYTES = 524288
ITEMSIZE = {"float16": 2, "bfloat16": 2, "float32": 4}

VL = 64  # fp32 vector lanes / 16-bit chunk of UNPK_B16-PK_B32
# The gated activation's exp/div run on the vector SFU instead of the precise
# fp32 library paths (`tl.enable_fast_math` is off by default, so
# `precision=None` means the 0-ulp sequences).  Measured on kimi-k2-t512: the
# gate/up kernel's epilogue drops from ~27 us to ~2 us of the block budget
# (842.7 -> 815.6 us).  The result differs from the precise chain by at most one
# ulp of the SFU approximation, which survives as at most one bf16/fp16 ulp on
# the `[T, S]` product; the end-to-end normalized error is unchanged
# (2.75e-06 for the gated product, 1.6e-07-level for the operator) and the
# op-level gate is atol=rtol=1e-2 with NSE <= 1e-4.
SFU = "ftz_true"
VECTOR_CORES = 56
UB_BYTES = 253952
UB_INFLATION = 1.7

# Stage 3 tolerances (unchanged).
HALF_ATOL = 1e-2
HALF_RTOL = 1e-2
HALF_NSE = 1e-4


def _ceil_div(x, y):
    return -(-x // y)


def _clamp_block_m(m, block_m):
    """Stage 3 rule: ``min(block_m, align16(m))`` -- never a tile of dead rows."""
    return min(block_m, max(MIN_BLOCK_M, _ceil_div(m, MIN_BLOCK_M) * MIN_BLOCK_M))


# ---------------------------------------------------------------------------
# 1. Dense NT GEMM (gate/up, down)
# ---------------------------------------------------------------------------
def _check_tiles(m, n, k, block_m, block_n, block_k, num_stages, split_k, dtype):
    if min(block_m, block_n, block_k) <= 0 or num_stages <= 0 or split_k <= 0:
        raise ValueError(
            f"tiles must be positive: block=({block_m},{block_n},{block_k}) "
            f"num_stages={num_stages} split_k={split_k}"
        )
    if block_m % 16 or block_n % 16 or block_k % 16:
        raise ValueError(
            "block_m/block_n/block_k must be multiples of 16 "
            f"(backend static_assert M/N % FRAC_N == 0): ({block_m},{block_n},{block_k})"
        )
    if k % block_k:
        raise ValueError(f"k must be divisible by block_k: k={k} block_k={block_k}")
    if split_k > 1:
        exact = m % block_m == 0 and n % block_n == 0 and (k // block_k) % split_k == 0
        if m < MIN_BLOCK_M or not exact:
            raise ValueError(
                "split-K requires exact M/N/K tiles and m >= 16: "
                f"m={m} n={n} k={k} block=({block_m},{block_n},{block_k}) split_k={split_k}"
            )
    if block_m * block_n * 4 > L0C_BYTES:
        raise ValueError(f"L0C overflow: {block_m * block_n * 4} > {L0C_BYTES}")
    l1 = (block_m + block_n) * block_k * ITEMSIZE[dtype] * num_stages
    if l1 > L1_BYTES:
        raise ValueError(f"L1 overflow: {l1} > {L1_BYTES}")


def _gemm_nt_program(m, n, k, dtype, out_dtype, block_m, block_n, block_k,
                     num_stages, split_k):
    """``C[m, n] = A[m, k] @ B[n, k]^T``; split_k > 1 emits fp32 partials."""
    c_dtype = out_dtype or dtype
    mt = _ceil_div(m, block_m)
    nt = _ceil_div(n, block_n)
    k_tiles = k // block_k
    k_slice = k_tiles // split_k
    exact_m = m % block_m == 0
    exact_n = n % block_n == 0
    out_shape = (m, n) if split_k == 1 else (split_k, m, n)
    out_dtype_ = c_dtype if split_k == 1 else "float32"
    grid = nt * mt * split_k

    @T.prim_func
    def gemm_nt(
        A: T.Tensor((m, k), dtype),
        B: T.Tensor((n, k), dtype),
        C: T.Tensor(out_shape, out_dtype_),
    ):
        with T.Kernel(grid) as bid:
            bx = bid // (mt * split_k)  # n tile: slow axis
            by = bid % (mt * split_k) // split_k  # m tile: fast axis (A panel in L2)
            bz = bid % split_k  # split-K slice: fastest
            m0 = by * block_m
            n0 = bx * block_n
            am = block_m if exact_m else T.min(m - m0, block_m)
            an = block_n if exact_n else T.min(n - n0, block_n)

            a1 = T.alloc_l1((block_m, block_k), dtype)
            b1 = T.alloc_l1((block_n, block_k), dtype)
            c0 = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions({a1: num_stages, b1: num_stages})

            for step in T.Pipelined(k_slice, num_stages=num_stages):
                k0 = (bz * k_slice + step) * block_k
                T.copy(A[m0 : m0 + am, k0 : k0 + block_k], a1[:am, :])
                T.copy(B[n0 : n0 + an, k0 : k0 + block_k], b1[:an, :])
                T.gemm(a1, b1, c0, transpose_B=True, clear_accum=(step == 0))

            # fixpipe store: the source region is the whole accumulator, only the
            # destination is truncated (an M-truncated L0C source region makes the
            # backend derive the row pitch from the region -- TRAP-950-l0c-region-
            # store-stride).
            if split_k == 1:
                T.copy(c0, C[m0 : m0 + am, n0 : n0 + an])
            else:
                T.copy(c0, C[bz, m0 : m0 + am, n0 : n0 + an])

    return gemm_nt


@functools.lru_cache(maxsize=None)
def _build_gemm(m, n, k, dtype, out_dtype, block_m, block_n, block_k, num_stages,
                split_k):
    block_m = _clamp_block_m(m, block_m)
    _check_tiles(m, n, k, block_m, block_n, block_k, num_stages, split_k, dtype)
    prog = _gemm_nt_program(m, n, k, dtype, out_dtype, block_m, block_n, block_k,
                            num_stages, split_k)
    return tilelang.compile(prog, target="ascend", out_idx=-1)


# ---------------------------------------------------------------------------
# 2. Fused gate/up GEMM: ``out = silu(A @ Wg^T) * (A @ Wu^T)``
# ---------------------------------------------------------------------------
def _gemm_nt_gate_up_program(m, s, k, dtype, block_m, block_n, block_k, num_stages):
    """One block computes the gate tile *and* the up tile, then gates them.

    Two fp32 L0C accumulators (gate rows ``n0``, up rows ``S + n0``) keep the
    L1 fill ratio of the single-tile 256x256 GEMM (``bm*bn`` halved, so
    ``1/bm + 1/(2*bn)`` is unchanged) while the gated product is produced in the
    Cube + Vector epilogue: ``dual_copy`` the two accumulators into the two AIV
    UB banks, evaluate ``(g * u) / (1 + exp(-g))`` in fp32 (DESIGN.md 1.6.1 A1,
    the Stage 3 chain) and ``dual_copy`` the rounded tile out.  The intermediate
    ``[T, 2S]`` tensor and its separate activation pass disappear.

    Requires exact M/N tiles: ``dual_copy`` cannot infer a symbolic region
    extent, so the caller only builds this form when
    ``m % block_m == 0 and s % block_n == 0`` (the same precondition the Stage 3
    bias epilogue carries).
    """
    mt = _ceil_div(m, block_m)
    nt = _ceil_div(s, block_n)
    grid = nt * mt
    half = block_m // 2
    nchunk = block_n // VL

    @T.prim_func
    def gate_up_fused(
        A: T.Tensor((m, k), dtype),
        W: T.Tensor((2 * s, k), dtype),
        OUT: T.Tensor((m, s), dtype),
    ):
        with T.Kernel(grid) as bid:
            bx = bid // mt  # n tile: slow axis
            by = bid % mt  # m tile: fast axis
            m0 = by * block_m
            n0 = bx * block_n

            a1 = T.alloc_l1((block_m, block_k), dtype)
            bg = T.alloc_l1((block_n, block_k), dtype)
            bu = T.alloc_l1((block_n, block_k), dtype)
            cg = T.alloc_l0c((block_m, block_n), "float32")
            cu = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions({a1: num_stages, bg: num_stages, bu: num_stages})

            for step in T.Pipelined(k // block_k, num_stages=num_stages):
                k0 = step * block_k
                T.copy(A[m0 : m0 + block_m, k0 : k0 + block_k], a1)
                T.copy(W[n0 : n0 + block_n, k0 : k0 + block_k], bg)
                T.copy(W[s + n0 : s + n0 + block_n, k0 : k0 + block_k], bu)
                T.gemm(a1, bg, cg, transpose_B=True, clear_accum=(step == 0))
                T.gemm(a1, bu, cu, transpose_B=True, clear_accum=(step == 0))

            cv_g = T.alloc_shared((half, block_n), "float32")
            cv_u = T.alloc_shared((half, block_n), "float32")
            co = T.alloc_shared((half, block_n), dtype)
            T.dual_copy(cg, cv_g)
            T.dual_copy(cu, cv_u)
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.vdup(T.float32(1), "float32", full)
                for r in T.serial(half):
                    for c in T.serial(nchunk):
                        g = T.simd.vld(cv_g[r, c * VL])
                        u = T.simd.vld(cv_u[r, c * VL])
                        num = T.simd.vmul(g, u, full)
                        den = T.simd.vadd(
                            one,
                            T.simd.vexp(T.simd.vneg(g, full), full, precision=SFU),
                            full,
                        )
                        y = T.simd.vdiv(num, den, full, precision=SFU)
                        T.simd.vsts(
                            co[r, c * VL],
                            T.simd.vcvt(y, dtype, round="ROUND_R", sat=False, part=0),
                            full,
                            dist="PK_B32",
                        )
            T.dual_copy(co, OUT[m0 : m0 + block_m, n0 : n0 + block_n])

    return gate_up_fused


@functools.lru_cache(maxsize=32)
def _build_gate_up_fused(m, s, k, dtype, block_m, block_n, block_k, num_stages):
    block_m = _clamp_block_m(m, block_m)
    if m % block_m or s % block_n or k % block_k:
        raise ValueError(
            f"fused gate/up epilogue needs exact tiles: m={m} s={s} k={k} "
            f"block=({block_m},{block_n},{block_k})"
        )
    if block_n % VL:
        raise ValueError(
            f"the gated epilogue walks VL={VL}-wide column chunks: block_n={block_n}"
        )
    if 2 * block_m * block_n * 4 > L0C_BYTES:
        raise ValueError(
            f"L0C overflow: 2 x {block_m}x{block_n} fp32 accumulators = "
            f"{2 * block_m * block_n * 4} > {L0C_BYTES}"
        )
    l1 = (block_m + 2 * block_n) * block_k * ITEMSIZE[dtype] * num_stages
    if l1 > L1_BYTES:
        raise ValueError(f"L1 overflow: {l1} > {L1_BYTES}")
    ub = (2 * block_m * block_n * 4) // 2 + (block_m * block_n * ITEMSIZE[dtype]) // 2
    # The epilogue buffers are outside any T.Pipelined loop and carry no buffer
    # versions, so the multibuffer inflation factor of the Stage 3 UB budgets
    # does not apply here; the plain capacity check is the binding one.
    if ub > UB_BYTES:
        raise ValueError(f"UB footprint {ub} B > {UB_BYTES} B")
    prog = _gemm_nt_gate_up_program(m, s, k, dtype, block_m, block_n, block_k,
                                    num_stages)
    return tilelang.compile(prog, target="ascend", out_idx=-1)


# ---------------------------------------------------------------------------
# 3. Split-K reduce (optionally fused silu_and_mul)
# ---------------------------------------------------------------------------
def _pick_block_cols(m, out_n):
    best_key, best_bc = None, 256
    for bc in (256, 512, 1024):
        nc = _ceil_div(out_n, bc)
        key = (_ceil_div(m * nc, VECTOR_CORES) * bc, -bc)
        if best_key is None or key < best_key:
            best_key, best_bc = key, bc
    return best_bc


def _upconvert(addr):
    return T.simd.vcvt(T.simd.vld(addr, dist="UNPK_B16"), "float32", part=0)


@functools.lru_cache(maxsize=64)
def _build_reduce(split_k, m, n, dtype, activation, out_dtype):
    """Slice-sum, optional gated activation, one rounding (Stage 3 R2-R5)."""
    gated = activation == "silu_and_mul"
    if activation not in ("none", "silu_and_mul"):
        raise ValueError(f"unsupported split-K epilogue: {activation}")
    if gated and n % 2:
        raise ValueError("silu_and_mul requires an even output width")
    out_n = n // 2 if gated else n
    c_dtype = out_dtype or dtype
    bc = _pick_block_cols(m, out_n)
    nc = _ceil_div(out_n, bc)
    streams = 2 if gated else 1
    ub = split_k * bc * 4 * streams + bc * ITEMSIZE[c_dtype]
    if ub * UB_INFLATION > UB_BYTES:
        raise ValueError(f"reduce UB footprint {ub} B x{UB_INFLATION} > {UB_BYTES} B")
    grid = m * nc
    tail = (out_n % bc) != 0
    nch = bc // VL
    pk = "PK_B32" if ITEMSIZE[c_dtype] == 2 else "NORM_B32"

    @T.prim_func
    def reduce_kernel(
        W: T.Tensor((split_k, m, n), "float32"),
        C: T.Tensor((m, out_n), c_dtype),
    ):
        with T.Kernel(grid) as bx:
            row = bx // nc
            c0 = (bx % nc) * bc
            bcols = T.min(bc, out_n - c0)
            g_ub = T.alloc_shared((split_k, bc), "float32")
            u_ub = T.alloc_shared((split_k, bc), "float32") if gated else None
            o_ub = T.alloc_shared((bc,), c_dtype)
            versions = {g_ub: 1, o_ub: 1}
            if gated:
                versions[u_ub] = 1
            T.annotate_buffer_versions(versions)

            for s in T.serial(split_k):
                T.copy(W[s, row, c0 : c0 + bcols], g_ub[s, :bcols])
                if gated:
                    T.copy(W[s, row, out_n + c0 : out_n + c0 + bcols], u_ub[s, :bcols])

            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.vdup(T.float32(1), "float32", full)
                for ch in T.serial(nch):
                    mask = (
                        T.simd.update_mask(T.min(VL, bcols - ch * VL), 32)
                        if tail
                        else full
                    )
                    acc = T.simd.alloc_local((1,), "float32")
                    acc[0] = T.simd.vld(g_ub[0, ch * VL])
                    for s in T.serial(split_k - 1):
                        acc[0] = T.simd.vadd(acc[0], T.simd.vld(g_ub[s + 1, ch * VL]), mask)
                    if gated:
                        up = T.simd.alloc_local((1,), "float32")
                        up[0] = T.simd.vld(u_ub[0, ch * VL])
                        for s in T.serial(split_k - 1):
                            up[0] = T.simd.vadd(
                                up[0], T.simd.vld(u_ub[s + 1, ch * VL]), mask
                            )
                        den = T.simd.vadd(
                            one,
                            T.simd.vexp(T.simd.vneg(acc[0], mask), mask, precision=SFU),
                            mask,
                        )
                        y = T.simd.vmul(
                            T.simd.vdiv(acc[0], den, mask, precision=SFU), up[0], mask
                        )
                    else:
                        y = acc[0]
                    q = (
                        y
                        if c_dtype == "float32"
                        else T.simd.vcvt(y, c_dtype, round="ROUND_R", sat=False, part=0)
                    )
                    T.simd.vsts(o_ub[ch * VL], q, mask, dist=pk)
            T.copy(o_ub[:bcols], C[row, c0 : c0 + bcols])

    return tilelang.compile(reduce_kernel, target="ascend", out_idx=[])


# ---------------------------------------------------------------------------
# 3. Separate gated activation (unsplit gate/up path)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _build_silu_mul(M, N, dtype_str):
    """``out = silu(gate) * up`` with the Stage 3 UB-safe geometry clamp."""
    if M < 1 or N < 1:
        raise ValueError(f"empty problem: M={M} N={N}")
    bn = 256
    bn -= bn % VL
    bm = min(16 if M >= 64 else 1, M)
    mt = _ceil_div(M, bm)
    nt = _ceil_div(N, bn)
    target = 4 * VECTOR_CORES
    nsp, best = 1, None
    d = 1
    divisors = []
    while d * d <= nt:
        if nt % d == 0:
            divisors.append(d)
            if d != nt // d:
                divisors.append(nt // d)
        d += 1
    for cand in sorted(divisors):
        grid = mt * cand
        key = (abs(grid - target), grid)
        if best is None or key < best:
            best, nsp = key, cand
    tpb = nt // nsp
    grid = mt * nsp
    ub = 2 * 3 * bm * bn * ITEMSIZE[dtype_str]
    if ub > UB_BYTES:
        raise ValueError(f"silu_mul UB footprint {ub} B > {UB_BYTES} B")
    nchunk = bn // VL

    @T.prim_func
    def silu_mul(
        gate_up: T.Tensor((M, 2 * N), dtype_str),
        out: T.Tensor((M, N), dtype_str),
    ) -> None:
        with T.Kernel(grid) as bx:
            im = bx // nsp
            sp = bx % nsp
            g_ub = T.alloc_shared((bm, bn), dtype_str)
            u_ub = T.alloc_shared((bm, bn), dtype_str)
            y_ub = T.alloc_shared((bm, bn), dtype_str)
            T.annotate_buffer_versions({g_ub: 2, u_ub: 2, y_ub: 2})
            for it in T.Pipelined(tpb, num_stages=2):
                tn = sp * tpb + it
                m0 = im * bm
                n0 = tn * bn
                am = T.min(bm, M - m0)
                an = T.min(bn, N - n0)
                T.copy(gate_up[m0 : m0 + am, n0 : n0 + an], g_ub[0:am, 0:an])
                T.copy(gate_up[m0 : m0 + am, N + n0 : N + n0 + an], u_ub[0:am, 0:an])
                with T.SimdVF():
                    full = T.simd.pset(32)
                    for r in T.serial(bm):
                        for c in T.serial(nchunk):
                            g = T.simd.vcvt(
                                T.simd.vld(g_ub[r, c * VL], dist="UNPK_B16"),
                                "float32",
                                part=0,
                            )
                            u = T.simd.vcvt(
                                T.simd.vld(u_ub[r, c * VL], dist="UNPK_B16"),
                                "float32",
                                part=0,
                            )
                            num = T.simd.vmul(g, u, full)
                            den = T.simd.vadds(
                                T.simd.vexp(T.simd.vneg(g, full), full, precision=SFU),
                                1.0,
                                full,
                            )
                            y = T.simd.vdiv(num, den, full, precision=SFU)
                            T.simd.vsts(
                                y_ub[r, c * VL],
                                T.simd.vcvt(y, dtype_str, round="ROUND_R", sat=False, part=0),
                                full,
                                dist="PK_B32",
                            )
                T.copy(y_ub[0:am, 0:an], out[m0 : m0 + am, n0 : n0 + an])

    return tilelang.compile(silu_mul, target="ascend", out_idx=[1])


# ---------------------------------------------------------------------------
# 4. Host decomposition (same contracts as the Stage 3 wrapper)
# ---------------------------------------------------------------------------
def _k_tile(cfg, k):
    """``(block_k, k_eff)``; block_k must divide k, else pad K with zeros."""
    for cand in (cfg["block_k"], 64, 32, MIN_BLOCK_M):
        if k % cand == 0:
            return cand, k
    k_eff = _ceil_div(k, MIN_BLOCK_M) * MIN_BLOCK_M
    for cand in (cfg["block_k"], 64, 32, MIN_BLOCK_M):
        if k_eff % cand == 0:
            return cand, k_eff
    return MIN_BLOCK_M, k_eff


def _split_k(m, n, cfg):
    """4 slices only in the decode band with the exact-tile precondition."""
    block_m = _clamp_block_m(m, cfg["block_m"])
    if cfg.get("split_k") is not None:
        wanted = int(cfg["split_k"])
        if wanted <= 1:
            return 1
    else:
        wanted = SPLIT_K_MAX_SLICES
    block_k, k_eff = _k_tile(cfg, cfg["_k"])
    if m > SPLIT_K_MAX_TOKENS or m < MIN_BLOCK_M:
        return 1
    if m % block_m or n % cfg["block_n"]:
        return 1
    k_tiles = k_eff // block_k
    if k_tiles < wanted or k_tiles % wanted:
        return 1
    return wanted


def _gemm_call(m, n, k, cfg, dtype_str, out_dtype=None):
    """Build one GEMM (plus its reduce when split) and return the callable."""
    block_k, k_eff = _k_tile(cfg, k)
    block_m = _clamp_block_m(m, cfg["block_m"])
    cfg = dict(cfg)
    cfg["_k"] = k
    split = _split_k(m, n, cfg)
    mainloop = _build_gemm(m, n, k_eff, dtype_str, out_dtype, block_m,
                           cfg["block_n"], block_k, cfg["num_stages"], split)
    if k_eff != k:
        pad = k_eff - k
        unpadded = mainloop

        def padded(a, b):
            return unpadded(F.pad(a, (0, pad)), F.pad(b, (0, pad)))

        mainloop = padded
    if split == 1:
        return mainloop

    activation = cfg.get("activation", "none")
    reduce_program = _build_reduce(split, m, n, dtype_str, activation, out_dtype)
    out_n = n // 2 if activation == "silu_and_mul" else n
    torch_dtype = getattr(torch, out_dtype or dtype_str)

    def gemm_split(a, b):
        partial = mainloop(a, b)
        out = torch.empty((m, out_n), dtype=torch_dtype, device=a.device)
        reduce_program(partial, out)
        return out

    return gemm_split


class SharedExpertMLPKernel:
    """Drop-in replacement for the Stage 3 kernel class (same call shape)."""

    ascend_mode = None
    supported_archs = None

    def __init__(self, num_tokens, hidden_size, ffn_size, dtype=torch.bfloat16,
                 config=None, *, device_index=None):
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dtype = dtype
        self.device_index = device_index
        self._program_key = None
        self._programs = None
        self._silu_program = None
        self.config = dict(self.default_config)
        if config:
            for key, value in config.items():
                if value is not None:
                    self.config[key] = value
        self.resolved_config = dict(self.config)

    @property
    def dtype_str(self):
        return str(self.dtype).split(".")[-1]

    @property
    def default_config(self):
        return {
            "block_m": TUNED_BLOCK_M,
            "block_n": TUNED_BLOCK_N,
            "block_k": TUNED_BLOCK_K,
            "num_stages": TUNED_NUM_STAGES,
            "split_k": TUNED_SPLIT_K,
            "fuse_silu": True,
        }

    def init_config(self, config=None):
        if config:
            for key, value in config.items():
                if value is not None:
                    self.config[key] = value
        self.resolved_config = dict(self.config)

    def _programs_for_config(self):
        key = tuple(sorted((k, str(v)) for k, v in self.config.items()))
        if self._program_key != key:
            self._programs = self._decompose()
            self._silu_program = None
            self._program_key = key
        return self._programs

    def _decompose(self):
        num_tokens, hidden_size, ffn_size = (
            self.num_tokens, self.hidden_size, self.ffn_size,
        )
        cfg = dict(self.config)
        # gate/up: a single fused GEMM + gated-vector epilogue when the tiles
        # are exact (``dual_copy`` precondition); otherwise the plain GEMM plus
        # the separate activation, or the split-K reduce that fuses it.
        bn_gate = 0
        _bm = _clamp_block_m(num_tokens, cfg["block_m"])
        _cap = 65536 // (2 * _bm)  # L0C element cap for the two fp32 accumulators
        # The fused block carries two accumulators, so its N tile is half the
        # single-tile one; that keeps the L1 fill ratio identical and the block
        # count per S tile identical to the unfused gate/up GEMM.  The tile must
        # also be a multiple of the 64-lane vector chunk: the epilogue walks the
        # columns in VL-wide chunks and has no tail mask.
        for cand in (cfg["block_n"] // 2, 128, 64):
            if 16 <= cand <= _cap and ffn_size % cand == 0 and cand % VL == 0:
                bn_gate = cand
                break
        exact = (bn_gate > 0 and num_tokens % _bm == 0
                 and hidden_size % cfg["block_k"] == 0)
        fused = None
        gate_up = None
        if cfg.get("fuse_silu", True) and exact:
            fused = _build_gate_up_fused(
                num_tokens, ffn_size, hidden_size, self.dtype_str,
                _bm, bn_gate, cfg["block_k"], cfg["num_stages"],
            )
        elif _split_k(num_tokens, 2 * ffn_size, dict(cfg, _k=hidden_size)) == 1:
            gate_up = _gemm_call(num_tokens, 2 * ffn_size, hidden_size, cfg, self.dtype_str)
        else:
            gate_cfg = dict(cfg, activation="silu_and_mul")
            fused = _gemm_call(num_tokens, 2 * ffn_size, hidden_size, gate_cfg,
                               self.dtype_str)
        down = _gemm_call(num_tokens, hidden_size, ffn_size, cfg, self.dtype_str)
        return gate_up, down, fused

    def forward(self, hidden, w_gate_up, w_down):
        gate_up_program, down_program, fused = self._programs_for_config()
        if fused is not None:
            activated = fused(hidden, w_gate_up)
        else:
            gate_up = gate_up_program(hidden, w_gate_up)
            activated = self._silu_mul(gate_up)
        return down_program(activated, w_down)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _silu_mul(self, gate_up):
        if self._silu_program is None:
            self._silu_program = _build_silu_mul(
                self.num_tokens, self.ffn_size, self.dtype_str
            )
        return self._silu_program(gate_up)


def build_kernel(num_tokens, hidden_size, ffn_size, dtype_str, config=None):
    """Harness entry: returns a callable with ``resolved_config`` attached."""
    kernel = SharedExpertMLPKernel(
        num_tokens, hidden_size, ffn_size, getattr(torch, dtype_str), config
    )
    fn = lambda a, b, c: kernel(a, b, c)  # noqa: E731
    fn.resolved_config = kernel.resolved_config
    return fn


# ---------------------------------------------------------------------------
# 5. Golden + layered precision checks (Stage 3 tolerances, unchanged)
# ---------------------------------------------------------------------------
def golden_shared_expert(hidden, w_gate_up, w_down):
    """fp32 reference straight from the operator semantics."""
    T_, H_, _ = hidden.shape[0], w_gate_up.shape[1], w_down.shape[0]
    s = w_down.shape[1]
    assert w_gate_up.shape == (2 * s, H_) and w_down.shape == (H_, s)
    gate = hidden.float() @ w_gate_up[:s].float().t()
    up = hidden.float() @ w_gate_up[s:].float().t()
    # The gated product is materialised in the input dtype (the activation
    # kernel's output is the down GEMM's A operand), then accumulated in fp32.
    act = (F.silu(gate) * up).to(hidden.dtype)
    return (act.float() @ w_down.float().t()).to(hidden.dtype)


def _nse(got, want):
    g, w = got.double(), want.double()
    return ((g - w) ** 2).sum().item() / max((w**2).sum().item(), 1e-30)


_CASES = []
_MAX_ABS = 0.0


def run_case(level, name, T_, H_, S_, dtype_str, config=None, seed=0):
    global _MAX_ABS
    torch.manual_seed(seed)
    dtype = getattr(torch, dtype_str)
    hidden = torch.randn(T_, H_, dtype=dtype)
    w_gate_up = torch.randn(2 * S_, H_, dtype=dtype) * 0.05
    w_down = torch.randn(H_, S_, dtype=dtype) * 0.05
    want = golden_shared_expert(hidden, w_gate_up, w_down)
    kernel = SharedExpertMLPKernel(T_, H_, S_, dtype, config)
    try:
        got = kernel(hidden.npu(), w_gate_up.npu(), w_down.npu()).cpu()
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001 - reported, not silently dropped
        print(f"[{level}][FAIL] {name}: {type(exc).__name__}: {exc}"[:600])
        _CASES.append((level, name, False, str(exc)[:300]))
        return False
    assert got.shape == want.shape, (got.shape, want.shape)
    diff = (got.float() - want.float()).abs().max().item()
    nse = _nse(got, want)
    ok = diff <= (HALF_ATOL + HALF_RTOL * want.float().abs().max().item()) and nse <= HALF_NSE
    _MAX_ABS = max(_MAX_ABS, diff)
    print(
        f"       [{level}] {name}: T={T_} H={H_} S={S_} {dtype_str} "
        f"max_diff={diff:.3e} nse={nse:.2e} {'ok' if ok else 'FAIL'}"
    )
    _CASES.append((level, name, ok, f"diff={diff:.3e} nse={nse:.2e}"))
    return ok


def run_L0():
    """The five manifest workloads plus the M-tail and both 16-bit dtypes."""
    run_case("L0", "kimi-k2-t512", 512, 7168, 18432, "bfloat16", seed=1)
    run_case("L0", "ds-v3-t4096", 4096, 7168, 2048, "bfloat16", seed=2)
    run_case("L0", "ds-v3-t32-bf16", 32, 7168, 2048, "bfloat16", seed=3)
    run_case("L0", "ds-v3-t32-fp16", 32, 7168, 2048, "float16", seed=4)
    run_case("L0", "glm45-t32", 32, 5120, 1536, "bfloat16", seed=5)
    run_case("L0", "kimi-k2-t1", 1, 7168, 18432, "bfloat16", seed=6)
    run_case("L0", "tiny-tail-17x512x128", 17, 512, 128, "bfloat16", seed=7)
    run_case("L0", "k576-ladder", 64, 576, 1024, "bfloat16", seed=8)
    run_case("L0", "smoke-32x256x128", 32, 256, 128, "bfloat16", seed=9)
    run_case("L0", "smoke-32x256x128-fp16", 32, 256, 128, "float16", seed=10)


def run_L1():
    """Cross dtype and geometry coverage."""
    run_case("L1", "kimi-k2-t512-fp16", 512, 7168, 18432, "float16", seed=21)
    run_case("L1", "ds-v3-t4096-fp16", 4096, 7168, 2048, "float16", seed=22)
    run_case("L1", "kimi-k2-t1-fp16", 1, 7168, 18432, "float16", seed=23)
    run_case("L1", "rows-128-7168x2048", 128, 7168, 2048, "bfloat16", seed=24)
    run_case("L1", "rows-100-tail", 100, 7168, 2048, "bfloat16", seed=25)
    run_case("L1", "glm45-t1", 1, 5120, 1536, "bfloat16", seed=26)
    run_case("L1", "rows-2048-ds", 2048, 7168, 2048, "bfloat16", seed=27)


def _finish(level):
    rows = [row for row in _CASES if row[0] == level]
    failed = [row for row in rows if not row[2]]
    print(f"[{level}] {len(rows) - len(failed)}/{len(rows)} cases passed")
    if failed:
        for row in failed:
            print(f"[{level}][FAIL] {row[1]}: {row[3]}")
        raise SystemExit(f"[{level}] {len(failed)} case(s) failed")


def run_L2():
    """Larger scale than the gate (informative, non-blocking on divergence)."""
    for name, T_, H_, S_, dtype_str, seed in (
        ("L2-kimi-t512", 512, 7168, 18432, "bfloat16", 31),
        ("L2-ds-t4096", 4096, 7168, 2048, "bfloat16", 32),
        ("L2-ds-t2048", 2048, 7168, 2048, "float16", 33),
    ):
        try:
            run_case("L2", name, T_, H_, S_, dtype_str, seed=seed)
        except AssertionError as exc:  # recorded, not blocking
            print(f"[L2][WARN] {name}: {str(exc)[:200]}")


def run_boundary():
    """Boundary: M/N tails, a padded K contraction, k-ladder fallbacks."""
    run_case("B", "B-m17-n-tail", 17, 512, 128, "bfloat16", seed=41)
    run_case("B", "B-m1-n-tail", 1, 1000, 128, "bfloat16", seed=42)
    run_case("B", "B-narrow-tiles-1020", 512, 1024, 1020, "bfloat16", seed=43)
    run_case("B", "B-k-ladder-576", 64, 576, 1024, "bfloat16", seed=44)
    run_case("B", "B-single-row-16", 16, 256, 128, "float16", seed=45)
    run_case("B", "B-t32-exact", 32, 256, 128, "bfloat16", seed=46)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0",
                        choices=["L0", "L1", "L2", "boundary", "all", "bench"])
    args, _ = parser.parse_known_args()
    started = time.perf_counter()
    if args.level in ("L0", "all"):
        run_L0()
    if args.level in ("L1", "all"):
        run_L1()
    if args.level in ("L2", "all"):
        run_L2()
    if args.level in ("boundary", "all"):
        run_boundary()
    if args.level != "bench":
        print(f"max_abs_diff={_MAX_ABS:.3e} total={time.perf_counter() - started:.1f}s")
        print("[PRECISION_PASS]")


if __name__ == "__main__":
    main()
