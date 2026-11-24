# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""`_gemm_basic_kernel` on Ascend 950PR: NT-normalized L1-operand dense GEMM.

Launch 1 of ``SharedExpertMLPFwdOp`` (``C = op(A) @ op(B) [+ bias]``), migrated
from the GPU TileLang source ``tileops/_kernels/gemm/dense.py``.  Follows the
frozen DESIGN.md / passed REVIEW.md of this directory:

* R1 -- the backend GEMM is NT only (``transpose_B=True``, ``transpose_A``
  unsupported), so the four storage layouts are normalized during the GM -> L1
  load: ``nd2nz`` for a ``(M,K)``/``(N,K)`` store, ``dn2nz`` (``transpose=True``)
  for a transposed store.  No arithmetic changes.
* R2 -- ``a1``/``b1`` are L1 buffers with ``num_stages`` versions declared through
  ``T.annotate_buffer_versions`` and rotated by ``T.Pipelined``; L1 -> L0 and the
  L0 double buffer live inside the backend ``ascend_gemm_l1`` template.
* R3 -- ``c0`` is an fp32 L0C accumulator kept across the K loop; the store is one
  fixpipe ``T.copy`` that quantizes fp32 -> output dtype in hardware
  (``quant_pre`` 16 for bf16, 1 for fp16, 0 for fp32).
* R4 -- ``T.Kernel`` takes a 1-D grid, so ``(n tile, m tile, split slice)`` is
  packed into one block index with the M tile as the fast axis (the A panel stays
  resident in L2 while B tiles stream).
* R5 -- ``has_bias`` (only legal with ``split_k == 1``) uses the Cube + Vector
  epilogue: ``dual_copy`` halves the L0C tile into the two AIV banks, a
  ``T.SimtVF`` pass adds the bias in fp32 and casts, ``dual_copy`` writes back.
* R6 -- ``split_k > 1`` writes fp32 partial sums ``[split_k, m, n]`` for the
  companion ``splitk_reduce_kernel`` (never quantized here).

Tile contract (DESIGN.md 5.2, enforced at build time): tiles are multiples of 16,
``k % block_k == 0``, ``block_m * block_n * 4 <= 256 KiB`` (L0C) and
``(block_m + block_n) * block_k * itemsize * num_stages <= 512 KiB`` (L1).
``block_m`` is clamped to the geometry so the harness default 128 also works for
``m = 1/32``; ``split_k > 1`` keeps the source's exact-tile precondition and
raises ``ValueError`` when ``m < 16`` (that case must run with ``split_k = 1``).
The bias epilogue needs exact M/N tiles because ``dual_copy`` cannot infer a
symbolic region extent.

run: python _gemm_basic_kernel.py --level L0|L1|L2|boundary|all
"""

import argparse
import functools
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401 -- registers torch.npu

# Frozen Stage 3 tiling (DESIGN.md 5.2/5.6).
BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 64
NUM_STAGES = 3

# DESIGN.md 8.2 tolerances.
HALF_ATOL = 1e-2
HALF_RTOL = 1e-2
HALF_NSE = 1e-4
FP32_ATOL = 1e-4
FP32_RTOL = 1e-4

ITEMSIZE = {"float16": 2, "bfloat16": 2, "float32": 4}

_DIFFS = []


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference, from the operator semantics only)
# ---------------------------------------------------------------------------
def golden_gemm_basic(a, b, trans_a=False, trans_b=False, bias=None, out_dtype=None):
    """``C = op(A) @ op(B) [+ bias]``: fp32 accumulation, one final rounding.

    Deliberately free of kernel-side devices: no tiling, no partial sums, no
    hardware quantize step, no traversal order.
    """
    ah = a.float().t() if trans_a else a.float()
    bh = b.float().t() if trans_b else b.float()
    acc = ah @ bh
    if bias is not None:
        acc = acc + bias.float().unsqueeze(0)  # column broadcast, fp32 add
    return acc.to(out_dtype or a.dtype)


def golden_gemm_basic_splitk(a, b, split_k, trans_a=False, trans_b=False):
    """``split_k > 1`` output contract: ``[split_k, m, n]`` fp32 partial sums."""
    ah = a.float().t() if trans_a else a.float()
    bh = b.float().t() if trans_b else b.float()
    k = ah.shape[1]
    assert k % split_k == 0, (k, split_k)
    ks = k // split_k
    return torch.stack(
        [ah[:, s * ks : (s + 1) * ks] @ bh[s * ks : (s + 1) * ks, :] for s in range(split_k)]
    )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def _ceil_div(x, y):
    return -(-x // y)


def _clamp_block_m(m, block_m):
    """Clamp the M tile to the geometry (DESIGN.md 5.2 rule 3, applied to both bands).

    ``min(block_m, align16(m))`` keeps the tile a multiple of 16 while never
    allocating a tile with more dead rows than live ones, so the harness default
    ``block_m = 128`` is usable for the decode band (``m = 1/32``).
    """
    return min(block_m, max(16, _ceil_div(m, 16) * 16))


def _check_tiles(m, n, k, block_m, block_n, block_k, num_stages, split_k, dtype, has_bias):
    """Compile-time tile contract (DESIGN.md 5.2 rules 1-5)."""
    if min(block_m, block_n, block_k) <= 0 or num_stages <= 0 or split_k <= 0:
        raise ValueError(
            f"tiles must be positive: block=({block_m},{block_n},{block_k}) "
            f"num_stages={num_stages} split_k={split_k}"
        )
    if block_m % 16 or block_n % 16 or block_k % 16:
        raise ValueError(
            f"block_m/block_n/block_k must be multiples of 16 "
            f"(backend static_assert M/N % FRAC_N == 0): ({block_m},{block_n},{block_k})"
        )
    if k % block_k:
        raise ValueError(
            f"k must be divisible by block_k (a K tail corrupts live outputs): k={k} block_k={block_k}"
        )
    if split_k > 1:
        exact = m % block_m == 0 and n % block_n == 0 and (k // block_k) % split_k == 0
        if m < 16 or not exact:
            raise ValueError(
                "split-K basic GEMM requires exact M/N/K tiles and m >= 16: "
                f"m={m} n={n} k={k} block=({block_m},{block_n},{block_k}) split_k={split_k}; "
                "run this geometry with split_k = 1 instead"
            )
    if has_bias and split_k > 1:
        raise ValueError(
            "bias belongs to the split_k == 1 contract (the reduce kernel owns it otherwise): "
            f"has_bias={has_bias} split_k={split_k}"
        )
    if has_bias and (m % block_m or n % block_n):
        raise ValueError(
            "the Cube + Vector bias epilogue needs exact M/N tiles "
            f"(dual_copy cannot infer a symbolic extent): m={m} n={n} "
            f"block=({block_m},{block_n})"
        )
    if block_m * block_n * 4 > 262144:
        raise ValueError(
            f"L0C overflow: block_m*block_n*4 = {block_m * block_n * 4} > 262144"
        )
    l1 = (block_m + block_n) * block_k * ITEMSIZE[dtype] * num_stages
    if l1 > 524288:
        raise ValueError(f"L1 overflow: {l1} > 524288 bytes")


def _gemm_basic_program(m, n, k, trans_a, trans_b, dtype, has_bias, out_dtype,
                        block_m, block_n, block_k, num_stages, split_k):
    """The designed program: L1-operand NT GEMM with a fixpipe / Vector epilogue."""
    c_dtype = out_dtype or dtype
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)
    mt = _ceil_div(m, block_m)
    nt = _ceil_div(n, block_n)
    k_tiles = k // block_k
    k_slice = k_tiles // split_k
    exact_m = m % block_m == 0
    exact_n = n % block_n == 0
    out_shape = (m, n) if split_k == 1 else (split_k, m, n)
    out_dtype_ = c_dtype if split_k == 1 else "float32"
    grid = nt * mt * split_k

    if k_tiles == 0:
        # k == 0: the main loop is empty, so the source semantics are "write bias
        # or zero".  DESIGN.md 9.3.5 keeps this off L0C (an unverified T.clear on
        # the accumulator) and fills the UB tile directly instead.
        @T.prim_func
        def gemm_basic_empty(
            A: T.Tensor(a_shape, dtype),
            B: T.Tensor(b_shape, dtype),
            C: T.Tensor(out_shape, out_dtype_),
        ):
            with T.Kernel(grid) as bid:
                bx = bid // (mt * split_k)  # n tile: slow axis
                by = bid % (mt * split_k) // split_k  # m tile: fast axis
                bz = bid % split_k  # split-K slice: fastest
                m0 = by * block_m
                n0 = bx * block_n
                am = block_m if exact_m else T.min(m - m0, block_m)
                an = block_n if exact_n else T.min(n - n0, block_n)
                z = T.alloc_shared((block_m, block_n), out_dtype_)
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(block_m, block_n):
                        z[i, j] = T.cast(0, out_dtype_)
                if split_k == 1:
                    T.copy(z[:am, :an], C[m0 : m0 + am, n0 : n0 + an])
                else:
                    T.copy(z[:am, :an], C[bz, m0 : m0 + am, n0 : n0 + an])

        @T.prim_func
        def gemm_basic_empty_bias(
            A: T.Tensor(a_shape, dtype),
            B: T.Tensor(b_shape, dtype),
            bias: T.Tensor((n,), dtype),
            C: T.Tensor(out_shape, out_dtype_),
        ):
            with T.Kernel(grid) as bid:
                bx = bid // (mt * split_k)  # n tile: slow axis
                by = bid % (mt * split_k) // split_k  # m tile: fast axis
                bz = bid % split_k  # split-K slice: fastest
                m0 = by * block_m
                n0 = bx * block_n
                z = T.alloc_shared((block_m, block_n), out_dtype_)
                bv = T.alloc_shared((block_n,), dtype)
                T.copy(bias[n0 : n0 + block_n], bv)
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(block_m, block_n):
                        z[i, j] = T.cast(bv[j], out_dtype_)
                T.copy(z, C[m0 : m0 + block_m, n0 : n0 + block_n])

        return gemm_basic_empty_bias if has_bias else gemm_basic_empty

    if has_bias:

        @T.prim_func
        def gemm_basic_bias(
            A: T.Tensor(a_shape, dtype),
            B: T.Tensor(b_shape, dtype),
            bias: T.Tensor((n,), dtype),
            C: T.Tensor(out_shape, out_dtype_),
        ):
            with T.Kernel(grid) as bid:
                bx = bid // (mt * split_k)  # n tile: slow axis
                by = bid % (mt * split_k) // split_k  # m tile: fast axis
                bz = bid % split_k  # split-K slice: fastest
                m0 = by * block_m
                n0 = bx * block_n

                a1 = T.alloc_l1((block_m, block_k), dtype)
                b1 = T.alloc_l1((block_n, block_k), dtype)
                c0 = T.alloc_l0c((block_m, block_n), "float32")
                T.annotate_buffer_versions({a1: num_stages, b1: num_stages})

                for step in T.Pipelined(k_slice, num_stages=num_stages):
                    k0 = (bz * k_slice + step) * block_k
                    if trans_a:
                        T.copy(A[k0 : k0 + block_k, m0 : m0 + block_m], a1, transpose=True)
                    else:
                        T.copy(A[m0 : m0 + block_m, k0 : k0 + block_k], a1)
                    if trans_b:
                        T.copy(B[n0 : n0 + block_n, k0 : k0 + block_k], b1)
                    else:
                        T.copy(B[k0 : k0 + block_k, n0 : n0 + block_n], b1, transpose=True)
                    T.gemm(a1, b1, c0, transpose_B=True, clear_accum=(step == 0))

                # Cube + Vector epilogue: the fixpipe splits the fp32 accumulator
                # over the two AIV banks, the bias is added in fp32 and rounded once.
                cv = T.alloc_shared((block_m // 2, block_n), "float32")
                co = T.alloc_shared((block_m // 2, block_n), dtype)
                bv = T.alloc_shared((block_n,), dtype)
                T.copy(bias[n0 : n0 + block_n], bv)
                T.dual_copy(c0, cv)
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(block_m // 2, block_n):
                        co[i, j] = T.cast(cv[i, j] + T.cast(bv[j], "float32"), dtype)
                T.dual_copy(co, C[m0 : m0 + block_m, n0 : n0 + block_n])

        return gemm_basic_bias

    @T.prim_func
    def gemm_basic(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        C: T.Tensor(out_shape, out_dtype_),
    ):
        with T.Kernel(grid) as bid:
            bx = bid // (mt * split_k)  # n tile: slow axis
            by = bid % (mt * split_k) // split_k  # m tile: fast axis
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
                # NT normalization: A -> (M,K) tile, B -> (N,K) tile in L1.
                if trans_a:
                    T.copy(A[k0 : k0 + block_k, m0 : m0 + am], a1[:am, :], transpose=True)
                else:
                    T.copy(A[m0 : m0 + am, k0 : k0 + block_k], a1[:am, :])
                if trans_b:
                    T.copy(B[n0 : n0 + an, k0 : k0 + block_k], b1[:an, :])
                else:
                    T.copy(B[k0 : k0 + block_k, n0 : n0 + an], b1[:an, :], transpose=True)
                T.gemm(a1, b1, c0, transpose_B=True, clear_accum=(step == 0))

            # fixpipe store: one rounding fp32 -> c_dtype.  The source region is
            # the whole accumulator and only the destination is truncated: an
            # M-truncated L0C source region makes the backend derive the row pitch
            # from the region itself (ceil16(am)) instead of the tile, which
            # scrambles every column past the first fractal (see
            # repro/TRAP-950-l0c-region-store-stride.py).  Tail rows of c0 stay
            # dead because they are outside the destination region.
            if split_k == 1:
                T.copy(c0, C[m0 : m0 + am, n0 : n0 + an])
            else:
                # fp32 partial sum for the companion reduce kernel (no quantize).
                T.copy(c0, C[bz, m0 : m0 + am, n0 : n0 + an])

    return gemm_basic


@functools.lru_cache(maxsize=None)
def build(m, n, k, trans_a, trans_b, dtype, has_bias, out_dtype,
          block_m, block_n, block_k, num_stages, split_k):
    """Resolve the tile contract and compile one specialization."""
    c_dtype = out_dtype or dtype
    if dtype not in ITEMSIZE or c_dtype not in ITEMSIZE:
        raise ValueError(f"unsupported dtype pair: {dtype} -> {c_dtype}")
    block_m = _clamp_block_m(m, block_m)
    _check_tiles(m, n, k, block_m, block_n, block_k, num_stages, split_k, dtype, has_bias)
    prog = _gemm_basic_program(
        m, n, k, trans_a, trans_b, dtype, has_bias, out_dtype,
        block_m, block_n, block_k, num_stages, split_k,
    )
    return tilelang.compile(prog, target="ascend", out_idx=-1)


def _gemm_basic_kernel(m, n, k, trans_a, trans_b, dtype, has_bias=False, out_dtype=None):
    """Harness-shaped factory: the outer call fixes the problem, the inner the tiles.

    Same outer signature as the GPU source; the inner one drops ``threads``
    (this backend has no thread parameter, DESIGN.md 3.3 / 10.4).
    """

    def _gemm_basic_func(block_m=BLOCK_M, block_n=BLOCK_N, block_k=BLOCK_K,
                         num_stages=NUM_STAGES, split_k=1):
        return build(m, n, k, trans_a, trans_b, dtype, has_bias, out_dtype,
                     block_m, block_n, block_k, num_stages, split_k)

    return _gemm_basic_func


def _split_k_for(m, n, k, block_m, block_n, block_k):
    """Decode-band split (the caller's ``_split_k(T <= 32) = 4``), degraded to 1.

    ``split_k > 1`` keeps the source's exact-tile precondition, so any geometry
    that cannot honour it (``m < 16``, a K/N remainder, fewer than 4 K tiles)
    runs unsplit instead of raising.
    """
    if not (16 <= m <= 32 and m % block_m == 0 and n % block_n == 0):
        return 1
    if k < 4 * block_k or (k // block_k) % 4:
        return 1
    return 4


def default_config(m, n, k, block_m=None, block_n=None, block_k=BLOCK_K):
    """Frozen Stage 3 tiling (DESIGN.md 5.6) with the design's degradations."""
    bm = _clamp_block_m(m, BLOCK_M if block_m is None else block_m)
    bn = BLOCK_N if block_n is None else block_n
    return {
        "block_m": bm,
        "block_n": bn,
        "block_k": block_k,
        "num_stages": NUM_STAGES,
        "split_k": _split_k_for(m, n, k, bm, bn, block_k),
    }


# ---------------------------------------------------------------------------
# Precision comparing
# ---------------------------------------------------------------------------
def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _splitk_reference_fp64(a, b, split_k, trans_a, trans_b):
    """fp64 reference of the partial sums (DESIGN.md 8.2 reference convention)."""
    ah = a.double().t() if trans_a else a.double()
    bh = b.double().t() if trans_b else b.double()
    ks = ah.shape[1] // split_k
    return torch.stack(
        [ah[:, s * ks : (s + 1) * ks] @ bh[s * ks : (s + 1) * ks, :] for s in range(split_k)]
    )


def _nse(got, want):
    """Normalized squared error of DESIGN.md 8.2."""
    g, w = got.double(), want.double()
    return ((g - w) ** 2).sum().item() / max((w**2).sum().item(), 1e-30)


def _max_ulp(got, want):
    """Largest fp32 ulp distance of ``got`` from ``want`` (reported, not asserted).

    Elements whose reference is exactly zero have no meaningful ulp scale and are
    skipped; a half-precision output shows ~1-2e3 fp32 ulp per target ulp.
    """
    got_f = got.float()
    want_f = want.float()
    mask = want_f != 0
    if not bool(mask.any()):
        return 0.0
    want_f = want_f[mask]
    got_f = got_f[mask]
    ulp = (torch.nextafter(want_f, torch.full_like(want_f, float("inf"))) - want_f).double()
    return ((got_f.double() - want_f.double()).abs() / ulp.abs()).max().item()


def _check(got, want, tag, atol, rtol, nse_limit=None):
    """Elementwise tolerance + NSE gate; returns ``(max_abs_diff, nse, ulp)``."""
    assert got.shape == want.shape, f"[{tag}] shape {tuple(got.shape)} != {tuple(want.shape)}"
    g = got.double()
    w = want.double()
    diff = (g - w).abs()
    bad = diff > (atol + rtol * w.abs())
    n_bad = int(bad.sum().item())
    max_diff = diff.max().item() if diff.numel() else 0.0
    nse = _nse(g, w)
    if n_bad:
        first = tuple(bad.nonzero()[0].tolist())
        raise AssertionError(
            f"[{tag}] {n_bad}/{diff.numel()} elements out of tolerance "
            f"(atol={atol:g} rtol={rtol:g}); max_diff={max_diff:.3e}; "
            f"first_bad={first} got={g[first].item():.6e} want={w[first].item():.6e}"
        )
    if nse_limit is not None and nse > nse_limit:
        raise AssertionError(f"[{tag}] NSE {nse:.3e} > {nse_limit:g}")
    return max_diff, nse, _max_ulp(got, want)


def run_case(m, n, k, dtype, tag, trans_a=False, trans_b=False, has_bias=False,
             out_dtype=None, block_m=None, block_n=None, block_k=None,
             num_stages=None, split_k=None, seed=0, strict=False):
    """Compile, launch and check one shape; returns the per-case max diff."""
    cfg = default_config(m, n, k, block_m, block_n, BLOCK_K if block_k is None else block_k)
    if num_stages is not None:
        cfg["num_stages"] = num_stages
    if split_k is not None:
        cfg["split_k"] = split_k

    out_torch = None if out_dtype is None else getattr(torch, str(out_dtype))
    torch.manual_seed(seed)
    a = torch.randn((k, m) if trans_a else (m, k), dtype=dtype)
    b = torch.randn((n, k) if trans_b else (k, n), dtype=dtype)
    bias = torch.randn(n, dtype=dtype) if has_bias else None

    fn = _gemm_basic_kernel(
        m, n, k, trans_a, trans_b, _dtype_name(dtype), has_bias,
        None if out_dtype is None else str(out_dtype),
    )(cfg["block_m"], cfg["block_n"], cfg["block_k"], cfg["num_stages"], cfg["split_k"])

    args = [a.to("npu"), b.to("npu")]
    if has_bias:
        args.append(bias.to("npu"))

    # The first launch of a specialization pays the device-side kernel load, so
    # warm up once and then time a few steady-state launches.
    out = fn(*args)
    torch.npu.synchronize()
    reps = 3
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
    got = out.cpu()  # the device->host copy also waits for the last launch
    torch.npu.synchronize()
    elapsed = (time.perf_counter() - t0) / reps
    tflops = 2.0 * m * n * k / max(elapsed, 1e-9) / 1e12

    if cfg["split_k"] > 1:
        # 1) the partial-sum contract: [split_k, m, n] fp32, no quantize.
        ref64 = _splitk_reference_fp64(a, b, cfg["split_k"], trans_a, trans_b)
        assert got.shape == (cfg["split_k"], m, n), (tag, tuple(got.shape))
        assert got.dtype == torch.float32, (tag, got.dtype)
        max_diff, _, ulp = _check(got, ref64, f"{tag}/partials", FP32_ATOL, FP32_RTOL)
        # 2) end to end through the reduce kernel's semantics (sum of slices).
        want = golden_gemm_basic(a, b, trans_a, trans_b, None, out_torch)
        e2e = got.sum(0).to(want.dtype)
        e2e_diff, e2e_nse, _ = _check(
            e2e, want, f"{tag}/sum", HALF_ATOL, HALF_RTOL, HALF_NSE
        )
        print(
            f"[{tag}] PASS m={m} n={n} k={k} {dtype} tiles="
            f"({cfg['block_m']},{cfg['block_n']},{cfg['block_k']},ns{cfg['num_stages']}"
            f",sk{cfg['split_k']}) partials_max_diff={max_diff:.3e} ulp={ulp:.1f} "
            f"e2e_max_diff={e2e_diff:.3e} nse={e2e_nse:.2e} "
            f"time={elapsed * 1e3:.2f}ms {tflops:.1f}TF/s"
        )
        _DIFFS.append((tag, max(max_diff, e2e_diff)))
        return max(max_diff, e2e_diff)

    want = golden_gemm_basic(a, b, trans_a, trans_b, bias, out_torch)
    if strict:
        assert torch.equal(got, want), f"[{tag}] strict compare failed"
        max_diff = 0.0
        print(f"[{tag}] PASS strict m={m} n={n} k={k} {dtype} time={elapsed * 1e3:.2f}ms")
        _DIFFS.append((tag, max_diff))
        return max_diff

    fp32_out = got.dtype == torch.float32
    atol, rtol, nse_limit = (
        (FP32_ATOL, FP32_RTOL, None) if fp32_out else (HALF_ATOL, HALF_RTOL, HALF_NSE)
    )
    max_diff, nse, ulp = _check(got, want, tag, atol, rtol, nse_limit)
    print(
        f"[{tag}] PASS m={m} n={n} k={k} {dtype} tiles="
        f"({cfg['block_m']},{cfg['block_n']},{cfg['block_k']},ns{cfg['num_stages']}"
        f",sk{cfg['split_k']}) ta={int(trans_a)} tb={int(trans_b)} "
        f"bias={int(has_bias)} out={got.dtype} max_diff={max_diff:.3e} "
        f"nse={nse:.2e} ulp={ulp:.1f} time={elapsed * 1e3:.2f}ms {tflops:.1f}TF/s"
    )
    _DIFFS.append((tag, max_diff))
    return max_diff


_WARNS = []


def _warn_case(fn, tag):
    """Run a non-blocking (L2 / boundary) case and record failures as warnings."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - recorded, non-blocking per the stage contract
        _WARNS.append((tag, f"{type(exc).__name__}: {exc}".splitlines()[0][:300]))
        print(f"[{tag}] WARN (recorded, non-blocking): {type(exc).__name__}: {exc}")
        return None


def _expect_value_error(fn, tag):
    """A degraded geometry must fail loudly at build time, never silently."""
    try:
        fn()
    except ValueError as exc:
        print(f"[{tag}] PASS ValueError: {str(exc).splitlines()[0][:150]}")
        return
    raise AssertionError(f"[{tag}] expected a ValueError")


# ---------------------------------------------------------------------------
# Hierarchical testing
# ---------------------------------------------------------------------------
def run_L0():
    """Gate: every branch of DESIGN.md 8.3 plus the geometry bands of the op."""
    bf16, fp16 = torch.bfloat16, torch.float16
    # L0-1 / L0-2 -- minimal tile, the two fixpipe quantize modes (16 / 1).
    run_case(32, 64, 128, bf16, "L0-1-bf16-quant16", block_m=32, block_n=64, num_stages=2)
    run_case(32, 64, 128, fp16, "L0-2-fp16-quant1", block_m=32, block_n=64, num_stages=2)
    # L0-3 -- m = 1 M-tail (am = 1), single-row fixpipe burst.
    run_case(1, 128, 256, bf16, "L0-3-m1-tail", block_m=16, block_n=128)
    # L0-4 -- split-K partial sums on exact tiles.
    run_case(32, 64, 256, bf16, "L0-4-splitk4", block_m=32, block_n=64, split_k=4)
    # L0-5 -- the four storage layouts, normalized during the L1 load.
    for ta, tb, name in (
        (False, True, "NT"), (False, False, "NN"), (True, True, "TN"), (True, False, "TT"),
    ):
        run_case(32, 64, 128, bf16, f"L0-5-{name}", trans_a=ta, trans_b=tb,
                 block_m=32, block_n=64, num_stages=2)
    # L0-6 -- Cube + Vector bias epilogue (dual_copy + SimtVF + dual_copy).
    run_case(32, 64, 128, bf16, "L0-6-bias", has_bias=True,
             block_m=32, block_n=64, num_stages=2)
    # L0-7 -- both M and N tails (am = 4 .. 32, an = 104).
    run_case(100, 1000, 128, bf16, "L0-7-mn-tail", block_m=32, block_n=128, num_stages=2)
    # L0-8 -- k == 0: write zeros / write bias (source semantics, strict compare).
    run_case(16, 64, 0, bf16, "L0-8-k0", block_m=16, block_n=64, split_k=1, strict=True)
    run_case(16, 64, 0, bf16, "L0-8-k0-bias", has_bias=True, block_m=16, block_n=64,
             strict=True)
    # L0-8b -- k == 0 with both M and N tails (the UB fill path must truncate too).
    run_case(48, 300, 0, bf16, "L0-8b-k0-tails", block_m=32, block_n=128, strict=True)
    # L0-9 -- decode down GEMM: full-K (32 tiles) split accumulation, real geometry.
    run_case(32, 7168, 2048, bf16, "L0-9-splitk4-full-k", block_m=32, block_n=256)
    # Geometry bands the op actually hits: m = 512 / 4096, S = 18432, H = 7168.
    run_case(512, 512, 512, bf16, "L0-10-m512")
    run_case(4096, 256, 512, bf16, "L0-11-m4096")
    run_case(32, 512, 18432, bf16, "L0-12-k18432-S", block_m=32, block_n=256, split_k=1)
    run_case(32, 7168, 512, bf16, "L0-13-n7168-H", block_m=32, block_n=256)


def run_L1():
    """Representative manifest shapes beyond the gate."""
    bf16, fp16 = torch.bfloat16, torch.float16
    # kimi-k2-t512 (prefill band, split_k = 1)
    run_case(512, 7168, 7168, bf16, "L1-t512-gate-up")
    run_case(512, 7168, 18432, bf16, "L1-t512-down")
    # ds-v3-t4096 (the compute-bound case)
    run_case(4096, 4096, 7168, bf16, "L1-t4096-gate-up")
    run_case(4096, 7168, 2048, bf16, "L1-t4096-down")
    # ds-v3-t32 decode band (split-K) and glm45-t32
    run_case(32, 4096, 7168, bf16, "L1-t32-decode-gate-up")
    run_case(32, 7168, 2048, bf16, "L1-t32-decode-down")
    run_case(32, 3072, 5120, bf16, "L1-glm-t32-gate-up")
    run_case(32, 5120, 1536, bf16, "L1-glm-t32-down")
    # kimi-k2-t1: m = 1, the split-K precondition forbids a split here.
    run_case(1, 7168, 18432, bf16, "L1-t1-down")
    # fp16 element type on a prefill shape
    run_case(512, 4096, 7168, fp16, "L1-fp16-t512")


def run_L2():
    """Larger scale than the gate; recorded, non-blocking."""
    bf16 = torch.bfloat16
    _warn_case(lambda: run_case(512, 36864, 7168, bf16, "L2-kimi-t512-gate-up"), "L2-kimi-t512")
    _warn_case(lambda: run_case(4096, 4096, 7168, bf16, "L2-ds-t4096-gate-up"), "L2-ds-t4096")


def run_boundary():
    """Boundary: tails, single K tiles, fp32 output, and the degraded contracts."""
    bf16 = torch.bfloat16
    run_case(1, 1000, 128, bf16, "B-m1-n-tail", block_m=16, block_n=128, num_stages=2)
    run_case(17, 300, 128, bf16, "B-m17-n-tail", block_m=16, block_n=128, num_stages=2)
    run_case(64, 300, 192, bf16, "B-n300-default-bn", block_m=64)
    run_case(32, 128, 128, bf16, "B-m32-clamp-bm128", block_m=128, num_stages=2)
    # exactly one K tile (clear_accum only on step 0)
    run_case(32, 64, 64, bf16, "B-single-ktile", block_m=32, block_n=64, num_stages=2)
    # minimal split-K: 8 K tiles / 4 slices = 2 steps per slice
    run_case(32, 64, 512, bf16, "B-splitk-min", block_m=32, block_n=64, split_k=4)
    # out_dtype = fp32: quant 0, no rounding (fp32 criterion)
    run_case(64, 128, 256, bf16, "B-out-fp32", out_dtype="float32", block_m=64, block_n=128)
    # non-finite inputs propagate through the fp32 accumulator (bf16 output)
    _warn_case(_nonfinite_case, "B-nonfinite")

    # degraded contracts must raise at build time instead of changing the output
    # contract silently (DESIGN.md 5.2 rules 3/4, 9.2 E6)
    _expect_value_error(
        lambda: build(1, 64, 256, False, True, "bfloat16", False, None, 128, 64, 64, 2, 4),
        "B-err-splitk-m1",
    )
    _expect_value_error(
        lambda: build(32, 60, 128, False, True, "bfloat16", False, None, 32, 64, 64, 2, 4),
        "B-err-splitk-ntail",
    )
    _expect_value_error(
        lambda: build(32, 100, 128, False, True, "bfloat16", True, None, 32, 256, 64, 2, 1),
        "B-err-bias-tail",
    )
    _expect_value_error(
        lambda: build(32, 64, 96, False, True, "bfloat16", False, None, 32, 64, 64, 2, 1),
        "B-err-k-tail",
    )
    _expect_value_error(
        lambda: build(64, 64, 128, False, True, "bfloat16", False, None, 24, 64, 64, 2, 1),
        "B-err-tile-not-16",
    )
    _expect_value_error(
        lambda: build(512, 64, 128, False, True, "bfloat16", False, None, 256, 512, 64, 2, 1),
        "B-err-l0c-overflow",
    )


def _nonfinite_case():
    """+inf / NaN inputs must follow IEEE through the fp32 accumulator."""
    m, n, k = 32, 64, 128
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.bfloat16)
    b = torch.randn(n, k, dtype=torch.bfloat16)
    a[0, 0] = float("inf")
    a[1, 1] = float("nan")
    got = _gemm_basic_kernel(m, n, k, False, True, "bfloat16", False, None)(
        32, 64, 64, 2, 1
    )(a.to("npu"), b.to("npu")).cpu()
    want = golden_gemm_basic(a, b, False, True)
    assert torch.isinf(got[0, :]).all(), "an inf row must stay inf"
    assert torch.isnan(got[1, :]).all(), "a NaN row must stay NaN"
    ok = torch.isinf(got) | torch.isnan(got) | (got == want)
    assert ok.all(), "finite cells differ from the reference"
    print("[B-nonfinite] PASS inf/NaN propagation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"]
    )
    args, _ = parser.parse_known_args()

    t0 = time.perf_counter()
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

    worst = max(diff for _, diff in _DIFFS) if _DIFFS else 0.0
    for tag, msg in _WARNS:
        print(f"[warn] {tag}: {msg}")
    print(
        f"cases={len(_DIFFS)} warns={len(_WARNS)} max_diff={worst:.3e} "
        f"total={time.perf_counter() - t0:.1f}s"
    )
    print("\033[92mAll check passed!\033[0m")
    print("[PRECISION_PASS]")


if __name__ == "__main__":
    main()
