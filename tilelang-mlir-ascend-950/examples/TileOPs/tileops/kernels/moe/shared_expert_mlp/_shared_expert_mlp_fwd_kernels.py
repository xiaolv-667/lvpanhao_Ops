"""Extracted TileLang kernel implementation (curated reference).

Op: ``SharedExpertMLPFwdOp`` — the shared expert of an MoE layer, a dense gated
MLP ``down(silu(hidden @ gate.T) * (hidden @ up.T))`` with gate and up stacked
in ``w_gate_up``.

Provenance: ``extract_tl_kernel.py --op-name SharedExpertMLPFwdOp
--gpu-repo-root /workspace/gpu_mirror_950pr`` (resolution pattern B) pulled the
transitive closure of the GPU ``SharedExpertMLPKernel``.  That kernel does not
own a fused TileLang program: it *decomposes* the expert into child GEMM
kernels of the generic dense-GEMM library (``GEMMCpAsyncKernel`` /
``GEMMTMAKernel`` / ``GEMMTemplate``) plus its own activation kernel, so the
raw closure is the whole SM90 GEMM zoo (eleven dense GEMM variants, the split-K
reduction, a grouped GEMM, and the trace API) and does not import (its
``trace = _Trace()`` constant has no definition in the emitted file).  This
file keeps the kernels the op actually runs through that decomposition,
verbatim:

  - ``_gemm_basic_kernel`` — the portable (non-warp-specialized) pipelined
    dense GEMM, i.e. the path the op takes on a target without SM90's
    TMA/WGMMA.  It serves both the gate/up GEMM (``n = 2S``, ``k = H``) and the
    down GEMM (``n = H``, ``k = S``), both ``trans_b=True``.
  - ``splitk_reduce_kernel`` — the split-K epilogue the op uses in the decode
    band (``T <= 32``), with the ``silu_and_mul`` activation fused for the
    gate/up product.
  - ``_silu_mul_fused_kernel`` — the op's own activation kernel,
    ``gate_up[M, 2N] -> silu(gate) * up[M, N]``.
  - ``_add_bias`` / ``_main`` — the two helpers ``_gemm_basic_kernel`` and
    ``splitk_reduce_kernel`` call, copied verbatim from
    ``tileops/_kernels/gemm/dense.py`` so this file is self-contained.

The dropped variants (``_gemm_kernel``, ``_gemm_coop2_kernel``,
``_gemm_pingpong_kernel``, ``_gemm_simple_kernel``, ``_gemm_stream_kernel``,
``_gemm_swap_ab_kernel``, ``_gemm_coop2s_kernel``, ``_gemm_coop2_splitk_kernel``,
``_gemm_splitk_kernel``, ``_gemm_coop2_swap_kernel``, ``_grouped_gemm_kernel``)
are generic library kernels reached only through the SM90 dispatch of
``GEMMTMAKernel`` / ``GEMMTemplate``; none of them belongs to this op, and the
selection policy that picks among them is carried by the kernel class in
``shared_expert_mlp.py``.

These are the GPU (CUDA-target) implementations.  The NPU kernel component
reimplements them for the selected NPU target: K1 (decorator/target), K2
(grid + sync), K3 (``threads`` removed from the factory's inner callable) and
K4 (padding/masking) apply per function.
"""

# --- external imports (preserved) ---
from collections.abc import Callable

import tilelang
import tilelang.language as T
import math

# --- module-level constants referenced by the kernels ---
# None: the retained kernels read no module constant (the dropped variants read
# ``STR_TO_DTYPE`` / ``SWAP_AB_MPAD`` / the trace API; none of those is kept).

# --- helper functions called by the kernels (copied verbatim) ---
@T.macro
def _add_bias(acc, bias, start, n, *, by_row=False, helper_layout=False):
    """Add FP32 column bias, or row bias for a transposed output tile.

    Out-of-range lanes read the last bias and are excluded from output stores.
    ``helper_layout`` uses the 64-row WGMMA fragment layout; its C++ helper
    supports warpgroups outside threads ``[0, 128)``, unlike TileLang 0.1.12's
    ``T.Parallel`` lowering for these fragments."""
    rows, cols = acc.shape
    if helper_layout:
        T.call_extern(
            "handle",
            f"tileops::dense_add_{'row' if by_row else 'col'}_bias_64xN<{int(cols)}>",
            acc.data,
            T.address_of(bias[0]),
            start,
            n,
        )
    else:
        for i, j in T.Parallel(rows, cols):
            acc[i, j] += T.cast(bias[T.min(start + (i if by_row else j), n - 1)], acc.dtype)
def _main(
    name: str,
    body: Callable,
    operands: tuple,
    c: tuple,
    bias: tuple | None = None,
) -> Callable:
    """The program running ``body(*operands, bias, c)``, with ``bias`` a parameter
    between the operands and C, or ``None`` when *bias* is ``None``.

    *operands* holds one ``(shape, dtype)`` pair, the parameter ``w``, or two, ``a`` and
    ``b``; *c* and *bias* are one pair each. The program's symbol is ``{name}_bias_main``
    with a bias, else ``{name}_main``.
    """
    c_shape, c_dtype = c
    if len(operands) == 1:
        ((w_shape, w_dtype),) = operands
        if bias is None:

            @T.prim_func
            def main(
                w: T.Tensor(w_shape, w_dtype),  # type: ignore
                c: T.Tensor(c_shape, c_dtype),  # type: ignore
            ) -> None:
                body(w, None, c)

        else:
            bias_shape, bias_dtype = bias

            @T.prim_func
            def main(
                w: T.Tensor(w_shape, w_dtype),  # type: ignore
                bias: T.Tensor(bias_shape, bias_dtype),  # type: ignore
                c: T.Tensor(c_shape, c_dtype),  # type: ignore
            ) -> None:
                body(w, bias, c)

    else:
        (a_shape, a_dtype), (b_shape, b_dtype) = operands
        if bias is None:

            @T.prim_func
            def main(
                a: T.Tensor(a_shape, a_dtype),  # type: ignore
                b: T.Tensor(b_shape, b_dtype),  # type: ignore
                c: T.Tensor(c_shape, c_dtype),  # type: ignore
            ) -> None:
                body(a, b, None, c)

        else:
            bias_shape, bias_dtype = bias

            @T.prim_func
            def main(
                a: T.Tensor(a_shape, a_dtype),  # type: ignore
                b: T.Tensor(b_shape, b_dtype),  # type: ignore
                bias: T.Tensor(bias_shape, bias_dtype),  # type: ignore
                c: T.Tensor(c_shape, c_dtype),  # type: ignore
            ) -> None:
                body(a, b, bias, c)

    symbol = f"{name}_main" if bias is None else f"{name}_bias_main"
    return main.with_attr("global_symbol", symbol)

# --- TileLang kernel functions (@tilelang.jit) ---
def _gemm_basic_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    has_bias: bool = False,
    out_dtype: str | None = None,
) -> Callable:
    """Pipelined dense GEMM ``C = op(A) @ op(B)`` for any tensor-core target.

    The non-warp-specialized counterpart of ``_gemm_kernel``: the same four
    ``(trans_a, trans_b)`` layouts, but with ``T.Pipelined`` software
    pipelining and plain ``T.gemm`` so it compiles on pre-SM90 targets
    (sm80 / sm86 / sm89). SMEM tile shapes follow the storage layout and the
    ``T.gemm`` transpose flags reconcile them with the logical (M, K) x
    (K, N) contraction (cf. the WGMMA version, which forwards them to WGMMA).
    *has_bias* builds a ``split_k == 1`` program taking ``bias[n]`` before C; split
    slices leave the bias to the reduction. A ``split_k == 1`` C is *out_dtype*, or
    *dtype* when it is ``None``.
    """
    accum_dtype = "float"
    c_dtype = out_dtype or dtype

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_basic_func(
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        num_stages: int = 2,
        threads: int = 128,
        split_k: int = 1,
    ) -> Callable:
        # Built here, not closed over: the autotuner refuses a closure that holds a tuple.
        a_shape = (k, m) if trans_a else (m, k)
        b_shape = (n, k) if trans_b else (k, n)
        # SMEM tile shapes follow the storage layout; the T.gemm transpose
        # flags reconcile them with the logical (M,K) x (K,N) contraction.
        a_tile = (block_k, block_m) if trans_a else (block_m, block_k)
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)
        n_exact = n % block_n == 0
        k_tiles = T.ceildiv(k, block_k)
        if split_k < 1 or k_tiles % split_k:
            raise ValueError("split_k must divide the K tile count")
        if split_k > 1 and (m % block_m or not n_exact or k % block_k):
            raise ValueError("split-K basic GEMM requires exact M/N/K tiles")
        k_slice = k_tiles // split_k
        output_shape = (m, n) if split_k == 1 else (split_k, m, n)
        output_dtype = c_dtype if split_k == 1 else accum_dtype

        @T.macro
        def _gemm_basic_body(a, b, bias, c):
            with T.Kernel(
                T.ceildiv(n, block_n), T.ceildiv(m, block_m), split_k, threads=threads
            ) as (
                bx,
                by,
                bz,
            ):
                a_smem = T.alloc_shared(a_tile, dtype)
                b_smem = T.alloc_shared(b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                if split_k == 1:
                    c_smem = T.alloc_shared((block_m, block_n), c_dtype)
                    T.annotate_layout(
                        {
                            a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                            b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                            c_smem: tilelang.layout.make_swizzled_layout(c_smem),
                        }
                    )
                else:
                    T.annotate_layout(
                        {
                            a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                            b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        }
                    )

                # L2 rasterization: same panel traversal as the BMM kernel.
                T.use_swizzle(10, enable=True)

                T.clear(c_local)
                m_start = by * block_m
                n_start = bx * block_n

                for ki in T.Pipelined(k_slice, num_stages=num_stages):
                    k_start = (bz * k_slice + ki) * block_k
                    # M/N tail reads land in c_local rows/cols that the
                    # epilogue guard skips; a K tail would corrupt live
                    # outputs, so block_k must divide k (enforced by the
                    # config chain / autotune filter).
                    if trans_a:
                        T.copy(
                            a[k_start : k_start + block_k, m_start : m_start + block_m],
                            a_smem,
                        )
                    else:
                        T.copy(
                            a[m_start : m_start + block_m, k_start : k_start + block_k],
                            a_smem,
                        )
                    if trans_b:
                        T.copy(
                            b[n_start : n_start + block_n, k_start : k_start + block_k],
                            b_smem,
                            eviction_policy="evict_first" if split_k > 1 else None,
                        )
                    elif n_exact:
                        T.copy(
                            b[k_start : k_start + block_k, n_start : n_start + block_n],
                            b_smem,
                        )
                    else:
                        # NN with an N tail: the b tile's N rows can be narrower than
                        # the 4/8/16-byte cp_async accesses, so mask the copy instead.
                        for i, j in T.Parallel(block_k, block_n):
                            b_smem[i, j] = T.if_then_else(
                                n_start + j < n,
                                b[k_start + i, n_start + j],
                                T.cast(0, dtype),
                            )
                    T.gemm(
                        a_smem,
                        b_smem,
                        c_local,
                        transpose_A=trans_a,
                        transpose_B=trans_b,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                if split_k == 1:
                    if has_bias:
                        _add_bias(c_local, bias, n_start, n)
                    T.copy(c_local, c_smem)
                    for i, j in T.Parallel(block_m, block_n):
                        if m_start + i < m and n_start + j < n:
                            c[m_start + i, n_start + j] = c_smem[i, j]
                else:
                    T.copy(
                        c_local,
                        c[
                            bz,
                            m_start : m_start + block_m,
                            n_start : n_start + block_n,
                        ],
                    )

        return _main(
            "_gemm_basic",
            _gemm_basic_body,
            ((a_shape, dtype), (b_shape, dtype)),
            (output_shape, output_dtype),
            ((n,), dtype) if has_bias and split_k == 1 else None,
        )

    return _gemm_basic_func
def splitk_reduce_kernel(
    split_k: int,
    m: int,
    n: int,
    dtype: str = "float16",
    activation: str = "none",
    has_bias: bool = False,
    out_dtype: str | None = None,
) -> Callable:
    """Reduce the split-K fp32 workspace into the final output.

    Sums ``w[split_k, m, n]`` over the slice axis in fp32, adds ``bias[n]`` (in *dtype*)
    under *has_bias*, and casts to *out_dtype*, or *dtype* when it is ``None``, at the
    boundary. Bandwidth-trivial elementwise kernel; the workspace of the shapes worth
    splitting fits in L2.

    ``C`` is a parameter, not a JIT-allocated output: the caller allocates it before
    the mainloop, so no allocation sits between the two launches.
    """
    accum_dtype = "float"
    c_dtype = out_dtype or dtype
    if activation not in ("none", "silu_and_mul"):
        raise ValueError(f"unsupported split-K epilogue: {activation}")
    gated = activation == "silu_and_mul"
    if gated and n % 2:
        raise ValueError("silu_and_mul requires an even output width")
    out_n = n // 2 if gated else n

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: gated},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _splitk_reduce_func(elems_per_cta: int = 1024) -> Callable:
        def _slice_sum(w, bias, gi, gj):
            expr = w[0, gi, gj]
            for s in range(1, split_k):
                expr = expr + w[s, gi, gj]
            if has_bias:
                expr = expr + T.cast(bias[gj], accum_dtype)
            return expr

        @T.macro
        def _splitk_reduce_body(w, bias, c):
            total = m * out_n
            with T.Kernel(T.ceildiv(total, elems_per_cta), threads=256) as bx:
                base = bx * elems_per_cta
                for t in T.Parallel(elems_per_cta):
                    idx = base + t
                    if idx < total:
                        gi = idx // out_n
                        gj = idx % out_n
                        if gated:
                            gate = _slice_sum(w, bias, gi, gj)
                            up = _slice_sum(w, bias, gi, out_n + gj)
                            c[gi, gj] = T.cast(gate * T.sigmoid(gate) * up, c_dtype)
                        else:
                            c[gi, gj] = T.cast(_slice_sum(w, bias, gi, gj), c_dtype)

        return _main(
            "_splitk_reduce",
            _splitk_reduce_body,
            (((split_k, m, n), accum_dtype),),
            ((m, out_n), c_dtype),
            ((n,), dtype) if has_bias else None,
        )

    return _splitk_reduce_func
def _silu_mul_fused_kernel(M: int, N: int, dtype_str: str):
    """Map ``gate_up[M, 2N]`` to ``silu(gate) * up[M, N]``."""
    dtype = dtype_str

    @tilelang.jit(
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_m, block_n, threads):
        gate_up_shape = (M, N * 2)
        out_shape = (M, N)

        @T.prim_func
        def _main(
            gate_up: T.Tensor(gate_up_shape, dtype),
            out: T.Tensor(out_shape, dtype),
        ):
            num_blocks_m = math.ceil(M / block_m)
            num_blocks_n = math.ceil(N / block_n)
            with T.Kernel(num_blocks_m * num_blocks_n, threads=threads) as (bx,):
                bx_m = bx // num_blocks_n
                bx_n = bx % num_blocks_n
                m_start = bx_m * block_m
                n_start = bx_n * block_n

                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < M and n_start + j < N:
                        g = gate_up[m_start + i, n_start + j]
                        u = gate_up[m_start + i, n_start + j + N]
                        g_f32 = T.cast(g, "float32")
                        u_f32 = T.cast(u, "float32")
                        sigmoid_g = T.sigmoid(g_f32)
                        result = g_f32 * sigmoid_g * u_f32
                        out[m_start + i, n_start + j] = T.cast(result, dtype)

        return _main

    return _func

__all__ = ["_gemm_basic_kernel", "_silu_mul_fused_kernel", "splitk_reduce_kernel"]
