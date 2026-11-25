# Tuned tiling knobs of the Stage 4 kernel (module-level so the wrapper's
# baseline/perf_opt switch block can reference them, and so splice-time
# experiment branches can override them).
#   _TILE_ELEMS -- elements one sub-tile may hold (UB envelope; the compiler
#                  counts every multi-buffered version separately).
#   _STEP_BIG   -- rows one block walks when the call splits into few column
#                  chunks (a flat geometry needs taller blocks to fill 56 cores).
#   _STEP_SMALL -- rows one block walks when a row block splits into many chunks
#                  (the chunks already provide the parallelism, and the decode
#                  calls must keep enough blocks).
#   _SUB_DIV    -- sub-tile rows target: bt = step_target // _SUB_DIV (capped by
#                  the UB envelope and the wrapper's pair budget).
#   _TAB_VEC    -- per-lane vector width of the cos/sin row gather.
#   _ROT_VEC    -- per-lane vector width of the in-place rotate pass (1 = let the
#                  mapping pick; the SIMT backend caps it at 8 B anyway).
_TILE_ELEMS = 4096
_STEP_BIG = 32
_STEP_SMALL = 16
_SUB_DIV = 4
_TAB_VEC = 4
_ROT_VEC = 1


def _rope_positions_kernel(
    query_heads: int,
    key_heads: int,
    head_size: int,
    rotary_dim: int,
    max_position: int,
    dtype: str,
    position_dtype: str,
    offset_dtype: str | None,
    neox: bool,
    rot_vec: int,
    tail_vec: int,
    row_vec: int,
):
    """The frozen factory: ``(units_per_block, staged)`` returns the compiled program.

    Candidate ``c5`` (Stage 4 tuned): one UB buffer per block walks ``nsub``
    sub-tiles of ``bt`` rows in a two-stage ``T.Pipelined`` loop (``T.copy``
    GM->UB -> one in-place rotate pass -> ``T.copy`` UB->GM), and the cos/sin
    gather for the whole block span is hoisted out of that loop and moved
    ``_TAB_VEC`` columns per lane.

    Structure comes from the measurements in ``perf_opt/opt_log.md``:

    * GM traffic over ``T.copy`` instead of per-lane vector accesses: the same
      payload costs 322 us per-lane (8 B) / 245 us (16 B) vs 124 us over the DMA
      path on 950PR, so the tile must not be moved by the vector pipe.
    * One UB buffer rotated in place (Stage 3 used two buffers and five UB
      passes for one read + one write), and the copied tail columns need no
      handling at all because they are never touched.
    * Bigger blocks: the per-block fixed cost dominates small tiles (a 4-row
      block: 292 us, 16 rows: 206 us, 32 rows: 150 us, 64 rows: 142 us on the
      llama3-8b-b2-2k case), but a block taller than the call's row count would
      starve the 56 vector cores on the decode cases, so the row step is chosen
      from the number of column chunks: few chunks -> taller blocks.
    * The table gather (two guarded 4 B per-lane GM reads with an int64 row
      scale) is the largest single remaining cost when done per sub-tile;
      hoisting it to the block and widening it to 4 columns per lane takes the
      dominant case from 195.7 us to 150.4 us.

    ``units_per_block`` stays the wrapper's pair budget (2048) and is used only
    as a floor for the sub-tile pair count; the tiling itself follows the tuned
    constants above.  ``rot_vec`` / ``tail_vec`` / ``row_vec`` selected CUDA
    access widths and are accepted but unused here; ``staged`` accepts both
    values and compiles this one program.
    """
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > head_size:
        raise ValueError("rotary_dim must be positive, even and <= head_size")
    if query_heads <= 0 or key_heads < 0:
        raise ValueError("query_heads must be positive and key_heads not negative")
    if max_position <= 0:
        raise ValueError("max_position must be positive")
    if dtype not in _TORCH_DTYPE:
        raise ValueError(f"dtype: {dtype!r} is not one of {tuple(_TORCH_DTYPE)}")
    if position_dtype not in ("int32", "int64"):
        raise ValueError(f"position_dtype: {position_dtype!r} is not int32 or int64")
    if offset_dtype not in (None, "int32", "int64"):
        raise ValueError(f"offset_dtype: {offset_dtype!r} is not int32, int64 or None")
    if offset_dtype is not None and not key_heads:
        raise ValueError("offsets: an offsets call also carries a key")

    dim, rotary = head_size, rotary_dim
    half = rotary // 2
    hpc_cap = max(1, 1024 // dim)

    # A 2D UB buffer's row pitch must be a whole number of 32 B sectors, or the
    # device rejects the ``T.copy`` with error 507035 (no compile diagnostic).
    # ``head_size``/``rotary_dim`` may be odd or non-pow2 (65 / 24 / 2), so the
    # staging buffers are padded to the sector while every copy and every lane
    # access still works on the logical ``cols`` / ``half`` extent.
    _sector = 32 // _TORCH_DTYPE[dtype].itemsize

    def _pad(n):
        return max(_sector, ((n + _sector - 1) // _sector) * _sector)

    def _vf_threads(extent):
        """Threads for the in-place rotate VF: the largest power of two <= 256 that
        divides the pass's lane count.

        The rotate pass reads and writes one UB buffer, so the pipeline puts a
        shared-memory sync into it; such a sync can only be lowered when every
        participating warp is fully active, i.e. when the lane count is a whole
        number of warps.  Benchmark geometries have 512..2048 lanes per pass and
        keep 256 threads; odd ones (one key head with rotary_dim 24 -> 48 lanes)
        would otherwise leave a half warp and fail to compile (ThreadSync
        "participating threads are not warp-uniform").
        """
        t = _THREADS
        while t > 1 and extent % t:
            t //= 2
        return t

    def _hpc(heads):
        return max(d for d in range(1, min(hpc_cap, heads) + 1) if heads % d == 0)

    hpc_q = _hpc(query_heads)
    hpc_k = _hpc(key_heads) if key_heads else 0
    cols_q = hpc_q * dim
    cols_k = hpc_k * dim
    cols_max = max(cols_q, cols_k)
    nch_q = query_heads // hpc_q
    nch_k = key_heads // hpc_k if key_heads else 0
    q_width, k_width = query_heads * dim, key_heads * dim
    has_off = offset_dtype is not None
    out_idx = [6, 7] if has_off else [5, 6] if key_heads else [4]
    tvec = _TAB_VEC if (_TAB_VEC > 1 and half % _TAB_VEC == 0) else 1
    rvec = _ROT_VEC if (neox and _ROT_VEC > 1 and half % _ROT_VEC == 0) else 1

    def _program(units_per_block, staged):
        # Rows one block walks: a row block splits into ``nch_q + nch_k`` column
        # chunks, so a geometry with few chunks needs taller blocks to fill the
        # 56 vector cores -- and one with many chunks must keep the blocks small
        # or the decode calls starve.
        units = nch_q + nch_k
        step_target = _STEP_SMALL if units > 5 else _STEP_BIG
        ub_rows = max(1, _TILE_ELEMS // cols_max)
        budget_rows = max(1, int(units_per_block) // (hpc_q * half))
        bt = max(1, min(step_target // _SUB_DIV, ub_rows, max(budget_rows, 1)))
        nsub = max(1, (step_target + bt - 1) // bt)
        step = bt * nsub  # rows one block walks
        cols_pad = _pad(cols_max)
        half_pad = _pad(half)
        tokens = T.dynamic("num_tokens")
        nrb = T.ceildiv(tokens, step)
        nq = nrb * nch_q
        nblocks = nq + nrb * nch_k

        @T.macro
        def side(X, Y, POS, OFF, COS, SIN, row_ub, cs_ub, sn_ub, x_ub, hpc, cols, rthreads, rb, cb):
            """One tensor's column chunk at ``cb`` over ``nsub`` sub-tiles from ``rb``."""
            c0 = cb * cols
            base = rb * step
            with T.SimtVF(threads=_THREADS):
                for i in T.Parallel(step):
                    r = T.min(base + i, tokens - 1)
                    seen = T.cast(POS[r], "int64")
                    if has_off:
                        seen = seen + T.cast(OFF[r], "int64")
                    row_ub[i] = T.cast(T.max(0, T.min(seen, max_position - 1)), "int32")
                if tvec > 1:
                    for i, kk in T.Parallel(step, half // tvec):
                        for v in T.vectorized(tvec):
                            cs_ub[i, kk * tvec + v] = COS[row_ub[i], kk * tvec + v]
                    for i, kk in T.Parallel(step, half // tvec):
                        for v in T.vectorized(tvec):
                            sn_ub[i, kk * tvec + v] = SIN[row_ub[i], kk * tvec + v]
                else:
                    for i, k in T.Parallel(step, half):
                        cs_ub[i, k] = COS[row_ub[i], k]
                    for i, k in T.Parallel(step, half):
                        sn_ub[i, k] = SIN[row_ub[i], k]
            for s in T.Pipelined(nsub, num_stages=2):
                row0 = base + s * bt
                nr = T.min(bt, tokens - row0)
                T.copy(X[row0 : row0 + nr, c0 : c0 + cols], x_ub[0:nr, 0:cols])
                with T.SimtVF(threads=rthreads):
                    if rvec > 1:
                        for i, h, kk in T.Parallel(bt, hpc, half // rvec):
                            for v in T.vectorized(rvec):
                                k = kk * rvec + v
                                lo = h * dim + k
                                hi = lo + half
                                cs = T.cast(cs_ub[s * bt + i, k], "float32")
                                sn = T.cast(sn_ub[s * bt + i, k], "float32")
                                x_lo = T.cast(x_ub[i, lo], "float32")
                                x_hi = T.cast(x_ub[i, hi], "float32")
                                x_ub[i, lo] = T.cast(x_lo * cs - x_hi * sn, dtype)
                                x_ub[i, hi] = T.cast(x_hi * cs + x_lo * sn, dtype)
                    else:
                        for i, p in T.Parallel(bt, hpc * half):
                            h = p // half
                            k = p - h * half
                            lo = h * dim + (k if neox else 2 * k)
                            hi = lo + (half if neox else 1)
                            cs = T.cast(cs_ub[s * bt + i, k], "float32")
                            sn = T.cast(sn_ub[s * bt + i, k], "float32")
                            x_lo = T.cast(x_ub[i, lo], "float32")
                            x_hi = T.cast(x_ub[i, hi], "float32")
                            x_ub[i, lo] = T.cast(x_lo * cs - x_hi * sn, dtype)
                            x_ub[i, hi] = T.cast(x_hi * cs + x_lo * sn, dtype)
                T.copy(x_ub[0:nr, 0:cols], Y[row0 : row0 + nr, c0 : c0 + cols])

        if has_off:

            @T.prim_func
            def main(
                positions: T.Tensor((tokens,), position_dtype),
                query: T.Tensor((tokens, q_width), dtype),
                key: T.Tensor((tokens, k_width), dtype),
                offsets: T.Tensor((tokens,), offset_dtype),
                cos_table: T.Tensor((max_position, half), dtype),
                sin_table: T.Tensor((max_position, half), dtype),
                query_out: T.Tensor((tokens, q_width), dtype),
                key_out: T.Tensor((tokens, k_width), dtype),
            ) -> None:
                with T.Kernel(nblocks) as bx:
                    x_ub = T.alloc_shared((bt, cols_pad), dtype)
                    cs_ub = T.alloc_shared((step, half_pad), dtype)
                    sn_ub = T.alloc_shared((step, half_pad), dtype)
                    row_ub = T.alloc_shared((step,), "int32")
                    if bx < nq:
                        side(
                            query, query_out, positions, offsets, cos_table, sin_table,
                            row_ub, cs_ub, sn_ub, x_ub, hpc_q, cols_q, _vf_threads(bt * hpc_q * half),
                            bx // nch_q, bx % nch_q,
                        )
                    else:
                        t = bx - nq
                        side(
                            key, key_out, positions, offsets, cos_table, sin_table,
                            row_ub, cs_ub, sn_ub, x_ub, hpc_k, cols_k, _vf_threads(bt * hpc_k * half),
                            t // nch_k, t % nch_k,
                        )

        elif key_heads:

            @T.prim_func
            def main(
                positions: T.Tensor((tokens,), position_dtype),
                query: T.Tensor((tokens, q_width), dtype),
                key: T.Tensor((tokens, k_width), dtype),
                cos_table: T.Tensor((max_position, half), dtype),
                sin_table: T.Tensor((max_position, half), dtype),
                query_out: T.Tensor((tokens, q_width), dtype),
                key_out: T.Tensor((tokens, k_width), dtype),
            ) -> None:
                with T.Kernel(nblocks) as bx:
                    x_ub = T.alloc_shared((bt, cols_pad), dtype)
                    cs_ub = T.alloc_shared((step, half_pad), dtype)
                    sn_ub = T.alloc_shared((step, half_pad), dtype)
                    row_ub = T.alloc_shared((step,), "int32")
                    if bx < nq:
                        side(
                            query, query_out, positions, positions, cos_table, sin_table,
                            row_ub, cs_ub, sn_ub, x_ub, hpc_q, cols_q, _vf_threads(bt * hpc_q * half),
                            bx // nch_q, bx % nch_q,
                        )
                    else:
                        t = bx - nq
                        side(
                            key, key_out, positions, positions, cos_table, sin_table,
                            row_ub, cs_ub, sn_ub, x_ub, hpc_k, cols_k, _vf_threads(bt * hpc_k * half),
                            t // nch_k, t % nch_k,
                        )

        else:

            @T.prim_func
            def main(
                positions: T.Tensor((tokens,), position_dtype),
                query: T.Tensor((tokens, q_width), dtype),
                cos_table: T.Tensor((max_position, half), dtype),
                sin_table: T.Tensor((max_position, half), dtype),
                query_out: T.Tensor((tokens, q_width), dtype),
            ) -> None:
                with T.Kernel(nblocks) as bx:
                    x_ub = T.alloc_shared((bt, cols_pad), dtype)
                    cs_ub = T.alloc_shared((step, half_pad), dtype)
                    sn_ub = T.alloc_shared((step, half_pad), dtype)
                    row_ub = T.alloc_shared((step,), "int32")
                    side(
                        query, query_out, positions, positions, cos_table, sin_table,
                        row_ub, cs_ub, sn_ub, x_ub, hpc_q, cols_q, _vf_threads(bt * hpc_q * half),
                        bx // nch_q, bx % nch_q,
                    )

        return tilelang.compile(main, target="ascend", out_idx=out_idx)

    return _program


