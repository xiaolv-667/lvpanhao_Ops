# Round-3 knobs (module level so splice.py --constants can branch on them).
#   _SUBTILES -- sub-tiles a block walks in a two-stage ``T.Pipelined`` loop.
#   _TAB_VEC  -- per-lane vector width of the cos/sin row gather.
#   _ROT_VEC  -- per-lane vector width of the in-place rotate pass (NeoX only).
_SUBTILES = globals().get("_SUBTILES", 8)
_TAB_VEC = globals().get("_TAB_VEC", 4)
_ROT_VEC = globals().get("_ROT_VEC", 1)


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

    Candidate ``c4`` (Stage 4): one UB buffer per block walks ``_SUBTILES``
    sub-tiles of ``bt`` rows in a two-stage ``T.Pipelined`` loop (DMA in ->
    in-place rotate -> DMA out), and -- unlike ``c3`` -- the table gather for the
    **whole block span** is hoisted out of that loop and widened to ``_TAB_VEC``
    columns per lane.

    Why: the per-row cos/sin gather (two 4 B per-lane GM reads with an int64 row
    scale, guarded, then a block barrier) is the largest remaining cost of this
    kernel on 950PR -- measured on the llama3-8b-b2-2k case, removing it takes
    195.7 us to 134.4 us while removing the whole rotate pass only takes it to
    158.9 us.  Gathering it once per block instead of once per sub-tile removes
    the repeated un-hidden GM latency, and the 4-wide run removes 3/4 of the
    gather instructions.

    ``bt`` is the per-sub-tile row count off the wrapper's pair budget:
    ``bt = max(1, min(units_per_block // (hpc * half), _TILE_ELEMS // cols))``;
    a block spans ``bt * _SUBTILES`` rows.  ``rot_vec`` / ``tail_vec`` /
    ``row_vec`` selected CUDA access widths and are accepted but unused here;
    ``staged`` accepts both values and compiles this one program.
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
    nsub = int(_SUBTILES)
    tvec = _TAB_VEC if (_TAB_VEC > 1 and half % _TAB_VEC == 0) else 1
    rvec = _ROT_VEC if (neox and _ROT_VEC > 1 and half % _ROT_VEC == 0) else 1

    def _program(units_per_block, staged):
        bt = max(
            1,
            min(
                int(units_per_block) // (hpc_q * half),
                max(1, _TILE_ELEMS // cols_max),
            ),
        )
        step = bt * nsub  # rows one block walks
        tokens = T.dynamic("num_tokens")
        nrb = T.ceildiv(tokens, step)
        nq = nrb * nch_q
        nblocks = nq + nrb * nch_k

        @T.macro
        def side(X, Y, POS, OFF, COS, SIN, row_ub, cs_ub, sn_ub, x_ub, hpc, cols, rb, cb):
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
                with T.SimtVF(threads=_THREADS):
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
                    x_ub = T.alloc_shared((bt, cols_max), dtype)
                    cs_ub = T.alloc_shared((step, half), dtype)
                    sn_ub = T.alloc_shared((step, half), dtype)
                    row_ub = T.alloc_shared((step,), "int32")
                    if bx < nq:
                        side(
                            query, query_out, positions, offsets, cos_table, sin_table,
                            row_ub, cs_ub, sn_ub, x_ub, hpc_q, cols_q,
                            bx // nch_q, bx % nch_q,
                        )
                    else:
                        t = bx - nq
                        side(
                            key, key_out, positions, offsets, cos_table, sin_table,
                            row_ub, cs_ub, sn_ub, x_ub, hpc_k, cols_k,
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
                    x_ub = T.alloc_shared((bt, cols_max), dtype)
                    cs_ub = T.alloc_shared((step, half), dtype)
                    sn_ub = T.alloc_shared((step, half), dtype)
                    row_ub = T.alloc_shared((step,), "int32")
                    if bx < nq:
                        side(
                            query, query_out, positions, positions, cos_table, sin_table,
                            row_ub, cs_ub, sn_ub, x_ub, hpc_q, cols_q,
                            bx // nch_q, bx % nch_q,
                        )
                    else:
                        t = bx - nq
                        side(
                            key, key_out, positions, positions, cos_table, sin_table,
                            row_ub, cs_ub, sn_ub, x_ub, hpc_k, cols_k,
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
                    x_ub = T.alloc_shared((bt, cols_max), dtype)
                    cs_ub = T.alloc_shared((step, half), dtype)
                    sn_ub = T.alloc_shared((step, half), dtype)
                    row_ub = T.alloc_shared((step,), "int32")
                    side(
                        query, query_out, positions, positions, cos_table, sin_table,
                        row_ub, cs_ub, sn_ub, x_ub, hpc_q, cols_q,
                        bx // nch_q, bx % nch_q,
                    )

        return tilelang.compile(main, target="ascend", out_idx=out_idx)

    return _program


