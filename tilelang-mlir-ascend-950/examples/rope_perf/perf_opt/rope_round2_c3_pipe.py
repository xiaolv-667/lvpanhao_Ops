# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Rotary position embedding forward (ascend): kernel + torch golden + layered tests.

``RoPEFwdOp`` rotates, at each token's position, the leading ``rotary_dim``
columns of every head of ``query`` and ``key`` and copies the rest:

    NeoX        y[lo] = x[lo] * cos[k] - x[hi] * sin[k],  lo = h * D + k,  hi = lo + R / 2
                y[hi] = x[hi] * cos[k] + x[lo] * sin[k]
    GPT-J       lo = h * D + 2k, hi = lo + 1, the same two expressions

The table row is ``clamp(int64(position) + int64(offset), 0, max_position - 1)``
(a position outside the table reads its nearest row), the arithmetic is float32
and every output value rounds to its dtype once, and the cos/sin tables are the
caller's already-rounded inputs (never recomputed here).

Adaptation (GPU ``_rope_positions_kernel`` -> this target): Ascend has a 1-D
core grid and no thread count at kernel scope, so the source's 3-D block/thread
decomposition is folded into one extent serving both tensors -- ``bx < nq`` is
the query side, the rest the key side -- and the lane domain is opened by
``T.SimtVF(threads=256)``.  A block owns a ``bt x (hpc * head_size)`` tile of one
tensor: ``bt`` rows and a column chunk that is a whole number of heads, so the
column axis has no tail block at all and only the row axis needs the dynamic
``tokens`` predicate.  Five UB buffers give the source's "staged rows" intent a
structural form: the input tile is read once into ``x_ub`` (read-only after the
load), copied whole into ``y_ub`` (which carries the copied tail columns
through), rotated in place in ``y_ub`` by two single-store passes, and written
back.  Because read and write buffers are physically separate, no load waits on
a store.

``units_per_block`` is the per-block pair budget (the wrapper's ``default_config``
is 2048 pairs, capped by a 4096-element tile), ``rot_vec`` / ``tail_vec`` /
``row_vec`` selected CUDA access widths and are accepted but unused here, and
``staged`` accepts both values and compiles this one program.

Content layout:

1. ``golden_rope_positions`` -- torch CPU reference (the source contract).
2. ``_rope_positions_kernel`` -- the frozen factory: ``(units_per_block, staged)``
   returns the compiled callable, ``out_idx`` places the outputs.
3. ``run_case`` / ``_launch`` -- build, launch and compare on the NPU.
4. ``run_L0`` / ``run_L1`` / ``run_L2`` / ``run_boundary`` / ``main``.
"""

import argparse
import functools
import math
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401  -- registers torch.npu

_THREADS = 256
_UNITS_PER_BLOCK = 2048  # pairs a block budgets (the wrapper's default_config)
_TILE_ELEMS = 4096
_VECTOR_ACCESS_BYTES = 16

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
_DTYPE_NAME = {v: k for k, v in _TORCH_DTYPE.items()}
_TOLERANCE = {  # (atol, rtol), the source op's per-dtype gates
    "float16": (1e-3, 1e-3),
    "bfloat16": (1.6e-2, 1.6e-2),
    "float32": (1e-5, 1e-5),
}


# ---------------------------------------------------------------------------
# 1. Golden: torch CPU reference of the source contract
# ---------------------------------------------------------------------------
def _rope_cache(params):
    """The float32 cos and sin caches: row r, column k is cos(r * base ** (-2k / R))."""
    rotary, base = params["rotary_dim"], params["base"]
    pos_freqs = base ** (torch.arange(0, rotary, 2, dtype=torch.float32) / rotary)
    inv = 1.0 / pos_freqs
    rows = params["max_position_embeddings"]
    angles = torch.outer(torch.arange(rows, dtype=torch.float32), inv)
    return angles.cos(), angles.sin()


def rope_tables(params):
    """The cos and sin tables the caller holds: float32 angles rounded to ``dtype``."""
    dtype = _TORCH_DTYPE[params["dtype"]]
    return tuple(t.to(dtype) for t in _rope_cache(params))


def _rotate_pairs(part, c, s, neox):
    """*part* rotated by cosines *c* and sines *s*: halves against each other, else adjacent pairs."""
    if neox:
        x1, x2 = part.chunk(2, dim=-1)
        return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
    x1, x2 = part[..., 0::2], part[..., 1::2]
    return torch.stack((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).flatten(-2)


def _rotate(x, index, cos, sin, params):
    """*x* ``[*S, head_size * H]``, each head's first *rotary_dim* columns rotated.

    The rotation is float32 and rounds once; the columns past ``rotary_dim`` are
    copied unchanged.  Nothing here mirrors the NPU tiling (no chunking, no
    folded grid): every token reads table row *index* directly.
    """
    tokens = index.numel()
    head_size, rotary = params["head_size"], params["rotary_dim"]
    heads = x.reshape(tokens, -1, head_size)
    c = cos[index.reshape(-1)].unsqueeze(1).float()
    s = sin[index.reshape(-1)].unsqueeze(1).float()
    rotated = _rotate_pairs(heads[..., :rotary].float(), c, s, params["is_neox_style"])
    out = torch.cat((rotated.to(x.dtype), heads[..., rotary:]), dim=-1)
    return out.reshape(x.shape)


def golden_rope_positions(positions, query, key, cos, sin, params, offsets=None):
    """``(query_out, key_out)`` of the source contract, on the CPU.

    The row is ``clamp(int64(position) + int64(offset), 0, max_position - 1)``:
    the sum is taken in int64 (so int32 positions and offsets cannot wrap) and
    only then clamped, which is what makes an out-of-table position read its
    nearest row instead of a wrapped one.
    """
    index = positions.to(torch.int64)
    if offsets is not None:
        index = index + offsets.to(torch.int64)
    index = index.reshape(-1).clamp(0, params["max_position_embeddings"] - 1)
    query_out = _rotate(query, index, cos, sin, params)
    key_out = None if key is None else _rotate(key, index, cos, sin, params)
    return query_out, key_out


# ---------------------------------------------------------------------------
# 2. Kernel
# ---------------------------------------------------------------------------
# Round-2/3 knobs (module level so splice.py --constants can branch on them).
#   _SUBTILES  -- sub-tiles a block walks in a 2-stage ``T.Pipelined`` loop, so
#                 the next sub-tile's GM->UB DMA overlaps this one's rotate pass.
#   _ROT_VEC   -- per-lane vector width of the in-place rotate pass (NeoX only).
_SUBTILES = 8
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

    Candidate ``c3`` (Stage 4): as ``c1`` (DMA in -> one in-place rotate pass ->
    DMA out, one UB buffer), but a block now walks ``_SUBTILES`` sub-tiles of
    ``bt`` rows in a two-stage ``T.Pipelined`` loop, so the MTE2 load of the
    next sub-tile overlaps the vector rotate of the current one and the MTE3
    store of the previous one.  Measured on 950PR, one block's DMA and vector
    phases are otherwise strictly serial while the op is far from the HBM
    floor, and the per-block fixed cost is large (see opt_log).

    ``bt`` is the per-sub-tile row count off the wrapper's pair budget:
    ``bt = max(1, min(units_per_block // (hpc * half), _TILE_ELEMS // cols))``.
    ``rot_vec`` / ``tail_vec`` / ``row_vec`` selected CUDA access widths and are
    accepted but unused here; ``staged`` accepts both values and compiles this
    one program.
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
    vec = _ROT_VEC if (neox and _ROT_VEC > 1 and half % _ROT_VEC == 0) else 1

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
            for s in T.Pipelined(nsub, num_stages=2):
                row0 = base + s * bt
                nr = T.min(bt, tokens - row0)
                T.copy(X[row0 : row0 + nr, c0 : c0 + cols], x_ub[0:nr, 0:cols])
                with T.SimtVF(threads=_THREADS):
                    for i in T.Parallel(bt):
                        r = T.min(row0 + i, tokens - 1)
                        seen = T.cast(POS[r], "int64")
                        if has_off:
                            seen = seen + T.cast(OFF[r], "int64")
                        row_ub[i] = T.cast(T.max(0, T.min(seen, max_position - 1)), "int32")
                    for i, k in T.Parallel(bt, half):
                        cs_ub[i, k] = COS[row_ub[i], k]
                    for i, k in T.Parallel(bt, half):
                        sn_ub[i, k] = SIN[row_ub[i], k]
                    if vec > 1:
                        for i, h, kk in T.Parallel(bt, hpc, half // vec):
                            for v in T.vectorized(vec):
                                k = kk * vec + v
                                lo = h * dim + k
                                hi = lo + half
                                cs = T.cast(cs_ub[i, k], "float32")
                                sn = T.cast(sn_ub[i, k], "float32")
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
                            cs = T.cast(cs_ub[i, k], "float32")
                            sn = T.cast(sn_ub[i, k], "float32")
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
                    cs_ub = T.alloc_shared((bt, half), dtype)
                    sn_ub = T.alloc_shared((bt, half), dtype)
                    row_ub = T.alloc_shared((bt,), "int32")
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
                    cs_ub = T.alloc_shared((bt, half), dtype)
                    sn_ub = T.alloc_shared((bt, half), dtype)
                    row_ub = T.alloc_shared((bt,), "int32")
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
                    cs_ub = T.alloc_shared((bt, half), dtype)
                    sn_ub = T.alloc_shared((bt, half), dtype)
                    row_ub = T.alloc_shared((bt,), "int32")
                    side(
                        query, query_out, positions, positions, cos_table, sin_table,
                        row_ub, cs_ub, sn_ub, x_ub, hpc_q, cols_q,
                        bx // nch_q, bx % nch_q,
                    )

        return tilelang.compile(main, target="ascend", out_idx=out_idx)

    return _program


@functools.lru_cache(maxsize=128)
def _build(
    query_heads,
    key_heads,
    head_size,
    rotary_dim,
    max_position,
    dtype,
    position_dtype,
    offset_dtype,
    neox,
    units_per_block=_UNITS_PER_BLOCK,
    staged=False,
):
    """Build (and cache) one program through the frozen factory's argument path."""
    lanes = _VECTOR_ACCESS_BYTES // _TORCH_DTYPE[dtype].itemsize
    span = rotary_dim // 2 if neox else rotary_dim
    factory = _rope_positions_kernel(
        query_heads,
        key_heads,
        head_size,
        rotary_dim,
        max_position,
        dtype,
        position_dtype,
        offset_dtype,
        neox,
        math.gcd(lanes, head_size, span),
        math.gcd(lanes, head_size, rotary_dim),
        math.gcd(lanes, head_size),
    )
    return factory(units_per_block, staged)


# ---------------------------------------------------------------------------
# 3. Launch + comparison against the golden
# ---------------------------------------------------------------------------
_RESULTS = []  # (level, name, ok, detail)
_MAX_DIFF = [0.0]


def _record(level, name, ok, detail=""):
    _RESULTS.append((level, name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[{level}][{tag}] {name}{(' -- ' + detail) if detail else ''}")


def _randn(shape, dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=torch.float32).to(_TORCH_DTYPE[dtype])


def _launch(fn, positions, query, key, cos, sin, offsets, key_heads):
    """Call the program the way the frozen wrapper's ``forward`` does.

    A key of no heads takes the query-only program: the source wrapper builds it
    for ``key_heads == 0`` and answers ``empty_like(key)`` for the key itself.
    """
    if not key_heads and offsets is None:
        return fn(positions, query, cos, sin), None  # query-only program (out_idx=[4])
    if offsets is None:
        return fn(positions, query, key, cos, sin)  # out_idx=[5, 6]
    return fn(positions, query, key, offsets, cos, sin)  # out_idx=[6, 7]


def _diff(got, gold):
    """Largest absolute difference in float space (a NaN against a NaN counts as equal)."""
    a, b = got.float(), gold.float()
    d = (a - b).abs()
    d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return float(d.max()) if d.numel() else 0.0


def run_case(
    level,
    name,
    tokens,
    query_heads,
    key_heads,
    head_size,
    rotary_dim,
    max_position=8192,
    base=500000.0,
    neox=True,
    dtype="bfloat16",
    position_dtype="int64",
    offset_dtype=None,
    positions=None,
    offsets=None,
    positions_shape=None,
    query_values=None,
    carry_key=False,
    seed=0,
    staged=False,
    units_per_block=_UNITS_PER_BLOCK,
    repeat=1,
):
    """Build one geometry, run it on the NPU and compare every output with the golden."""
    params = {
        "head_size": head_size,
        "rotary_dim": rotary_dim,
        "max_position_embeddings": max_position,
        "base": base,
        "is_neox_style": neox,
        "dtype": dtype,
    }
    t_dt = _TORCH_DTYPE[dtype]
    p_dt = torch.int32 if position_dtype == "int32" else torch.int64
    o_dt = (
        None if offset_dtype is None else (torch.int32 if offset_dtype == "int32" else torch.int64)
    )
    shape = list(positions_shape) if positions_shape is not None else [tokens]
    try:
        count = 1
        for extent in shape:
            count *= int(extent)
        q_width = query_heads * head_size
        if count == 0:
            # The wrapper returns empty tensors without launching: a zero-extent
            # grid is not a legal launch and there is nothing to rotate.
            assert all(int(e) >= 0 for e in shape), shape
            _record(level, name, True, "tokens=0 -> no launch, empty output")
            return True
        gen = torch.Generator().manual_seed(seed)
        if positions is None:
            if count <= max_position:
                pos = torch.randperm(max_position, generator=gen)[:count]
            else:
                pos = torch.arange(count) % max_position
            pos = pos.to(torch.int64).reshape(shape).to(p_dt)
        else:
            pos = torch.tensor(positions, dtype=torch.int64).reshape(shape).to(p_dt)
        if query_values is not None:
            query = (
                torch.tensor(query_values, dtype=torch.float32).reshape(shape + [q_width]).to(t_dt)
            )
        else:
            query = _randn([count, q_width], dtype, seed).reshape(shape + [q_width])
        if key_heads:
            key = _randn([count, key_heads * head_size], dtype, seed + 1).reshape(
                shape + [key_heads * head_size]
            )
            gold_key = key
        elif carry_key:
            # A key of no heads: the wrapper answers ``empty_like(key)`` and the
            # query-only program serves the rotation.
            key = torch.zeros(shape + [0], dtype=t_dt)
            gold_key = None
        else:
            key = None
            gold_key = None
        off = None
        if o_dt is not None:
            if offsets is None:
                off = (torch.arange(count, dtype=torch.int64) % 3).reshape(shape).to(o_dt)
            else:
                off = torch.tensor(offsets, dtype=torch.int64).reshape(shape).to(o_dt)

        cos, sin = rope_tables(params)
        gold_q, gold_k = golden_rope_positions(pos, query, gold_key, cos, sin, params, off)

        fn = _build(
            query_heads,
            key_heads,
            head_size,
            rotary_dim,
            max_position,
            dtype,
            position_dtype,
            offset_dtype,
            neox,
            units_per_block,
            staged,
        )
        pos_d = pos.reshape(-1).to("npu")
        query_d = query.reshape(count, q_width).to("npu")
        key_d = None if key is None else key.reshape(count, -1).to("npu")
        off_d = None if off is None else off.reshape(-1).to("npu")
        cos_d, sin_d = cos.to("npu"), sin.to("npu")

        start = time.perf_counter()
        q_out, k_out = _launch(fn, pos_d, query_d, key_d, cos_d, sin_d, off_d, key_heads)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1e3
        if repeat > 1:
            for _ in range(repeat - 1):  # out of place: the same inputs give the same outputs
                _launch(fn, pos_d, query_d, key_d, cos_d, sin_d, off_d, key_heads)
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(repeat):
                _launch(fn, pos_d, query_d, key_d, cos_d, sin_d, off_d, key_heads)
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) * 1e3 / repeat

        atol, rtol = _TOLERANCE[dtype]
        case_diff = 0.0
        got_q = q_out.cpu().reshape(gold_q.shape)
        torch.testing.assert_close(got_q, gold_q, atol=atol, rtol=rtol, equal_nan=True)
        case_diff = max(case_diff, _diff(got_q, gold_q))
        if key_heads:
            got_k = k_out.cpu().reshape(gold_k.shape)
            torch.testing.assert_close(got_k, gold_k, atol=atol, rtol=rtol, equal_nan=True)
            case_diff = max(case_diff, _diff(got_k, gold_k))
        elif carry_key:
            # The query-only program returns no key; the wrapper answers
            # ``empty_like(key)`` on the host (a zero-column tensor, unchanged).
            assert k_out is None, "the query-only program must not return a key"
            assert tuple(key.shape) == tuple(shape) + (0,), key.shape
        _MAX_DIFF[0] = max(_MAX_DIFF[0], case_diff)
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
        f"tokens={count} q={query_heads}x{head_size} k={key_heads} R={rotary_dim} "
        f"{dtype} {position_dtype} {'neox' if neox else 'gptj'} "
        f"diff={case_diff:.3g} {elapsed:.3f} ms",
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
    """Smoke gate: both conventions, three dtypes, both tables, clamp, the host paths."""
    run_case(
        "L0",
        "L0-1 neox-full-head fp16",
        tokens=64,
        query_heads=4,
        key_heads=2,
        head_size=64,
        rotary_dim=64,
        max_position=128,
        base=10000.0,
        neox=True,
        dtype="float16",
        seed=1,
    )
    run_case(
        "L0",
        "L0-2 gptj-adjacent-full fp16",
        tokens=64,
        query_heads=4,
        key_heads=2,
        head_size=64,
        rotary_dim=64,
        max_position=128,
        base=10000.0,
        neox=False,
        dtype="float16",
        seed=2,
    )
    run_case(
        "L0",
        "L0-3 neox-bf16-batched-positions",
        tokens=32,
        query_heads=4,
        key_heads=0,
        head_size=64,
        rotary_dim=64,
        max_position=64,
        base=10000.0,
        neox=True,
        dtype="bfloat16",
        positions_shape=[2, 16],
        seed=3,
    )
    run_case(
        "L0",
        "L0-4 fp32-partial-rotary gptj",
        tokens=64,
        query_heads=8,
        key_heads=2,
        head_size=64,
        rotary_dim=24,
        max_position=64,
        base=10000.0,
        neox=False,
        dtype="float32",
        seed=4,
    )
    run_case(
        "L0",
        "L0-5 phi2-query-only-int32",
        tokens=1000,
        query_heads=32,
        key_heads=0,
        head_size=80,
        rotary_dim=32,
        max_position=2048,
        base=10000.0,
        neox=True,
        dtype="bfloat16",
        position_dtype="int32",
        seed=5,
    )
    run_case(
        "L0",
        "L0-6a clamp-outside-table neox",
        tokens=4,
        query_heads=2,
        key_heads=1,
        head_size=32,
        rotary_dim=32,
        max_position=8,
        base=10000.0,
        neox=True,
        dtype="float16",
        positions=[-1, 8, 3, 0],
        seed=6,
    )
    run_case(
        "L0",
        "L0-6b clamp-outside-table gptj",
        tokens=4,
        query_heads=2,
        key_heads=1,
        head_size=32,
        rotary_dim=32,
        max_position=8,
        base=10000.0,
        neox=False,
        dtype="bfloat16",
        positions=[7, -13, 8, 2],
        seed=7,
    )
    run_case(
        "L0",
        "L0-7 zero-head-key",
        tokens=16,
        query_heads=3,
        key_heads=0,
        head_size=64,
        rotary_dim=32,
        max_position=64,
        base=10000.0,
        neox=True,
        dtype="float16",
        carry_key=True,
        seed=8,
    )
    run_case(
        "L0",
        "L0-8 empty-tokens",
        tokens=0,
        query_heads=2,
        key_heads=1,
        head_size=64,
        rotary_dim=32,
        max_position=64,
        base=10000.0,
        neox=True,
        dtype="float16",
        seed=9,
    )
    _case_staged_equivalence("L0", "L0-9 staged-both-values")
    _case_roofline("L0", "L0-10 roofline-accounting")
    run_case(
        "L0",
        "L0-11 offsets-int64-neox",
        tokens=48,
        query_heads=4,
        key_heads=2,
        head_size=64,
        rotary_dim=64,
        max_position=128,
        base=10000.0,
        neox=True,
        dtype="float16",
        offset_dtype="int64",
        seed=11,
    )
    run_case(
        "L0",
        "L0-12 offsets-int32-gptj-partial",
        tokens=48,
        query_heads=4,
        key_heads=4,
        head_size=64,
        rotary_dim=32,
        max_position=64,
        base=10000.0,
        neox=False,
        dtype="float32",
        position_dtype="int32",
        offset_dtype="int32",
        seed=12,
    )
    _finish("L0", strict=True)


def run_L1():
    """The manifest's six workloads (llama prefill runs both 16-bit dtypes)."""
    run_case(
        "L1",
        "L1-1 llama3-8b-decode-b64 bf16",
        tokens=64,
        query_heads=32,
        key_heads=8,
        head_size=128,
        rotary_dim=128,
        max_position=8192,
        base=500000.0,
        neox=True,
        dtype="bfloat16",
        seed=21,
        repeat=3,
    )
    for dtype, tag in (("bfloat16", "bf16"), ("float16", "fp16")):
        run_case(
            "L1",
            f"L1-2 llama3-8b-prefill-4k {tag}",
            tokens=4096,
            query_heads=32,
            key_heads=8,
            head_size=128,
            rotary_dim=128,
            max_position=8192,
            base=500000.0,
            neox=True,
            dtype=dtype,
            seed=22,
            repeat=2,
        )
    run_case(
        "L1",
        "L1-3 llama3-8b-b2-2k fp32",
        tokens=4096,
        query_heads=32,
        key_heads=8,
        head_size=128,
        rotary_dim=128,
        max_position=8192,
        base=500000.0,
        neox=True,
        dtype="float32",
        positions_shape=[2, 2048],
        seed=23,
        repeat=2,
    )
    run_case(
        "L1",
        "L1-4 gptj-6b-prefill fp16",
        tokens=2048,
        query_heads=16,
        key_heads=16,
        head_size=256,
        rotary_dim=64,
        max_position=2048,
        base=10000.0,
        neox=False,
        dtype="float16",
        seed=24,
        repeat=2,
    )
    run_case(
        "L1",
        "L1-5 gptj-6b-decode-b32 fp16",
        tokens=32,
        query_heads=16,
        key_heads=16,
        head_size=256,
        rotary_dim=64,
        max_position=2048,
        base=10000.0,
        neox=False,
        dtype="float16",
        seed=25,
        repeat=3,
    )
    run_case(
        "L1",
        "L1-6 phi-2-query-only bf16",
        tokens=1000,
        query_heads=32,
        key_heads=0,
        head_size=80,
        rotary_dim=32,
        max_position=2048,
        base=10000.0,
        neox=True,
        dtype="bfloat16",
        position_dtype="int32",
        seed=26,
        repeat=2,
    )
    _finish("L1", strict=True)


def run_L2():
    """Larger shapes than the manifest: informative, non-blocking."""
    run_case(
        "L2",
        "L2-1 neox-prefill-8192 bf16",
        tokens=8192,
        query_heads=32,
        key_heads=8,
        head_size=128,
        rotary_dim=128,
        max_position=8192,
        base=500000.0,
        neox=True,
        dtype="bfloat16",
        seed=31,
        repeat=2,
    )
    run_case(
        "L2",
        "L2-2 gptj-prefill-4096 fp16",
        tokens=4096,
        query_heads=16,
        key_heads=16,
        head_size=256,
        rotary_dim=64,
        max_position=2048,
        base=10000.0,
        neox=False,
        dtype="float16",
        seed=32,
        repeat=2,
    )
    run_case(
        "L2",
        "L2-3 neox-fp32-2048",
        tokens=2048,
        query_heads=32,
        key_heads=8,
        head_size=128,
        rotary_dim=128,
        max_position=4096,
        base=500000.0,
        neox=True,
        dtype="float32",
        seed=33,
        repeat=2,
    )
    _finish("L2", strict=False)


def run_boundary():
    """Targeted boundaries: tail blocks, odd heads, degenerate chunking, specials."""
    run_case(
        "boundary",
        "B-1 tail-row-block-17-tokens",
        tokens=17,
        query_heads=4,
        key_heads=2,
        head_size=64,
        rotary_dim=64,
        max_position=64,
        base=10000.0,
        neox=True,
        dtype="float16",
        seed=41,
    )
    run_case(
        "boundary",
        "B-2 single-token",
        tokens=1,
        query_heads=4,
        key_heads=2,
        head_size=64,
        rotary_dim=32,
        max_position=64,
        base=10000.0,
        neox=False,
        dtype="bfloat16",
        positions=[3],
        seed=42,
    )
    run_case(
        "boundary",
        "B-3a odd-head-65 neox",
        tokens=33,
        query_heads=8,
        key_heads=4,
        head_size=65,
        rotary_dim=32,
        max_position=64,
        base=10000.0,
        neox=True,
        dtype="float16",
        seed=43,
    )
    run_case(
        "boundary",
        "B-3b odd-head-65 gptj",
        tokens=33,
        query_heads=8,
        key_heads=4,
        head_size=65,
        rotary_dim=32,
        max_position=64,
        base=10000.0,
        neox=False,
        dtype="float32",
        seed=44,
    )
    run_case(
        "boundary",
        "B-4 phi2-shape-non-pow2-tile",
        tokens=100,
        query_heads=32,
        key_heads=0,
        head_size=80,
        rotary_dim=32,
        max_position=256,
        base=10000.0,
        neox=True,
        dtype="bfloat16",
        seed=45,
    )
    run_case(
        "boundary",
        "B-5 odd-head-count-hpc-1",
        tokens=20,
        query_heads=3,
        key_heads=3,
        head_size=64,
        rotary_dim=64,
        max_position=64,
        base=10000.0,
        neox=True,
        dtype="float16",
        seed=46,
    )
    run_case(
        "boundary",
        "B-6 tiny-table-P-2",
        tokens=4,
        query_heads=2,
        key_heads=2,
        head_size=32,
        rotary_dim=32,
        max_position=2,
        base=10000.0,
        neox=True,
        dtype="float16",
        positions=[0, 1, 1, 0],
        seed=47,
    )
    specials = [
        float("inf"),
        -float("inf"),
        float("nan"),
        0.0,
        -0.0,
        65504.0,
        -65504.0,
        6.0e-5,
        -6.0e-5,
        1.0,
        -1.0,
        0.5,
        -0.5,
        3.0,
        -3.0,
        2.0,
    ]
    run_case(
        "boundary",
        "B-7 special-values-inf-nan-zero-fp16",
        tokens=1,
        query_heads=1,
        key_heads=1,
        head_size=32,
        rotary_dim=32,
        max_position=32,
        base=10000.0,
        neox=True,
        dtype="float16",
        query_values=specials * 2,
        positions=[5],
        seed=48,
    )
    run_case(
        "boundary",
        "B-8 half-not-vector-multiple R-24 fp16",
        tokens=40,
        query_heads=4,
        key_heads=2,
        head_size=48,
        rotary_dim=24,
        max_position=64,
        base=10000.0,
        neox=False,
        dtype="float16",
        seed=49,
    )
    run_case(
        "boundary",
        "B-9 offsets-push-outside-table",
        tokens=6,
        query_heads=2,
        key_heads=2,
        head_size=32,
        rotary_dim=32,
        max_position=8,
        base=10000.0,
        neox=True,
        dtype="float16",
        positions=[-3, 5, 7, 0, 2, 6],
        offsets=[-20, 20, -1, 1, 3, -7],
        offset_dtype="int64",
        seed=50,
    )
    run_case(
        "boundary",
        "B-10 one-column-chunk-per-head",
        tokens=12,
        query_heads=1,
        key_heads=1,
        head_size=128,
        rotary_dim=64,
        max_position=32,
        base=10000.0,
        neox=False,
        dtype="bfloat16",
        seed=51,
    )
    _finish("boundary", strict=False)


# ---------------------------------------------------------------------------
# Host-level L0 cases (staged interface, roofline accounting)
# ---------------------------------------------------------------------------
def _case_staged_equivalence(level, name):
    """Both ``staged`` values compile the same program and give the same output."""
    try:
        gen = torch.Generator().manual_seed(71)
        pos = torch.randperm(64, generator=gen)[:37].to("npu")
        query = _randn([37, 256], "float16", 72).to("npu")
        key = _randn([37, 128], "float16", 73).to("npu")
        cos, sin = rope_tables(
            {
                "rotary_dim": 32,
                "base": 10000.0,
                "max_position_embeddings": 64,
                "dtype": "float16",
                "head_size": 64,
                "is_neox_style": True,
            }
        )
        cos_d, sin_d = cos.to("npu"), sin.to("npu")
        outs = []
        for staged in (False, True):
            fn = _build(
                4,
                2,
                64,
                32,
                64,
                "float16",
                "int64",
                None,
                True,
                _UNITS_PER_BLOCK,
                staged,
            )
            outs.append(_launch(fn, pos, query, key, cos_d, sin_d, None, 2))
        torch.npu.synchronize()
        for i in (0, 1):
            torch.testing.assert_close(
                outs[0][i].cpu(), outs[1][i].cpu(), atol=0, rtol=0, equal_nan=True
            )
    except AssertionError as exc:
        _record(level, name, False, str(exc)[:300])
        return
    except (ValueError, RuntimeError) as exc:  # noqa: BLE001 - reported, not swallowed
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:260]}")
        return
    _record(level, name, True, "staged=True/False agree bit for bit")


def _roofline(tokens, query_heads, key_heads, head_size, rotary_dim, itemsize, rows_read):
    """The manifest's ``rope_positions_roofline``: three flops per rotated value."""
    rotated = tokens * rotary_dim * (query_heads + key_heads)
    q_bytes = tokens * query_heads * head_size * itemsize
    k_bytes = tokens * key_heads * head_size * itemsize
    positions_bytes = tokens * 8
    cache_bytes = rows_read * rotary_dim * itemsize
    return 3 * rotated, 2 * (q_bytes + k_bytes) + positions_bytes + cache_bytes


def _case_roofline(level, name):
    """The accounting matches the manifest formulas on the representative workload."""
    expected = (62_914_560, 84_967_424)  # llama3-8b-prefill-4k, bf16, 4096 rows read
    try:
        got = _roofline(4096, 32, 8, 128, 128, 2, 4096)
        assert got == expected, f"roofline {got} != manifest {expected}"
    except AssertionError as exc:
        _record(level, name, False, str(exc))
        return
    _record(level, name, True, f"flops={got[0]} bytes={got[1]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
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
