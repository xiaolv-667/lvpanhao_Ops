# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""`_ssd_chunk_coupling_kernel` on Ascend 950PR: causal `tril(C @ B^T)` per chunk.

TileOPs migration of `SSDChunkCouplingFwdOp` (GPU TileLang source
``tileops/_kernels/mamba/ssd_chunk_coupling.py``).  For every
``(batch, chunk, group)`` the kernel computes

    cb[b, c, g, l, s] = sum_n C[b, c*q + l, g, n] * B[b, c*q + s, g, n]   (s <= l)
                      = 0                                                  (s > l)

with ``q = chunk_len``, ``n < d_state`` and one fp32 accumulation followed by a
single rounding to the input dtype (fp16 / bf16).

Structure (DESIGN.md 3.3 / 5.2 / 6, frozen; REVIEW.md passed, 0 blockers):

* one-dimensional ``T.Kernel(grid)``, ``grid = B * NC * G * nlt * nst`` with the
  scalar decode order ``bcg -> lt -> st`` (DESIGN.md 0.6 R1);
* ``BL = BL`` square output tile, ``BK = BK`` K blocks; ``C``/``B`` row blocks are
  copied straight out of the 4-D strided ``[B, S, G, N]`` layout into L1 with
  ``nd2nz`` (no host permute, no in-kernel transpose) and consumed with
  ``T.gemm(transpose_B=True, clear_accum=kk == 0)`` (DESIGN.md 0.6 R2);
* strictly-upper tiles take ``trips = 0`` so their K loop never runs, and the
  unified masked epilogue writes the whole tile as exact zeros (DESIGN.md 0.6 R4,
  1.6.1 O3);
* the epilogue is the platform's Cube+Vector mixed tail (DESIGN.md 0.6 R3):
  ``acc(L0C fp32) --dual_copy(M split)--> cv(UB fp32) --SimtVF select+cast-->
  co(UB dtype) --dual_copy--> cb``; the AIV sub-block id selects the tile half so
  the predicate is ``(j - i) <= l0 + sid*BL/2 - s0`` (strength reduced, 1.6.1 O2).
  The id is read with ``T.call_extern("int64", "asc_get_sub_block_id")``: the
  ``T.Vector(vector=2) as sid`` spelling of DESIGN.md 3.3 compiles but returns
  wrong values nondeterministically on this toolchain (see RETROSPECTIVE.md
  stage 3 and repro/TRAP-950-aiv-sid-vector-frame.py).

The mask is a ``T.if_then_else`` select (never a multiply): ``nan * 0`` would
break the "exact zero above the diagonal" contract (verify_equiv.py E3).

Content layout:

1. ``golden_ssd_chunk_coupling`` -- independent torch CPU reference (fp32
   accumulate, ``tril`` select, one cast), taken from the source repo's
   ``ssd_chunk_coupling_fwd_ref`` and *not* from the tiling above.
2. ``_ssd_chunk_coupling_kernel`` -- shape-specialised kernel factory.
3. ``run_case`` -- launch on the NPU and compare with the golden.
4. ``run_L0`` / ``run_L1`` / ``run_L2`` / ``run_boundary`` / ``main``.

run: python _ssd_chunk_coupling_kernel.py --level L0|L1|L2|boundary|all
"""

import argparse
import functools
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401  -- registers torch.npu

_TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}
_DTYPES = tuple(_TORCH_DTYPE)
_ATOL = 1e-3  # DESIGN.md 8.2 (COUPLING_ATOL of examples/TileOPs/tests/ops/test_mamba.py)
_RTOL = 1e-3  # DESIGN.md 8.2 (COUPLING_RTOL)
_DEFAULT_DTYPE = "float16"
_BL_LADDER = (128, 64, 32)  # DESIGN.md 5.2 (dual_copy 2:1 split needs BL >= 32)
_MIN_D_STATE = 16  # one Cube K fractal


# ---------------------------------------------------------------------------
# 1. Golden: independent torch reference (source repo ssd_chunk_coupling_fwd_ref)
# ---------------------------------------------------------------------------
def golden_ssd_chunk_coupling(C_mat, B_mat, chunk_len):
    """``cb[b,c,g,l,s] = sum_n C[b,cQ+l,g,n] * B[b,cQ+s,g,n]`` for ``s <= l`` else 0.

    fp32 accumulation with a single rounding to ``C_mat.dtype``; the masked half
    is an exact ``0`` of the output dtype (``torch.where``, not a multiply), so
    non-finite inputs cannot leak through as ``nan * 0``.
    """
    batch, seq_len, n_groups, d_state = C_mat.shape
    num_chunks = seq_len // chunk_len
    c = C_mat.reshape(batch, num_chunks, chunk_len, n_groups, d_state).float()
    b = B_mat.reshape(batch, num_chunks, chunk_len, n_groups, d_state).float()
    cb = torch.einsum("bcqgn,bcsgn->bcgqs", c, b)
    mask = torch.tril(torch.ones(chunk_len, chunk_len, dtype=torch.bool))
    zero = torch.zeros((), dtype=cb.dtype)
    return torch.where(mask, cb, zero).to(C_mat.dtype)


# ---------------------------------------------------------------------------
# 2. Kernel factory (DESIGN.md 3.3 / 5.2, one compiled artefact per shape)
# ---------------------------------------------------------------------------
def _tile_shape(chunk_len, d_state):
    """``BL``/``BK`` of DESIGN.md 5.2 plus the contract checks of 5.4 / 9.1."""
    block_l = None
    for cand in _BL_LADDER:
        if chunk_len % cand == 0:
            block_l = cand
            break
    if block_l is None:
        raise ValueError(
            f"chunk_len={chunk_len} is not a multiple of 32 (DESIGN.md 5.4/9.1: "
            "the tile boundary must be static and dual_copy needs BL >= 32)"
        )
    if d_state < _MIN_D_STATE:
        raise ValueError(f"d_state={d_state} < {_MIN_D_STATE} (one Cube K fractal)")
    block_n = 64 if d_state % 64 == 0 else 16
    return block_l, block_n


@functools.lru_cache(maxsize=32)
def _ssd_chunk_coupling_kernel(batch, num_chunks, n_groups, chunk_len, d_state, dtype):
    """Compiled kernel for one geometry (``out_idx=-1`` allocates ``cb``)."""
    if dtype not in _TORCH_DTYPE:
        raise ValueError(f"dtype={dtype!r} not in {_DTYPES}")
    for name, value in (
        ("batch", batch),
        ("num_chunks", num_chunks),
        ("n_groups", n_groups),
        ("chunk_len", chunk_len),
        ("d_state", d_state),
    ):
        if value <= 0:
            raise ValueError(f"{name}={value} must be positive")

    BL, BK = _tile_shape(chunk_len, d_state)
    Q, N, G = chunk_len, d_state, n_groups
    S = num_chunks * Q
    nlt, nst = Q // BL, Q // BL
    grid = batch * num_chunks * n_groups * nlt * nst
    k_trips = -(-N // BK)
    c_shape = (batch, S, G, N)
    cb_shape = (batch, num_chunks, G, Q, Q)

    @T.prim_func
    def main(
        C_mat: T.Tensor(c_shape, dtype),
        B_mat: T.Tensor(c_shape, dtype),
        cb: T.Tensor(cb_shape, dtype),
    ):
        with T.Kernel(grid) as bid:
            # 1) task decode (DESIGN.md 0.6 R1): bcg -> lt -> st, block-level scalars.
            bcg = bid // (nlt * nst)
            rem = bid % (nlt * nst)
            bb = bcg // (num_chunks * G)
            cg = bcg % (num_chunks * G)
            bc = cg // G
            bg = cg % G
            l0 = (rem // nst) * BL
            s0 = (rem % nst) * BL

            cl1 = T.alloc_l1((BL, BK), dtype)  # C row block
            bl1 = T.alloc_l1((BL, BK), dtype)  # B row block
            x0 = T.alloc_l0a((BL, BK), dtype)
            y0 = T.alloc_l0b((BL, BK), dtype)
            acc = T.alloc_l0c((BL, BL), "float32")
            cv = T.alloc_shared((BL // 2, BL), "float32")  # L0C -> UB (fp32 only)
            co = T.alloc_shared((BL // 2, BL), dtype)  # masked + cast half tile
            T.annotate_buffer_versions({cl1: 2, bl1: 2})

            # 2) K loop; a strictly-upper tile keeps the code path but trips = 0.
            trips = T.if_then_else(s0 <= l0 + BL - 1, k_trips, 0)
            for kk in T.Pipelined(trips, num_stages=2):
                n0 = kk * BK
                nk = T.min(N - n0, BK)
                T.copy(
                    C_mat[bb, bc * Q + l0 : bc * Q + l0 + BL, bg, n0 : n0 + nk],
                    cl1[:, :nk],
                )
                T.copy(
                    B_mat[bb, bc * Q + s0 : bc * Q + s0 + BL, bg, n0 : n0 + nk],
                    bl1[:, :nk],
                )
                T.copy(cl1[:, :nk], x0[:, :nk])
                T.copy(bl1[:, :nk], y0[:, :nk])
                T.gemm(
                    x0[:, :nk],
                    y0[:, :nk],
                    acc,
                    transpose_B=True,
                    clear_accum=kk == 0,
                )

            # 3) masked epilogue on both AIVs (DESIGN.md 0.6 R3 / 1.6.1 O1+O2).
            # The AIV sub-block id must be read as an extern scalar: the
            # ``T.Vector(vector=2) as sid`` spelling of DESIGN.md 3.3 compiles but
            # returns wrong values on this toolchain (measured: 5-7 of 8 launches
            # corrupt a 64-row half, nondeterministically; probe P-epl).  The
            # extern form is verified by repro/TRAP-950-aiv-sid-vector-frame.py
            # and matches repro/CONST-950-aiv-sid-capture.py in the knowledge base.
            T.dual_copy(acc, cv)
            sid = T.call_extern("int64", "asc_get_sub_block_id")
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(BL // 2, BL):
                    co[i, j] = T.cast(
                        T.if_then_else(
                            (j - i) <= (l0 + T.cast(sid, "int32") * (BL // 2) - s0),
                            cv[i, j],
                            T.float32(0.0),
                        ),
                        dtype,
                    )
            T.dual_copy(co, cb[bb, bc, bg, l0 : l0 + BL, s0 : s0 + BL])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _launch(batch, num_chunks, n_groups, chunk_len, d_state, dtype):
    """Compiled kernel for one geometry (stable call shape for the tests)."""
    return _ssd_chunk_coupling_kernel(batch, num_chunks, n_groups, chunk_len, d_state, dtype)


# ---------------------------------------------------------------------------
# 3. Launch + comparison with the golden
# ---------------------------------------------------------------------------
_RESULTS = []  # (level, name, ok, detail)
_MAX_DIFF = [0.0]
_MAX_WHERE = [""]


def _record(level, name, ok, detail=""):
    _RESULTS.append((level, name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[{level}][{tag}] {name}{(' -- ' + detail) if (detail and not ok) else ''}", flush=True)
    if not ok:
        print(f"       detail: {detail}", flush=True)


def _inputs(batch, num_chunks, chunk_len, n_groups, d_state, dtype, seed, scale=1.0):
    """Deterministic CPU inputs rounded to the kernel dtype."""
    gen = torch.Generator().manual_seed(seed)
    shape = (batch, num_chunks * chunk_len, n_groups, d_state)
    c = (torch.randn(shape, generator=gen, dtype=torch.float32) * scale).to(_TORCH_DTYPE[dtype])
    b = (torch.randn(shape, generator=gen, dtype=torch.float32) * scale).to(_TORCH_DTYPE[dtype])
    return c, b


def _compare(got, gold, atol=_ATOL, rtol=_RTOL, dtype_out=None):
    """Assert_close-style check with a one-output-ulp allowance.

    ``atol/rtol`` stay the manifest gate (DESIGN.md 8.2).  On top of that a
    difference of at most one ulp of the *output* dtype is accepted: both the
    golden and the kernel accumulate in fp32 and round once, so two legitimate
    fp32 accumulation orders may land on opposite sides of a rounding boundary
    (observed on bf16 only, 16 of 1.36M elements of the 2p7b workload, always
    exactly 1 ulp).  The count and the largest ulp distance are reported.
    """
    a, b = got.float(), gold.float()
    diff = (a - b).abs()
    tol = atol + rtol * b.abs()
    if dtype_out is not None:
        mant_bits = 7 if dtype_out == "bfloat16" else 10
        normal_min = 2.0**-126 if dtype_out == "bfloat16" else 2.0**-14
        mag = b.abs().clamp(min=normal_min)
        exponent = torch.floor(torch.log2(mag))
        ulp = torch.pow(torch.tensor(2.0), exponent - mant_bits)
        tol = torch.maximum(tol, ulp)
    bad = diff > tol
    n_bad = int(bad.sum())
    nan_bad = int(((a != a) != (b != b)).sum())
    max_diff = float(diff.max()) if diff.numel() else 0.0
    n_diff = int((diff > 0).sum())
    ulp_max = 0.0
    if dtype_out is not None and n_diff:
        # Only magnitudes whose ulp is above the absolute gate are informative.
        big = ulp >= atol
        if bool(big.any()):
            ulp_max = float((diff[big] / ulp[big]).max())
    metric = f"max_diff={max_diff:.3e} differing={n_diff}/{diff.numel()} max_ulp={ulp_max:.2f}"
    if n_bad == 0 and nan_bad == 0:
        return True, max_diff, metric
    idx = bad.nonzero()
    detail = f"{n_bad} element(s) over atol={atol} rtol={rtol} (nan-mismatch {nan_bad}); {metric}"
    if idx.numel():
        i = tuple(int(v) for v in idx[0])
        detail += f"; first at {i} got={a[i].item():.6g} gold={b[i].item():.6g}"
    return False, max_diff, detail


def run_case(
    level,
    name,
    batch,
    num_chunks,
    chunk_len,
    n_groups,
    d_state,
    dtype=_DEFAULT_DTYPE,
    seed=0,
    repeat=1,
    scale=1.0,
    noncontiguous=False,
):
    """Build one geometry, run it on the NPU and compare with the golden."""
    dt = _TORCH_DTYPE[dtype]
    c, b = _inputs(batch, num_chunks, chunk_len, n_groups, d_state, dtype, seed, scale)
    gold = golden_ssd_chunk_coupling(c, b, chunk_len)
    try:
        fn = _launch(batch, num_chunks, n_groups, chunk_len, d_state, dtype)
        dev = "npu"
        if noncontiguous:
            # Host contract of the source op: the wrapper normalises strided
            # inputs with .contiguous(); the kernel itself always sees this view.
            c_full = torch.zeros(
                (batch, 2 * num_chunks * chunk_len, n_groups, d_state), dtype=dt
            )
            b_full = torch.zeros_like(c_full)
            c_full[:, ::2] = c
            b_full[:, ::2] = b
            c_dev_full, b_dev_full = c_full.to(dev), b_full.to(dev)
            c_slice, b_slice = c_dev_full[:, ::2], b_dev_full[:, ::2]
            if c_slice.is_contiguous():
                raise AssertionError("device view is contiguous; the case would not test strides")
            c_dev, b_dev = c_slice.contiguous(), b_slice.contiguous()
        else:
            c_dev, b_dev = c.to(dev), b.to(dev)
        # Warm-up: the first launch after compilation also pays binary load and
        # the first output allocation of this size, both outside the kernel.
        fn(c_dev, b_dev)
        torch.npu.synchronize()
        times = []
        for _ in range(repeat):
            started = time.perf_counter()
            out = fn(c_dev, b_dev)
            torch.npu.synchronize()
            times.append((time.perf_counter() - started) * 1e3)
        elapsed = sorted(times)[len(times) // 2]
        got = out.cpu()
    except Exception as exc:  # noqa: BLE001 - a device failure must be reported, not raised
        _record(level, name, False, f"{type(exc).__name__}: {exc}".splitlines()[0][:300])
        return False

    ok, max_diff, metric = _compare(got, gold, dtype_out=dtype)
    if max_diff > _MAX_DIFF[0]:
        _MAX_DIFF[0] = max_diff
        _MAX_WHERE[0] = name
    tag = (
        f"B={batch} NC={num_chunks} Q={chunk_len} G={n_groups} N={d_state} {dtype} "
        f"repeat={repeat} grid={batch * num_chunks * n_groups * (chunk_len // _tile_shape(chunk_len, d_state)[0]) ** 2} "
        f"launch={elapsed:.3f} ms; {metric}"
    )
    _record(level, name, ok, metric if not ok else f"max_diff={max_diff:.3e} launch={elapsed:.3f} ms")
    print(f"       {tag}", flush=True)
    return ok


# ---------------------------------------------------------------------------
# 4. Semantic assertions that a tolerance comparison cannot prove
# ---------------------------------------------------------------------------
def _run_raw(batch, num_chunks, chunk_len, n_groups, d_state, dtype, seed, scale=1.0):
    c, b = _inputs(batch, num_chunks, chunk_len, n_groups, d_state, dtype, seed, scale)
    fn = _launch(batch, num_chunks, n_groups, chunk_len, d_state, dtype)
    out = fn(c.to("npu"), b.to("npu")).cpu()
    torch.npu.synchronize()
    return c, b, out


def _case_mask_semantics(level, name, batch, num_chunks, chunk_len, n_groups, d_state,
                         dtype=_DEFAULT_DTYPE, seed=0):
    """Diagonal kept + strictly-upper triangle exactly zero + both AIV halves.

    The diagonal check pins two things a tolerance test alone can hide:
    ``s == l`` must be *kept* (not ``<``), and the row offset must follow the AIV
    sub-block id, so row ``BL/2`` (AIV 1) has to carry its own inner product
    (DESIGN.md 9.1 item 9 / REVIEW.md dimension 6 suggestion 4).
    """
    Q, BL = chunk_len, _tile_shape(chunk_len, d_state)[0]
    c, b, out = _run_raw(batch, num_chunks, chunk_len, n_groups, d_state, dtype, seed)
    problems = []
    cf = c.float().reshape(batch, num_chunks, Q, n_groups, d_state)
    bf = b.float().reshape(batch, num_chunks, Q, n_groups, d_state)
    # ip[..., l] = C[b,c,l,g,:] . B[b,c,l,g,:] = the reference diagonal value.
    ip = (cf * bf).sum(-1).permute(0, 1, 3, 2)  # (batch, NC, G, Q)

    upper = torch.triu(torch.ones(Q, Q, dtype=torch.bool), diagonal=1)
    n_upper = int(upper.sum()) * batch * num_chunks * n_groups
    n_nonzero = int((out.float()[..., upper] != 0).sum())
    if n_nonzero:
        problems.append(f"{n_nonzero}/{n_upper} strictly-upper elements are not exact 0")

    diag = out.float().diagonal(dim1=-2, dim2=-1)  # (batch, NC, G, Q)
    d_err = (diag - ip).abs().max().item()
    d_scale = ip.abs().max().item()
    if d_err > _ATOL + _RTOL * d_scale:
        problems.append(f"diagonal differs from the inner product by {d_err:.3e}")
    if d_scale == 0.0:
        problems.append("diagonal is all zero (inputs should give a non-zero inner product)")

    # Both AIV halves, explicit: (BL/2, BL/2) is AIV 1 on the diagonal, (BL/2, 0)
    # is AIV 1 below the diagonal, (0, BL/2) is AIV 0 strictly above.  A row from
    # the second l-tile (Q > BL) proves the tile decode as well.
    checks = [((0, 0), True), ((BL // 2, BL // 2), True), ((BL // 2, 0), True),
              ((0, BL // 2), False), ((BL // 2 - 1, BL // 2), False)]
    if Q >= 2 * BL:
        checks += [((BL, 0), True), ((BL + BL // 2, BL // 2), True), ((0, BL), False)]
    for (l, s), keep in checks:
        if l >= Q or s >= Q:
            continue
        got_cells = out.float()[..., l, s]  # (batch, NC, G)
        if keep:
            # Reference = C row l . B row s (not the diagonal inner product).
            want = (cf[:, :, l] * bf[:, :, s]).sum(-1)  # (batch, NC, G)
            err = (got_cells - want).abs().max().item()
            scale = want.abs().max().item()
            if err > _ATOL + _RTOL * scale:
                problems.append(f"cb[...,{l},{s}] != C row {l} . B row {s} (err {err:.3e})")
        elif (got_cells != 0).any():
            problems.append(f"cb[...,{l},{s}] is not exact 0")

    n_zero_tiles = max(Q // BL - 1, 0)  # strictly-upper tiles of this grid
    detail = (
        f"diag_err={d_err:.3e} upper_nonzero={n_nonzero}/{n_upper} "
        f"BL={BL} strict_upper_tiles={n_zero_tiles} dtype={dtype}"
    )
    ok = not problems
    _record(level, name, ok, "; ".join(problems) if problems else detail)
    print(f"       {detail}", flush=True)
    return ok


def _case_contract(level, name):
    """The factory refuses the shapes DESIGN.md 5.4 / 9.1 declare unsupported."""
    problems = []
    rejects = [
        ((1, 2, 1, 48, 64, "float16"), "chunk_len=48 (chunk_len % 32 != 0)"),
        ((1, 2, 1, 16, 64, "float16"), "chunk_len=16 (BL would be < 32)"),
        ((1, 2, 1, 64, 8, "float16"), "d_state=8 (< one Cube K fractal)"),
        ((1, 2, 1, 64, 64, "float32"), "float32 inputs"),
        ((0, 2, 1, 64, 64, "float16"), "batch=0"),
        ((1, 0, 1, 64, 64, "float16"), "num_chunks=0"),
    ]
    for args, why in rejects:
        try:
            _ssd_chunk_coupling_kernel(*args)
            problems.append(f"{why}: no ValueError")
        except ValueError:
            pass
    # A cached factory must return the very same compiled artefact.
    first = _launch(1, 2, 1, 64, 64, "float16")
    again = _launch(1, 2, 1, 64, 64, "float16")
    if first is not again:
        problems.append("factory cache miss for a repeated geometry")
    ok = not problems
    _record(level, name, ok, "; ".join(problems) if problems else "6 rejects + cache identity")
    print("       6 unsupported contracts rejected, cached artefact reused", flush=True)
    return ok


def _case_allocation_contract(level, name, batch, num_chunks, chunk_len, n_groups, d_state,
                              dtype=_DEFAULT_DTYPE, seed=0):
    """``out_idx=-1``: every call returns a fresh buffer and leaves the old one intact.

    Mirrors ``test_ssd_chunk_coupling_fwd_cached`` in
    ``examples/TileOPs/tests/ops/test_mamba.py``.
    """
    c, b = _inputs(batch, num_chunks, chunk_len, n_groups, d_state, dtype, seed)
    fn = _launch(batch, num_chunks, n_groups, chunk_len, d_state, dtype)
    c_dev, b_dev = c.to("npu"), b.to("npu")
    first = fn(c_dev, b_dev)
    first_cpu = first.cpu()
    second = fn(-c_dev, b_dev)
    torch.npu.synchronize()
    second_cpu, first_after = second.cpu(), first.cpu()
    problems = []
    if first.data_ptr() == second.data_ptr():
        problems.append("second call reused the first output allocation")
    if not torch.equal(first_cpu, first_after):
        problems.append("second call mutated the first output")
    ok2, diff2, detail2 = _compare(second_cpu, golden_ssd_chunk_coupling(-c, b, chunk_len))
    ok1, diff1, _ = _compare(first_after, golden_ssd_chunk_coupling(c, b, chunk_len))
    if not ok1:
        problems.append(f"first output drifted (max_diff {diff1:.3e})")
    if not ok2:
        problems.append(f"second output wrong (max_diff {diff2:.3e}; {detail2})")
    ok = not problems
    _record(level, name, ok, "; ".join(problems) if problems else
            f"fresh buffer, first intact, max_diff={max(diff1, diff2):.3e}")
    return ok


# ---------------------------------------------------------------------------
# 5. Layered cases
# ---------------------------------------------------------------------------
def run_L0():
    """Gate: 4 manifest workloads + mask semantics + the trips=0/trips=2 paths."""
    # First device contact is the smoke pair with its diagonal assertions: the
    # AIV sub-block direction and the mask direction are proven before anything
    # heavier runs (DESIGN.md 8.3 / REVIEW.md hard prerequisite).
    run_case("L0", "L0-1 manifest-smoke-fp16", 1, 2, 64, 1, 64, "float16", seed=1, repeat=2)
    run_case("L0", "L0-2 manifest-smoke-bf16", 1, 2, 64, 1, 64, "bfloat16", seed=2, repeat=2)
    _case_mask_semantics("L0", "L0-3 mask-diagonal-and-upper", 1, 2, 64, 1, 64, "float16", seed=3)
    # Manifest workloads, verbatim from examples/TileOPs/tileops/manifest/mamba.yaml.
    run_case("L0", "L0-4 mamba2-780m-fp16", 1, 16, 256, 1, 128, "float16", seed=4)
    run_case("L0", "L0-5 mamba2-2p7b-bf16", 4, 8, 256, 1, 128, "bfloat16", seed=5)
    run_case("L0", "L0-6 mamba2-grouped-prefill-fp16", 3, 12, 256, 8, 128, "float16", seed=6)
    # trips = 0 (strictly-upper tiles) and trips = 2 on the same grid, BL=128.
    _case_mask_semantics("L0", "L0-7 trips-0-and-2-bl128", 1, 2, 256, 1, 128, "float16", seed=7)
    # chunk_len % 32 != 0 -> rejected; chunk_len=96 -> BL=32 positive path.
    _case_contract("L0", "L0-8 chunk-len-contract")
    run_case("L0", "L0-9 chunk-len-96-bl32", 1, 2, 96, 1, 64, "float16", seed=9)
    _finish("L0", strict=True)


def _case_zero_inputs(level, name, batch, num_chunks, chunk_len, n_groups, d_state,
                      dtype=_DEFAULT_DTYPE, seed=0):
    """All-zero inputs must give an all-zero output (no uninitialised tile leaks)."""
    c, b, out = _run_raw(batch, num_chunks, chunk_len, n_groups, d_state, dtype, seed, 0.0)
    n_nonzero = int((out != 0).sum())
    ok = n_nonzero == 0
    _record(level, name, ok, f"{n_nonzero} non-zero element(s) for all-zero inputs")
    return ok


def _finish(level, strict):
    rows = [r for r in _RESULTS if r[0] == level]
    failed = [r for r in rows if not r[2]]
    print(
        f"[{level}] {len(rows) - len(failed)}/{len(rows)} cases passed, "
        f"max_diff={_MAX_DIFF[0]:.6g}",
        flush=True,
    )
    if failed and strict:
        raise AssertionError(f"[{level}] {len(failed)} blocking case(s) failed")
    if failed:
        print(f"[{level}][WARN] {len(failed)} non-blocking case(s) failed", flush=True)


def run_L1():
    """Representative shapes: every manifest geometry on the other dtype + contracts."""
    run_case("L1", "L1-1 bf16-780m-shape", 1, 16, 256, 1, 128, "bfloat16", seed=11, repeat=2)
    run_case("L1", "L1-2 fp16-2p7b-shape", 4, 8, 256, 1, 128, "float16", seed=12, repeat=2)
    run_case("L1", "L1-3 bf16-grouped-prefill", 3, 12, 256, 8, 128, "bfloat16", seed=13)
    run_case("L1", "L1-4 fp16-g4-n64", 2, 4, 256, 4, 64, "float16", seed=14, repeat=2)
    run_case("L1", "L1-5 fp16-q64-g8", 2, 8, 64, 8, 64, "float16", seed=15, repeat=2)
    run_case("L1", "L1-6 bf16-q128-n64", 2, 4, 128, 2, 64, "bfloat16", seed=16)
    run_case("L1", "L1-7 fp16-noncontiguous", 1, 2, 64, 1, 64, "float16", seed=17,
             noncontiguous=True)
    _case_allocation_contract("L1", "L1-8 out-idx-allocation-contract", 1, 2, 64, 1, 64,
                              "float16", seed=18)
    _finish("L1", strict=True)


def run_L2():
    """Larger informative geometries (non-blocking; the golden is CPU einsum)."""
    run_case("L2", "L2-1 fp16-b2-s8192-g2", 2, 32, 256, 2, 128, "float16", seed=21)
    run_case("L2", "L2-2 bf16-b6-s3072", 6, 12, 256, 1, 128, "bfloat16", seed=22)
    run_case("L2", "L2-3 fp16-b8-s2048-g8", 8, 8, 256, 8, 128, "float16", seed=23)
    run_case("L2", "L2-4 fp16-long-b1-s16384", 1, 64, 256, 1, 128, "float16", seed=24)
    _finish("L2", strict=False)


def run_boundary():
    """Sub-tile sizes, K tails, group strides, zero inputs, magnitude extremes."""
    run_case("boundary", "B-1 q32-bl32", 1, 4, 32, 1, 64, "float16", seed=31)
    run_case("boundary", "B-2 q96-bl32-n96-ktail", 1, 2, 96, 1, 96, "float16", seed=32)
    run_case("boundary", "B-3 n16-single-k-fractal", 1, 2, 64, 1, 16, "float16", seed=33)
    run_case("boundary", "B-4 bf16-q64-g8-b2", 2, 4, 64, 8, 64, "bfloat16", seed=34)
    run_case("boundary", "B-5 noncontiguous-strided", 2, 3, 64, 2, 64, "bfloat16", seed=35,
             noncontiguous=True)
    _case_zero_inputs("boundary", "B-6 zero-inputs", 1, 2, 64, 1, 64, "float16", seed=36)
    run_case("boundary", "B-7 fp16-large-magnitude", 1, 2, 64, 1, 64, "float16", seed=37,
             scale=8.0)
    run_case("boundary", "B-8 single-chunk-q256", 1, 1, 256, 1, 128, "float16", seed=38)
    run_case("boundary", "B-9 bf16-q128-g1-n64", 1, 2, 128, 1, 64, "bfloat16", seed=39)
    run_case("boundary", "B-10 q64-g4-n128", 1, 4, 64, 4, 128, "float16", seed=40)
    _finish("boundary", strict=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0",
                        choices=["L0", "L1", "L2", "boundary", "all"])
    args, _ = parser.parse_known_args()
    levels = {"L0": run_L0, "L1": run_L1, "L2": run_L2, "boundary": run_boundary}
    chosen = list(levels.values()) if args.level == "all" else [levels[args.level]]
    started = time.perf_counter()
    for run in chosen:
        run()
    print(f"[TIME] {time.perf_counter() - started:.1f} s for level={args.level}", flush=True)
    print(f"[PRECISION_PASS] max_diff={_MAX_DIFF[0]:.6g} ({_MAX_WHERE[0]})", flush=True)


if __name__ == "__main__":
    main()
