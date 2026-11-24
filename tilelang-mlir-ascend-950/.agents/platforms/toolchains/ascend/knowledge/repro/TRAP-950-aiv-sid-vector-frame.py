"""TRAP-950-aiv-sid-vector-frame: `with T.Vector(vector=2) as sid` silently
First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-23; origin_task batch2-20261123.
yields a wrong AIV sub-block id -- the kernel compiles and runs, the numbers drift.

Found while implementing ``_ssd_chunk_coupling_kernel`` (stage 3, probe P-epl).

Context
-------
The causal-mask epilogue of that kernel splits one L0C accumulator between the two
AIVs with ``T.dual_copy(acc, cv)`` (2:1 M split) and then needs the AIV's own half
index inside a ``T.SimtVF`` to rebuild the absolute row:

    co[i, j] = (j - i) <= (l0 + sid * BL/2 - s0) ? cv[i, j] : 0

DESIGN.md 3.3 spelled that as

    with T.Vector(vector=2) as sid:
        with T.SimtVF(threads=128):
            ...
        T.dual_copy(co, cb[...])

and the compile-only design probe (P10/P11/P12) produced a tidy ``__mix__(1, 2)``
kernel with ``asc_copy_l0c2ub(dual_dst_mode=1)`` and a per-lane SIMT predicate, so
nothing at compile time rejects it.

What actually happens
---------------------
Measured on Ascend950PR_9579, TileLang ascend 0.1.15 / CANN 9.2.1:

* ``T.Vector(vector=2) as sid`` -> **wrong values, nondeterministically**.  On this
  geometry 5 of 8 launches return corrupted lower-triangle cells (a whole 64-row
  half is stale) and 7 distinct results are seen for 8 launches of the same kernel
  on the same inputs.  The strictly-upper triangle is still exactly 0, so a
  "does the output look causal / is it close" spot check can miss it entirely.
* ``sid = T.call_extern("int64", "asc_get_sub_block_id")`` + the same ``T.SimtVF``
  and ``T.dual_copy`` sequence: exact on all 8 launches and bit-identical.

Note the scope: on a *single-block* grid (``grid == 1``) both spellings happened to
be correct here, so the corruption surfaces when several blocks are scheduled per
core -- i.e. exactly the multi-wave regime of a real workload.  ``T.Vector`` is
therefore never a valid way to read the sub-block id in this toolchain.

Relation to CONST-950-aiv-sid-capture (MLA bwd): that entry promotes the extern
call because ``T.Vector`` was rejected at compile time with ``Need input to be in
SSA form dup sid``.  Here the frame is accepted, which is strictly more dangerous:
the failure is numeric, not a compile error.  Gate any new sid user on a numeric
assertion (below), not on "it compiled".

How to detect it in a new kernel
--------------------------------
Encode the tile coordinates into the operands so every output cell identifies its
own (l, s): with A row ``l = e_l`` and B row ``s = 16*(s%64) + n`` the coupling
matrix must be exactly ``16*(s%64) + l`` below the diagonal -- all values stay
below 2^11, so fp16 represents them exactly and no tolerance is needed.  A per-cell
comparison then localises the broken half immediately.

Run:

    source .agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python \
        examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel/repro/TRAP-950-aiv-sid-vector-frame.py
"""

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

DTYPE = "float16"
Q, NC, G, N = 256, 2, 8, 128  # grid = NC*G*(Q/BL)^2 = 64 blocks, BL = 128
BL, BK = 128, 64
TRIALS = 8


def build(mode):
    """One batch element of the coupling kernel; ``mode`` selects the sid source."""
    nlt = nst = Q // BL
    grid = NC * G * nlt * nst
    k_trips = N // BK

    @T.prim_func
    def main(
        C_mat: T.Tensor((1, NC * Q, G, N), DTYPE),
        B_mat: T.Tensor((1, NC * Q, G, N), DTYPE),
        cb: T.Tensor((1, NC, G, Q, Q), DTYPE),
    ):
        with T.Kernel(grid) as bid:
            bcg = bid // (nlt * nst)
            rem = bid % (nlt * nst)
            bc = bcg // G
            bg = bcg % G
            l0 = (rem // nst) * BL
            s0 = (rem % nst) * BL
            cl1 = T.alloc_l1((BL, BK), DTYPE)
            bl1 = T.alloc_l1((BL, BK), DTYPE)
            x0 = T.alloc_l0a((BL, BK), DTYPE)
            y0 = T.alloc_l0b((BL, BK), DTYPE)
            acc = T.alloc_l0c((BL, BL), "float32")
            cv = T.alloc_shared((BL // 2, BL), "float32")
            co = T.alloc_shared((BL // 2, BL), DTYPE)
            T.annotate_buffer_versions({cl1: 2, bl1: 2})
            trips = T.if_then_else(s0 <= l0 + BL - 1, k_trips, 0)
            for kk in T.Pipelined(trips, num_stages=2):
                n0 = kk * BK
                nk = T.min(N - n0, BK)
                T.copy(C_mat[0, bc * Q + l0 : bc * Q + l0 + BL, bg, n0 : n0 + nk], cl1[:, :nk])
                T.copy(B_mat[0, bc * Q + s0 : bc * Q + s0 + BL, bg, n0 : n0 + nk], bl1[:, :nk])
                T.copy(cl1[:, :nk], x0[:, :nk])
                T.copy(bl1[:, :nk], y0[:, :nk])
                T.gemm(x0[:, :nk], y0[:, :nk], acc, transpose_B=True, clear_accum=kk == 0)
            T.dual_copy(acc, cv)
            if mode == "vector_frame":
                # The trap: compiles and runs, wrong (AIV-dependent) values.
                with T.Vector(vector=2) as sid:
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(BL // 2, BL):
                            co[i, j] = T.cast(
                                T.if_then_else((j - i) <= (l0 + sid * (BL // 2) - s0),
                                               cv[i, j], T.float32(0.0)), DTYPE)
                    T.dual_copy(co, cb[0, bc, bg, l0 : l0 + BL, s0 : s0 + BL])
            else:  # "extern": the verified spelling.
                sid = T.call_extern("int64", "asc_get_sub_block_id")
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(BL // 2, BL):
                        co[i, j] = T.cast(
                            T.if_then_else(
                                (j - i) <= (l0 + T.cast(sid, "int32") * (BL // 2) - s0),
                                cv[i, j], T.float32(0.0)), DTYPE)
                T.dual_copy(co, cb[0, bc, bg, l0 : l0 + BL, s0 : s0 + BL])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def operands():
    """C row l = e_{l%N}; B row s = 16*(s%64) + n  =>  cb[l,s] = 16*(s%64) + (l%N)."""
    seq = NC * Q
    c = torch.zeros(1, seq, G, N, dtype=torch.float16)
    b = torch.zeros(1, seq, G, N, dtype=torch.float16)
    n = torch.arange(N, dtype=torch.float32)
    for row in range(seq):
        c[0, row, :, row % N] = 1.0
        b[0, row, :, :] = (n + 16.0 * (row % 64)).to(torch.float16)
    return c, b


def reference():
    idx = torch.arange(Q)
    ref = 16.0 * (idx[None, :] % 64) + (idx % N)[:, None]
    return torch.tril(ref).to(torch.float16), torch.tril(torch.ones(Q, Q, dtype=torch.bool))


def main():
    c, b = operands()
    ref, lower = reference()
    n_lower = int(lower.sum())
    c_dev, b_dev = c.to("npu"), b.to("npu")

    fn = build("extern")
    bad_runs, sums = 0, set()
    for _ in range(TRIALS):
        out = fn(c_dev, b_dev).cpu()
        torch.npu.synchronize()
        blk = out.float()[0, 0, 0]
        bad_runs += int((((blk - ref.float()).abs() > 1e-3) & lower).any())
        sums.add(round(float(blk.sum()), 3))
    print(f"extern      : {bad_runs}/{TRIALS} corrupt launches, distinct results {len(sums)}",
          flush=True)
    assert bad_runs == 0, "asc_get_sub_block_id form must be exact on every launch"
    assert len(sums) == 1, "asc_get_sub_block_id form must be deterministic"

    trap_fn = build("vector_frame")
    bad_runs, bad_cells, sums = 0, [], set()
    for _ in range(TRIALS):
        out = trap_fn(c_dev, b_dev).cpu()
        torch.npu.synchronize()
        blk = out.float()[0, 0, 0]
        cells = int((((blk - ref.float()).abs() > 1e-3) & lower).sum())
        bad_cells.append(cells)
        bad_runs += int(cells > 0)
        sums.add(round(float(blk.sum()), 3))
    print(f"vector_frame: {bad_runs}/{TRIALS} corrupt launches, wrong lower cells {bad_cells}, "
          f"distinct results {len(sums)}", flush=True)
    print(f"              (reference lower-triangle cells: {n_lower})", flush=True)
    # The trap is a silent value corruption whose exact cells vary run to run.
    assert bad_runs >= 1, "expected T.Vector(vector=2) to corrupt at least one launch"
    assert len(sums) > 1, "expected the T.Vector form to be nondeterministic"

    print("[TRAP-CONFIRMED] T.Vector(vector=2) sid corrupts values; "
          "asc_get_sub_block_id is exact and deterministic", flush=True)


if __name__ == "__main__":
    main()
