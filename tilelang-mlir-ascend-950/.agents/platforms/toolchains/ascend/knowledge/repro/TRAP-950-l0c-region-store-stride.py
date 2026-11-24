"""TRAP-950-l0c-region-store-stride: a truncated L0C *source* region corrupts the
First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-23; origin_task batch2-20261123.
fixpipe store pitch (every column past the first 16 goes wrong).

Found while implementing ``_gemm_basic_kernel`` (stage 3, M-tail cases L0-7/m=100).

Context
-------
The natural way to exclude the dead rows of an L0 accumulator is to truncate the
copy on both sides::

    T.copy(c0[:am, :an], C[m0:m0+am, n0:n0+an])

That is exactly what ``copy.cc`` documents as a "compact L0 region" (zero-start
tail dims), and it compiles.  At run time it is wrong whenever ``am`` does not
round up to the tile row count: the lowering derives the source row pitch from
the *region* instead of the buffer::

    loop_src_stride = compact_src_region.outer1->extent * info.row_frac;

so for a ``(32, 64)`` fp32 L0C tile stored with ``am = 16`` the emitted call is
``asc_copy_l0c2gm(..., 64, 16, 64, 16, ...)`` (last-but-one is the source
stride = ceil16(am) = 16) instead of ``..., 32, ...``.  Measured effect: columns
0..15 of the tile are correct and every column from 16 on is garbage
(87% of the M-tail rows wrong in the m=100/n=1000 case; the same happens for
``am = 4/8/16`` with ``block_m = 32``).

The pitch is only consistent when ``ceil16(am) == block_m``, i.e. when the tile
row count is 16 or the region misses at most its last 16-row group.  That is why
a ``block_m = 16`` kernel looks fine with the truncated form.

What this script asserts
------------------------
1. ``T.copy(c0, C[m0:m0+am, n0:n0+an])`` -- full source region, destination
   truncated -- reproduces the reference (<= 2 bf16 ulp) for ``am = 16`` and an
   M tail with ``m % block_m != 0``; the lowering then takes the burst extents
   from the destination and the source pitch from the whole tile.
2. The truncated-source form of the same program is wrong by O(10) on the tail
   rows (asserted, so a future compiler fix fails this script loudly).
3. With ``block_m = 16`` both forms agree, confirming the ``ceil16(am) ==
   block_m`` condition rather than "partial stores never work".

Run (single card, task dir or repo root; needs the ascend env):

    source .agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python \
        examples/shared_expert_mlp/_gemm_basic_kernel/repro/TRAP-950-l0c-region-store-stride.py
"""

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401 -- registers torch.npu

DTYPE = "bfloat16"
K = 128


def build(m, n, bm, bn, truncated_source):
    """One block per output tile; ``truncated_source`` selects the store form."""
    mt, nt = -(-m // bm), -(-n // bn)

    @T.prim_func
    def main(
        A: T.Tensor((m, K), DTYPE),
        B: T.Tensor((n, K), DTYPE),
        C: T.Tensor((m, n), DTYPE),
    ):
        with T.Kernel(nt * mt) as bid:
            bx = bid // mt
            by = bid % mt
            m0, n0 = by * bm, bx * bn
            am = bm if m % bm == 0 else T.min(m - m0, bm)
            an = bn if n % bn == 0 else T.min(n - n0, bn)
            a1 = T.alloc_l1((bm, 64), DTYPE)
            b1 = T.alloc_l1((bn, 64), DTYPE)
            c0 = T.alloc_l0c((bm, bn), "float32")
            T.annotate_buffer_versions({a1: 2, b1: 2})
            for step in T.Pipelined(K // 64, num_stages=2):
                k0 = step * 64
                T.copy(A[m0 : m0 + am, k0 : k0 + 64], a1[:am, :])
                T.copy(B[n0 : n0 + an, k0 : k0 + 64], b1[:an, :])
                T.gemm(a1, b1, c0, transpose_B=True, clear_accum=(step == 0))
            if truncated_source:
                T.copy(c0[:am, :an], C[m0 : m0 + am, n0 : n0 + an])
            else:
                T.copy(c0, C[m0 : m0 + am, n0 : n0 + an])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def run(m, n, bm, bn, truncated_source):
    torch.manual_seed(0)
    a = torch.randn(m, K, dtype=torch.bfloat16)
    b = torch.randn(n, K, dtype=torch.bfloat16)
    want = a.float() @ b.float().t()
    fn = build(m, n, bm, bn, truncated_source)
    got = fn(a.to("npu"), b.to("npu")).cpu().float()
    err = (got - want).abs()
    # DESIGN.md 8.2 half-precision criterion (atol = rtol = 1e-2).
    viol = int((err > 0.01 + 0.01 * want.abs()).sum().item())
    store = [
        line.strip()
        for line in fn.get_kernel_source().splitlines()
        if "asc_copy_l0c2gm" in line
    ]
    return err.max().item(), viol, store


def main():
    for m, bm in ((48, 32), (36, 32), (100, 32)):
        full, full_viol, store = run(m, n=64, bm=bm, bn=64, truncated_source=False)
        trunc, trunc_viol, store_t = run(m, n=64, bm=bm, bn=64, truncated_source=True)
        print(f"m={m} bm={bm}: full_source max_err={full:.3e} violations={full_viol}")
        print(f"m={m} bm={bm}: truncated   max_err={trunc:.3e} violations={trunc_viol}")
        print("  full-source store:", store[0][:150])
        print("  truncated   store:", store_t[0][:150])
        assert full_viol == 0, f"full-source form must match the reference, got {full:.3e}"
        assert trunc > 10.0 and trunc_viol > 0, (
            "the truncated-source form is expected to be broken here; if it passes, "
            f"the backend pitch bug was fixed -- re-check the store form (max_err={trunc:.3e})"
        )

    # block_m == 16: ceil16(am) == block_m, so both forms agree (this is why the
    # engram_decode kernel, whose tile is 16 rows, never saw the bug).
    for m in (48, 40):
        full, full_viol, _ = run(m, n=64, bm=16, bn=64, truncated_source=False)
        trunc, trunc_viol, _ = run(m, n=64, bm=16, bn=64, truncated_source=True)
        print(f"m={m} bm=16: full={full:.3e}/{full_viol} truncated={trunc:.3e}/{trunc_viol}")
        assert full_viol == 0 and trunc_viol == 0, (m, full, trunc)

    print("\033[92mTRAP-950-l0c-region-store-stride reproduced and the workaround verified\033[0m")


if __name__ == "__main__":
    main()
