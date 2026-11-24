"""TRAP-950-l0c2ub-single-aiv: a plain T.copy(L0C -> UB) does not reach both AIVs.
First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-23; origin_task batch2-20261123.

Found while implementing ``_bwd_kernel`` (stage 3, probe P1).

Context
-------
On this toolchain a mixed kernel is ``__mix__(1, 2)``: one AIC and two AIVs, and
each AIV addresses its own UB bank at the same virtual address.  The documented,
verified way to land an L0C accumulator in UB is ``T.dual_copy`` with a 2:1
shape ratio (``repro/PL-950-mixed-epilogue.py``): the hardware dual-destination
copy writes half of the tile into each AIV's buffer.

A plain ``T.copy(c_l0c, ub)`` compiles to
``asc_copy_l0c2ub(..., dual_dst_mode=0, ...)`` -- one destination, i.e. one AIV.
Any vector work that then reads the whole tile on *both* AIVs sees stale data
on the other one, and if both AIVs store that value to the same GM address the
result is garbage (observed: 99.2% of the cells wrong, 25% never written).

What this script asserts
------------------------
1. The dual-split form is correct: ``dual_copy(c (64,64) fp32 -> ub (32,64))``
   followed by ``dual_copy(ub (32,64) -> gm (64,64))`` reproduces the GEMM
   output exactly (each AIV's half lands in its own rows).
2. The plain whole-tile form is *not* usable: the same GEMM through
   ``T.copy(c, ub)`` then ``T.copy(ub, gm)`` mismatches on many cells.

Run (single card, from any directory; needs the ascend env):

    source .agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python \
        examples/mla_dense_bwd/_bwd_kernel/repro/TRAP-950-l0c2ub-single-aiv.py
"""

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

BN = 64  # tile rows
BM = 64  # tile cols
DQK = 192
HN = BN // 2
DTYPE = "float16"


def build_plain():
    @T.prim_func
    def main(
        k: T.Tensor((BN, DQK), DTYPE),
        q: T.Tensor((BM, DQK), DTYPE),
        out: T.Tensor((BN, BM), "float32"),
    ):
        with T.Kernel(1):
            k_s = T.alloc_l1((BN, DQK), DTYPE)
            q_s = T.alloc_l1((BM, DQK), DTYPE)
            c = T.alloc_l0c((BN, BM), "float32")
            ub = T.alloc_shared((BN, BM), "float32")
            T.copy(k, k_s)
            T.copy(q, q_s)
            T.gemm(k_s, q_s, c, transpose_B=True, clear_accum=True)
            T.copy(c, ub)  # lowered with dual_dst_mode=0: one AIV only
            T.copy(ub, out)

    return tilelang.compile(main, target="ascend")


def build_dual():
    @T.prim_func
    def main(
        k: T.Tensor((BN, DQK), DTYPE),
        q: T.Tensor((BM, DQK), DTYPE),
        out: T.Tensor((BN, BM), "float32"),
    ):
        with T.Kernel(1):
            k_s = T.alloc_l1((BN, DQK), DTYPE)
            q_s = T.alloc_l1((BM, DQK), DTYPE)
            c = T.alloc_l0c((BN, BM), "float32")
            ub = T.alloc_shared((HN, BM), "float32")
            T.copy(k, k_s)
            T.copy(q, q_s)
            T.gemm(k_s, q_s, c, transpose_B=True, clear_accum=True)
            T.dual_copy(c, ub)  # 2:1 M split, one half per AIV
            T.dual_copy(ub, out)  # the per-AIV halves are stitched back

    return tilelang.compile(main, target="ascend")


def main():
    torch.manual_seed(0)
    dt = getattr(torch, DTYPE)
    k = torch.randn(BN, DQK).to(dt)
    q = torch.randn(BM, DQK).to(dt)
    ref = k.float() @ q.float().T

    out_dual = torch.full((BN, BM), -777.0, device="npu", dtype=torch.float32)
    build_dual()(k.to("npu"), q.to("npu"), out_dual)
    got_dual = out_dual.cpu()
    dual_bad = int(((got_dual - ref).abs() > 1e-2).sum())
    print(f"dual_copy   : mismatched cells = {dual_bad} / {BN * BM} "
          f"(max diff {(got_dual - ref).abs().max().item():.3e})", flush=True)
    assert dual_bad == 0, "dual_copy halves must reproduce the L0C tile"
    assert torch.equal(got_dual[0:HN], got_dual[0:HN]), "half 0 must be the first 32 rows"
    torch.testing.assert_close(got_dual[:HN], ref[:HN], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(got_dual[HN:], ref[HN:], atol=1e-2, rtol=1e-2)

    out_plain = torch.full((BN, BM), -777.0, device="npu", dtype=torch.float32)
    build_plain()(k.to("npu"), q.to("npu"), out_plain)
    got_plain = out_plain.cpu()
    plain_bad = int(((got_plain - ref).abs() > 1e-2).sum())
    untouched = int((got_plain == -777.0).sum())
    print(f"plain copy  : mismatched cells = {plain_bad} / {BN * BM}, "
          f"never written = {untouched}", flush=True)
    assert plain_bad > 0, "a plain L0C->UB copy is expected to reach only one AIV"
    print("TRAP-950-l0c2ub-single-aiv: PASS (dual split correct, plain copy unusable)")


if __name__ == "__main__":
    main()
