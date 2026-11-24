# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Repro: UB vector store -> vector load needs an explicit memory barrier.
First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-23; origin_task batch2-20261123.

Target: ascend-950pr-9579 / dav-3510, tilelang 0.1.15 + CANN 9.2.1.

Found while implementing the Engram GateConv forward, whose single launch keeps a
row window in UB and reads it back in the same kernel.  Every symptom traced to
one rule:

    a vector store to UB is not visible to a later vector load of the same UB
    buffer unless T.simd.mem_bar("VST_VLD") sits between them.

Observed symptoms without the barrier (run-to-run nondeterministic, so they are
*reported* rather than asserted):

  * ``vsts(dist="PK_B32")`` (16-bit pack store) followed by ``vld(dist="UNPK_B16")``
    returned an all-zero row (64/64 lanes zero) in one build and the correct row
    in another;
  * a single-lane store (``PAT_VL1`` + ``extent=1``) read back with
    ``vld(dist="BRC_B32")`` returned a stale scalar, which scaled a whole row by
    1.38x and made the convolution window wrong;
  * the same fp32 store/load window read produced different results on
    consecutive launches of one compiled kernel.

Assertions below are the positive evidence:

  A  ``PK_B32`` store + DMA copy out is exact (the bytes written are correct).
  B2 the same 16-bit row read back with ``UNPK_B16`` is exact *after* mem_bar.
  C  fp32 ``vsts`` -> mem_bar -> ``vld`` is exact.
  D  single-lane store -> mem_bar -> ``BRC_B32`` returns the stored scalar.

B (no barrier) is informational: it prints what the read returned this run.

run:
  cd <task dir>
  source <repo>/.agents/platforms/toolchains/ascend/env-950pr.sh
  ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python \
      repro/TRAP-950-simd-ub-store-load.py
"""

import torch
import tilelang
import tilelang.ascend.language as T

N = 64


@tilelang.jit(out_idx=[1], target="ascend")
def build_pack_then_dma():
    """A: 16-bit pack store, then DMA copy out."""

    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), OUT: T.Tensor((N,), "float16")):
        with T.Kernel(1) as bx:
            x_ub = T.alloc_shared((N,), "float32")
            h_ub = T.alloc_shared((N,), "float16")
            T.copy(X, x_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                v = T.simd.vld(x_ub[0])
                nh = T.simd.vcvt(v, "float16", round="ROUND_R", sat=False, part=0)
                T.simd.vsts(h_ub[0], nh, full, dist="PK_B32")
            T.copy(h_ub, OUT)

    return main


@tilelang.jit(out_idx=[1], target="ascend")
def build_pack_then_unpack(barrier: bool):
    """B/B2: 16-bit pack store, then UNPK_B16 read back (with or without mem_bar)."""

    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), OUT: T.Tensor((N,), "float32")):
        with T.Kernel(1) as bx:
            x_ub = T.alloc_shared((N,), "float32")
            h_ub = T.alloc_shared((N,), "float16")
            o_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                v = T.simd.vld(x_ub[0])
                nh = T.simd.vcvt(v, "float16", round="ROUND_R", sat=False, part=0)
                T.simd.vsts(h_ub[0], nh, full, dist="PK_B32")
                if barrier:
                    T.simd.mem_bar("VST_VLD")
                back = T.simd.vcvt(T.simd.vld(h_ub[0], dist="UNPK_B16"), "float32", part=0)
                T.simd.vsts(o_ub[0], back, full)
            T.copy(o_ub, OUT)

    return main


@tilelang.jit(out_idx=[1], target="ascend")
def build_fp32_roundtrip():
    """C: fp32 store -> mem_bar -> fp32 load."""

    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), OUT: T.Tensor((N,), "float32")):
        with T.Kernel(1) as bx:
            x_ub = T.alloc_shared((N,), "float32")
            f_ub = T.alloc_shared((N,), "float32")
            o_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                v = T.simd.vld(x_ub[0])
                T.simd.vsts(f_ub[0], v, full)
                T.simd.mem_bar("VST_VLD")
                T.simd.vsts(o_ub[0], T.simd.vld(f_ub[0]), full)
            T.copy(o_ub, OUT)

    return main


@tilelang.jit(out_idx=[1], target="ascend")
def build_scalar_broadcast():
    """D: single-lane store -> mem_bar -> BRC_B32 read of that scalar."""

    @T.prim_func
    def main(R: T.Tensor((N,), "float32"), OUT: T.Tensor((N,), "float32")):
        with T.Kernel(1) as bx:
            r_ub = T.alloc_shared((N,), "float32")
            o_ub = T.alloc_shared((N,), "float32")
            s_ub = T.alloc_shared((8,), "float32")
            T.copy(R, r_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                scalar = T.simd.vld(r_ub[0])
                T.simd.vsts(s_ub[0], scalar, one, extent=1)
                T.simd.mem_bar("VST_VLD")
                bcast = T.simd.vld(s_ub[0], dist="BRC_B32")
                T.simd.vsts(o_ub[0], bcast, full)
            T.copy(o_ub, OUT)

    return main


def _out(fn, *args):
    """out_idx returns a list for several outputs and a bare tensor for one."""
    got = fn(*args)
    if isinstance(got, (list, tuple)):
        got = got[0]
    return got.cpu().float()


def main():
    fail = []

    x = (torch.arange(1, N + 1, dtype=torch.float32) * 0.011).npu()
    want16 = x.cpu().half().float()

    got16 = _out(build_pack_then_dma(), x)
    ok_a = torch.equal(got16, want16)
    print(f"[A]  PK_B32 store + DMA copy out          exact={ok_a}")
    if not ok_a:
        fail.append("A")

    got_b = _out(build_pack_then_unpack(False), x)
    print(
        f"[B]  PK_B32 store + UNPK_B16 (no barrier)  informational: "
        f"exact={torch.equal(got_b, want16)} nonzero_lanes={int((got_b != 0).sum())}/64"
    )

    got_b2 = _out(build_pack_then_unpack(True), x)
    ok_b2 = torch.equal(got_b2, want16)
    print(f"[B2] PK_B32 store + mem_bar + UNPK_B16    exact={ok_b2}")
    if not ok_b2:
        fail.append("B2")

    got_c = _out(build_fp32_roundtrip(), x)
    ok_c = torch.equal(got_c, x.cpu())
    print(f"[C]  fp32 store + mem_bar + fp32 load     exact={ok_c}")
    if not ok_c:
        fail.append("C")

    r = torch.arange(1, N + 1, dtype=torch.float32).npu()
    got_d = _out(build_scalar_broadcast(), r)
    ok_d = bool((got_d == r.cpu()[0]).all())
    print(f"[D]  single-lane store + mem_bar + BRC    broadcast={ok_d} (value {float(got_d[0])})")
    if not ok_d:
        fail.append("D")

    print("REPRO_RESULT:", "ALL_ASSERTIONS_HOLD" if not fail else f"FAILED {fail}")
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
