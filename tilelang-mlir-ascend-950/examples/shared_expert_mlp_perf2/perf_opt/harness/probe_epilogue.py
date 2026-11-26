"""Probe: the fused gate/up epilogue variants (wall clock, screening only).

The epilogue is the only AIV work in the gate/up kernel and it runs *after* the
K loop inside the same block, so its cost adds to the block time.  The Stage 4
version uses the precise fp32 `vexp`/`vdiv` (``tl.enable_fast_math`` is off by
default), which are the slow library paths.  This probe isolates:

  v0  precise exp/div, serial chunks        (the Stage 4 shape)
  v1  SFU exp/div  (precision="ftz_true")
  v2  v1 + 2-way independent chunk unroll
  v3  v2 with num_stages=4

Screening numbers only; anything that wins is re-measured through
``capture.py`` (msopprof Task Duration) before it is recorded.

Usage::

    python probe_epilogue.py --impl ../SharedExpertMLPFwdOp.py [--m 512]
"""

import argparse
import importlib.util
import os
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T

VL = 64
ITEMSIZE = {"float16": 2, "bfloat16": 2, "float32": 4}
L0C_BYTES, L1_BYTES, UB_BYTES = 262144, 524288, 253952


def gate_up_fused(m, s, k, dtype, bm, bn, bk, ns, variant="v0"):
    half = bm // 2
    nchunk = bn // VL
    assert bn % VL == 0 and 2 * bm * bn * 4 <= L0C_BYTES
    assert (bm + 2 * bn) * bk * ITEMSIZE[dtype] * ns <= L1_BYTES
    prec = "ftz_true" if variant != "v0" else None

    @T.prim_func
    def main(A: T.Tensor((m, k), dtype), W: T.Tensor((2 * s, k), dtype),
             O: T.Tensor((m, s), dtype)):
        with T.Kernel((s // bn) * (m // bm)) as bid:
            bx = bid // (m // bm)
            by = bid % (m // bm)
            m0, n0 = by * bm, bx * bn
            a1 = T.alloc_l1((bm, bk), dtype)
            bg = T.alloc_l1((bn, bk), dtype)
            bu = T.alloc_l1((bn, bk), dtype)
            cg = T.alloc_l0c((bm, bn), "float32")
            cu = T.alloc_l0c((bm, bn), "float32")
            T.annotate_buffer_versions({a1: ns, bg: ns, bu: ns})
            for step in T.Pipelined(k // bk, num_stages=ns):
                k0 = step * bk
                T.copy(A[m0 : m0 + bm, k0 : k0 + bk], a1)
                T.copy(W[n0 : n0 + bn, k0 : k0 + bk], bg)
                T.copy(W[s + n0 : s + n0 + bn, k0 : k0 + bk], bu)
                T.gemm(a1, bg, cg, transpose_B=True, clear_accum=(step == 0))
                T.gemm(a1, bu, cu, transpose_B=True, clear_accum=(step == 0))
            cv_g = T.alloc_shared((half, bn), "float32")
            cv_u = T.alloc_shared((half, bn), "float32")
            co = T.alloc_shared((half, bn), dtype)
            T.dual_copy(cg, cv_g)
            T.dual_copy(cu, cv_u)
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.vdup(T.float32(1), "float32", full)
                if variant in ("v0", "v1"):
                    for r in T.serial(half):
                        for c in T.serial(nchunk):
                            g = T.simd.vld(cv_g[r, c * VL])
                            u = T.simd.vld(cv_u[r, c * VL])
                            num = T.simd.vmul(g, u, full)
                            den = T.simd.vadd(
                                one, T.simd.vexp(T.simd.vneg(g, full), full, precision=prec), full
                            )
                            y = T.simd.vdiv(num, den, full, precision=prec)
                            T.simd.vsts(co[r, c * VL],
                                        T.simd.vcvt(y, dtype, round="ROUND_R", sat=False, part=0),
                                        full, dist="PK_B32")
                else:  # v2 / v3: two independent chains per iteration body
                    for r in T.serial(half):
                        g0 = T.simd.vld(cv_g[r, 0])
                        u0 = T.simd.vld(cv_u[r, 0])
                        g1 = T.simd.vld(cv_g[r, VL])
                        u1 = T.simd.vld(cv_u[r, VL])
                        n0v = T.simd.vmul(g0, u0, full)
                        n1v = T.simd.vmul(g1, u1, full)
                        d0 = T.simd.vadd(one, T.simd.vexp(T.simd.vneg(g0, full), full,
                                                          precision=prec), full)
                        d1 = T.simd.vadd(one, T.simd.vexp(T.simd.vneg(g1, full), full,
                                                          precision=prec), full)
                        y0 = T.simd.vdiv(n0v, d0, full, precision=prec)
                        y1 = T.simd.vdiv(n1v, d1, full, precision=prec)
                        T.simd.vsts(co[r, 0],
                                    T.simd.vcvt(y0, dtype, round="ROUND_R", sat=False, part=0),
                                    full, dist="PK_B32")
                        T.simd.vsts(co[r, VL],
                                    T.simd.vcvt(y1, dtype, round="ROUND_R", sat=False, part=0),
                                    full, dist="PK_B32")

            T.dual_copy(co, O[m0 : m0 + bm, n0 : n0 + bn])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def time_fn(fn, *args, reps=5):
    out = fn(*args)
    torch.npu.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--s", type=int, default=18432)
    ap.add_argument("--k", type=int, default=7168)
    args = ap.parse_args()
    m, s, k = args.m, args.s, args.k
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.bfloat16).npu() * 0.1
    w = torch.randn(2 * s, k, dtype=torch.bfloat16).npu() * 0.02

    ref = None
    for variant, ns in (("v0", 3), ("v1", 3), ("v2", 3), ("v3", 4)):
        try:
            fn = gate_up_fused(m, s, k, "bfloat16", 256, 128, 128, ns, variant)
            us, out = time_fn(fn, a, w)
            gt = (w[:s].float() @ a.float().t()).t()
            up = (w[s:].float() @ a.float().t()).t()
            import torch.nn.functional as F
            want = (F.silu(gt) * up)
            got = out.float()
            nse = ((got - want) ** 2).sum().item() / (want**2).sum().item()
            print(f"{variant} ns={ns}: {us:8.1f} us  nse={nse:.3e}")
            if ref is None:
                ref = us
            else:
                print(f"            vs v0: {100.0 * (us / ref - 1):+.2f}%")
        except Exception as exc:  # noqa: BLE001
            print(f"{variant} ns={ns}: REJECTED {type(exc).__name__}: {str(exc)[:160]}")


if __name__ == "__main__":
    main()
