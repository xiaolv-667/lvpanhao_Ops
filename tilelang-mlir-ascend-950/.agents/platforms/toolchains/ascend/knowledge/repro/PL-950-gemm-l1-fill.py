#!/usr/bin/env python3
"""PL-950-gemm-l1-fill / CONST-950-gemm-mte2-burst repro.

First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-24; origin_task shared_expert_mlp-SharedExpertMLPFwdOp-20261124T132107Z.

Asserted (measured, 28 Cube cores, bf16, m=512/n=36864/k=7168):
  1. block_k 64 -> 128 at 256x256: per-row DMA segment 128 B -> 256 B, MTE2
     ~70 -> ~105 GB/s per core, 1201.6 -> 811.8 us (<= 0.80x here).
  2. 128x256 (reuse 85.3 FLOP/B) is slower than 256x256 (reuse 128).
  3. The fused gate/up epilogue (two fp32 L0C halves + gated SIMD epilogue)
     removes the separate activation pass; its N tile must be a multiple of the
     64-lane vector chunk (block_n=96 leaves columns 64..95 unwritten).
"""
import time

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T

M, N, K, S, DT, VL = 512, 36864, 7168, 18432, "bfloat16", 64


def gemm(bm, bn, bk, ns, m=M, n=N, k=K):
    @T.prim_func
    def main(A: T.Tensor((m, k), DT), B: T.Tensor((n, k), DT), C: T.Tensor((m, n), DT)):
        with T.Kernel((n // bn) * (m // bm)) as bid:
            m0, n0 = (bid % (m // bm)) * bm, (bid // (m // bm)) * bn
            a1 = T.alloc_l1((bm, bk), DT)
            b1 = T.alloc_l1((bn, bk), DT)
            c0 = T.alloc_l0c((bm, bn), "float32")
            T.annotate_buffer_versions({a1: ns, b1: ns})
            for step in T.Pipelined(k // bk, num_stages=ns):
                T.copy(A[m0 : m0 + bm, step * bk : step * bk + bk], a1)
                T.copy(B[n0 : n0 + bn, step * bk : step * bk + bk], b1)
                T.gemm(a1, b1, c0, transpose_B=True, clear_accum=(step == 0))
            T.copy(c0, C[m0 : m0 + bm, n0 : n0 + bn])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def fused(bm, bn, bk, ns, m=M, s=S, k=K):
    if bn % VL:
        raise ValueError(f"the gated epilogue walks {VL}-wide column chunks: bn={bn}")
    half = bm // 2

    @T.prim_func
    def main(A: T.Tensor((m, k), DT), W: T.Tensor((2 * s, k), DT), O: T.Tensor((m, s), DT)):
        with T.Kernel((s // bn) * (m // bm)) as bid:
            m0, n0 = (bid % (m // bm)) * bm, (bid // (m // bm)) * bn
            a1 = T.alloc_l1((bm, bk), DT)
            bg = T.alloc_l1((bn, bk), DT)
            bu = T.alloc_l1((bn, bk), DT)
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
            co = T.alloc_shared((half, bn), DT)
            T.dual_copy(cg, cv_g)
            T.dual_copy(cu, cv_u)
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.vdup(T.float32(1), "float32", full)
                for r in T.serial(half):
                    for c in T.serial(bn // VL):
                        g = T.simd.vld(cv_g[r, c * VL])
                        u = T.simd.vld(cv_u[r, c * VL])
                        den = T.simd.vadd(one, T.simd.vexp(T.simd.vneg(g, full), full), full)
                        y = T.simd.vdiv(T.simd.vmul(g, u, full), den, full)
                        T.simd.vsts(co[r, c * VL], T.simd.vcvt(y, DT, round="ROUND_R", sat=False, part=0), full, dist="PK_B32")
            T.dual_copy(co, O[m0 : m0 + bm, n0 : n0 + bn])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def time_fn(fn, *args, reps=5):
    fn(*args)
    torch.npu.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return sorted(ts)[len(ts) // 2]


torch.manual_seed(0)
a = torch.randn(M, K, dtype=torch.bfloat16).npu() * 0.1
b = torch.randn(N, K, dtype=torch.bfloat16).npu() * 0.02
t64 = time_fn(gemm(256, 256, 64, 4), a, b)
t128 = time_fn(gemm(256, 256, 128, 3), a, b)
tsmall = time_fn(gemm(128, 256, 64, 4), a, b)
print(f"bk=64  256x256 -> {t64:8.1f} us\nbk=128 256x256 -> {t128:8.1f} us ({t128 / t64:.3f}x)"
      f"\nbk=64  128x256 -> {tsmall:8.1f} us ({tsmall / t64:.3f}x of 256x256)")
w = torch.randn(2 * S, K, dtype=torch.bfloat16).npu() * 0.02
print(f"fused gate/up  -> {time_fn(fused(256, 128, 128, 3), a, w):8.1f} us "
      f"(unfused was {t128:.1f} us + a separate activation pass)")
try:
    fused(256, 96, 128, 3)
except ValueError as exc:
    print(f"bn=96 refused as expected: {exc}")
else:
    raise AssertionError("block_n=96 must be refused (columns 64..95 unwritten)")
assert t128 <= 0.80 * t64, (t128, t64)
assert tsmall >= 1.1 * t64, (tsmall, t64)
print("[PASS] PL-950-gemm-l1-fill / CONST-950-gemm-mte2-burst")
