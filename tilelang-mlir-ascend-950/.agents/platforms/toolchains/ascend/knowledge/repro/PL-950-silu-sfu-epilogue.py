#!/usr/bin/env python3
"""PL-950-silu-sfu-epilogue / VP-950-SIMD-SFU-PRECISION / CONST-950-grid-tail-wave repro.

Entries: PL-950-silu-sfu-epilogue (pattern), CONST-950-grid-tail-wave (constant),
VP-950-SIMD-SFU-PRECISION / VP-950-GEMM-TAIL-WAVE (evidence rows of the same task).

First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-25; origin_task shared_expert_mlp perf2 (Stage 4 round 2).

Two measured facts:

1. In a mixed Cube+Vector kernel the gated epilogue runs *after* the K loop, so
   its cost is added to the block time.  ``T.simd.vexp``/``vdiv`` with
   ``precision=None`` follow ``tl.enable_fast_math`` (off by default) and take
   the precise 0-ulp sequences; ``precision="ftz_true"`` (bare SFU) is ~3.2%
   faster on the whole gate/up kernel (842.7 -> 815.6 us at
   m=512, S=18432, k=7168, bf16) with <= 1 bf16 ulp of output change.
2. Tail waves: the same kernel costs 5.66 us/tile at 288 blocks (10.29 waves on
   28 Cube cores) but 5.35/5.33 us/tile at 224/252 blocks (8/9 whole waves),
   i.e. the partial last wave costs ~6.2% and cannot be removed by tile choice
   (tiles must be multiples of 16).

Run it from a directory that is NOT the repository root (the repo's own
``tilelang/`` source tree shadows the installed wheel) -- e.g. ``cd /tmp``:

Run (task platform prototype command):
    bash -c 'source .agents/platforms/toolchains/ascend/env-950pr.sh && \
             export TILELANG_CACHE_DIR=/tmp/tl-repro && \
             python .agents/platforms/toolchains/ascend/knowledge/repro/PL-950-silu-sfu-epilogue.py'
"""
import statistics
import time

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T

VL = 64


def gate_up_fused(m, s, k, bm, bn, bk, ns, sfu):
    half = bm // 2
    prec = "ftz_true" if sfu else None

    @T.prim_func
    def main(A: T.Tensor((m, k), "bfloat16"), W: T.Tensor((2 * s, k), "bfloat16"),
             O: T.Tensor((m, s), "bfloat16")):
        with T.Kernel((s // bn) * (m // bm)) as bid:
            m0, n0 = (bid % (m // bm)) * bm, (bid // (m // bm)) * bn
            a1 = T.alloc_l1((bm, bk), "bfloat16")
            bg = T.alloc_l1((bn, bk), "bfloat16")
            bu = T.alloc_l1((bn, bk), "bfloat16")
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
            co = T.alloc_shared((half, bn), "bfloat16")
            T.dual_copy(cg, cv_g)
            T.dual_copy(cu, cv_u)
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.vdup(T.float32(1), "float32", full)
                for r in T.serial(half):
                    for c in T.serial(bn // VL):
                        g = T.simd.vld(cv_g[r, c * VL])
                        u = T.simd.vld(cv_u[r, c * VL])
                        den = T.simd.vadd(one, T.simd.vexp(T.simd.vneg(g, full), full,
                                                           precision=prec), full)
                        y = T.simd.vdiv(T.simd.vmul(g, u, full), den, full, precision=prec)
                        T.simd.vsts(co[r, c * VL],
                                    T.simd.vcvt(y, "bfloat16", round="ROUND_R", sat=False, part=0),
                                    full, dist="PK_B32")
            T.dual_copy(co, O[m0 : m0 + bm, n0 : n0 + bn])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def median_us(fn, args, reps=5):
    fn(*args)
    torch.npu.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


M, S, K = 512, 18432, 7168
torch.manual_seed(0)
a = torch.randn(M, K, dtype=torch.bfloat16).npu() * 0.1
w = torch.randn(2 * S, K, dtype=torch.bfloat16).npu() * 0.02

precise = gate_up_fused(M, S, K, 256, 128, 128, 3, False)
sfu = gate_up_fused(M, S, K, 256, 128, 128, 3, True)
t0 = median_us(precise, (a, w))
t1 = median_us(sfu, (a, w))
o0, o1 = precise(a, w), sfu(a, w)
torch.npu.synchronize()
ulp = (o0.view(torch.int16).to(torch.int32) - o1.view(torch.int16).to(torch.int32)).abs().max().item()
print(f"precise exp/div -> {t0:8.1f} us ; SFU -> {t1:8.1f} us  ({100 * (t1 / t0 - 1):+.2f}%)")
print(f"max |bf16 bit diff| = {ulp}")
assert t1 <= 0.98 * t0, (t0, t1)
assert ulp <= 1, ulp

print("tail wave (same kernel, 28 Cube cores):")
for S2 in (14336, 16128, 18432):
    w2 = torch.randn(2 * S2, K, dtype=torch.bfloat16).npu() * 0.02
    fn = gate_up_fused(M, S2, K, 256, 128, 128, 3, True)
    t = median_us(fn, (a, w2), reps=4)
    tiles = S2 // 128
    print(f"  S={S2} blocks={2 * tiles} waves={2 * tiles / 28:.2f} {t:8.1f} us {t / tiles:6.2f} us/tile")
print("[PASS] PL-950-silu-sfu-epilogue")
