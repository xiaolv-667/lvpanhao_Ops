"""CONST-950-clear-pad-all-versions.py -- ``T.clear`` on a multi-versioned UB buffer.

Entry: pattern-library constants -> UB budget / pad-column zeroing.
Phenomenon: ``T.clear(buf)`` on an ``alloc_shared`` buffer that
``annotate_buffer_versions`` gives two versions is lowered to a PIPE_S scalar
store loop (8 B per store, ``#pragma unroll``) that zeroes the *whole
allocation*, i.e. both versions and therefore every pad lane; without the clear
those lanes are never written at all.  The unrolled loop also dominates the
compile of a padded geometry (about 15x here).

Assertions (constant kind: structure of the generated source; the numbers are
printed for humans): two zero-fill loops of 4096 stores, one per operand, each
covering 2 versions x 16384 B; two PIPE_S notifies; nothing of the sort in the
non-clearing variant; clearing is far more expensive to compile.

First verified: 2026-11-23, tilelang 0.1.15 + CANN 9.2.1, Ascend950PR_9579,
compile only (never launched).  origin_task: mla_dense_bwd / _delta_kernel
Stage 3 (RETROSPECTIVE.md, Stage 3 section).  Re-verification: none.

    python CONST-950-clear-pad-all-versions.py      # needs the ascend env
"""
import re
import time

import tilelang
import tilelang.ascend.language as T

ROWS, DW, DIM, HEADS, SEQ, DT = 64, 128, 64, 4, 128, "bfloat16"
BUFFER_BYTES = ROWS * DW * 2  # one version of o_ub / do_ub (16384 B)


def build(clear):
    nfull = SEQ // ROWS

    @T.prim_func
    def main(
        o: T.Tensor((SEQ, HEADS, DIM), DT),
        do: T.Tensor((SEQ, HEADS, DIM), DT),
        delta: T.Tensor((1, HEADS, SEQ), "float32"),
    ):
        with T.Kernel(HEADS) as h:
            o_ub = T.alloc_shared((ROWS, DW), DT)
            do_ub = T.alloc_shared((ROWS, DW), DT)
            d_ub = T.alloc_shared((ROWS, 8), "float32")
            T.annotate_buffer_versions({o_ub: 2, do_ub: 2})
            if clear:  # build-time branch: pad columns must be zero on both operands
                T.clear(o_ub)
                T.clear(do_ub)
            for it in T.Pipelined(nfull, num_stages=2):
                s0 = it * ROWS
                # explicit destination region: the source has DIM < DW columns
                T.copy(o[s0 : s0 + ROWS, h, :], o_ub[0:ROWS, 0:DIM])
                T.copy(do[s0 : s0 + ROWS, h, :], do_ub[0:ROWS, 0:DIM])
                with T.SimdVF():
                    one = T.simd.pset(32, "PAT_VL1")
                    for i in T.serial(ROWS):
                        vo = T.simd.vld(o_ub[i, 0])
                        vd = T.simd.vld(do_ub[i, 0])
                        prod = T.simd.vadd(
                            T.simd.vmul(
                                T.simd.vcvt(vo, "float32", part=0), T.simd.vcvt(vd, "float32", part=0)
                            ),
                            T.simd.vmul(
                                T.simd.vcvt(vo, "float32", part=1), T.simd.vcvt(vd, "float32", part=1)
                            ),
                        )
                        T.simd.vsts(d_ub[i, 0], T.simd.vcadd(prod), one, extent=1)
                T.copy(d_ub[0:ROWS, 0:1], delta[0, h, s0 : s0 + ROWS])

    started = time.perf_counter()
    fn = tilelang.compile(main, target="ascend", out_idx=[2])
    return fn.get_kernel_source(), time.perf_counter() - started


tilelang.disable_cache()  # the generated source must come from a fresh compile
src_clear, t_clear = build(True)
src_plain, t_plain = build(False)

loop = r"for \(int32_t \w+ = 0; \w+ < (\d+); \+\+\w+\)"  # one counter per clear
fills = [int(n) for n in re.findall(loop, src_clear)]
zero_fill = [n for n in fills if n * 8 == 2 * BUFFER_BYTES]  # 4096 x 8 B = both versions
print(f"one version = {BUFFER_BYTES} B, two versions = {2 * BUFFER_BYTES} B")
print(f"clear: zero-fill loops = {zero_fill} (bytes {sum(zero_fill) * 8}), "
      f"PIPE_S notifies = {src_clear.count('asc_sync_notify(PIPE_S, PIPE_MTE2')}, "
      f"compile = {t_clear:.1f} s")
plain_fills = [int(n) for n in re.findall(loop, src_plain)]
print(f"plain: zero-fill loops = {[n for n in plain_fills if n == 4096]}, compile = {t_plain:.1f} s")

assert len(zero_fill) == 2, "expected one zero-fill loop per cleared operand"
assert sum(zero_fill) * 8 == 4 * BUFFER_BYTES, "the clear must zero both versions"
assert src_clear.count("asc_sync_notify(PIPE_S, PIPE_MTE2") == 2, "one PIPE_S notify per operand"
assert 4096 not in plain_fills, "the plain variant must not fill"
assert t_clear > 3 * t_plain, f"clear is not a compile-time outlier: {t_clear:.1f} vs {t_plain:.1f}"
print("PASS: T.clear zeroes the whole multi-versioned allocation (PIPE_S store loop); "
      "the un-cleared variant writes nothing")
