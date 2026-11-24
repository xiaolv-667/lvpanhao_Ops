"""CONST-950-tail-no-multibuffer.py -- a tail segment disables UB multi-buffering.

Entry: pattern-library constants -> tail-segment geometry / UB versions.
Phenomenon: when a UB buffer is also written *outside* ``T.Pipelined`` (the
static-``nlast`` tail), AutoSchedule refuses the requested 2 versions, so the
pipelined loads keep one UB base instead of alternating -- performance only.
Assertions: no-tail (S = 128, 2 pipelined iterations) has both GM->UB
destinations varying with the tile index and no warning; tail (S = 131 =
2 x 64 + 3) has no varying destination and reports the refusal.
First verified: 2026-11-23, tilelang 0.1.15 + CANN 9.2.1, Ascend950PR_9579,
compile only (never launched); the same geometries matched the golden on the
device (max_diff 3.8e-06) in the parent task.  origin_task: mla_dense_bwd /
_delta_kernel Stage 3 (RETROSPECTIVE.md, Stage 3 section).  Re-verification: none.

    python CONST-950-tail-no-multibuffer.py      # needs the ascend env
"""
import os
import tempfile

import tilelang
import tilelang.ascend.language as T

ROWS, DW, DIM, HEADS, DT = 64, 128, 128, 2, "bfloat16"

def _capture_stderr():
    """Capture fd 2: the AutoSchedule warnings are written below ``sys.stderr``."""
    saved, tmp = os.dup(2), tempfile.TemporaryFile()
    os.dup2(tmp.fileno(), 2)

    def restore():
        os.dup2(saved, 2)
        os.close(saved)
        tmp.seek(0)
        return tmp.read().decode("utf-8", "ignore")

    return restore

@T.macro
def _rows(o_ub, do_ub, d_ub, rows):
    """One SIMD row loop.  Must be a ``T.macro``: a plain python helper called
    from the traced body is not source-transformed, so its ``T.serial`` fails."""
    with T.SimdVF():
        one = T.simd.pset(32, "PAT_VL1")
        for i in T.serial(rows):
            vo, vd = T.simd.vld(o_ub[i, 0]), T.simd.vld(do_ub[i, 0])
            prod = T.simd.vadd(
                T.simd.vmul(T.simd.vcvt(vo, "float32", part=0), T.simd.vcvt(vd, "float32", part=0)),
                T.simd.vmul(T.simd.vcvt(vo, "float32", part=1), T.simd.vcvt(vd, "float32", part=1)),
            )
            T.simd.vsts(d_ub[i, 0], T.simd.vcadd(prod), one, extent=1)

def build(seqlen):
    nfull, nlast = seqlen // ROWS, seqlen - (seqlen // ROWS) * ROWS  # build-time

    @T.prim_func
    def main(
        o: T.Tensor((seqlen, HEADS, DIM), DT),
        do: T.Tensor((seqlen, HEADS, DIM), DT),
        delta: T.Tensor((1, HEADS, seqlen), "float32"),
    ):
        with T.Kernel(HEADS) as h:  # one core per head, no intra-core split
            o_ub = T.alloc_shared((ROWS, DW), DT)
            do_ub = T.alloc_shared((ROWS, DW), DT)
            d_ub = T.alloc_shared((ROWS, 8), "float32")
            T.annotate_buffer_versions({o_ub: 2, do_ub: 2})
            if nfull > 0:
                for it in T.Pipelined(nfull, num_stages=2):
                    s0 = it * ROWS
                    T.copy(o[s0 : s0 + ROWS, h, :], o_ub[0:ROWS, 0:DIM])
                    T.copy(do[s0 : s0 + ROWS, h, :], do_ub[0:ROWS, 0:DIM])
                    _rows(o_ub, do_ub, d_ub, ROWS)
                    T.copy(d_ub[0:ROWS, 0:1], delta[0, h, s0 : s0 + ROWS])
            if nlast:  # static tail: the three extents use nlast, no run-time predicate
                s0 = nfull * ROWS
                T.copy(o[s0 : s0 + nlast, h, :], o_ub[0:nlast, 0:DIM])
                T.copy(do[s0 : s0 + nlast, h, :], do_ub[0:nlast, 0:DIM])
                _rows(o_ub, do_ub, d_ub, nlast)
                T.copy(d_ub[0:nlast, 0:1], delta[0, h, s0 : s0 + nlast])

    log = _capture_stderr()
    fn = tilelang.compile(main, target="ascend", out_idx=[2])
    return fn.get_kernel_source(), log()

tilelang.disable_cache()  # the pass must run: a cached kernel prints no warning
plain_src, plain_log = build(128)  # no tail: 2 pipelined iterations
tail_src, tail_log = build(131)  # nfull = 2, nlast = 3

# a versioned load addresses its UB destination with the tile index
versioned = lambda src: [  # noqa: E731
    ln for ln in src.splitlines() if "asc_copy_gm2ub" in ln and "it *" in ln.split("), (__gm__")[0]
]
plain_v, tail_v = len(versioned(plain_src)), len(versioned(tail_src))
warn = "did not enable multi-buffering"
print(f"no-tail (S=128): versioned loads = {plain_v}, warning = {warn in plain_log}")
print(f"tail    (S=131): versioned loads = {tail_v}, warning = {warn in tail_log}")
print("   ", next((ln.strip()[:112] for ln in tail_log.splitlines() if warn in ln), "no warning"))
assert plain_v == 2, "the tail-free geometry must load through both UB versions"
assert tail_v == 0, "the tail geometry must fall back to a single UB version"
assert warn in tail_log and warn not in plain_log, "warning/version mismatch"
print("PASS: a segment written outside T.Pipelined drops the requested UB versions")
