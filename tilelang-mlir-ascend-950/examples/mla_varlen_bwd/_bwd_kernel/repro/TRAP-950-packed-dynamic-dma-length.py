#!/usr/bin/env python3
"""TRAP-950: a ``T.dynamic`` extent turns every hot DMA length into a runtime value.

Why this repro exists
---------------------
The packed ``_bwd_kernel`` reuses the dense C2 body verbatim (same tile, same 5
explicit-L0 GEMMs, same epilogue), yet one block-iteration costs **3.62 us**
instead of the dense production **2.93 us** (both measured on
``ascend-950pr-9579``; see ``evidence/msprof_pipe_*_s3.csv`` and the dense
``evidence/msprof_pipe_capture1.csv``).  msprof attributes the whole difference
to the MTE2 pipes:

    AIC mte2   0.697 -> 1.107 us/iteration   (active BW 118.5 -> 80.5 GB/s)
    AIV mte2   0.112 -> 0.348 us/iteration   (lse/delta, 512 B per iteration)
    cube       0.756 -> 0.756 us/iteration   (unchanged)
    mte1       0.668 -> 0.692 us/iteration   (unchanged)
    fixpipe    0.562 -> 0.589 us/iteration   (unchanged)

This script isolates the mechanism at *compile* level: the per-iteration region
``q[qoff : qoff + 64, h, :]`` has a **runtime** start (``q_start`` comes from
``cu_seqlens_q``) inside a tensor whose row count is ``T.dynamic("total_q")``, so
the OOB pass cannot prove the region is in bounds and emits

    asc_copy_gm2l1_nd2nz(..., shape0 = min(64, max((total_q - q_start) - kq*64, 0)), 192, ...)

whereas the dense kernel -- same copy, static ``S_q`` -- emits a **constant**
``64``.  Every hot DMA of the iteration (``q``, ``do``, and their two transposed
copies) inherits the runtime length, which is what the MTE2 numbers above show.

What this asserts (deterministic, no device needed)
--------------------------------------------------
1. static extent  -> the emitted q-copy row count is the literal ``64``;
2. dynamic extent -> it is ``min(64, max(...))`` (a runtime value);
3. the dynamic form also widens the address arithmetic to ``int64_t``.

Run
---
    cd examples/mla_varlen_bwd/_bwd_kernel
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    TMPDIR=/workspace ASCEND_RT_VISIBLE_DEVICES="" \
      /workspace/venv-ascend950/bin/python repro/TRAP-950-packed-dynamic-dma-length.py

Exits 0 with ``TRAP_CONFIRMED`` when all three assertions hold.
"""

import re
import sys

import tilelang
import tilelang.ascend.language as T

ROWS, DIM, HEADS, TOTAL = 64, 192, 128, 32768


def build(dynamic: bool):
    """One kernel that loops the *same* 64x192 GM->L1 copy 8 times."""
    tq = T.dynamic("total_q") if dynamic else TOTAL

    @T.prim_func
    def main(src: T.Tensor((tq, HEADS, DIM), "bfloat16"),  # type: ignore
             off: T.Tensor((1,), "int32"),  # type: ignore
             out: T.Tensor((1,), "float32")) -> None:  # type: ignore
        with T.Kernel(1) as bid:
            l1 = T.alloc_l1((ROWS, DIM), "bfloat16")
            o = T.alloc_local([1], "int32")
            o[0] = off[0]
            for i in T.serial(8):
                start = o[0] + i * ROWS if dynamic else i * ROWS
                T.copy(src[start : start + ROWS, 5, :], l1)
            out[0] = T.cast(o[0], "float32")

    return tilelang.compile(main, target="ascend")


def copy_calls(src: str):
    """The generated GM->L1 lines of the benchmark loop."""
    return [line.strip() for line in src.splitlines()
            if "asc_copy_gm2l1" in line and "src" in line]


def main() -> int:
    static_src = build(False).get_kernel_source()
    dyn_src = build(True).get_kernel_source()
    static_calls, dyn_calls = copy_calls(static_src), copy_calls(dyn_src)
    print(f"[repro] static calls={len(static_calls)} dynamic calls={len(dyn_calls)}")
    for c in static_calls[:1]:
        print("[repro] static :", c[:200])
    for c in dyn_calls[:1]:
        print("[repro] dynamic:", c[:200])

    problems = []
    if not static_calls:
        problems.append("the static build emitted no GM->L1 copy")
    elif not re.search(r",\s*64,\s*192,\s*0,\s*0\)", static_calls[0]):
        problems.append("static build: row count is not the literal 64")
    if not dyn_calls:
        problems.append("the dynamic build emitted no GM->L1 copy")
    else:
        if "min(64" not in dyn_calls[0]:
            problems.append("dynamic build: row count is not a runtime min(64, ...)")
        if "int64_t" not in dyn_calls[0]:
            problems.append("dynamic build: address arithmetic is not widened to int64_t")

    if problems:
        print("[repro] TRAP_NOT_CONFIRMED:", "; ".join(problems))
        return 1
    print("[repro] TRAP_CONFIRMED: static extent -> constant DMA length; "
          "T.dynamic extent -> runtime min(64, ...) length + int64 addressing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
