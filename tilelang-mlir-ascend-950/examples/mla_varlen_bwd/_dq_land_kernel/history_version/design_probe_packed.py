# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Design probe (no device): the packed call path builds the same landed kernel.

Why this probe exists
---------------------
``MLAVarlenBwdKernel`` and ``MLADenseBwdKernel`` share one module-level
``_dq_land_kernel(heads, dim_qk, scale, dtype)``; the packed class only differs
in that it hands the already packed ``(T_q, H, D_qk)`` tensors straight to
``_run``.  This probe checks the two claims the design makes about that path,
without a device (the accelerator is busy with another operator's Stage 3):

1. the factory compiles the packed form for both dtypes with ``out_idx=[1]``
   (the JIT-allocated output, whose first extent is the *packed* total), and
2. the generated device code still contains the SIMD compute + store chain
   (``vlds_norm|vmuls|vcvt`` and ``vsts_pack_b32|vsts_norm``) -- the dense
   task's Stage 3 finding was that a lost body still compiles, so a bare
   ``COMPILE OK`` is not evidence.

It also prints a sha256 of each generated source.  ``T_q`` is a run-time symbol
(``T.dynamic("total_q")``), so the source is a function of
``(heads, dim_qk, scale, dtype)`` only: this hash is what any later geometry --
dense or packed, any request-length split -- compiles to.

Run (never launches):
    cd examples/mla_varlen_bwd/_dq_land_kernel
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        history_version/design_probe_packed.py
"""

import hashlib
import pathlib
import re

import tilelang
import tilelang.ascend.language as T

_SEG = 64
_R = 64
_G = 56  # Ascend950PR_9579 [SoCInfo] vector_core_cnt
_UB = 253952
_SCALE = 192 ** -0.5
_DIST = "PK_B32"
_COMPUTE = ("vlds_norm", "vmuls", "vcvt")
_STORE = ("vsts_pack_b32", "vsts_norm")
_OUT = pathlib.Path(__file__).resolve().parent


def build(heads, dim_qk, scale, dtype, versions=2):
    dw = ((dim_qk + _SEG - 1) // _SEG) * _SEG
    nseg = dw // _SEG
    assert versions * (dw * _R * 4 + dw * _R * 2) <= _UB, "UB budget"
    total = T.dynamic("total_q")

    @T.prim_func
    def _land(
        dq_acc: T.Tensor((total, heads, dim_qk), "float32"),  # type: ignore
        dq: T.Tensor((total, heads, dim_qk), dtype),  # type: ignore
    ) -> None:
        ntiles = T.ceildiv(total, _R) * heads
        with T.Kernel(_G) as bx:
            acc = T.alloc_shared((_R, dw), "float32")
            out = T.alloc_shared((_R, dw), dtype)
            T.annotate_buffer_versions({acc: versions, out: versions})
            cpt = T.ceildiv(ntiles, _G)
            it0 = bx * cpt
            it1 = T.min(it0 + cpt, ntiles)
            for it in T.Pipelined(it0, it1, num_stages=versions):
                h = it % heads
                t0 = (it // heads) * _R
                n = T.min(_R, total - t0)
                T.copy(dq_acc[t0 : t0 + n, h, :], acc[0:n, 0:dim_qk])
                with T.SimdVF():
                    m = T.simd.pset(32, "PAT_ALL")
                    for i in T.serial(_R):
                        for s in T.serial(nseg):
                            v = T.simd.vcvt(
                                T.simd.vmuls(T.simd.vld(acc[i, s * _SEG]), float(scale)),
                                dtype,
                                part=0,
                                sat=False,
                            )
                            T.simd.vsts(out[i, s * _SEG], v, m, _DIST, extent=_SEG)
                T.copy(out[0:n, 0:dim_qk], dq[t0 : t0 + n, h, :])

    return tilelang.compile(_land, target="ascend", out_idx=[1])


def main():
    ok = True
    for dtype in ("float16", "bfloat16"):
        kernel = build(8, 192, _SCALE, dtype)
        src = kernel.get_kernel_source()
        missing = [t for t in _COMPUTE if t not in src]
        store = [t for t in _STORE if t in src]
        digest = hashlib.sha256(src.encode()).hexdigest()[:16]
        lines = len(src.splitlines())
        path = _OUT / f"probe_packed_gen_{dtype}.txt"
        path.write_text(src, encoding="utf-8")
        # T_q must stay symbolic: no baked row count in the device code.
        symbolic = "total_q" in src
        print(
            f"[{dtype}] COMPILE OK  lines={lines}  sha256={digest}  "
            f"compute={[t for t in _COMPUTE if t in src]}  store={store}  "
            f"total_q_symbolic={symbolic}",
            flush=True,
        )
        cvt = re.findall(r"vcvt<[a-z0-9]+>\([^;]*", src)
        if cvt:
            print(f"    {cvt[0][:150]}", flush=True)
        if missing or not store or not symbolic:
            ok = False
            print(f"    MISSING compute={missing} store={store} symbolic={symbolic}", flush=True)
    print("[PASS] packed path keeps the SIMD body" if ok else "[FAIL] body lost")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
