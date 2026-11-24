"""Design probe #3 (Stage 1, _dq_land_kernel) -- trigger of the SIMD body drop.

Probe #2 established: `T.SimdVF` + `T.simd.*` survives with a **segment loop
written inside the prim_func body** (m1-m4) and the DMA cannot cast (m5/m6
semantic check).  Every form that used a module-level python helper
(`_row_body`, which unrolls the 3 segments at trace time) lost the body.
This probe isolates that difference:

  e8   static tensor, R=4, segment loop **inside** the body  (control, == m2)
  e9   static tensor, R=64, block-index addressing, same form (== real geometry)
  e10  dynamic tensor, R=64, block-index addressing, clamped copies, same form
  e11  dynamic tensor, R=64, block-index addressing, **python-unrolled** segments
       (the form that failed) -- negative control

Run (no NPU, no launch):
    cd examples/mla_dense_bwd/_dq_land_kernel
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        history_version/design_probe_dynamic.py [e8|e9|e10|e11|all]
"""

import os
import sys
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

DQ = 192
SEG = 64
R = 64
H = 2
STATIC_ROWS = 256
SCALE = 0.08838834764831845

_OPS = ("vlds", "vmuls", "vcvt", "vsts", "asc_copy")


def _row_body(acc, out, i, scale, dtype, m64):
    """Python-unrolled segments (negative-control form)."""
    for s in range(3):
        T.simd.vsts(
            out[i, s * SEG],
            T.simd.vcvt(T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), scale), dtype, part=0),
            m64,
            "NORM_B16",
            extent=SEG,
        )


def _report(name, fn, out_idx=None):
    print(f"\n===== {name} =====", flush=True)
    try:
        kernel = tilelang.compile(fn, target="ascend", out_idx=[] if out_idx is None else out_idx)
        print("COMPILE: OK", flush=True)
    except Exception:  # noqa: BLE001
        print("COMPILE: FAIL", flush=True)
        traceback.print_exc(limit=3)
        return None
    src = kernel.get_kernel_source()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"probe_{name}_gen.txt")
    with open(path, "w") as fh:
        fh.write(src)
    hits = [ln.strip() for ln in src.splitlines() if any(o in ln for o in _OPS)]
    print(f"    SIMD body present: {any('vlds' in h for h in hits)}", flush=True)
    for ln in hits:
        print("   ", ln[:180], flush=True)
    return src


def make_e8():
    """Static tensor, R=4, segment loop inside the prim_func body."""

    @T.prim_func
    def main(dq_acc: T.Tensor((8, H, DQ), "float32"),  # type: ignore
             dq: T.Tensor((8, H, DQ), "float16")):  # type: ignore
        with T.Kernel(1) as bx:
            acc = T.alloc_shared((4, DQ), "float32")
            out = T.alloc_shared((4, DQ), "float16")
            T.copy(dq_acc[0:4, 0, :], acc)
            with T.SimdVF():
                m = T.simd.pset(16, "PAT_VL64")
                for i in T.serial(4):
                    for s in T.serial(3):
                        v = T.simd.vcvt(
                            T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), SCALE), "float16", part=0
                        )
                        T.simd.vsts(out[i, s * SEG], v, m, "NORM_B16", extent=SEG)
            T.copy(out, dq[0:4, 0, :])

    return main


def _make_block_static():
    @T.prim_func
    def main(dq_acc: T.Tensor((STATIC_ROWS, H, DQ), "float32"),  # type: ignore
             dq: T.Tensor((STATIC_ROWS, H, DQ), "float16")):  # type: ignore
        with T.Kernel((STATIC_ROWS // R) * H) as bx:
            h = bx % H
            t0 = (bx // H) * R
            acc = T.alloc_shared((R, DQ), "float32")
            out = T.alloc_shared((R, DQ), "float16")
            T.copy(dq_acc[t0 : t0 + R, h, :], acc)
            with T.SimdVF():
                m = T.simd.pset(16, "PAT_VL64")
                for i in T.serial(R):
                    for s in T.serial(3):
                        v = T.simd.vcvt(
                            T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), SCALE), "float16", part=0
                        )
                        T.simd.vsts(out[i, s * SEG], v, m, "NORM_B16", extent=SEG)
            T.copy(out, dq[t0 : t0 + R, h, :])

    return main


def _make_block_dynamic(unrolled=False):
    if unrolled:

        @T.prim_func
        def main(dq_acc: T.Tensor((T.dynamic("total_q"), H, DQ), "float32"),  # type: ignore
                 dq: T.Tensor((T.dynamic("total_q"), H, DQ), "float16")):  # type: ignore
            total = dq_acc.shape[0]
            with T.Kernel(T.ceildiv(total, R) * H) as bx:
                h = bx % H
                t0 = (bx // H) * R
                n = T.min(R, total - t0)
                acc = T.alloc_shared((R, DQ), "float32")
                out = T.alloc_shared((R, DQ), "float16")
                T.copy(dq_acc[t0 : t0 + n, h, :], acc[0:n, 0:DQ])
                with T.SimdVF():
                    m = T.simd.pset(16, "PAT_VL64")
                    for i in T.serial(R):
                        _row_body(acc, out, i, SCALE, "float16", m)
                T.copy(out[0:n, 0:DQ], dq[t0 : t0 + n, h, :])

        return main

    @T.prim_func
    def main(dq_acc: T.Tensor((T.dynamic("total_q"), H, DQ), "float32"),  # type: ignore
             dq: T.Tensor((T.dynamic("total_q"), H, DQ), "float16")):  # type: ignore
        total = dq_acc.shape[0]
        with T.Kernel(T.ceildiv(total, R) * H) as bx:
            h = bx % H
            t0 = (bx // H) * R
            n = T.min(R, total - t0)
            acc = T.alloc_shared((R, DQ), "float32")
            out = T.alloc_shared((R, DQ), "float16")
            T.copy(dq_acc[t0 : t0 + n, h, :], acc[0:n, 0:DQ])
            with T.SimdVF():
                m = T.simd.pset(16, "PAT_VL64")
                for i in T.serial(R):
                    for s in T.serial(3):
                        v = T.simd.vcvt(
                            T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), SCALE), "float16", part=0
                        )
                        T.simd.vsts(out[i, s * SEG], v, m, "NORM_B16", extent=SEG)
            T.copy(out[0:n, 0:DQ], dq[t0 : t0 + n, h, :])

    return main


CASES = {
    "e8": make_e8,
    "e9": _make_block_static,
    "e10": lambda: _make_block_dynamic(False),
    "e11": lambda: _make_block_dynamic(True),
}




def _make_persistent(num_stages):
    """Static grid (= vector cores) + per-core contiguous dynamic tile range."""

    @T.prim_func
    def main(dq_acc: T.Tensor((T.dynamic("total_q"), H, DQ), "float32"),  # type: ignore
             dq: T.Tensor((T.dynamic("total_q"), H, DQ), "float16")):  # type: ignore
        total = dq_acc.shape[0]
        ntiles = T.ceildiv(total, R) * H
        with T.Kernel(56) as bx:
            acc = T.alloc_shared((R, DQ), "float32")
            out = T.alloc_shared((R, DQ), "float16")
            T.annotate_buffer_versions({acc: 2, out: 2})
            cpt = T.ceildiv(ntiles, 56)
            it0 = bx * cpt
            it1 = T.min(it0 + cpt, ntiles)
            for it in T.Pipelined(it0, it1, num_stages=num_stages):
                h = it % H
                t0 = (it // H) * R
                n = T.min(R, total - t0)
                T.copy(dq_acc[t0 : t0 + n, h, :], acc[0:n, 0:DQ])
                with T.SimdVF():
                    m = T.simd.pset(16, "PAT_VL64")
                    for i in T.serial(R):
                        for s in T.serial(3):
                            v = T.simd.vcvt(
                                T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), SCALE),
                                "float16",
                                part=0,
                                sat=False,
                            )
                            T.simd.vsts(out[i, s * SEG], v, m, "NORM_B16", extent=SEG)
                T.copy(out[0:n, 0:DQ], dq[t0 : t0 + n, h, :])

    return main


CASES["e12"] = lambda: _make_persistent(2)
CASES["e13"] = lambda: _make_persistent(1)


def make_e14():
    """PRIMARY form compiled with the real `out_idx=[1]` (output allocation)."""
    return _make_persistent(2)


CASES["e14"] = make_e14


def make_e15():
    """PRIMARY form, `out_idx=[1]`, symbol bound from an outer python variable
    (the source operator's `total = T.dynamic("total_q")` pattern)."""
    total = T.dynamic("total_q")

    @T.prim_func
    def main(dq_acc: T.Tensor((total, H, DQ), "float32"),  # type: ignore
             dq: T.Tensor((total, H, DQ), "float16")):  # type: ignore
        ntiles = T.ceildiv(total, R) * H
        with T.Kernel(56) as bx:
            acc = T.alloc_shared((R, DQ), "float32")
            out = T.alloc_shared((R, DQ), "float16")
            T.annotate_buffer_versions({acc: 2, out: 2})
            cpt = T.ceildiv(ntiles, 56)
            it0 = bx * cpt
            it1 = T.min(it0 + cpt, ntiles)
            for it in T.Pipelined(it0, it1, num_stages=2):
                h = it % H
                t0 = (it // H) * R
                n = T.min(R, total - t0)
                T.copy(dq_acc[t0 : t0 + n, h, :], acc[0:n, 0:DQ])
                with T.SimdVF():
                    m = T.simd.pset(16, "PAT_VL64")
                    for i in T.serial(R):
                        for s in T.serial(3):
                            v = T.simd.vcvt(
                                T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), SCALE),
                                "float16",
                                part=0,
                                sat=False,
                            )
                            T.simd.vsts(out[i, s * SEG], v, m, "NORM_B16", extent=SEG)
                T.copy(out[0:n, 0:DQ], dq[t0 : t0 + n, h, :])

    return main


CASES["e15"] = make_e15


def main(argv):
    todo = list(CASES) if (not argv or argv[0] == "all") else argv
    for name in todo:
        if name not in CASES:
            print(f"unknown case {name!r}; choose from {list(CASES)}")
            return 2
        _report(name, CASES[name](), out_idx=[1] if name in ("e14", "e15") else None)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
