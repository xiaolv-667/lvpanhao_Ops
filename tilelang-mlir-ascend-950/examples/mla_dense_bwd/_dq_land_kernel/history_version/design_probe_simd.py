"""Design probe #2 (Stage 1, _dq_land_kernel) -- which SIMD store form survives.

Probe #1 (`design_probe_compile.py`, k1/k7) showed that the intended chain
`vld -> vmuls -> vcvt(f32->f16) -> vsts(NORM_B16, extent=64)` compiles but the
**whole compute body is dropped** from the device code (only the two default
masks survive).  This probe isolates the store form on a 4-row static geometry
and reports, per variant, which SIMD instructions reach the generated code.

Variants:
  m1  vld -> vmuls -> vsts back into the SAME fp32 buffer (dist default
      NORM_B32, all-lane fp32 mask) -- is a plain register store lowered?
  m2  vld -> vmuls -> vcvt(f16, part=0) -> vsts fp16, mask PAT_VL64,
      dist="NORM_B16", extent=64  (probe #1 form)
  m3  same as m2 without the explicit `extent` argument
  m4  same as m2 but mask = pset(16, "PAT_ALL")
  m5  pure `T.copy(acc_row_f32, out_row_f16)` per row -- does the platform DMA
      cast on a UB->UB copy?
  m6  vld -> vmuls -> vsts back into fp32, then `T.copy(acc, dq)` to GM fp16
      (cast delegated to the UB->GM copy)

Run (no NPU, no launch):
    cd examples/mla_dense_bwd/_dq_land_kernel
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        history_version/design_probe_simd.py [m1|m2|m3|m4|m5|m6|all]
"""

import os
import sys
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

DQ = 192
SEG = 64
R = 4
TOTAL = 8  # static rows for the micro cases (keeps the probe minimal)
H = 2
SCALE = 0.08838834764831845

_OPS = ("vlds", "vmuls", "vcvt", "vsts", "asc_copy", "asc_storealign")


def _row_body(acc, out, i, scale, dtype, m64):
    """One row of the 192-wide epilogue, python-unrolled over the 3 segments."""
    for s in range(3):
        T.simd.vsts(
            out[i, s * SEG],
            T.simd.vcvt(T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), scale), dtype, part=0),
            m64,
            "NORM_B16",
            extent=SEG,
        )


def _report(name, fn):
    print(f"\n===== {name} =====", flush=True)
    try:
        kernel = tilelang.compile(fn, target="ascend", out_idx=[])
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
    if not hits:
        print("    (no SIMD/copy instruction in the device code -> body dropped)", flush=True)
    for ln in hits:
        print("   ", ln[:200], flush=True)
    return src


def make_m1():
    @T.prim_func
    def main(dq_acc: T.Tensor((TOTAL, H, DQ), "float32"), dq: T.Tensor((TOTAL, H, DQ), "float32")):
        with T.Kernel(1) as bx:
            acc = T.alloc_shared((R, DQ), "float32")
            T.copy(dq_acc[0:R, 0, :], acc)
            with T.SimdVF():
                ma = T.simd.pset(32, "PAT_ALL")
                for i in T.serial(R):
                    for s in range(3):
                        T.simd.vsts(
                            acc[i, s * SEG],
                            T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), SCALE),
                            ma,
                        )
            T.copy(acc, dq[0:R, 0, :])

    return main


def _make_cast(kind):
    @T.prim_func
    def main(dq_acc: T.Tensor((TOTAL, H, DQ), "float32"), dq: T.Tensor((TOTAL, H, DQ), "float16")):
        with T.Kernel(1) as bx:
            acc = T.alloc_shared((R, DQ), "float32")
            out = T.alloc_shared((R, DQ), "float16")
            T.copy(dq_acc[0:R, 0, :], acc)
            with T.SimdVF():
                if kind == "all":
                    m = T.simd.pset(16, "PAT_ALL")
                else:
                    m = T.simd.pset(16, "PAT_VL64")
                for i in T.serial(R):
                    for s in range(3):
                        v = T.simd.vcvt(
                            T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), SCALE), "float16", part=0
                        )
                        if kind == "noext":
                            T.simd.vsts(out[i, s * SEG], v, m, "NORM_B16")
                        else:
                            T.simd.vsts(out[i, s * SEG], v, m, "NORM_B16", extent=SEG)
            T.copy(out, dq[0:R, 0, :])

    return main


def make_m5():
    """Per-row UB->UB copy with dtype change."""

    @T.prim_func
    def main(dq_acc: T.Tensor((TOTAL, H, DQ), "float32"), dq: T.Tensor((TOTAL, H, DQ), "float16")):
        with T.Kernel(1) as bx:
            acc = T.alloc_shared((R, DQ), "float32")
            out = T.alloc_shared((R, DQ), "float16")
            T.copy(dq_acc[0:R, 0, :], acc)
            for i in T.serial(R):
                T.copy(acc[i : i + 1, 0:DQ], out[i : i + 1, 0:DQ])
            T.copy(out, dq[0:R, 0, :])

    return main


def make_m6():
    """Scale in fp32 in place, then let the UB->GM copy cast to fp16."""

    @T.prim_func
    def main(dq_acc: T.Tensor((TOTAL, H, DQ), "float32"), dq: T.Tensor((TOTAL, H, DQ), "float16")):
        with T.Kernel(1) as bx:
            acc = T.alloc_shared((R, DQ), "float32")
            T.copy(dq_acc[0:R, 0, :], acc)
            with T.SimdVF():
                ma = T.simd.pset(32, "PAT_ALL")
                for i in T.serial(R):
                    for s in range(3):
                        T.simd.vsts(
                            acc[i, s * SEG],
                            T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), SCALE),
                            ma,
                        )
            T.copy(acc, dq[0:R, 0, :])

    return main


CASES = {
    "m1": make_m1,
    "m2": lambda: _make_cast("vl64"),
    "m3": lambda: _make_cast("noext"),
    "m4": lambda: _make_cast("all"),
    "m5": make_m5,
    "m6": make_m6,
}


# ---------------------------------------------------------------------------
# Bisect: which surrounding construct makes the SIMD VF body disappear?
# b1 = m2 (known good) ... b4 adds one construct at a time.
# ---------------------------------------------------------------------------
def _bisect(dynamic_copy=False, versions=False, pipelined=False, rows=R):
    @T.prim_func
    def main(dq_acc: T.Tensor((T.dynamic("total_q"), H, DQ), "float32"),  # type: ignore
             dq: T.Tensor((T.dynamic("total_q"), H, DQ), "float16")):  # type: ignore
        total = dq_acc.shape[0]
        with T.Kernel(1) as bx:
            acc = T.alloc_shared((rows, DQ), "float32")
            out = T.alloc_shared((rows, DQ), "float16")
            if versions:
                T.annotate_buffer_versions({acc: 2, out: 2})
            n = T.min(rows, total)
            if pipelined:
                for _it in T.Pipelined(0, 4, num_stages=2):
                    if dynamic_copy:
                        T.copy(dq_acc[0:n, 0, :], acc[0:n, 0:DQ])
                    else:
                        T.copy(dq_acc[0:rows, 0, :], acc)
                    with T.SimdVF():
                        m = T.simd.pset(16, "PAT_VL64")
                        for i in T.serial(rows):
                            _row_body(acc, out, i, SCALE, "float16", m)
                    if dynamic_copy:
                        T.copy(out[0:n, 0:DQ], dq[0:n, 0, :])
                    else:
                        T.copy(out, dq[0:rows, 0, :])
            else:
                if dynamic_copy:
                    T.copy(dq_acc[0:n, 0, :], acc[0:n, 0:DQ])
                else:
                    T.copy(dq_acc[0:rows, 0, :], acc)
                with T.SimdVF():
                    m = T.simd.pset(16, "PAT_VL64")
                    for i in T.serial(rows):
                        _row_body(acc, out, i, SCALE, "float16", m)
                if dynamic_copy:
                    T.copy(out[0:n, 0:DQ], dq[0:n, 0, :])
                else:
                    T.copy(out, dq[0:rows, 0, :])

    return main


CASES.update(
    {
        "b2": lambda: _bisect(dynamic_copy=True),
        "b3": lambda: _bisect(versions=True),
        "b4": lambda: _bisect(pipelined=True),
        "b5": lambda: _bisect(rows=64),
        "b6": lambda: _bisect(dynamic_copy=True, rows=64),
        "r4": lambda: _bisect(rows=4),
        "r8": lambda: _bisect(rows=8),
        "r16": lambda: _bisect(rows=16),
        "r32": lambda: _bisect(rows=32),
        "r24": lambda: _bisect(rows=24),
        "r6": lambda: _bisect(rows=6),
        "r5": lambda: _bisect(rows=5),
    }
)


def main(argv):
    todo = list(CASES) if (not argv or argv[0] == "all") else argv
    for name in todo:
        if name not in CASES:
            print(f"unknown case {name!r}; choose from {list(CASES)}")
            return 2
        _report(name, CASES[name]())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
