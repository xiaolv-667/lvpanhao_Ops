"""Design probe (Stage 1, _dq_land_kernel) -- D-3 compile-only evidence.

The task's card is held by another operator's Stage 3, so this probe **never
launches**: it only compiles `target="ascend"` programs on the host and prints
the generated device code.  `ASCEND_RT_VISIBLE_DEVICES` is forced empty.

The variants answer the questions DESIGN.md depends on (see §3.5 and §9):

  k1  PRIMARY: static grid = vector cores (persistent), per-core contiguous
      tile range, `T.Pipelined(it0, it1, num_stages=2)` with **dynamic** bounds,
      dynamic row count `n = min(R, total - t0)` on both copies, dynamic
      `T.serial(n)` row loop, `vld -> vmuls -> vcvt(f32->f16) -> vsts`.
      -> does a dynamic trip count + 2 buffer versions compile?
      -> does the emitted copy carry a `min(...)` row clamp?
  k2  same body, `num_stages=1` (control: pipelining off)
  k3  flat dynamic grid `T.Kernel(ceildiv(total, R) * H)`, one tile per block
      (structural alternative of §1.6.3 candidate A1)
  k4  copy-cast: `T.copy(acc_f32_ub, dq_gm_f16)` without any vcvt  -> does the
      platform DMA cast fp32 UB -> fp16 GM?  (candidate rejected/accepted)
  k5  SIMT projection: `T.SimtVF(threads=128)` + `T.Parallel` + `T.cast`
      (the fallback path whose conversion semantics PL-950-conversion-layout
      already verified against torch)
  k6  bf16 variant of k1 (bfloat16 target dtype)

Run (no NPU, no launch):
    cd examples/mla_dense_bwd/_dq_land_kernel
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        history_version/design_probe_compile.py [k1|k2|k3|k4|k5|k6|all]
"""

import os
import sys
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

DQ = 192  # dim_qk of MLADenseBwdOp (manifest attention.yaml)
SEG = 64  # fp32 lanes of one 2048-bit vector register (2048 / 32)
NSEG = DQ // SEG  # 3 segments per row
R = 64  # rows per (tile) -- UB budget see DESIGN.md 5.2
G = 56  # vector cores of Ascend950PR_9579
SCALE = 0.08838834764831845  # D_qk ** -0.5, the op's default sm_scale

_KEY = (
    "asc_copy_gm2ub",
    "asc_copy_ub2gm",
    "vlds_norm",
    "vmuls",
    "vcvt",
    "vsts_norm",
    "for (",
    "min(",
)


def _dump(name, fn, want=26):
    print(f"\n===== {name} =====", flush=True)
    try:
        kernel = tilelang.compile(fn, target="ascend", out_idx=[])
        print("COMPILE: OK", flush=True)
    except Exception:  # noqa: BLE001 - a probe reports, it never raises
        print("COMPILE: FAIL", flush=True)
        traceback.print_exc(limit=3)
        return False
    src = kernel.get_kernel_source()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"probe_{name}_gen.txt")
    with open(path, "w") as fh:
        fh.write(src)
    print(f"    (full device code: {path})", flush=True)
    for ln in src.splitlines():
        s = ln.strip()
        if any(k in s for k in _KEY):
            print("   ", s[:260], flush=True)
    return True


def _row_body(acc, out, i, scale, dtype, m64):
    """One row: NSEG x (vld -> vmuls -> vcvt) + vsts, python-unrolled.

    `NSEG` is a python int, so this loop runs while tracing and emits `NSEG`
    independent statement groups; no vector value is carried across an IR
    `for` frame (the failure mode recorded in _delta_kernel DESIGN.md 3.3).
    """
    for s in range(NSEG):
        v = T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), scale)
        T.simd.vsts(
            out[i, s * SEG],
            T.simd.vcvt(v, dtype, part=0),
            m64,
            "NORM_B16",
            extent=SEG,
        )


@tilelang.jit
def _noop():  # placeholder to keep `tilelang.jit` imported-visible
    pass


def make_k1(dtype="float16", heads=8, num_stages=2):
    """Persistent static grid + pipelined dynamic tile range + SIMD epilogue."""

    @T.prim_func
    def main(
        dq_acc: T.Tensor((T.dynamic("total_q"), heads, DQ), "float32"),  # type: ignore
        dq: T.Tensor((T.dynamic("total_q"), heads, DQ), dtype),  # type: ignore
    ) -> None:
        total = dq_acc.shape[0]
        with T.Kernel(G) as bx:
            acc = T.alloc_shared((R, DQ), "float32")
            out = T.alloc_shared((R, DQ), dtype)
            T.annotate_buffer_versions({acc: 2, out: 2})
            ntiles = T.ceildiv(total, R) * heads
            cpt = T.ceildiv(ntiles, G)
            it0 = bx * cpt
            it1 = T.min(it0 + cpt, ntiles)
            for it in T.Pipelined(it0, it1, num_stages=num_stages):
                h = it % heads
                t0 = (it // heads) * R
                n = T.min(R, total - t0)
                T.copy(dq_acc[t0 : t0 + n, h, :], acc[0:n, 0:DQ])
                with T.SimdVF():
                    m64 = T.simd.pset(16, "PAT_VL64")
                    for i in T.serial(n):
                        _row_body(acc, out, i, SCALE, dtype, m64)
                T.copy(out[0:n, 0:DQ], dq[t0 : t0 + n, h, :])

    return main


def make_k2(dtype="float16", heads=8):
    return make_k1(dtype=dtype, heads=heads, num_stages=1)


def make_k3(dtype="float16", heads=8):
    """Flat dynamic grid: one (tile, head) per block, no in-core loop."""

    @T.prim_func
    def main(
        dq_acc: T.Tensor((T.dynamic("total_q"), heads, DQ), "float32"),  # type: ignore
        dq: T.Tensor((T.dynamic("total_q"), heads, DQ), dtype),  # type: ignore
    ) -> None:
        total = dq_acc.shape[0]
        with T.Kernel(T.ceildiv(total, R) * heads) as bx:
            h = bx % heads
            t0 = (bx // heads) * R
            n = T.min(R, total - t0)
            acc = T.alloc_shared((R, DQ), "float32")
            out = T.alloc_shared((R, DQ), dtype)
            T.copy(dq_acc[t0 : t0 + n, h, :], acc[0:n, 0:DQ])
            with T.SimdVF():
                m64 = T.simd.pset(16, "PAT_VL64")
                for i in T.serial(n):
                    _row_body(acc, out, i, SCALE, dtype, m64)
            T.copy(out[0:n, 0:DQ], dq[t0 : t0 + n, h, :])

    return main


def make_k4(dtype="float16", heads=8):
    """Candidate: let the UB->GM copy perform the fp32 -> fp16 cast."""

    @T.prim_func
    def main(
        dq_acc: T.Tensor((T.dynamic("total_q"), heads, DQ), "float32"),  # type: ignore
        dq: T.Tensor((T.dynamic("total_q"), heads, DQ), dtype),  # type: ignore
    ) -> None:
        total = dq_acc.shape[0]
        with T.Kernel(T.ceildiv(total, R) * heads) as bx:
            h = bx % heads
            t0 = (bx // heads) * R
            n = T.min(R, total - t0)
            acc = T.alloc_shared((R, DQ), "float32")
            T.copy(dq_acc[t0 : t0 + n, h, :], acc[0:n, 0:DQ])
            with T.SimdVF():
                for i in T.serial(n):
                    for s in range(NSEG):
                        T.simd.vsts(
                            acc[i, s * SEG],
                            T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), SCALE),
                            T.simd.pset(32, "PAT_VL64"),
                            "NORM_B32",
                            extent=SEG,
                        )
            T.copy(acc[0:n, 0:DQ], dq[t0 : t0 + n, h, :])

    return main


def make_k5(dtype="float16", heads=8):
    """Fallback projection: SIMT element loop with T.cast (verified semantics)."""

    @T.prim_func
    def main(
        dq_acc: T.Tensor((T.dynamic("total_q"), heads, DQ), "float32"),  # type: ignore
        dq: T.Tensor((T.dynamic("total_q"), heads, DQ), dtype),  # type: ignore
    ) -> None:
        total = dq_acc.shape[0]
        with T.Kernel(G) as bx:
            acc = T.alloc_shared((R, DQ), "float32")
            out = T.alloc_shared((R, DQ), dtype)
            T.annotate_buffer_versions({acc: 2, out: 2})
            ntiles = T.ceildiv(total, R) * heads
            cpt = T.ceildiv(ntiles, G)
            it0 = bx * cpt
            it1 = T.min(it0 + cpt, ntiles)
            for it in T.Pipelined(it0, it1, num_stages=2):
                h = it % heads
                t0 = (it // heads) * R
                n = T.min(R, total - t0)
                T.copy(dq_acc[t0 : t0 + n, h, :], acc[0:n, 0:DQ])
                with T.SimtVF(threads=128):
                    for i, d in T.Parallel(n, DQ):
                        out[i, d] = T.cast(acc[i, d] * SCALE, dtype)
                T.copy(out[0:n, 0:DQ], dq[t0 : t0 + n, h, :])

    return main


def make_k6(heads=8):
    return make_k1(dtype="bfloat16", heads=heads, num_stages=2)


def make_k7(dtype="float16", heads=8):
    """k1 + `T.clear(out)` before the loop (buffer-init / versioning fix test)."""

    @T.prim_func
    def main(
        dq_acc: T.Tensor((T.dynamic("total_q"), heads, DQ), "float32"),  # type: ignore
        dq: T.Tensor((T.dynamic("total_q"), heads, DQ), dtype),  # type: ignore
    ) -> None:
        total = dq_acc.shape[0]
        with T.Kernel(G) as bx:
            acc = T.alloc_shared((R, DQ), "float32")
            out = T.alloc_shared((R, DQ), dtype)
            T.annotate_buffer_versions({acc: 2, out: 2})
            T.clear(out)
            ntiles = T.ceildiv(total, R) * heads
            cpt = T.ceildiv(ntiles, G)
            it0 = bx * cpt
            it1 = T.min(it0 + cpt, ntiles)
            for it in T.Pipelined(it0, it1, num_stages=2):
                h = it % heads
                t0 = (it // heads) * R
                n = T.min(R, total - t0)
                T.copy(dq_acc[t0 : t0 + n, h, :], acc[0:n, 0:DQ])
                with T.SimdVF():
                    m64 = T.simd.pset(16, "PAT_VL64")
                    for i in T.serial(n):
                        _row_body(acc, out, i, SCALE, dtype, m64)
                T.copy(out[0:n, 0:DQ], dq[t0 : t0 + n, h, :])

    return main


def _make_static_rows(dtype="float16", heads=8, num_stages=2, persistent=True):
    """Corrected primary form: **static** row loop, clamped copies.

    `T.serial(n)` with a dynamic bound (k1/k3/k7) makes the toolchain drop the
    whole SIMD body (see the probe log), so the row loop stays static `R` and
    the tail is expressed only through the two DMA extents
    `n = min(R, total - t0)`.  Rows [n, R) therefore compute on uninitialised UB
    and are never copied out (the store copy carries the same clamp).
    """

    @T.prim_func
    def main(
        dq_acc: T.Tensor((T.dynamic("total_q"), heads, DQ), "float32"),  # type: ignore
        dq: T.Tensor((T.dynamic("total_q"), heads, DQ), dtype),  # type: ignore
    ) -> None:
        total = dq_acc.shape[0]
        ntiles = T.ceildiv(total, R) * heads
        with T.Kernel(G if persistent else ntiles) as bx:
            acc = T.alloc_shared((R, DQ), "float32")
            out = T.alloc_shared((R, DQ), dtype)
            T.annotate_buffer_versions({acc: 2, out: 2})
            if persistent:
                cpt = T.ceildiv(ntiles, G)
                it0 = bx * cpt
                it1 = T.min(it0 + cpt, ntiles)
                for it in T.Pipelined(it0, it1, num_stages=num_stages):
                    h = it % heads
                    t0 = (it // heads) * R
                    n = T.min(R, total - t0)
                    T.copy(dq_acc[t0 : t0 + n, h, :], acc[0:n, 0:DQ])
                    with T.SimdVF():
                        m64 = T.simd.pset(16, "PAT_VL64")
                        for i in T.serial(R):
                            _row_body(acc, out, i, SCALE, dtype, m64)
                    T.copy(out[0:n, 0:DQ], dq[t0 : t0 + n, h, :])
            else:
                h = bx % heads
                t0 = (bx // heads) * R
                n = T.min(R, total - t0)
                T.copy(dq_acc[t0 : t0 + n, h, :], acc[0:n, 0:DQ])
                with T.SimdVF():
                    m64 = T.simd.pset(16, "PAT_VL64")
                    for i in T.serial(R):
                        _row_body(acc, out, i, SCALE, dtype, m64)
                T.copy(out[0:n, 0:DQ], dq[t0 : t0 + n, h, :])

    return main


def make_n1(dtype="float16"):
    return _make_static_rows(dtype=dtype, num_stages=2, persistent=True)


def make_n2(dtype="float16"):
    return _make_static_rows(dtype=dtype, num_stages=1, persistent=True)


def make_n4(dtype="float16"):
    return _make_static_rows(dtype=dtype, persistent=False)


def make_n5(dtype="float16", heads=8):
    """SIMT fallback, static rows: `T.Parallel(R, DQ)` + `T.cast`."""

    @T.prim_func
    def main(
        dq_acc: T.Tensor((T.dynamic("total_q"), heads, DQ), "float32"),  # type: ignore
        dq: T.Tensor((T.dynamic("total_q"), heads, DQ), dtype),  # type: ignore
    ) -> None:
        total = dq_acc.shape[0]
        ntiles = T.ceildiv(total, R) * heads
        with T.Kernel(ntiles) as bx:
            h = bx % heads
            t0 = (bx // heads) * R
            n = T.min(R, total - t0)
            acc = T.alloc_shared((R, DQ), "float32")
            out = T.alloc_shared((R, DQ), dtype)
            T.copy(dq_acc[t0 : t0 + n, h, :], acc[0:n, 0:DQ])
            with T.SimtVF(threads=128):
                for i, d in T.Parallel(R, DQ):
                    out[i, d] = T.cast(acc[i, d] * SCALE, dtype)
            T.copy(out[0:n, 0:DQ], dq[t0 : t0 + n, h, :])

    return main


CASES = {
    "k1": lambda: make_k1(),
    "k2": lambda: make_k2(),
    "k3": lambda: make_k3(),
    "k4": lambda: make_k4(),
    "k5": lambda: make_k5(),
    "k6": lambda: make_k6(),
    "k7": lambda: make_k7(),
    "n1": lambda: make_n1(),
    "n2": lambda: make_n2(),
    "n4": lambda: make_n4(),
    "n5": lambda: make_n5(),
}


def main(argv):
    todo = list(CASES) if (not argv or argv[0] == "all") else argv
    ok = {}
    for name in todo:
        if name not in CASES:
            print(f"unknown case {name!r}; choose from {list(CASES)}")
            return 2
        ok[name] = _dump(name, CASES[name]())
    print("\n===== summary =====", flush=True)
    for name, good in ok.items():
        print(f"  {name}: {'COMPILE OK' if good else 'COMPILE FAIL'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
