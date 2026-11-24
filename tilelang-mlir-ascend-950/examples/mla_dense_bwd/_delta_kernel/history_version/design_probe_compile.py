"""Design probe 1 (Stage 1, _delta_kernel) -- D-3 compile-only API evidence.

Nothing is launched (``ASCEND_RT_VISIBLE_DEVICES=""``): every kernel below is only
built, and the generated source is printed so the design can cite what the
compiler actually emits for

  P1  the strided GM->UB region copy of one head's rows  ``o[r0:r0+R, h, :]``
      (does it become one descriptor / one instruction, or R 1-D copies?)
  P2  the contiguous UB->GM store of the R fp32 row sums
      (``d_ub[:, 0:1]`` from a row-padded (R, 8) staging buffer)
  P3  the explicit SIMD row reduction inside ``T.SimdVF``
      (vld fp16x128 -> 2x vcvt part 0/1 -> 2x vmul -> vadd -> vcadd -> vsts)
      for float16 and bfloat16 inputs
  P4  the persistent form: ``T.Pipelined`` tile loop with a nested SimdVF body
      plus ``T.annotate_buffer_versions`` double buffering

Run (no NPU, no kernel launch):
    cd examples/mla_dense_bwd/_delta_kernel
    source .agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        history_version/design_probe_compile.py [p1|p2|p3|p4|all]
"""

from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

B, S, H, D = 1, 512, 8, 128  # manifest smoke geometry (heads/D_v), reduced S
R = 64  # rows per tile
SR = 8  # splits of the row axis per (b, h) == S // R here


def _dump(name: str, fn, want=90):
    print(f"\n===== {name} =====", flush=True)
    try:
        kernel = tilelang.compile(fn, target="ascend", out_idx=[])
        print("COMPILE: OK", flush=True)
    except Exception:  # noqa: BLE001 - a probe reports, it never raises
        print("COMPILE: FAIL", flush=True)
        traceback.print_exc(limit=4)
        return False
    src = ""
    for attr in ("get_kernel_source", "get_source"):
        if hasattr(kernel, attr):
            try:
                src = str(getattr(kernel, attr)())
                break
            except Exception:  # noqa: BLE001 - diagnostic only
                pass
    if not src:
        print("(no source accessor)", flush=True)
        return True
    lines = [ln for ln in src.splitlines() if ln.strip()]
    print(f"-- generated source: {len(lines)} non-empty lines", flush=True)
    interesting = [
        ln
        for ln in lines
        if any(
            k in ln
            for k in (
                "copy",
                "Copy",
                "DataCopy",
                "Nd2Nd",
                "nd2nd",
                "vld",
                "vsts",
                "vcvt",
                "vmul",
                "vcadd",
                "vadd",
                "set_flag",
                "wait_flag",
                "for (",
            )
        )
    ]
    for ln in interesting[:want]:
        print("   ", ln.strip(), flush=True)
    if len(interesting) > want:
        print(f"    ... ({len(interesting) - want} more matching lines)", flush=True)
    return True


def p1():
    """Strided head-slice load + contiguous row-sum store (copy paths only)."""

    @T.prim_func
    def main(
        o: T.Tensor((B * S, H, D), "float16"),
        do: T.Tensor((B * S, H, D), "float16"),
        delta: T.Tensor((B, H, S), "float32"),
    ) -> None:
        with T.Kernel(B * H * SR) as bx:
            h = (bx // SR) % H
            sp = bx % SR
            o_ub = T.alloc_shared((R, D), "float16")
            do_ub = T.alloc_shared((R, D), "float16")
            d_ub = T.alloc_shared((R, 8), "float32")
            r0 = sp * R
            T.copy(o[r0 : r0 + R, h, :], o_ub)
            T.copy(do[r0 : r0 + R, h, :], do_ub)
            for i in T.serial(R):
                d_ub[i, 0] = 1.0
            T.copy(d_ub[:, 0:1], delta[bx // (SR * H), h, r0 : r0 + R])

    return main


def p2(dtype: str):
    """Full primary body: SIMD row dot product, R rows per tile, one program."""

    @T.prim_func
    def main(
        o: T.Tensor((B * S, H, D), dtype),
        do: T.Tensor((B * S, H, D), dtype),
        delta: T.Tensor((B, H, S), "float32"),
    ) -> None:
        with T.Kernel(B * H * SR) as bx:
            h = (bx // SR) % H
            sp = bx % SR
            o_ub = T.alloc_shared((R, D), dtype)
            do_ub = T.alloc_shared((R, D), dtype)
            d_ub = T.alloc_shared((R, 8), "float32")
            r0 = sp * R
            T.copy(o[r0 : r0 + R, h, :], o_ub)
            T.copy(do[r0 : r0 + R, h, :], do_ub)
            with T.SimdVF():
                one = T.simd.pset(32, "PAT_VL1")
                for i in T.serial(R):
                    vo = T.simd.vld(o_ub[i, 0])
                    vd = T.simd.vld(do_ub[i, 0])
                    o_lo = T.simd.vcvt(vo, "float32", part=0)
                    o_hi = T.simd.vcvt(vo, "float32", part=1)
                    d_lo = T.simd.vcvt(vd, "float32", part=0)
                    d_hi = T.simd.vcvt(vd, "float32", part=1)
                    prod = T.simd.vadd(
                        T.simd.vmul(o_lo, d_lo), T.simd.vmul(o_hi, d_hi)
                    )
                    total = T.simd.vcadd(prod)
                    T.simd.vsts(d_ub[i, 0], total, one, extent=1)
            T.copy(d_ub[:, 0:1], delta[bx // (SR * H), h, r0 : r0 + R])

    return main


def p4(dtype: str = "float16"):
    """Persistent form: pipelined tile loop, nested SimdVF, double buffering."""
    nsp = 4  # splits per (b, h); tiles per (b, h) = S // R = 8
    tpb = (S // R) // nsp

    @T.prim_func
    def main(
        o: T.Tensor((B * S, H, D), dtype),
        do: T.Tensor((B * S, H, D), dtype),
        delta: T.Tensor((B, H, S), "float32"),
    ) -> None:
        with T.Kernel(B * H * nsp) as bx:
            h = (bx // nsp) % H
            sp = bx % nsp
            o_ub = T.alloc_shared((R, D), dtype)
            do_ub = T.alloc_shared((R, D), dtype)
            d_ub = T.alloc_shared((R, 8), "float32")
            T.annotate_buffer_versions({o_ub: 2, do_ub: 2})
            for step in T.Pipelined(tpb, num_stages=2):
                r0 = (sp * tpb + step) * R
                T.copy(o[r0 : r0 + R, h, :], o_ub)
                T.copy(do[r0 : r0 + R, h, :], do_ub)
                with T.SimdVF():
                    one = T.simd.pset(32, "PAT_VL1")
                    for i in T.serial(R):
                        vo = T.simd.vld(o_ub[i, 0])
                        vd = T.simd.vld(do_ub[i, 0])
                        o_lo = T.simd.vcvt(vo, "float32", part=0)
                        o_hi = T.simd.vcvt(vo, "float32", part=1)
                        d_lo = T.simd.vcvt(vd, "float32", part=0)
                        d_hi = T.simd.vcvt(vd, "float32", part=1)
                        prod = T.simd.vadd(
                            T.simd.vmul(o_lo, d_lo), T.simd.vmul(o_hi, d_hi)
                        )
                        total = T.simd.vcadd(prod)
                        T.simd.vsts(d_ub[i, 0], total, one, extent=1)
                T.copy(d_ub[:, 0:1], delta[bx // (nsp * H), h, r0 : r0 + R])

    return main


def main():
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    if choice in ("p1", "all"):
        _dump("P1 strided head-slice copy + row-sum store", p1(), want=60)
    if choice in ("p2", "all"):
        _dump("P2a SIMD row dot (float16)", p2("float16"), want=60)
        _dump("P2b SIMD row dot (bfloat16)", p2("bfloat16"), want=20)
    if choice in ("p4", "all"):
        _dump("P4 pipelined persistent + nested SimdVF", p4(), want=60)


if __name__ == "__main__":
    main()
