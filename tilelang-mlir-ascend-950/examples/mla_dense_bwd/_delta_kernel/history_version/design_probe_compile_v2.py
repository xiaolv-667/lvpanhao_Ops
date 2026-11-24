"""Design probe v2 (Stage 1 revision 1, _delta_kernel) -- D-3 compile-only evidence.

Answers the three blocking findings of REVIEW.md by building (never launching)
the *corrected* §3.3 forms:

  F1  nvec == 1, build-time specialised single expression, dst-sliced copies,
      D_v == DW == 128, persistent `T.Pipelined` + double buffering  (§3.3 main form)
  F2  D_v = 64 < DW = 128: destination region `o_ub[0:R, 0:D_v]` + `T.clear` on
      BOTH operands (blocker 2 fix); prints the two copy descriptors
  F3  D_v = 64 with an UNSLICED destination -> reproduces the blocker-2 failure
      (`PlanMTECopy ... got src=4096 and dst=8192`)
  F4  tail tile: `nfull` full tiles in the pipelined loop + one build-time
      `nlast` segment whose two copies AND the row loop use the static `nlast`
      extent (blocker 3 fix), S=65 / R=64 / nlast=1
  F5  nvec == 2 (D_v = 256): hand-unrolled two-segment single expression
  F6  nvec == 2 through a python-level unroll helper (checks that a helper
      called while tracing can bind several `vld` registers and return one
      expression -- the generic form of the nvec > 1 rule)
  F7  the v0 §3.3 form `for v in range(nvec)` + `acc` rebind -> reproduces the
      blocker-1 failure (`Immutable variable 'acc' is used outside its defining
      region`).  NOTE: this case makes the toolchain abort the process, so it is
      *not* part of `all` -- run `... design_probe_compile_v2.py f7` on its own.

Run (no NPU, no kernel launch):
    cd examples/mla_dense_bwd/_delta_kernel
    source .agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        history_version/design_probe_compile_v2.py [f1|f2|f3|f4|f5|f6|f7|all]
"""

import os
import sys
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

VL = 128  # 16-bit vector width (2048 bit / 16 bit)
R = 64  # rows per tile
DT = "float16"


def _dump(name: str, fn, want: int = 12, only_copy: bool = False) -> bool:
    print(f"\n===== {name} =====", flush=True)
    try:
        kernel = tilelang.compile(fn, target="ascend", out_idx=[])
        print("COMPILE: OK", flush=True)
    except Exception:  # noqa: BLE001 - a probe reports, it never raises
        print("COMPILE: FAIL", flush=True)
        traceback.print_exc(limit=3)
        return False
    if only_copy:
        for ln in kernel.get_kernel_source().splitlines():
            s = ln.strip()
            if any(t in s for t in ("asc_copy_gm2ub", "asc_copy_ub2gm")):
                print("   ", s[:170], flush=True)
        return True
    hits = [
        ln.strip()
        for ln in kernel.get_kernel_source().splitlines()
        if any(
            k in ln
            for k in (
                "asc_copy_gm2ub",
                "asc_copy_ub2gm",
                "vlds_norm",
                "vcvt",
                "vmul",
                "vadd",
                "vcadd",
                "vsts_norm",
                "asc_sync_notify",
                "for (",
            )
        )
    ]
    for ln in hits[:want]:
        print("   ", ln[:170], flush=True)
    if len(hits) > want:
        print(f"    ... ({len(hits) - want} more matching lines)", flush=True)
    return True


def _row_dot_expr(o_ub, do_ub, i: int, nvec: int):
    """Python-level (build-time) unrolling of the `nvec` segment dot products.

    Never passes a vector value across an IR `for` frame: every `T.simd.vld`
    binds a register in the *current* scope and the segments are combined into
    ONE nested expression `vadd(vadd(p0,p1),p2)...`.
    """
    acc = None
    for v in range(nvec):
        vo = T.simd.vld(o_ub[i, v * VL])
        vd = T.simd.vld(do_ub[i, v * VL])
        part = T.simd.vadd(
            T.simd.vmul(T.simd.vcvt(vo, "float32", part=0), T.simd.vcvt(vd, "float32", part=0)),
            T.simd.vmul(T.simd.vcvt(vo, "float32", part=1), T.simd.vcvt(vd, "float32", part=1)),
        )
        acc = part if acc is None else T.simd.vadd(acc, part)
    return acc


# ----------------------------------------------------------------------------- F1/F2
def persistent(D: int, S: int, H: int, nsp: int, dt: str = DT, sliced: bool = True,
               clear_both: bool = True):
    """§3.3 main form: D_v <= DW, persistent split, dst-sliced copies."""
    B = 1
    DW = ((D + VL - 1) // VL) * VL
    nvec = DW // VL
    nt = (S + R - 1) // R
    tpb = nt // nsp
    nfull = nt  # S % R == 0 in F1/F2

    @T.prim_func
    def main(
        o: T.Tensor((B * S, H, D), dt),
        do: T.Tensor((B * S, H, D), dt),
        delta: T.Tensor((B, H, S), "float32"),
    ) -> None:
        with T.Kernel(B * H * nsp) as bx:
            bh = bx // nsp
            sp = bx % nsp
            b, h = bh // H, bh % H
            o_ub = T.alloc_shared((R, DW), dt)
            do_ub = T.alloc_shared((R, DW), dt)
            d_ub = T.alloc_shared((R, 8), "float32")
            T.annotate_buffer_versions({o_ub: 2, do_ub: 2})
            if DW != D:  # build-time: pad columns must be zero on BOTH operands
                T.clear(o_ub)
                T.clear(do_ub)
            for it in T.Pipelined(tpb, num_stages=2):
                s0 = (sp * tpb + it) * R
                r0 = b * S + s0
                if sliced:
                    T.copy(o[r0 : r0 + R, h, :], o_ub[0:R, 0:D])
                    T.copy(do[r0 : r0 + R, h, :], do_ub[0:R, 0:D])
                else:
                    T.copy(o[r0 : r0 + R, h, :], o_ub)
                    T.copy(do[r0 : r0 + R, h, :], do_ub)
                with T.SimdVF():
                    one = T.simd.pset(32, "PAT_VL1")
                    for i in T.serial(R):
                        # nvec == 1 / nvec == 2 / ... : one build-time expression
                        prod = _row_dot_expr(o_ub, do_ub, i, nvec)
                        T.simd.vsts(d_ub[i, 0], T.simd.vcadd(prod), one, extent=1)
                T.copy(d_ub[0:R, 0:1], delta[b, h, s0 : s0 + R])

    return main


# --------------------------------------------------------------------------------- F4
def tail(S: int = 65, H: int = 8, D: int = 128):
    """Tail segment uses the static `nlast` extent for both copies and the row loop."""
    B = 1
    nt = (S + R - 1) // R
    nfull = S // R  # full R-row tiles
    nlast = S - nfull * R  # build-time constant (1 here)

    @T.prim_func
    def main(
        o: T.Tensor((B * S, H, D), DT),
        do: T.Tensor((B * S, H, D), DT),
        delta: T.Tensor((B, H, S), "float32"),
    ) -> None:
        with T.Kernel(B * H) as bx:
            b, h = bx // H, bx % H
            o_ub = T.alloc_shared((R, D), DT)
            do_ub = T.alloc_shared((R, D), DT)
            d_ub = T.alloc_shared((R, 8), "float32")
            T.annotate_buffer_versions({o_ub: 2, do_ub: 2})
            if nfull > 0:  # build-time: only full tiles go through the pipeline
                for it in T.Pipelined(nfull, num_stages=2):
                    s0 = it * R
                    r0 = b * S + s0
                    T.copy(o[r0 : r0 + R, h, :], o_ub[0:R, 0:D])
                    T.copy(do[r0 : r0 + R, h, :], do_ub[0:R, 0:D])
                    with T.SimdVF():
                        one = T.simd.pset(32, "PAT_VL1")
                        for i in T.serial(R):
                            prod = _row_dot_expr(o_ub, do_ub, i, 1)
                            T.simd.vsts(d_ub[i, 0], T.simd.vcadd(prod), one, extent=1)
                    T.copy(d_ub[0:R, 0:1], delta[b, h, s0 : s0 + R])
            if nlast != R:  # build-time: the tail segment, static nlast everywhere
                s0 = nfull * R
                r0 = b * S + s0
                T.copy(o[r0 : r0 + nlast, h, :], o_ub[0:nlast, 0:D])
                T.copy(do[r0 : r0 + nlast, h, :], do_ub[0:nlast, 0:D])
                with T.SimdVF():
                    one = T.simd.pset(32, "PAT_VL1")
                    for i in T.serial(nlast):
                        prod = _row_dot_expr(o_ub, do_ub, i, 1)
                        T.simd.vsts(d_ub[i, 0], T.simd.vcadd(prod), one, extent=1)
                T.copy(d_ub[0:nlast, 0:1], delta[b, h, s0 : s0 + nlast])

    return main


# --------------------------------------------------------------------------------- F7
def design_v0_form(D: int = 128, S: int = 512, H: int = 8):
    """v0 §3.3 verbatim: `for v in range(nvec)` + `acc` rebind (expected to fail)."""
    B = 1
    DW = ((D + VL - 1) // VL) * VL
    nvec = DW // VL
    SR = S // R

    @T.prim_func
    def main(
        o: T.Tensor((B * S, H, D), DT),
        do: T.Tensor((B * S, H, D), DT),
        delta: T.Tensor((B, H, S), "float32"),
    ) -> None:
        with T.Kernel(B * H * SR) as bx:
            h = (bx // SR) % H
            sp = bx % SR
            o_ub = T.alloc_shared((R, DW), DT)
            do_ub = T.alloc_shared((R, DW), DT)
            d_ub = T.alloc_shared((R, 8), "float32")
            r0 = sp * R
            T.copy(o[r0 : r0 + R, h, :], o_ub)
            T.copy(do[r0 : r0 + R, h, :], do_ub)
            with T.SimdVF():
                one = T.simd.pset(32, "PAT_VL1")
                for i in T.serial(R):
                    acc = None
                    for v in range(nvec):
                        vo = T.simd.vld(o_ub[i, v * VL])
                        vd = T.simd.vld(do_ub[i, v * VL])
                        lo = T.simd.vadd(
                            T.simd.vmul(T.simd.vcvt(vo, "float32", part=0),
                                        T.simd.vcvt(vd, "float32", part=0)),
                            T.simd.vmul(T.simd.vcvt(vo, "float32", part=1),
                                        T.simd.vcvt(vd, "float32", part=1)),
                        )
                        acc = lo if acc is None else T.simd.vadd(acc, lo)
                    T.simd.vsts(d_ub[i, 0], T.simd.vcadd(acc), one, extent=1)
            T.copy(d_ub[0:R, 0:1], delta[bx // (SR * H), h, r0 : r0 + R])

    return main


def main():
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    ok = {}
    if choice in ("f1", "all"):
        ok["F1 D=128 nvec=1 specialised (persistent, sliced dst)"] = _dump(
            "F1 D=128 nvec=1 specialised single expression, persistent + 2 versions",
            persistent(128, 512, 8, 8), want=14)
    if choice in ("f2", "all"):
        ok["F2 D=64 DW=128 sliced dst + clear both"] = _dump(
            "F2 D=64 < DW=128: dst slice o_ub[0:R,0:D] + T.clear on both operands",
            persistent(64, 256, 4, 1), want=6, only_copy=True)
    if choice in ("f3", "all"):
        ok["F3 D=64 unsliced dst (must FAIL)"] = _dump(
            "F3 D=64 with UNSLICED destination (v0 §3.3 form; expected FAIL)",
            persistent(64, 256, 4, 1, sliced=False), want=4, only_copy=True)
    if choice in ("f4", "all"):
        ok["F4 tail S=65 nlast=1 static extents"] = _dump(
            "F4 tail segment: nfull pipelined full tiles + static nlast=1 segment",
            tail(), want=8, only_copy=True)
    if choice in ("f8", "all"):
        ok["F8 S=1 (nfull=0, tail only)"] = _dump(
            "F8 S=1: no full tile at all (`tpb == 0` branch skips the pipelined loop)",
            tail(S=1, H=1, D=128), want=6, only_copy=True)
    if choice in ("f5", "all"):
        ok["F5 D=256 nvec=2 hand-unrolled"] = _dump(
            "F5 D=256 nvec=2: two segments combined in one build-time expression",
            persistent(256, 512, 8, 8), want=10)
    if choice in ("f6", "all"):
        ok["F6 D=256 nvec=2 via helper"] = _dump(
            "F6 D=192 nvec=2 (DW=256) through the python-level unroll helper",
            persistent(192, 512, 8, 8), want=10)
    if choice == "f7":
        # Negative control only: the toolchain aborts the process here, so it never
        # runs as part of `all` (the SUMMARY below would be lost).
        ok["F7 v0 loop form (must FAIL)"] = _dump(
            "F7 v0 §3.3 `for v in range(nvec)` + acc rebind (expected FAIL)",
            design_v0_form(), want=4)
    if choice == "all":
        print("\n(F7 negative control skipped in `all`: it aborts the process; run `f7` alone)", flush=True)
    print("\n===== SUMMARY =====", flush=True)
    for k, v in ok.items():
        print(f"   {'OK  ' if v else 'FAIL'}  {k}", flush=True)


if __name__ == "__main__":
    main()
