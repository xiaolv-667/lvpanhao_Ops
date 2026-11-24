"""Design probe (Stage 1, `mla_varlen_bwd/_delta_kernel`) -- D-3 compile-only evidence.

The packed `_delta_kernel` reuses the dense core (1-D core grid, `R`-row UB tile,
per-row SIMD `vcadd`, compressed write-back, build-time `nvec` expansion,
destination slicing + two-sided `T.clear`).  What differs is the *dynamic* row
axis: the factory key is `(batch, heads, dim_v, 0, dtype)`, so ``T_q`` is not a
build-time constant and no static tail segment is possible.  These probes build
-- they never launch -- the forms this design needs and the two negative
controls that justify its hard constraints.

  P1  main form, D_v == DW == 128, heads = 8: static 56-core persistent grid,
      dynamic tile stream (`ntiles = ceildiv(total, R) * heads`), dynamic-extent
      copies on both sides, `T.Pipelined(it0, it1)` two versions, per-row
      `vcadd`, compressed head-major write-back (`delta[h, t0:t0+n]`).
  P2  pad path D_v = 64 < DW = 128 with the *dynamic* extent: destination
      slicing `o_ub[0:n, 0:D_v]` + `T.clear` on both operands.
  P3  `nvec == 2` (D_v = 192) through the build-time expression helper.
  P4  narrowed ``T.Tensor`` 2-D output `delta[(heads, total)]` + `out_idx=[2]`
      allocation (the packed output contract, dynamic second axis).
  P5  heads = 128 geometry: dumps the head-major base-address expression so the
      ``h * total_q + t0`` offset and the static 4 B inner stride are visible.
  N1  negative control: D_v = 64 with an UNSLICED destination -> reproduces the
      platform rule "whole buffer to whole buffer needs equal element counts".
  N2  negative control: the SIMD *statement* moved into a module-level python
      helper -> compiles, but the compute body is silently dropped (the
      trap the design forbids; the expression helper of P3 is the allowed form).
  N3  negative control: the tail write-back with the STATIC `R` extent under a
      dynamic ``total`` -> compiles, which is exactly why the runtime clamp has
      to appear in both copies (no compiler diagnostic exists for it).

Run (no NPU, nothing is launched):

    cd examples/mla_varlen_bwd/_delta_kernel
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        history_version/design_probe_packed.py [p1|p2|p3|p4|p5|n1|n2|n3|all]
"""

import functools
import os
import sys
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

VL = 128  # 16-bit vector width (2048 bit / 16 bit)
R = 64  # rows per tile (UB budget, DESIGN.md 4.5)
G = 56  # Ascend950PR_9579 vector_core_cnt
DT = "float16"

_HITS = (
    "asc_copy_gm2ub",
    "asc_copy_ub2gm",
    "vlds_norm",
    "vcvt",
    "vmul",
    "vadd",
    "vcadd",
    "vsts_norm",
    "asc_sync_notify",
    "asc_sync_wait",
    "for (",
    "clear",
)
_SIMD_TOKENS = ("vlds_norm", "vcvt", "vmul", "vcadd", "vsts_norm")


def _dump(name: str, fn, want: int = 40, tokens=_HITS, out_idx=()):
    print(f"\n===== {name} =====", flush=True)
    try:
        kernel = tilelang.compile(fn, target="ascend", out_idx=list(out_idx))
        print("COMPILE: OK", flush=True)
    except Exception:  # noqa: BLE001 - a probe reports, it never raises
        print("COMPILE: FAIL", flush=True)
        traceback.print_exc(limit=3)
        return None
    src = kernel.get_kernel_source()
    hits = [ln.strip() for ln in src.splitlines() if any(k in ln for k in tokens)]
    for ln in hits[:want]:
        print("   ", ln[:200], flush=True)
    print(f"    [source lines: {len(src.splitlines())}]", flush=True)
    return src


def _row_dot(o_ub, do_ub, i, nvec):
    """Build-time expression helper: `nvec` is a python int, so this loop runs
    while tracing and only concatenates PrimExprs (no IR ForFrame, no statement)."""
    part = None
    for v in range(nvec):
        vo = T.simd.vld(o_ub[i, v * VL])
        vd = T.simd.vld(do_ub[i, v * VL])
        cur = T.simd.vadd(
            T.simd.vmul(
                T.simd.vcvt(vo, "float32", part=0), T.simd.vcvt(vd, "float32", part=0)
            ),
            T.simd.vmul(
                T.simd.vcvt(vo, "float32", part=1), T.simd.vcvt(vd, "float32", part=1)
            ),
        )
        part = cur if part is None else T.simd.vadd(part, cur)
    return T.simd.vcadd(part)


def _build(heads, dim_v, dtype=DT, *, slice_dst=True, static_writeback=False,
           statement_helper=False):
    dw = ((dim_v + VL - 1) // VL) * VL
    nvec = dw // VL
    total = T.dynamic("total_q")

    @T.prim_func
    def main(
        o: T.Tensor((total, heads, dim_v), dtype),  # type: ignore
        do: T.Tensor((total, heads, dim_v), dtype),  # type: ignore
        delta: T.Tensor((heads, total), "float32"),  # type: ignore
    ):
        ntiles = T.ceildiv(total, R) * heads
        with T.Kernel(G) as bx:
            o_ub = T.alloc_shared((R, dw), dtype)
            do_ub = T.alloc_shared((R, dw), dtype)
            d_ub = T.alloc_shared((R, 8), "float32")
            T.annotate_buffer_versions({o_ub: 2, do_ub: 2})
            if dw != dim_v:
                T.clear(o_ub)
                T.clear(do_ub)
            cpt = T.ceildiv(ntiles, G)
            it0 = bx * cpt
            it1 = T.min(it0 + cpt, ntiles)
            for it in T.Pipelined(it0, it1, num_stages=2):
                h = it % heads
                t0 = (it // heads) * R
                n = T.min(R, total - t0)
                m = R if static_writeback else n  # N3: the wrong (static) form
                if slice_dst:
                    T.copy(o[t0 : t0 + n, h, :], o_ub[0:n, 0:dim_v])
                    T.copy(do[t0 : t0 + n, h, :], do_ub[0:n, 0:dim_v])
                else:
                    T.copy(o[t0 : t0 + n, h, :], o_ub)
                    T.copy(do[t0 : t0 + n, h, :], do_ub)
                with T.SimdVF():
                    if statement_helper:
                        # N2: the store statement lives in a helper -> dropped.
                        for i in T.serial(R):
                            _st_helper(d_ub, i, _row_dot(o_ub, do_ub, i, nvec))
                    else:
                        one = T.simd.pset(32, "PAT_VL1")
                        for i in T.serial(R):
                            T.simd.vsts(
                                d_ub[i, 0],
                                _row_dot(o_ub, do_ub, i, nvec),
                                one,
                                extent=1,
                            )
                T.copy(d_ub[0:m, 0:1], delta[h, t0 : t0 + m])

    return main


def _st_helper(d_ub, i, value):
    one = T.simd.pset(32, "PAT_VL1")
    T.simd.vsts(d_ub[i, 0], value, one, extent=1)


def p1():
    src = _dump("P1 main D_v=128 H=8 (dynamic total, persistent 56)", _build(8, 128))
    return src


def p2():
    src = _dump("P2 pad D_v=64 < DW=128 (dst slice + both clears)", _build(4, 64))
    return src


def p3():
    src = _dump("P3 nvec=2 D_v=192 (build-time expression helper)", _build(4, 192))
    return src


def p4():
    print("\n===== P4 out_idx allocation, delta (heads, total) fp32 =====", flush=True)
    try:
        kernel = tilelang.compile(_build(8, 128), target="ascend", out_idx=[2])
        print("COMPILE: OK", flush=True)
    except Exception:  # noqa: BLE001
        print("COMPILE: FAIL", flush=True)
        traceback.print_exc(limit=3)
        return None
    src = kernel.get_kernel_source()
    for ln in src.splitlines():
        s = ln.strip()
        if "asc_copy_ub2gm" in s or "delta" in s or "total_q" in s:
            print("   ", s[:200], flush=True)
    return src


def p5():
    """heads = 128: show the dynamic head-major base address and the static stride."""
    src = _dump(
        "P5 heads=128 (head-major base h*total_q + t0)",
        _build(128, 128),
        tokens=("asc_copy_gm2ub", "asc_copy_ub2gm"),
    )
    return src


def n1():
    print("\n===== N1 D_v=64, UNSLICED destination (expected FAIL) =====", flush=True)
    kernel = _dump("N1", _build(4, 64, slice_dst=False))
    return kernel


def n2():
    print("\n===== N2 statement helper (expected: SIMD body dropped) =====", flush=True)
    src = _dump("N2", _build(4, 128, statement_helper=True), want=6)
    if src:
        missing = [t for t in _SIMD_TOKENS if t not in src]
        print(f"    [SIMD tokens missing from device code: {missing}]", flush=True)
    return src


def n3():
    print("\n===== N3 static-R write-back under dynamic total (compiles: the bug) =====", flush=True)
    src = _dump("N3", _build(4, 128, static_writeback=True),
                tokens=("asc_copy_ub2gm",))
    return src


PROBES = {"p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "n1": n1, "n2": n2, "n3": n3}


def main(argv):
    names = argv[1:] or ["all"]
    if names == ["all"]:
        names = list(PROBES)
    failed = []
    for name in names:
        fn = PROBES.get(name)
        if fn is None:
            print(f"unknown probe {name!r}; choose from {sorted(PROBES)}", flush=True)
            return 2
        try:
            fn()
        except Exception:  # noqa: BLE001 - a probe run continues
            print(f"PROBE {name}: EXCEPTION", flush=True)
            traceback.print_exc(limit=3)
            failed.append(name)
    print(f"\n[DONE] probes={names} exceptions={failed}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
