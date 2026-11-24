"""Design probe (Stage 1, ``splitk_reduce_kernel``) -- D-3 compile-only API evidence.

Nothing is launched (``ASCEND_RT_VISIBLE_DEVICES=""``): every kernel below is only
built, and the generated source is scanned for what the compiler actually emits.

Probed structures (DESIGN.md 3.3 / 3.4):

  P1  non-gated + bias, split_k=4, (m,n,out_n)=(32,7168,7168), BC=1024
      -- split_k slice DMAs, chained vadd, bias UNPK_B16, PK_B32 store
  P2  gated + bias (silu_and_mul), split_k=4, (32,4096,2048), BC=512
      -- two column streams (j and out_n+j) in one block
  P3  gated + bias with a partial column block (out_n=1536, BC=1024)
      -- runtime ``bc = min(BC, out_n-c0)`` region + ``update_mask`` tail
  P4  split_k=1, no bias, activation="none" (the degenerate copy) in fp16 and
      bf16 -- the m<16 fallback the wrapper needs (split_k=1)
  P5  out_dtype="float32" (fp32 output store, NORM_B32 instead of PK_B32)
  P6  glm45-t32 gate/up geometry (32,3072,1536), BC=512 -- divisor path, no mask

Run (no NPU, no kernel launch):
    cd examples/shared_expert_mlp/splitk_reduce_kernel
    source .agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        history_version/design_probe_compile.py [p1|p2|p3|p4|p5|p6|all]
"""

import os
import sys
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

VL = 64  # fp32 vector register width (PL-950-vector-execution / engram_gate_conv)
ITEMS = {"float16": 2, "bfloat16": 2, "float32": 4}


def build(split_k, m, n, activation, has_bias, dtype, out_dtype, BC):
    """The DESIGN.md 3.3 structure, one build-time configuration per call."""
    gated = activation == "silu_and_mul"
    out_n = n // 2 if gated else n
    c_dtype = out_dtype or dtype
    c_items = ITEMS[c_dtype]
    nc = (out_n + BC - 1) // BC
    grid = m * nc
    tail = (out_n % BC) != 0
    nch = BC // VL
    pk = "PK_B32" if c_items == 2 else "NORM_B32"

    @T.prim_func
    def main(
        W: T.Tensor((split_k, m, n), "float32"),
        BIAS: T.Tensor((n,), dtype),
        C: T.Tensor((m, out_n), c_dtype),
    ):
        with T.Kernel(grid) as bx:
            row = bx // nc
            c0 = (bx % nc) * BC
            bc = T.min(BC, out_n - c0)

            g_ub = T.alloc_shared((split_k, BC), "float32")
            u_ub = T.alloc_shared((split_k, BC), "float32") if gated else None
            b_ub = T.alloc_shared((2 if gated else 1, BC), dtype) if has_bias else None
            o_ub = T.alloc_shared((BC,), c_dtype)

            for s in T.serial(split_k):
                T.copy(W[s, row, c0 : c0 + bc], g_ub[s, :bc])
                if gated:
                    T.copy(W[s, row, out_n + c0 : out_n + c0 + bc], u_ub[s, :bc])
            if has_bias:
                T.copy(BIAS[c0 : c0 + bc], b_ub[0, :bc])
                if gated:
                    T.copy(BIAS[out_n + c0 : out_n + c0 + bc], b_ub[1, :bc])

            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.vdup(T.float32(1), "float32", full)
                for ch in T.serial(nch):
                    mask = T.simd.update_mask(T.min(VL, bc - ch * VL), 32) if tail else full
                    g = T.simd.alloc_local((1,), "float32")
                    g[0] = T.simd.vld(g_ub[0, ch * VL])
                    for s in T.serial(split_k - 1):
                        g[0] = T.simd.vadd(g[0], T.simd.vld(g_ub[s + 1, ch * VL]), mask)
                    if has_bias:
                        g[0] = T.simd.vadd(
                            g[0],
                            T.simd.vcvt(
                                T.simd.vld(b_ub[0, ch * VL], dist="UNPK_B16"),
                                "float32",
                                part=0,
                            ),
                            mask,
                        )
                    if gated:
                        u = T.simd.alloc_local((1,), "float32")
                        u[0] = T.simd.vld(u_ub[0, ch * VL])
                        for s in T.serial(split_k - 1):
                            u[0] = T.simd.vadd(
                                u[0], T.simd.vld(u_ub[s + 1, ch * VL]), mask
                            )
                        if has_bias:
                            u[0] = T.simd.vadd(
                                u[0],
                                T.simd.vcvt(
                                    T.simd.vld(b_ub[1, ch * VL], dist="UNPK_B16"),
                                    "float32",
                                    part=0,
                                ),
                                mask,
                            )
                        # A1: silu(g)*u rewritten as g/(1+exp(-g))*u (one vmul fewer)
                        den = T.simd.vadd(
                            one, T.simd.vexp(T.simd.vneg(g[0], mask), mask), mask
                        )
                        y = T.simd.vmul(T.simd.vdiv(g[0], den, mask), u[0], mask)
                    else:
                        y = g[0]
                    # fp32 output: no vcvt (vcvt fp32->fp32 is rejected by the
                    # front end), store the accumulator vector directly.
                    ys = (
                        y
                        if c_dtype == "float32"
                        else T.simd.vcvt(y, c_dtype, round="ROUND_R", sat=False, part=0)
                    )
                    T.simd.vsts(o_ub[ch * VL], ys, mask, dist=pk)
            T.copy(o_ub[:bc], C[row, c0 : c0 + bc])

    return main


PROBES = {
    "p1": lambda: build(4, 32, 7168, "none", True, "float16", None, 1024),
    "p2": lambda: build(4, 32, 4096, "silu_and_mul", True, "bfloat16", None, 512),
    "p3": lambda: build(4, 32, 3072, "silu_and_mul", True, "bfloat16", None, 1024),
    "p4a": lambda: build(1, 32, 7168, "none", False, "float16", None, 1024),
    "p4b": lambda: build(1, 32, 7168, "none", False, "bfloat16", None, 1024),
    "p5": lambda: build(4, 32, 7168, "none", True, "float16", "float32", 1024),
    "p6": lambda: build(4, 32, 3072, "silu_and_mul", True, "bfloat16", None, 512),
}

MARKERS = {
    "P1": ("asc_copy_gm2ub", "vadd", "vsts", "asc_copy_ub2gm"),
    "P2": ("vlds_unpack", "vexp", "vdiv", "vsts_pack_b32"),
    "P3": ("update_mask", "vsts_pack_b32"),
    "P4": ("vld", "vsts"),
}


def _dump(name: str, fn, shown=60):
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
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    for ln in lines[:shown]:
        print("  " + ln, flush=True)
    print(f"  ... ({len(lines)} lines total)", flush=True)
    for tag, pats in MARKERS.items():
        hit = {p: sum(p in ln for ln in lines) for p in pats}
        if any(hit.values()):
            print(f"  MARKER {tag}: {hit}", flush=True)
    # scalar-loop tripwire: a float2 scalar loop means a vector op was lowered away
    print(
        "  TRIPWIRE float2-loop:",
        sum(("float2" in ln and "for" in ln) for ln in lines),
        flush=True,
    )
    return True


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    keys = list(PROBES) if which == "all" else [which]
    ok = {}
    for k in keys:
        ok[k] = _dump(k, PROBES[k]())
    print("\nSUMMARY:", {k: ("OK" if v else "FAIL") for k, v in ok.items()}, flush=True)
