# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""TRAP-950-prim-func-helper-body: what an eager `@T.prim_func` body may delegate.
First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-23; origin_task batch2-20261123.

Measured on tilelang ascend 0.1.15 (site-packages) + CANN 9.2.1, ascend-950pr-9579,
device free (compile only).  `tilelang.language.prim_func` AST-mutates the decorated
function and *executes* it; the mutation is what turns `for ... in T.serial(...)`
into an IR loop and `buf[i] = value` into a builder store.  A helper called from the
body is not mutated, so:

  A) `for s in T.serial(n):` inside the helper
     -> TypeError: 'ForFrame' object is not iterable
  B) `with T.serial(n) as s:` inside the helper (the context form) + a store
     -> TypeError: 'Buffer' object does not support item assignment
  C) a helper returning a *vector expression* built from `T.simd.vld` / `vcvt` /
     `vmul` / `vadd` on arguments supplied by the caller  -> compiles (supported;
     this is how the kernel's `_upconvert` bias promote is written)
  D) `T.simd.pset(...)` created *inside* the helper
     -> Unresolved call ir.Op(name="tl.simd.pset") : the mask constructor has to be
        emitted from the mutated call site, not from an unmutated helper

Both A and B are reported against the *decorated* function, so the traceback points
at the `@T.prim_func` definition rather than at the helper -- easy to misread as a
signature or annotation problem.

Consequence for kernels whose `T.prim_func` signature depends on a build-time flag
(here: the optional `BIAS` parameter of `splitk_reduce_kernel`): the body has to be
written once per signature variant inside the decorated function; it cannot be
shared through a helper.

Run: cd <task_dir> && source .agents/platforms/toolchains/ascend/env-950pr.sh
     ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
         repro/TRAP-950-prim-func-helper-body.py
"""

import os

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

N = 256
SCALE = 2.0


def _helper_loop_form(x, y):
    """Variant A: `for ... in T.serial` inside an unmutated helper."""
    for i in T.serial(N):
        y[i] = x[i] * SCALE


def _helper_store_form(x, y):
    """Variant B: the context form plus a buffer store."""
    with T.serial(N) as i:
        y[i] = x[i] * SCALE


def _helper_value(addr, mask):
    """Variant C: returns a vector expression; only the ops take part, no store."""
    return T.simd.vmul(T.simd.vld(addr), T.simd.vdup(T.float32(SCALE), "float32", mask), mask)


def _helper_pset(addr):
    """Variant D: the mask constructor created inside the helper (unsupported)."""
    mask = T.simd.pset(32)
    return T.simd.vmul(T.simd.vld(addr), T.simd.vdup(T.float32(SCALE), "float32", mask), mask)


def _build(kind):
    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(1) as bx:
            _ = bx
            x = T.alloc_shared((N,), "float32")
            y = T.alloc_shared((N,), "float32")
            T.copy(X, x)
            if kind == "inline":
                for i in T.serial(N):
                    y[i] = x[i] * SCALE
            elif kind == "helper_loop":
                _helper_loop_form(x, y)
            elif kind == "helper_store":
                _helper_store_form(x, y)
            elif kind == "helper_value":
                with T.SimdVF():
                    full = T.simd.pset(32)
                    for ch in T.serial(N // 64):
                        T.simd.vsts(y[ch * 64], _helper_value(x[ch * 64], full), full)
            else:  # helper_pset
                with T.SimdVF():
                    for ch in T.serial(N // 64):
                        T.simd.vsts(y[ch * 64], _helper_pset(x[ch * 64]), None)
            T.copy(y, Y)

    return main


EXPECTED = {
    "inline": None,
    "helper_loop": "'ForFrame' object is not iterable",
    "helper_store": "'Buffer' object does not support item assignment",
    "helper_value": None,
    "helper_pset": "Unresolved call",
}

failures = 0
for kind, want in EXPECTED.items():
    try:
        tilelang.compile(_build(kind), target="ascend", out_idx=[])
        got = None
    except Exception as exc:  # noqa: BLE001 - the repro reports, it never raises
        got = f"{exc}".splitlines()[0].strip()
    if want is None:
        ok = got is None
        print(f"[{kind}] compile ok={ok}" + ("" if ok else f" unexpected: {got[:200]}"))
    else:
        ok = got is not None and want in got
        print(f"[{kind}] refused as documented={ok} -> {got[:120]}")
    failures += 0 if ok else 1

print(f"TOTAL checks: {len(EXPECTED)}, failures: {failures}")
print("[TRAP_CONFIRMED]" if failures == 0 else "[TRAP_NOT_REPRODUCED]")
raise SystemExit(1 if failures else 0)
