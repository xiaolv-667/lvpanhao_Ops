"""TRAP-950-shared-shift-copy. First verified: TileLang ascend 0.1.15 / CANN 9.2.1 / Ascend950PR_9579.

A shared->shared (UB->UB) region ``T.copy`` whose destination offset is not
element-aligned delivers lane-wrong data, and a doubling (Hillis-Steele) prefix
scan that carries its running value with such a copy silently loses the shifted
term: the scan still runs to completion and produces plausible integers, but the
result is already wrong at the first doubling step and stays wrong.

Measured on Ascend950PR_9579 (real device, int32):

  * variant "copy_shift": ``T.copy(src[0:N-k], dst[k:N])`` then
    ``out[e] = src[e] + dst[e]`` -- wrong for every k in 1..32 (the values do
    not match the reference shift; in a neighbouring geometry the same form was
    observed one lane early, so do not rely on the error being a clean shift).
  * variant "in_buffer_scan" (the design's original form):
    ``s_cs[e] = s_scan[N+e] + s_scan[N+e-k]`` followed by
    ``T.copy(s_cs[0:N], s_scan[N:2N])`` -- wrong at step 0 and after the full
    doubling scan, while every intermediate value stays a plausible integer.
  * variant "two_buffer" (the fix, asserted below): keep two buffers with a
    zero halo in the lower half and never copy between them --
    ``dst[N+e] = src[N+e] + src[N+e-k]`` with ``src/dst`` alternating per step.
    Correct inclusive scan and correct exclusive prefix (shift by one lane),
    for N = 384 and N = 64.

Run (device, from the task directory):
    source .agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python \
        repro/TRAP-950-shared-shift-copy.py

Exit code 0 = the trap still reproduces and the two-buffer workaround still
holds. A non-zero exit means the toolchain changed: re-check both assertions
before trusting either claim.
"""

import math

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

STEPS_N = 384
STEPS_SMALL = 64


def build_copy_shift(n, k):
    """Shift by copy: dst[k:n] <- src[0:n-k], then add the two buffers."""

    @T.prim_func
    def main(X: T.Tensor((n,), "int32"), Y: T.Tensor((n,), "int32")):
        with T.Kernel(1):
            src = T.alloc_shared((n,), "int32")
            dst = T.alloc_shared((n,), "int32")
            out = T.alloc_shared((n,), "int32")
            T.copy(X, src)
            with T.SimtVF(threads=n):
                for e in T.Parallel(n):
                    dst[e] = 0
            T.copy(src[0 : n - k], dst[k:n])
            with T.SimtVF(threads=n):
                for e in T.Parallel(n):
                    out[e] = src[e] + dst[e]
            T.copy(out, Y)

    return tilelang.compile(main, target="ascend", out_idx=[])


def build_in_buffer_scan(n, steps):
    """The design's original doubling scan: one 2n buffer plus a copy-back."""

    @T.prim_func
    def main(X: T.Tensor((n,), "int32"), Out: T.Tensor((n,), "int32")):
        with T.Kernel(1):
            s_scan = T.alloc_shared((2 * n,), "int32")
            s_cs = T.alloc_shared((n,), "int32")
            with T.SimtVF(threads=n):
                for e in T.Parallel(n):
                    s_scan[n + e] = X[e]
            with T.SimtVF(threads=n):
                for e in T.Parallel(n):
                    s_scan[e] = 0
            for d in range(steps):
                k = 1 << d
                with T.SimtVF(threads=n):
                    for e in T.Parallel(n):
                        s_cs[e] = s_scan[n + e] + s_scan[n + e - k]
                T.copy(s_cs[0:n], s_scan[n : 2 * n])
            T.copy(s_scan[n : 2 * n], Out)

    return tilelang.compile(main, target="ascend", out_idx=[])


def build_two_buffer_scan(n, steps):
    """The fix: two buffers, zero halo in the lower half, no shared->shared copy."""

    @T.prim_func
    def main(X: T.Tensor((n,), "int32"), Inc: T.Tensor((n,), "int32"), Pref: T.Tensor((n + 4,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((2 * n,), "int32")
            b = T.alloc_shared((2 * n,), "int32")
            pref = T.alloc_shared((n + 4,), "int32")
            with T.SimtVF(threads=n):
                for e in T.Parallel(n):
                    a[n + e] = X[e]
            with T.SimtVF(threads=n):
                for e in T.Parallel(n):
                    a[e] = 0
            with T.SimtVF(threads=2 * n):
                for e in T.Parallel(2 * n):
                    b[e] = 0
            for d in range(steps):
                k = 1 << d
                if d % 2 == 0:
                    with T.SimtVF(threads=n):
                        for e in T.Parallel(n):
                            b[n + e] = a[n + e] + a[n + e - k]
                else:
                    with T.SimtVF(threads=n):
                        for e in T.Parallel(n):
                            a[n + e] = b[n + e] + b[n + e - k]
            if steps % 2 == 0:
                T.copy(a[n : 2 * n], Inc)
                with T.SimtVF(threads=n + 4):
                    for e in T.Parallel(n + 4):
                        if e <= n:
                            pref[e] = T.if_then_else(e == 0, 0, a[n + e - 1])
            else:
                T.copy(b[n : 2 * n], Inc)
                with T.SimtVF(threads=n + 4):
                    for e in T.Parallel(n + 4):
                        if e <= n:
                            pref[e] = T.if_then_else(e == 0, 0, b[n + e - 1])
            T.copy(pref, Pref)

    return tilelang.compile(main, target="ascend", out_idx=[])


def trap_copy_shift(n=STEPS_N):
    gen = torch.Generator().manual_seed(1)
    x = torch.randint(0, 100, (n,), generator=gen, dtype=torch.int32)
    checked = 0
    for k in (1, 2, 4, 8, 16, 32):
        y = torch.full((n,), -1, dtype=torch.int32, device="npu")
        build_copy_shift(n, k)(x.npu(), y)
        torch.npu.synchronize()
        want = x.clone()
        want[k:] = x[k:] + x[: n - k]
        if torch.equal(y.cpu(), want):
            print(f"[copy_shift] k={k}: unexpectedly correct")
        else:
            got = y.cpu()
            shifted = torch.cat([torch.zeros(1, dtype=got.dtype), want[:-1]])
            note = "off by one lane" if torch.equal(got, shifted) else "lane-wrong, not a clean shift"
            print(f"[copy_shift] k={k}: WRONG as recorded ({note})")
            checked += 1
    assert checked >= 5, "the copy-shift defect no longer reproduces: re-verify the scan design"


def trap_in_buffer_scan(n=STEPS_N):
    gen = torch.Generator().manual_seed(3)
    x = (torch.randint(0, 3, (n,), generator=gen, dtype=torch.int32) * 64)

    # one step: the copy-back loses the shifted term
    out1 = torch.full((n,), -1, dtype=torch.int32, device="npu")
    build_in_buffer_scan(n, 1)(x.npu(), out1)
    torch.npu.synchronize()
    want1 = x.clone()
    want1[1:] = x[1:] + x[:-1]
    assert not torch.equal(out1.cpu(), want1), "step 0 is now correct: re-verify the workaround"
    bad1 = (out1.cpu() != want1).nonzero().flatten()[:4].tolist()
    print(f"[in_buffer_scan] n={n}: step 0 WRONG at {bad1} (shift silently dropped)")

    # full doubling scan: the accumulated result is wrong
    steps = int(math.ceil(math.log2(n)))
    out = torch.full((n,), -1, dtype=torch.int32, device="npu")
    build_in_buffer_scan(n, steps)(x.npu(), out)
    torch.npu.synchronize()
    want = torch.cumsum(x.long(), 0).to(torch.int32)
    assert not torch.equal(out.cpu(), want), "the in-buffer scan is now correct: re-verify the workaround"
    print(f"[in_buffer_scan] n={n}: {steps}-step scan is wrong (defect reproduces)")


def fix_two_buffer_scan():
    for n in (STEPS_N, STEPS_SMALL):
        steps = int(math.ceil(math.log2(n)))
        gen = torch.Generator().manual_seed(5)
        x = (torch.randint(0, 3, (n,), generator=gen, dtype=torch.int32) * 64)
        inc = torch.full((n,), -1, dtype=torch.int32, device="npu")
        pref = torch.full((n + 4,), -1, dtype=torch.int32, device="npu")
        build_two_buffer_scan(n, steps)(x.npu(), inc, pref)
        torch.npu.synchronize()
        want_inc = torch.cumsum(x.long(), 0).to(torch.int32)
        want_pref = torch.zeros(n + 4, dtype=torch.int32)
        want_pref[1 : n + 1] = want_inc
        assert torch.equal(inc.cpu(), want_inc), f"two-buffer inclusive scan wrong for n={n}"
        assert torch.equal(pref.cpu()[: n + 1], want_pref[: n + 1]), f"two-buffer exclusive prefix wrong for n={n}"
        print(f"[two_buffer_scan] n={n}: inclusive and exclusive prefix both exact")


def main():
    trap_copy_shift()
    trap_in_buffer_scan()
    fix_two_buffer_scan()
    print("RESULT: trap reproduced, two-buffer workaround exact (Ascend950PR_9579, int32)")


if __name__ == "__main__":
    main()
