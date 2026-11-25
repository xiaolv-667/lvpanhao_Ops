"""TRAP-950-ub-atomic-vf-scope -- repro.

First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-25;
origin_task moe_permute_align-moe_permute_align-20261124T171256Z (Stage 4).
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa

Two compile-time traps around a per-lane UB atomic (``T.atomic_add(s_cur[eid], 1)``)
in the SIMT VF form:

1. declaring the loop-local buffers **outside** the VF region makes the compiler
   pass them into ``__simt_vf__`` as pointers:
   ``error: pointer arguments to simt_vf functions must reside in '__gm__ or
   '__ubuf__' address space``;
2. putting a ``T.Parallel`` loop **outside** any VF block:
   ``[TileLang Semantic Check] T.Parallel loop 'j' must be inside a VF block``.

Usable form: ``with T.Kernel(grid) as a, T.SimtVF(threads=N):`` as one header, with
``T.alloc_local`` / ``T.alloc_shared`` inside it, exactly as the Stage 3 scatter
does.  That form compiles and runs (it is what the tuned scatter uses).

Run (device required; compile-only, one launch at the end):

    python TRAP-950-ub-atomic-vf-scope.py
"""

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

N, E, GRID, THREADS = 4096, 128, 8, 128
EPT = N // (GRID * THREADS)


def _right_form():
    @T.prim_func
    def main(flat: T.Tensor((N,), "int32"), out: T.Tensor((GRID, E), "int32")):
        with T.Kernel(GRID) as bid, T.SimtVF(threads=THREADS):
            s_hist = T.alloc_shared((E,), "int32")
            eid = T.alloc_local([1], "int32")
            for k in T.serial(E // THREADS):
                for j in T.Parallel(THREADS):
                    s_hist[k * THREADS + j] = 0
            for i in T.serial(EPT):
                for j in T.Parallel(THREADS):
                    idx = (bid * EPT + i) * THREADS + j
                    eid[0] = flat[idx]
                    T.atomic_add(s_hist[eid[0]], T.int32(1))
            T.copy(s_hist, out[bid, :])

    return tilelang.compile(main, target="ascend", out_idx=[])


def _locals_outside_vf():
    @T.prim_func
    def main(flat: T.Tensor((N,), "int32"), out: T.Tensor((GRID, E), "int32")):
        with T.Kernel(GRID) as bid:
            s_hist = T.alloc_shared((E,), "int32")
            eid = T.alloc_local([1], "int32")          # declared at kernel scope
            with T.SimtVF(threads=THREADS):
                for k in T.serial(E // THREADS):
                    for j in T.Parallel(THREADS):
                        s_hist[k * THREADS + j] = 0
                for i in T.serial(EPT):
                    for j in T.Parallel(THREADS):
                        idx = (bid * EPT + i) * THREADS + j
                        eid[0] = flat[idx]
                        T.atomic_add(s_hist[eid[0]], T.int32(1))
            T.copy(s_hist, out[bid, :])

    return tilelang.compile(main, target="ascend", out_idx=[])


def _parallel_outside_vf():
    @T.prim_func
    def main(flat: T.Tensor((N,), "int32"), out: T.Tensor((GRID, E), "int32")):
        with T.Kernel(GRID) as bid:
            s_hist = T.alloc_shared((E,), "int32")
            for j in T.Parallel(THREADS):              # no VF block at all
                s_hist[j] = 0
            T.copy(s_hist, out[bid, :])

    return tilelang.compile(main, target="ascend", out_idx=[])


def main():
    prog = _right_form()
    print("header form (T.Kernel + T.SimtVF, locals inside): COMPILE_OK")

    for name, builder, needle in (
        ("locals declared outside the VF region", _locals_outside_vf,
         "pointer arguments to simt_vf functions"),
        ("T.Parallel outside any VF block", _parallel_outside_vf,
         "must be inside a VF block"),
    ):
        try:
            builder()
        except Exception as exc:  # noqa: BLE001 - the message is the assertion
            text = str(exc)
            assert needle in text, text[:800]
            print(f"{name}: COMPILE_FAIL as expected ({needle!r})")
        else:
            raise AssertionError(f"{name}: compiled, expected the documented failure")

    gen = torch.Generator().manual_seed(5)
    flat = torch.randint(0, E, (N,), generator=gen, dtype=torch.int32).npu()
    out = torch.zeros((GRID, E), dtype=torch.int32, device="npu")
    prog(flat, out)
    torch.npu.synchronize()
    got = out.cpu().sum(0).to(torch.int32)
    expected = torch.bincount(flat.cpu().long(), minlength=E).to(torch.int32)
    assert torch.equal(got, expected), (got[:8], expected[:8])
    print("[TRAP-950-ub-atomic-vf-scope] header form runs and matches bincount: PASS")


if __name__ == "__main__":
    main()
