"""TRAP-950-simt-thread-cap. First verified: TileLang ascend 0.1.15 / CANN 9.2.1 / Ascend950PR_9579.

``T.SimtVF(threads=N)`` is capped at **2048** by the compiler: the generated
``LAUNCH_BOUND`` must be an integer constant in ``(0, 2048]``.  ``N = 2048``
compiles, ``N = 2049`` and above fail inside the device compiler step
(``Ascend device compilation failed`` ... ``Maximum number of SIMT threads
returned by LAUNCH_BOUND must be an integer constant in range (0, 2048]``).

That cap is easy to cross without noticing, because the natural lane count of a
"one lane per expert" kernel is ``E_pad = align_up(E, 64)`` and MoE models with
more than 1984 experts -- or a halo buffer of width ``2 * E_pad`` -- exceed it
while every CPU-side check still looks fine.

Measured on Ascend950PR_9579 (real device, int32):

  * ``threads=2048``: compiles, and the store lands correctly.
  * ``threads=2049``, ``2304``, ``2560``: compile fails (the reported failure of
    ``_make_align_kernel`` at E=1100 was ``2 * E_pad = 2304``).
  * the fix, ``_lane_spans``: keep every region at or below 2048 lanes and walk
    the range in consecutive spans, adding the span offset to the lane index
    (``for _o, _n in _lane_spans(2304)``).  The last assertion below checks that
    two 1152-lane passes over the same 2304 lanes produce the reference result.

Run (device, from the task directory):
    source .agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python \
        repro/TRAP-950-simt-thread-cap.py

Exit code 0 = the cap still exists at 2048 and the span-split form is exact.
"""

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

CAP = 2048
LANES = 2304          # 2 * E_pad for E_pad = 1152 (the failing E=1100 geometry)


def lane_spans(count, cap=CAP):
    """Cover [0, count) with consecutive spans of at most ``cap`` lanes."""
    if count <= cap:
        return [(0, count)]
    return [(off, min(cap, count - off)) for off in range(0, count, cap)]


def build_single(threads, n):
    """One lane region of width ``threads`` over ``n`` elements."""

    @T.prim_func
    def main(X: T.Tensor((64,), "int32"), Y: T.Tensor((LANES,), "int32")):
        with T.Kernel(1) as _b:
            y = T.alloc_shared((LANES,), "int32")
            with T.SimtVF(threads=threads):
                for i in T.Parallel(n):
                    y[i] = X[0]
            T.copy(y, Y)

    return main


def build_split(x_value_lanes):
    """The fix: write the same lanes in spans of at most ``cap`` lanes."""

    @T.prim_func
    def main(X: T.Tensor((64,), "int32"), Y: T.Tensor((LANES,), "int32")):
        with T.Kernel(1) as _b:
            y = T.alloc_shared((LANES,), "int32")
            for off, n in lane_spans(x_value_lanes):
                with T.SimtVF(threads=n):
                    for i in T.Parallel(n):
                        y[i + off] = X[0] + off
            T.copy(y, Y)

    return main


def compile_only(name, fn, expect_ok):
    try:
        tilelang.compile(fn, target="ascend", out_idx=[])
        ok = True
        detail = "COMPILE_OK"
    except Exception as exc:  # noqa: BLE001
        ok = False
        txt = " ".join(str(exc).split())
        detail = "COMPILE_FAIL" + (
            " (LAUNCH_BOUND)" if ("LAUNCH_BOUND" in txt or "SIMT threads" in txt) else ""
        )
    print(f"[{name}] {detail}")
    return ok == expect_ok


def main():
    assert lane_spans(CAP) == [(0, CAP)], "the cap-sized range must stay a single span"
    assert lane_spans(LANES) == [(0, CAP), (CAP, LANES - CAP)], lane_spans(LANES)
    assert all(n <= CAP for _, n in lane_spans(LANES + 12345)), "span width exceeded the cap"
    assert sum(n for _, n in lane_spans(LANES + 12345)) == LANES + 12345, "spans must cover the range"
    print(f"[lane_spans] {LANES} -> {lane_spans(LANES)}; every span <= {CAP}")

    good = True
    good &= compile_only("threads=2048", build_single(CAP, CAP), expect_ok=True)
    for n in (CAP + 1, 2304, 2560):
        good &= compile_only(f"threads={n}", build_single(n, n), expect_ok=False)

    # the split form must not only compile but produce the reference values
    fn = tilelang.compile(build_split(LANES), target="ascend", out_idx=[])
    x = torch.full((64,), 7, dtype=torch.int32, device="npu")
    y = torch.full((LANES,), -1, dtype=torch.int32, device="npu")
    fn(x, y)
    torch.npu.synchronize()
    want = torch.empty(LANES, dtype=torch.int32)
    for off, n in lane_spans(LANES):
        want[off : off + n] = 7 + off
    split_ok = torch.equal(y.cpu(), want)
    print(f"[split form] two spans over {LANES} lanes: {'exact' if split_ok else 'WRONG'}")
    assert split_ok, "the span-split form did not write the reference values"

    assert good, "a thread-count expectation above did not hold: re-check the cap"
    print(f"RESULT: cap = {CAP} lanes confirmed; span split exact (Ascend950PR_9579, int32)")


if __name__ == "__main__":
    main()
