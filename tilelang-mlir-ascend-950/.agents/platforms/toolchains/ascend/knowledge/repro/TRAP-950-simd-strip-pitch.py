# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Repro: rows of a 2D UB buffer written by ``T.copy`` must be 32 B aligned.
First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-23; origin_task batch2-20261123.

Target: ascend-950pr-9579 / dav-3510, tilelang 0.1.15 + CANN 9.2.1.

Found while implementing the Engram GateConv backward
(``examples/engram_gate_conv_bwd``): the four saved fp32 row scalars
(``alpha`` / ``rrms_h`` / ``rrms_k`` / ``rrms_v``) are staged once per batch row
into a ``(4, seq_len)`` fp32 UB strip and then read per token with
``vld(sc[k, t], dist="BRC_B32")``.  With ``seq_len`` in {1, 2, 4} -- or with the
strip sized exactly to ``seq_len`` when ``seq_len`` is not a multiple of 8
(the ``(3, 401, 256)`` geometry) -- the kernel died with

    npuSynchronizeDevice: NPU function error: device error type 3, error code 507035

and no compiler diagnostic.  Padding the row pitch to
``max(8, ceil(seq_len / 8) * 8)`` fp32 elements (32 B) made every geometry pass.

The rule: every row of the destination buffer must start on a 32 B boundary, so
the fp32 row pitch must be at least 8 elements and a multiple of 8.  The number
of elements *copied* is free: case B copies one fp32 element per row.

Assertions (measured on the device):
  A  pitch 8 (32 B): 8 elements per row, 4 rows   -> exact
  B  pitch 8 (32 B): 1 element per row, 4 rows    -> exact
  C  pitch 408 (1632 B): 401 elements per row     -> exact   (seq_len = 401)
  D  pitch 1 (4 B): 1 element per row             -> faults (device error 507035).
     Not asserted: the fault poisons the process, so it runs last and only with
     ``--fault``; its measured output is reported, not compared.

run:
  cd <kernel task dir>
  source <repo>/.agents/platforms/toolchains/ascend/env-950pr.sh
  ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python \
      repro/TRAP-950-simd-strip-pitch.py            # A, B, C
  ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python \
      repro/TRAP-950-simd-strip-pitch.py --fault    # A, B, C, then D
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
import torch_npu  # noqa: F401  # registers torch.npu


@tilelang.jit(out_idx=[1], target="ascend")
def build_strip(pitch: int, rows: int, cols: int):
    """Copy ``rows`` strips of ``cols`` fp32 values at ``pitch`` fp32 per row."""

    @T.prim_func
    def main(X: T.Tensor((rows, cols), "float32"), OUT: T.Tensor((rows, cols), "float32")):
        with T.Kernel(1) as bx:
            sc = T.alloc_shared((rows, pitch), "float32")
            T.annotate_buffer_versions({sc: 1})
            for k in T.serial(rows):
                T.copy(X[k, :], sc[k, :cols])
            T.copy(sc[:, :cols], OUT)

    return main


def check(tag, pitch, rows, cols):
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(rows, cols, generator=gen, dtype=torch.float32)
    fn = build_strip(pitch, rows, cols)
    got = fn(x.to("npu"))
    torch.npu.synchronize()
    ok = torch.equal(got.cpu(), x)
    print(f"  {tag}: pitch={pitch} ({pitch * 4} B) rows={rows} cols={cols} exact={ok}",
          flush=True)
    assert ok, f"{tag}: the strip did not round trip"
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault", action="store_true", help="also run the 4 B pitch case")
    args = parser.parse_args()

    print("[A] 32 B pitch, full rows", flush=True)
    check("A", 8, 4, 8)
    print("[B] 32 B pitch, single element per row", flush=True)
    check("B", 8, 4, 1)
    print("[C] 32 B pitch, seq_len = 401", flush=True)
    check("C", 408, 4, 401)

    if args.fault:
        print("[D] 4 B pitch (expected: device error 507035)", flush=True)
        fn = build_strip(1, 4, 1)
        fn(torch.randn(4, 1, dtype=torch.float32).to("npu"))
        try:
            torch.npu.synchronize()
            print("  D: surprisingly survived", flush=True)
        except RuntimeError as exc:
            print(f"  D: {str(exc).splitlines()[0]}", flush=True)
        return

    print("REPRO_RESULT: ALL_ASSERTIONS_HOLD", flush=True)


if __name__ == "__main__":
    main()
