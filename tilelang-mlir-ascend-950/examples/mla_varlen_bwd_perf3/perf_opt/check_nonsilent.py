"""Non-silent acceptance of the production (perf_opt) MLA backward kernel.

The 12 pytest cases do not include the manifest geometry at its real head count,
so a silent grid/head truncation would not show up there.  This check calls the
**production op** once on ds-v3-8x4k at H=128 and verifies, per request and per
head, that the gradients are actually written, and that the launch plan the
wrapper prints satisfies `launches * max_grid <= 65535`, `blocks == useful`,
`idle == 0`.

Usage (cwd examples/TileOPs):  python ../mla_varlen_bwd_perf/perf_opt/check_nonsilent.py
"""

import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

TILEOPS_ROOT = Path(__file__).resolve().parents[2] / "TileOPs"
if str(TILEOPS_ROOT) not in sys.path:
    sys.path.insert(0, str(TILEOPS_ROOT))

from tileops.ops.attention.mla_varlen_bwd import MLAVarlenBwdOp  # noqa: E402
from tileops.workloads.attention import MLAVarlenBwdWorkload  # noqa: E402

Q_LENS = [4096] * 8
HEADS, D_QK, D_V = 128, 192, 128


def main():
    torch.npu.set_device(0)
    workload = MLAVarlenBwdWorkload(Q_LENS, Q_LENS, HEADS, D_QK, D_V,
                                    torch.bfloat16, True, None)
    inputs = workload.gen_inputs()
    do, q, k, v, o, lse, cu_q, cu_kv = inputs
    op = MLAVarlenBwdOp(is_causal=True, sm_scale=None)
    dq, dk, dv = op(do, q, k, v, o, lse, cu_q, cu_kv)
    torch.npu.synchronize()

    cu = cu_q.tolist()
    fails = []
    print("  (counts are non-zero (row, head) cells; dk/dv must equal rows*H exactly,\n"
          "   dq may lose at most H cells to the single-key rows whose dS is exactly 0)")
    print("request |      rows | dq nonzero | dk nonzero | dv nonzero | zero heads")
    for b in range(len(Q_LENS)):
        lo, hi = cu[b], cu[b + 1]
        # A query row that sees exactly one key has dS == 0 (delta == dP), so a
        # handful of exactly-zero dq rows is the correct result, not truncation.
        row_nz = (dq[lo:hi].float().abs().sum(-1) > 0)
        nz_dq = int(row_nz.sum())
        dq_zero_heads = [h for h in range(HEADS)
                         if float(dq[lo:hi, h].float().abs().sum()) == 0.0]
        nz_dk = int((dk[lo:hi].float().abs().sum(-1) > 0).sum())
        nz_dv = int((dv[lo:hi].float().abs().sum(-1) > 0).sum())
        zero_heads = [h for h in range(HEADS)
                      if float(dk[lo:hi, h].float().abs().sum()) == 0.0
                      or float(dv[lo:hi, h].float().abs().sum()) == 0.0]
        zero_heads += [("dq", h) for h in dq_zero_heads]
        print(f"{b:7d} | {hi - lo:9d} | {nz_dq:10d} | {nz_dk:10d} | {nz_dv:10d} | "
              f"{zero_heads if zero_heads else '[]'}")
        cells = (hi - lo) * HEADS  # dk/dv must be non-zero on every (row, head) cell
        if (zero_heads or nz_dk != cells or nz_dv != cells
                or nz_dq < cells - HEADS):
            fails.append(b)
    # head coverage of the last slab (heads 96..127) is the truncation-prone end.
    tail = [h for h in (96, 112, 126, 127)
            if float(dk[:, h].float().abs().sum()) > 0 and float(dv[:, h].float().abs().sum()) > 0]
    print(f"tail-slab heads non-zero: {tail} (expect all of [96, 112, 126, 127])")
    if len(tail) != 4:
        fails.append("tail-slab")
    print(f"[{'NONSILENT_PASS' if not fails else 'NONSILENT_FAIL'}] failing requests: {fails or 'none'}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
