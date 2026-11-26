"""CONST-950: MLA backward main pass at the production shape.

Two revisions are recorded here; the script measures whichever body
``_bwd_kernel.py`` currently carries.

**v4 / A5 rotation (2026-11-26, current body)** -- `T.serial` + two statically
expanded half bodies + the lagged ``GB(k-1)``:

    kernel-only Task Duration    95.19 / 95.38 ms  (1.877 us per block-iteration)
    aic_cube_ratio               41.4-42.4 %       (v3 25.2 %)
    aiv_vec_ratio                88.6-89.1 %       (v3 45.7 % -> the AIV is now
                                                    the binding unit)
    aic_mte1_ratio / mte2 / fixpipe / scalar  0.383 / 0.417 / 0.378 / 0.048
                                 -> the four AIC pipes sum to 1.65 of the block
                                    time, i.e. they now *overlap* (v3: 2.74 of
                                    2.97 us, i.e. serial); this is the direct
                                    evidence that the rotation worked
    aic_mte2_active_bw           103.4 GB/s per core  (v3 118.5)
    icache_miss_rate             0.0

**v3 (pre-rotation, kept as the reference point)** -- `T.Pipelined(num_stages=2)`,
measured 2026-11-25 on `ascend-950pr-9579` (dav-3510, CANN 9.2.1, tilelang
0.1.15), `ds-v3-b2-8k`: B=2, H=128, S_q=S_kv=8192, D_qk=192, D_v=128, bfloat16,
causal, grid = 22016 blocks, 1,420,032 block-iterations.

    kernel-only Task Duration   150.7 / 151.0 ms   (2.971 us per block-iteration,
                                                   28 cores; 3.02x the v1 455.3 ms)
    aic_cube_ratio              25.2 %    (aiv_vec_ratio 45.7 %)
    aic_mte1_time               43.1 us per block = 0.67 us per iteration
    aic_mte1_active_bw          299.7 GB/s per core  -> 217,088 B / 0.67 us
    aic_mte2_active_bw          118.5 GB/s per core
    aic_time / aiv_time         188.8 / 191.3 us per block (both units saturated)
    per-iteration AIC pipe sum  cube 0.756 + mte1 0.668 + mte2 0.672 +
                                fixpipe 0.561 + scalar 0.086 = 2.74 us of 2.97 us
                                -> inside one AIC the pipes are serial, not overlapped

Two consequences worth carrying forward:

* `aic_mte1_active_bw` is ~0.30 TB/s per core, an order of magnitude below the
  "1.0-2.0 TB/s" unverified assumption of DESIGN.md 1.6.4 G, so the L1->L0
  staging of a single-version L0 (217,088 B/iteration) costs 0.67 us of the
  2.97 us iteration.  The design's 0.5 us fallback trigger is crossed but the
  prescribed `block_n=80` fallback does not compile (see
  `TRAP-950-nd2nz-rows-multiple16.py`); the next legal step down is 64, which
  doubles the q/do fill traffic.
* The four AIC pipes sum to almost exactly the iteration time, i.e. the gain of
  this revision comes from moving work onto the Cube pipe, not from overlapping
  it; overlap (Stage 4) is the remaining lever.

Ratio capture (the numbers above are the mean over blocks of the CSVs):

    msopprof --output=<dir> --launch-count=1 --warm-up=0 --kernel-name=_bwd_main_kernel \
      --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization \
      <venv>/bin/python <this file> --launch-only

(`--launch-count=1` profiles the single launch; the CSVs land in
`<dir>/OPPROF_*/PipeUtilization.csv`; Stage 3 copies are in `../evidence/`.)

Run:

    cd /tmp && source <repo>/.agents/platforms/toolchains/ascend/env-950pr.sh \
      && TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
         <repo>/examples/mla_dense_bwd/_bwd_kernel/repro/CONST-950-mla-bwd-cube-split.py --time
"""

import argparse
import importlib.util
import os
import sys
import time

import torch
import torch_npu  # noqa: F401

_OP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_bwd_kernel.py"
)
B, S, H, D, DV = 2, 8192, 128, 192, 128
T_OP_LIMIT_MS = 115.0     # scheduling contract, both revisions
T_OP_V3_MS = 150.7        # measured, T.Pipelined body
T_OP_V4_MS = 95.2         # measured, A5 rotation body


def _load():
    spec = importlib.util.spec_from_file_location("mla_bwd", _OP)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mla_bwd"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_args(dev="npu"):
    dt = torch.bfloat16
    q = torch.randn((B, S, H, D), dtype=torch.float32, device=dev).to(dt)
    k = torch.randn((B, S, H, D), dtype=torch.float32, device=dev).to(dt)
    v = torch.randn((B, S, H, DV), dtype=torch.float32, device=dev).to(dt)
    dout = torch.randn((B, S, H, DV), dtype=torch.float32, device=dev).to(dt)
    lse = torch.zeros(B, H, S, dtype=torch.float32, device=dev)
    delta = torch.zeros(B, H, S, dtype=torch.float32, device=dev)
    dq = torch.zeros(B, S, H, D, dtype=torch.float32, device=dev)
    dk = torch.empty(B, S, H, D, dtype=dt, device=dev)
    dv = torch.empty(B, S, H, DV, dtype=dt, device=dev)
    cu = torch.zeros(B + 1, dtype=torch.int32, device=dev)
    return (q, k, v, dout, lse, delta, cu, cu, dq, dk, dv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    args = ap.parse_args()
    kb = _load()
    fn = kb._launch(B, H, D, DV, S, S, True, D**-0.5, "bfloat16")
    targs = make_args()
    if not args.time:
        print("[info] pass --time to measure; measured 150.7 ms, cube 25.2 %, vec 45.7 %")
        return 0
    best = float("inf")
    for _ in range(3):
        targs[8].zero_()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn(*targs)
        torch.npu.synchronize()
        best = min(best, time.perf_counter() - t0)
    ntiles = (S + 95) // 96
    iters = sum((S // 64) - max(0, (t * 96) // 64) for t in range(ntiles)) * H * B
    ms = best * 1e3
    print(f"[CONST] T_op = {ms:.3f} ms over {iters} block-iterations "
          f"({best * 1e6 * 28 / iters:.3f} us/iteration); v3 {T_OP_V3_MS} ms, "
          f"v4 {T_OP_V4_MS} ms, acceptance <= {T_OP_LIMIT_MS} ms")
    assert ms <= T_OP_LIMIT_MS, ms
    # The rotation band: within +-8% of the measured v4 value.  A v3 body would
    # land at 150.7 ms and fail here (the two revisions are 1.58x apart), so
    # this also asserts that the rotated schedule is the one being measured.
    assert 0.92 * T_OP_V4_MS <= ms <= 1.08 * T_OP_V4_MS, ms
    print("[PASS] production shape in the A5 band and under the acceptance line")
    return 0


if __name__ == "__main__":
    sys.exit(main())
