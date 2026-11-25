"""Stage 3 baseline entry for profiling: one benchmark workload, one call.

The Stage 3 file is a read-only upstream artifact whose ``__main__`` only runs
its layered precision levels, so this wrapper imports its factory and repeats
the launch the op's ``forward`` does -- the same call the perf_opt candidate's
``--case`` entry makes.

Usage (task dir):
  python perf_opt/baseline_case.py --case ds-v3-mtp-n3 [--timeit 20]
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401  -- registers torch.npu

BASELINE = (
    Path(__file__).resolve().parents[2]
    / "TileOPs/tileops/kernels/sampling/chain_speculative_sampling"
    / "chain_speculative_sampling_kernel/_chain_speculative_sampling_kernel.py"
)
TILEOPS_ROOT = Path(__file__).resolve().parents[2] / "TileOPs"
if str(TILEOPS_ROOT) not in sys.path:
    sys.path.insert(0, str(TILEOPS_ROOT))


def load_baseline():
    spec = importlib.util.spec_from_file_location("css_baseline", BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def workload_shape(label):
    from tileops.manifest import load_workloads

    for workload in load_workloads("ChainSpeculativeSamplingFwdOp"):
        if workload.get("label") == label:
            batch, num_draft, vocab = tuple(workload["draft_probs_shape"])
            return batch, num_draft, vocab
    raise ValueError(f"no benchmark workload labelled {label!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--timeit", type=int, default=0)
    args = ap.parse_args()

    baseline = load_baseline()
    batch, num_draft, vocab = workload_shape(args.case)
    draft, ids, target, seed, offset = baseline._make_inputs(batch, num_draft, vocab, 24)
    dev = "npu"
    dd, di, dt = draft.to(dev), ids.to(dev), target.to(dev)
    ds, do = seed.to(dev), offset.to(dev)
    out_t = torch.empty(batch, num_draft + 1, dtype=torch.int32, device=dev)
    out_a = torch.empty(batch, dtype=torch.int32, device=dev)
    fn = baseline._chain_speculative_sampling_kernel(batch, num_draft, vocab)
    call = (
        dd.view(-1), di.view(-1), dt.view(-1),
        ds.view(torch.int32).reshape(-1), do.view(torch.int32).reshape(-1),
        out_t.view(-1), out_a.view(-1),
    )
    fn(*call)
    torch.npu.synchronize()
    if args.timeit:
        for _ in range(3):
            fn(*call)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.timeit):
            fn(*call)
        torch.npu.synchronize()
        print(f"baseline case={args.case} wall={(time.perf_counter() - t0) / args.timeit * 1e6:.1f} us",
              flush=True)
    else:
        print(f"baseline case={args.case} B={batch} N={num_draft} V={vocab}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
