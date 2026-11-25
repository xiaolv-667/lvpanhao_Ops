"""Wall-clock sweep over the tune workloads for a set of stage spellings.

Diagnostic only (the recorded metric is the platform's msprof Task Duration);
used to rank branches quickly.  ``--reference`` times the Stage 3 baseline
kernel on the same inputs for a same-run comparison.

Usage (task dir):
  python perf_opt/sweep_case.py --variants select,mul,hoist --walk lane
"""

import argparse
import importlib.util
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(HERE))
import ChainSpeculativeSamplingFwdOp as cand  # noqa: E402

BASELINE = (
    Path(__file__).resolve().parents[2]
    / "TileOPs/tileops/kernels/sampling/chain_speculative_sampling"
    / "chain_speculative_sampling_kernel/_chain_speculative_sampling_kernel.py"
)

TUNE = [
    "llama-8b-b16-n4",
    "llama-8b-b64-n4",
    "ds-v3-mtp-n1",
    "ds-v3-mtp-n3",
    "qwen3-235b-eagle-n5",
    "llama-8b-b17-n8",
    "llama2-7b-b256-n1",
]


def load_baseline():
    spec = importlib.util.spec_from_file_location("css_baseline", BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timeit(run, iters=20, warmup=5):
    for _ in range(warmup):
        run()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="select,mul,hoist")
    ap.add_argument("--walk", default="lane")
    ap.add_argument("--labels", default=",".join(TUNE))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--reference", action="store_true")
    args = ap.parse_args()

    baseline = load_baseline() if (args.reference or "reference" in args.variants) else None
    rows = []
    for label in args.labels.split(","):
        batch, num_draft, vocab = cand._workload_shape(label)
        draft, ids, target, seed, offset = cand._make_inputs(batch, num_draft, vocab, 24)
        dd, di, dt = draft.to("npu"), ids.to("npu"), target.to("npu")
        ds, do = seed.to("npu"), offset.to("npu")
        for variant in args.variants.split(","):
            if variant == "reference":
                fn = baseline._chain_speculative_sampling_kernel(batch, num_draft, vocab)
            else:
                cand._KNOBS.update({"fold": variant, "walk": args.walk, "recomp": "branch"})
                cand._chain_speculative_sampling_kernel.cache_clear()
                fn = cand._chain_speculative_sampling_kernel(batch, num_draft, vocab)
            out_t = torch.empty(batch, num_draft + 1, dtype=torch.int32, device="npu")
            out_a = torch.empty(batch, dtype=torch.int32, device="npu")
            call = (
                dd.view(-1), di.view(-1), dt.view(-1),
                ds.view(torch.int32).reshape(-1), do.view(torch.int32).reshape(-1),
                out_t.view(-1), out_a.view(-1),
            )
            us = timeit(lambda: fn(*call), args.iters)
            rows.append((label, variant, us))
            print(f"{label:24s} {variant:10s} {us:9.1f} us", flush=True)
            del fn
    print()
    best = {}
    for label, variant, us in rows:
        best.setdefault(label, []).append((us, variant))
    for label, items in best.items():
        items.sort()
        base = dict((v, u) for u, v in items)
        ref = base.get("reference") or base.get("select")
        summary = "  ".join(f"{v}={u:.1f}us({ref / u:.2f}x)" for u, v in items)
        print(f"{label:24s} {summary}")


if __name__ == "__main__":
    main()
