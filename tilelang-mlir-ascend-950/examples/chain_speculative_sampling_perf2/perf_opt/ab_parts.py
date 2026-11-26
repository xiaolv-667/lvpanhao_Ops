"""Interleaved multi-run A/B of the split parameters (diagnostic ranking).

Times knob sets on the same inputs in one process, alternating them run by run,
and reports the median of the per-run means.  The recorded metric stays the
platform's msprof Task Duration; this only ranks candidates.

Usage (task dir):
  python perf_opt/ab_parts.py --label ds-v3-mtp-n3 \
      --set "none=split=none" --set "p4=split=k3,parts=4" --set "p16=split=k3,parts=16"
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ChainSpeculativeSamplingFwdOp as cand  # noqa: E402

DEFAULTS = {"fold": "hoist", "recomp": "branch", "walk": "lane",
            "resid": "mul", "chain": "local", "split": "none", "parts": 0}


def build(batch, num_draft, vocab, knobs):
    cand._KNOBS.clear()
    cand._KNOBS.update(DEFAULTS)
    cand._KNOBS.update(knobs)
    cand._chain_speculative_sampling_kernel.cache_clear()
    return cand._chain_speculative_sampling_kernel(batch, num_draft, vocab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="ds-v3-mtp-n3")
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    batch, num_draft, vocab = cand._workload_shape(args.label)
    draft, ids, target, seed, offset = cand._make_inputs(batch, num_draft, vocab, 24)
    dd, di, dt = draft.to("npu"), ids.to("npu"), target.to("npu")
    ds, do = seed.to("npu"), offset.to("npu")
    out_t = torch.empty(batch, num_draft + 1, dtype=torch.int32, device="npu")
    out_a = torch.empty(batch, dtype=torch.int32, device="npu")
    call = (
        dd.view(-1), di.view(-1), dt.view(-1),
        ds.view(torch.int32).reshape(-1), do.view(torch.int32).reshape(-1),
        out_t.view(-1), out_a.view(-1),
    )
    arms = []
    for spec in args.set:
        name, _, knobs = spec.partition("=")
        parsed = dict(item.split("=") for item in knobs.split(",")) if knobs else {}
        if "parts" in parsed:
            parsed["parts"] = int(parsed["parts"])
        fn = build(batch, num_draft, vocab, parsed)
        plan = fn.describe() if hasattr(fn, "describe") else {"split": "none"}
        arms.append((name, plan, fn))
    print(f"{args.label}: B={batch} N={num_draft} V={vocab}", flush=True)
    for name, plan, _ in arms:
        print(f"  {name}: {plan}", flush=True)

    samples = {name: [] for name, _, _ in arms}
    for _ in range(3):
        for _, _, fn in arms:
            fn(*call)
    torch.npu.synchronize()
    for _ in range(args.rounds):
        for name, _, fn in arms:
            t0 = time.perf_counter()
            for _ in range(args.iters):
                fn(*call)
            torch.npu.synchronize()
            samples[name].append((time.perf_counter() - t0) / args.iters * 1e6)
    base = None
    for name, _, _ in arms:
        med = statistics.median(samples[name])
        if base is None:
            base = med
        runs = " ".join(f"{s:.0f}" for s in samples[name])
        print(f"  {name:6s} median={med:8.1f} us  ({base / med:.2f}x)  runs=[{runs}]", flush=True)


if __name__ == "__main__":
    main()
