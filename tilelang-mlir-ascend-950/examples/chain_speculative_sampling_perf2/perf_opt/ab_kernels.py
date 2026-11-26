"""Per-kernel timing of the split pipeline (diagnostic).

Times the three launches of the k3 split separately, so a regression can be
attributed to the chain test, the fold or the locate stage.

Usage (task dir):
  python perf_opt/ab_kernels.py --label qwen3-235b-eagle-n5 --parts 8
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


def timed(fn, args, iters=20):
    for _ in range(5):
        fn(*args)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="qwen3-235b-eagle-n5")
    ap.add_argument("--parts", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    batch, num_draft, vocab = cand._workload_shape(args.label)
    draft, ids, target, seed, offset = cand._make_inputs(batch, num_draft, vocab, 24)
    dd, di, dt = draft.to("npu"), ids.to("npu"), target.to("npu")
    ds, do = seed.to("npu"), offset.to("npu")
    dp, dpi, tp = dd.view(-1), di.view(-1), dt.view(-1)
    sw, ow = ds.view(torch.int32).reshape(-1), do.view(torch.int32).reshape(-1)
    out_t = torch.empty(batch, num_draft + 1, dtype=torch.int32, device="npu")
    out_a = torch.empty(batch, dtype=torch.int32, device="npu")
    ot, oa = out_t.view(-1), out_a.view(-1)

    for split in ("none", "k3"):
        cand._KNOBS.clear()
        cand._KNOBS.update({"fold": "hoist", "recomp": "branch", "walk": "lane",
                            "resid": "mul", "chain": "local",
                            "split": split, "parts": args.parts})
        cand._chain_speculative_sampling_kernel.cache_clear()
        fn = cand._chain_speculative_sampling_kernel(batch, num_draft, vocab)
        if split == "none":
            us = timed(fn, (dp, dpi, tp, sw, ow, ot, oa), args.iters)
            print(f"  fused        : {us:8.1f} us", flush=True)
            continue
        print(f"  plan: {fn.describe()}", flush=True)
        ns, u, tile_sum = fn.workspace(dp.device)
        t_chain = timed(fn.k_chain, (dp, dpi, tp, sw, ow, ns, u), args.iters)
        t_fold = timed(fn.k_fold, (dp, tp, ns, tile_sum), args.iters)
        t_loc = timed(fn.k_locate, (dp, dpi, tp, ns, u, tile_sum, ot, oa), args.iters)
        t_all = timed(fn, (dp, dpi, tp, sw, ow, ot, oa), args.iters)
        print(f"  chain        : {t_chain:8.1f} us", flush=True)
        print(f"  fold         : {t_fold:8.1f} us", flush=True)
        print(f"  locate       : {t_loc:8.1f} us", flush=True)
        print(f"  sum          : {t_chain + t_fold + t_loc:8.1f} us", flush=True)
        print(f"  end to end   : {t_all:8.1f} us", flush=True)


if __name__ == "__main__":
    main()
