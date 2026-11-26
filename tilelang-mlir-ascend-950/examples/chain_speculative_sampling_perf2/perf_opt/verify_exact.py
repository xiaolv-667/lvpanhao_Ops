"""Bit-exactness gate: this round's candidate vs the round-1 tuned kernel (baseline).

Runs the baseline kernel factory and the ``perf_opt`` candidate on the same
inputs and requires ``output_token_ids`` *and* ``num_accepted`` to agree bit for
bit on every case, plus agreement with the torch reference on ``num_accepted``
(the op's contract).  Cases: the benchmark workloads, the L0/L1 shapes, the
control-flow paths (all accepted / first reject / mid reject / zero residual),
a shifted (``storage_offset=1``) view, and an odd vocabulary.

Usage (task dir):
  python perf_opt/verify_exact.py [--knobs split=k3]
"""

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

import ChainSpeculativeSamplingFwdOp as cand  # noqa: E402

BASELINE = (
    Path(__file__).resolve().parents[2]
    / "chain_speculative_sampling_perf/perf_opt/chain_speculative_sampling.py"
)

# (name, batch, num_draft, vocab, gen_seed, mode, shift)
CASES = [
    ("smoke (2,2,64)", 2, 2, 64, 11, "random", False),
    ("(4,3,64)", 4, 3, 64, 12, "random", False),
    ("all accepted (4,3,64)", 4, 3, 64, 13, "all_accepted", False),
    ("reject first (4,3,64)", 4, 3, 64, 14, "reject_first", False),
    ("reject mid (4,5,64)", 4, 5, 64, 15, "reject_mid", False),
    ("B=1 (1,2,64)", 1, 2, 64, 16, "random", False),
    ("llama-8b-b16-n4", 16, 4, 128256, 21, "random", False),
    ("llama-8b-b64-n4", 64, 4, 128256, 22, "random", False),
    ("ds-v3-mtp-n1", 64, 1, 129280, 23, "random", False),
    ("ds-v3-mtp-n3", 64, 3, 129280, 24, "random", False),
    ("qwen3-235b-eagle-n5", 32, 5, 151936, 25, "random", False),
    ("llama-8b-b17-n8", 17, 8, 128256, 26, "random", False),
    ("llama2-7b-b256-n1", 256, 1, 32000, 27, "random", False),
    ("V not a multiple of 4 (8,3,4099)", 8, 3, 4099, 31, "random", False),
    ("19 tiles (2,3,151936)", 2, 3, 151936, 32, "random", False),
    ("per-core 6 rows (300,3,32000)", 300, 3, 32000, 33, "random", False),
    ("N=600 (4,600,64)", 4, 600, 64, 34, "random", False),
    ("V one tile (16,2,8192)", 16, 2, 8192, 35, "random", False),
    ("V past a tile (16,2,8193)", 16, 2, 8193, 36, "random", False),
    ("all accepted big V (2,4,32000)", 2, 4, 32000, 44, "all_accepted", False),
    ("zero residual (4,3,64)", 4, 3, 64, 55, "zero_residual", False),
    ("shifted (2,2,64)", 2, 2, 64, 51, "random", True),
    ("shifted odd V (3,3,4099)", 3, 3, 4099, 52, "random", True),
    ("one-token V (4,2,1)", 4, 2, 1, 53, "random", False),
    ("B=256 N=3 (256,3,32000)", 256, 3, 32000, 41, "random", False),
    ("all rejected (64,3,129280)", 64, 3, 129280, 61, "all_rejected", False),
]


def load_baseline():
    spec = importlib.util.spec_from_file_location("css_baseline", BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(module, inputs):
    draft, ids, target, seed, offset = inputs
    batch, num_draft, _ = draft.shape
    out_tokens = torch.empty(batch, num_draft + 1, dtype=torch.int32, device="npu")
    out_acc = torch.empty(batch, dtype=torch.int32, device="npu")
    fn = module._chain_speculative_sampling_kernel(batch, num_draft, draft.shape[2])
    fn(
        draft.contiguous().view(-1),
        ids.contiguous().view(-1),
        target.contiguous().view(-1),
        seed.view(torch.int32).reshape(-1),
        offset.view(torch.int32).reshape(-1),
        out_tokens.view(-1),
        out_acc.view(-1),
    )
    return out_tokens, out_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knobs", default="")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    if args.knobs:
        for item in args.knobs.split(","):
            key, value = item.split("=")
            cand._KNOBS[key] = value
    print(f"knobs={cand._KNOBS}", flush=True)
    baseline = load_baseline()

    failures = 0
    for name, batch, num_draft, vocab, gen_seed, mode, shift in CASES:
        if args.only and args.only not in name:
            continue
        draft, ids, target, seed, offset = cand._make_inputs(batch, num_draft, vocab, gen_seed, mode)
        if shift:
            dev = "npu"
            dd, dt = cand._shifted_views(draft, target, dev)
        else:
            dd, dt = draft.to("npu"), target.to("npu")
        inputs = (dd, ids.to("npu"), dt, seed.to("npu"), offset.to("npu"))
        torch.npu.synchronize()
        base_tok, base_acc = _run(baseline, inputs)
        torch.npu.synchronize()
        got_tok, got_acc = _run(cand, inputs)
        torch.npu.synchronize()
        bt, ba = base_tok.cpu(), base_acc.cpu()
        gt, ga = got_tok.cpu(), got_acc.cpu()
        ref_tok, ref_acc = cand.golden_chain_speculative_sampling(
            draft, ids, target, seed, offset
        )
        tok_diff = int((gt != bt).sum()) if batch else 0
        acc_diff = int((ga != ba).sum()) if batch else 0
        ref_diff = int((ga != ref_acc).sum())
        ok = tok_diff == 0 and acc_diff == 0 and ref_diff == 0
        failures += 0 if ok else 1
        print(
            f"[{'PASS' if ok else 'FAIL'}] {name:34s} tok_diff={tok_diff} "
            f"acc_diff={acc_diff} acc_vs_ref={ref_diff} "
            f"n=[{int(ga.min())},{int(ga.max())}] drawn!=golden={int((gt[torch.arange(batch), ga.long()] != ref_tok[torch.arange(batch), ref_acc.long()]).sum())}",
            flush=True,
        )
    print(f"{'ALL CASES BIT-EXACT' if not failures else f'{failures} CASE(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
