"""Production-wiring bit-exactness check for ``MoEPermuteAlignFwdOp``.

Runs the op through the **active** wrapper import block (no injection shim) and
compares against ``MoEPermuteAlignWorkload.ref_program``:

* ``num_tokens_post_pad`` -- ``torch.equal`` (exact);
* ``expert_ids``           -- ``torch.equal`` (exact, whole array including the
                              blocks past the padded length);
* ``sorted_token_ids``     -- each expert's routed range sorted into ascending
                              route order, then ``torch.equal`` (the operator
                              contract fixes each expert's routed *set*; the order
                              inside a range is documented as free, and the frozen
                              test ``_assert_route_equivalent`` normalises the same
                              way).  Whether the raw array also matched ascending
                              order is reported, not required.

    python verify_production_bitexact.py            # manifest + edge cases
    python verify_production_bitexact.py --all      # adds the full pytest matrix
"""

import argparse
import math

import torch

from tileops.device import get_device_backend
from tileops.ops.moe.permute_align import MoEPermuteAlignFwdOp
from tileops.workloads.moe import MoEPermuteAlignWorkload

MANIFEST = [
    (1, 8, 384, 64), (32, 8, 384, 64), (512, 8, 384, 64), (4096, 8, 384, 64),
    (1, 8, 256, 64), (32, 8, 256, 64), (512, 8, 256, 64), (4096, 8, 256, 64),
    (1, 8, 128, 64), (32, 8, 128, 64), (512, 8, 128, 64), (4096, 8, 128, 64),
]
EDGE = [
    (4, 2, 4, 4, "smoke-tiny"),
    (16, 2, 8, 16, "small-bs16"),
    (1024, 8, 64, 128, "large-bs128"),
    (1, 2, 4, 4, "single-token"),
    (8, 1, 4, 4, "top-k-1"),
    (100, 2, 8, 16, "sb-numel200"),
    (100, 6, 32, 64, "sb-numel600"),
    (100, 6, 33, 64, "experts-33"),
    (1023, 1, 8, 16, "routes-1023"),
    (512, 2, 8, 16, "lb-boundary-1024"),
    (64, 8, 1024, 64, "experts-1024"),
    (64, 8, 1100, 64, "experts-past-threads"),
    (0, 8, 128, 64, "no-routes"),
    (0, 8, 8, 16, "no-routes-few-experts"),
    (1, 1, 128, 64, "one-route"),
    (2048, 1, 1, 64, "one-expert"),
    (64, 8, 128, 1, "bs1"),
    (64, 8, 128, 48, "bs-not-pow2"),
    (129, 8, 128, 64, "routes-1032"),
    (64, 8, 512, 64, "experts-512"),
    (64, 8, 513, 64, "experts-513"),
]
SKEWED = [(32, 4, 8, 16), (256, 8, 384, 64)]


def _normalise(sorted_ids, topk_ids, num_experts, block_size):
    flat = topk_ids.flatten().long()
    counts = torch.bincount(flat, minlength=num_experts)[:num_experts]
    padded = (counts + block_size - 1) // block_size * block_size
    starts = torch.cumsum(padded, 0) - padded
    out = sorted_ids.clone()
    for expert in range(num_experts):
        n = int(counts[expert])
        if n > 1:
            lo = int(starts[expert])
            out[lo:lo + n] = torch.sort(out[lo:lo + n]).values
    return out


def check(total_tokens, top_k, num_experts, block_size, label, skew=False):
    test = MoEPermuteAlignWorkload(total_tokens, top_k, num_experts, block_size)
    if skew:  # every route on expert 0
        inputs = torch.zeros((total_tokens, top_k), dtype=torch.int32,
                             device=get_device_backend().name)
    else:
        inputs = test.gen_inputs()[0]
    ref = test.ref_program(inputs)
    with torch.no_grad():
        got = MoEPermuteAlignFwdOp(num_experts, block_size)(inputs)
    got_ids, got_experts, got_count = (t.cpu() for t in got)
    ref_ids, ref_experts, ref_count = (t.cpu() for t in ref)

    numel = total_tokens * top_k
    tail = ref_count.item()
    assert torch.equal(got_count, ref_count), f"{label}: num_tokens_post_pad differs"
    assert torch.equal(got_experts, ref_experts), f"{label}: expert_ids differs"
    normed = _normalise(got_ids, inputs.cpu(), num_experts, block_size)
    assert torch.equal(normed, ref_ids), f"{label}: sorted_token_ids differs"
    raw = "ascending-order-identical" if torch.equal(got_ids, ref_ids) else "set-identical"
    print(f"  {label:22s} numel={numel:6d} E={num_experts:4d} B={block_size:3d} "
          f"P={ref_ids.numel():6d} blocks={ref_experts.numel():5d} "
          f"post_pad={tail:6d}  bit-exact=YES ({raw})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="also the full test matrix")
    args = parser.parse_args()

    print("manifest workloads:")
    for tokens, top_k, E, B in MANIFEST:
        check(tokens, top_k, E, B, f"{E}-t{tokens}")
    print("edge geometries:")
    for case in EDGE:
        check(*case)
    print("skewed (all routes on expert 0):")
    for tokens, top_k, E, B in SKEWED:
        check(tokens, top_k, E, B, f"skew-{E}-t{tokens}", skew=True)
    if args.all:
        print("full pytest matrix (tokens, top_k, E, B):")
        for tokens, top_k, E, B in [(16, 2, 8, 16), (1024, 8, 64, 128), (8, 1, 4, 4)]:
            check(tokens, top_k, E, B, f"extra-{E}-t{tokens}")
    print("[BIT-EXACT] all cases match the reference (post_pad/expert_ids exact, "
          "sorted_token_ids exact after the contract's per-expert normalisation)")


if __name__ == "__main__":
    main()
