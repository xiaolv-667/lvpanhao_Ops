"""Production acceptance checks for the adopted round-2 kernel (op layer).

Runs the *integration package* exactly as the benchmark and the op tests do
(``PagedKVCacheWriteFwdOp()`` with no eager geometry, so the wrapper resolves the
geometry and routes the tile through the tuned ``pick_config``):

1. fp8 ``qwen3-235b-fp8`` — both pools compared byte-for-byte with the workload
   reference (raw bytes, NaN sign canonicalised), plus a repeat-launch
   determinism check: launch 0 against the reference, launches 1 and 2 against
   launch 0 (the quantised build owns its synchronization now, so a race would
   show up here rather than in a compile error).
2. prefill ``llama-8b-prefill`` — byte-for-byte comparison (the 16-bit path).

    python production_check.py        # cwd = examples/TileOPs
"""

import os
import sys

import torch
import torch_npu  # noqa: F401

from tileops.ops.attention.paged_cache_write import PagedKVCacheWriteFwdOp
from tileops.workloads.attention import PagedKVCacheWriteWorkload

CASES = {
    "fp8-4096x512": (4096, 4, 128, 16, 4096, torch.bfloat16, torch.float8_e4m3fn),
    "prefill-8192x1024": (8192, 8, 128, 16, 4096, torch.bfloat16, torch.bfloat16),
}


def canonical(byte_view):
    """Backend canonicalises f32->e4m3 NaN to 0x7F; the contract ignores the NaN sign."""
    return torch.where(byte_view == 0xFF, torch.tensor(0x7F, dtype=torch.uint8), byte_view)


def compare(tag, pools, want):
    ok = True
    for got, ref in zip(pools, want):
        g = got.cpu().contiguous().view(torch.uint8)
        e = ref.cpu().contiguous().view(torch.uint8)
        if got.dtype is torch.float8_e4m3fn:
            g, e = canonical(g), canonical(e)
        if not torch.equal(g, e):
            ok = False
            print(f"[{tag}] FAIL: {int((g != e).sum())} of {g.numel()} bytes differ")
    return ok


def main():
    ok = True
    for name, spec in CASES.items():
        tokens, heads, dim, page, pages, dtype, cache_dtype = spec
        w = PagedKVCacheWriteWorkload(tokens, heads, dim, page, pages, dtype, cache_dtype)
        inputs = w.gen_inputs()
        gold = w.ref_program(*inputs)
        op = PagedKVCacheWriteFwdOp()

        op(*inputs)  # launch 0 on pristine pools
        torch.npu.synchronize()
        ok &= compare(f"{name} launch0-vs-ref", (inputs[2], inputs[3]), gold)
        snap = (inputs[2].clone(), inputs[3].clone())
        for it in (1, 2):
            op(*inputs)
            torch.npu.synchronize()
            ok &= compare(f"{name} launch{it}-vs-launch0", (inputs[2], inputs[3]), snap)
        kernel = next(iter(op._kernel_cache.values()))
        print(f"[{name}] tile={kernel._compiled_config} byte_exact+deterministic={ok}")
    print(f"[production] {'BYTE_EXACT_DETERMINISTIC_PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
