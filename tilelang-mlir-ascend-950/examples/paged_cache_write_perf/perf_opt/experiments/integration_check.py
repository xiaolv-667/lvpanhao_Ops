"""Integration-path byte-exact check + tile probe (production wiring acceptance).

Runs the op layer exactly as ``benchmarks/ops/bench_paged_cache_write.py`` and
``tests/ops/test_paged_kv_cache_write.py`` do (``PagedKVCacheWriteFwdOp()`` with
no eager geometry), asserts the pools are byte-exact against the workload's
``ref_program``, and prints the tile the tuned ``pick_config`` resolved.

    python integration_check.py [--workload fp8|prefill|all]
"""

import argparse
import sys

import torch
import torch_npu  # noqa: F401

from tileops.ops.attention.paged_cache_write import PagedKVCacheWriteFwdOp
from tileops.workloads.attention import PagedKVCacheWriteWorkload

CASES = {
    "fp8": dict(tokens=4096, heads_kv=4, dim=128, page_size=16, num_pages=4096,
                dtype=torch.bfloat16, cache_dtype=torch.float8_e4m3fn),
    "prefill": dict(tokens=8192, heads_kv=8, dim=128, page_size=16, num_pages=4096,
                    dtype=torch.bfloat16, cache_dtype=torch.bfloat16),
    "decode": dict(tokens=64, heads_kv=8, dim=128, page_size=16, num_pages=4096,
                   dtype=torch.bfloat16, cache_dtype=torch.bfloat16),
}


def nan_canonical(byte_view):
    """Backend canonicalises f32->e4m3 NaN to 0x7F; the contract ignores the NaN sign."""
    return torch.where(byte_view == 0xFF, torch.tensor(0x7F, dtype=torch.uint8), byte_view)


def check(name, spec):
    workload = PagedKVCacheWriteWorkload(
        spec["tokens"], spec["heads_kv"], spec["dim"], spec["page_size"], spec["num_pages"],
        spec["dtype"], spec["cache_dtype"],
    )
    inputs = workload.gen_inputs()
    # the reference clones the pools before writing them, so read the pools the
    # op mutated (inputs[2], inputs[3]) and compare them with the clones
    gold = workload.ref_program(*inputs)
    op = PagedKVCacheWriteFwdOp()
    out = op(*inputs)
    assert out is None, "the op must return None (in-place contract)"
    kernel = next(iter(op._kernel_cache.values()))
    cfg = kernel._compiled_config
    ok = True
    for got, want in zip((inputs[2], inputs[3]), gold):
        got_c, want_c = got.cpu(), want.cpu()
        torch.testing.assert_close(got_c, want_c, atol=0, rtol=0, equal_nan=True)
        gb = got_c.contiguous().view(torch.uint8)
        wb = want_c.contiguous().view(torch.uint8)
        if got_c.dtype is torch.float8_e4m3fn:
            gb, wb = nan_canonical(gb), nan_canonical(wb)
        if not torch.equal(gb, wb):
            ok = False
            n = int((gb != wb).sum())
            print(f"[{name}] FAIL {n} of {gb.numel()} bytes differ")
    bytes_moved = 2 * spec["tokens"] * spec["heads_kv"] * spec["dim"] * 2 + 2 * spec["tokens"] * spec["heads_kv"] * spec["dim"] * (1 if spec["cache_dtype"] is torch.float8_e4m3fn else 2)
    print(f"[{name}] byte_exact={ok} tile={cfg} bytes={bytes_moved}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="all", choices=["fp8", "prefill", "decode", "all"])
    args = ap.parse_args()
    names = list(CASES) if args.workload == "all" else [args.workload]
    ok = all([check(n, CASES[n]) for n in names])
    print(f"[integration] {'BYTE_EXACT_PASS' if ok else 'BYTE_EXACT_FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
