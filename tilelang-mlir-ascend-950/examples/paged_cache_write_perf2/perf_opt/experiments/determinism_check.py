"""Manual-sync determinism: the same inputs must give byte-identical pools on every launch.

With ``tl.enable_auto_schedule=False`` the kernel owns its synchronization, so a
race would show up as a run-to-run difference rather than a compile error.
Runs the delivered file's fp8 case three times on one set of buffers and compares
the two pools byte-for-byte with the workload reference after each launch.
"""

import importlib.util
import os
import sys

import torch
import torch_npu  # noqa: F401

PERF_OPT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PERF_OPT)

# the delivered module itself (not the production wrapper, which still points at
# the previous round's kernel)
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "pcw_round2", os.path.join(PERF_OPT, "PagedKVCacheWriteFwdOp.py"))
mod = importlib.util.module_from_spec(_spec)
sys.modules["pcw_round2"] = mod
_spec.loader.exec_module(mod)

from tileops.workloads.attention import PagedKVCacheWriteWorkload  # noqa: E402


def canonical(byte_view):
    return torch.where(byte_view == 0xFF, torch.tensor(0x7F, dtype=torch.uint8), byte_view)


def main():
    w = PagedKVCacheWriteWorkload(4096, 4, 128, 16, 4096, torch.bfloat16,
                                  torch.float8_e4m3fn)
    inputs = w.gen_inputs()
    gold = w.ref_program(*inputs)
    def op(k, v, kp, vp, slots, ks, vs):
        return mod._write(k, v, kp, vp, slots, ks, vs)
    ok = True

    def compare(tag, want_pools):
        nonlocal ok
        for got, want in zip((inputs[2], inputs[3]), want_pools):
            g = canonical(got.cpu().contiguous().view(torch.uint8))
            e = canonical(want.cpu().contiguous().view(torch.uint8))
            if not torch.equal(g, e):
                ok = False
                print(f"[det] {tag}: {int((g != e).sum())} bytes differ")

    # launch 0 on pristine pools: must equal the workload reference
    op(*inputs)
    torch.npu.synchronize()
    compare("launch 0 vs reference", gold)
    snap = (inputs[2].clone(), inputs[3].clone())
    # replays: the write is idempotent, so every replay must reproduce launch 0
    for it in range(1, 3):
        op(*inputs)
        torch.npu.synchronize()
        compare(f"launch {it} vs launch 0", snap)
    print(f"[det] {'DETERMINISTIC_PASS' if ok else 'DETERMINISTIC_FAIL'} "
          f"(3 launches, fp8 4096x512, byte-exact per launch)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
