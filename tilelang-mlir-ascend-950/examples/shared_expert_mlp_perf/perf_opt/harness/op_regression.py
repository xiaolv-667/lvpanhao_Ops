"""Op-level precision regression for a ``perf_opt`` candidate.

Mirrors ``examples/TileOPs/tests/ops/test_fused_moe_shared_expert.py`` case for
case, with the same workload objects, the same reference program and the same
comparison policy (``tileops.workloads.moe.moe_close``: elementwise
``atol = rtol = 1e-2`` plus normalized squared error ``<= 1e-4``).  It drives
the candidate kernel class directly (the Stage 5 wrapper still points at the
Stage 3 package, which this stage must not modify).

Usage::

    python op_regression.py --impl ../SharedExpertMLPFwdOp.py [--only smoke]
"""

import argparse
import importlib.util
import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401 -- registers torch.npu

TILEOPS_DIR = "/workspace/tilelang-mlir-ascend-950/examples/TileOPs"
sys.path.insert(0, TILEOPS_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tileops.manifest import load_workloads  # noqa: E402
from tileops.workloads.moe import SharedExpertMLPWorkload, moe_close  # noqa: E402

# (id, T, H, S, dtype, tier)
CASES = [
    ("matches-its-reference-bf16", 32, 256, 128, "bfloat16", "smoke"),
    ("matches-its-reference-fp16", 32, 256, 128, "float16", "smoke"),
    ("ragged", 600, 256, 7384, "float16", "smoke"),
    ("one-wave", 1024, 256, 512, "bfloat16", "smoke"),
    ("separate", 4096, 256, 512, "bfloat16", "full"),
    ("two-waves", 4096, 256, 2048, "bfloat16", "full"),
    ("narrow-expert", 512, 512, 256, "float16", "full"),
    ("narrow-tiles", 512, 1024, 1020, "bfloat16", "full"),
    ("splits-k-wide-expert", 32, 2048, 1408, "bfloat16", "smoke"),
    ("splits-k-narrow-model", 32, 1024, 6144, "bfloat16", "full"),
    ("gate-up-row-order", 16, 256, 128, "bfloat16", "smoke"),
]


def load_module(path):
    spec = importlib.util.spec_from_file_location("cand_op", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cand_op"] = module
    spec.loader.exec_module(module)
    return module


def manifest_cases():
    rows = []
    for workload in load_workloads("SharedExpertMLPFwdOp"):
        tokens, hidden = workload["hidden_states_shape"]
        ffn = workload["w_down_shape"][1]
        label = workload.get("label", f"t{tokens}-h{hidden}-s{ffn}")
        for dtype_name in workload["dtypes"]:
            rows.append((f"manifest-{label}-{dtype_name}", tokens, hidden, ffn,
                         dtype_name, "manifest"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--only", default="", help="smoke | full | manifest | ''")
    args = ap.parse_args()
    module = load_module(os.path.abspath(args.impl))

    cases = CASES + manifest_cases()
    if args.only:
        cases = [case for case in cases if case[5] == args.only]

    failures = []
    for name, tokens, hidden, ffn, dtype_name, tier in cases:
        dtype = getattr(torch, dtype_name)
        workload = SharedExpertMLPWorkload(tokens, hidden, ffn, dtype)
        inputs = workload.gen_inputs()
        kernel = module.SharedExpertMLPKernel(tokens, hidden, ffn, dtype)
        try:
            with torch.no_grad():
                got = kernel(*inputs)
            ref = workload.ref_program(*inputs)
            if name == "gate-up-row-order":
                gate, up = inputs[1].chunk(2, dim=0)
                swapped = torch.cat((up, gate), dim=0)
                with torch.no_grad():
                    bad = kernel(inputs[0], swapped, inputs[2])
                try:
                    moe_close(bad, ref)
                    raise AssertionError("a swapped gate/up order must not match")
                except AssertionError as exc:
                    if "must not match" in str(exc):
                        raise
                moe_close(got, ref)
            else:
                moe_close(got, ref)
            assert got.shape == ref.shape and got.dtype == ref.dtype
            print(f"[PASS][{tier}] {name} T={tokens} H={hidden} S={ffn} {dtype_name}")
        except Exception as exc:  # noqa: BLE001 - reported per case
            failures.append((name, f"{type(exc).__name__}: {exc}".splitlines()[0][:220]))
            print(f"[FAIL][{tier}] {name} T={tokens} H={hidden} S={ffn} {dtype_name}: "
                  f"{failures[-1][1]}")
            traceback.print_exc(limit=2)
    print(f"cases={len(cases)} failed={len(failures)}")
    if failures:
        print("[PRECISION_FAIL]")
        raise SystemExit(1)
    print("[PRECISION_PASS]")


if __name__ == "__main__":
    main()
