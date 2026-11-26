"""Bitwise / tolerance parity of a candidate kernel against the Stage 3 baseline.

Runs both impls on the same inputs in one process and compares the three
gradients.  ``dk``/``dv`` are assignments from a fixed-order L0C accumulation and
must be **bitwise** identical; ``dq_acc`` is a store-atomic fp32 accumulation
whose arrival order is unspecified, so it is compared with the DESIGN.md 8.2
relative tolerance (1e-3) plus a bitwise-equal report.

Usage:
    python perf_opt/verify_parity.py --case mla-varlen-bwd-smoke-bfloat16 \
        --cand perf_opt/MLAVarlenBwdOp_round1_c1.py
    python perf_opt/verify_parity.py --custom '{"q_lens":[300,512],"kv_lens":[700,96],"heads":3}' \
        --case parity-ragged --cand perf_opt/...
"""

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401  -- registers torch.npu

HERE = Path(__file__).resolve().parent
TILEOPS_ROOT = HERE.parents[1] / "TileOPs"
if str(TILEOPS_ROOT) not in sys.path:
    sys.path.insert(0, str(TILEOPS_ROOT))

from mla_bwd_case import (  # noqa: E402
    BASELINE_MODULE, BASELINE_PACKAGE, INPUT_KEYS, benchmark_cases, build_inputs,
    run_module_call,
)


class Impl:
    def __init__(self, module, label):
        package = importlib.import_module(BASELINE_PACKAGE)
        helpers = importlib.import_module(BASELINE_MODULE)
        self.label = label
        self.delta = getattr(module, "_delta_kernel", package._delta_kernel)
        self.land = getattr(module, "_dq_land_kernel", package._dq_land_kernel)
        self.build = getattr(module, "build_kernel", helpers.build_kernel)


def load_module(spec):
    if spec == "baseline":
        return Impl(importlib.import_module(BASELINE_MODULE), "baseline")
    path = Path(spec).resolve()
    spec_obj = importlib.util.spec_from_file_location("parity_cand", path)
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules["parity_cand"] = module
    spec_obj.loader.exec_module(module)
    return Impl(module, path.name)


def compare(name, got, ref, atol, rtol):
    a, b = got.float(), ref.float()
    exact = bool(torch.equal(a, b))
    gap = torch.nan_to_num((a - b).abs(), nan=0.0, posinf=0.0, neginf=0.0)
    bad = int((gap > atol + rtol * b.abs()).sum())
    print(f"    {name:8s} bitwise={'yes' if exact else 'no ':3s} "
          f"max_abs_diff={float(gap.max()):.3e} cells_over_tol={bad} "
          f"ref_absmax={float(b.abs().max()):.4g}")
    return exact, bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--custom", default="")
    parser.add_argument("--cand", required=True)
    args = parser.parse_args()

    if args.custom:
        raw = json.loads(args.custom)
        kwargs = dict(q_lens=list(raw["q_lens"]), kv_lens=list(raw.get("kv_lens", raw["q_lens"])),
                      heads=int(raw["heads"]), dim_qk=int(raw.get("dim_qk", 192)),
                      dim_v=int(raw.get("dim_v", 128)),
                      is_causal=bool(raw.get("is_causal", True)), sm_scale=None,
                      dtype=getattr(torch, raw.get("dtype", "bfloat16")))
        tensors = build_inputs(kwargs)
    else:
        kwargs = benchmark_cases()[args.case]
        tensors = build_inputs(kwargs)

    torch.npu.set_device(0)
    inputs = {key: tensors[key].to("npu") for key in INPUT_KEYS}
    base = load_module("baseline")
    cand = load_module(args.cand)
    ref = run_module_call(base, kwargs, inputs)
    got = run_module_call(cand, kwargs, inputs)
    # A second baseline pass: dq_acc's atomic order is not reproducible, so the
    # pair (baseline, candidate) must be read against (baseline, baseline).
    ref2 = run_module_call(base, kwargs, inputs)
    print(f"[{args.case}] q_lens={kwargs['q_lens']} kv_lens={kwargs['kv_lens']} "
          f"H={kwargs['heads']} {kwargs['dtype']}")
    print("  baseline vs baseline (run-to-run floor):")
    compare("dk", ref2[1], ref[1], 0.0, 0.0)
    compare("dv", ref2[2], ref[2], 0.0, 0.0)
    compare("dq32", ref2[0].float(), ref[0].float(), 0.0, 0.0)
    print(f"  candidate {cand.label} vs baseline:")
    ok_dk = compare("dk", got[1], ref[1], 0.0, 0.0)
    ok_dv = compare("dv", got[2], ref[2], 0.0, 0.0)
    ok_dq = compare("dq32", got[0].float(), ref[0].float(), 1e-3, 1e-3)
    ok = ok_dk[0] and ok_dv[0] and ok_dq[1] == 0
    print(f"  [{'PARITY_PASS' if ok else 'PARITY_FAIL'}] dk/dv bitwise, dq_acc within 1e-3")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
