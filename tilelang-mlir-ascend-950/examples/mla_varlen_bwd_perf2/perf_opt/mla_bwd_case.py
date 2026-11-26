"""One MLA-varlen-bwd call for one benchmark workload, for a chosen kernel impl.

The Stage 3 wrapper (``mla_varlen_bwd.py``) launches three kernels per call:
``prep(o, do)`` (delta), one ``_bwd_main`` per head slab, and ``land(dq_acc)``,
with ``dq_acc`` zeroed once before the first slab.  This runner reproduces that
call sequence *verbatim* so a measured ``T_op`` is comparable with the wrapper's
and with the production report (``t_op_us`` = sum over the captured ops).

Two-phase use (the profiled process must not run input generation, or the
profiler's capture window fills with ``StatelessNormal``):

    # phase 1 - build the case inputs once (outside the profiler)
    python perf_opt/mla_bwd_case.py --case ds-v3-mixed-bfloat16 \
        --save-inputs /path/inp.pt

    # phase 2 - the profiled process
    python perf_opt/mla_bwd_case.py --impl baseline \
        --inputs /path/inp.pt --warm-up 5 --launch-count 10

``--impl`` is ``baseline`` (the frozen Stage 3 package), ``op`` (the TileOPs op
layer, i.e. the production path) or a path to a candidate kernel module that
exposes the module API of the tuned main pass (missing ``_delta_kernel`` /
``_dq_land_kernel`` / helpers fall back to the Stage 3 package, which is what
the wrapper's switch block assumes).
"""

import argparse
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401  -- registers torch.npu

TILEOPS_ROOT = Path(__file__).resolve().parents[2] / "TileOPs"
if str(TILEOPS_ROOT) not in sys.path:
    sys.path.insert(0, str(TILEOPS_ROOT))

BASELINE_PACKAGE = (
    "tileops.kernels.attention.mla_varlen_bwd.mla_varlen_bwd_kernel"
)
BASELINE_MODULE = BASELINE_PACKAGE + "._bwd_kernel"
INPUT_KEYS = ("do", "q", "k", "v", "o", "lse", "cu_q", "cu_kv")


class Impl:
    """The four entry points one impl must provide (missing ones fall back)."""

    def __init__(self, module, *, label):
        package = importlib.import_module(BASELINE_PACKAGE)
        helpers = importlib.import_module(BASELINE_MODULE)
        self.label = label
        self.module = module
        self.path = getattr(module, "__file__", "")
        self.delta = getattr(module, "_delta_kernel", package._delta_kernel)
        self.land = getattr(module, "_dq_land_kernel", package._dq_land_kernel)
        self.build = getattr(module, "build_kernel", helpers.build_kernel)


def load_impl(spec_str):
    """``('module', Impl)`` or ``('op', OpClass)`` for one impl selector."""
    if spec_str == "op":
        from tileops.ops.attention.mla_varlen_bwd import MLAVarlenBwdOp

        return "op", MLAVarlenBwdOp
    if spec_str == "baseline":
        module = importlib.import_module(BASELINE_MODULE)
        return "module", Impl(module, label="baseline")
    path = Path(spec_str).resolve()
    if not path.is_file():
        raise SystemExit(f"--impl: {path} is not a file")
    name = "mla_bwd_candidate_kernel"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return "module", Impl(module, label=path.name)


def benchmark_cases():
    """The benchmark's parameter sweep: ``{case_id: workload kwargs}``."""
    from tileops.manifest import load_workloads

    cases = {}
    for workload in load_workloads("MLAVarlenBwdOp"):
        for dtype_name in workload["dtypes"]:
            cases[f"{workload['label']}-{dtype_name}"] = dict(
                q_lens=list(workload["q_lens"]),
                kv_lens=list(workload["kv_lens"]),
                heads=workload["H"],
                dim_qk=workload["D_qk"],
                dim_v=workload["D_v"],
                is_causal=workload.get("is_causal", False),
                sm_scale=workload.get("sm_scale"),
                dtype=getattr(torch, dtype_name),
            )
    return cases


def build_inputs(kwargs):
    from tileops.workloads.attention import MLAVarlenBwdWorkload

    workload = MLAVarlenBwdWorkload(
        kwargs["q_lens"], kwargs["kv_lens"], kwargs["heads"], kwargs["dim_qk"],
        kwargs["dim_v"], kwargs["dtype"], kwargs["is_causal"], kwargs["sm_scale"],
    )
    do, q, k, v, o, lse, cu_q, cu_kv = workload.gen_inputs()
    return dict(do=do, q=q, k=k, v=v, o=o, lse=lse, cu_q=cu_q, cu_kv=cu_kv)


def case_kwargs(workload, dtype_name):
    """``kwargs`` of one case id, without touching the manifest module."""
    return dict(
        q_lens=list(workload["q_lens"]),
        kv_lens=list(workload["kv_lens"]),
        heads=workload["H"], dim_qk=workload["D_qk"], dim_v=workload["D_v"],
        is_causal=workload.get("is_causal", False),
        sm_scale=workload.get("sm_scale"),
        dtype=getattr(torch, dtype_name),
    )


def run_module_call(impl, kwargs, tensors):
    """The wrapper's ``_run``, verbatim, on one impl."""
    dtype_name = str(kwargs["dtype"]).split(".")[-1]
    batch = len(kwargs["q_lens"])
    heads, dim_qk, dim_v = kwargs["heads"], kwargs["dim_qk"], kwargs["dim_v"]
    scale = dim_qk ** -0.5 if kwargs["sm_scale"] is None else kwargs["sm_scale"]
    q, k, v, o, do = (tensors[name] for name in ("q", "k", "v", "o", "do"))
    lse, cu_q, cu_kv = tensors["lse"], tensors["cu_q"], tensors["cu_kv"]

    prep = impl.delta(batch, heads, dim_v, 0, dtype_name)
    land = impl.land(heads, dim_qk, scale, dtype_name)
    fn, plan, _cfg = impl.build(
        batch, heads, dim_qk, dim_v, kwargs["is_causal"], scale, dtype_name,
        int(k.shape[0]),
    )
    device = q.device
    delta = prep(o, do)
    # The wrapper's zeroing: one ``ZerosLike`` before the first slab.
    dq_acc = torch.zeros(q.shape, dtype=torch.float32, device=device)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    h_off = torch.tensor(
        [launch.head_off for launch in plan.launches], dtype=torch.int32, device=device
    )
    for index in range(len(plan.launches)):
        fn(q, k, v, do, lse, delta, cu_q, cu_kv, dq_acc, dk, dv,
           h_off[index:index + 1])
    dq = land(dq_acc)
    return dq, dk, dv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", default="baseline")
    parser.add_argument("--case", default="")
    parser.add_argument("--save-inputs", default="")
    parser.add_argument("--inputs", default="")
    parser.add_argument("--custom", default="",
                        help='JSON geometry override, e.g. \'{"q_lens":[5376,...],'
                             '"heads":128}\' (exploration only)')
    parser.add_argument("--warm-up", type=int, default=0)
    parser.add_argument("--no-setup", action="store_true",
                        help="skip the untimed setup call (the compile cache is "
                             "already warm from a pre-run); the profiled process then "
                             "issues exactly one operator call")
    parser.add_argument("--launch-count", type=int, default=1)
    args = parser.parse_args()

    if args.save_inputs:
        if args.custom:
            raw = json.loads(args.custom)
            cases = {args.case: dict(
                q_lens=raw["q_lens"], kv_lens=raw.get("kv_lens", raw["q_lens"]),
                heads=raw["heads"], dim_qk=raw.get("dim_qk", 192),
                dim_v=raw.get("dim_v", 128), is_causal=raw.get("is_causal", True),
                sm_scale=raw.get("sm_scale"), dtype=getattr(torch, raw.get("dtype", "bfloat16")),
            )}
        else:
            cases = benchmark_cases()
        if args.case not in cases:
            raise SystemExit(f"--case {args.case}: known {sorted(cases)}")
        tensors = build_inputs(cases[args.case])
        payload = {key: tensors[key].detach().cpu() for key in INPUT_KEYS}
        payload["_meta"] = {
            "case": args.case,
            "kwargs": {k: (str(v) if k == "dtype" else v)
                       for k, v in cases[args.case].items()},
        }
        torch.save(payload, args.save_inputs)
        print(f"[inputs] saved {args.case} -> {args.save_inputs} "
              f"T_q={tuple(payload['q'].shape)} T_kv={tuple(payload['k'].shape)}",
              flush=True)
        return 0

    if not args.inputs:
        raise SystemExit("--inputs or --save-inputs is required")
    payload = torch.load(args.inputs, weights_only=False)
    meta = payload["_meta"]
    kwargs = dict(meta["kwargs"])
    kwargs["dtype"] = getattr(torch, kwargs["dtype"].split(".")[-1])
    kind, impl = load_impl(args.impl)
    torch.npu.set_device(0)
    tensors = {key: payload[key].to("npu") for key in INPUT_KEYS}
    if kind == "op":
        op = impl(is_causal=kwargs["is_causal"], sm_scale=kwargs["sm_scale"])
        call = lambda: op(  # noqa: E731
            tensors["q"], tensors["k"], tensors["v"], tensors["o"], tensors["do"],
            tensors["lse"], tensors["cu_q"], tensors["cu_kv"],
        )
    else:
        call = lambda: run_module_call(impl, kwargs, tensors)  # noqa: E731
    if not args.no_setup:
        start = time.perf_counter()
        call()
        torch.npu.synchronize()
        print(f"[setup] first call {time.perf_counter() - start:.1f} s (compile)",
              flush=True)
    for _ in range(args.warm_up):
        call()
    torch.npu.synchronize()
    start = time.perf_counter()
    result = None
    for _ in range(max(args.launch_count, 1)):
        result = call()
    torch.npu.synchronize()
    per_call = (time.perf_counter() - start) / max(args.launch_count, 1)
    dq, dk, dv = result
    checksum = float(dk.float().abs().sum() + dv.float().abs().sum()
                     + dq.float().abs().sum())
    print(f"[events] per-call {per_call * 1e3:.3f} ms (not the reported metric), "
          f"checksum={checksum:.6f} impl={impl.label if kind == 'module' else 'op'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
