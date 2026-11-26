"""One MLA-dense-bwd call for one benchmark workload, for a chosen kernel impl.

``MLADenseBwdOp`` launches, per call (``mla_dense_bwd.py`` ``_run`` /
``forward``):

1. ``torch.arange`` x2 (``cu_seqlens_q`` / ``cu_seqlens_kv``) -- one ``Range`` op,
2. ``torch.zeros`` for the fp32 ``dq_acc`` buffer -- one ``ZerosLike`` op,
3. ``prep(o, do)`` -- ``_delta_main_kernel``,
4. ``main(q4, k4, v4, do4, lse, delta, cu_q, cu_kv, dq_acc4, dk4, dv4)`` --
   ``_bwd_main_kernel`` (the 4-D batch views of DESIGN.md 0.2 / 5.4),
5. ``land(dq_acc)`` -- ``_dq_land_main_kernel``.

This runner reproduces that call sequence *verbatim* so a measured ``T_op`` is
comparable with the wrapper's and with the production report (``t_op_us`` = sum
over every captured launch).

Two-phase use (the profiled process must not run input generation, or the
profiler's capture window fills with ``StatelessNormal`` / forward kernels):

    # phase 1 - build the case inputs once (outside the profiler)
    python perf_opt/mla_dense_case.py --case ds-v3-b2-8k-bfloat16 \
        --save-inputs /workspace/mla_dense_bwd_opt/inputs/x.pt

    # phase 2 - the profiled process
    python perf_opt/mla_dense_case.py --impl baseline \
        --inputs /path/inp.pt --warm-up 5 --launch-count 1

``--impl`` is ``baseline`` (the Stage 3 package), ``op`` (the TileOPs op layer,
i.e. the production path) or a path to a candidate kernel module that exposes the
module API of the tuned main pass (``_bwd_kernel``; missing ``_delta_kernel`` /
``_dq_land_kernel`` fall back to the Stage 3 package, which is what the wrapper's
switch block assumes).
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

BASELINE_PACKAGE = "tileops.kernels.attention.mla_dense_bwd.mla_dense_bwd_kernel"
BASELINE_MODULE = BASELINE_PACKAGE + "._bwd_kernel"
INPUT_KEYS = ("do", "q", "k", "v", "o", "lse")
# One call launches: Range x2 (``cu_seqlens_q`` and ``cu_seqlens_kv`` are two
# separate ``torch.arange`` calls -- measured ``n=2`` in one capture window),
# ZerosLike (the fp32 ``dq_acc``), ``_delta_main``, ``_bwd_main``,
# ``_dq_land_main``.  Verified by ``window_ok`` on every capture.
LAUNCHES_PER_CALL = 6


class Impl:
    """The three entry points one impl must provide (missing ones fall back)."""

    def __init__(self, module, *, label):
        package = importlib.import_module(BASELINE_PACKAGE)
        self.label = label
        self.module = module
        self.path = getattr(module, "__file__", "")
        self.delta = getattr(module, "_delta_kernel", package._delta_kernel)
        self.land = getattr(module, "_dq_land_kernel", package._dq_land_kernel)
        self.bwd = getattr(module, "_bwd_kernel", package._bwd_kernel)


def load_impl(spec_str):
    """``('module', Impl)`` or ``('op', OpClass)`` for one impl selector."""
    if spec_str == "op":
        from tileops.ops.attention.mla_dense_bwd import MLADenseBwdOp

        return "op", MLADenseBwdOp
    if spec_str == "baseline":
        module = importlib.import_module(BASELINE_MODULE)
        return "module", Impl(module, label="baseline")
    path = Path(spec_str).resolve()
    if not path.is_file():
        raise SystemExit(f"--impl: {path} is not a file")
    name = "mla_dense_candidate_kernel"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return "module", Impl(module, label=path.name)


def benchmark_cases():
    """The benchmark's parameter sweep: ``{case_id: workload kwargs}``."""
    from tileops.manifest import load_workloads

    cases = {}
    for workload in load_workloads("MLADenseBwdOp"):
        for dtype_name in workload["dtypes"]:
            cases[f"{workload['label']}-{dtype_name}"] = dict(
                batch=int(workload["B"]),
                seq_q=int(workload["S_q"]),
                seq_kv=int(workload["S_kv"]),
                heads=int(workload["H"]),
                dim_qk=int(workload["D_qk"]),
                dim_v=int(workload["D_v"]),
                is_causal=workload.get("is_causal", False),
                sm_scale=workload.get("sm_scale"),
                dtype=getattr(torch, dtype_name),
            )
    return cases


def build_inputs(kwargs):
    from tileops.workloads.attention import MLADenseBwdWorkload

    workload = MLADenseBwdWorkload(
        kwargs["batch"], kwargs["seq_q"], kwargs["seq_kv"], kwargs["heads"],
        kwargs["dim_qk"], kwargs["dim_v"], kwargs["dtype"], kwargs["is_causal"],
        kwargs["sm_scale"],
    )
    do, q, k, v, o, lse = workload.gen_inputs()
    return dict(do=do, q=q, k=k, v=v, o=o, lse=lse)


def case_kwargs(workload, dtype_name):
    """``kwargs`` of one case id, without touching the manifest module."""
    return dict(
        batch=int(workload["B"]), seq_q=int(workload["S_q"]),
        seq_kv=int(workload["S_kv"]), heads=int(workload["H"]),
        dim_qk=int(workload["D_qk"]), dim_v=int(workload["D_v"]),
        is_causal=workload.get("is_causal", False),
        sm_scale=workload.get("sm_scale"),
        dtype=getattr(torch, dtype_name),
    )


def run_module_call(impl, kwargs, tensors):
    """The wrapper's ``forward`` + ``_run``, verbatim, on one impl.

    Inputs are the packed ``(T, H, D)`` tensors the op layer takes; the main
    pass receives the zero-copy 4-D batch views ``(B, S, H, D)``.
    """
    dtype_name = str(kwargs["dtype"]).split(".")[-1]
    batch, seq_q, seq_kv = kwargs["batch"], kwargs["seq_q"], kwargs["seq_kv"]
    heads, dim_qk, dim_v = kwargs["heads"], kwargs["dim_qk"], kwargs["dim_v"]
    scale = dim_qk ** -0.5 if kwargs["sm_scale"] is None else kwargs["sm_scale"]
    # ``forward`` receives BSHD 4-D tensors and flattens the token axis; the
    # three launches then take these packed 3-D views.
    q, k, v, o, do = (
        tensors[name].view(-1, heads, tensors[name].shape[-1])
        for name in ("q", "k", "v", "o", "do")
    )
    lse = tensors["lse"]

    prep = impl.delta(batch, heads, dim_v, seq_q, dtype_name)
    land = impl.land(heads, dim_qk, scale, dtype_name)
    main = impl.bwd(batch, heads, dim_qk, dim_v, seq_q, seq_kv,
                    kwargs["is_causal"], scale, dtype_name)

    # --- wrapper ``forward``: offsets built per call (Range op) ---------------
    device = q.device
    cu_q = torch.arange(0, (batch + 1) * seq_q, seq_q, dtype=torch.int32, device=device)
    cu_kv = torch.arange(0, (batch + 1) * seq_kv, seq_kv, dtype=torch.int32, device=device)
    # --- wrapper ``_run`` ----------------------------------------------------
    delta = prep(o, do)
    dq_acc = torch.zeros(q.shape, dtype=torch.float32, device=device)  # ZerosLike
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    main(
        q.view(batch, seq_q, heads, dim_qk),
        k.view(batch, seq_kv, heads, dim_qk),
        v.view(batch, seq_kv, heads, dim_v),
        do.view(batch, seq_q, heads, dim_v),
        lse,
        delta,
        cu_q,
        cu_kv,
        dq_acc.view(batch, seq_q, heads, dim_qk),
        dk.view(batch, seq_kv, heads, dim_qk),
        dv.view(batch, seq_kv, heads, dim_v),
    )
    dq = land(dq_acc)
    return dq.view(q.shape), dk, dv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", default="baseline")
    parser.add_argument("--case", default="")
    parser.add_argument("--save-inputs", default="")
    parser.add_argument("--inputs", default="")
    parser.add_argument("--custom", default="",
                        help='JSON geometry override, e.g. \'{"batch":1,"seq_q":8192,'
                             '"seq_kv":8192,"heads":128}\' (exploration only)')
    parser.add_argument("--warm-up", type=int, default=0)
    parser.add_argument("--launch-count", type=int, default=1)
    args = parser.parse_args()

    if args.save_inputs:
        if args.custom:
            raw = json.loads(args.custom)
            cases = {args.case: dict(
                batch=int(raw["batch"]), seq_q=int(raw["seq_q"]),
                seq_kv=int(raw.get("seq_kv", raw["seq_q"])), heads=int(raw["heads"]),
                dim_qk=int(raw.get("dim_qk", 192)), dim_v=int(raw.get("dim_v", 128)),
                is_causal=bool(raw.get("is_causal", True)), sm_scale=raw.get("sm_scale"),
                dtype=getattr(torch, raw.get("dtype", "bfloat16")),
            )}
        else:
            cases = benchmark_cases()
        if args.case not in cases:
            raise SystemExit(f"--case {args.case}: known {sorted(cases)}")
        started = time.perf_counter()
        tensors = build_inputs(cases[args.case])
        payload = {key: tensors[key].detach().cpu() for key in INPUT_KEYS}
        payload["_meta"] = {
            "case": args.case,
            "kwargs": {k: (str(v) if k == "dtype" else v)
                       for k, v in cases[args.case].items()},
        }
        torch.save(payload, args.save_inputs)
        print(f"[inputs] saved {args.case} -> {args.save_inputs} "
              f"q={tuple(payload['q'].shape)} in {time.perf_counter() - started:.1f} s",
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
            tensors["lse"],
        )
    else:
        call = lambda: run_module_call(impl, kwargs, tensors)  # noqa: E731
    start = time.perf_counter()
    call()
    torch.npu.synchronize()
    print(f"[setup] first call {time.perf_counter() - start:.1f} s (compile)", flush=True)
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
