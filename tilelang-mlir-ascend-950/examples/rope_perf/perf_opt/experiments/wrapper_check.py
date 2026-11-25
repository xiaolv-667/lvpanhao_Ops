"""Drop-in check: run the **op layer** on the tuned factory, exactly as the
wrapper's ``perf_opt`` import block will after the conductor flips it.

The wrapper module re-exports ``_rope_positions_kernel`` from
``rope_kernel/__init__``; this script swaps that module global for the tuned
factory (what ``from .rope_kernel.perf_opt._rope_positions_kernel import
_rope_positions_kernel`` will do at Stage 5) and then runs

* the seven benchmark workloads (``tileops.manifest`` -> ``RoPEFwdWorkload``),
  compared with the op's own torch reference, and
* the op-layer fixture shapes (smoke / full / boundary dtypes).

Usage (cwd = examples/TileOPs so that ``tileops`` resolves):

    python <this file> --impl ../rope_perf/perf_opt/rope.py
"""

import argparse
import importlib.util
import os
import sys

import torch
import torch_npu  # noqa: F401

from tileops.manifest import load_workloads
from tileops.ops.rope.rope import RoPEFwdOp
from tileops.workloads.rope import RoPEFwdWorkload, rope_reference, rope_tables
import tileops.kernels.rope.rope.rope as wrapper


def load_factory(path):
    name = "rope_perf_opt_factory"
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod._rope_positions_kernel


def check(op, wl, params):
    inputs = wl.gen_inputs()
    positions, query, key = inputs
    ref_q, ref_k = rope_reference(
        params, *rope_tables(params, query), positions, query, key, None
    )
    got_q, got_k = op(positions, query, key)
    tol = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 1.6e-2}[query.dtype]
    torch.testing.assert_close(got_q, ref_q, atol=tol, rtol=tol, equal_nan=True)
    if key is not None:
        torch.testing.assert_close(got_k, ref_k, atol=tol, rtol=tol, equal_nan=True)
    diff = float((got_q.float() - ref_q.float()).abs().max())
    if key is not None:
        diff = max(diff, float((got_k.float() - ref_k.float()).abs().max()))
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    args = ap.parse_args()
    factory = load_factory(args.impl)
    wrapper._rope_positions_kernel = factory

    cases = []
    for w in load_workloads("RoPEFwdOp"):
        key_shape = w.get("key_shape")
        pos_dt = {"int32": torch.int32, "int64": torch.int64}[w.get("position_dtype", "int64")]
        for dtype_str in w["dtypes"]:
            cases.append(
                dict(
                    label=f"{w.get('label', 'rope')}-{dtype_str}",
                    positions_shape=tuple(w["positions_shape"]),
                    query_shape=tuple(w["query_shape"]),
                    key_shape=None if key_shape is None else tuple(key_shape),
                    head_size=w["head_size"],
                    rotary_dim=w["rotary_dim"],
                    max_position_embeddings=w["max_position_embeddings"],
                    base=w["base"],
                    is_neox_style=w["is_neox_style"],
                    dtype=getattr(torch, dtype_str),
                    position_dtype=pos_dt,
                )
            )
    # op-layer fixture shapes (smoke / full / boundary dtype coverage)
    cases += [
        dict(label="fixture-smoke-fp32", positions_shape=(64,), query_shape=(64, 256),
             key_shape=(64, 128), head_size=64, rotary_dim=64,
             max_position_embeddings=2048, base=10000.0, is_neox_style=True,
             dtype=torch.float32, position_dtype=torch.int64),
        dict(label="fixture-full-gptj", positions_shape=(2048,), query_shape=(2048, 4096),
             key_shape=(2048, 4096), head_size=256, rotary_dim=64,
             max_position_embeddings=2048, base=10000.0, is_neox_style=False,
             dtype=torch.float16, position_dtype=torch.int64),
        dict(label="fixture-tiny", positions_shape=(3,), query_shape=(3, 130),
             key_shape=(3, 65), head_size=65, rotary_dim=32,
             max_position_embeddings=7, base=10000.0, is_neox_style=True,
             dtype=torch.float16, position_dtype=torch.int32),
    ]
    ok = 0
    for c in cases:
        wl = RoPEFwdWorkload(
            c["positions_shape"], c["query_shape"], c["key_shape"], c["head_size"],
            c["rotary_dim"], c["max_position_embeddings"], c["base"],
            c["is_neox_style"], c["dtype"], c["position_dtype"],
        )
        params = wl.params()
        op = RoPEFwdOp(
            head_size=c["head_size"], rotary_dim=c["rotary_dim"],
            max_position_embeddings=c["max_position_embeddings"], base=c["base"],
            is_neox_style=c["is_neox_style"], dtype=c["dtype"],
        )
        try:
            diff = check(op, wl, params)
            print(f"[PASS] {c['label']:<34} max_diff={diff:.3g}")
            ok += 1
        except AssertionError as exc:
            print(f"[FAIL] {c['label']:<34} {str(exc)[:200]}")
    print(f"{ok}/{len(cases)} op-layer cases passed")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
