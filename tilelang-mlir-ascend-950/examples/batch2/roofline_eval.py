#!/usr/bin/env python3
"""Evaluate a TileOPs manifest roofline (flops/bytes) for one benchmark case.

The report's ``performance_tops`` comes from msprof's per-kernel counters, so it
cannot be used as the whole-operator FLOPs.  This helper evaluates the manifest
``roofline.vars`` + ``flops`` + ``bytes`` expressions against the case ``params``
and returns the device-agnostic work of one operator call.

The manifest formulas are written against the *workload* vocabulary (shape keys
such as ``key_shape`` or ``dY_shape`` and dimension names such as ``S_q``).  A
report only stores the ``params`` the benchmark built, which is sometimes a
flattened / nested subset of that vocabulary.  ``build_env`` therefore

  * flattens nested param dicts (``op_params`` / ``config``),
  * maps the report's dimension names onto the manifest's aliases,
  * synthesises the ``*_shape`` keys the benchmark did not record, from the
    scalar dimensions it did record (and documents each synthesis),
  * pre-fills the handful of vars whose manifest definition is prose rather
    than an expression (``written``, ``in_elem_bytes``, ``rows_read`` ...).

Every synthesised value is listed in ``assumptions`` in the returned dict so a
reader can see exactly what the FLOPs/Bytes number rests on.

Usage:
    python roofline_eval.py --manifest-dir <examples/TileOPs/tileops/manifest> \
                            --op MLADenseBwdOp --params '<json>'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _product(seq):
    out = 1
    for v in seq:
        out *= v
    return out


SAFE = {"__builtins__": {}, "min": min, "max": max, "sum": sum, "len": len,
        "abs": abs, "list": list, "zip": zip, "range": range,
        "product": _product}

DTYPE_BYTES = {
    "float16": 2, "bfloat16": 2, "float32": 4, "float8_e4m3fn": 1,
    "int32": 4, "int64": 8,
}


def load_entry(manifest_dir: Path, op: str):
    for path in sorted(manifest_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        if op in data:
            return data[op]
    raise KeyError(op)


def _put(env: dict, assumptions: list, key, value, why: str):
    """Set ``key`` unless the caller/params already supplied a usable value."""
    if key in env and env[key] is not None:
        return
    env[key] = value
    if why:
        assumptions.append(f"{key}={value!r} ({why})")


def build_env(params: dict):
    """Turn a report ``params`` blob into the manifest expression namespace."""
    env: dict = {}
    assumptions: list[str] = []

    # 1. params, with nested dicts flattened (MoE stores num_experts/block_size
    #    under ``op_params``; some benches use ``config``).
    for k, v in params.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                env.setdefault(kk, vv)
        else:
            env[k] = v

    # 2. every dtype string is also a name, so a formula may write
    #    ``cache_dtype == float8_e4m3fn``.
    for name in DTYPE_BYTES:
        env.setdefault(name, name)

    dt = params.get("dtype") or params.get("cache_dtype") or "bfloat16"
    env["elem_bytes"] = DTYPE_BYTES.get(str(dt), 2)

    # 3. attention / generic dimension aliases used by the MLA + MHA formulas.
    if isinstance(params.get("hidden_states_shape"), list):
        hs = params["hidden_states_shape"]
        if len(hs) > 0:
            _put(env, assumptions, "T", hs[0], "hidden_states_shape[0]")
        if len(hs) > 1:
            _put(env, assumptions, "H", hs[1], "hidden_states_shape[1]")
    if isinstance(params.get("w_down_shape"), list) and len(params["w_down_shape"]) > 1:
        _put(env, assumptions, "S", params["w_down_shape"][1], "w_down_shape[1]")
    for alias, pkey in (("B", "batch"), ("S_q", "seqlen_q"), ("S_kv", "seqlen_kv"),
                        ("D_qk", "dim_qk"), ("D_v", "dim_v"), ("M", "M"),
                        ("seq_len", "seq_len"), ("N", "N"), ("G", "G"),
                        ("NC", "NC"), ("chunk_len", "chunk_len")):
        if pkey in params:
            _put(env, assumptions, alias, params[pkey], f"params.{pkey}")
    if params.get("heads") is not None:
        env["H"] = params["heads"]

    # 4. family-specific shape synthesis ------------------------------------
    # 4a. Engram GateConv fwd/bwd reports only carry (M, seq_len, d); the
    #     manifest reads H_shape/dY_shape/conv_w_shape.
    if all(k in params for k in ("M", "seq_len", "d")):
        m, s, dd = params["M"], params["seq_len"], params["d"]
        for shape_key in ("H_shape", "dY_shape"):
            _put(env, assumptions, shape_key, [m, s, dd],
                 f"synthesised from M={m}, seq_len={s}, d={dd}")
        _put(env, assumptions, "conv_w_shape", [4, dd],
             "conv_kernel_size=4 taps over d (Engram GateConv signature)")
    # 4b. Engram Decode reports (batch, d_mem, d, conv_len, max_conv_len) but no
    #     tensor shapes; the manifest reads e_t/h_t/conv_state/new_conv_state.
    if all(k in params for k in ("batch", "d_mem", "d")):
        b, dmem, dd = params["batch"], params["d_mem"], params["d"]
        L = params.get("conv_len", params.get("L"))
        mcl = params.get("max_conv_len", L)
        _put(env, assumptions, "e_t_shape", [b, dmem], "engram_decode signature")
        _put(env, assumptions, "h_t_shape", [b, dd], "engram_decode signature")
        if L is not None:
            _put(env, assumptions, "conv_state_shape", [b, L, dd], "cached history")
        _put(env, assumptions, "new_conv_state_shape", [b, mcl, dd],
             "shifted history written back (max_conv_len long)")
    # 4c. MoE routing reports ``shape``/``tokens``, the manifest wants
    #     ``topk_ids_shape``; flops is 0 and only the index traffic matters.
    if "topk_ids_shape" not in env and params.get("shape") is not None:
        _put(env, assumptions, "topk_ids_shape", params["shape"],
             "report `shape` is the routed-id matrix [T, K]")
    if "topk_ids_shape" in env and "index_bytes" not in env:
        _put(env, assumptions, "index_bytes", 4, "int32 index traffic")
    # 4c-bis. Chain speculative sampling reports only (batch, num_draft, vocab);
    #     the manifest reads the three probability / id shapes.
    if all(k in params for k in ("batch", "num_draft", "vocab")):
        b, nd, vocab = params["batch"], params["num_draft"], params["vocab"]
        _put(env, assumptions, "draft_probs_shape", [b, nd, vocab],
             "draft_probs is [B, N, V] per the signature")
        _put(env, assumptions, "target_probs_shape", [b, nd + 1, vocab],
             "target_probs is [B, N + 1, V] per the signature")
        _put(env, assumptions, "draft_token_ids_shape", [b, nd],
             "draft_token_ids is [B, N] per the signature")
    # 4d. SSD chunk-coupling reports the scalar geometry only; the manifest
    #     reads C_shape/B_shape.
    if "C_shape" not in env and all(k in params for k in
                                    ("batch", "num_chunks", "chunk_len",
                                     "n_groups", "d_state")):
        seq_len = params["num_chunks"] * params["chunk_len"]
        shape = [params["batch"], seq_len, params["n_groups"], params["d_state"]]
        _put(env, assumptions, "C_shape", shape,
             "seq_len = num_chunks * chunk_len from the report geometry")
        _put(env, assumptions, "B_shape", shape, "same geometry as C")
        _put(env, assumptions, "x_shape", shape, "not used, kept for the scan formula")
    if "chunk_size" not in env and "chunk_len" in params:
        _put(env, assumptions, "chunk_size", params["chunk_len"],
             "the report calls the chunk length chunk_len")
    # 4e. Paged write: the manifest names ``written`` (slots >= 0) and the two
    #     element widths instead of a single elem_bytes.
    if "k_shape" in env:
        row = _product(env["k_shape"][1:])
        _put(env, assumptions, "row", row, "product(k_shape[1:])")
        _put(env, assumptions, "written", params.get("tokens", env["k_shape"][0]),
             "the bench writes every token (distinct non-negative slots)")
        _put(env, assumptions, "in_elem_bytes",
             DTYPE_BYTES.get(str(params.get("dtype", dt)), 2), "input element width")
        _put(env, assumptions, "cache_elem_bytes",
             DTYPE_BYTES.get(str(params.get("cache_dtype", dt)), 2),
             "cache element width")
    # 4f. RoPE reads only the cos/sin cache rows its positions name.
    if "query_shape" in env:
        _put(env, assumptions, "rows_read",
             min(_product(env.get("positions_shape") or [0]),
                 params.get("max_position_embeddings") or 1 << 60),
             "upper bound: positions cover a contiguous row range")

    # 5. raw ``*_shape`` params always win (setdefault only fills gaps).
    for k, v in params.items():
        if k.endswith("_shape"):
            env.setdefault(k, v)

    return env, assumptions


def eval_roofline(manifest_dir: Path, op: str, params: dict):
    entry = load_entry(manifest_dir, op)
    roof = entry.get("roofline") or {}
    env, assumptions = build_env(params)

    # variables in dependency order (several passes: vars may reference later vars)
    for _ in range(4):
        for name, expr in (roof.get("vars") or {}).items():
            try:
                env[name] = eval(expr, SAFE, env)  # noqa: S307
            except Exception:
                pass
    try:
        flops = eval(str(roof.get("flops", "0")), SAFE, env)  # noqa: S307
    except Exception:
        flops = None
    try:
        byte_expr = roof.get("bytes")
        size = eval(str(byte_expr), SAFE, env) if byte_expr else None
    except Exception:
        size = None
    return {"op": op, "flops": flops, "bytes": size,
            "assumptions": assumptions,
            "vars": {k: env.get(k) for k in
                     ("T", "H", "S", "B", "S_q", "S_kv", "D_qk", "D_v", "elem_bytes")}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--op", required=True)
    ap.add_argument("--params", required=True)
    args = ap.parse_args(argv)
    print(json.dumps(eval_roofline(Path(args.manifest_dir), args.op, json.loads(args.params)),
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    main()
