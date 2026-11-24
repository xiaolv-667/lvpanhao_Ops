#!/usr/bin/env python3
"""Correctness at the *dominant* workload shapes of the migrated TileOPs ops.

The in-tree correctness suite runs small shapes (64 tokens, H=4/8, ...); the
benchmark matrix in ``examples/batch2/mfu_table.json`` runs the production
shapes but only times them.  This script closes that gap: it runs each
operator's dominant workload (the ``examples/batch2/mfu_operators.csv`` row)
once and compares the result against a reference or against itself under an
exact decomposition (slice self-consistency).

It creates no kernel/operator code and mutates nothing in-tree -- only
``examples/batch2/workload_correctness.jsonl`` (one JSON object per case, so a
later crash cannot lose earlier results) and, with ``--merge``,
``examples/batch2/workload_correctness.json``.

Usage (from ``examples/TileOPs``, interpreter ``/workspace/venv-ascend950/bin/python``):

    python ../batch2/verify_workload_shapes.py --list
    python ../batch2/verify_workload_shapes.py --only engram_decode
    python ../batch2/verify_workload_shapes.py --merge

Every case prints ``RESULT {json}`` on stdout.  Each case is independent: run
them one process at a time on a single card, cheapest first.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TILEOPS_ROOT = _HERE.parent / "TileOPs"
if str(_TILEOPS_ROOT) not in sys.path:
    # The dev root (repo ``tilelang/``) must not shadow the installed package,
    # so add the TileOPs checkout explicitly instead of relying on cwd.
    sys.path.insert(0, str(_TILEOPS_ROOT))

import torch  # noqa: E402

from tileops.device import get_device_backend  # noqa: E402

JSONL = _HERE / "workload_correctness.jsonl"
JSON_OUT = _HERE / "workload_correctness.json"
DEFAULT_SEED = 20261124

_EXPLICIT_MANT = {
    torch.bfloat16: 7,
    torch.float16: 10,
    torch.float32: 23,
    torch.float64: 52,
}


# ---------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------


def _f32(t: torch.Tensor) -> torch.Tensor:
    """float32 view of *t* on **the tensor's own device**.

    The container's cgroup caps host memory at 32 GiB, so multi-GB tensors are
    never copied to the host: reductions run on the device and only scalars
    cross over.  bf16/fp16 -> fp32 is exact, so this changes no comparison.
    """
    return t.detach().float()


def _where(cond: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.where(cond, x, torch.zeros_like(x))


def pair_stats(got: torch.Tensor, ref: torch.Tensor) -> dict:
    """max |got - ref| and max relative error where ref is not negligible."""
    a, b = _f32(got), _f32(ref)
    diff = (a - b).abs()
    finite = torch.isfinite(a) & torch.isfinite(b)
    max_abs = float(_where(finite, diff).max()) if finite.any() else float("nan")
    nan_mismatch = int((torch.isnan(a) ^ torch.isnan(b)).sum())
    both_nan = int((torch.isnan(a) & torch.isnan(b)).sum())
    denom = b.abs()
    scale = float(_where(finite, denom).max()) if finite.any() else 0.0
    mask = finite & (denom > 1e-6 * max(scale, 1e-30))
    max_rel = (
        float((diff / denom.clamp_min(1e-30)).masked_fill(~mask, 0.0).max())
        if mask.any()
        else 0.0
    )
    return {
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "nan_mismatch": nan_mismatch,
        "both_nan": both_nan,
    }


def allclose_stats(
    got: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float
) -> dict:
    """Element count outside ``atol + rtol * |ref|`` (NaN matches NaN)."""
    a, b = _f32(got), _f32(ref)
    tol = atol + rtol * b.abs()
    bad = (a - b).abs() > tol
    bad = bad | (torch.isnan(a) ^ torch.isnan(b))
    n_bad = int(bad.sum())
    numel = int(a.numel())
    return {
        "ok": n_bad == 0,
        "n_bad": n_bad,
        "numel": numel,
        "bad_frac": n_bad / max(numel, 1),
    }


def max_ulp_err(got: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype) -> float:
    """Largest error in units of the output dtype's last place at |ref|."""
    explicit = _EXPLICIT_MANT.get(dtype)
    if explicit is None:
        return float("nan")
    a, b = _f32(got), _f32(ref)
    diff = (a - b).abs()
    mag = b.abs()
    ok = torch.isfinite(a) & torch.isfinite(b) & (mag > 0)
    if not ok.any():
        return 0.0
    expo = torch.floor(torch.log2(mag.clamp_min(2.0**-126)))
    ulp = torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=mag.device), expo - 1 - explicit
    )
    ratio = torch.where(ok, diff / ulp, torch.zeros_like(diff))
    return float(ratio.max())


def exact_stats(got: torch.Tensor, ref: torch.Tensor) -> dict:
    """Byte-exact equality (NaN matches NaN)."""
    a, b = got.detach(), ref.detach()
    if a.dtype is torch.float8_e4m3fn:
        # aclnn has no float8 comparison on this stack; the pools are small.
        a, b = a.cpu().view(torch.uint8), b.cpu().view(torch.uint8)
    same = a == b
    if same.dtype.is_floating_point:
        same = same | (torch.isnan(a) & torch.isnan(b))
    n_bad = int((~same).sum())
    return {"ok": n_bad == 0, "n_bad": n_bad, "numel": int(same.numel())}


def zero_frac(t: torch.Tensor) -> float:
    return float((t.detach() == 0).float().mean())


def nonfinite_frac(t: torch.Tensor) -> float:
    return float((~torch.isfinite(t.detach().float())).float().mean())


def k_ulp_global(got: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype) -> float:
    """DESIGN-style ULP diagnostic: ``max|got-ref| / ulp(max|ref|)``.

    DESIGN §8.3 (L0-12/L0-21) defines ``k_ulp`` with the ulp of the tensor's
    *largest magnitude*, not per element, so it reads as "how many roundings of
    the biggest output value is the worst error".
    """
    explicit = _EXPLICIT_MANT.get(dtype)
    if explicit is None:
        return float("nan")
    a, b = _f32(got), _f32(ref)
    scale = float(b.abs().max())
    if scale == 0:
        return 0.0
    expo = math.floor(math.log2(scale))
    return float((a - b).abs().max()) / (2.0 ** (expo - explicit))


def describe_outputs(names, got, ref, dtype, atol, rtol):
    """Per-output stats plus the aggregate the case's criterion uses."""
    per, worst_abs, worst_rel, worst_ulp = {}, 0.0, 0.0, 0.0
    ok, n_bad, numel = True, 0, 0
    for name, g, r in zip(names, got, ref, strict=True):
        st = pair_stats(g, r)
        ac = allclose_stats(g, r, atol, rtol)
        ulp = max_ulp_err(g, r, dtype)
        per[name] = {
            **st,
            "max_ulp": ulp,
            "k_ulp_global": k_ulp_global(g, r, dtype),
            "n_bad": ac["n_bad"],
            "bad_frac": ac["bad_frac"],
            "zero_frac_got": zero_frac(g),
        }
        worst_abs = max(worst_abs, st["max_abs_err"])
        worst_rel = max(worst_rel, st["max_rel_err"])
        worst_ulp = max(worst_ulp, ulp)
        ok = ok and ac["ok"]
        n_bad += ac["n_bad"]
        numel += ac["numel"]
    return {
        "per_output": per,
        "max_abs_err": worst_abs,
        "max_rel_err": worst_rel,
        "max_ulp": worst_ulp,
        "n_bad": n_bad,
        "numel": numel,
        "bad_frac": n_bad / max(numel, 1),
        "ok": ok,
    }


# ---------------------------------------------------------------------------
# case plumbing
# ---------------------------------------------------------------------------

CASES = {}
# case id -> (operator, dominant-case label from mfu_table.json / manifest)
CASE_META = {
    "engram_gate_conv_fwd": ("EngramGateConvFwdOp", "train-s96"),
    "engram_gate_conv_bwd": ("EngramGateConvBwdOp", "train-s96"),
    "engram_decode": ("EngramDecodeFwdOp", "serving-b4"),
    "moe_permute_align_ds_v3": ("MoEPermuteAlignFwdOp", "ds-v3-t4096"),
    "moe_permute_align_kimi_k2": ("MoEPermuteAlignFwdOp", "kimi-k2-t4096"),
    "paged_kv_cache_write": (
        "PagedKVCacheWriteFwdOp",
        "qwen3-235b-fp8-bfloat16-float8_e4m3fn",
    ),
    "rope_llama3_b2_2k": ("RoPEFwdOp", "llama3-8b-b2-2k"),
    "ssd_chunk_coupling": ("SSDChunkCouplingFwdOp", "mamba2-grouped-prefill"),
    "chain_speculative_sampling": (
        "ChainSpeculativeSamplingFwdOp",
        "ds-v3-mtp-n3-float32",
    ),
    "shared_expert_mlp": ("SharedExpertMLPFwdOp", "512x7168x18432"),
    "mla_dense_bwd": ("MLADenseBwdOp", "ds-v3-b2-8k"),
    "mla_varlen_bwd": ("MLAVarlenBwdOp", "ds-v3-mixed"),
    "mla_varlen_bwd_8x4k": ("MLAVarlenBwdOp", "ds-v3-8x4k"),
}


def case(name):
    def register(fn):
        CASES[name] = fn
        return fn

    return register


def _seed(seed=DEFAULT_SEED):
    torch.manual_seed(seed)
    dev = get_device_backend()
    dev.manual_seed_all(seed)
    return seed


def _mem_reset():
    with contextlib.suppress(Exception):
        torch.npu.reset_peak_memory_stats()


def _mem_peak_mb():
    try:
        return torch.npu.max_memory_allocated() / 1e6
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. Engram GateConv fwd / bwd / decode
# ---------------------------------------------------------------------------


@case("engram_gate_conv_fwd")
def engram_gate_conv_fwd():
    """train-s96: M=3, seq=96, d=768, bf16 (manifest ``train-s96``)."""
    from tileops.ops.sequence_modeling.engram_gate_conv import EngramGateConvFwdOp
    from tileops.workloads.sequence_modeling import EngramGateConvFwdWorkload

    M, seq_len, d, dtype = 3, 96, 768, torch.bfloat16
    seed = _seed()
    wl = EngramGateConvFwdWorkload(M, seq_len, d, dtype)
    op = EngramGateConvFwdOp()
    inputs = wl.gen_inputs()
    ref = wl.ref_program(*inputs)
    with torch.no_grad():
        got = op(*inputs)
    res = describe_outputs(
        ["Y", "v_hat", "alpha", "rrms_h", "rrms_k", "rrms_v"],
        got,
        ref,
        dtype,
        atol=0.2,
        rtol=0.1,
    )
    # alpha / rrms_* are fp32 outputs the suite compares with the same band; a
    # strict fp32 diagnostic tells us whether that band hides real drift.
    strict = pair_stats(got[2], ref[2])["max_abs_err"]
    return {
        "method": "reference (PyTorch fp32 forward) vs op",
        "criterion": {"kind": "allclose", "atol": 0.2, "rtol": 0.1},
        "dtype": "bfloat16",
        "shapes": {"H": [M, seq_len, d], "conv_w": [4, d]},
        "status": "pass" if res["ok"] else "fail",
        "max_abs_err": res["max_abs_err"],
        "extra": {**res, "strict_fp32_alpha_max_abs_err": strict},
        "seed": seed,
    }


@case("engram_gate_conv_bwd")
def engram_gate_conv_bwd():
    """train-s96: M=3, seq=96, d=768, bf16; six gradients compared."""
    from tileops.ops.sequence_modeling.engram_gate_conv_bwd import EngramGateConvBwdOp
    from tileops.workloads.sequence_modeling import EngramGateConvBwdWorkload

    M, seq_len, d, dtype = 3, 96, 768, torch.bfloat16
    seed = _seed()
    wl = EngramGateConvBwdWorkload(M, seq_len, d, dtype)
    op = EngramGateConvBwdOp()
    inputs = wl.gen_inputs()
    ref = wl.ref_program(*inputs)
    with torch.no_grad():
        got = op(*inputs)
    res = describe_outputs(
        ["dH", "dk", "dv", "dw_h", "dw_v", "dconv_w"],
        got,
        ref,
        dtype,
        atol=0.3,
        rtol=0.2,
    )
    return {
        "method": "reference (autograd through the fp32 forward) vs op",
        "criterion": {"kind": "allclose", "atol": 0.3, "rtol": 0.2},
        "dtype": "bfloat16",
        "shapes": {"H": [M, seq_len, d], "conv_w": [4, d]},
        "status": "pass" if res["ok"] else "fail",
        "max_abs_err": res["max_abs_err"],
        "extra": res,
        "seed": seed,
    }


@case("engram_decode")
def engram_decode():
    """serving-b4: B=4, d_mem=1024, d=512, cache 20, taps 4, dilation 5, fp16."""
    from tileops.ops.sequence_modeling.engram_decode import EngramDecodeFwdOp
    from tileops.workloads.sequence_modeling import EngramDecodeWorkload

    batch, d_mem, d, max_conv_len, k, dilation, dtype = (
        4,
        1024,
        512,
        20,
        4,
        5,
        torch.float16,
    )
    seed = _seed()
    wl = EngramDecodeWorkload(batch, d_mem, d, max_conv_len, k, dilation, dtype)
    op = EngramDecodeFwdOp(max_conv_len, k, dilation)
    inputs = wl.gen_inputs()
    ref = wl.ref_program(*inputs)
    with torch.no_grad():
        got = op(*inputs)
    res = describe_outputs(
        ["y_t", "new_conv_state"], got, ref, dtype, atol=0.05, rtol=0.05
    )
    # The whole updated cache must be right, not just y_t; report the untouched
    # (pure shift) prefix separately.
    prefix_ok = int(
        (
            _f32(got[1])[:, : max_conv_len - 1] == _f32(ref[1])[:, : max_conv_len - 1]
        ).sum()
    )
    return {
        "method": "reference (fp32 single-step decode, incl. cache update) vs op",
        "criterion": {"kind": "allclose", "atol": 0.05, "rtol": 0.05},
        "dtype": "float16",
        "shapes": {
            "e_t": [batch, d_mem],
            "h_t": [batch, d],
            "conv_state": [batch, max_conv_len, d],
        },
        "status": "pass" if res["ok"] else "fail",
        "max_abs_err": res["max_abs_err"],
        "extra": {**res, "shifted_prefix_entries_equal": prefix_ok},
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# 2. MoE permute/align (index op: route-equivalent exactness)
# ---------------------------------------------------------------------------


def _route_canonical(
    topk_ids, sorted_ids, expert_ids, total_padded, block_size, num_experts
):
    """Compare a routing result to the contract independent of intra-expert order."""
    numel = topk_ids.numel()
    flat = topk_ids.flatten().cpu().long()
    counts = torch.bincount(flat, minlength=num_experts)
    padded = (counts + block_size - 1) // block_size * block_size
    starts = torch.cumsum(padded, 0) - padded
    expect_total = int(padded.sum())

    problems = []
    if int(total_padded) != expect_total:
        problems.append(f"num_tokens_post_pad {int(total_padded)} != {expect_total}")

    ids = sorted_ids.cpu().long()
    if ids.numel() < expect_total:
        problems.append(f"sorted_token_ids too short: {ids.numel()} < {expect_total}")
        return {"ok": False, "problems": problems}
    # Every slot in an expert's padded range: the expert's routed flat indices
    # in any order, then the ``numel`` filler.
    for e in range(num_experts):
        lo, hi = int(starts[e]), int(starts[e] + padded[e])
        got_sorted = torch.sort(ids[lo:hi]).values
        routed = torch.nonzero(flat == e).flatten()
        want = torch.cat(
            [routed, torch.full((int(padded[e] - counts[e]),), numel, dtype=torch.long)]
        )
        if not torch.equal(got_sorted, torch.sort(want).values):
            n_diff = int((got_sorted != torch.sort(want).values).sum())
            problems.append(f"expert {e}: {n_diff} slot(s) differ")
            if len(problems) > 8:
                break
    # Filler past the padded length.
    tail = ids[expect_total:]
    if tail.numel() and int((tail != numel).sum()):
        problems.append(f"{int((tail != numel).sum())} filler entries != numel")
    # Block owners.
    owners = expert_ids.cpu().long()
    expected_owners = []
    for b in range(expect_total // block_size):
        block_start = b * block_size
        e = int(
            torch.searchsorted(starts + padded, torch.tensor(block_start), right=True)
        )
        expected_owners.append(e)
    expected_owners = torch.tensor(expected_owners, dtype=torch.long)
    if owners.numel() < expected_owners.numel() or not torch.equal(
        owners[: expected_owners.numel()], expected_owners
    ):
        problems.append("expert_ids (block owners) differ")
    return {"ok": not problems, "problems": problems}


@case("moe_permute_align_ds_v3")
def moe_permute_align_ds_v3():
    """ds-v3-t4096: topk_ids [4096, 8], 256 experts, block 64, int32."""
    return _moe_permute_align(4096, 8, 256, 64, "ds-v3-t4096")


@case("moe_permute_align_kimi_k2")
def moe_permute_align_kimi_k2():
    """kimi-k2-t4096: topk_ids [4096, 8], 384 experts, block 64, int32."""
    return _moe_permute_align(4096, 8, 384, 64, "kimi-k2-t4096")


def _moe_permute_align(tokens, top_k, num_experts, block_size, label):
    from tileops.ops.moe.permute_align import MoEPermuteAlignFwdOp
    from tileops.workloads.moe import MoEPermuteAlignWorkload

    seed = _seed()
    wl = MoEPermuteAlignWorkload(tokens, top_k, num_experts, block_size)
    op = MoEPermuteAlignFwdOp(num_experts=num_experts, block_size=block_size)
    (topk_ids,) = wl.gen_inputs()
    ref_ids, ref_experts, ref_count = wl.ref_program(topk_ids)
    got_ids, got_experts, got_count = op(topk_ids)
    cmp = _route_canonical(
        topk_ids, got_ids, got_experts, int(got_count), block_size, num_experts
    )
    # The reference itself must pass the same canonical check (self-check of
    # the comparison harness).
    ref_cmp = _route_canonical(
        topk_ids, ref_ids, ref_experts, int(ref_count), block_size, num_experts
    )
    return {
        "method": "reference (python routing) vs op; route-equivalent exact compare",
        "criterion": {"kind": "exact_structural", "order_within_expert": "unspecified"},
        "dtype": "int32",
        "shapes": {
            "topk_ids": [tokens, top_k],
            "num_experts": num_experts,
            "block_size": block_size,
        },
        "status": "pass" if (cmp["ok"] and ref_cmp["ok"]) else "fail",
        "max_abs_err": 0.0 if cmp["ok"] and ref_cmp["ok"] else None,
        "extra": {
            "op_vs_contract": cmp,
            "reference_vs_contract": ref_cmp,
            "num_tokens_post_pad": int(got_count),
            "num_tokens_post_pad_ref": int(ref_count),
            "sorted_len": int(got_ids.numel()),
        },
        "seed": seed,
        "label": label,
    }


# ---------------------------------------------------------------------------
# 3. Paged KV cache write (index op: byte-exact)
# ---------------------------------------------------------------------------


@case("paged_kv_cache_write")
def paged_kv_cache_write():
    """qwen3-235b-fp8: k [4096,4,128] bf16 -> cache [4096,16,4,128] fp8_e4m3."""
    from tileops.ops.attention.paged_cache_write import PagedKVCacheWriteFwdOp
    from tileops.workloads.attention import PagedKVCacheWriteWorkload

    tokens, heads_kv, dim, page_size, num_pages = 4096, 4, 128, 16, 4096
    dtype, cache_dtype = torch.bfloat16, torch.float8_e4m3fn
    seed = _seed()
    wl = PagedKVCacheWriteWorkload(
        tokens, heads_kv, dim, page_size, num_pages, dtype, cache_dtype
    )
    op = PagedKVCacheWriteFwdOp()
    inputs = wl.gen_inputs()
    before_k = inputs[2].clone()
    ref_k, ref_v = wl.ref_program(*inputs)
    with torch.no_grad():
        ret = op(*inputs)
    got_k, got_v = inputs[2], inputs[3]
    # Rows no slot names must be untouched; the workload's reference clones, so
    # compare the whole pools exactly.
    st_k = exact_stats(got_k, ref_k)
    st_v = exact_stats(got_v, ref_v)
    # Diagnostic: how many rows actually changed (a no-op kernel would pass an
    # "untouched rows" check, so count written rows too).
    rows = num_pages * page_size
    changed = int(
        (
            got_k.view(torch.uint8).view(rows, -1)
            != before_k.view(torch.uint8).view(rows, -1)
        )
        .any(1)
        .sum()
    )
    ok = st_k["ok"] and st_v["ok"]
    return {
        "method": "reference (host index-scatter with saturating fp8 cast) vs in-place op",
        "criterion": {"kind": "exact", "atol": 0, "rtol": 0},
        "dtype": f"{dtype}->{cache_dtype}",
        "shapes": {
            "k": [tokens, heads_kv, dim],
            "k_cache": [num_pages, page_size, heads_kv, dim],
        },
        "status": "pass" if ok else "fail",
        "max_abs_err": 0.0 if ok else None,
        "extra": {
            "k_cache": st_k,
            "v_cache": st_v,
            "mutated_rows": changed,
            "slot_min": int(inputs[4].min()),
            "slot_max": int(inputs[4].max()),
            "k_scale": float(inputs[5]),
            "v_scale": float(inputs[6]),
            "returns_none": ret is None,
        },
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# 4. RoPE (llama3-8b-b2-2k, fp32)
# ---------------------------------------------------------------------------


@case("rope_llama3_b2_2k")
def rope_llama3_b2_2k():
    """positions [2,2048], q [2,2048,4096], k [2,2048,1024], fp32, NeoX."""
    from tileops.ops.rope.rope import RoPEFwdOp
    from tileops.workloads.rope import RoPEFwdWorkload

    seed = _seed()
    wl = RoPEFwdWorkload(
        positions_shape=(2, 2048),
        query_shape=(2, 2048, 4096),
        key_shape=(2, 2048, 1024),
        head_size=128,
        rotary_dim=128,
        max_position_embeddings=8192,
        base=500000.0,
        is_neox_style=True,
        dtype=torch.float32,
    )
    op = RoPEFwdOp(**wl.params())
    positions, query, key = wl.gen_inputs()
    ref_q, ref_k = wl.ref_program(positions, query, key)
    with torch.no_grad():
        got_q, got_k = op(positions, query, key)
    res = describe_outputs(
        ["query", "key"],
        [got_q, got_k],
        [ref_q, ref_k],
        torch.float32,
        atol=1e-5,
        rtol=1e-5,
    )
    # Un-rotated tail columns of each head must be copied through exactly (this
    # workload rotates every column, so the tail is empty and the check is a
    # no-op; it is kept for the general case).
    heads_q = query.reshape(*query.shape[:-1], -1, 128)
    tail_ref = heads_q[..., 128:]
    if tail_ref.numel():
        tail_st = exact_stats(
            got_q.reshape(*got_q.shape[:-1], -1, 128)[..., 128:], tail_ref
        )
    else:
        tail_st = {"skipped": "rotary_dim == head_size, no un-rotated tail"}
    return {
        "method": "reference (vLLM NeoX formula, fp32) vs op",
        "criterion": {"kind": "allclose", "atol": 1e-5, "rtol": 1e-5},
        "dtype": "float32",
        "shapes": {
            "positions": [2, 2048],
            "query": [2, 2048, 4096],
            "key": [2, 2048, 1024],
        },
        "status": "pass" if res["ok"] else "fail",
        "max_abs_err": res["max_abs_err"],
        "extra": {
            **res,
            "non_rotated_tail_exact": tail_st,
            "tail_numel": int(tail_ref.numel()),
        },
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# 5. SSD chunk coupling (mamba2-grouped-prefill)
# ---------------------------------------------------------------------------


@case("ssd_chunk_coupling")
def ssd_chunk_coupling():
    """C/B [3,3072,8,128], chunk 256, fp16 (manifest ``mamba2-grouped-prefill``)."""
    from tileops.ops.mamba.ssd_chunk_coupling import SSDChunkCouplingFwdOp
    from tileops.testing.mamba2_reference import ssd_chunk_coupling_fwd_ref
    from tileops.workloads.mamba import SSDChunkCouplingFwdWorkload

    batch, num_chunks, chunk_len, n_groups, d_state = 3, 12, 256, 8, 128
    dtype = torch.float16
    assert num_chunks * chunk_len == 3072, "manifest seq_len is 3072"
    seed = _seed()
    wl = SSDChunkCouplingFwdWorkload(
        batch, num_chunks, chunk_len, n_groups, d_state, dtype
    )
    op = SSDChunkCouplingFwdOp(chunk_len)
    C_mat, B_mat = wl.gen_inputs()
    ref = ssd_chunk_coupling_fwd_ref(C_mat, B_mat, num_chunks, chunk_len, dtype)
    with torch.no_grad():
        got = op(C_mat, B_mat)
    st = pair_stats(got, ref)
    ac = allclose_stats(got, ref, 1e-3, 1e-3)
    # Off-diagonal blocks must be exactly zero (causal mask).
    cf = _f32(got).reshape(batch, num_chunks, n_groups, chunk_len, chunk_len)
    upper = torch.triu(torch.ones(chunk_len, chunk_len, dtype=torch.bool), diagonal=1)
    offdiag_nonzero = int((cf[..., upper] != 0).sum())
    return {
        "method": "reference (fp32 einsum + tril mask) vs op",
        "criterion": {"kind": "allclose", "atol": 1e-3, "rtol": 1e-3},
        "dtype": "float16",
        "shapes": {
            "C": [batch, num_chunks * chunk_len, n_groups, d_state],
            "B": [batch, num_chunks * chunk_len, n_groups, d_state],
            "chunk_size": chunk_len,
        },
        "status": "pass" if ac["ok"] else "fail",
        "max_abs_err": st["max_abs_err"],
        "extra": {
            **st,
            **ac,
            "masked_upper_nonzero": offdiag_nonzero,
            "max_ulp": max_ulp_err(got, ref, dtype),
        },
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# 6. Chain speculative sampling (ds-v3-mtp-n3)
# ---------------------------------------------------------------------------


@case("chain_speculative_sampling")
def chain_speculative_sampling():
    """ds-v3-mtp-n3: B=64, N=3, V=129280, float32 (dominant case)."""
    from tileops.ops.sampling.chain_speculative_sampling import (
        ChainSpeculativeSamplingFwdOp,
    )
    from tileops.workloads.sampling import (
        ChainSpeculativeSamplingFwdWorkload,
        verify_chain_outputs,
    )

    batch, num_draft, vocab = 64, 3, 129280
    seed = _seed()
    wl = ChainSpeculativeSamplingFwdWorkload(batch, num_draft, vocab)
    op = ChainSpeculativeSamplingFwdOp()
    inputs = wl.gen_inputs()
    ref_tokens, ref_accepted = wl.ref_program(*inputs)
    with torch.no_grad():
        got_tokens, got_accepted = op(*inputs)
    acc_exact = exact_stats(got_accepted, ref_accepted)
    tok_exact = exact_stats(got_tokens, ref_tokens)
    problems = []
    try:
        verify_chain_outputs(got_tokens, got_accepted, inputs[0], inputs[1], inputs[2])
        structure_ok = True
    except AssertionError as e:  # noqa: BLE001 - reported, not raised
        structure_ok = False
        problems.append(str(e)[:400])
    # Output range and per-position non-negative-token sanity.
    tok = got_tokens.cpu()
    range_ok = bool(((tok >= -1) & (tok < vocab)).all())
    accepted_hist = torch.bincount(
        got_accepted.cpu().long(), minlength=num_draft + 1
    ).tolist()
    ref_hist = torch.bincount(
        ref_accepted.cpu().long(), minlength=num_draft + 1
    ).tolist()
    # Token-level: the op and the reference draw from the same uniform stream,
    # so the accepted prefix must agree; the post-prefix token need not.
    agree = (tok[:, :num_draft] == ref_tokens[:, :num_draft].cpu()) | (
        torch.arange(num_draft)[None] >= got_accepted.cpu()[:, None]
    )
    ok = acc_exact["ok"] and structure_ok and range_ok
    return {
        "method": "reference (Philox-keyed chain sampling) vs op: num_accepted exact + contract",
        "criterion": {
            "kind": "exact_accepted+structure",
            "note": "post-prefix token follows the same distribution but is not required to be the same index",
        },
        "dtype": "float32",
        "shapes": {
            "draft_probs": [batch, num_draft, vocab],
            "target_probs": [batch, num_draft + 1, vocab],
        },
        "status": "pass" if ok else "fail",
        "max_abs_err": 0.0 if acc_exact["ok"] else None,
        "extra": {
            "num_accepted_exact": acc_exact,
            "token_exact": tok_exact,
            "token_match_rate": 1.0 - tok_exact["n_bad"] / max(tok_exact["numel"], 1),
            "accepted_prefix_agree_frac": float(agree.float().mean()),
            "structure_ok": structure_ok,
            "range_ok": range_ok,
            "problems": problems,
            "accepted_hist_got": accepted_hist,
            "accepted_hist_ref": ref_hist,
        },
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# 7. Shared expert MLP (512 x 7168 x 18432, bf16)
# ---------------------------------------------------------------------------


@case("shared_expert_mlp")
def shared_expert_mlp():
    """512x7168x18432 bf16: the manifest kimi-k2-t512 row (dominant by MFU)."""
    from tileops.ops.moe.shared_expert_mlp import SharedExpertMLPFwdOp
    from tileops.workloads.moe import SharedExpertMLPWorkload

    tokens, hidden, ffn, dtype = 512, 7168, 18432, torch.bfloat16
    seed = _seed()
    wl = SharedExpertMLPWorkload(tokens, hidden, ffn, dtype)
    op = SharedExpertMLPFwdOp()
    inputs = wl.gen_inputs()
    ref = wl.ref_program(*inputs)
    with torch.no_grad():
        got = op(*inputs)
    st = pair_stats(got, ref)
    ac = allclose_stats(got, ref, 1e-2, 1e-2)
    # Normalised squared error, the policy's second half (moe_close bound 1e-4).
    # Accumulated on the host in float64, one 4 MiB chunk at a time (the device
    # has no float64 on this stack).
    g, r = got.detach().float().reshape(-1), ref.detach().float().reshape(-1)
    err = torch.zeros((), dtype=torch.float64)
    energy = torch.zeros((), dtype=torch.float64)
    for a, b in zip(g.split(1 << 20), r.split(1 << 20), strict=True):
        a, b = a.cpu().double(), b.cpu().double()
        err += ((a - b) ** 2).sum()
        energy += (a * a + b * b).sum()
    nsq = float(err / energy)
    nsq_ok = nsq <= 1e-4
    return {
        "method": "reference (fp32 gated MLP) vs op; moe_close policy",
        "criterion": {
            "kind": "moe_close",
            "atol": 1e-2,
            "rtol": 1e-2,
            "nsq_bound": 1e-4,
        },
        "dtype": "bfloat16",
        "shapes": {
            "hidden_states": [tokens, hidden],
            "w_gate_up": [2 * ffn, hidden],
            "w_down": [hidden, ffn],
        },
        "status": "pass" if (ac["ok"] and nsq_ok) else "fail",
        "max_abs_err": st["max_abs_err"],
        "extra": {
            **st,
            **ac,
            "nsq_err": nsq,
            "nsq_ok": nsq_ok,
            "max_ulp": max_ulp_err(got, ref, dtype),
        },
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# 8. MLA dense backward (ds-v3-b2-8k) -- the heavy one
# ---------------------------------------------------------------------------


def _mla_compare(names, got, ref, dtype, atol, rtol):
    return describe_outputs(names, got, ref, dtype, atol, rtol)


def _mla_delta_convention_probe(
    got_dq, ref_dq, q, k, v, do, o, scale, bounds_q, bounds_kv, n_rows=3
):
    """Recompute the worst ``dQ`` rows two ways and attribute the difference.

    ``got_dq`` / ``ref_dq`` are ``(T, H, D_qk)`` packed; request ``r`` owns rows
    ``bounds_q[r]:bounds_q[r+1]`` of ``q``/``do`` and ``bounds_kv[r]:bounds_kv[r+1]``
    of ``k``/``v`` (bottom-right causal, as the workload builds them).

    The repo reference forms ``dS = P * (dP - delta)`` with
    ``delta = sum_j P_j (do . v_j)`` in fp32, while the op documents and
    implements the FlashAttention convention ``delta = sum_d o * do`` on the
    *forward output* ``o`` (rounded to bf16).  With few visible keys ``P`` is
    near-uniform and ``dS`` is a difference of nearly equal large numbers, so
    the two conventions can differ well beyond the elementwise tolerance even
    though neither is a runaway error.  For the worst rows this recomputes
    ``dQ`` on the host with both deltas and reports which one the kernel
    followed.
    """
    import bisect

    gf, rf = _f32(got_dq), _f32(ref_dq)
    diff = (gf - rf).abs()
    tol = 5e-3 + 1.6e-2 * rf.abs()
    bad = (diff > tol).nonzero()
    if not bad.numel():
        return {"skipped": "no element outside the criterion"}
    order = diff[bad[:, 0], bad[:, 1], bad[:, 2]].argsort(descending=True)
    ordered = bad[order]
    # How many keys each outlier's query row can see: the cancellation that makes
    # the two delta conventions diverge is strongest when that count is small.
    hist = {}
    for t in ordered[:20000]:
        row = int(t[0])
        r = bisect.bisect_right(bounds_q, row) - 1
        local = row - bounds_q[r]
        kv_len = bounds_kv[r + 1] - bounds_kv[r]
        n_vis = min(local + 1, kv_len)
        hist[n_vis] = hist.get(n_vis, 0) + 1
    rows = []
    for t in ordered[:n_rows]:
        row, h, d = (int(x) for x in t)
        r = bisect.bisect_right(bounds_q, row) - 1
        local = row - bounds_q[r]
        kv_len = bounds_kv[r + 1] - bounds_kv[r]
        n = min(local + 1, kv_len)
        qh = q[row, h].to("cpu", torch.float32)
        kh = k[bounds_kv[r] : bounds_kv[r] + n, h].to("cpu", torch.float32)
        vh = v[bounds_kv[r] : bounds_kv[r] + n, h].to("cpu", torch.float32)
        doh = do[row, h].to("cpu", torch.float32)
        oc = o[row, h].to("cpu", torch.float32)
        p = torch.softmax((kh @ qh) * scale, -1)
        dp = vh @ doh
        deltas = {
            "reference_sum_P_times_dP": (p * dp).sum(),
            "op_convention_o_dot_do": (oc * doh).sum(),
        }
        out = {}
        for name, delta in deltas.items():
            val = float(((p * (dp - delta)) @ kh)[d] * scale)
            out[name] = {
                "computed": val,
                "err_vs_kernel": abs(val - float(gf[row, h, d])),
                "err_vs_reference": abs(val - float(rf[row, h, d])),
            }
        rows.append(
            {
                "index": [row, h, d],
                "request": r,
                "visible_keys": n,
                "kernel": float(gf[row, h, d]),
                "reference": float(rf[row, h, d]),
                "fp32_recompute": out,
            }
        )
    return {
        "n_bad": int(bad.shape[0]),
        "visible_keys_histogram": hist,
        "worst_rows": rows,
    }


def _worst_elements(got, ref, atol, rtol, names, k=6):
    """Top-k elements outside ``atol + rtol*|ref|`` per output, for the record."""
    out = {}
    for name, g, r in zip(names, got, ref, strict=True):
        gf, rf = _f32(g), _f32(r)
        diff = (gf - rf).abs()
        tol = atol + rtol * rf.abs()
        bad = diff > tol
        if not bool(bad.any()):
            out[name] = {"n_bad": 0}
            continue
        flat = torch.nonzero(bad.reshape(-1)).flatten()
        vals = diff.reshape(-1)[flat]
        order = vals.argsort(descending=True)[:k]
        rows = []
        for i in order:
            idx = flat[i]
            coords = []
            rem = int(idx)
            for size in reversed(gf.shape):
                coords.append(rem % size)
                rem //= size
            coords.reverse()
            tv = tuple(coords)
            gv, rv = float(gf[tv]), float(rf[tv])
            ulp = max_ulp_err(torch.tensor([gv]), torch.tensor([rv]), g.dtype)
            rows.append(
                {
                    "index": coords,
                    "got": gv,
                    "ref": rv,
                    "err": abs(gv - rv),
                    "tol": float(tol[tv]),
                    "ulp_at_ref": ulp,
                }
            )
        out[name] = {"n_bad": int(bad.sum()), "worst": rows}
    return out


def _head_liveness(t: torch.Tensor, head_axis: int) -> dict:
    """Per-head totals of |t|: a head slab the kernel never wrote reads exactly 0."""
    a = t.detach().float()
    heads = a.shape[head_axis]
    dims = tuple(d for d in range(a.ndim) if d != head_axis)
    totals = a.abs().sum(dim=dims)
    return {
        "heads": heads,
        "dead_heads": int((totals == 0).sum()),
        "nonfinite": int((~torch.isfinite(a)).sum()),
        "abs_max": float(a.abs().max()),
        "zero_frac": zero_frac(t),
        "per_head_abs_sum_min": float(totals.min()),
        "per_head_abs_sum_max": float(totals.max()),
    }


# ---------------------------------------------------------------------------
# Contract-aligned MLA backward reference (verification side).
#
# The operator's own contract (DESIGN §1 / §8.2, `_delta_kernel.golden_delta`)
# is: `P` is rebuilt from the *given* `lse` and the row statistic is
# `delta[t,h] = sum_d o[t,h,d]*do[t,h,d]` computed in fp32 from the forward
# output `o` **as the op receives it** (already rounded to the input dtype).
# `MLADenseBwdWorkload.ref_program` / `MLAVarlenBwdWorkload.ref_program` instead
# recompute `delta = sum_j P_j (do . v_j)` in fp32.  Both are mathematically
# equivalent, but they are not equivalent *numerically* when only a couple of
# keys are visible (dS is then a difference of two nearly equal large numbers).
#
# This reference feeds the backward the same `lse` and `delta` the op gets, so
# the comparison isolates the backward implementation given identical
# statistics.  `delta_source="probs"` reproduces the workload reference's
# convention, which is used as a self-check of this implementation.
# ---------------------------------------------------------------------------


def mla_delta_from_o(o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
    """fp32 ``delta[t, h] = sum_d o[t,h,d] * do[t,h,d]`` (the op's contract).

    Same definition as ``_delta_kernel.golden_delta``; the only freedom is the
    fp32 reduction order over ``D_v``, which is ulp-level.
    """
    return (o.float() * do.float()).sum(-1)


def _contract_slabs(q, k, lse_req, *, is_causal, scale):
    """Yield ``(heads, rows, probs)`` with ``probs = exp(score*scale - lse)``.

    ``lse_req`` is the caller's fp32 ``(H, q_len)`` log-sum-exp for this request,
    read as-is (the op does the same); a row with no visible key has
    ``lse = -inf`` and yields zero probabilities.
    """
    q_len, num_heads, _ = q.shape
    kv_len = k.shape[0]
    key = k.float()
    position = torch.arange(q_len, device=q.device) + (kv_len - q_len)
    key_index = torch.arange(kv_len, device=q.device)
    heads_per_slab, rows_per_slab = 8, 1024
    for h0 in range(0, num_heads, heads_per_slab):
        for r0 in range(0, q_len, rows_per_slab):
            heads = slice(h0, h0 + heads_per_slab)
            rows = slice(r0, r0 + rows_per_slab)
            scores = (
                torch.einsum("shd,nhd->hsn", q[rows, heads].float(), key[:, heads])
                * scale
            )
            lse_rows = lse_req[heads, r0 : r0 + rows_per_slab].unsqueeze(-1)
            if is_causal:
                hidden = key_index[None, :] > position[rows, None]
                scores = scores.masked_fill(hidden[None], float("-inf"))
            yield heads, rows, torch.exp(scores - lse_rows).nan_to_num(0.0)


def mla_bwd_reference_contract(
    do,
    q,
    k,
    v,
    o,
    lse,
    bounds_q,
    bounds_kv,
    *,
    is_causal=True,
    scale=None,
    delta_source="contract",
    quantize_pd=False,
):
    """fp32 ``(dq, dk, dv)`` from the same ``lse`` and ``delta`` the op consumes.

    ``do``/``q``/``k``/``v``/``o`` are ``(T, H, D)`` packed in token-major order
    and ``lse`` is ``(H, T)``; request ``b`` owns rows
    ``bounds_q[b]:bounds_q[b+1]`` (and ``bounds_kv``) as in the workload.
    ``delta_source="contract"`` uses ``sum_d o*do`` (the op's `_delta_kernel`);
    ``"probs"`` uses ``sum_j P_j (do . v_j)`` (the workload reference) and is
    only for validating this implementation.

    ``quantize_pd=True`` additionally rounds ``P`` and ``dS`` to the input dtype
    before the GEMMs, which is the kernel's documented semantics ("``P`` 量化到
    输入 dtype", DESIGN §8.1 golden).  It is kept as a separate *diagnostic*
    variant: the primary criterion uses the fp32-arithmetic reference, which is
    stricter on this axis.
    """
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    dq = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
    dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
    if delta_source == "contract":
        delta = mla_delta_from_o(o, do)
    elif delta_source != "probs":
        raise ValueError(f"unknown delta_source {delta_source!r}")
    for q0, q1, k0, k1 in zip(
        bounds_q, bounds_q[1:], bounds_kv, bounds_kv[1:], strict=False
    ):
        key, value = k[k0:k1].float(), v[k0:k1].float()
        for heads, rows, probs in _contract_slabs(
            q[q0:q1], k[k0:k1], lse[:, q0:q1], is_causal=is_causal, scale=scale
        ):
            grad_out = do[q0:q1][rows, heads].float()
            dprobs = torch.einsum("shd,nhd->hsn", grad_out, value[:, heads])
            if delta_source == "probs":
                d = (probs * dprobs).sum(-1, keepdim=True)
            else:
                d = delta[q0:q1][rows, heads].T.unsqueeze(-1)
            dscores = probs * (dprobs - d)
            if quantize_pd:
                # The kernel quantizes P and dS to the input dtype before the
                # tensor-core GEMMs (source semantics, DESIGN §8.1).
                probs = probs.to(q.dtype).float()
                dscores = dscores.to(q.dtype).float()
            dv[k0:k1, heads] += torch.einsum("hsn,shd->nhd", probs, grad_out)
            dk[k0:k1, heads] += torch.einsum(
                "hsn,shd->nhd", dscores, q[q0:q1][rows, heads].float()
            )
            dq[q0:q1][rows, heads] = torch.einsum(
                "hsn,nhd->shd", dscores, key[:, heads]
            )
    return dq * scale, dk * scale, dv


def _contract_reference_multi(wl_dtype, grads, like):
    return tuple(
        g.reshape(t.shape).to(wl_dtype) for g, t in zip(grads, like, strict=True)
    )


@case("mla_dense_bwd")
def mla_dense_bwd():
    """ds-v3-b2-8k: B=2, S=8192, H=128, D_qk=192, D_v=128, bf16, causal."""
    from tileops.ops.attention.mla_dense_bwd import MLADenseBwdOp
    from tileops.workloads.attention import MLADenseBwdWorkload

    batch, seq, heads, d_qk, d_v = 2, 8192, 128, 192, 128
    dtype = torch.bfloat16
    seed = _seed()
    wl = MLADenseBwdWorkload(batch, seq, seq, heads, d_qk, d_v, dtype)
    op = MLADenseBwdOp(is_causal=True)
    inputs = wl.gen_inputs()
    do, q, k, v, o, lse = inputs
    with torch.no_grad():
        got = op(*inputs)

    # --- primary criterion: contract-aligned reference -------------------------
    # Same `lse` and same `delta = rowsum(o*do)` (fp32, from the bf16 `o` the op
    # receives) as the kernel's `_delta_kernel`; only the backward arithmetic is
    # reimplemented independently.
    bounds = [i * seq for i in range(batch + 1)]
    contract = mla_bwd_reference_contract(
        do.reshape(batch * seq, heads, d_v),
        q.reshape(batch * seq, heads, d_qk),
        k.reshape(batch * seq, heads, d_qk),
        v.reshape(batch * seq, heads, d_v),
        o.reshape(batch * seq, heads, d_v),
        lse.permute(1, 0, 2).reshape(heads, batch * seq),
        bounds,
        bounds,
        is_causal=True,
        scale=d_qk**-0.5,
    )
    # Diagnostic variant: the same statistics, with the kernel's documented
    # P/dS -> input-dtype quantization (DESIGN §8.1 golden).  Not the criterion.
    golden = mla_bwd_reference_contract(
        do.reshape(batch * seq, heads, d_v),
        q.reshape(batch * seq, heads, d_qk),
        k.reshape(batch * seq, heads, d_qk),
        v.reshape(batch * seq, heads, d_v),
        o.reshape(batch * seq, heads, d_v),
        lse.permute(1, 0, 2).reshape(heads, batch * seq),
        bounds,
        bounds,
        is_causal=True,
        scale=d_qk**-0.5,
        quantize_pd=True,
    )
    ref_golden = _contract_reference_multi(dtype, golden, (q, k, v))
    ref_contract = _contract_reference_multi(dtype, contract, (q, k, v))
    res_c = _mla_compare(
        ["dq", "dk", "dv"], got, ref_contract, dtype, atol=5e-3, rtol=1.6e-2
    )
    res_g = _mla_compare(
        ["dq", "dk", "dv"], got, ref_golden, dtype, atol=5e-3, rtol=1.6e-2
    )
    worst_c = (
        _worst_elements(got, ref_contract, 5e-3, 1.6e-2, ["dq", "dk", "dv"])
        if not res_c["ok"]
        else {"skipped": "all elements inside the criterion"}
    )

    # --- historical record: the workload reference's own delta convention -------
    ref_legacy = wl.ref_program(*inputs)
    res_l = _mla_compare(
        ["dq", "dk", "dv"], got, ref_legacy, dtype, atol=5e-3, rtol=1.6e-2
    )
    live = {
        name: _head_liveness(t, head_axis=2)
        for name, t in zip(["dq", "dk", "dv"], got, strict=True)
    }

    # --- slice self-consistency -------------------------------------------------
    # Both the batch axis and the head axis are exactly decomposable: request b's
    # rows and head h's columns never interact.  A grid that silently truncates
    # leaves whole slabs at zero, which the full-vs-slice compare and the
    # liveness check both catch.
    slices = []
    for b in range(batch):
        sub = (
            do[b : b + 1],
            q[b : b + 1],
            k[b : b + 1],
            v[b : b + 1],
            o[b : b + 1],
            lse[b : b + 1],
        )
        with torch.no_grad():
            g = MLADenseBwdOp(is_causal=True)(*sub)
        for name, full_t, sub_t in zip(["dq", "dk", "dv"], got, g, strict=True):
            st = pair_stats(sub_t, full_t[b : b + 1])
            slices.append(
                {
                    "split": f"batch={b}",
                    "output": name,
                    "max_abs_err": st["max_abs_err"],
                    "exact": exact_stats(sub_t, full_t[b : b + 1])["ok"],
                }
            )
    for h0 in range(0, heads, 32):
        h1 = h0 + 32
        sub = (
            do[:, :, h0:h1],
            q[:, :, h0:h1],
            k[:, :, h0:h1],
            v[:, :, h0:h1],
            o[:, :, h0:h1],
            lse[:, h0:h1],
        )
        with torch.no_grad():
            g = MLADenseBwdOp(is_causal=True)(*sub)
        for name, full_t, sub_t in zip(["dq", "dk", "dv"], got, g, strict=True):
            st = pair_stats(sub_t, full_t[:, :, h0:h1])
            slices.append(
                {
                    "split": f"heads={h0}:{h1}",
                    "output": name,
                    "max_abs_err": st["max_abs_err"],
                    "exact": exact_stats(sub_t, full_t[:, :, h0:h1])["ok"],
                }
            )
    slice_max = max(s["max_abs_err"] for s in slices)
    slice_ok = slice_max <= 5e-3
    live_ok = all(v["dead_heads"] == 0 and v["nonfinite"] == 0 for v in live.values())
    per_batch_zero = [
        {name: zero_frac(t[b]) for name, t in zip(["dq", "dk", "dv"], got, strict=True)}
        for b in range(batch)
    ]
    attribution = (
        _mla_delta_convention_probe(
            got[0].reshape(batch * seq, heads, d_qk),
            ref_legacy[0].reshape(batch * seq, heads, d_qk),
            q.reshape(batch * seq, heads, d_qk),
            k.reshape(batch * seq, heads, d_qk),
            v.reshape(batch * seq, heads, d_v),
            do.reshape(batch * seq, heads, d_v),
            o.reshape(batch * seq, heads, d_v),
            d_qk**-0.5,
            bounds_q=bounds,
            bounds_kv=bounds,
        )
        if not res_l["ok"]
        else {"skipped": "all elements inside the criterion"}
    )
    ok = res_c["ok"] and slice_ok and live_ok
    return {
        "method": "contract-aligned reference (same lse+delta) + slice self-consistency + liveness",
        "criterion": {
            "kind": "allclose",
            "atol": 5e-3,
            "rtol": 1.6e-2,
            "self_consistency_max_abs_err": 5e-3,
        },
        "dtype": "bfloat16",
        "shapes": {
            "q": [batch, seq, heads, d_qk],
            "k": [batch, seq, heads, d_qk],
            "v": [batch, seq, heads, d_v],
            "do": [batch, seq, heads, d_v],
        },
        "status": "pass" if ok else "fail",
        "verdict": "pass (contract-aligned)" if ok else "fail",
        "max_abs_err": res_c["max_abs_err"],
        "extra": {
            "contract_aligned_reference": res_c,
            "design_golden_variant": {
                "stats": res_g,
                "note": (
                    "same lse/delta as the op, plus P and dS rounded to the input "
                    "dtype before the GEMMs (the kernel's documented semantics, "
                    "DESIGN §8.1 golden). Diagnostic only -- the primary criterion "
                    "uses the fp32-arithmetic reference."
                ),
            },
            "contract_worst_elements": worst_c,
            "legacy_convention": {
                "reference": res_l,
                "note": (
                    "MLADenseBwdWorkload.ref_program recomputes delta = sum_j P_j(do.v_j) "
                    "in fp32; the op's _delta_kernel uses delta = sum_d o*do with the "
                    "bf16 forward output. Both are mathematically equivalent; the "
                    "difference is a numerical convention that the bf16 elementwise "
                    "band does not cover on rows with very few visible keys."
                ),
                "outlier_attribution": attribution,
            },
            "self_consistency": {
                "ok": slice_ok,
                "max_abs_err": slice_max,
                "slices": slices,
            },
            "liveness": live,
            "liveness_ok": live_ok,
            "per_batch_zero_frac": per_batch_zero,
            "all_exact_slices": all(s["exact"] for s in slices),
        },
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# 9. MLA varlen backward (ds-v3-mixed)
# ---------------------------------------------------------------------------


@case("mla_varlen_bwd")
def mla_varlen_bwd():
    """ds-v3-mixed (dominant): q_lens=[512,1024,2048,4096,8192,16384], H=128, bf16."""
    return _mla_varlen([512, 1024, 2048, 4096, 8192, 16384], "ds-v3-mixed")


@case("mla_varlen_bwd_8x4k")
def mla_varlen_bwd_8x4k():
    """ds-v3-8x4k: eight 4096-token requests, H=128, bf16 -- the case the grid-cap
    bug was found on (re-checked here at the fixed head slab)."""
    return _mla_varlen([4096] * 8, "ds-v3-8x4k")


def _mla_varlen(q_lens, label):
    from tileops.ops.attention.mla_varlen_bwd import MLAVarlenBwdOp
    from tileops.workloads.attention import MLAVarlenBwdWorkload

    heads, d_qk, d_v = 128, 192, 128
    dtype = torch.bfloat16
    seed = _seed()
    wl = MLAVarlenBwdWorkload(q_lens, q_lens, heads, d_qk, d_v, dtype)
    op = MLAVarlenBwdOp(is_causal=True)
    inputs = wl.gen_inputs()
    do, q, k, v, o, lse, cu_q, cu_kv = inputs
    with torch.no_grad():
        got = op(*inputs)

    # --- primary criterion: contract-aligned reference -------------------------
    # `lse` is the fp32 tensor the op receives (head-major for packed requests);
    # `delta` is `rowsum(o*do)` in fp32 from the bf16 `o` the op receives, i.e.
    # exactly what `_delta_kernel` produces.
    kv_offs = [int(x) for x in cu_kv.tolist()]
    q_offs = [int(x) for x in cu_q.tolist()]
    contract = mla_bwd_reference_contract(
        do, q, k, v, o, lse, q_offs, kv_offs, is_causal=True, scale=d_qk**-0.5
    )
    # Diagnostic variant: same statistics + the kernel's documented P/dS ->
    # input-dtype quantization (DESIGN §8.1 golden).  Not the criterion.
    golden = mla_bwd_reference_contract(
        do,
        q,
        k,
        v,
        o,
        lse,
        q_offs,
        kv_offs,
        is_causal=True,
        scale=d_qk**-0.5,
        quantize_pd=True,
    )
    ref_golden = tuple(g.to(dtype) for g in golden)
    ref_contract = tuple(g.to(dtype) for g in contract)
    res_c = _mla_compare(
        ["dq", "dk", "dv"], got, ref_contract, dtype, atol=5e-3, rtol=1.6e-2
    )
    res_g = _mla_compare(
        ["dq", "dk", "dv"], got, ref_golden, dtype, atol=5e-3, rtol=1.6e-2
    )
    worst_c = (
        _worst_elements(got, ref_contract, 5e-3, 1.6e-2, ["dq", "dk", "dv"])
        if not res_c["ok"]
        else {"skipped": "all elements inside the criterion"}
    )

    # --- historical record: the workload reference's own delta convention -------
    ref_legacy = wl.ref_program(*inputs)
    res_l = _mla_compare(
        ["dq", "dk", "dv"], got, ref_legacy, dtype, atol=5e-3, rtol=1.6e-2
    )

    # Per-request liveness: a truncated grid drops whole requests (or whole head
    # slabs of them) and looks like an exact zero, so check every request/head.
    offs = [int(x) for x in cu_q.tolist()]
    live_per_request = []
    for r, qlen in enumerate(q_lens):
        a0, a1 = offs[r], offs[r + 1]
        dq_r = _f32(got[0][a0:a1]).reshape(qlen, heads, d_qk)
        dk_r = _f32(got[1][a0:a1]).reshape(qlen, heads, d_qk)
        dv_r = _f32(got[2][a0:a1]).reshape(qlen, heads, d_v)
        for name, t in (("dq", dq_r), ("dk", dk_r), ("dv", dv_r)):
            per_head = t.abs().sum(dim=(0, 2))
            live_per_request.append(
                {
                    "request": r,
                    "len": qlen,
                    "output": name,
                    "zero_frac": float((t == 0).float().mean()),
                    "dead_heads": int((per_head == 0).sum()),
                    "nonfinite": int((~torch.isfinite(t)).sum()),
                    "abs_max": float(t.abs().max()),
                }
            )
    live_ok = all(
        v["dead_heads"] == 0 and v["nonfinite"] == 0 for v in live_per_request
    )

    # Per-request self-consistency: each packed request is independent, so
    # running it alone must reproduce its slice of the full result exactly.
    slices = []
    for r, qlen in enumerate(q_lens):
        a0, a1 = offs[r], offs[r + 1]
        sub_cu = torch.zeros(2, dtype=torch.int32, device=cu_q.device)
        sub_cu[1] = qlen
        sub = (
            do[a0:a1],
            q[a0:a1],
            k[a0:a1],
            v[a0:a1],
            o[a0:a1],
            lse[:, a0:a1],
            sub_cu,
            sub_cu,
        )
        with torch.no_grad():
            g = MLAVarlenBwdOp(is_causal=True)(*sub)
        for name, full_t, sub_t in zip(["dq", "dk", "dv"], got, g, strict=True):
            st = pair_stats(sub_t, full_t[a0:a1])
            slices.append(
                {
                    "request": r,
                    "len": qlen,
                    "output": name,
                    "max_abs_err": st["max_abs_err"],
                    "exact": exact_stats(sub_t, full_t[a0:a1])["ok"],
                }
            )
    slice_max = max(s["max_abs_err"] for s in slices)
    slice_ok = slice_max <= 5e-3
    attribution = (
        _mla_delta_convention_probe(
            got[0],
            ref_legacy[0],
            q,
            k,
            v,
            do,
            o,
            d_qk**-0.5,
            bounds_q=offs,
            bounds_kv=kv_offs,
        )
        if not res_l["ok"]
        else {"skipped": "all elements inside the criterion"}
    )
    ok = res_c["ok"] and live_ok and slice_ok
    return {
        "method": "contract-aligned reference (same lse+delta) + per-request self-consistency + liveness",
        "criterion": {
            "kind": "allclose",
            "atol": 5e-3,
            "rtol": 1.6e-2,
            "self_consistency_max_abs_err": 5e-3,
        },
        "dtype": "bfloat16",
        "shapes": {
            "q_lens": q_lens,
            "heads": heads,
            "D_qk": d_qk,
            "D_v": d_v,
            "total_q": int(q.shape[0]),
        },
        "status": "pass" if ok else "fail",
        "verdict": "pass (contract-aligned)" if ok else "fail",
        "max_abs_err": res_c["max_abs_err"],
        "label": label,
        "extra": {
            "contract_aligned_reference": res_c,
            "design_golden_variant": {
                "stats": res_g,
                "note": (
                    "same lse/delta as the op, plus P and dS rounded to the input "
                    "dtype before the GEMMs (the kernel's documented semantics, "
                    "DESIGN §8.1 golden). Diagnostic only -- the primary criterion "
                    "uses the fp32-arithmetic reference."
                ),
            },
            "contract_worst_elements": worst_c,
            "legacy_convention": {
                "reference": res_l,
                "note": (
                    "MLAVarlenBwdWorkload.ref_program recomputes delta = sum_j P_j(do.v_j) "
                    "in fp32; the op's _delta_kernel uses delta = sum_d o*do with the "
                    "bf16 forward output. Both are mathematically equivalent; the "
                    "difference is a numerical convention that the bf16 elementwise "
                    "band does not cover on rows with very few visible keys."
                ),
                "outlier_attribution": attribution,
            },
            "self_consistency": {
                "ok": slice_ok,
                "max_abs_err": slice_max,
                "slices": slices,
            },
            "liveness": live_per_request,
            "liveness_ok": live_ok,
            "all_exact_slices": all(s["exact"] for s in slices),
        },
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_case(name: str) -> dict:
    started = time.time()
    _mem_reset()
    operator, label = CASE_META.get(name, (name, name))
    base = {
        "operator": operator,
        "case": name,
        "harness": "verify_workload_shapes.py",
        "dominant_case": label,
    }
    try:
        payload = CASES[name]()
        payload = {**base, **payload}
        payload["extra"] = payload.get("extra") or {}
    except Exception:  # noqa: BLE001 - one bad case must not lose the rest
        payload = {
            **base,
            "method": "-",
            "criterion": None,
            "dtype": None,
            "shapes": None,
            "status": "unverified",
            "max_abs_err": None,
            "error": traceback.format_exc(limit=8),
        }
    payload["elapsed_s"] = round(time.time() - started, 2)
    peak = _mem_peak_mb()
    if peak is not None:
        payload["npu_peak_mem_MB"] = round(peak, 1)
    return payload


def append_jsonl(payload: dict) -> None:
    with JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


def merge() -> None:
    rows = []
    if JSONL.exists():
        for line in JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    # Last result per case wins (re-runs overwrite).
    latest = {}
    for row in rows:
        latest[row["case"]] = row
    ordered = [latest[name] for name in CASES if name in latest]
    JSON_OUT.write_text(
        json.dumps({"cases": ordered, "source": JSONL.name}, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {JSON_OUT} ({len(ordered)} case(s))")


def mla_reference_selfcheck() -> dict:
    """Validate the contract-aligned reference against the workload reference.

    With ``delta_source="probs"`` the new reference reproduces the workload's
    own ``mla_bwd_reference`` (P recomputed from q/k/v and
    ``delta = sum_j P_j (do.v_j)``, fp32).  Agreement there shows that the new
    code differs from the shipped oracle *only* in the two contract inputs
    (given ``lse`` and ``delta = rowsum(o*do)``), not in the backward arithmetic.
    Runs on the smoke geometry, so it is cheap.
    """
    from tileops.workloads.attention import (
        MLAVarlenBwdWorkload,
        mla_bwd_reference,
    )

    q_lens = [256, 384]
    heads, d_qk, d_v = 8, 192, 128
    dtype = torch.bfloat16
    seed = _seed()
    wl = MLAVarlenBwdWorkload(q_lens, q_lens, heads, d_qk, d_v, dtype)
    do, q, k, v, o, lse, cu_q, cu_kv = wl.gen_inputs()
    bounds_q = [int(x) for x in cu_q.tolist()]
    bounds_kv = [int(x) for x in cu_kv.tolist()]
    scale = d_qk**-0.5
    shipped = mla_bwd_reference(
        q, k, v, do, bounds_q, bounds_kv, is_causal=True, sm_scale=scale
    )
    new = mla_bwd_reference_contract(
        do,
        q,
        k,
        v,
        o,
        lse,
        bounds_q,
        bounds_kv,
        is_causal=True,
        scale=scale,
        delta_source="probs",
    )
    per = {}
    for name, a, b in zip(["dq", "dk", "dv"], new, shipped, strict=True):
        st = pair_stats(a, b)
        per[name] = st["max_abs_err"]
    return {
        "operator": "(harness self-check)",
        "case": "mla_reference_selfcheck",
        "method": "new contract reference (delta_source='probs') vs shipped mla_bwd_reference",
        "criterion": {"kind": "fp32 agreement", "atol": 1e-5, "rtol": 1e-5},
        "dtype": "bfloat16",
        "shapes": {"q_lens": q_lens, "heads": heads, "D_qk": d_qk, "D_v": d_v},
        "max_abs_err": max(per.values()),
        "per_output_max_abs_err": per,
        "status": "pass"
        if all(
            allclose_stats(a, b, 1e-5, 1e-5)["ok"]
            for a, b in zip(new, shipped, strict=True)
        )
        else "fail",
        "seed": seed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated case names")
    ap.add_argument("--list", action="store_true", help="list case names and exit")
    ap.add_argument(
        "--merge", action="store_true", help="build the combined JSON from the jsonl"
    )
    ap.add_argument(
        "--no-append", action="store_true", help="print only, do not touch the jsonl"
    )
    ap.add_argument(
        "--selfcheck",
        action="store_true",
        help="validate the contract-aligned MLA reference (fp32 agreement, smoke shape)",
    )
    args = ap.parse_args()

    if args.list:
        for name in CASES:
            print(name)
        return 0
    if args.merge:
        merge()
        return 0
    if args.selfcheck:
        payload = mla_reference_selfcheck()
        print("RESULT " + json.dumps(payload, default=str))
        return 0 if payload["status"] == "pass" else 1

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        print(f"unknown case(s): {unknown}", file=sys.stderr)
        return 2

    backend = get_device_backend()
    print(f"# device={backend.name} {backend.get_device_name(0)}", file=sys.stderr)
    for name in names:
        payload = run_case(name)
        if not args.no_append:
            append_jsonl(payload)
        print("RESULT " + json.dumps(payload, default=str), flush=True)
        if payload["status"] != "pass":
            print(f"# {name}: {payload['status']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
