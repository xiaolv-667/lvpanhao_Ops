#!/usr/bin/env python3
"""Re-capture whole-operator msprof times for multi-launch NPU operators.

The TileOPs reporter prefers a single-kernel-name filter, so operators that
launch several kernels per call are reported as *invalid* (unfiltered) or as
*valid but incomplete* (filtered to one launch).  This driver forces an
unfiltered capture (bogus ``TILEOPS_MSPROF_KERNEL_NAME`` makes the harness fall
back to "no filter") and then sums every captured launch into T_op.

Policy (same as ``msprof_sum.py``):
    * one pytest invocation == one profiled operator call (0 warm-up, 1 launch)
    * every kernel launched by that call is summed, including torch-side
      device ops (ZerosLike/Range) that are part of the operator's forward
    * host-padding ops (``Pad*``) are excluded

Output: examples/batch2/t_op_results.json
    Each record carries ``status``: ok | blocked.  A blocked record keeps the
    reason (msprof failure text, missing artifact, timeout) and never reports a
    fabricated 0 as the time.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path("/workspace/tilelang-mlir-ascend-950")
TILEOPS = ROOT / "examples/TileOPs"
CAP = ROOT / "examples/batch2/msprof_capture"
RESULTS = ROOT / "examples/batch2/t_op_results.json"
VENV = "/workspace/venv-ascend950/bin/python"
HOST_PAD = re.compile(r"^Pad", re.I)

# operator, bench file, test function, node id, task dir name
CASES = [
    ("MLADenseBwdOp", "bench_mla_dense_bwd.py", "test_mla_dense_bwd_bench",
     "mla-dense-bwd-smoke-bfloat16", "mla_dense_bwd"),
    ("MLAVarlenBwdOp", "bench_mla_varlen_bwd.py", "test_mla_varlen_bwd_bench",
     "ds-v3-8x4k-bfloat16", "mla_varlen_bwd"),
    ("MLAVarlenBwdOp", "bench_mla_varlen_bwd.py", "test_mla_varlen_bwd_bench",
     "ds-v3-8x4k-float16", "mla_varlen_bwd"),
    ("MLAVarlenBwdOp", "bench_mla_varlen_bwd.py", "test_mla_varlen_bwd_bench",
     "ds-v3-mixed-bfloat16", "mla_varlen_bwd"),
    ("MLAVarlenBwdOp", "bench_mla_varlen_bwd.py", "test_mla_varlen_bwd_bench",
     "kimi-k2-4x8k-bfloat16", "mla_varlen_bwd"),
    ("EngramDecodeFwdOp", "bench_engram.py", "test_engram_decode_bench",
     "serving-b8-bfloat16", "engram_decode"),
    ("EngramDecodeFwdOp", "bench_engram.py", "test_engram_decode_bench",
     "decode-partial-cache-float16", "engram_decode"),
    ("EngramGateConvBwdOp", "bench_engram.py", "test_engram_gate_conv_bwd_bench",
     "train-s32-float16", "engram_gate_conv_bwd"),
    ("EngramGateConvBwdOp", "bench_engram.py", "test_engram_gate_conv_bwd_bench",
     "train-s64-float16", "engram_gate_conv_bwd"),
    ("EngramGateConvBwdOp", "bench_engram.py", "test_engram_gate_conv_bwd_bench",
     "train-s128-bfloat16", "engram_gate_conv_bwd"),
    ("EngramGateConvBwdOp", "bench_engram.py", "test_engram_gate_conv_bwd_bench",
     "train-s96-bfloat16", "engram_gate_conv_bwd"),
    # smoke cases the integration report never recorded (each op's first
    # manifest workload).  Single-kernel ops, so this only needs the event
    # latency, but measuring them keeps the table's case set complete.
    ("EngramGateConvFwdOp", "bench_engram.py", "test_engram_gate_conv_fwd_bench",
     "train-s32-float16", "engram_gate_conv"),
    ("PagedKVCacheWriteFwdOp", "bench_paged_cache_write.py", "test_paged_cache_write_bench",
     "paged-write-smoke-float16-float16", "paged_cache_write"),
    ("SSDChunkCouplingFwdOp", "bench_mamba.py", "test_ssd_chunk_coupling_fwd_bench",
     "ssd-chunk-coupling-smoke-float16", "ssd_chunk_coupling"),
    ("ChainSpeculativeSamplingFwdOp", "bench_chain_speculative_sampling.py",
     "test_chain_speculative_sampling_bench",
     "chain-sampling-smoke-float32", "chain_speculative_sampling"),
]

FAIL_PATTERNS = ("profiling failed", "msprof", "No OpBasicInfo", "returned exit",
                 "No space left", "RuntimeError", "ERROR")


def sum_capture(path: Path):
    """Sum one duration per launched kernel name, excluding Pad* host ops."""
    per: dict[str, float] = {}
    for p in path.rglob("OpBasicInfo*.csv"):
        try:
            for row in csv.DictReader(p.open()):
                name = (row.get("Op Name") or "").strip()
                raw = (row.get("Task Duration(us)") or "").strip()
                if not name or not raw or HOST_PAD.search(name):
                    continue
                dur = float(raw)
                per[name] = per.get(name, 0.0) + dur
        except (OSError, ValueError):
            continue
    return (sum(per.values()), per) if per else (0.0, {})


def latest_bench_json(since: float):
    best = None
    for p in TILEOPS.glob("profile_run_msprof_*.json"):
        try:
            if p.stat().st_mtime >= since - 1:
                if best is None or p.stat().st_mtime > best.stat().st_mtime:
                    best = p
        except OSError:
            continue
    return best


def failure_reason(text: str) -> str:
    for line in (text or "").splitlines():
        low = line.lower()
        if any(pat.lower() in low for pat in FAIL_PATTERNS):
            return line.strip()[:400]
    tail = (text or "").strip().splitlines()
    return tail[-1][:400] if tail else "no output"


def load_results():
    if not RESULTS.exists():
        return []
    try:
        return json.loads(RESULTS.read_text())
    except json.JSONDecodeError:
        # never lose the previous good records on a half-written file
        return []


def run_case(op, bench, test_fn, node, task, rec_by_key):
    outdir = CAP / f"{op}_{node}"
    # start from a clean artifact dir: a stale CSV from a previous attempt must
    # never be summed together with this run's trace.
    if outdir.exists():
        for p in sorted(outdir.rglob("*"), reverse=True):
            try:
                p.unlink() if p.is_file() else p.rmdir()
            except OSError:
                pass
    outdir.mkdir(parents=True, exist_ok=True)

    # msprof stages its raw trace in tempfile.mkdtemp(prefix="tileops_msprof_"):
    # the large MLA cases write several GB there.  The container's root overlay
    # is only 5 GB and ENOSPC is exactly what used to make msprof "fail", so
    # point TMPDIR at the 300 GB /workspace volume.
    scratch = ROOT / "examples/batch2/tmp_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "TILEOPS_MSPROF_KERNEL_NAME": "__force_unfiltered__",
        "TILEOPS_MSPROF_KEEP_OUTPUT": "1",
        "TILEOPS_MSPROF_OUTPUT_DIR": str(outdir),
        "TILEOPS_TASK_DIR": str(ROOT / "examples" / task),
        "ASCEND_RT_VISIBLE_DEVICES": "0",
        "TMPDIR": str(scratch),
    })
    started = time.time()
    print(f"\n=== {op} / {node}", flush=True)
    try:
        proc = subprocess.run(
            [VENV, "-m", "pytest", f"benchmarks/ops/{bench}::{test_fn}[{node}]", "-q"],
            cwd=TILEOPS, env=env, capture_output=True, text=True, timeout=2400)
    except subprocess.TimeoutExpired:
        return {"operator": op, "node": node, "t_op_us": None, "status": "blocked",
                "reason": "pytest timeout (2400s)", "breakdown": {}}

    out = (proc.stdout or "") + (proc.stderr or "")
    t_op, per = sum_capture(outdir)
    rec = {"operator": op, "node": node, "breakdown": per,
           "pytest_rc": proc.returncode,
           "artifact": str(outdir)}
    if t_op <= 0:
        rec.update({"t_op_us": None, "status": "blocked",
                    "reason": failure_reason(out)})
        print(f"  BLOCKED rc={proc.returncode}: {rec['reason']}", flush=True)
        return rec

    rec.update({"t_op_us": round(t_op, 3), "status": "ok"})

    # attach the reporter's own numbers + params for the case, when available
    bj = latest_bench_json(started)
    if bj:
        try:
            data = json.loads(bj.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}
        label = node
        for suffix in ("-bfloat16", "-float16", "-float32"):
            if label.endswith(suffix):
                label = label[: -len(suffix)]
                break
        # Each pytest invocation benchmarks exactly one case, so the matching
        # record is the same operator's only record; prefer an exact/prefix
        # label match when the bench stored a longer id (e.g. the paged write
        # bench appends both T and KV dtype to the node id).
        chosen = None
        for r in data.get("records", []):
            if r.get("operator") != op:
                continue
            lab = str(r.get("label"))
            if chosen is None:
                chosen = r
            if lab == label or node == lab or node.startswith(lab):
                chosen = r
                break
        if chosen is not None:
            r = chosen
            res = r.get("result", {})
            lat = res.get("latency_us")
            rec.update({
                "reported_us": lat,
                "tflops_reported": res.get("tflops"),
                "bw_reported": res.get("bandwidth_tbs"),
                "params": r.get("params"),
                "bench_label": r.get("label"),
            })
            if lat:
                rec["flops"] = (res.get("tflops") or 0) * 1e12 * lat * 1e-6
                rec["bytes"] = (res.get("bandwidth_tbs") or 0) * 1e12 * lat * 1e-6
    print(f"  ok T_op={rec['t_op_us']:.3f} us  {rec['breakdown']}", flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="substring filter on the node id")
    args = ap.parse_args(argv)

    CAP.mkdir(parents=True, exist_ok=True)
    results = load_results()
    for r in results:
        # older records predate the status field
        r.setdefault("status", "ok" if r.get("t_op_us") else "blocked")
    by_key = {(r.get("operator"), r.get("node")): r for r in results}

    for op, bench, test_fn, node, task in CASES:
        if args.only and args.only not in node:
            continue
        if (op, node) in by_key and by_key[(op, node)].get("status") == "ok":
            print(f"skip {op} / {node} (already ok)", flush=True)
            continue
        rec = run_case(op, bench, test_fn, node, task, by_key)
        by_key[(op, node)] = rec
        ordered = list(by_key.values())
        RESULTS.write_text(json.dumps(ordered, ensure_ascii=False, indent=1))

    print(f"\nwrote {RESULTS} ({len(by_key)} records)", flush=True)


if __name__ == "__main__":
    main()
