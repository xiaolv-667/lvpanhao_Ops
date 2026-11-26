"""Emit ``perf_opt/workload_inventory.json`` and ``perf_opt/perf_records.jsonl``.

Round 2 of the rope tuning (task ``examples/rope_perf2``): the baseline is the
round-1 product ``examples/rope_perf/perf_opt/rope.py``; the candidate is the
SIMD-rotate version.  Every row carries the task platform's evidence fields
exactly as ``platform_context.PlatformContext.evidence()`` returns them, the
artifact sha256 at emission time, and the raw capture directory
(``msprof_raw_path`` = the platform's ``measurement.raw_evidence_field``).

Kernel-only T_op: one ``msopprof`` capture around one operator call, one launch
(``main_kernel``) per call.
"""

import datetime
import hashlib
import json
import sys
from pathlib import Path

TOOLS = Path("/workspace/tilelang-mlir-ascend-950/.agents/tools")
sys.path.insert(0, str(TOOLS))
from platform_context import get_platform, task_state_path  # noqa: E402

HERE = Path(__file__).resolve().parent.parent          # perf_opt/
TASK = HERE.parent                                     # examples/rope_perf2
KERNEL_ID = (
    "tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py"
    "::_rope_positions_kernel::main"
)
BENCH = "examples/TileOPs/benchmarks/ops/bench_rope.py::test_rope_fwd_bench"
BASELINE_ARTIFACT = "../../rope_perf/perf_opt/rope.py"

WORKLOADS = {
    "llama3-8b-b2-2k-float32": dict(
        label="llama3-8b-b2-2k", marks=[],
        shape={"positions": [2, 2048], "query": [2, 2048, 4096], "key": [2, 2048, 1024]},
        dtype="float32",
        params=dict(tokens=4096, query_heads=32, key_heads=8, head_size=128,
                    rotary_dim=128, max_position_embeddings=8192, base=500000.0,
                    is_neox_style=True, position_dtype="int64"),
    ),
    "llama3-8b-prefill-4k-bfloat16": dict(
        label="llama3-8b-prefill-4k", marks=[],
        shape={"positions": [4096], "query": [4096, 4096], "key": [4096, 1024]},
        dtype="bfloat16",
        params=dict(tokens=4096, query_heads=32, key_heads=8, head_size=128,
                    rotary_dim=128, max_position_embeddings=8192, base=500000.0,
                    is_neox_style=True, position_dtype="int64"),
    ),
    "llama3-8b-prefill-4k-float16": dict(
        label="llama3-8b-prefill-4k", marks=[],
        shape={"positions": [4096], "query": [4096, 4096], "key": [4096, 1024]},
        dtype="float16",
        params=dict(tokens=4096, query_heads=32, key_heads=8, head_size=128,
                    rotary_dim=128, max_position_embeddings=8192, base=500000.0,
                    is_neox_style=True, position_dtype="int64"),
    ),
    "gptj-6b-prefill-float16": dict(
        label="gptj-6b-prefill", marks=[],
        shape={"positions": [2048], "query": [2048, 4096], "key": [2048, 4096]},
        dtype="float16",
        params=dict(tokens=2048, query_heads=16, key_heads=16, head_size=256,
                    rotary_dim=64, max_position_embeddings=2048, base=10000.0,
                    is_neox_style=False, position_dtype="int64"),
    ),
    "phi-2-query-only-bfloat16": dict(
        label="phi-2-query-only", marks=[],
        shape={"positions": [1000], "query": [1000, 2560], "key": None},
        dtype="bfloat16",
        params=dict(tokens=1000, query_heads=32, key_heads=0, head_size=80,
                    rotary_dim=32, max_position_embeddings=2048, base=10000.0,
                    is_neox_style=True, position_dtype="int32"),
    ),
    "llama3-8b-decode-b64-bfloat16": dict(
        label="llama3-8b-decode-b64", marks=[],
        shape={"positions": [64], "query": [64, 4096], "key": [64, 1024]},
        dtype="bfloat16",
        params=dict(tokens=64, query_heads=32, key_heads=8, head_size=128,
                    rotary_dim=128, max_position_embeddings=8192, base=500000.0,
                    is_neox_style=True, position_dtype="int64"),
    ),
    "gptj-6b-decode-b32-float16": dict(
        label="gptj-6b-decode-b32", marks=[],
        shape={"positions": [32], "query": [32, 4096], "key": [32, 4096]},
        dtype="float16",
        params=dict(tokens=32, query_heads=16, key_heads=16, head_size=256,
                    rotary_dim=64, max_position_embeddings=2048, base=10000.0,
                    is_neox_style=False, position_dtype="int64"),
    ),
}


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    platform = get_platform()
    state = json.loads(task_state_path(TASK).read_text())
    if not state.get("platform_runtime"):
        raise SystemExit("task stage not started: platform_runtime missing")
    evidence = platform.evidence()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    inv = {
        "schema": "stage4-workload-inventory-v1",
        "benchmark_source": BENCH,
        "op_name": "RoPEFwdOp",
        "kernel_id": KERNEL_ID,
        "generated_from": (
            "tileops.manifest:RoPEFwdOp workloads (rope.yaml) as expanded by "
            "bench_rope.py::test_rope_fwd_bench; 7 parametrized cases, no skip or "
            "smoke mark -> all kind=tune"
        ),
        "workloads": [],
    }
    for wid, meta in WORKLOADS.items():
        inv["workloads"].append({
            "benchmark_source": BENCH,
            "kernel_id": KERNEL_ID,
            "workload_id": wid,
            "label": meta["label"],
            "marks": meta["marks"],
            "shape": meta["shape"],
            "dtype": meta["dtype"],
            "params": meta["params"],
            "kind": "tune",
            "reason": ("winner merged: DMA-staged tile + T.simd vector-register rotate "
                       "(NeoX fp32/16-bit; SIMT fallback otherwise), 3-stage sub-tile "
                       "pipeline, block-hoisted 4-wide cos/sin gather"),
            "tuning_status": "winner_merged",
            "precision_pass": True,
        })
    (HERE / "workload_inventory.json").write_text(json.dumps(inv, indent=2) + "\n")

    rows = []
    baseline = {
        "llama3-8b-b2-2k-float32": (148.289, "profiles/baseline/base_c7_r0"),
        "llama3-8b-prefill-4k-bfloat16": (103.129, "profiles/baseline/base_llama3-8b-prefill-4k-bfloat16_r0"),
        "llama3-8b-prefill-4k-float16": (103.199, "profiles/baseline/base_llama3-8b-prefill-4k-float16_r0"),
        "gptj-6b-prefill-float16": (85.661, "profiles/baseline/base_gptj-6b-prefill-float16_r0"),
        "phi-2-query-only-bfloat16": (14.952, "profiles/baseline/base_phi-2-query-only-bfloat16_r0"),
        "llama3-8b-decode-b64-bfloat16": (9.628, "profiles/baseline/base_llama3-8b-decode-b64-bfloat16_r0"),
        "gptj-6b-decode-b32-float16": (5.997, "profiles/baseline/base_gptj-6b-decode-b32-float16_r0"),
    }
    for wid, (dur, rawdir) in baseline.items():
        rows.append(dict(round=0, candidate_id="baseline-round1", parent_id=None,
                         phase="baseline", artifact=BASELINE_ARTIFACT,
                         workload_id=wid, duration_us=dur, raw=rawdir,
                         note="read-only round-1 product (perf_opt/rope.py of task "
                              "examples/rope_perf); mean of the independent captures"))

    cands = [
        (1, "r1-ns3", "baseline-round1", "experiments/r1_ns3.py", {
            "llama3-8b-b2-2k-float32": (145.291, "experiments/r1_ns3_r0"),
        }),
        (1, "r1-ns4", "baseline-round1", "experiments/r1_ns4.py", {
            "llama3-8b-b2-2k-float32": (145.897, "experiments/r1_ns4_r0"),
        }),
        (1, "r1-bt8", "baseline-round1", "experiments/r1_bt8.py", {
            "llama3-8b-b2-2k-float32": (147.174, "experiments/r1_bt8_r0"),
        }),
        (1, "r1-bt16", "baseline-round1", "experiments/r1_bt16.py", {
            "llama3-8b-b2-2k-float32": (145.790, "experiments/r1_bt16_r0"),
        }),
        (2, "r2-simd-rotate", "r1-ns3", "experiments/r3_simd.py", {
            "llama3-8b-b2-2k-float32": (121.354, "experiments/r3_simd_r0"),
            "llama3-8b-prefill-4k-bfloat16": (79.079, "r1/r3_llama3-8b-prefill-4k-bfloat16_r0"),
            "llama3-8b-prefill-4k-float16": (79.206, "r1/r3_llama3-8b-prefill-4k-float16_r0"),
            "gptj-6b-prefill-float16": (79.880, "r1/r3_gptj-6b-prefill-float16_r0"),
            "phi-2-query-only-bfloat16": (14.191, "r1/r3_phi-2-query-only-bfloat16_r0"),
            "llama3-8b-decode-b64-bfloat16": (7.014, "r1/r3_llama3-8b-decode-b64-bfloat16_r0"),
            "gptj-6b-decode-b32-float16": (5.913, "r1/r3_gptj-6b-decode-b32-float16_r0"),
        }),
        (2, "r2-simd-ns2", "r2-simd-rotate", "experiments/r3_simd_ns2.py", {
            "llama3-8b-b2-2k-float32": (129.386, "r1/r3_simd_ns2_r0"),
        }),
        (2, "r2-simd-bt8", "r2-simd-rotate", "experiments/r3_simd_bt8.py", {
            "llama3-8b-b2-2k-float32": (123.708, "r1/r3_simd_bt8_r0"),
        }),
        (2, "r2-simd-step64", "r2-simd-rotate", "experiments/r3_simd_step64.py", {
            "llama3-8b-b2-2k-float32": (123.612, "r1/r3_simd_step64_r0"),
        }),
        (2, "r2-simd-unroll", "r2-simd-rotate", "experiments/r4_simd_unroll.py", {
            "llama3-8b-b2-2k-float32": (122.990, "r1/r4_unroll_r0"),
        }),
        (2, "r2-simd-step16", "r2-simd-rotate", "experiments/r3_simd_s16.py", {
            "llama3-8b-b2-2k-float32": (123.882, "r1/r3s16_llama3-8b-b2-2k-float32_r0"),
            "llama3-8b-prefill-4k-bfloat16": (80.772, "r1/r3s16_llama3-8b-prefill-4k-bfloat16_r0"),
        }),
    ]
    for rnd, cid, parent, artifact, table in cands:
        digest = sha(HERE / artifact)
        for wid, (dur, rawdir) in table.items():
            rows.append(dict(round=rnd, candidate_id=cid, parent_id=parent,
                             phase="candidate", artifact=artifact,
                             workload_id=wid, duration_us=dur, raw=rawdir,
                             note="single msprof capture of one operator call "
                                  "(or the mean of the captures listed)"))

    final = {
        "llama3-8b-b2-2k-float32": (123.220, "final_llama3-8b-b2-2k-float32"),
        "llama3-8b-prefill-4k-bfloat16": (79.320, "final_llama3-8b-prefill-4k-bfloat16"),
        "llama3-8b-prefill-4k-float16": (78.467, "final_llama3-8b-prefill-4k-float16"),
        "gptj-6b-prefill-float16": (80.639, "final_gptj-6b-prefill-float16"),
        "phi-2-query-only-bfloat16": (14.140, "final_phi-2-query-only-bfloat16"),
        "llama3-8b-decode-b64-bfloat16": (7.189, "final_llama3-8b-decode-b64-bfloat16"),
        "gptj-6b-decode-b32-float16": (5.753, "final_gptj-6b-decode-b32-float16"),
    }
    final_digest = sha(HERE / "RoPEFwdOp.py")
    for wid, (dur, tag) in final.items():
        rows.append(dict(round=3, candidate_id="final-simd", parent_id="r2-simd-rotate",
                         phase="final", artifact="RoPEFwdOp.py", workload_id=wid,
                         duration_us=dur, raw=f"profiles/final/{tag}_r0",
                         note="mean (n=2) / median (n=3) of the independent captures "
                              "under profiles/final/"))


    # production check: the new artifact installed at the wrapper's perf_opt path for
    # one single-operator report, then the round-1 file restored (adoption is Stage 5's)
    merged = {
        "llama3-8b-b2-2k-float32": (120.106002, "main_kernel_x1rje8be"),
        "llama3-8b-prefill-4k-bfloat16": (76.060, "main_kernel_dtyxgmhq"),
        "llama3-8b-prefill-4k-float16": (75.321, "main_kernel_jefg9zbt"),
        "gptj-6b-prefill-float16": (78.350, "main_kernel_h5l2v55g"),
        "phi-2-query-only-bfloat16": (12.546, "main_kernel_9z2u1hee"),
        "llama3-8b-decode-b64-bfloat16": (6.132, "main_kernel_zgzlwsg_"),
        "gptj-6b-decode-b32-float16": (4.967, "main_kernel_z4imtdpr"),
    }
    run_id = "20261125_123729_120109_RoPEFwdOp"
    for wid, (dur, art) in merged.items():
        rows.append(dict(round=4, candidate_id="merged-production-check", parent_id="final-simd",
                         phase="merged", artifact="RoPEFwdOp.py", workload_id=wid,
                         duration_us=dur,
                         raw=(f"../../TileOPs/reports/tileops/{run_id}/artifacts/msprof/{art}"),
                         note=(f"single-operator report {run_id} with RoPEFwdOp.py installed at "
                               "rope_kernel/perf_opt/_rope_positions_kernel.py; round-1 file "
                               "restored afterwards")))

    with open(HERE / "perf_records.jsonl", "w") as f:
        for r in rows:
            rec = {
                **evidence,
                "round": r["round"],
                "candidate_id": r["candidate_id"],
                "parent_id": r["parent_id"],
                "kernel_id": KERNEL_ID,
                "workload_id": r["workload_id"],
                "phase": r["phase"],
                "artifact_path": r["artifact"],
                "artifact_sha256": (final_digest if r["phase"] in ("final", "merged")
                                    else sha(HERE / r["artifact"])),
                "duration_us": r["duration_us"],
                "l0_pass": True,
                "msprof_raw_path": r["raw"],
                "timestamp": stamp,
                "note": r["note"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(inv['workloads'])} workloads, {len(rows)} records")


if __name__ == "__main__":
    main()
