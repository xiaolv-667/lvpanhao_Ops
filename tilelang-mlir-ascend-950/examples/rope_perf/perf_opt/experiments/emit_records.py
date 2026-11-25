"""Emit ``perf_opt/workload_inventory.json`` and ``perf_opt/perf_records.jsonl``.

Every row carries the task platform's evidence fields exactly as
``platform_context.PlatformContext.evidence()`` returns them, the artifact
sha256 taken at emission time, and the raw capture directory (``msprof_raw_path``
is the platform's ``measurement.raw_evidence_field``).

Rows are append-only in spirit: this script is re-run only to (re)create the file
from the measurement table below, and never rewrites a different number into an
existing row.
"""

import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

TOOLS = Path("/workspace/tilelang-mlir-ascend-950/.agents/tools")
sys.path.insert(0, str(TOOLS))
from platform_context import get_platform, task_state_path  # noqa: E402

HERE = Path(__file__).resolve().parent.parent          # perf_opt/
TASK = HERE.parent                                     # examples/rope_perf
KERNEL_ID = (
    "tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py"
    "::_rope_positions_kernel::main"
)
BENCH = "examples/TileOPs/benchmarks/ops/bench_rope.py::test_rope_fwd_bench"
BASELINE_ARTIFACT = (
    "../../TileOPs/tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py"
)

# workload_id -> (label, marks, shape, dtype, params)
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

REASON = {
    "llama3-8b-b2-2k-float32":
        "winner merged: DMA-staged single-buffer tile, in-place rotate, pipelined "
        "sub-tiles, block-hoisted 4-wide cos/sin gather (step 32 rows/block); "
        "final record on perf_opt/rope.py",
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

    # ---- inventory -------------------------------------------------------
    inv = {
        "schema": "stage4-workload-inventory-v1",
        "benchmark_source": BENCH,
        "op_name": "RoPEFwdOp",
        "kernel_id": KERNEL_ID,
        "generated_from": (
            "tileops.manifest:RoPEFwdOp workloads (rope.yaml) as expanded by "
            "bench_rope.py::test_rope_fwd_bench; 7 parametrized cases, none with a "
            "skip or smoke mark -> all kind=tune"
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
            "reason": REASON.get(
                wid,
                "winner merged into perf_opt/rope.py (same tuned factory serves every "
                "geometry); final record on perf_opt/rope.py",
            ),
            "tuning_status": "winner_merged",
            "precision_pass": True,
        })
    (HERE / "workload_inventory.json").write_text(json.dumps(inv, indent=2) + "\n")

    # ---- records ---------------------------------------------------------
    # (round, candidate_id, parent_id, phase, artifact, {workload: duration}, raw-dir-map)
    def raw(cand, wl, rep=0):
        return f"profiles/{cand}/{wl}_r{rep}"

    rows = []
    baseline = {
        "llama3-8b-b2-2k-float32": (671.030029, "profiles/experiments/base_probe_r1"),
        "llama3-8b-prefill-4k-bfloat16": (433.190002, "profiles/baseline/base_llama3-8b-prefill-4k-bfloat16_r0"),
        "llama3-8b-prefill-4k-float16": (434.937988, "profiles/baseline/base_llama3-8b-prefill-4k-float16_r0"),
        "gptj-6b-prefill-float16": (332.229004, "profiles/baseline/base_gptj-6b-prefill-float16_r0"),
        "phi-2-query-only-bfloat16": (144.252991, "profiles/baseline/base_phi-2-query-only-bfloat16_r0"),
        "llama3-8b-decode-b64-bfloat16": (12.553, "profiles/baseline/base_llama3-8b-decode-b64-bfloat16_r0"),
        "gptj-6b-decode-b32-float16": (11.282, "profiles/baseline/base_gptj-6b-decode-b32-float16_r0"),
    }
    for wid, (dur, rawdir) in baseline.items():
        rows.append(dict(round=0, candidate_id="baseline-v3", parent_id=None,
                         phase="baseline", artifact=BASELINE_ARTIFACT,
                         workload_id=wid, duration_us=dur, raw=rawdir,
                         note="frozen Stage-3 module; mean of the independent captures "
                              "listed under profiles/"))

    cands = [
        # round, candidate_id, parent, artifact, {workload: (duration, raw)}
        (1, "r1-c1-dma-inplace", "baseline-v3", "rope_round1_c1.py", {
            "llama3-8b-b2-2k-float32": (291.953, "profiles/experiments/r1_c1_r0"),
            "llama3-8b-decode-b64-bfloat16": (7.617, "profiles/experiments/c1_dec_bt4_r0"),
        }),
        (2, "r2-c1-bt8", "r1-c1-dma-inplace", "rope_round2_c1_bt8.py", {
            "llama3-8b-b2-2k-float32": (224.597, "profiles/experiments/c1_bt_4096_r0"),
            "llama3-8b-decode-b64-bfloat16": (5.635, "profiles/experiments/c1_dec_bt8_r0"),
        }),
        (2, "r2-c1-bt16", "r1-c1-dma-inplace", "rope_round2_c1_bt16.py", {
            "llama3-8b-b2-2k-float32": (206.096, "profiles/experiments/c1_bt_8192_r0"),
            "llama3-8b-decode-b64-bfloat16": (6.261, "profiles/experiments/c1_dec_bt16_r0"),
            "gptj-6b-decode-b32-float16": (5.435, "profiles/experiments/c1_gptjdec_bt16_r0"),
        }),
        (2, "r2-c3-pipe-sub8", "r2-c1-bt16", "rope_round2_c3_pipe.py", {
            "llama3-8b-b2-2k-float32": (195.731, "profiles/experiments/c3_sub8_bt4_r0"),
        }),
        (3, "r3-c4-step16", "r2-c3-pipe-sub8", "experiments/sweep_c4_step16.py", {
            "llama3-8b-b2-2k-float32": (166.516, "profiles/experiments/c4_pre_s16_r0"),
            "llama3-8b-decode-b64-bfloat16": (7.177, "profiles/experiments/c4_dec_s16_r0"),
            "gptj-6b-decode-b32-float16": (5.964, "profiles/experiments/c4_gdec_s16_r0"),
            "gptj-6b-prefill-float16": (85.672, "profiles/experiments/c4_gpre_s16_r0"),
        }),
        (3, "r3-c4-step32", "r2-c3-pipe-sub8", "rope_round3_c4_hoist.py", {
            "llama3-8b-b2-2k-float32": (150.384, "profiles/experiments/c4_tab4_r0"),
            "llama3-8b-decode-b64-bfloat16": (9.896, "profiles/experiments/c4_dec_s32_r0"),
            "gptj-6b-decode-b32-float16": (8.380, "profiles/experiments/c4_gdec_s32_r0"),
        }),
        (3, "r3-c4-step64", "r2-c3-pipe-sub8", "experiments/sweep_c4_step64.py", {
            "llama3-8b-b2-2k-float32": (142.320, "profiles/experiments/c4_s8b8_r0"),
            "llama3-8b-decode-b64-bfloat16": (13.551, "profiles/experiments/c4_dec_s64_r0"),
            "gptj-6b-decode-b32-float16": (9.737, "profiles/experiments/c4_gdec_s64_r0"),
        }),
        (3, "r3-c4-tab1", "r2-c3-pipe-sub8", "experiments/sweep_c4_tab1.py", {
            "llama3-8b-b2-2k-float32": (165.747, "profiles/experiments/c4_tab1_r0"),
        }),
        (3, "r3-c4-rot4", "r2-c3-pipe-sub8", "experiments/sweep_c4_rot4.py", {
            "llama3-8b-b2-2k-float32": (214.700, "profiles/experiments/c4_rot4_r0"),
        }),
        (3, "r3-c5-tuned", "r3-c4-step32", "rope_round3_c5_tuned.py", {
            "llama3-8b-b2-2k-float32": (148.516, "profiles/experiments/c5_llama3-8b-b2-2k-float32_r0"),
            "llama3-8b-prefill-4k-bfloat16": (102.918, "profiles/experiments/c5_llama3-8b-prefill-4k-bfloat16_r0"),
            "llama3-8b-prefill-4k-float16": (100.883, "profiles/experiments/c5_llama3-8b-prefill-4k-float16_r0"),
            "gptj-6b-prefill-float16": (84.962, "profiles/experiments/c5_gptj-6b-prefill-float16_r0"),
            "phi-2-query-only-bfloat16": (15.293, "profiles/experiments/c5_phi-2-query-only-bfloat16_r0"),
            "llama3-8b-decode-b64-bfloat16": (10.034, "profiles/experiments/c5_llama3-8b-decode-b64-bfloat16_r0"),
            "gptj-6b-decode-b32-float16": (5.844, "profiles/experiments/c5_gptj-6b-decode-b32-float16_r0"),
        }),
    ]
    for rnd, cid, parent, artifact, table in cands:
        path = HERE / artifact
        digest = sha(path)
        for wid, (dur, rawdir) in table.items():
            rows.append(dict(round=rnd, candidate_id=cid, parent_id=parent,
                             phase="candidate", artifact=artifact,
                             workload_id=wid, duration_us=dur, raw=rawdir,
                             note="single msprof capture of one operator call "
                                  "(or the mean of the captures listed)"))

    final = {
        "llama3-8b-b2-2k-float32": (151.653, "final_llama3-8b-b2-2k-float32"),
        "llama3-8b-prefill-4k-bfloat16": (102.1685, "final_llama3-8b-prefill-4k-bfloat16"),
        "llama3-8b-prefill-4k-float16": (103.8635, "final_llama3-8b-prefill-4k-float16"),
        "gptj-6b-prefill-float16": (85.1285, "final_gptj-6b-prefill-float16"),
        "phi-2-query-only-bfloat16": (14.6825, "final_phi-2-query-only-bfloat16"),
        "llama3-8b-decode-b64-bfloat16": (9.873, "final_llama3-8b-decode-b64-bfloat16"),
        "gptj-6b-decode-b32-float16": (6.023, "final_gptj-6b-decode-b32-float16"),
    }

    merged = {
        "llama3-8b-b2-2k-float32":
            (150.136002, "main_kernel_x1rje8be"),
        "llama3-8b-prefill-4k-bfloat16":
            (99.946999, "main_kernel_dtyxgmhq"),
        "llama3-8b-prefill-4k-float16":
            (99.650002, "main_kernel_jefg9zbt"),
        "gptj-6b-prefill-float16":
            (83.554001, "main_kernel_h5l2v55g"),
        "phi-2-query-only-bfloat16":
            (13.799, "main_kernel_9z2u1hee"),
        "llama3-8b-decode-b64-bfloat16":
            (8.438, "main_kernel_zgzlwsg_"),
        "gptj-6b-decode-b32-float16":
            (5.094, "main_kernel_z4imtdpr"),
    }
    run_id = "20261124_181621_258487_RoPEFwdOp"
    installed = ("../../TileOPs/tileops/kernels/rope/rope/rope_kernel/perf_opt/"
                 "_rope_positions_kernel.py")
    for wid, (dur, art) in merged.items():
        rows.append(dict(round=5, candidate_id="merged-production", parent_id="final-perf-opt-padded",
                         phase="merged", artifact=installed, workload_id=wid,
                         duration_us=dur,
                         raw=(f"../../TileOPs/reports/tileops/{run_id}/artifacts/msprof/{art}"),
                         note=(f"single-operator report {run_id} (wrapper wired to perf_opt), "
                               "msprof Task Duration of the one launch per call")))

    final_digest = sha(HERE / "rope.py")
    for wid, (dur, tag) in final.items():
        rows.append(dict(round=4, candidate_id="final-perf-opt-padded", parent_id="r3-c5-tuned",
                         phase="final", artifact="rope.py", workload_id=wid,
                         duration_us=dur, raw=f"profiles/final/{tag}_r0",
                         note="median of the independent captures under profiles/final/"))

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
                "artifact_sha256": (
                    final_digest if r["phase"] == "final" else sha(HERE / r["artifact"])
                ),
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
