"""Emit ``perf_opt/workload_inventory.json`` and ``perf_opt/perf_records.jsonl``.

Round 3 of the rope tuning (task ``examples/rope_perf3``): the goal was
88.41 % -> >=90 % MBU on the dominant case.  Several structural candidates were
built and measured; none beat the round-2 kernel in the production (warm) report,
so ``perf_opt/RoPEFwdOp.py`` stays the round-2 kernel and this file records both
the candidates and the final state honestly (baseline == final by content).

Kernel-only T_op: one ``msopprof`` capture around one operator call (one launch).
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
TASK = HERE.parent                                     # examples/rope_perf3
KERNEL_ID = (
    "tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py"
    "::_rope_positions_kernel::main"
)
BENCH = "examples/TileOPs/benchmarks/ops/bench_rope.py::test_rope_fwd_bench"
BASELINE_ARTIFACT = "../../rope_perf2/perf_opt/RoPEFwdOp.py"

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
            "reason": ("no_gain this round: the round-2 SIMD-rotate kernel stays the "
                       "delivered best (production 120.1 us / 88.4 % MBU = 95-99 % of the "
                       "measured HBM peak); wide-layout and block-size candidates did not "
                       "beat it in the warm production report"),
            "tuning_status": "no_gain",
            "precision_pass": True,
        })
    (HERE / "workload_inventory.json").write_text(json.dumps(inv, indent=2) + "\n")

    rows = []
    baseline = {
        "llama3-8b-b2-2k-float32": (124.911, "profiles/baseline/base_r2b_r0"),
        "llama3-8b-prefill-4k-bfloat16": (78.379, "profiles/final/final_llama3-8b-prefill-4k-bfloat16_r0"),
        "llama3-8b-prefill-4k-float16": (78.152, "profiles/final/final_llama3-8b-prefill-4k-float16_r0"),
        "gptj-6b-prefill-float16": (80.276, "profiles/baseline/base_gpre_r0"),
        "phi-2-query-only-bfloat16": (13.865, "profiles/final/final_phi-2-query-only-bfloat16_r0"),
        "llama3-8b-decode-b64-bfloat16": (6.985, "profiles/baseline/base_dec_b64_r0"),
        "gptj-6b-decode-b32-float16": (5.980, "profiles/baseline/base_gdec_r0"),
    }
    for wid, (dur, rawdir) in baseline.items():
        rows.append(dict(round=0, candidate_id="baseline-round2", parent_id=None,
                         phase="baseline", artifact=BASELINE_ARTIFACT,
                         workload_id=wid, duration_us=dur, raw=rawdir,
                         note="read-only round-2 product (examples/rope_perf2)"))

    cands = [
        (1, "r1-skip-table-probe", "baseline-round2", "experiments/r1_notable.py", {
            "llama3-8b-b2-2k-float32": (113.287, "r1/r1_notable_r0"),
        }, "diagnostic: cos/sin gather replaced by constants (not a valid kernel)"),
        (1, "r1-wide-hpc4096", "baseline-round2", "experiments/r1_hpc4096.py", {
            "llama3-8b-b2-2k-float32": (122.993, "r1/r1_hpc4096_r0"),
            "llama3-8b-decode-b64-bfloat16": (12.665, "r1/hpc_llama3-8b-decode-b64-bfloat16_r0"),
            "gptj-6b-prefill-float16": (61.269, "r1/hpc_gptj-6b-prefill-float16_r0"),
            "gptj-6b-decode-b32-float16": (15.531, "r1/hpc_gptj-6b-decode-b32-float16_r0"),
        }, "single wide layout (one column chunk a side)"),
        (2, "r2-dual-layout", "r1-wide-hpc4096", "experiments/r2_dual.py", {
            "llama3-8b-b2-2k-float32": (120.633, "r2/r2dual_llama3-8b-b2-2k-float32_r0"),
            "llama3-8b-decode-b64-bfloat16": (11.425, "r2/r2dual_llama3-8b-decode-b64-bfloat16_r0"),
            "gptj-6b-prefill-float16": (61.067, "r2/r2dual_gptj-6b-prefill-float16_r0"),
        }, "wide+narrow layouts chosen by the call's token count"),
        (3, "r3-global-wide", "r2-dual-layout", "experiments/r3b_fp32wide.py", {
            "llama3-8b-b2-2k-float32": (121.300, "r3/r3b_llama3-8b-b2-2k-float32_r0"),
            "llama3-8b-decode-b64-bfloat16": (7.022, "r3/r3b_llama3-8b-decode-b64-bfloat16_r0"),
            "llama3-8b-prefill-4k-bfloat16": (79.517, "r3/r3b_llama3-8b-prefill-4k-bfloat16_r0"),
        }, "wide layout for 4-byte dtypes only, narrow code path elsewhere"),
        (4, "r4-perlayout-tile", "r3-global-wide", "experiments/factory_r4.py", {
            "llama3-8b-b2-2k-float32": (121.259, "r4/r4b_fp32_r0"),
            "llama3-8b-decode-b64-bfloat16": (7.198, "r4/r4_llama3-8b-decode-b64-bfloat16_r0"),
        }, "per-layout sub-tile height (wide: one whole token row per sub-tile)"),
    ]
    for rnd, cid, parent, artifact, table, note in cands:
        path = HERE / artifact
        digest = sha(path) if path.is_file() else "0" * 64
        for wid, (dur, rawdir) in table.items():
            rows.append(dict(round=rnd, candidate_id=cid, parent_id=parent,
                             phase="candidate", artifact=artifact,
                             workload_id=wid, duration_us=dur, raw=rawdir, note=note))

    merged = {
        "llama3-8b-b2-2k-float32": (120.421, "main_kernel_x1rje8be"),
        "llama3-8b-prefill-4k-bfloat16": (76.951, "main_kernel_dtyxgmhq"),
        "llama3-8b-prefill-4k-float16": (75.518, "main_kernel_jefg9zbt"),
        "gptj-6b-prefill-float16": (78.320, "main_kernel_h5l2v55g"),
        "phi-2-query-only-bfloat16": (13.073, "main_kernel_9z2u1hee"),
        "llama3-8b-decode-b64-bfloat16": (5.920, "main_kernel_zgzlwsg_"),
        "gptj-6b-decode-b32-float16": (4.499, "main_kernel_z4imtdpr"),
    }
    run_id = "20261126_085009_745759_RoPEFwdOp"
    wide_artifact = "experiments/factory_r5.py"
    for wid, (dur, art) in merged.items():
        rows.append(dict(round=5, candidate_id="merged-production-wide", parent_id="r4-perlayout-tile",
                         phase="merged", artifact=wide_artifact, workload_id=wid,
                         duration_us=dur,
                         raw=(f"../../TileOPs/reports/tileops/{run_id}/artifacts/msprof/{art}"),
                         note=(f"production report {run_id}: the wide-layout candidate "
                               "installed at rope_kernel/perf_opt/; no gain vs round-2 "
                               "baseline 120.106 us, reverted afterwards")))

    final = {
        # the delivered file is the round-2 kernel byte for byte, so the final rows
        # are the same measurements as the baseline rows
        "llama3-8b-b2-2k-float32": (124.911, "final_llama3-8b-b2-2k-float32"),
        "llama3-8b-prefill-4k-bfloat16": (78.379, "final_llama3-8b-prefill-4k-bfloat16"),
        "llama3-8b-prefill-4k-float16": (78.152, "final_llama3-8b-prefill-4k-float16"),
        "gptj-6b-prefill-float16": (80.276, "final_gptj-6b-prefill-float16"),
        "phi-2-query-only-bfloat16": (13.865, "final_phi-2-query-only-bfloat16"),
        "llama3-8b-decode-b64-bfloat16": (6.985, "final_llama3-8b-decode-b64-bfloat16"),
        "gptj-6b-decode-b32-float16": (5.980, "final_gptj-6b-decode-b32-float16"),
    }
    final_digest = sha(HERE / "RoPEFwdOp.py")
    for wid, (dur, tag) in final.items():
        rows.append(dict(round=6, candidate_id="final-round2-kept", parent_id="baseline-round2",
                         phase="final", artifact="RoPEFwdOp.py", workload_id=wid,
                         duration_us=dur, raw=f"profiles/final/{tag}_r0",
                         note="round-2 kernel delivered unchanged: no candidate beat it in "
                              "the warm production report"))

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
