"""CPU-only checks for the MFU/MU acceptance layer (no NPU required)."""

from __future__ import annotations

import math

from tileops.manifest import load_manifest
from tileops.reporting.acceptance import compute_case, evaluate_roofline, load_peaks


def test_shared_expert_compute_bound_uses_mfu():
    manifest = load_manifest()
    params = {
        "hidden_states_shape": [512, 7168],
        "w_gate_up_shape": [36864, 7168],
        "w_down_shape": [7168, 18432],
        "dtype": "bfloat16",
    }
    acc = compute_case(manifest["SharedExpertMLPFwdOp"], params, 2150.18,
                       peaks=load_peaks(), reported_us=716.78)
    assert acc["class"] == "compute"
    assert acc["metric"] == "MFU"
    # headline uses the vendor peak (378); the measured peak (370.5) is the
    # supplementary column
    assert math.isclose(acc["value"], 49.94, rel_tol=1e-2)
    assert math.isclose(acc["value_measured"], 50.96, rel_tol=1e-2)
    assert acc["pass"] is False
    assert "report-underreports: use t_op_us" in acc["notes"]


def test_mla_dense_is_compute_bound():
    manifest = load_manifest()
    params = {
        "batch": 1, "dim_qk": 192, "dim_v": 128, "dtype": "bfloat16",
        "heads": 128, "is_causal": True, "seqlen_kv": 4096, "seqlen_q": 4096,
    }
    acc = compute_case(manifest["MLADenseBwdOp"], params, 58268.2, peaks=load_peaks())
    assert acc["class"] == "compute"
    assert acc["metric"] == "MFU"
    assert 7.0 < acc["value"] < 9.5


def test_rope_is_memory_bound():
    manifest = load_manifest()
    params = {
        "query_shape": [4096, 128], "key_shape": [4096, 8, 128], "head_size": 128,
        "is_neox_style": True, "base": 10000.0, "dtype": "bfloat16",
        "position_dtype": "int64", "positions_shape": [4096],
    }
    cube = compute_case(manifest["RoPEFwdOp"], params, 429.43, peaks=load_peaks())
    acc = compute_case(
        manifest["RoPEFwdOp"], params, 429.43, peaks=load_peaks(), unit="vector"
    )
    assert acc["class"] == "memory"
    # MFU is the single acceptance metric even for memory-bound shapes; MU is
    # kept only as a diagnostic and reports the MFU's physical ceiling.
    assert acc["metric"] == "MFU"
    assert acc["compute_unit"] == "vector"
    assert acc["peak_used"]["compute_tflops"] == 47.0
    assert math.isclose(acc["value"], cube["value"] * 378.0 / 47.0, rel_tol=1e-6)
    assert acc["value_measured"] is None  # no Vector calibration yet
    assert acc["mu_value"] > acc["value"]
    assert acc["mfu_ceiling_percent"] > cube["mfu_ceiling_percent"]
    assert any("capped at AI/balance" in note for note in acc["notes"])


def test_compute_unit_map_covers_every_migrated_operator():
    from tileops.reporting.acceptance import load_compute_units

    units = load_compute_units()
    expected = {
        "SharedExpertMLPFwdOp": "cube",
        "MLADenseBwdOp": "vector",
        "MLAVarlenBwdOp": "vector",
        "RoPEFwdOp": "vector",
        "PagedKVCacheWriteFwdOp": "vector",
        "ChainSpeculativeSamplingFwdOp": "vector",
        "MoEPermuteAlignFwdOp": "vector",
        "SSDChunkCouplingFwdOp": "vector",
        "EngramDecodeFwdOp": "vector",
        "EngramGateConvFwdOp": "vector",
        "EngramGateConvBwdOp": "vector",
    }
    for operator, unit in expected.items():
        assert units[operator]["unit"] == unit
        assert units[operator]["evidence"]


def test_moe_has_no_arithmetic_and_small_traffic_is_latency():
    manifest = load_manifest()
    params = {"topk_ids_shape": [4096, 8], "num_experts": 384,
              "block_size": 64, "dtype": "int32"}
    flops, nbytes = evaluate_roofline(manifest["MoEPermuteAlignFwdOp"], params)
    assert flops == 0.0
    assert nbytes and nbytes > 0
    acc = compute_case(manifest["MoEPermuteAlignFwdOp"], params, 643.55, peaks=load_peaks())
    assert acc["class"] == "latency"
    assert acc["metric"] == "latency"


def test_engram_bwd_bytes_use_tap_alias():
    manifest = load_manifest()
    params = {"M": 2, "seq_len": 64, "d": 512, "dtype": "float16"}
    flops, nbytes = evaluate_roofline(manifest["EngramGateConvBwdOp"], params)
    assert flops and flops > 0
    assert nbytes and nbytes > 0


def test_throughput_above_peak_is_flagged_invalid():
    """A >peak throughput is physically impossible and must not be published."""
    manifest = load_manifest()
    params = {
        "dim_qk": 192, "dim_v": 128, "dtype": "bfloat16", "heads": 128,
        "is_causal": True, "sm_scale": None,
        "q_lens": [4096] * 8, "kv_lens": [4096] * 8,
    }
    # 20.7 ms is the truncated-grid measurement (only 1.5% of the work ran).
    acc = compute_case(manifest["MLAVarlenBwdOp"], params, 20732.0, peaks=load_peaks())
    assert acc["status"] == "invalid"
    assert acc["pass"] is False
    assert any("physically-impossible" in note for note in acc["notes"])


def test_aggregate_keeps_mfu_and_mu_in_separate_buckets():
    from tileops.reporting.acceptance import aggregate

    cases = [
        {"operator": "DemoOp", "acceptance": {
            "class": "compute", "metric": "MFU", "value": 40.0, "pass": False,
            "notes": []}},
        {"operator": "DemoOp", "acceptance": {
            "class": "memory", "metric": "MU", "value": 20.0, "pass": False,
            "notes": []}},
        {"operator": "DemoOp", "acceptance": {
            "class": "memory", "metric": "MU", "value": 30.0, "pass": False,
            "notes": []}},
        # a >peak capture must not become the operator's best value
        {"operator": "DemoOp", "acceptance": {
            "status": "invalid", "class": "compute", "metric": "MFU", "value": 183.0,
            "pass": False, "notes": ["physically-impossible"]}},
    ]
    summary = aggregate(cases)["DemoOp"]
    assert summary["metrics"]["MFU"]["best"] == 40.0
    assert summary["metrics"]["MU"]["best"] == 30.0
    # headline follows the metric covering the most cases
    assert summary["metric"] == "MU"
    assert summary["best"] == 30.0
    assert summary["failed"] == 4
    assert summary["invalid"] == 1


def test_invalid_acceptance_fails_the_benchmark_stage():
    """A truncated-grid capture must not let the benchmark stage pass."""
    from tileops.reporting.analyzer import analyze_run

    params = {
        "dim_qk": 192, "dim_v": 128, "dtype": "bfloat16", "heads": 128,
        "is_causal": True, "sm_scale": None,
        "q_lens": [4096] * 8, "kv_lens": [4096] * 8,
    }
    benchmark = {
        "status": "present",
        "profiling_mode_requested": "msprof",
        "records": [
            {
                "operator": "MLAVarlenBwdOp",
                "case_id": "varlen-8x4k",
                "label": "ds-v3-8x4k",
                "tag": "tileops",
                "params": params,
                "result": {
                    "latency_us": 20774.0,
                    "prof_mode": "msprof",
                    "msprof": {"captured_op_count": 4, "t_op_us": 20732.0},
                },
            }
        ],
    }
    correctness = {
        "status": "passed", "tests": 1, "passed": 1, "failed": 0, "errors": 0, "cases": []
    }
    run = analyze_run(
        operator="MLAVarlenBwdOp",
        correctness=correctness,
        correctness_exit_code=0,
        benchmark=benchmark,
        benchmark_exit_code=0,
    )
    assert run["summary"]["invalid_acceptance_count"] == 1
    assert run["summary"]["benchmark_passed"] is False
    assert run["status"] == "partial"
    assert any("exceed the physical peak" in warning for warning in run["warnings"])


def test_multi_kernel_capture_is_valid_when_t_op_is_aggregated():
    from tileops.reporting.analyzer import _profile_validity

    metadata = {
        "prof_mode": "msprof",
        "msprof": {"captured_op_count": 4, "t_op_us": 490616.4},
    }
    valid, reason = _profile_validity(metadata)
    assert valid is True and reason is None
    # Without the aggregation the old guard still applies.
    metadata["msprof"].pop("t_op_us")
    valid, reason = _profile_validity(metadata)
    assert valid is False and "4 distinct ops" in reason
