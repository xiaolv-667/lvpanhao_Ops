# TileOPs Evaluation Report: MLADenseBwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-23T11:42:52.130107+00:00 |
| evaluation_scope | MLADenseBwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261123_114002_550215_MLADenseBwdOp |
| correctness_target | tests/ops/test_mla_dense_bwd.py |
| benchmark_target | benchmarks/ops/bench_mla_dense_bwd.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend950PR_9579 x 1 |
| cpu | x86_64 |
| cann | 9.2.1 |
| driver | 25.7.rc1.10 |
| pytorch | 2.13.0+cpu |
| pytorch_npu | 2.13.0rc1 |
| tilelang | 0.1.15 |
| python | 3.12.15 |
| os | Linux-6.6.0-159.4.3.154.oe2403sp4.x86_64-x86_64-with-glibc2.35 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 10 | 0 |

## Operator Analysis / 算子分析

| Operator | Total Time | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|---:|
| MLADenseBwdOp | N/A | 100.0% (10/10) | 1.32e-02 | 5 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### MLADenseBwdOp

<details>
<summary>Session Timing / Session 耗时 — N/A</summary>

> Timing data unavailable: operator mismatch: expected 'MLADenseBwdOp', got 'mla_dense_bwd'

</details>

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| ds-v3-4k | 208.0010 | N/A | {"batch": 1, "dim_qk": 192, "dim_v": 128, "heads": 128, "is_causal": true, "seqlen_kv": 4096, "seqlen_q": 4096, "sm_scale": null} | bfloat16 | msprof | N/A | 6.4628 |
| ds-v3-4k | 208.3330 | N/A | {"batch": 1, "dim_qk": 192, "dim_v": 128, "heads": 128, "is_causal": true, "seqlen_kv": 4096, "seqlen_q": 4096, "sm_scale": null} | float16 | msprof | N/A | 6.4525 |
| ds-v3-b2-8k | 455066.5833 | N/A | {"batch": 2, "dim_qk": 192, "dim_v": 128, "heads": 128, "is_causal": true, "seqlen_kv": 8192, "seqlen_q": 8192, "sm_scale": null} | bfloat16 | events | N/A | 0.0118 |
| kimi-k2-8k | 113774.4148 | N/A | {"batch": 1, "dim_qk": 192, "dim_v": 128, "heads": 64, "is_causal": true, "seqlen_kv": 8192, "seqlen_q": 8192, "sm_scale": null} | bfloat16 | events | N/A | 0.0118 |
| ds-v2-lite-3k | 16621.6948 | N/A | {"batch": 4, "dim_qk": 192, "dim_v": 128, "heads": 16, "is_causal": true, "seqlen_kv": 3072, "seqlen_q": 3072, "sm_scale": null} | bfloat16 | events | N/A | 0.0303 |

## Warnings

- MLADenseBwdOp/MLADenseBwdOp-2dde29e75a7b: invalid tileops profile: unfiltered capture contains 5 distinct ops
- MLADenseBwdOp/MLADenseBwdOp-4f18d71d7fe0: invalid tileops profile: unfiltered capture contains 5 distinct ops
- MLADenseBwdOp/MLADenseBwdOp-e05b56840d90: requested msprof but recorded events
- MLADenseBwdOp/MLADenseBwdOp-c80131794817: requested msprof but recorded events
- MLADenseBwdOp/MLADenseBwdOp-fc86608b1f73: requested msprof but recorded events
- correctness passed, but the benchmark stage did not produce valid records
