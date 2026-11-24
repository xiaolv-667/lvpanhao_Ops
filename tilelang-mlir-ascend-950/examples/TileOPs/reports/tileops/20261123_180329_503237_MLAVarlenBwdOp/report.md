# TileOPs Evaluation Report: MLAVarlenBwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-23T18:07:23.754674+00:00 |
| evaluation_scope | MLAVarlenBwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261123_180329_503237_MLAVarlenBwdOp |
| correctness_target | tests/ops/test_mla_varlen_bwd.py |
| benchmark_target | benchmarks/ops/bench_mla_varlen_bwd.py |

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
| 100.0% | 1 | 12 | 0 |

## Operator Analysis / 算子分析

| Operator | Total Time | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|---:|
| MLAVarlenBwdOp | N/A | 100.0% (12/12) | 1.77e-02 | 4 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### MLAVarlenBwdOp

<details>
<summary>Session Timing / Session 耗时 — N/A</summary>

> Timing data unavailable: operator mismatch: expected 'MLAVarlenBwdOp', got 'mla_varlen_bwd'

</details>

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| ds-v3-8x4k | 20774.7768 | N/A | {"dim_qk": 192, "dim_v": 128, "heads": 128, "is_causal": true, "kv_lens": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], "q_lens": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], "sm_scale": null} | bfloat16 | events | N/A | 0.5177 |
| ds-v3-8x4k | 20776.9335 | N/A | {"dim_qk": 192, "dim_v": 128, "heads": 128, "is_causal": true, "kv_lens": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], "q_lens": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], "sm_scale": null} | float16 | events | N/A | 0.5176 |
| ds-v3-mixed | 1727128.5889 | N/A | {"dim_qk": 192, "dim_v": 128, "heads": 128, "is_causal": true, "kv_lens": [512, 1024, 2048, 4096, 8192, 16384], "q_lens": [512, 1024, 2048, 4096, 8192, 16384], "sm_scale": null} | bfloat16 | events | N/A | 0.0061 |
| kimi-k2-4x8k | 472266.7908 | N/A | {"dim_qk": 192, "dim_v": 128, "heads": 64, "is_causal": true, "kv_lens": [8192, 8192, 8192, 8192], "q_lens": [8192, 8192, 8192, 8192], "sm_scale": null} | bfloat16 | events | N/A | 0.0114 |

## Warnings

- MLAVarlenBwdOp/MLAVarlenBwdOp-42781b32a36f: requested msprof but recorded events
- MLAVarlenBwdOp/MLAVarlenBwdOp-3e7917abd57b: requested msprof but recorded events
- MLAVarlenBwdOp/MLAVarlenBwdOp-22e55d4f19ed: requested msprof but recorded events
- MLAVarlenBwdOp/MLAVarlenBwdOp-40ce6d108415: requested msprof but recorded events
- correctness passed, but the benchmark stage did not produce valid records
