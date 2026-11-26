# TileOPs Evaluation Report: MLADenseBwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-26T04:08:08.535599+00:00 |
| evaluation_scope | MLADenseBwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261126_040450_369928_MLADenseBwdOp |
| git_commit | 4e5f56462476703856120678f82e2365ee8d9164 |
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

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Acceptance |
|---|---:|---:|---:|---:|
| MLADenseBwdOp | 100.0% (10/10) | 1.32e-02 | 5 | MFU 38.3% (0P/5F) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### MLADenseBwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| ds-v3-4k | 220.5060 | 13613.7370 | MFU 34.8% (cube, 实测 35.5%) FAIL | {"batch": 1, "dim_qk": 192, "dim_v": 128, "heads": 128, "is_causal": true, "seqlen_kv": 4096, "seqlen_q": 4096, "sm_scale": null} | bfloat16 | msprof | N/A | 131.7 TFLOP/s |
| ds-v3-4k | 222.8110 | 13593.7582 | MFU 34.9% (cube, 实测 35.6%) FAIL | {"batch": 1, "dim_qk": 192, "dim_v": 128, "heads": 128, "is_causal": true, "seqlen_kv": 4096, "seqlen_q": 4096, "sm_scale": null} | float16 | msprof | N/A | 131.9 TFLOP/s |
| ds-v3-b2-8k | 968.8980 | 99091.6833 | MFU 38.3% (cube, 实测 39.1%) FAIL | {"batch": 2, "dim_qk": 192, "dim_v": 128, "heads": 128, "is_causal": true, "seqlen_kv": 8192, "seqlen_q": 8192, "sm_scale": null} | bfloat16 | msprof | N/A | 144.7 TFLOP/s |
| kimi-k2-8k | 220.9000 | 24860.5079 | MFU 38.1% (cube, 实测 38.9%) FAIL | {"batch": 1, "dim_qk": 192, "dim_v": 128, "heads": 64, "is_causal": true, "seqlen_kv": 8192, "seqlen_q": 8192, "sm_scale": null} | bfloat16 | msprof | N/A | 144.2 TFLOP/s |
| ds-v2-lite-3k | 73.8640 | 3747.2269 | MFU 35.6% (cube, 实测 36.3%) FAIL | {"batch": 4, "dim_qk": 192, "dim_v": 128, "heads": 16, "is_causal": true, "seqlen_kv": 3072, "seqlen_q": 3072, "sm_scale": null} | bfloat16 | msprof | N/A | 134.6 TFLOP/s |
