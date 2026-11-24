# TileOPs Evaluation Report: MLADenseBwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-24T09:27:53.055779+00:00 |
| evaluation_scope | MLADenseBwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261124_092427_744436_MLADenseBwdOp |
| git_commit | 2e11033aa79814201ada6ac2e1e7196a3c0287a1 |
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
| MLADenseBwdOp | 100.0% (10/10) | 1.32e-02 | 5 | MFU 67.0% (0P/5F) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### MLADenseBwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| ds-v3-4k | 208.3210 | 58270.7867 | MFU 65.5% (vector) FAIL | {"batch": 1, "dim_qk": 192, "dim_v": 128, "heads": 128, "is_causal": true, "seqlen_kv": 4096, "seqlen_q": 4096, "sm_scale": null} | bfloat16 | msprof | N/A | 30.8 TFLOP/s |
| ds-v3-4k | 206.6660 | 58272.4738 | MFU 65.5% (vector) FAIL | {"batch": 1, "dim_qk": 192, "dim_v": 128, "heads": 128, "is_causal": true, "seqlen_kv": 4096, "seqlen_q": 4096, "sm_scale": null} | float16 | msprof | N/A | 30.8 TFLOP/s |
| ds-v3-b2-8k | 957.4080 | 455298.4921 | MFU 67.0% (vector) FAIL | {"batch": 2, "dim_qk": 192, "dim_v": 128, "heads": 128, "is_causal": true, "seqlen_kv": 8192, "seqlen_q": 8192, "sm_scale": null} | bfloat16 | msprof | N/A | 31.5 TFLOP/s |
| kimi-k2-8k | 208.4680 | 113770.5775 | MFU 67.0% (vector) FAIL | {"batch": 1, "dim_qk": 192, "dim_v": 128, "heads": 64, "is_causal": true, "seqlen_kv": 8192, "seqlen_q": 8192, "sm_scale": null} | bfloat16 | msprof | N/A | 31.5 TFLOP/s |
| ds-v2-lite-3k | 72.3380 | 16625.8601 | MFU 64.5% (vector) FAIL | {"batch": 4, "dim_qk": 192, "dim_v": 128, "heads": 16, "is_causal": true, "seqlen_kv": 3072, "seqlen_q": 3072, "sm_scale": null} | bfloat16 | msprof | N/A | 30.3 TFLOP/s |
