# TileOPs Evaluation Report: MLAVarlenBwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-26T03:04:27.668224+00:00 |
| evaluation_scope | MLAVarlenBwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261126_025737_402618_MLAVarlenBwdOp |
| git_commit | 4e5f56462476703856120678f82e2365ee8d9164 |
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

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Acceptance |
|---|---:|---:|---:|---:|
| MLAVarlenBwdOp | 100.0% (12/12) | 1.77e-02 | 4 | MFU 27.4% (0P/4F) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### MLAVarlenBwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| ds-v3-8x4k | 32868.7266 | 138353.9829 | MFU 27.4% (cube, 实测 28.0%) FAIL | {"dim_qk": 192, "dim_v": 128, "heads": 128, "is_causal": true, "kv_lens": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], "q_lens": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], "sm_scale": null} | bfloat16 | msprof | N/A | 103.7 TFLOP/s |
| ds-v3-8x4k | 32850.6602 | 138298.7625 | MFU 27.4% (cube, 实测 28.0%) FAIL | {"dim_qk": 192, "dim_v": 128, "heads": 128, "is_causal": true, "kv_lens": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], "q_lens": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], "sm_scale": null} | float16 | msprof | N/A | 103.7 TFLOP/s |
| ds-v3-mixed | 108625.0547 | 441697.6709 | MFU 22.9% (cube, 实测 23.4%) FAIL | {"dim_qk": 192, "dim_v": 128, "heads": 128, "is_causal": true, "kv_lens": [512, 1024, 2048, 4096, 8192, 16384], "q_lens": [512, 1024, 2048, 4096, 8192, 16384], "sm_scale": null} | bfloat16 | msprof | N/A | 86.5 TFLOP/s |
| kimi-k2-4x8k | 1663.2380 | 140302.3807 | MFU 27.0% (cube, 实测 27.6%) FAIL | {"dim_qk": 192, "dim_v": 128, "heads": 64, "is_causal": true, "kv_lens": [8192, 8192, 8192, 8192], "q_lens": [8192, 8192, 8192, 8192], "sm_scale": null} | bfloat16 | msprof | N/A | 102.2 TFLOP/s |
