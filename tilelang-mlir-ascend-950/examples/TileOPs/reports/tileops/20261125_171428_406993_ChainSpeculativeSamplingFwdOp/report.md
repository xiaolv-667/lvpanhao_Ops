# TileOPs Evaluation Report: ChainSpeculativeSamplingFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-25T17:18:06.160995+00:00 |
| evaluation_scope | ChainSpeculativeSamplingFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261125_171428_406993_ChainSpeculativeSamplingFwdOp |
| git_commit | 4e5f56462476703856120678f82e2365ee8d9164 |
| correctness_target | tests/ops/test_sampling.py |
| benchmark_target | benchmarks/ops/bench_chain_speculative_sampling.py |

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
| 100.0% | 1 | 9 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Acceptance |
|---|---:|---:|---:|---:|
| ChainSpeculativeSamplingFwdOp | 100.0% (9/9) | 0.00e+00 | 7 | MFU 0.345% (0P/7F) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### ChainSpeculativeSamplingFwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| llama-8b-b16-n4-float32 | 43.5180 | 140.1930 | MFU 0.191% (vector, 带宽上限 5.22%) FAIL | {"batch": 16, "num_draft": 4, "vocab": 128256} | N/A | msprof | main | 0.059 TB/s |
| llama-8b-b64-n4-float32 | 84.8480 | 393.6180 | MFU 0.272% (vector, 带宽上限 5.22%) FAIL | {"batch": 64, "num_draft": 4, "vocab": 128256} | N/A | msprof | main | 0.083 TB/s |
| ds-v3-mtp-n1-float32 | 83.0640 | 313.1060 | MFU 0.345% (vector, 带宽上限 5.22%) FAIL | {"batch": 64, "num_draft": 1, "vocab": 129280} | N/A | msprof | main | 0.106 TB/s |
| ds-v3-mtp-n3-float32 | 86.2490 | 346.0930 | MFU 0.312% (vector, 带宽上限 5.22%) FAIL | {"batch": 64, "num_draft": 3, "vocab": 129280} | N/A | msprof | main | 0.096 TB/s |
| qwen3-235b-eagle-n5-float32 | 47.2840 | 374.3370 | MFU 0.169% (vector, 带宽上限 5.22%) FAIL | {"batch": 32, "num_draft": 5, "vocab": 151936} | N/A | msprof | main | 0.052 TB/s |
| llama-8b-b17-n8-float32 | 44.2250 | 155.3840 | MFU 0.183% (vector, 带宽上限 5.22%) FAIL | {"batch": 17, "num_draft": 8, "vocab": 128256} | N/A | msprof | main | 0.056 TB/s |
| llama2-7b-b256-n1-float32 | 191.3710 | 399.5910 | MFU 0.267% (vector, 带宽上限 5.22%) FAIL | {"batch": 256, "num_draft": 1, "vocab": 32000} | N/A | msprof | main | 0.082 TB/s |
