# TileOPs Evaluation Report: EngramDecodeFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-25T00:51:47.933255+00:00 |
| evaluation_scope | EngramDecodeFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261125_005026_893589_EngramDecodeFwdOp |
| git_commit | b42b9dcab78733f222394efe9e394392a47844f2 |
| correctness_target | tests/ops/test_engram.py |
| benchmark_target | benchmarks/ops/bench_engram.py |

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
| 100.0% | 1 | 16 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Acceptance |
|---|---:|---:|---:|---:|
| EngramDecodeFwdOp | 100.0% (16/16) | 3.81e-05 | 4 | MFU 2.20% (0P/4F) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### EngramDecodeFwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| serving-b1 | 3.0670 | 5.7410 | MFU 0.197% (vector, 带宽上限 3.33%) FAIL | {"batch": 1, "conv_kernel_size": 4, "conv_len": 12, "d": 256, "d_mem": 512, "dilation": 3, "eps": 1e-06, "max_conv_len": 12} | float16 | msprof | N/A | 0.1 TFLOP/s |
| serving-b4 | 4.6050 | 8.1650 | MFU 2.20% (vector, 带宽上限 12.6%) FAIL | {"batch": 4, "conv_kernel_size": 4, "conv_len": 20, "d": 512, "d_mem": 1024, "dilation": 5, "eps": 1e-06, "max_conv_len": 20} | float16 | msprof | N/A | 0.280 TB/s |
| serving-b8 | 3.1950 | 6.0940 | MFU 1.48% (vector, 带宽上限 20.9%) FAIL | {"batch": 8, "conv_kernel_size": 4, "conv_len": 18, "d": 256, "d_mem": 512, "dilation": 3, "eps": 1e-06, "max_conv_len": 18} | bfloat16 | msprof | N/A | 0.113 TB/s |
| decode-partial-cache | 2.9110 | 8.8910 | MFU 0.254% (vector, 带宽上限 6.58%) FAIL | {"batch": 2, "conv_kernel_size": 4, "conv_len": 5, "d": 256, "d_mem": 512, "dilation": 3, "eps": 1e-06, "max_conv_len": 12} | float16 | msprof | N/A | 0.062 TB/s |
