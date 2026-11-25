# TileOPs Evaluation Report: EngramGateConvFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-24T22:24:26.188679+00:00 |
| evaluation_scope | EngramGateConvFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261124_222322_139293_EngramGateConvFwdOp |
| git_commit | 2d56d90ab3d6697dbbe13f8e4ce33f3d2fc56c75 |
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
| 100.0% | 1 | 13 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Acceptance |
|---|---:|---:|---:|---:|
| EngramGateConvFwdOp | 100.0% (13/13) | 1.15e-03 | 3 | MFU 1.54% (0P/3F) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### EngramGateConvFwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| train-s64 | 6.5940 | 6.5940 | MFU 0.610% (vector, 带宽上限 9.70%) FAIL | {"M": 2, "d": 512, "eps": 1e-06, "seq_len": 64} | float16 | msprof | main | 0.101 TB/s |
| train-s128 | 5.6090 | 5.6090 | MFU 0.360% (vector, 带宽上限 9.71%) FAIL | {"M": 1, "d": 256, "eps": 1e-06, "seq_len": 128} | bfloat16 | msprof | main | 0.2 TFLOP/s |
| train-s96 | 8.8340 | 8.8340 | MFU 1.54% (vector, 带宽上限 9.77%) FAIL | {"M": 3, "d": 768, "eps": 1e-06, "seq_len": 96} | bfloat16 | msprof | main | 0.252 TB/s |
