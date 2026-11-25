# TileOPs Evaluation Report: EngramGateConvBwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-24T23:20:36.960288+00:00 |
| evaluation_scope | EngramGateConvBwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261124_231932_732777_EngramGateConvBwdOp |
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
| 100.0% | 1 | 17 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Acceptance |
|---|---:|---:|---:|---:|
| EngramGateConvBwdOp | 100.0% (17/17) | 2.05e-03 | 3 | MFU 2.61% (0P/3F) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### EngramGateConvBwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| train-s64 | 6.0930 | 8.8090 | MFU 1.08% (vector, 带宽上限 14.3%) FAIL | {"M": 2, "d": 512, "eps": 1e-06, "seq_len": 64} | float16 | msprof | main | 0.121 TB/s |
| train-s128 | 4.8730 | 7.1390 | MFU 0.671% (vector, 带宽上限 14.3%) FAIL | {"M": 1, "d": 256, "eps": 1e-06, "seq_len": 128} | bfloat16 | msprof | main | 0.075 TB/s |
| train-s96 | 9.8030 | 12.3930 | MFU 2.61% (vector, 带宽上限 14.5%) FAIL | {"M": 3, "d": 768, "eps": 1e-06, "seq_len": 96} | bfloat16 | msprof | main | 0.288 TB/s |
