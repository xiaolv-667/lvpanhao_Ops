# TileOPs Evaluation Report: RoPEFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-24T18:18:26.494838+00:00 |
| evaluation_scope | RoPEFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261124_181621_258487_RoPEFwdOp |
| git_commit | 2d56d90ab3d6697dbbe13f8e4ce33f3d2fc56c75 |
| correctness_target | tests/ops/test_rope.py |
| benchmark_target | benchmarks/ops/bench_rope.py |

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
| 100.0% | 1 | 34 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Acceptance |
|---|---:|---:|---:|---:|
| RoPEFwdOp | 100.0% (34/34) | 5.02e-08 | 7 | MFU 1.82% (0P/7F) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### RoPEFwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| llama3-8b-decode-b64 | 8.4380 | 8.4380 | MFU 0.248% (vector, 带宽上限 2.55%) FAIL | key_shape=[64, 1024]; positions_shape=[64]; query_shape=[64, 4096] | dtype=bfloat16; position_dtype=int64 | msprof | main | 0.1 TFLOP/s |
| llama3-8b-prefill-4k | 99.9470 | 99.9470 | MFU 1.34% (vector, 带宽上限 2.55%) FAIL | key_shape=[4096, 1024]; positions_shape=[4096]; query_shape=[4096, 4096] | dtype=bfloat16; position_dtype=int64 | msprof | main | 0.840 TB/s |
| llama3-8b-prefill-4k | 99.6500 | 99.6500 | MFU 1.34% (vector, 带宽上限 2.55%) FAIL | key_shape=[4096, 1024]; positions_shape=[4096]; query_shape=[4096, 4096] | dtype=float16; position_dtype=int64 | msprof | main | 0.842 TB/s |
| llama3-8b-b2-2k | 150.1360 | 150.1360 | MFU 1.82% (vector, 带宽上限 2.61%) FAIL | key_shape=[2, 2048, 1024]; positions_shape=[2, 2048]; query_shape=[2, 2048, 4096] | dtype=float32; position_dtype=int64 | msprof | main | 1.118 TB/s |
| gptj-6b-prefill | 83.5540 | 83.5540 | MFU 0.320% (vector, 带宽上限 0.638%) FAIL | key_shape=[2048, 4096]; positions_shape=[2048]; query_shape=[2048, 4096] | dtype=float16; position_dtype=int64 | msprof | main | 0.803 TB/s |
| gptj-6b-decode-b32 | 5.0940 | 5.0940 | MFU 0.082% (vector, 带宽上限 0.638%) FAIL | key_shape=[32, 4096]; positions_shape=[32]; query_shape=[32, 4096] | dtype=float16; position_dtype=int64 | msprof | main | 0.0 TFLOP/s |
| phi-2-query-only | 13.7990 | 13.7990 | MFU 0.474% (vector, 带宽上限 1.02%) FAIL | key_shape=N/A; positions_shape=[1000]; query_shape=[1000, 2560] | dtype=bfloat16; position_dtype=int32 | msprof | main | 0.742 TB/s |
