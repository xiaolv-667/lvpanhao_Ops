# TileOPs Evaluation Report: RoPEFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-25T12:39:37.473151+00:00 |
| evaluation_scope | RoPEFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261125_123729_120109_RoPEFwdOp |
| git_commit | 4e5f56462476703856120678f82e2365ee8d9164 |
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
| RoPEFwdOp | 100.0% (34/34) | 5.02e-08 | 7 | MFU 2.28% (0P/7F) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### RoPEFwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| llama3-8b-decode-b64 | 6.1320 | 6.1320 | MFU 0.341% (vector, 带宽上限 2.55%) FAIL | key_shape=[64, 1024]; positions_shape=[64]; query_shape=[64, 4096] | dtype=bfloat16; position_dtype=int64 | msprof | main | 0.2 TFLOP/s |
| llama3-8b-prefill-4k | 76.0600 | 76.0600 | MFU 1.76% (vector, 带宽上限 2.55%) FAIL | key_shape=[4096, 1024]; positions_shape=[4096]; query_shape=[4096, 4096] | dtype=bfloat16; position_dtype=int64 | msprof | main | 1.103 TB/s |
| llama3-8b-prefill-4k | 75.3210 | 75.3210 | MFU 1.78% (vector, 带宽上限 2.55%) FAIL | key_shape=[4096, 1024]; positions_shape=[4096]; query_shape=[4096, 4096] | dtype=float16; position_dtype=int64 | msprof | main | 1.114 TB/s |
| llama3-8b-b2-2k | 120.1060 | 120.1060 | MFU 2.28% (vector, 带宽上限 2.61%) FAIL | key_shape=[2, 2048, 1024]; positions_shape=[2, 2048]; query_shape=[2, 2048, 4096] | dtype=float32; position_dtype=int64 | msprof | main | 1.397 TB/s |
| gptj-6b-prefill | 78.3500 | 78.3500 | MFU 0.342% (vector, 带宽上限 0.638%) FAIL | key_shape=[2048, 4096]; positions_shape=[2048]; query_shape=[2048, 4096] | dtype=float16; position_dtype=int64 | msprof | main | 0.857 TB/s |
| gptj-6b-decode-b32 | 4.9670 | 4.9670 | MFU 0.084% (vector, 带宽上限 0.638%) FAIL | key_shape=[32, 4096]; positions_shape=[32]; query_shape=[32, 4096] | dtype=float16; position_dtype=int64 | msprof | main | 0.0 TFLOP/s |
| phi-2-query-only | 12.5460 | 12.5460 | MFU 0.521% (vector, 带宽上限 1.02%) FAIL | key_shape=N/A; positions_shape=[1000]; query_shape=[1000, 2560] | dtype=bfloat16; position_dtype=int32 | msprof | main | 0.817 TB/s |
