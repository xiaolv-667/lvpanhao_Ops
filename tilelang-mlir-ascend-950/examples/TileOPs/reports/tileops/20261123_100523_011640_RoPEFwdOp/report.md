# TileOPs Evaluation Report: RoPEFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-23T10:07:31.253689+00:00 |
| evaluation_scope | RoPEFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261123_100523_011640_RoPEFwdOp |
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

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| RoPEFwdOp | 100.0% (34/34) | 5.02e-08 | 7 | 2.82% – 10.82% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### RoPEFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| llama3-8b-decode-b64 | 11.2880 | 4.75 | key_shape=[64, 1024]; positions_shape=[64]; query_shape=[64, 4096] | dtype=bfloat16; position_dtype=int64 | msprof | main | 1.6000 |
| llama3-8b-prefill-4k | 429.4340 | 8.33 | key_shape=[4096, 1024]; positions_shape=[4096]; query_shape=[4096, 4096] | dtype=bfloat16; position_dtype=int64 | msprof | main | 1.6000 |
| llama3-8b-prefill-4k | 431.0900 | 8.29 | key_shape=[4096, 1024]; positions_shape=[4096]; query_shape=[4096, 4096] | dtype=float16; position_dtype=int64 | msprof | main | 1.6000 |
| llama3-8b-b2-2k | 669.6121 | 10.82 | key_shape=[2, 2048, 1024]; positions_shape=[2, 2048]; query_shape=[2, 2048, 4096] | dtype=float32; position_dtype=int64 | msprof | main | 1.6000 |
| gptj-6b-prefill | 330.0280 | 8.41 | key_shape=[2048, 4096]; positions_shape=[2048]; query_shape=[2048, 4096] | dtype=float16; position_dtype=int64 | msprof | main | 1.6000 |
| gptj-6b-decode-b32 | 10.3230 | 4.15 | key_shape=[32, 4096]; positions_shape=[32]; query_shape=[32, 4096] | dtype=float16; position_dtype=int64 | msprof | main | 1.6000 |
| phi-2-query-only | 141.7400 | 2.82 | key_shape=N/A; positions_shape=[1000]; query_shape=[1000, 2560] | dtype=bfloat16; position_dtype=int32 | msprof | main | 1.6000 |
