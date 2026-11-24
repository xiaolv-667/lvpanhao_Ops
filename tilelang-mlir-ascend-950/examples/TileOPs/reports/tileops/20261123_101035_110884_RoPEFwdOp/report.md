# TileOPs Evaluation Report: RoPEFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-23T10:12:48.552513+00:00 |
| evaluation_scope | RoPEFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261123_101035_110884_RoPEFwdOp |
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

| Operator | Total Time | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|---:|
| RoPEFwdOp | N/A | 100.0% (34/34) | 5.02e-08 | 7 | 2.93% – 8.39% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### RoPEFwdOp

<details>
<summary>Session Timing / Session 耗时 — N/A</summary>

> Timing data unavailable: operator mismatch: expected 'RoPEFwdOp', got 'rope'

</details>

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| llama3-8b-decode-b64 | 10.9250 | 4.67 | key_shape=[64, 1024]; positions_shape=[64]; query_shape=[64, 4096] | dtype=bfloat16; position_dtype=int64 | msprof | main | 1.6000 |
| llama3-8b-prefill-4k | 430.3610 | 8.32 | key_shape=[4096, 1024]; positions_shape=[4096]; query_shape=[4096, 4096] | dtype=bfloat16; position_dtype=int64 | msprof | main | 1.6000 |
| llama3-8b-prefill-4k | 429.7560 | 8.32 | key_shape=[4096, 1024]; positions_shape=[4096]; query_shape=[4096, 4096] | dtype=float16; position_dtype=int64 | msprof | main | 1.6000 |
| llama3-8b-b2-2k | 671.1190 | 6.25 | key_shape=[2, 2048, 1024]; positions_shape=[2, 2048]; query_shape=[2, 2048, 4096] | dtype=float32; position_dtype=int64 | msprof | main | 1.6000 |
| gptj-6b-prefill | 330.0120 | 8.39 | key_shape=[2048, 4096]; positions_shape=[2048]; query_shape=[2048, 4096] | dtype=float16; position_dtype=int64 | msprof | main | 1.6000 |
| gptj-6b-decode-b32 | 10.3020 | 4.17 | key_shape=[32, 4096]; positions_shape=[32]; query_shape=[32, 4096] | dtype=float16; position_dtype=int64 | msprof | main | 1.6000 |
| phi-2-query-only | 141.3280 | 2.93 | key_shape=N/A; positions_shape=[1000]; query_shape=[1000, 2560] | dtype=bfloat16; position_dtype=int32 | msprof | main | 1.6000 |
