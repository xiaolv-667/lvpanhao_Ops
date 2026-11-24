# TileOPs Evaluation Report: SSDChunkCouplingFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-23T13:57:42.592710+00:00 |
| evaluation_scope | SSDChunkCouplingFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261123_135638_847395_SSDChunkCouplingFwdOp |
| correctness_target | tests/ops/test_mamba.py |
| benchmark_target | benchmarks/ops/bench_mamba.py |

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
| 100.0% | 1 | 8 | 0 |

## Operator Analysis / 算子分析

| Operator | Total Time | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|---:|
| SSDChunkCouplingFwdOp | N/A | 100.0% (8/8) | 1.58e-04 | 3 | 14.47% – 18.11% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### SSDChunkCouplingFwdOp

<details>
<summary>Session Timing / Session 耗时 — N/A</summary>

> Timing data unavailable: operator mismatch: expected 'SSDChunkCouplingFwdOp', got 'ssd_chunk_coupling'

</details>

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| mamba2-780m | 12.6430 | 14.47 | {"batch": 1, "chunk_len": 256, "d_state": 128, "n_groups": 1, "num_chunks": 16} | float16 | msprof | main | 1.6000 |
| mamba2-2p7b | 22.9100 | 15.24 | {"batch": 4, "chunk_len": 256, "d_state": 128, "n_groups": 1, "num_chunks": 8} | bfloat16 | msprof | main | 1.6000 |
| mamba2-grouped-prefill | 171.0000 | 18.11 | {"batch": 3, "chunk_len": 256, "d_state": 128, "n_groups": 8, "num_chunks": 12} | float16 | msprof | main | 1.6000 |
