# TileOPs Evaluation Report: SSDChunkCouplingFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-25T18:38:37.984940+00:00 |
| evaluation_scope | SSDChunkCouplingFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261125_183730_853046_SSDChunkCouplingFwdOp |
| git_commit | 4e5f56462476703856120678f82e2365ee8d9164 |
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

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Acceptance |
|---|---:|---:|---:|---:|
| SSDChunkCouplingFwdOp | 100.0% (8/8) | 1.58e-04 | 3 | MFU 94.0% (1P/2F) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### SSDChunkCouplingFwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| mamba2-780m | 6.1600 | 6.1600 | MFU 46.4% (vector) FAIL | {"batch": 1, "chunk_len": 256, "d_state": 128, "n_groups": 1, "num_chunks": 16} | float16 | msprof | main | 0.681 TB/s |
| mamba2-2p7b | 11.8080 | 11.8080 | MFU 48.4% (vector) FAIL | {"batch": 4, "chunk_len": 256, "d_state": 128, "n_groups": 1, "num_chunks": 8} | bfloat16 | msprof | main | 0.710 TB/s |
| mamba2-grouped-prefill | 54.6640 | 54.6640 | MFU 94.0% (vector) PASS | {"batch": 3, "chunk_len": 256, "d_state": 128, "n_groups": 8, "num_chunks": 12} | float16 | msprof | main | 1.381 TB/s |
