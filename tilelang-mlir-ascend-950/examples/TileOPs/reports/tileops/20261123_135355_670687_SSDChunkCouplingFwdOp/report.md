# TileOPs Evaluation Report: SSDChunkCouplingFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-23T13:54:12.953140+00:00 |
| evaluation_scope | SSDChunkCouplingFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261123_135355_670687_SSDChunkCouplingFwdOp |
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
| 66.7% | 1 | 12 | 4 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| SSDChunkCouplingFwdOp | 66.7% (8/12) | 1.58e-04 | 0 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### SSDChunkCouplingFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| - | No performance records | - | - | - | - | - | - |

## Failed Cases / 失败用例

| Stage | Test | Outcome | Message |
|---|---|---|---|
| Correctness | tests.ops.test_mamba::test_ssd_chunk_scan_fwd[1-2-64-4-64-32-1-dtype0] | failed | AttributeError: type object 'PassConfigKey' has no attribute 'TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION' |
| Correctness | tests.ops.test_mamba::test_ssd_chunk_scan_fwd[1-2-128-4-128-32-1-dtype1] | failed | AttributeError: type object 'PassConfigKey' has no attribute 'TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION' |
| Correctness | tests.ops.test_mamba::test_ssd_chunk_scan_fwd[2-4-64-8-64-64-2-dtype2] | failed | AttributeError: type object 'PassConfigKey' has no attribute 'TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION' |
| Correctness | tests.ops.test_mamba::test_ssd_chunk_scan_fwd[2-2-64-4-64-32-2-dtype3] | failed | AttributeError: type object 'PassConfigKey' has no attribute 'TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION' |
