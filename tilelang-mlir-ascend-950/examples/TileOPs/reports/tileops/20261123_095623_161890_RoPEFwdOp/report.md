# TileOPs Evaluation Report: RoPEFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-23T09:57:40.169028+00:00 |
| evaluation_scope | RoPEFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261123_095623_161890_RoPEFwdOp |
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
| 94.1% | 1 | 34 | 2 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| RoPEFwdOp | 94.1% (32/34) | 5.02e-08 | 0 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### RoPEFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| - | No performance records | - | - | - | - | - | - |

## Failed Cases / 失败用例

| Stage | Test | Outcome | Message |
|---|---|---|---|
| Correctness | tests.ops.test_rope::test_rope_query_off_a_vector_boundary[neox] | failed | RuntimeError: copy_between_host_and_device_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:102 NPU function error: device error type 3, error code is 507035 [ERROR] 2026-11-23-09:57:37 (PID:203014, Device:0, RankID:-1) ERR00100 PTA call acl api failed |
| Correctness | tests.ops.test_rope::test_rope_query_off_a_vector_boundary[gptj] | failed | RuntimeError: copy_between_host_and_device_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:102 NPU function error: device error type 3, error code is 507035 [ERROR] 2026-11-23-09:57:37 (PID:203014, Device:0, RankID:-1) ERR00100 PTA call acl api failed |
