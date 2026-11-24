# TileOPs Evaluation Report: MLADenseBwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-23T11:39:15.367888+00:00 |
| evaluation_scope | MLADenseBwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261123_113734_679663_MLADenseBwdOp |
| correctness_target | tests/ops/test_mla_dense_bwd.py |
| benchmark_target | benchmarks/ops/bench_mla_dense_bwd.py |

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
| 20.0% | 1 | 10 | 8 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| MLADenseBwdOp | 20.0% (2/10) | N/A | 0 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### MLADenseBwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| - | No performance records | - | - | - | - | - | - |

## Failed Cases / 失败用例

| Stage | Test | Outcome | Message |
|---|---|---|---|
| Correctness | tests.ops.test_mla_dense_bwd::test_mla_dense_bwd_op[smoke-bwd-bf16] | failed | ValueError: Kernel expected 11 inputs, but 3 are provided. |
| Correctness | tests.ops.test_mla_dense_bwd::test_mla_dense_bwd_op[smoke-bwd-fp16] | failed | ValueError: Kernel expected 11 inputs, but 3 are provided. |
| Correctness | tests.ops.test_mla_dense_bwd::test_mla_dense_bwd_op[full-bwd-s200] | failed | ValueError: Kernel expected 11 inputs, but 3 are provided. |
| Correctness | tests.ops.test_mla_dense_bwd::test_mla_dense_bwd_op[full-bwd-q128-kv320] | failed | ValueError: Kernel expected 11 inputs, but 3 are provided. |
| Correctness | tests.ops.test_mla_dense_bwd::test_mla_dense_bwd_op[full-bwd-q320-kv128] | failed | ValueError: Kernel expected 11 inputs, but 3 are provided. |
| Correctness | tests.ops.test_mla_dense_bwd::test_mla_dense_bwd_op[full-bwd-noncausal] | failed | ValueError: Kernel expected 11 inputs, but 3 are provided. |
| Correctness | tests.ops.test_mla_dense_bwd::test_mla_dense_bwd_op[full-bwd-sm-scale] | failed | ValueError: Kernel expected 11 inputs, but 3 are provided. |
| Correctness | tests.ops.test_mla_dense_bwd::test_mla_dense_bwd_general_kernel | failed | ValueError: Kernel expected 11 inputs, but 3 are provided. |
