# TileOPs Evaluation Report: SharedExpertMLPFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-23T15:12:27.561416+00:00 |
| evaluation_scope | SharedExpertMLPFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261123_151219_181414_SharedExpertMLPFwdOp |
| correctness_target | tests/ops/test_fused_moe_shared_expert.py |
| benchmark_target | benchmarks/ops/bench_fused_moe.py |

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
| 96.0% | 1 | 25 | 1 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| SharedExpertMLPFwdOp | 96.0% (24/25) | 1.97e-04 | 0 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### SharedExpertMLPFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| - | No performance records | - | - | - | - | - | - |

## Failed Cases / 失败用例

| Stage | Test | Outcome | Message |
|---|---|---|---|
| Correctness | tests.ops.test_fused_moe_shared_expert::test_shared_expert_mlp_rejects_an_invalid_tensor_contract[down] | failed | AssertionError: Regex pattern did not match.   Expected regex: 'w_down must be'   Actual message: 'w_gate_up must be [2 * S, H] = (128, 64) for hidden_states (8, 64) and w_down (64, 64); got (64, 64)' |
