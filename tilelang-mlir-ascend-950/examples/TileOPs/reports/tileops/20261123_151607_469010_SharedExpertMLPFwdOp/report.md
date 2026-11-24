# TileOPs Evaluation Report: SharedExpertMLPFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-23T15:20:05.269669+00:00 |
| evaluation_scope | SharedExpertMLPFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261123_151607_469010_SharedExpertMLPFwdOp |
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
| 100.0% | 1 | 25 | 0 |

## Operator Analysis / 算子分析

| Operator | Total Time | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|---:|
| SharedExpertMLPFwdOp | N/A | 100.0% (25/25) | 1.97e-04 | 6 | 30.55% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### SharedExpertMLPFwdOp

<details>
<summary>Session Timing / Session 耗时 — N/A</summary>

> Timing data unavailable: operator mismatch: expected 'SharedExpertMLPFwdOp', got 'shared_expert_mlp'

</details>

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| 1x7168x18432 | 227.4760 | N/A | hidden_states_shape=[1, 7168]; w_down_shape=[7168, 18432]; w_gate_up_shape=[36864, 7168] | bfloat16 | msprof | N/A | 3.4850 |
| 512x7168x18432 | 716.7780 | 30.55 | hidden_states_shape=[512, 7168]; w_down_shape=[7168, 18432]; w_gate_up_shape=[36864, 7168] | bfloat16 | msprof | N/A | 1.6000 |
| 32x7168x2048 | 27.9040 | N/A | hidden_states_shape=[32, 7168]; w_down_shape=[7168, 2048]; w_gate_up_shape=[4096, 7168] | bfloat16 | msprof | N/A | 3.1894 |
| 32x7168x2048 | 28.1790 | N/A | hidden_states_shape=[32, 7168]; w_down_shape=[7168, 2048]; w_gate_up_shape=[4096, 7168] | float16 | msprof | N/A | 3.1583 |
| 4096x7168x2048 | 608.4310 | N/A | hidden_states_shape=[4096, 7168]; w_down_shape=[7168, 2048]; w_gate_up_shape=[4096, 7168] | bfloat16 | msprof | N/A | 0.3378 |
| 32x5120x1536 | 16.6750 | N/A | hidden_states_shape=[32, 5120]; w_down_shape=[5120, 1536]; w_gate_up_shape=[3072, 5120] | bfloat16 | msprof | N/A | 2.8690 |

## Warnings

- SharedExpertMLPFwdOp/SharedExpertMLPFwdOp-158d51ea59b2: invalid tileops profile: unfiltered capture contains 2 distinct ops
- SharedExpertMLPFwdOp/SharedExpertMLPFwdOp-7c06f86a85de: invalid tileops profile: unfiltered capture contains 2 distinct ops
- SharedExpertMLPFwdOp/SharedExpertMLPFwdOp-1f9e953ef7ef: invalid tileops profile: unfiltered capture contains 2 distinct ops
- SharedExpertMLPFwdOp/SharedExpertMLPFwdOp-371c3c2386e3: invalid tileops profile: unfiltered capture contains 2 distinct ops
- SharedExpertMLPFwdOp/SharedExpertMLPFwdOp-69103b45338d: invalid tileops profile: unfiltered capture contains 2 distinct ops
- SharedExpertMLPFwdOp/SharedExpertMLPFwdOp-0703f2622a51: invalid tileops profile: unfiltered capture contains 2 distinct ops
- correctness passed, but the benchmark stage did not produce valid records
