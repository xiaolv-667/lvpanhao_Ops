# TileOPs Evaluation Report: SharedExpertMLPFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-24T14:23:21.756889+00:00 |
| evaluation_scope | SharedExpertMLPFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261124_141934_327213_SharedExpertMLPFwdOp |
| git_commit | 2d56d90ab3d6697dbbe13f8e4ce33f3d2fc56c75 |
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

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Acceptance |
|---|---:|---:|---:|---:|
| SharedExpertMLPFwdOp | 100.0% (25/25) | 1.97e-04 | 6 | MFU 89.2% (0P/6F) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### SharedExpertMLPFwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| 1x7168x18432 | 177.3800 | 535.0800 | MFU 0.392% (cube, 实测 0.400%, 带宽上限 0.423%) FAIL | hidden_states_shape=[1, 7168]; w_down_shape=[7168, 18432]; w_gate_up_shape=[36864, 7168] | bfloat16 | msprof | N/A | 1.482 TB/s |
| 512x7168x18432 | 835.0040 | 1203.3130 | MFU 89.2% (cube, 实测 91.1%) FAIL | hidden_states_shape=[512, 7168]; w_down_shape=[7168, 18432]; w_gate_up_shape=[36864, 7168] | bfloat16 | msprof | N/A | 337.3 TFLOP/s |
| 32x7168x2048 | 44.3030 | 65.2330 | MFU 11.4% (cube, 实测 11.7%, 带宽上限 13.4%) FAIL | hidden_states_shape=[32, 7168]; w_down_shape=[7168, 2048]; w_gate_up_shape=[4096, 7168] | bfloat16 | msprof | N/A | 1.364 TB/s |
| 32x7168x2048 | 43.8170 | 64.4330 | MFU 11.6% (cube, 实测 11.8%, 带宽上限 13.4%) FAIL | hidden_states_shape=[32, 7168]; w_down_shape=[7168, 2048]; w_gate_up_shape=[4096, 7168] | float16 | msprof | N/A | 1.381 TB/s |
| 4096x7168x2048 | 763.1150 | 1132.6320 | MFU 84.3% (cube, 实测 86.0%) FAIL | hidden_states_shape=[4096, 7168]; w_down_shape=[7168, 2048]; w_gate_up_shape=[4096, 7168] | bfloat16 | msprof | N/A | 318.6 TFLOP/s |
| 32x5120x1536 | 27.0460 | 40.2000 | MFU 9.94% (cube, 实测 10.1%, 带宽上限 13.4%) FAIL | hidden_states_shape=[32, 5120]; w_down_shape=[5120, 1536]; w_gate_up_shape=[3072, 5120] | bfloat16 | msprof | N/A | 1.190 TB/s |
