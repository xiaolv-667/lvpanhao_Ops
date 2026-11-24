# TileOPs Evaluation Report: SharedExpertMLPFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-24T09:12:37.541963+00:00 |
| evaluation_scope | SharedExpertMLPFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261124_090845_407965_SharedExpertMLPFwdOp |
| git_commit | 2e11033aa79814201ada6ac2e1e7196a3c0287a1 |
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
| SharedExpertMLPFwdOp | 100.0% (25/25) | 1.97e-04 | 6 | MFU 50.8% (0P/6F) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### SharedExpertMLPFwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| 1x7168x18432 | 240.8670 | 723.0160 | MFU 0.290% (cube, 实测 0.296%, 带宽上限 0.423%) FAIL | hidden_states_shape=[1, 7168]; w_down_shape=[7168, 18432]; w_gate_up_shape=[36864, 7168] | bfloat16 | msprof | N/A | 1.096 TB/s |
| 512x7168x18432 | 716.7760 | 2146.9630 | MFU 50.0% (cube, 实测 51.0%) FAIL | hidden_states_shape=[512, 7168]; w_down_shape=[7168, 18432]; w_gate_up_shape=[36864, 7168] | bfloat16 | msprof | N/A | 189.1 TFLOP/s |
| 32x7168x2048 | 30.8960 | 105.8040 | MFU 7.05% (cube, 实测 7.19%, 带宽上限 13.4%) FAIL | hidden_states_shape=[32, 7168]; w_down_shape=[7168, 2048]; w_gate_up_shape=[4096, 7168] | bfloat16 | msprof | N/A | 0.841 TB/s |
| 32x7168x2048 | 28.1890 | 102.7590 | MFU 7.26% (cube, 实测 7.40%, 带宽上限 13.4%) FAIL | hidden_states_shape=[32, 7168]; w_down_shape=[7168, 2048]; w_gate_up_shape=[4096, 7168] | float16 | msprof | N/A | 0.866 TB/s |
| 4096x7168x2048 | 607.6480 | 1879.0469 | MFU 50.8% (cube, 实测 51.8%) FAIL | hidden_states_shape=[4096, 7168]; w_down_shape=[7168, 2048]; w_gate_up_shape=[4096, 7168] | bfloat16 | msprof | N/A | 192.0 TFLOP/s |
| 32x5120x1536 | 16.8720 | 59.9110 | MFU 6.67% (cube, 实测 6.80%, 带宽上限 13.4%) FAIL | hidden_states_shape=[32, 5120]; w_down_shape=[5120, 1536]; w_gate_up_shape=[3072, 5120] | bfloat16 | msprof | N/A | 0.799 TB/s |
