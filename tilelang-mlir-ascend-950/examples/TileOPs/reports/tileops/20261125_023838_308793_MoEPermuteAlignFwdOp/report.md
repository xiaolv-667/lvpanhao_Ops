# TileOPs Evaluation Report: MoEPermuteAlignFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-25T02:42:08.871287+00:00 |
| evaluation_scope | MoEPermuteAlignFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261125_023838_308793_MoEPermuteAlignFwdOp |
| git_commit | b42b9dcab78733f222394efe9e394392a47844f2 |
| correctness_target | tests/ops/test_moe_permute_align.py |
| benchmark_target | benchmarks/ops/bench_moe_permute_align.py |

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
| 100.0% | 1 | 30 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Acceptance |
|---|---:|---:|---:|---:|
| MoEPermuteAlignFwdOp | 100.0% (30/30) | 5.17e+01 | 12 | lat 29.7us (0P/0F) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### MoEPermuteAlignFwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| kimi-k2-t1 | 10.5420 | 12.4990 | lat 12.5us | [1, 8] | int32 | msprof | N/A | N/A |
| kimi-k2-t32 | 11.6460 | 14.0600 | lat 14.1us | [32, 8] | int32 | msprof | N/A | N/A |
| kimi-k2-t512 | 5.3810 | 18.6530 | lat 18.7us | [512, 8] | int32 | msprof | N/A | N/A |
| kimi-k2-t4096 | 7.2630 | 29.7170 | lat 29.7us | [4096, 8] | int32 | msprof | N/A | N/A |
| ds-v3-t1 | 9.9200 | 12.2530 | lat 12.3us | [1, 8] | int32 | msprof | N/A | N/A |
| ds-v3-t32 | 11.2470 | 13.6460 | lat 13.6us | [32, 8] | int32 | msprof | N/A | N/A |
| ds-v3-t512 | 5.6190 | 19.2310 | lat 19.2us | [512, 8] | int32 | msprof | N/A | N/A |
| ds-v3-t4096 | 6.9850 | 27.6940 | lat 27.7us | [4096, 8] | int32 | msprof | N/A | N/A |
| qwen3-235b-t1 | 9.4870 | 11.5820 | lat 11.6us | [1, 8] | int32 | msprof | N/A | N/A |
| qwen3-235b-t32 | 10.4210 | 12.9170 | lat 12.9us | [32, 8] | int32 | msprof | N/A | N/A |
| qwen3-235b-t512 | 5.5770 | 16.8200 | lat 16.8us | [512, 8] | int32 | msprof | N/A | N/A |
| qwen3-235b-t4096 | 6.2330 | 27.3870 | lat 27.4us | [4096, 8] | int32 | msprof | N/A | N/A |
