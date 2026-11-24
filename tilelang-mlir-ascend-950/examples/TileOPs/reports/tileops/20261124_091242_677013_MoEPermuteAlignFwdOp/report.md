# TileOPs Evaluation Report: MoEPermuteAlignFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-24T09:16:12.160997+00:00 |
| evaluation_scope | MoEPermuteAlignFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261124_091242_677013_MoEPermuteAlignFwdOp |
| git_commit | 2e11033aa79814201ada6ac2e1e7196a3c0287a1 |
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
| MoEPermuteAlignFwdOp | 100.0% (30/30) | 7.41e+02 | 12 | lat 1731.1us (0P/0F) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### MoEPermuteAlignFwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| kimi-k2-t1 | 23.8100 | 26.5160 | lat 26.5us | [1, 8] | int32 | msprof | N/A | N/A |
| kimi-k2-t32 | 34.7790 | 43.3310 | lat 43.3us | [32, 8] | int32 | msprof | N/A | N/A |
| kimi-k2-t512 | 25.8700 | 129.1720 | lat 129.2us | [512, 8] | int32 | msprof | N/A | N/A |
| kimi-k2-t4096 | 45.0030 | 658.4490 | lat 658.4us | [4096, 8] | int32 | msprof | N/A | N/A |
| ds-v3-t1 | 18.1760 | 21.1680 | lat 21.2us | [1, 8] | int32 | msprof | N/A | N/A |
| ds-v3-t32 | 28.2260 | 37.5120 | lat 37.5us | [32, 8] | int32 | msprof | N/A | N/A |
| ds-v3-t512 | 23.7940 | 153.2360 | lat 153.2us | [512, 8] | int32 | msprof | N/A | N/A |
| ds-v3-t4096 | 39.2360 | 939.0170 | lat 939.0us | [4096, 8] | int32 | msprof | N/A | N/A |
| qwen3-235b-t1 | 12.1400 | 15.2330 | lat 15.2us | [1, 8] | int32 | msprof | N/A | N/A |
| qwen3-235b-t32 | 23.3320 | 40.2840 | lat 40.3us | [32, 8] | int32 | msprof | N/A | N/A |
| qwen3-235b-t512 | 22.1150 | 250.9080 | lat 250.9us | [512, 8] | int32 | msprof | N/A | N/A |
| qwen3-235b-t4096 | 29.5340 | 1731.1150 | lat 1731.1us | [4096, 8] | int32 | msprof | N/A | N/A |
