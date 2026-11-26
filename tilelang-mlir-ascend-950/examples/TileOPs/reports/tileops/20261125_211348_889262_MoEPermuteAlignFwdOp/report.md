# TileOPs Evaluation Report: MoEPermuteAlignFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-25T21:17:26.639626+00:00 |
| evaluation_scope | MoEPermuteAlignFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261125_211348_889262_MoEPermuteAlignFwdOp |
| git_commit | 4e5f56462476703856120678f82e2365ee8d9164 |
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
| MoEPermuteAlignFwdOp | 100.0% (30/30) | 4.45e+01 | 12 | lat 27.3us (0P/0F) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### MoEPermuteAlignFwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| kimi-k2-t1 | 6.9760 | 11.4910 | lat 11.5us | [1, 8] | int32 | msprof | N/A | N/A |
| kimi-k2-t32 | 7.0550 | 11.9780 | lat 12.0us | [32, 8] | int32 | msprof | N/A | N/A |
| kimi-k2-t512 | 5.2640 | 17.1950 | lat 17.2us | [512, 8] | int32 | msprof | N/A | N/A |
| kimi-k2-t4096 | 9.5930 | 27.3400 | lat 27.3us | [4096, 8] | int32 | msprof | N/A | N/A |
| ds-v3-t1 | 7.0440 | 11.2070 | lat 11.2us | [1, 8] | int32 | msprof | N/A | N/A |
| ds-v3-t32 | 6.7760 | 11.4280 | lat 11.4us | [32, 8] | int32 | msprof | N/A | N/A |
| ds-v3-t512 | 5.4350 | 17.2030 | lat 17.2us | [512, 8] | int32 | msprof | N/A | N/A |
| ds-v3-t4096 | 9.4590 | 26.3400 | lat 26.3us | [4096, 8] | int32 | msprof | N/A | N/A |
| qwen3-235b-t1 | 6.4740 | 10.0790 | lat 10.1us | [1, 8] | int32 | msprof | N/A | N/A |
| qwen3-235b-t32 | 6.5220 | 10.5100 | lat 10.5us | [32, 8] | int32 | msprof | N/A | N/A |
| qwen3-235b-t512 | 5.3850 | 16.8270 | lat 16.8us | [512, 8] | int32 | msprof | N/A | N/A |
| qwen3-235b-t4096 | 9.5670 | 25.3820 | lat 25.4us | [4096, 8] | int32 | msprof | N/A | N/A |
