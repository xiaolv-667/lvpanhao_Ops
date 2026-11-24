# TileOPs Evaluation Report: MoEPermuteAlignFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-23T16:53:32.513869+00:00 |
| evaluation_scope | MoEPermuteAlignFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261123_164821_903151_MoEPermuteAlignFwdOp |
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

| Operator | Total Time | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|---:|
| MoEPermuteAlignFwdOp | N/A | 100.0% (30/30) | 7.39e+02 | 12 | 0.11% – 31.82% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### MoEPermuteAlignFwdOp

<details>
<summary>Session Timing / Session 耗时 — N/A</summary>

> Timing data unavailable: operator mismatch: expected 'MoEPermuteAlignFwdOp', got 'moe_permute_align'

</details>

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| kimi-k2-t1 | 23.3540 | 0.45 | [1, 8] | int32 | msprof | N/A | 1.6000 |
| kimi-k2-t32 | 34.5600 | 0.50 | [32, 8] | int32 | msprof | N/A | 1.6000 |
| kimi-k2-t512 | 26.2230 | 4.06 | [512, 8] | int32 | msprof | N/A | 1.6000 |
| kimi-k2-t4096 | 45.3210 | 31.82 | [4096, 8] | int32 | msprof | N/A | 1.6000 |
| ds-v3-t1 | 17.5740 | 0.27 | [1, 8] | int32 | msprof | N/A | 1.6000 |
| ds-v3-t32 | 28.1730 | 0.36 | [32, 8] | int32 | msprof | N/A | 1.6000 |
| ds-v3-t512 | 24.1450 | 2.94 | [512, 8] | int32 | msprof | N/A | 1.6000 |
| ds-v3-t4096 | 39.3220 | 20.61 | [4096, 8] | int32 | msprof | N/A | 1.6000 |
| qwen3-235b-t1 | 12.0200 | 0.11 | [1, 8] | int32 | msprof | N/A | 1.6000 |
| qwen3-235b-t32 | 22.6560 | 0.18 | [32, 8] | int32 | msprof | N/A | 1.6000 |
| qwen3-235b-t512 | 22.1840 | 1.60 | [512, 8] | int32 | msprof | N/A | 1.6000 |
| qwen3-235b-t4096 | 29.8380 | 10.84 | [4096, 8] | int32 | msprof | N/A | 1.6000 |

## Warnings

- MoEPermuteAlignFwdOp/MoEPermuteAlignFwdOp-c8b45b8c48af: invalid tileops profile: unfiltered capture contains 2 distinct ops
- MoEPermuteAlignFwdOp/MoEPermuteAlignFwdOp-f70e3f3fc64f: invalid tileops profile: unfiltered capture contains 2 distinct ops
- MoEPermuteAlignFwdOp/MoEPermuteAlignFwdOp-10f9faa78b43: invalid tileops profile: unfiltered capture contains 2 distinct ops
- MoEPermuteAlignFwdOp/MoEPermuteAlignFwdOp-850d29b641f0: invalid tileops profile: unfiltered capture contains 2 distinct ops
- MoEPermuteAlignFwdOp/MoEPermuteAlignFwdOp-a74e2f3e61cb: invalid tileops profile: unfiltered capture contains 2 distinct ops
- MoEPermuteAlignFwdOp/MoEPermuteAlignFwdOp-538eda3ff781: invalid tileops profile: unfiltered capture contains 2 distinct ops
- MoEPermuteAlignFwdOp/MoEPermuteAlignFwdOp-6ec2c88e5e56: invalid tileops profile: unfiltered capture contains 2 distinct ops
- MoEPermuteAlignFwdOp/MoEPermuteAlignFwdOp-f0afaf8d048b: invalid tileops profile: unfiltered capture contains 2 distinct ops
- MoEPermuteAlignFwdOp/MoEPermuteAlignFwdOp-1486cd8db708: invalid tileops profile: unfiltered capture contains 2 distinct ops
- MoEPermuteAlignFwdOp/MoEPermuteAlignFwdOp-c69dbce9eeb0: invalid tileops profile: unfiltered capture contains 2 distinct ops
- MoEPermuteAlignFwdOp/MoEPermuteAlignFwdOp-0426bac62ac4: invalid tileops profile: unfiltered capture contains 2 distinct ops
- MoEPermuteAlignFwdOp/MoEPermuteAlignFwdOp-6bf936a1c34e: invalid tileops profile: unfiltered capture contains 2 distinct ops
- correctness passed, but the benchmark stage did not produce valid records
