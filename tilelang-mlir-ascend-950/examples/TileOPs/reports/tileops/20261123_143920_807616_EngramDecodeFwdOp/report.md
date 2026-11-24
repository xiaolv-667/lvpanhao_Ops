# TileOPs Evaluation Report: EngramDecodeFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-23T14:42:12.047543+00:00 |
| evaluation_scope | EngramDecodeFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261123_143920_807616_EngramDecodeFwdOp |
| correctness_target | tests/ops/test_engram.py |
| benchmark_target | benchmarks/ops/bench_engram.py |

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
| 100.0% | 1 | 46 | 0 |

## Operator Analysis / 算子分析

| Operator | Total Time | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|---:|
| EngramDecodeFwdOp | N/A | 100.0% (28/28) | 3.81e-05 | 4 | 6.84% – 35.13% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### EngramDecodeFwdOp

<details>
<summary>Session Timing / Session 耗时 — N/A</summary>

> Timing data unavailable: operator mismatch: expected 'EngramDecodeFwdOp', got 'engram_decode'

</details>

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| serving-b1 | 3.1410 | 12.24 | {"batch": 1, "conv_kernel_size": 4, "conv_len": 12, "d": 256, "d_mem": 512, "dilation": 3, "eps": 1e-06, "max_conv_len": 12} | float16 | msprof | N/A | 1.6000 |
| serving-b4 | 4.5790 | 35.13 | {"batch": 4, "conv_kernel_size": 4, "conv_len": 20, "d": 512, "d_mem": 1024, "dilation": 5, "eps": 1e-06, "max_conv_len": 20} | float16 | msprof | N/A | 1.6000 |
| serving-b8 | 6.3850 | 6.84 | {"batch": 8, "conv_kernel_size": 4, "conv_len": 18, "d": 256, "d_mem": 512, "dilation": 3, "eps": 1e-06, "max_conv_len": 18} | bfloat16 | msprof | N/A | 1.6000 |
| decode-partial-cache | 3.0920 | 12.42 | {"batch": 2, "conv_kernel_size": 4, "conv_len": 5, "d": 256, "d_mem": 512, "dilation": 3, "eps": 1e-06, "max_conv_len": 12} | float16 | msprof | N/A | 1.6000 |

## Warnings

- EngramDecodeFwdOp/EngramDecodeFwdOp-9abd96bfe99e: invalid tileops profile: unfiltered capture contains 2 distinct ops
- EngramDecodeFwdOp/EngramDecodeFwdOp-0f76b47837e2: invalid tileops profile: unfiltered capture contains 2 distinct ops
- EngramDecodeFwdOp/EngramDecodeFwdOp-6b2786be08aa: invalid tileops profile: unfiltered capture contains 2 distinct ops
- EngramDecodeFwdOp/EngramDecodeFwdOp-a35fed079d24: invalid tileops profile: unfiltered capture contains 3 distinct ops
- correctness passed, but the benchmark stage did not produce valid records
