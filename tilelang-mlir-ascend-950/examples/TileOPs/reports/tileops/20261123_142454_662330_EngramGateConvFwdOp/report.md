# TileOPs Evaluation Report: EngramGateConvFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-23T14:28:12.365743+00:00 |
| evaluation_scope | EngramGateConvFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261123_142454_662330_EngramGateConvFwdOp |
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
| EngramGateConvFwdOp | N/A | 100.0% (28/28) | 1.15e-03 | 3 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### EngramGateConvFwdOp

<details>
<summary>Session Timing / Session 耗时 — N/A</summary>

> Timing data unavailable: operator mismatch: expected 'EngramGateConvFwdOp', got 'engram_gate_conv'

</details>

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| train-s64 | 8.4660 | N/A | {"M": 2, "d": 512, "eps": 1e-06, "seq_len": 64} | float16 | msprof | main | 0.0784 |
| train-s128 | 7.3480 | N/A | {"M": 1, "d": 256, "eps": 1e-06, "seq_len": 128} | bfloat16 | msprof | main | 0.0453 |
| train-s96 | 11.1170 | N/A | {"M": 3, "d": 768, "eps": 1e-06, "seq_len": 96} | bfloat16 | msprof | main | 0.2002 |

## Warnings

- EngramDecodeFwdOp/EngramDecodeFwdOp-9abd96bfe99e: invalid tileops profile: unfiltered capture contains 2 distinct ops
- EngramDecodeFwdOp/EngramDecodeFwdOp-0f76b47837e2: invalid tileops profile: unfiltered capture contains 2 distinct ops
- EngramDecodeFwdOp/EngramDecodeFwdOp-6b2786be08aa: invalid tileops profile: unfiltered capture contains 2 distinct ops
- EngramDecodeFwdOp/EngramDecodeFwdOp-a35fed079d24: invalid tileops profile: unfiltered capture contains 3 distinct ops
- correctness passed, but the benchmark stage did not produce valid records
