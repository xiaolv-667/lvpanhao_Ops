# TileOPs Evaluation Report: ChainSpeculativeSamplingFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-23T16:27:24.121435+00:00 |
| evaluation_scope | ChainSpeculativeSamplingFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261123_162521_290555_ChainSpeculativeSamplingFwdOp |
| correctness_target | tests/ops/test_sampling.py |
| benchmark_target | benchmarks/ops/bench_chain_speculative_sampling.py |

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
| 100.0% | 1 | 9 | 0 |

## Operator Analysis / 算子分析

| Operator | Total Time | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|---:|
| ChainSpeculativeSamplingFwdOp | N/A | 100.0% (9/9) | 0.00e+00 | 7 | 0.98% – 2.09% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### ChainSpeculativeSamplingFwdOp

<details>
<summary>Session Timing / Session 耗时 — N/A</summary>

> Timing data unavailable: operator mismatch: expected 'ChainSpeculativeSamplingFwdOp', got 'chain_speculative_sampling'

</details>

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| llama-8b-b16-n4-float32 | 1245.8091 | 0.98 | {"batch": 16, "num_draft": 4, "vocab": 128256} | N/A | msprof | main | 1.6000 |
| llama-8b-b64-n4-float32 | 2472.0908 | 2.02 | {"batch": 64, "num_draft": 4, "vocab": 128256} | N/A | msprof | main | 1.6000 |
| ds-v3-mtp-n1-float32 | 2529.2458 | 1.54 | {"batch": 64, "num_draft": 1, "vocab": 129280} | N/A | msprof | main | 1.6000 |
| ds-v3-mtp-n3-float32 | 2542.7910 | 1.91 | {"batch": 64, "num_draft": 3, "vocab": 129280} | N/A | msprof | main | 1.6000 |
| qwen3-235b-eagle-n5-float32 | 1427.4860 | 2.09 | {"batch": 32, "num_draft": 5, "vocab": 151936} | N/A | msprof | main | 1.6000 |
| llama-8b-b17-n8-float32 | 1259.7440 | 1.06 | {"batch": 17, "num_draft": 8, "vocab": 128256} | N/A | msprof | main | 1.6000 |
| llama2-7b-b256-n1-float32 | 1666.3409 | N/A | {"batch": 256, "num_draft": 1, "vocab": 32000} | N/A | msprof | main | 0.0197 |
