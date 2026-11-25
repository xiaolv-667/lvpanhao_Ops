# TileOPs Evaluation Report: PagedKVCacheWriteFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-24T21:21:21.040396+00:00 |
| evaluation_scope | PagedKVCacheWriteFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261124_211917_044010_PagedKVCacheWriteFwdOp |
| git_commit | 2d56d90ab3d6697dbbe13f8e4ce33f3d2fc56c75 |
| correctness_target | tests/ops/test_paged_kv_cache_write.py |
| benchmark_target | benchmarks/ops/bench_paged_cache_write.py |

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
| 100.0% | 1 | 15 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Acceptance |
|---|---:|---:|---:|---:|
| PagedKVCacheWriteFwdOp | 100.0% (15/15) | 0.00e+00 | 7 | MFU 0.557% / lat 3.2us (0P/3F, 1 invalid) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### PagedKVCacheWriteFwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| llama-8b-decode-bfloat16 | 3.2430 | 3.2430 | lat 3.2us | cache_shape=[4096, 16, 8, 128]; k_shape=[64, 8, 128] | cache_dtype=bfloat16; dtype=bfloat16 | msprof | N/A | 0.162 TB/s |
| llama-8b-decode-float16 | 3.1880 | 3.1880 | lat 3.2us | cache_shape=[4096, 16, 8, 128]; k_shape=[64, 8, 128] | cache_dtype=float16; dtype=float16 | msprof | N/A | 0.165 TB/s |
| llama-8b-prefill-bfloat16 | 41.4300 | 41.4300 | invalid (capture) | cache_shape=[4096, 16, 8, 128]; k_shape=[8192, 8, 128] | cache_dtype=bfloat16; dtype=bfloat16 | msprof | N/A | 1.621 TB/s |
| qwen3-235b-fp8-bfloat16-float8_e4m3fn | 32.0400 | 32.0400 | MFU 0.557% (vector, 带宽上限 2.26%) FAIL | cache_shape=[4096, 16, 4, 128]; k_shape=[4096, 4, 128] | cache_dtype=float8_e4m3fn; dtype=bfloat16 | msprof | N/A | 0.394 TB/s |
| gpt-oss-120b-bfloat16 | 4.6890 | 4.6890 | - | cache_shape=[4096, 16, 8, 64]; k_shape=[256, 8, 64] | cache_dtype=bfloat16; dtype=bfloat16 | msprof | N/A | 0.224 TB/s |
| odd-tokens-and-page-float16 | 1.7260 | 1.7260 | lat 1.7us | cache_shape=[5, 7, 2, 64]; k_shape=[7, 2, 64] | cache_dtype=float16; dtype=float16 | msprof | N/A | 0.004 TB/s |
| qwen3-235b-fp8-fp16-float16-float8_e4m3fn | 13.5260 | 13.5260 | MFU 0.330% (vector, 带宽上限 2.26%) FAIL | cache_shape=[512, 32, 4, 128]; k_shape=[1024, 4, 128] | cache_dtype=float8_e4m3fn; dtype=float16 | msprof | N/A | 0.233 TB/s |

## Warnings

- correctness passed, but the benchmark stage did not produce valid records
- 1 acceptance record(s) exceed the physical peak (truncated grid or wrong launch plan); benchmark marked failed
