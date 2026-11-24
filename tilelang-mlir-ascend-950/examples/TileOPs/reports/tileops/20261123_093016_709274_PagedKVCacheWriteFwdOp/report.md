# TileOPs Evaluation Report: PagedKVCacheWriteFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-23T09:32:45.878157+00:00 |
| evaluation_scope | PagedKVCacheWriteFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261123_093016_709274_PagedKVCacheWriteFwdOp |
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

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| PagedKVCacheWriteFwdOp | 100.0% (15/15) | 0.00e+00 | 7 | 0.06% – 3.16% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### PagedKVCacheWriteFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| llama-8b-decode-bfloat16 | 15.5510 | 1.36 | cache_shape=[4096, 16, 8, 128]; k_shape=[64, 8, 128] | cache_dtype=bfloat16; dtype=bfloat16 | msprof | N/A | 1.6000 |
| llama-8b-decode-float16 | 15.2120 | 1.39 | cache_shape=[4096, 16, 8, 128]; k_shape=[64, 8, 128] | cache_dtype=float16; dtype=float16 | msprof | N/A | 1.6000 |
| llama-8b-prefill-bfloat16 | 885.4130 | 3.16 | cache_shape=[4096, 16, 8, 128]; k_shape=[8192, 8, 128] | cache_dtype=bfloat16; dtype=bfloat16 | msprof | N/A | 1.6000 |
| qwen3-235b-fp8-bfloat16-float8_e4m3fn | 269.6120 | 2.46 | cache_shape=[4096, 16, 4, 128]; k_shape=[4096, 4, 128] | cache_dtype=float8_e4m3fn; dtype=bfloat16 | msprof | N/A | 1.6000 |
| gpt-oss-120b-bfloat16 | 25.3180 | 1.69 | cache_shape=[4096, 16, 8, 64]; k_shape=[256, 8, 64] | cache_dtype=bfloat16; dtype=bfloat16 | msprof | N/A | 1.6000 |
| odd-tokens-and-page-float16 | 6.4550 | 0.06 | cache_shape=[5, 7, 2, 64]; k_shape=[7, 2, 64] | cache_dtype=float16; dtype=float16 | msprof | N/A | 1.6000 |
| qwen3-235b-fp8-fp16-float16-float8_e4m3fn | 77.7980 | 2.18 | cache_shape=[512, 32, 4, 128]; k_shape=[1024, 4, 128] | cache_dtype=float8_e4m3fn; dtype=float16 | msprof | N/A | 1.6000 |
