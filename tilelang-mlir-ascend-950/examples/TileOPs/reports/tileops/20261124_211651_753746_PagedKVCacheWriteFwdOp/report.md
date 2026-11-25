# TileOPs Evaluation Report: PagedKVCacheWriteFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.0 |
| date | 2026-11-24T21:18:53.926218+00:00 |
| evaluation_scope | PagedKVCacheWriteFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261124_211651_753746_PagedKVCacheWriteFwdOp |
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
| PagedKVCacheWriteFwdOp | 100.0% (15/15) | 0.00e+00 | 7 | MFU 0.549% / lat 3.1us (0P/2F) |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Acceptance 为算子类别对应的指标（计算型 MFU / 访存型 MU / 时延型 latency）与达标计数。

## Operator Details / 算子明细

### PagedKVCacheWriteFwdOp

| Label | Latency (us) | T_op (us) | Acceptance | Shape / Parameters | DType | Mode | Kernel | Achieved |
|---|---:|---:|---|---|---|---|---|---:|
| llama-8b-decode-bfloat16 | 3.0710 | 3.0710 | lat 3.1us | cache_shape=[4096, 16, 8, 128]; k_shape=[64, 8, 128] | cache_dtype=bfloat16; dtype=bfloat16 | msprof | N/A | 0.171 TB/s |
| llama-8b-decode-float16 | 3.0270 | 3.0270 | lat 3.0us | cache_shape=[4096, 16, 8, 128]; k_shape=[64, 8, 128] | cache_dtype=float16; dtype=float16 | msprof | N/A | 0.173 TB/s |
| llama-8b-prefill-bfloat16 | 41.8340 | 41.8340 | - | cache_shape=[4096, 16, 8, 128]; k_shape=[8192, 8, 128] | cache_dtype=bfloat16; dtype=bfloat16 | msprof | N/A | 1.606 TB/s |
| qwen3-235b-fp8-bfloat16-float8_e4m3fn | 32.4860 | 32.4860 | MFU 0.549% (vector, 带宽上限 2.26%) FAIL | cache_shape=[4096, 16, 4, 128]; k_shape=[4096, 4, 128] | cache_dtype=float8_e4m3fn; dtype=bfloat16 | msprof | N/A | 0.388 TB/s |
| gpt-oss-120b-bfloat16 | 4.9990 | 4.9990 | - | cache_shape=[4096, 16, 8, 64]; k_shape=[256, 8, 64] | cache_dtype=bfloat16; dtype=bfloat16 | msprof | N/A | 0.210 TB/s |
| odd-tokens-and-page-float16 | 2.1940 | 2.1940 | lat 2.2us | cache_shape=[5, 7, 2, 64]; k_shape=[7, 2, 64] | cache_dtype=float16; dtype=float16 | msprof | N/A | 0.003 TB/s |
| qwen3-235b-fp8-fp16-float16-float8_e4m3fn | 13.2940 | 13.2940 | MFU 0.336% (vector, 带宽上限 2.26%) FAIL | cache_shape=[512, 32, 4, 128]; k_shape=[1024, 4, 128] | cache_dtype=float8_e4m3fn; dtype=float16 | msprof | N/A | 0.237 TB/s |
