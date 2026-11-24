# TileOPs Evaluation Report: MoEPermuteAlignFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-23T16:45:22.066689+00:00 |
| evaluation_scope | MoEPermuteAlignFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261123_164406_365988_MoEPermuteAlignFwdOp |
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
| 96.7% | 1 | 30 | 1 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| MoEPermuteAlignFwdOp | 96.7% (29/30) | 7.53e+02 | 0 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### MoEPermuteAlignFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| - | No performance records | - | - | - | - | - | - |

## Failed Cases / 失败用例

| Stage | Test | Outcome | Message |
|---|---|---|---|
| Correctness | tests.ops.test_moe_permute_align::test_permute_align_op[experts-past-threads] | failed | RuntimeError: Ascend device compilation failed. Command: /usr/local/Ascend/cann-9.2.1/bin/bisheng -O2 -fPIC -std=c++20 -mllvm -cce-aicore-dcpreload-args=false --npu-arch=dav-3510 -I/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/3rdparty/../src --cce-aicore-only /tmp/tmp1fg3o1s6/tl_kernel.asc -o /tmp/tmp1fg3o1s6/tl_kernel.rel.o Compiler output: /tmp/tmp1fg3o1s6/tl_kernel.asc:60:72: warning: 'block_idx' is deprecated in SIMT, use 'blockIdx.x' instead [-Wdeprecated-declarations] [+139 more lines; full report in element text] |
