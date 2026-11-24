# 950PR MFU 基线记录 v1（未优化）

> 日期：2026-11-24 ｜ 硬件：Ascend950PR_9579（28 Cube / 56 Vector，128 GiB）
> 状态：**Stage 4 优化前的基线**；本次只记录，不做任何优化。

## 口径

```
MFU = FLOPs / (T_op × 执行单元峰值)
T_op = 一次算子调用内所有 kernel 的耗时之和（同 stream）
```

| 项 | 取值 |
|---|---|
| 峰值（提交主口径） | Cube bf16/fp16 **378**、fp8/int8 756、fp32 189；Vector bf16/fp16 **47**、fp32 23；HBM **1.6 TB/s** |
| 实测峰值（附表） | Cube 370.5（8192³ matmul）、HBM 1.424（512 MiB copy） |
| 执行单元 | 由 msprof `ArithmeticUtilization` 实测判定（`compute_units.json`） |
| 目标 | 90%（sparse CV 50% / CV 80% / others 90%） |
| MU | 仅诊断，不参与达标 |
| invalid | `achieved > peak × 1.01`，不参与统计 |

## 11 个算子的 MFU（主导 case）

| 算子 | 单元 | 主导 case | AI | T_op | 实测 TFLOP/s | **MFU** | 上限 | MFU/上限 | MFU max | 达标 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| `ChainSpeculativeSamplingFwdOp` | vector | ds-v3-mtp-n3-float32 | 0.8 | 2542.8 µs | 0.0 | **0.04%** | 5.22% | 0.8% | 0.06% | 0/8 |
| `EngramDecodeFwdOp` | vector | serving-b4 | 3.7 | 8.7 µs | 1.0 | **2.06%** | 12.58% | 16.4% | 2.06% | 0/5 |
| `EngramGateConvBwdOp` | vector | train-s96 | 4.3 | 15.0 µs | 1.0 | **2.15%** | 14.47% | 14.9% | 2.15% | 0/4 |
| `EngramGateConvFwdOp` | vector | train-s96 | 2.9 | 11.0 µs | 0.6 | **1.23%** | 9.78% | 12.6% | 1.23% | 0/4 |
| `MLADenseBwdOp` | vector | ds-v3-b2-8k | 2666.7 | 455298.5 µs | 31.5 | **67.01%** | 100.00% | 67.0% | 67.04% | 0/6 |
| `MLAVarlenBwdOp` | vector | ds-v3-mixed | 3611.0 | 1728735.3 µs | 22.1 | **47.05%** | 100.00% | 47.0% | 64.84% | 0/5 |
| `MoEPermuteAlignFwdOp` | vector | kimi-k2-t4096 | 0.0 | 658.4 µs | - | **-** | - | - | - | n/a |
| `PagedKVCacheWriteFwdOp` | vector | qwen3-235b-fp8-bfloat16-float8_e4m3fn | 0.7 | 267.7 µs | 0.0 | **0.07%** | 2.26% | 2.9% | 0.07% | 0/2 |
| `RoPEFwdOp` | vector | llama3-8b-b2-2k | 0.4 | 671.1 µs | 0.1 | **0.41%** | 2.58% | 15.8% | 0.41% | 0/7 |
| `SSDChunkCouplingFwdOp` | vector | mamba2-grouped-prefill | 32.0 | 171.0 µs | 14.1 | **30.06%** | 100.00% | 30.1% | 30.06% | 0/4 |
| `SharedExpertMLPFwdOp` | cube | 512x7168x18432 | 502.8 | 2147.0 µs | 189.1 | **50.02%** | 100.00% | 50.0% | 50.80% | 0/6 |

说明：

- **主导 case** = 该算子计算量（FLOPs）最大的 workload；MoE 的 FLOPs=0，取流量最大者，MFU 记为 n/a；
- `MFU 上限 = AI / (单元峰值 / 1.6 TB/s)`：访存受限的形状，MFU 被带宽锁死，低不是实现问题；
- `MFU max` = 该算子所有有效 case 中的最高值；`达标` = MFU ≥ 90% 的 case 数；
- 当前 **0/11 算子、0/51 个有 MFU 的 case 达到 90%**；最接近的是 MLADense（67.0%）与 SharedExpert（50.0%）。

## 附录：逐 case（69 行）

| 算子 | case | dtype | 单元 | T_op µs | MFU % | 上限 % | 达标 |
|---|---|---|---|---:|---:|---:|:--:|
| `ChainSpeculativeSamplingFwdOp` | chain-sampling-smoke | float32 | vector | 26.9 | 0.00 | 4.51 | no |
| `ChainSpeculativeSamplingFwdOp` | ds-v3-mtp-n1-float32 | float32 | vector | 2529.2 | 0.04 | 5.22 | no |
| `ChainSpeculativeSamplingFwdOp` | ds-v3-mtp-n3-float32 | float32 | vector | 2542.8 | 0.04 | 5.22 | no |
| `ChainSpeculativeSamplingFwdOp` | llama-8b-b16-n4-float32 | float32 | vector | 1245.8 | 0.02 | 5.22 | no |
| `ChainSpeculativeSamplingFwdOp` | llama-8b-b17-n8-float32 | float32 | vector | 1259.7 | 0.02 | 5.22 | no |
| `ChainSpeculativeSamplingFwdOp` | llama-8b-b64-n4-float32 | float32 | vector | 2472.1 | 0.04 | 5.22 | no |
| `ChainSpeculativeSamplingFwdOp` | llama2-7b-b256-n1-float32 | float32 | vector | 1666.3 | 0.06 | 5.22 | no |
| `ChainSpeculativeSamplingFwdOp` | qwen3-235b-eagle-n5-float32 | float32 | vector | 1427.5 | 0.04 | 5.22 | no |
| `EngramDecodeFwdOp` | decode-partial-cache | float16 | vector | 5.9 | 0.39 | 6.58 | no |
| `EngramDecodeFwdOp` | decode-s1 | float16 | vector | 6.1 | 0.09 | 3.31 | no |
| `EngramDecodeFwdOp` | serving-b1 | float16 | vector | 6.3 | 0.18 | 3.33 | no |
| `EngramDecodeFwdOp` | serving-b4 | float16 | vector | 8.7 | 2.06 | 12.58 | no |
| `EngramDecodeFwdOp` | serving-b8 | bfloat16 | vector | 9.5 | 0.95 | 20.90 | no |
| `EngramGateConvBwdOp` | train-s128 | bfloat16 | vector | 8.8 | 0.55 | 14.32 | no |
| `EngramGateConvBwdOp` | train-s32 | float16 | vector | 7.2 | 0.16 | 13.45 | no |
| `EngramGateConvBwdOp` | train-s64 | float16 | vector | 10.2 | 0.93 | 14.29 | no |
| `EngramGateConvBwdOp` | train-s96 | bfloat16 | vector | 15.0 | 2.15 | 14.47 | no |
| `EngramGateConvFwdOp` | train-s128 | bfloat16 | vector | 7.6 | 0.27 | 9.71 | no |
| `EngramGateConvFwdOp` | train-s32 | float16 | vector | 9.2 | 0.05 | 9.35 | no |
| `EngramGateConvFwdOp` | train-s64 | float16 | vector | 8.2 | 0.49 | 9.70 | no |
| `EngramGateConvFwdOp` | train-s96 | bfloat16 | vector | 11.0 | 1.23 | 9.77 | no |
| `MLADenseBwdOp` | ds-v2-lite-3k | bfloat16 | vector | 16625.9 | 64.53 | 100.00 | no |
| `MLADenseBwdOp` | ds-v3-4k | bfloat16 | vector | 58270.8 | 65.46 | 100.00 | no |
| `MLADenseBwdOp` | ds-v3-4k | float16 | vector | 58272.5 | 65.45 | 100.00 | no |
| `MLADenseBwdOp` | ds-v3-b2-8k | bfloat16 | vector | 455298.5 | 67.01 | 100.00 | no |
| `MLADenseBwdOp` | kimi-k2-8k | bfloat16 | vector | 113770.6 | 67.04 | 100.00 | no |
| `MLADenseBwdOp` | mla-dense-bwd-smoke | bfloat16 | vector | 110.4 | 33.81 | 100.00 | no |
| `MLAVarlenBwdOp` | ds-v3-8x4k | bfloat16 | vector | 490592.4 | 62.20 | 100.00 | no |
| `MLAVarlenBwdOp` | ds-v3-8x4k | float16 | vector | 490626.6 | 62.19 | 100.00 | no |
| `MLAVarlenBwdOp` | ds-v3-mixed | bfloat16 | vector | 1728735.3 | 47.05 | 100.00 | no |
| `MLAVarlenBwdOp` | kimi-k2-4x8k | bfloat16 | vector | 470544.9 | 64.84 | 100.00 | no |
| `MLAVarlenBwdOp` | mla-varlen-bwd-smoke | bfloat16 | vector | 130.2 | 23.33 | 100.00 | no |
| `MoEPermuteAlignFwdOp` | ds-v3-t1 | int32 | vector | 21.2 | - | - | n/a |
| `MoEPermuteAlignFwdOp` | ds-v3-t32 | int32 | vector | 37.5 | - | - | n/a |
| `MoEPermuteAlignFwdOp` | ds-v3-t4096 | int32 | vector | 939.0 | - | - | n/a |
| `MoEPermuteAlignFwdOp` | ds-v3-t512 | int32 | vector | 153.2 | - | - | n/a |
| `MoEPermuteAlignFwdOp` | kimi-k2-t1 | int32 | vector | 26.5 | - | - | n/a |
| `MoEPermuteAlignFwdOp` | kimi-k2-t32 | int32 | vector | 43.3 | - | - | n/a |
| `MoEPermuteAlignFwdOp` | kimi-k2-t4096 | int32 | vector | 658.4 | - | - | n/a |
| `MoEPermuteAlignFwdOp` | kimi-k2-t512 | int32 | vector | 129.2 | - | - | n/a |
| `MoEPermuteAlignFwdOp` | qwen3-235b-t1 | int32 | vector | 15.2 | - | - | n/a |
| `MoEPermuteAlignFwdOp` | qwen3-235b-t32 | int32 | vector | 40.3 | - | - | n/a |
| `MoEPermuteAlignFwdOp` | qwen3-235b-t4096 | int32 | vector | 1731.1 | - | - | n/a |
| `MoEPermuteAlignFwdOp` | qwen3-235b-t512 | int32 | vector | 250.9 | - | - | n/a |
| `PagedKVCacheWriteFwdOp` | gpt-oss-120b-bfloat16 | bfloat16 | vector | 25.8 | - | - | n/a |
| `PagedKVCacheWriteFwdOp` | llama-8b-decode-bfloat16 | bfloat16 | vector | 15.2 | - | - | n/a |
| `PagedKVCacheWriteFwdOp` | llama-8b-decode-float16 | float16 | vector | 15.4 | - | - | n/a |
| `PagedKVCacheWriteFwdOp` | llama-8b-prefill-bfloat16 | bfloat16 | vector | 884.7 | - | - | n/a |
| `PagedKVCacheWriteFwdOp` | odd-tokens-and-page-float16 | float16 | vector | 6.5 | - | - | n/a |
| `PagedKVCacheWriteFwdOp` | paged-write-smoke-float16 | float16 | vector | 5.1 | - | - | n/a |
| `PagedKVCacheWriteFwdOp` | qwen3-235b-fp8-bfloat16-float8_e4m3fn | bfloat16 | vector | 267.7 | 0.07 | 2.26 | no |
| `PagedKVCacheWriteFwdOp` | qwen3-235b-fp8-fp16-float16-float8_e4m3fn | float16 | vector | 78.9 | 0.06 | 2.26 | no |
| `RoPEFwdOp` | gptj-6b-decode-b32 | float16 | vector | 10.3 | 0.04 | 0.64 | no |
| `RoPEFwdOp` | gptj-6b-prefill | float16 | vector | 330.0 | 0.08 | 0.64 | no |
| `RoPEFwdOp` | llama3-8b-b2-2k | float32 | vector | 671.1 | 0.41 | 2.58 | no |
| `RoPEFwdOp` | llama3-8b-decode-b64 | bfloat16 | vector | 10.9 | 0.19 | 2.52 | no |
| `RoPEFwdOp` | llama3-8b-prefill-4k | bfloat16 | vector | 430.4 | 0.31 | 2.52 | no |
| `RoPEFwdOp` | llama3-8b-prefill-4k | float16 | vector | 429.8 | 0.31 | 2.52 | no |
| `RoPEFwdOp` | phi-2-query-only | bfloat16 | vector | 141.3 | 0.05 | 1.01 | no |
| `SSDChunkCouplingFwdOp` | mamba2-2p7b | bfloat16 | vector | 22.9 | 24.93 | 100.00 | no |
| `SSDChunkCouplingFwdOp` | mamba2-780m | float16 | vector | 12.6 | 22.59 | 100.00 | no |
| `SSDChunkCouplingFwdOp` | mamba2-grouped-prefill | float16 | vector | 171.0 | 30.06 | 100.00 | no |
| `SSDChunkCouplingFwdOp` | ssd-chunk-coupling-smoke | float16 | vector | 2.7 | 0.42 | 36.31 | no |
| `SharedExpertMLPFwdOp` | 1x7168x18432 | bfloat16 | cube | 723.0 | 0.29 | 0.42 | no |
| `SharedExpertMLPFwdOp` | 32x5120x1536 | bfloat16 | cube | 59.9 | 6.67 | 13.36 | no |
| `SharedExpertMLPFwdOp` | 32x7168x2048 | bfloat16 | cube | 105.8 | 7.05 | 13.41 | no |
| `SharedExpertMLPFwdOp` | 32x7168x2048 | float16 | cube | 102.8 | 7.26 | 13.41 | no |
| `SharedExpertMLPFwdOp` | 4096x7168x2048 | bfloat16 | cube | 1879.0 | 50.80 | 100.00 | no |
| `SharedExpertMLPFwdOp` | 512x7168x18432 | bfloat16 | cube | 2147.0 | 50.02 | 100.00 | no |

## 溯源

- 数据表：`examples/batch2/mfu_table.json`（sha256 `4bf759127a7d4d05…`）
- 峰值策略：`tileops/reporting/peaks.json`、`compute_units.json`
- 报告：`examples/TileOPs/reports/tileops/`（6 个 2026-11-24 重跑 + 5 个既有通过）
- 逐 case CSV：`examples/batch2/mfu_cases.csv`；算子级 CSV：`mfu_operators.csv`

> 本记录为优化前基线；后续 Stage 4 的结果应另写 v2，不改写本文件。
