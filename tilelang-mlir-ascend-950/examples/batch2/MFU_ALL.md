# Batch2 — whole-operator time (T_op) and MFU / MU

Every case is measured as **one operator call**.  `T_op` sums every kernel the call launches (host `Pad*` ops excluded); `reported` is the number the TileOPs reporter wrote, which is often a single filtered kernel and therefore under-reports multi-kernel operators.

Peaks: **378.0 TFLOPS** Cube, **47 TFLOPS** Vector (bf16/fp16), **1.6 TB/s** bandwidth.  Machine balance is **236 FLOP/byte** for Cube and **29.4 FLOP/byte** for Vector.  MFU uses the peak of the unit the kernel actually runs on (vendor values; measured 370.5 / 1.424 is the supplementary column).

**Acceptance metric is MFU** (sparse CV 50% / CV 80% / others 90%).  MU is reported as a diagnostic only and never decides pass/fail.  `MFU 上限` = `AI / (unit peak / 1.6 TB/s)` is the physical ceiling bandwidth imposes on a memory-bound shape, so a tiny MFU there is not a kernel defect.

Peak provenance: Ascend950PR_9579 has 28 cube cores; the dense bf16/fp16 28 Cube cores x 4096 MAC/cycle x 2 x 1.65 GHz = 378 TFLOPS, which matches the vendor 28-core SKU column exactly.

## How T_op was obtained

- **recapture** — this session re-ran the case with `TILEOPS_MSPROF_KERNEL_NAME=__force_unfiltered__`, kept the artifact and summed every kernel the single profiled call launched (`Pad*` excluded).
- **report** — the case's stored msprof artifact already covered the whole call; `T_op` is the sum of its `OpBasicInfo*.csv`, or (when the reporter kept no artifact) the reporter's own `latency_us` for a validated capture.
- **blocked** — no usable measurement; `T_op` is left empty, never 0.

A single call can launch a kernel more than once under one name; those samples are all present and are summed (verified on `EngramGateConvBwdOp`, which launches `main_kernel` twice).

## msprof failures: root cause

The earlier `msprof profiling failed (...); falling back to events` runs were not a profiler or toolchain defect.  The container's root overlay (`/`) is a **5 GB** filesystem and was **100% full (0 bytes available)**: stale `/tmp/tileops_msprof_*` scratch dirs from previous captures had accumulated to ~9.7 GB (single dirs up to 4.7 GB).  msprof stages its raw trace there, so it died with `No space left on device`; the reporter then fell back to events, and the large cases produced an empty artifact dir.

After removing the stale scratch dirs and pointing `TMPDIR` at `/workspace` (297 GB free), the exact previously-failing command `bench_mla_dense_bwd.py::test_mla_dense_bwd_bench[ds-v3-b2-8k-bfloat16]` returned rc=0 in 95.5 s with a complete unfiltered trace.  Every re-captured case then succeeded.

## Invalid rows

Rows with achieved throughput above the cube peak are **not** efficiency figures and are excluded from every max/达标 statistic (`invalid`).

None in this run.  The 2026-11-24 head-slab fix removed the only known case (`MLAVarlenBwdOp`/`ds-v3-8x4k`, grid 66,560): it now runs 4 x 16,640 blocks and reports a real ~490 ms.  The trap and its evidence remain documented in `docs/950pr-mfu-acceptance.md` and `TRAP-950-grid-over-65535`.

## Per-operator summary

| operator | unit | type | binding | AI max | cases | blocked | MFU(主导) | MFU 上限 | MFU max | MU max (诊断) | 达标(MFU) | case 达标 | notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---|
| `MLADenseBwdOp` | cube | compute-bound | compute | 2666.7 | 6 | 0 | 40.30% | 100.00% | 40.30% | 8.51% | no | 0/6 | report under-reports |
| `MLAVarlenBwdOp` | cube | compute-bound | compute | 3611.0 | 5 | 0 | 25.12% | 100.00% | 28.23% | 6.30% | no | 0/5 | report under-reports |
| `MoEPermuteAlignFwdOp` | vector | memory-bound | bandwidth | 0.0 | 12 | 0 | n/a | n/a | n/a | 0.83% | no | 0/12 | report under-reports |
| `SharedExpertMLPFwdOp` | cube | compute-bound | compute | 1755.7 | 6 | 0 | 91.32% | 100.00% | 91.32% | 92.99% | yes | 1/6 | report under-reports |
| `EngramGateConvFwdOp` | vector | memory-bound | bandwidth | 2.9 | 4 | 0 | 1.54% | 9.77% | 1.54% | 15.75% | no | 0/4 | - |
| `PagedKVCacheWriteFwdOp` | vector | memory-bound | bandwidth | 0.7 | 8 | 0 | 0.91% | 2.26% | 0.91% | 97.26% | no | 0/8 | - |
| `RoPEFwdOp` | vector | memory-bound | bandwidth | 0.7 | 7 | 0 | 2.28% | 2.58% | 2.28% | 88.41% | no | 0/7 | - |
| `EngramDecodeFwdOp` | vector | memory-bound | bandwidth | 6.1 | 5 | 0 | 2.20% | 12.58% | 2.20% | 17.48% | no | 0/5 | report under-reports |
| `EngramGateConvBwdOp` | vector | memory-bound | bandwidth | 4.3 | 4 | 0 | 2.61% | 14.47% | 2.61% | 18.01% | no | 0/4 | - |
| `SSDChunkCouplingFwdOp` | vector | compute-bound | compute | 32.0 | 4 | 0 | 94.03% | 100.00% | 94.03% | 86.32% | yes | 1/4 | - |
| `ChainSpeculativeSamplingFwdOp` | vector | memory-bound | bandwidth | 0.7 | 8 | 0 | 0.31% | 5.22% | 0.34% | 6.61% | no | 0/8 | report under-reports |

`invalid` rows (grid above the 65,535-block limit) are excluded from every max / 达标 statistic; see the section above for the evidence.

## All cases

| operator | case | dtype | unit | reported µs | T_op µs | source | FLOPs | Bytes | AI | MFU % | MFU 上限 | 达标(MFU≥90%) | MU % | note |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|---:|---|
| `MLADenseBwdOp` | ds-v2-lite-3k | bfloat16 | cube | 71.799 | 3703.044 | report | 504235524096 | 504102912 | 1000.3 | 36.02 | 100.00 | no | 8.51 | report-underreports |
| `MLADenseBwdOp` | ds-v3-4k | bfloat16 | cube | 217.260 | 13158.864 | report | 1792646840320 | 1344274432 | 1333.5 | 36.04 | 100.00 | no | 6.38 | report-underreports |
| `MLADenseBwdOp` | ds-v3-4k | float16 | cube | 215.627 | 13168.978 | report | 1792646840320 | 1344274432 | 1333.5 | 36.01 | 100.00 | no | 6.38 | report-underreports |
| `MLADenseBwdOp` | ds-v3-b2-8k | bfloat16 | cube | 964.070 | 94130.114 | report | 14338887778304 | 5377097728 | 2666.7 | 40.30 | 100.00 | no | 3.57 | report-underreports |
| `MLADenseBwdOp` | kimi-k2-8k | bfloat16 | cube | 216.962 | 23776.831 | report | 3584721944576 | 1344274432 | 2666.7 | 39.89 | 100.00 | no | 3.53 | report-underreports |
| `MLADenseBwdOp` | mla-dense-bwd-smoke | bfloat16 | cube | 4.413 | 110.412 | recapture | 1754540032 | 10502144 | 167.1 | 4.20 | 70.72 | no | 5.94 | report-underreports |
| `MLAVarlenBwdOp` | ds-v3-8x4k | bfloat16 | cube | 31880.373 | 134392.538 | report | 14341174722560 | 10754195528 | 1333.5 | 28.23 | 100.00 | no | 5.00 | report-underreports |
| `MLAVarlenBwdOp` | ds-v3-8x4k | float16 | cube | 31879.594 | 134374.732 | report | 14341174722560 | 10754195528 | 1333.5 | 28.23 | 100.00 | no | 5.00 | report-underreports |
| `MLAVarlenBwdOp` | ds-v3-mixed | bfloat16 | cube | 98835.672 | 402647.725 | report | 38226104254464 | 10586161208 | 3611.0 | 25.12 | 100.00 | no | 1.64 | report-underreports |
| `MLAVarlenBwdOp` | kimi-k2-4x8k | bfloat16 | cube | 1664.005 | 140154.597 | report | 14338887778304 | 5377097768 | 2666.7 | 27.07 | 100.00 | no | 2.40 | report-underreports |
| `MLAVarlenBwdOp` | mla-varlen-bwd-smoke | bfloat16 | cube | 5.786 | 130.201 | recapture | 1427517952 | 13127704 | 108.7 | 2.90 | 46.03 | no | 6.30 | report-underreports |
| `MoEPermuteAlignFwdOp` | ds-v3-t1 | int32 | vector | 7.044 | 11.207 | report | 0 | 65848 | 0.0 | - | - | n/a | 0.37 | - |
| `MoEPermuteAlignFwdOp` | ds-v3-t32 | int32 | vector | 6.776 | 11.428 | report | 0 | 67844 | 0.0 | - | - | n/a | 0.37 | - |
| `MoEPermuteAlignFwdOp` | ds-v3-t4096 | int32 | vector | 9.459 | 26.340 | report | 0 | 329972 | 0.0 | - | - | n/a | 0.78 | report-underreports |
| `MoEPermuteAlignFwdOp` | ds-v3-t512 | int32 | vector | 5.435 | 17.203 | report | 0 | 98804 | 0.0 | - | - | n/a | 0.36 | report-underreports |
| `MoEPermuteAlignFwdOp` | kimi-k2-t1 | int32 | vector | 6.976 | 11.491 | report | 0 | 98608 | 0.0 | - | - | n/a | 0.54 | - |
| `MoEPermuteAlignFwdOp` | kimi-k2-t32 | int32 | vector | 7.055 | 11.978 | report | 0 | 100604 | 0.0 | - | - | n/a | 0.52 | - |
| `MoEPermuteAlignFwdOp` | kimi-k2-t4096 | int32 | vector | 9.593 | 27.340 | report | 0 | 362732 | 0.0 | - | - | n/a | 0.83 | report-underreports |
| `MoEPermuteAlignFwdOp` | kimi-k2-t512 | int32 | vector | 5.264 | 17.195 | report | 0 | 131564 | 0.0 | - | - | n/a | 0.48 | report-underreports |
| `MoEPermuteAlignFwdOp` | qwen3-235b-t1 | int32 | vector | 6.474 | 10.079 | report | 0 | 33088 | 0.0 | - | - | n/a | 0.21 | - |
| `MoEPermuteAlignFwdOp` | qwen3-235b-t32 | int32 | vector | 6.522 | 10.510 | report | 0 | 35084 | 0.0 | - | - | n/a | 0.21 | - |
| `MoEPermuteAlignFwdOp` | qwen3-235b-t4096 | int32 | vector | 9.567 | 25.382 | report | 0 | 297212 | 0.0 | - | - | n/a | 0.73 | report-underreports |
| `MoEPermuteAlignFwdOp` | qwen3-235b-t512 | int32 | vector | 5.385 | 16.827 | report | 0 | 66044 | 0.0 | - | - | n/a | 0.25 | report-underreports |
| `SharedExpertMLPFwdOp` | 1x7168x18432 | bfloat16 | cube | 174.854 | 532.831 | report | 792834048 | 792752128 | 1.0 | 0.39 | 0.42 | no | 92.99 | report-underreports |
| `SharedExpertMLPFwdOp` | 32x5120x1536 | bfloat16 | cube | 26.980 | 40.150 | report | 1510244352 | 47841280 | 31.6 | 9.95 | 13.36 | no | 74.47 | - |
| `SharedExpertMLPFwdOp` | 32x7168x2048 | bfloat16 | cube | 43.548 | 64.657 | report | 2818965504 | 88997888 | 31.7 | 11.53 | 13.41 | no | 86.03 | - |
| `SharedExpertMLPFwdOp` | 32x7168x2048 | float16 | cube | 43.669 | 65.292 | report | 2818965504 | 88997888 | 31.7 | 11.42 | 13.41 | no | 85.19 | - |
| `SharedExpertMLPFwdOp` | 4096x7168x2048 | bfloat16 | cube | 736.832 | 1107.166 | report | 360827584512 | 205520896 | 1755.7 | 86.22 | 100.00 | no | 11.60 | - |
| `SharedExpertMLPFwdOp` | 512x7168x18432 | bfloat16 | cube | 808.446 | 1175.983 | report | 405931032576 | 807403520 | 502.8 | 91.32 | 100.00 | yes | 42.91 | - |
| `EngramGateConvFwdOp` | train-s128 | bfloat16 | vector | 5.609 | 5.609 | report | 948992 | 332800 | 2.9 | 0.36 | 9.71 | no | 3.71 | - |
| `EngramGateConvFwdOp` | train-s32 | float16 | vector | 9.194 | 9.194 | recapture | 234944 | 85504 | 2.7 | 0.05 | 9.35 | no | 0.58 | - |
| `EngramGateConvFwdOp` | train-s64 | float16 | vector | 6.594 | 6.594 | report | 1890048 | 663552 | 2.8 | 0.61 | 9.70 | no | 6.29 | - |
| `EngramGateConvFwdOp` | train-s96 | bfloat16 | vector | 8.834 | 8.834 | report | 6390720 | 2225664 | 2.9 | 1.54 | 9.77 | no | 15.75 | - |
| `PagedKVCacheWriteFwdOp` | gpt-oss-120b-bfloat16 | bfloat16 | vector | 4.889 | 4.889 | report | 0 | 1050624 | 0.0 | - | - | n/a | 13.43 | - |
| `PagedKVCacheWriteFwdOp` | llama-8b-decode-bfloat16 | bfloat16 | vector | 3.117 | 3.117 | report | 0 | 524800 | 0.0 | - | - | n/a | 10.52 | - |
| `PagedKVCacheWriteFwdOp` | llama-8b-decode-float16 | float16 | vector | 3.351 | 3.351 | report | 0 | 524800 | 0.0 | - | - | n/a | 9.79 | - |
| `PagedKVCacheWriteFwdOp` | llama-8b-prefill-bfloat16 | bfloat16 | vector | 43.166 | 43.166 | report | 0 | 67174400 | 0.0 | - | - | n/a | 97.26 | - |
| `PagedKVCacheWriteFwdOp` | odd-tokens-and-page-float16 | float16 | vector | 2.212 | 2.212 | report | 0 | 7224 | 0.0 | - | - | n/a | 0.20 | - |
| `PagedKVCacheWriteFwdOp` | paged-write-smoke-float16 | float16 | vector | 5.070 | 5.070 | recapture | 0 | 5160 | 0.0 | - | - | n/a | 0.06 | - |
| `PagedKVCacheWriteFwdOp` | qwen3-235b-fp8-bfloat16-float8_e4m3fn | bfloat16 | vector | 19.621 | 19.621 | report | 8388608 | 12615688 | 0.7 | 0.91 | 2.26 | no | 40.19 | - |
| `PagedKVCacheWriteFwdOp` | qwen3-235b-fp8-fp16-float16-float8_e4m3fn | float16 | vector | 7.315 | 7.315 | report | 2097152 | 3153928 | 0.7 | 0.61 | 2.26 | no | 26.95 | - |
| `RoPEFwdOp` | gptj-6b-decode-b32 | float16 | vector | 4.967 | 4.967 | report | 196608 | 1052928 | 0.2 | 0.08 | 0.64 | no | 13.25 | - |
| `RoPEFwdOp` | gptj-6b-prefill | float16 | vector | 78.350 | 78.350 | report | 12582912 | 67387392 | 0.2 | 0.34 | 0.64 | no | 53.76 | - |
| `RoPEFwdOp` | llama3-8b-b2-2k | float32 | vector | 120.106 | 120.106 | report | 62914560 | 169902080 | 0.4 | 2.28 | 2.58 | no | 88.41 | - |
| `RoPEFwdOp` | llama3-8b-decode-b64 | bfloat16 | vector | 6.132 | 6.132 | report | 983040 | 1327616 | 0.7 | 0.34 | 2.52 | no | 13.53 | - |
| `RoPEFwdOp` | llama3-8b-prefill-4k | bfloat16 | vector | 76.060 | 76.060 | report | 62914560 | 84967424 | 0.7 | 1.76 | 2.52 | no | 69.82 | - |
| `RoPEFwdOp` | llama3-8b-prefill-4k | float16 | vector | 75.321 | 75.321 | report | 62914560 | 84967424 | 0.7 | 1.78 | 2.52 | no | 70.50 | - |
| `RoPEFwdOp` | phi-2-query-only | bfloat16 | vector | 12.546 | 12.546 | report | 3072000 | 10308000 | 0.3 | 0.52 | 1.01 | no | 51.35 | - |
| `EngramDecodeFwdOp` | decode-partial-cache | float16 | vector | 2.911 | 8.891 | report | 1060904 | 548864 | 1.9 | 0.25 | 6.58 | no | 3.86 | report-underreports |
| `EngramDecodeFwdOp` | decode-s1 | float16 | vector | 3.142 | 6.089 | recapture | 268308 | 275968 | 1.0 | 0.09 | 3.31 | no | 2.83 | - |
| `EngramDecodeFwdOp` | serving-b1 | float16 | vector | 3.067 | 5.741 | report | 530452 | 541696 | 1.0 | 0.20 | 3.33 | no | 5.90 | - |
| `EngramDecodeFwdOp` | serving-b4 | float16 | vector | 4.605 | 8.165 | report | 8437840 | 2283520 | 3.7 | 2.20 | 12.58 | no | 17.48 | - |
| `EngramDecodeFwdOp` | serving-b8 | bfloat16 | vector | 3.195 | 6.094 | report | 4243616 | 691200 | 6.1 | 1.48 | 20.90 | no | 7.09 | - |
| `EngramGateConvBwdOp` | train-s128 | bfloat16 | vector | 4.873 | 7.139 | report | 2253056 | 535552 | 4.2 | 0.67 | 14.32 | no | 4.69 | - |
| `EngramGateConvBwdOp` | train-s32 | float16 | vector | 4.805 | 7.223 | recapture | 556352 | 140800 | 4.0 | 0.16 | 13.45 | no | 1.22 | - |
| `EngramGateConvBwdOp` | train-s64 | float16 | vector | 6.093 | 8.809 | report | 4486400 | 1069056 | 4.2 | 1.08 | 14.29 | no | 7.58 | - |
| `EngramGateConvBwdOp` | train-s96 | bfloat16 | vector | 9.803 | 12.393 | report | 15181632 | 3571200 | 4.3 | 2.61 | 14.47 | no | 18.01 | - |
| `SSDChunkCouplingFwdOp` | mamba2-2p7b | bfloat16 | vector | 11.808 | 11.808 | report | 268435456 | 8388608 | 32.0 | 48.37 | 100.00 | no | 44.40 | - |
| `SSDChunkCouplingFwdOp` | mamba2-780m | float16 | vector | 6.160 | 6.160 | report | 134217728 | 4194304 | 32.0 | 46.36 | 100.00 | no | 42.56 | - |
| `SSDChunkCouplingFwdOp` | mamba2-grouped-prefill | float16 | vector | 54.664 | 54.664 | report | 2415919104 | 75497472 | 32.0 | 94.03 | 100.00 | yes | 86.32 | - |
| `SSDChunkCouplingFwdOp` | ssd-chunk-coupling-smoke | float16 | vector | 2.680 | 2.680 | recapture | 524288 | 49152 | 10.7 | 0.42 | 36.31 | no | 1.15 | - |
| `ChainSpeculativeSamplingFwdOp` | chain-sampling-smoke | float32 | vector | 26.862 | 26.862 | recapture | 394 | 608 | 0.6 | 0.00 | 4.51 | no | 0.00 | - |
| `ChainSpeculativeSamplingFwdOp` | ds-v3-mtp-n1-float32 | float32 | vector | 83.064 | 313.106 | report | 24821952 | 33097232 | 0.7 | 0.34 | 5.22 | no | 6.61 | report-underreports |
| `ChainSpeculativeSamplingFwdOp` | ds-v3-mtp-n3-float32 | float32 | vector | 86.249 | 346.093 | report | 24822208 | 33099280 | 0.7 | 0.31 | 5.22 | no | 5.98 | report-underreports |
| `ChainSpeculativeSamplingFwdOp` | llama-8b-b16-n4-float32 | float32 | vector | 43.518 | 140.193 | report | 6156432 | 8209552 | 0.7 | 0.19 | 5.22 | no | 3.66 | report-underreports |
| `ChainSpeculativeSamplingFwdOp` | llama-8b-b17-n8-float32 | float32 | vector | 44.225 | 155.384 | report | 6541345 | 8723736 | 0.7 | 0.18 | 5.22 | no | 3.51 | report-underreports |
| `ChainSpeculativeSamplingFwdOp` | llama-8b-b64-n4-float32 | float32 | vector | 84.848 | 393.618 | report | 24625728 | 32838160 | 0.7 | 0.27 | 5.22 | no | 5.21 | report-underreports |
| `ChainSpeculativeSamplingFwdOp` | llama2-7b-b256-n1-float32 | float32 | vector | 191.371 | 399.591 | report | 24576768 | 32774160 | 0.7 | 0.27 | 5.22 | no | 5.13 | report-underreports |
| `ChainSpeculativeSamplingFwdOp` | qwen3-235b-eagle-n5-float32 | float32 | vector | 47.284 | 374.337 | report | 14586208 | 19450640 | 0.7 | 0.17 | 5.22 | no | 3.25 | report-underreports |

## Per-operator conclusion

### `MLADenseBwdOp`

- **类型**: compute-bound (max AI 2666.7 FLOP/byte vs the cube balance 236.2).
- **受限资源**: compute — on the dominant case `ds-v3-b2-8k` MFU 40.30%, MU 3.57% (operator maxima: MFU 40.30%, MU 8.51%).
- **是否达标**: 否 (MFU 40.30% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `MLAVarlenBwdOp`

- **类型**: compute-bound (max AI 3611.0 FLOP/byte vs the cube balance 236.2).
- **受限资源**: compute — on the dominant case `ds-v3-mixed` MFU 25.12%, MU 1.64% (operator maxima: MFU 28.23%, MU 6.30%).
- **是否达标**: 否 (MFU 25.12% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `MoEPermuteAlignFwdOp`

- **类型**: memory-bound (max AI 0.0 FLOP/byte vs the vector balance 29.4).
- **受限资源**: bandwidth — on the dominant case `kimi-k2-t4096` MFU -%, MU 0.83% (operator maxima: MFU -%, MU 0.83%).
- **是否达标**: 否 (MFU -% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `SharedExpertMLPFwdOp`

- **类型**: compute-bound (max AI 1755.7 FLOP/byte vs the cube balance 236.2).
- **受限资源**: compute — on the dominant case `512x7168x18432` MFU 91.32%, MU 42.91% (operator maxima: MFU 91.32%, MU 92.99%).
- **是否达标**: 是 (MFU 91.32% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `EngramGateConvFwdOp`

- **类型**: memory-bound (max AI 2.9 FLOP/byte vs the vector balance 29.4).
- **受限资源**: bandwidth — on the dominant case `train-s96` MFU 1.54%, MU 15.75% (operator maxima: MFU 1.54%, MU 15.75%).
- **是否达标**: 否 (MFU 1.54% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `PagedKVCacheWriteFwdOp`

- **类型**: memory-bound (max AI 0.7 FLOP/byte vs the vector balance 29.4).
- **受限资源**: bandwidth — on the dominant case `qwen3-235b-fp8-bfloat16-float8_e4m3fn` MFU 0.91%, MU 40.19% (operator maxima: MFU 0.91%, MU 97.26%).
- **是否达标**: 否 (MFU 0.91% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `RoPEFwdOp`

- **类型**: memory-bound (max AI 0.7 FLOP/byte vs the vector balance 29.4).
- **受限资源**: bandwidth — on the dominant case `llama3-8b-b2-2k` MFU 2.28%, MU 88.41% (operator maxima: MFU 2.28%, MU 88.41%).
- **是否达标**: 否 (MFU 2.28% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `EngramDecodeFwdOp`

- **类型**: memory-bound (max AI 6.1 FLOP/byte vs the vector balance 29.4).
- **受限资源**: bandwidth — on the dominant case `serving-b4` MFU 2.20%, MU 17.48% (operator maxima: MFU 2.20%, MU 17.48%).
- **是否达标**: 否 (MFU 2.20% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `EngramGateConvBwdOp`

- **类型**: memory-bound (max AI 4.3 FLOP/byte vs the vector balance 29.4).
- **受限资源**: bandwidth — on the dominant case `train-s96` MFU 2.61%, MU 18.01% (operator maxima: MFU 2.61%, MU 18.01%).
- **是否达标**: 否 (MFU 2.61% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `SSDChunkCouplingFwdOp`

- **类型**: compute-bound (max AI 32.0 FLOP/byte vs the vector balance 29.4).
- **受限资源**: compute — on the dominant case `mamba2-grouped-prefill` MFU 94.03%, MU 86.32% (operator maxima: MFU 94.03%, MU 86.32%).
- **是否达标**: 是 (MFU 94.03% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `ChainSpeculativeSamplingFwdOp`

- **类型**: memory-bound (max AI 0.7 FLOP/byte vs the vector balance 29.4).
- **受限资源**: bandwidth — on the dominant case `ds-v3-mtp-n3-float32` MFU 0.31%, MU 5.98% (operator maxima: MFU 0.34%, MU 6.61%).
- **是否达标**: 否 (MFU 0.31% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.
