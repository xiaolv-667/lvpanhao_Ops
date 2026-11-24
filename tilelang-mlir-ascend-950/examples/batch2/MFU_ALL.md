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
| `ChainSpeculativeSamplingFwdOp` | vector | memory-bound | bandwidth | 0.7 | 8 | 0 | 0.04% | 5.22% | 0.06% | 1.23% | no | 0/8 | - |
| `EngramDecodeFwdOp` | vector | memory-bound | bandwidth | 6.1 | 5 | 0 | 2.06% | 12.58% | 2.06% | 16.35% | no | 0/5 | report under-reports |
| `EngramGateConvBwdOp` | vector | memory-bound | bandwidth | 4.3 | 4 | 0 | 2.15% | 14.47% | 2.15% | 14.86% | no | 0/4 | - |
| `EngramGateConvFwdOp` | vector | memory-bound | bandwidth | 2.9 | 4 | 0 | 1.23% | 9.77% | 1.23% | 12.59% | no | 0/4 | - |
| `MLADenseBwdOp` | vector | compute-bound | compute | 2666.7 | 6 | 0 | 67.01% | 100.00% | 67.04% | 5.94% | no | 0/6 | report under-reports |
| `MLAVarlenBwdOp` | vector | compute-bound | compute | 3611.0 | 5 | 0 | 47.05% | 100.00% | 64.84% | 6.30% | no | 0/5 | report under-reports |
| `MoEPermuteAlignFwdOp` | vector | memory-bound | bandwidth | 0.0 | 12 | 0 | n/a | n/a | n/a | 0.23% | no | 0/12 | report under-reports |
| `PagedKVCacheWriteFwdOp` | vector | memory-bound | bandwidth | 0.7 | 8 | 0 | 0.07% | 2.26% | 0.07% | 4.75% | no | 0/8 | - |
| `RoPEFwdOp` | vector | memory-bound | bandwidth | 0.7 | 7 | 0 | 0.41% | 2.58% | 0.41% | 15.82% | no | 0/7 | - |
| `SSDChunkCouplingFwdOp` | vector | compute-bound | compute | 32.0 | 4 | 0 | 30.06% | 100.00% | 30.06% | 27.59% | no | 0/4 | - |
| `SharedExpertMLPFwdOp` | cube | compute-bound | compute | 1755.7 | 6 | 0 | 50.02% | 100.00% | 50.80% | 68.53% | no | 0/6 | report under-reports |

`invalid` rows (grid above the 65,535-block limit) are excluded from every max / 达标 statistic; see the section above for the evidence.

## All cases

| operator | case | dtype | unit | reported µs | T_op µs | source | FLOPs | Bytes | AI | MFU % | MFU 上限 | 达标(MFU≥90%) | MU % | note |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|---:|---|
| `ChainSpeculativeSamplingFwdOp` | chain-sampling-smoke | float32 | vector | 26.862 | 26.862 | recapture | 394 | 608 | 0.6 | 0.00 | 4.51 | no | 0.00 | - |
| `ChainSpeculativeSamplingFwdOp` | ds-v3-mtp-n1-float32 | float32 | vector | 2529.246 | 2529.246 | report | 24821952 | 33097232 | 0.7 | 0.04 | 5.22 | no | 0.82 | - |
| `ChainSpeculativeSamplingFwdOp` | ds-v3-mtp-n3-float32 | float32 | vector | 2542.791 | 2542.791 | report | 24822208 | 33099280 | 0.7 | 0.04 | 5.22 | no | 0.81 | - |
| `ChainSpeculativeSamplingFwdOp` | llama-8b-b16-n4-float32 | float32 | vector | 1245.809 | 1245.809 | report | 6156432 | 8209552 | 0.7 | 0.02 | 5.22 | no | 0.41 | - |
| `ChainSpeculativeSamplingFwdOp` | llama-8b-b17-n8-float32 | float32 | vector | 1259.744 | 1259.744 | report | 6541345 | 8723736 | 0.7 | 0.02 | 5.22 | no | 0.43 | - |
| `ChainSpeculativeSamplingFwdOp` | llama-8b-b64-n4-float32 | float32 | vector | 2472.091 | 2472.091 | report | 24625728 | 32838160 | 0.7 | 0.04 | 5.22 | no | 0.83 | - |
| `ChainSpeculativeSamplingFwdOp` | llama2-7b-b256-n1-float32 | float32 | vector | 1666.341 | 1666.341 | report | 24576768 | 32774160 | 0.7 | 0.06 | 5.22 | no | 1.23 | - |
| `ChainSpeculativeSamplingFwdOp` | qwen3-235b-eagle-n5-float32 | float32 | vector | 1427.486 | 1427.486 | report | 14586208 | 19450640 | 0.7 | 0.04 | 5.22 | no | 0.85 | - |
| `EngramDecodeFwdOp` | decode-partial-cache | float16 | vector | 3.092 | 5.854 | recapture | 1060904 | 548864 | 1.9 | 0.39 | 6.58 | no | 5.86 | - |
| `EngramDecodeFwdOp` | decode-s1 | float16 | vector | 3.142 | 6.089 | recapture | 268308 | 275968 | 1.0 | 0.09 | 3.31 | no | 2.83 | - |
| `EngramDecodeFwdOp` | serving-b1 | float16 | vector | 3.141 | 6.305 | recapture | 530452 | 541696 | 1.0 | 0.18 | 3.33 | no | 5.37 | report-underreports |
| `EngramDecodeFwdOp` | serving-b4 | float16 | vector | 4.579 | 8.727 | recapture | 8437840 | 2283520 | 3.7 | 2.06 | 12.58 | no | 16.35 | - |
| `EngramDecodeFwdOp` | serving-b8 | bfloat16 | vector | 6.385 | 9.544 | recapture | 4243616 | 691200 | 6.1 | 0.95 | 20.90 | no | 4.53 | - |
| `EngramGateConvBwdOp` | train-s128 | bfloat16 | vector | 6.238 | 8.779 | recapture | 2253056 | 535552 | 4.2 | 0.55 | 14.32 | no | 3.81 | - |
| `EngramGateConvBwdOp` | train-s32 | float16 | vector | 4.805 | 7.223 | recapture | 556352 | 140800 | 4.0 | 0.16 | 13.45 | no | 1.22 | - |
| `EngramGateConvBwdOp` | train-s64 | float16 | vector | 6.997 | 10.245 | recapture | 4486400 | 1069056 | 4.2 | 0.93 | 14.29 | no | 6.52 | - |
| `EngramGateConvBwdOp` | train-s96 | bfloat16 | vector | 11.402 | 15.021 | recapture | 15181632 | 3571200 | 4.3 | 2.15 | 14.47 | no | 14.86 | - |
| `EngramGateConvFwdOp` | train-s128 | bfloat16 | vector | 7.619 | 7.619 | report | 948992 | 332800 | 2.9 | 0.27 | 9.71 | no | 2.73 | - |
| `EngramGateConvFwdOp` | train-s32 | float16 | vector | 9.194 | 9.194 | recapture | 234944 | 85504 | 2.7 | 0.05 | 9.35 | no | 0.58 | - |
| `EngramGateConvFwdOp` | train-s64 | float16 | vector | 8.230 | 8.230 | report | 1890048 | 663552 | 2.8 | 0.49 | 9.70 | no | 5.04 | - |
| `EngramGateConvFwdOp` | train-s96 | bfloat16 | vector | 11.049 | 11.049 | report | 6390720 | 2225664 | 2.9 | 1.23 | 9.77 | no | 12.59 | - |
| `MLADenseBwdOp` | ds-v2-lite-3k | bfloat16 | vector | 72.338 | 16625.860 | report | 504235524096 | 504102912 | 1000.3 | 64.53 | 100.00 | no | 1.90 | report-underreports |
| `MLADenseBwdOp` | ds-v3-4k | bfloat16 | vector | 208.321 | 58270.787 | report | 1792646840320 | 1344274432 | 1333.5 | 65.46 | 100.00 | no | 1.44 | report-underreports |
| `MLADenseBwdOp` | ds-v3-4k | float16 | vector | 206.666 | 58272.474 | report | 1792646840320 | 1344274432 | 1333.5 | 65.45 | 100.00 | no | 1.44 | report-underreports |
| `MLADenseBwdOp` | ds-v3-b2-8k | bfloat16 | vector | 957.408 | 455298.492 | report | 14338887778304 | 5377097728 | 2666.7 | 67.01 | 100.00 | no | 0.74 | report-underreports |
| `MLADenseBwdOp` | kimi-k2-8k | bfloat16 | vector | 208.468 | 113770.578 | report | 3584721944576 | 1344274432 | 2666.7 | 67.04 | 100.00 | no | 0.74 | report-underreports |
| `MLADenseBwdOp` | mla-dense-bwd-smoke | bfloat16 | vector | 4.413 | 110.412 | recapture | 1754540032 | 10502144 | 167.1 | 33.81 | 100.00 | no | 5.94 | report-underreports |
| `MLAVarlenBwdOp` | ds-v3-8x4k | bfloat16 | vector | 120967.758 | 490592.444 | report | 14341174722560 | 10754195528 | 1333.5 | 62.20 | 100.00 | no | 1.37 | report-underreports |
| `MLAVarlenBwdOp` | ds-v3-8x4k | float16 | vector | 120969.406 | 490626.594 | report | 14341174722560 | 10754195528 | 1333.5 | 62.19 | 100.00 | no | 1.37 | report-underreports |
| `MLAVarlenBwdOp` | ds-v3-mixed | bfloat16 | vector | 430522.125 | 1728735.319 | report | 38226104254464 | 10586161208 | 3611.0 | 47.05 | 100.00 | no | 0.38 | report-underreports |
| `MLAVarlenBwdOp` | kimi-k2-4x8k | bfloat16 | vector | 1669.053 | 470544.937 | report | 14338887778304 | 5377097768 | 2666.7 | 64.84 | 100.00 | no | 0.71 | report-underreports |
| `MLAVarlenBwdOp` | mla-varlen-bwd-smoke | bfloat16 | vector | 5.786 | 130.201 | recapture | 1427517952 | 13127704 | 108.7 | 23.33 | 100.00 | no | 6.30 | report-underreports |
| `MoEPermuteAlignFwdOp` | ds-v3-t1 | int32 | vector | 18.176 | 21.168 | report | 0 | 65848 | 0.0 | - | - | n/a | 0.19 | - |
| `MoEPermuteAlignFwdOp` | ds-v3-t32 | int32 | vector | 28.226 | 37.512 | report | 0 | 67844 | 0.0 | - | - | n/a | 0.11 | - |
| `MoEPermuteAlignFwdOp` | ds-v3-t4096 | int32 | vector | 39.236 | 939.017 | report | 0 | 329972 | 0.0 | - | - | n/a | 0.02 | report-underreports |
| `MoEPermuteAlignFwdOp` | ds-v3-t512 | int32 | vector | 23.794 | 153.236 | report | 0 | 98804 | 0.0 | - | - | n/a | 0.04 | report-underreports |
| `MoEPermuteAlignFwdOp` | kimi-k2-t1 | int32 | vector | 23.810 | 26.516 | report | 0 | 98608 | 0.0 | - | - | n/a | 0.23 | - |
| `MoEPermuteAlignFwdOp` | kimi-k2-t32 | int32 | vector | 34.779 | 43.331 | report | 0 | 100604 | 0.0 | - | - | n/a | 0.15 | - |
| `MoEPermuteAlignFwdOp` | kimi-k2-t4096 | int32 | vector | 45.003 | 658.449 | report | 0 | 362732 | 0.0 | - | - | n/a | 0.03 | report-underreports |
| `MoEPermuteAlignFwdOp` | kimi-k2-t512 | int32 | vector | 25.870 | 129.172 | report | 0 | 131564 | 0.0 | - | - | n/a | 0.06 | report-underreports |
| `MoEPermuteAlignFwdOp` | qwen3-235b-t1 | int32 | vector | 12.140 | 15.233 | report | 0 | 33088 | 0.0 | - | - | n/a | 0.14 | - |
| `MoEPermuteAlignFwdOp` | qwen3-235b-t32 | int32 | vector | 23.332 | 40.284 | report | 0 | 35084 | 0.0 | - | - | n/a | 0.05 | - |
| `MoEPermuteAlignFwdOp` | qwen3-235b-t4096 | int32 | vector | 29.534 | 1731.115 | report | 0 | 297212 | 0.0 | - | - | n/a | 0.01 | report-underreports |
| `MoEPermuteAlignFwdOp` | qwen3-235b-t512 | int32 | vector | 22.115 | 250.908 | report | 0 | 66044 | 0.0 | - | - | n/a | 0.02 | report-underreports |
| `PagedKVCacheWriteFwdOp` | gpt-oss-120b-bfloat16 | bfloat16 | vector | 25.768 | 25.768 | report | 0 | 1050624 | 0.0 | - | - | n/a | 2.55 | - |
| `PagedKVCacheWriteFwdOp` | llama-8b-decode-bfloat16 | bfloat16 | vector | 15.218 | 15.218 | report | 0 | 524800 | 0.0 | - | - | n/a | 2.16 | - |
| `PagedKVCacheWriteFwdOp` | llama-8b-decode-float16 | float16 | vector | 15.418 | 15.418 | report | 0 | 524800 | 0.0 | - | - | n/a | 2.13 | - |
| `PagedKVCacheWriteFwdOp` | llama-8b-prefill-bfloat16 | bfloat16 | vector | 884.653 | 884.653 | report | 0 | 67174400 | 0.0 | - | - | n/a | 4.75 | - |
| `PagedKVCacheWriteFwdOp` | odd-tokens-and-page-float16 | float16 | vector | 6.463 | 6.463 | report | 0 | 7224 | 0.0 | - | - | n/a | 0.07 | - |
| `PagedKVCacheWriteFwdOp` | paged-write-smoke-float16 | float16 | vector | 5.070 | 5.070 | recapture | 0 | 5160 | 0.0 | - | - | n/a | 0.06 | - |
| `PagedKVCacheWriteFwdOp` | qwen3-235b-fp8-bfloat16-float8_e4m3fn | bfloat16 | vector | 267.690 | 267.690 | report | 8388608 | 12615688 | 0.7 | 0.07 | 2.26 | no | 2.95 | - |
| `PagedKVCacheWriteFwdOp` | qwen3-235b-fp8-fp16-float16-float8_e4m3fn | float16 | vector | 78.891 | 78.891 | report | 2097152 | 3153928 | 0.7 | 0.06 | 2.26 | no | 2.50 | - |
| `RoPEFwdOp` | gptj-6b-decode-b32 | float16 | vector | 10.302 | 10.302 | report | 196608 | 1052928 | 0.2 | 0.04 | 0.64 | no | 6.39 | - |
| `RoPEFwdOp` | gptj-6b-prefill | float16 | vector | 330.012 | 330.012 | report | 12582912 | 67387392 | 0.2 | 0.08 | 0.64 | no | 12.76 | - |
| `RoPEFwdOp` | llama3-8b-b2-2k | float32 | vector | 671.119 | 671.119 | report | 62914560 | 169902080 | 0.4 | 0.41 | 2.58 | no | 15.82 | - |
| `RoPEFwdOp` | llama3-8b-decode-b64 | bfloat16 | vector | 10.925 | 10.925 | report | 983040 | 1327616 | 0.7 | 0.19 | 2.52 | no | 7.60 | - |
| `RoPEFwdOp` | llama3-8b-prefill-4k | bfloat16 | vector | 430.361 | 430.361 | report | 62914560 | 84967424 | 0.7 | 0.31 | 2.52 | no | 12.34 | - |
| `RoPEFwdOp` | llama3-8b-prefill-4k | float16 | vector | 429.756 | 429.756 | report | 62914560 | 84967424 | 0.7 | 0.31 | 2.52 | no | 12.36 | - |
| `RoPEFwdOp` | phi-2-query-only | bfloat16 | vector | 141.328 | 141.328 | report | 3072000 | 10308000 | 0.3 | 0.05 | 1.01 | no | 4.56 | - |
| `SSDChunkCouplingFwdOp` | mamba2-2p7b | bfloat16 | vector | 22.910 | 22.910 | report | 268435456 | 8388608 | 32.0 | 24.93 | 100.00 | no | 22.88 | - |
| `SSDChunkCouplingFwdOp` | mamba2-780m | float16 | vector | 12.643 | 12.643 | report | 134217728 | 4194304 | 32.0 | 22.59 | 100.00 | no | 20.73 | - |
| `SSDChunkCouplingFwdOp` | mamba2-grouped-prefill | float16 | vector | 171.000 | 171.000 | report | 2415919104 | 75497472 | 32.0 | 30.06 | 100.00 | no | 27.59 | - |
| `SSDChunkCouplingFwdOp` | ssd-chunk-coupling-smoke | float16 | vector | 2.680 | 2.680 | recapture | 524288 | 49152 | 10.7 | 0.42 | 36.31 | no | 1.15 | - |
| `SharedExpertMLPFwdOp` | 1x7168x18432 | bfloat16 | cube | 240.867 | 723.016 | report | 792834048 | 792752128 | 1.0 | 0.29 | 0.42 | no | 68.53 | report-underreports |
| `SharedExpertMLPFwdOp` | 32x5120x1536 | bfloat16 | cube | 16.872 | 59.911 | report | 1510244352 | 47841280 | 31.6 | 6.67 | 13.36 | no | 49.91 | report-underreports |
| `SharedExpertMLPFwdOp` | 32x7168x2048 | bfloat16 | cube | 30.896 | 105.804 | report | 2818965504 | 88997888 | 31.7 | 7.05 | 13.41 | no | 52.57 | report-underreports |
| `SharedExpertMLPFwdOp` | 32x7168x2048 | float16 | cube | 28.189 | 102.759 | report | 2818965504 | 88997888 | 31.7 | 7.26 | 13.41 | no | 54.13 | report-underreports |
| `SharedExpertMLPFwdOp` | 4096x7168x2048 | bfloat16 | cube | 607.648 | 1879.047 | report | 360827584512 | 205520896 | 1755.7 | 50.80 | 100.00 | no | 6.84 | report-underreports |
| `SharedExpertMLPFwdOp` | 512x7168x18432 | bfloat16 | cube | 716.776 | 2146.963 | report | 405931032576 | 807403520 | 502.8 | 50.02 | 100.00 | no | 23.50 | report-underreports |

## Per-operator conclusion

### `ChainSpeculativeSamplingFwdOp`

- **类型**: memory-bound (max AI 0.7 FLOP/byte vs the vector balance 29.4).
- **受限资源**: bandwidth — on the dominant case `ds-v3-mtp-n3-float32` MFU 0.04%, MU 0.81% (operator maxima: MFU 0.06%, MU 1.23%).
- **是否达标**: 否 (MFU 0.04% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `EngramDecodeFwdOp`

- **类型**: memory-bound (max AI 6.1 FLOP/byte vs the vector balance 29.4).
- **受限资源**: bandwidth — on the dominant case `serving-b4` MFU 2.06%, MU 16.35% (operator maxima: MFU 2.06%, MU 16.35%).
- **是否达标**: 否 (MFU 2.06% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `EngramGateConvBwdOp`

- **类型**: memory-bound (max AI 4.3 FLOP/byte vs the vector balance 29.4).
- **受限资源**: bandwidth — on the dominant case `train-s96` MFU 2.15%, MU 14.86% (operator maxima: MFU 2.15%, MU 14.86%).
- **是否达标**: 否 (MFU 2.15% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `EngramGateConvFwdOp`

- **类型**: memory-bound (max AI 2.9 FLOP/byte vs the vector balance 29.4).
- **受限资源**: bandwidth — on the dominant case `train-s96` MFU 1.23%, MU 12.59% (operator maxima: MFU 1.23%, MU 12.59%).
- **是否达标**: 否 (MFU 1.23% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `MLADenseBwdOp`

- **类型**: compute-bound (max AI 2666.7 FLOP/byte vs the vector balance 29.4).
- **受限资源**: compute — on the dominant case `ds-v3-b2-8k` MFU 67.01%, MU 0.74% (operator maxima: MFU 67.04%, MU 5.94%).
- **是否达标**: 否 (MFU 67.01% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `MLAVarlenBwdOp`

- **类型**: compute-bound (max AI 3611.0 FLOP/byte vs the vector balance 29.4).
- **受限资源**: compute — on the dominant case `ds-v3-mixed` MFU 47.05%, MU 0.38% (operator maxima: MFU 64.84%, MU 6.30%).
- **是否达标**: 否 (MFU 47.05% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `MoEPermuteAlignFwdOp`

- **类型**: memory-bound (max AI 0.0 FLOP/byte vs the vector balance 29.4).
- **受限资源**: bandwidth — on the dominant case `kimi-k2-t4096` MFU -%, MU 0.03% (operator maxima: MFU -%, MU 0.23%).
- **是否达标**: 否 (MFU -% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `PagedKVCacheWriteFwdOp`

- **类型**: memory-bound (max AI 0.7 FLOP/byte vs the vector balance 29.4).
- **受限资源**: bandwidth — on the dominant case `qwen3-235b-fp8-bfloat16-float8_e4m3fn` MFU 0.07%, MU 2.95% (operator maxima: MFU 0.07%, MU 4.75%).
- **是否达标**: 否 (MFU 0.07% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `RoPEFwdOp`

- **类型**: memory-bound (max AI 0.7 FLOP/byte vs the vector balance 29.4).
- **受限资源**: bandwidth — on the dominant case `llama3-8b-b2-2k` MFU 0.41%, MU 15.82% (operator maxima: MFU 0.41%, MU 15.82%).
- **是否达标**: 否 (MFU 0.41% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `SSDChunkCouplingFwdOp`

- **类型**: compute-bound (max AI 32.0 FLOP/byte vs the vector balance 29.4).
- **受限资源**: compute — on the dominant case `mamba2-grouped-prefill` MFU 30.06%, MU 27.59% (operator maxima: MFU 30.06%, MU 27.59%).
- **是否达标**: 否 (MFU 30.06% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.

### `SharedExpertMLPFwdOp`

- **类型**: compute-bound (max AI 1755.7 FLOP/byte vs the cube balance 236.2).
- **受限资源**: compute — on the dominant case `512x7168x18432` MFU 50.02%, MU 23.50% (operator maxima: MFU 50.80%, MU 68.53%).
- **是否达标**: 否 (MFU 50.02% on the dominant case vs target 90%; MFU is the acceptance metric, MU is diagnostic only).
- **阻塞/异常**: none — every case has a whole-call T_op.
