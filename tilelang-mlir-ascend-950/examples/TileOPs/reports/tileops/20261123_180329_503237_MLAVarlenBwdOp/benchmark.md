# NPU Benchmark Report
Generated: 2026-11-23 18:07:19

## Environment

- **Torch version**: 2.13.0+cpu
- **torch_npu version**: 2.13.0.rc1
- **NPU model**: Ascend950PR_9579
- **NPU count**: 1
- **Profiling mode**: msprof

## MLAVarlenBwdOp

### tileops

| Label | Latency (us) | Ratio (%) | bandwidth_tbs | prof_mode | tflops | dim_qk | dim_v | heads | is_causal | kv_lens | q_lens | sm_scale | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ds-v3-8x4k | 20774.7768 | N/A | 0.5177 | events | 690.3167 | 192 | 128 | 128 | True | [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096] | [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096] | None | bfloat16 |
| ds-v3-8x4k | 20776.9335 | N/A | 0.5176 | events | 690.2450 | 192 | 128 | 128 | True | [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096] | [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096] | None | float16 |
| ds-v3-mixed | 1727128.5889 | N/A | 0.0061 | events | 22.1327 | 192 | 128 | 128 | True | [512, 1024, 2048, 4096, 8192, 16384] | [512, 1024, 2048, 4096, 8192, 16384] | None | bfloat16 |
| kimi-k2-4x8k | 472266.7908 | N/A | 0.0114 | events | 30.3618 | 192 | 128 | 64 | True | [8192, 8192, 8192, 8192] | [8192, 8192, 8192, 8192] | None | bfloat16 |
