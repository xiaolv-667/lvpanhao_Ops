# NPU Benchmark Report
Generated: 2026-11-23 11:42:48

## Environment

- **Torch version**: 2.13.0+cpu
- **torch_npu version**: 2.13.0.rc1
- **NPU model**: Ascend950PR_9579
- **NPU count**: 1
- **Profiling mode**: msprof

## MLADenseBwdOp

### tileops

| Label | Latency (us) | Ratio (%) | artifact_dir | bandwidth_tbs | msprof | prof_mode | tflops | batch | dim_qk | dim_v | heads | is_causal | seqlen_kv | seqlen_q | sm_scale | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ds-v3-4k | 208.0010 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_114002_550215_MLADenseBwdOp/artifacts/msprof/Range_b2cb2b97546bcd85efbca38148de9ae1_high_performance_1001_hnpqj0r7 | 6.4628 | {'captured_op_count': 5, 'captured_op_names': ['Range_b2cb2b97546bcd85efbca38148de9ae1_high_performance_1001', 'ZerosLike_4_BYTES_high_performance_1', '_bwd_main_kernel', '_delta_main_kernel', '_dq_land_main_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 208.000992, 'max_latency_us': 57451.027344, 'min_latency_us': 2.224, 'op_name': 'Range_b2cb2b97546bcd85efbca38148de9ae1_high_performance_1001', 'sample_count': 6} | msprof | 8618.4533 | 1 | 192 | 128 | 128 | True | 4096 | 4096 | None | bfloat16 |
| ds-v3-4k | 208.3330 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_114002_550215_MLADenseBwdOp/artifacts/msprof/Range_b2cb2b97546bcd85efbca38148de9ae1_high_performance_1001_u6t4loi4 | 6.4525 | {'captured_op_count': 5, 'captured_op_names': ['Range_b2cb2b97546bcd85efbca38148de9ae1_high_performance_1001', 'ZerosLike_4_BYTES_high_performance_1', '_bwd_main_kernel', '_delta_main_kernel', '_dq_land_main_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 208.332993, 'max_latency_us': 57457.523438, 'min_latency_us': 2.1, 'op_name': 'Range_b2cb2b97546bcd85efbca38148de9ae1_high_performance_1001', 'sample_count': 6} | msprof | 8604.7189 | 1 | 192 | 128 | 128 | True | 4096 | 4096 | None | float16 |
| ds-v3-b2-8k | 455066.5833 | N/A | N/A | 0.0118 | N/A | events | 31.5094 | 2 | 192 | 128 | 128 | True | 8192 | 8192 | None | bfloat16 |
| kimi-k2-8k | 113774.4148 | N/A | N/A | 0.0118 | N/A | events | 31.5073 | 1 | 192 | 128 | 64 | True | 8192 | 8192 | None | bfloat16 |
| ds-v2-lite-3k | 16621.6948 | N/A | N/A | 0.0303 | N/A | events | 30.3360 | 4 | 192 | 128 | 16 | True | 3072 | 3072 | None | bfloat16 |
