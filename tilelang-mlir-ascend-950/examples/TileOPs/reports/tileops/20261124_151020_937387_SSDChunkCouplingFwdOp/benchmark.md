# NPU Benchmark Report
Generated: 2026-11-24 15:11:26

## Environment

- **Torch version**: 2.13.0+cpu
- **torch_npu version**: 2.13.0.rc1
- **NPU model**: Ascend950PR_9579
- **NPU count**: 1
- **Profiling mode**: msprof

## SSDChunkCouplingFwdOp

### tileops

| Label | Latency (us) | Derived TFLOP/s | Derived TB/s | artifact_dir | msprof | prof_mode | batch | chunk_len | d_state | n_groups | num_chunks | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mamba2-780m | 8.5040 | 15.7829 | 0.4932 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_151020_937387_SSDChunkCouplingFwdOp/artifacts/msprof/main_kernel_0yiik7lq | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 8.504, 'max_latency_us': 8.504, 'min_latency_us': 8.504, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 8.504}, 't_op_us': 8.504} | msprof | 1 | 256 | 128 | 1 | 16 | {'block_l': 64, 'block_n': 64, 'block_s': 64} | float16 |
| mamba2-2p7b | 16.0380 | 16.7375 | 0.5230 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_151020_937387_SSDChunkCouplingFwdOp/artifacts/msprof/main_kernel_26ugkzg_ | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 16.038, 'max_latency_us': 16.038, 'min_latency_us': 16.038, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 16.038}, 't_op_us': 16.038} | msprof | 4 | 256 | 128 | 1 | 8 | {'block_l': 64, 'block_n': 64, 'block_s': 64} | bfloat16 |
| mamba2-grouped-prefill | 83.1880 | 29.0417 | 0.9076 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_151020_937387_SSDChunkCouplingFwdOp/artifacts/msprof/main_kernel_39p1v6r0 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 83.188004, 'max_latency_us': 83.188004, 'min_latency_us': 83.188004, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 83.188004}, 't_op_us': 83.188004} | msprof | 3 | 256 | 128 | 8 | 12 | {'block_l': 64, 'block_n': 64, 'block_s': 64} | float16 |
