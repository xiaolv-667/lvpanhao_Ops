# NPU Benchmark Report
Generated: 2026-11-25 18:38:34

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
| mamba2-780m | 6.1600 | 21.7886 | 0.6809 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261125_183730_853046_SSDChunkCouplingFwdOp/artifacts/msprof/main_kernel_hwoz7rdx | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.16, 'max_latency_us': 6.16, 'min_latency_us': 6.16, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 6.16}, 't_op_us': 6.16} | msprof | 1 | 256 | 128 | 1 | 16 | {'block_l': 64, 'block_n': 64, 'block_s': 64} | float16 |
| mamba2-2p7b | 11.8080 | 22.7334 | 0.7104 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261125_183730_853046_SSDChunkCouplingFwdOp/artifacts/msprof/main_kernel_4jqjsnu3 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 11.808, 'max_latency_us': 11.808, 'min_latency_us': 11.808, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 11.808}, 't_op_us': 11.808} | msprof | 4 | 256 | 128 | 1 | 8 | {'block_l': 64, 'block_n': 64, 'block_s': 64} | bfloat16 |
| mamba2-grouped-prefill | 54.6640 | 44.1958 | 1.3811 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261125_183730_853046_SSDChunkCouplingFwdOp/artifacts/msprof/main_kernel_ia888lb6 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 54.664001, 'max_latency_us': 54.664001, 'min_latency_us': 54.664001, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 54.664001}, 't_op_us': 54.664001} | msprof | 3 | 256 | 128 | 8 | 12 | {'block_l': 64, 'block_n': 64, 'block_s': 64} | float16 |
