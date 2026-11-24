# NPU Benchmark Report
Generated: 2026-11-23 13:57:37

## Environment

- **Torch version**: 2.13.0+cpu
- **torch_npu version**: 2.13.0.rc1
- **NPU model**: Ascend950PR_9579
- **NPU count**: 1
- **Profiling mode**: msprof

## SSDChunkCouplingFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | chunk_len | d_state | n_groups | num_chunks | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mamba2-780m | 12.6430 | 14.4744 | 38.6770 | 1.6000 | 378.4704 | 8.9572 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_135638_847395_SSDChunkCouplingFwdOp/artifacts/msprof/main_kernel_mhczwh2g | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 12.643, 'max_latency_us': 12.643, 'min_latency_us': 12.643, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | 1 | 256 | 128 | 1 | 16 | {'block_l': 64, 'block_n': 64, 'block_s': 64} | float16 |
| mamba2-2p7b | 22.9100 | 15.2433 | 20.2675 | 1.6000 | 378.4704 | 4.9431 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_135638_847395_SSDChunkCouplingFwdOp/artifacts/msprof/main_kernel_c3um65wy | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 22.91, 'max_latency_us': 22.91, 'min_latency_us': 22.91, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | 4 | 256 | 128 | 1 | 8 | {'block_l': 64, 'block_n': 64, 'block_s': 64} | bfloat16 |
| mamba2-grouped-prefill | 171.0000 | 18.1120 | 2.2853 | 1.6000 | 378.4704 | 0.6623 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_135638_847395_SSDChunkCouplingFwdOp/artifacts/msprof/main_kernel_unoir147 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 171.0, 'max_latency_us': 171.0, 'min_latency_us': 171.0, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | 3 | 256 | 128 | 8 | 12 | {'block_l': 64, 'block_n': 64, 'block_s': 64} | float16 |
