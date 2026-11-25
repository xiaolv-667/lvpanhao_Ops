# NPU Benchmark Report
Generated: 2026-11-24 22:24:21

## Environment

- **Torch version**: 2.13.0+cpu
- **torch_npu version**: 2.13.0.rc1
- **NPU model**: Ascend950PR_9579
- **NPU count**: 1
- **Profiling mode**: msprof

## EngramGateConvFwdOp

### tileops

| Label | Latency (us) | Derived TFLOP/s | Derived TB/s | artifact_dir | msprof | prof_mode | M | d | eps | seq_len | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train-s64 | 6.5940 | 0.2866 | 0.1006 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_222322_139293_EngramGateConvFwdOp/artifacts/msprof/main_kernel_f0am7t6h | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.594, 'max_latency_us': 6.594, 'min_latency_us': 6.594, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 6.594}, 't_op_us': 6.594} | msprof | 2 | 512 | 1e-06 | 64 | float16 |
| train-s128 | 5.6090 | 0.1692 | 0.0593 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_222322_139293_EngramGateConvFwdOp/artifacts/msprof/main_kernel_s8gk4drn | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 5.609, 'max_latency_us': 5.609, 'min_latency_us': 5.609, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 5.609}, 't_op_us': 5.609} | msprof | 1 | 256 | 1e-06 | 128 | bfloat16 |
| train-s96 | 8.8340 | 0.7234 | 0.2519 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_222322_139293_EngramGateConvFwdOp/artifacts/msprof/main_kernel_5830lt06 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 8.834, 'max_latency_us': 8.834, 'min_latency_us': 8.834, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 8.834}, 't_op_us': 8.834} | msprof | 3 | 768 | 1e-06 | 96 | bfloat16 |
