# NPU Benchmark Report
Generated: 2026-11-24 23:20:33

## Environment

- **Torch version**: 2.13.0+cpu
- **torch_npu version**: 2.13.0.rc1
- **NPU model**: Ascend950PR_9579
- **NPU count**: 1
- **Profiling mode**: msprof

## EngramGateConvBwdOp

### tileops

| Label | Latency (us) | Derived TFLOP/s | Derived TB/s | artifact_dir | msprof | prof_mode | M | d | eps | seq_len | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train-s64 | 6.0930 | 0.7363 | 0.1755 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_231932_732777_EngramGateConvBwdOp/artifacts/msprof/main_kernel_h_ihhj2c | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.093, 'max_latency_us': 6.093, 'min_latency_us': 2.716, 'op_name': 'main_kernel', 'sample_count': 2, 't_op_breakdown': {'main_kernel': 8.809000000000001}, 't_op_us': 8.809000000000001} | msprof | 2 | 512 | 1e-06 | 64 | float16 |
| train-s128 | 4.8730 | 0.4624 | 0.1099 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_231932_732777_EngramGateConvBwdOp/artifacts/msprof/main_kernel_j6fy3ppx | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 4.873, 'max_latency_us': 4.873, 'min_latency_us': 2.266, 'op_name': 'main_kernel', 'sample_count': 2, 't_op_breakdown': {'main_kernel': 7.139}, 't_op_us': 7.139} | msprof | 1 | 256 | 1e-06 | 128 | bfloat16 |
| train-s96 | 9.8030 | 1.5487 | 0.3643 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_231932_732777_EngramGateConvBwdOp/artifacts/msprof/main_kernel_x2pmb1us | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 9.803, 'max_latency_us': 9.803, 'min_latency_us': 2.59, 'op_name': 'main_kernel', 'sample_count': 2, 't_op_breakdown': {'main_kernel': 12.393}, 't_op_us': 12.393} | msprof | 3 | 768 | 1e-06 | 96 | bfloat16 |
