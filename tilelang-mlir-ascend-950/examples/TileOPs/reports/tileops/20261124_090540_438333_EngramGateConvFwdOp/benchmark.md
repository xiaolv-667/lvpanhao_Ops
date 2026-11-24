# NPU Benchmark Report
Generated: 2026-11-24 09:08:29

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
| train-s64 | 8.2030 | 0.2304 | 0.0809 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_090540_438333_EngramGateConvFwdOp/artifacts/msprof/main_kernel_qnfi_7a2 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 8.203, 'max_latency_us': 8.203, 'min_latency_us': 8.203, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 8.203}, 't_op_us': 8.203} | msprof | 2 | 512 | 1e-06 | 64 | float16 |
| train-s128 | 7.5360 | 0.1259 | 0.0442 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_090540_438333_EngramGateConvFwdOp/artifacts/msprof/main_kernel_ef6z4yll | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 7.536, 'max_latency_us': 7.536, 'min_latency_us': 7.536, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 7.536}, 't_op_us': 7.536} | msprof | 1 | 256 | 1e-06 | 128 | bfloat16 |
| train-s96 | 10.9500 | 0.5836 | 0.2033 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_090540_438333_EngramGateConvFwdOp/artifacts/msprof/main_kernel_yjsq35fn | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 10.95, 'max_latency_us': 10.95, 'min_latency_us': 10.95, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 10.95}, 't_op_us': 10.95} | msprof | 3 | 768 | 1e-06 | 96 | bfloat16 |

## EngramGateConvBwdOp

### tileops

| Label | Latency (us) | Derived TFLOP/s | Derived TB/s | artifact_dir | msprof | prof_mode | M | d | eps | seq_len | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train-s64 | 7.1690 | 0.6258 | 0.1491 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_090540_438333_EngramGateConvFwdOp/artifacts/msprof/main_kernel_291233y0 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 7.169, 'max_latency_us': 7.169, 'min_latency_us': 3.37, 'op_name': 'main_kernel', 'sample_count': 2, 't_op_breakdown': {'main_kernel': 10.539}, 't_op_us': 10.539} | msprof | 2 | 512 | 1e-06 | 64 | float16 |
| train-s128 | 6.0670 | 0.3714 | 0.0883 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_090540_438333_EngramGateConvFwdOp/artifacts/msprof/main_kernel_n5fo4zf9 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.067, 'max_latency_us': 6.067, 'min_latency_us': 2.774, 'op_name': 'main_kernel', 'sample_count': 2, 't_op_breakdown': {'main_kernel': 8.841000000000001}, 't_op_us': 8.841000000000001} | msprof | 1 | 256 | 1e-06 | 128 | bfloat16 |
| train-s96 | 11.6830 | 1.2995 | 0.3057 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_090540_438333_EngramGateConvFwdOp/artifacts/msprof/main_kernel_42u70i0n | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 11.683, 'max_latency_us': 11.683, 'min_latency_us': 3.434, 'op_name': 'main_kernel', 'sample_count': 2, 't_op_breakdown': {'main_kernel': 15.117}, 't_op_us': 15.117} | msprof | 3 | 768 | 1e-06 | 96 | bfloat16 |

## EngramDecodeFwdOp

### tileops

| Label | Latency (us) | Derived TFLOP/s | Derived TB/s | artifact_dir | msprof | prof_mode | batch | conv_kernel_size | conv_len | d | d_mem | dilation | eps | max_conv_len | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| serving-b1 | 3.1370 | 0.1691 | 0.1727 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_090540_438333_EngramGateConvFwdOp/artifacts/msprof/engram_decode_step_kernel_w6r_o7y5 | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 3.137, 'max_latency_us': 3.137, 'min_latency_us': 2.606, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2, 't_op_breakdown': {'engram_decode_step_kernel': 2.606, 'engram_project_kernel': 3.137}, 't_op_us': 5.743} | msprof | 1 | 4 | 12 | 256 | 512 | 3 | 1e-06 | 12 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
| serving-b4 | 5.0200 | 1.6808 | 0.4549 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_090540_438333_EngramGateConvFwdOp/artifacts/msprof/engram_decode_step_kernel_qq433c3e | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 5.02, 'max_latency_us': 5.02, 'min_latency_us': 3.201, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2, 't_op_breakdown': {'engram_decode_step_kernel': 3.201, 'engram_project_kernel': 5.02}, 't_op_us': 8.221} | msprof | 4 | 4 | 20 | 512 | 1024 | 5 | 1e-06 | 20 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
| serving-b8 | 6.1070 | 0.6949 | 0.1132 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_090540_438333_EngramGateConvFwdOp/artifacts/msprof/engram_decode_step_kernel_3olsk8ch | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 6.107, 'max_latency_us': 6.107, 'min_latency_us': 2.933, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2, 't_op_breakdown': {'engram_decode_step_kernel': 2.933, 'engram_project_kernel': 6.107}, 't_op_us': 9.04} | msprof | 8 | 4 | 18 | 256 | 512 | 3 | 1e-06 | 18 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | bfloat16 |
| decode-partial-cache | 2.9790 | 0.3561 | 0.1842 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_090540_438333_EngramGateConvFwdOp/artifacts/msprof/PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021_v5sjo0er | {'captured_op_count': 3, 'captured_op_names': ['PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021', 'engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 2.979, 'max_latency_us': 3.002, 'min_latency_us': 2.887, 'op_name': 'PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021', 'sample_count': 3, 't_op_breakdown': {'PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021': 2.979, 'engram_decode_step_kernel': 2.887, 'engram_project_kernel': 3.002}, 't_op_us': 8.868} | msprof | 2 | 4 | 5 | 256 | 512 | 3 | 1e-06 | 12 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
