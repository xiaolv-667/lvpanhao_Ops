# NPU Benchmark Report
Generated: 2026-11-24 09:30:55

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
| train-s64 | 8.2270 | 0.2297 | 0.0807 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_092758_551541_EngramGateConvBwdOp/artifacts/msprof/main_kernel__u85o3av | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 8.227, 'max_latency_us': 8.227, 'min_latency_us': 8.227, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 8.227}, 't_op_us': 8.227} | msprof | 2 | 512 | 1e-06 | 64 | float16 |
| train-s128 | 7.5110 | 0.1263 | 0.0443 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_092758_551541_EngramGateConvBwdOp/artifacts/msprof/main_kernel_lmm5y9cr | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 7.511, 'max_latency_us': 7.511, 'min_latency_us': 7.511, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 7.511}, 't_op_us': 7.511} | msprof | 1 | 256 | 1e-06 | 128 | bfloat16 |
| train-s96 | 10.7170 | 0.5963 | 0.2077 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_092758_551541_EngramGateConvBwdOp/artifacts/msprof/main_kernel_tqqrdqm0 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 10.717, 'max_latency_us': 10.717, 'min_latency_us': 10.717, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 10.717}, 't_op_us': 10.717} | msprof | 3 | 768 | 1e-06 | 96 | bfloat16 |

## EngramGateConvBwdOp

### tileops

| Label | Latency (us) | Derived TFLOP/s | Derived TB/s | artifact_dir | msprof | prof_mode | M | d | eps | seq_len | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train-s64 | 7.0400 | 0.6373 | 0.1519 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_092758_551541_EngramGateConvBwdOp/artifacts/msprof/main_kernel_qs0f302t | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 7.04, 'max_latency_us': 7.04, 'min_latency_us': 3.3, 'op_name': 'main_kernel', 'sample_count': 2, 't_op_breakdown': {'main_kernel': 10.34}, 't_op_us': 10.34} | msprof | 2 | 512 | 1e-06 | 64 | float16 |
| train-s128 | 6.1320 | 0.3674 | 0.0873 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_092758_551541_EngramGateConvBwdOp/artifacts/msprof/main_kernel_vog5_b33 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.132, 'max_latency_us': 6.132, 'min_latency_us': 2.579, 'op_name': 'main_kernel', 'sample_count': 2, 't_op_breakdown': {'main_kernel': 8.711}, 't_op_us': 8.711} | msprof | 1 | 256 | 1e-06 | 128 | bfloat16 |
| train-s96 | 11.7270 | 1.2946 | 0.3045 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_092758_551541_EngramGateConvBwdOp/artifacts/msprof/main_kernel_g0jmac5a | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 11.727, 'max_latency_us': 11.727, 'min_latency_us': 3.449, 'op_name': 'main_kernel', 'sample_count': 2, 't_op_breakdown': {'main_kernel': 15.176}, 't_op_us': 15.176} | msprof | 3 | 768 | 1e-06 | 96 | bfloat16 |

## EngramDecodeFwdOp

### tileops

| Label | Latency (us) | Derived TFLOP/s | Derived TB/s | artifact_dir | msprof | prof_mode | batch | conv_kernel_size | conv_len | d | d_mem | dilation | eps | max_conv_len | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| serving-b1 | 3.2180 | 0.1648 | 0.1683 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_092758_551541_EngramGateConvBwdOp/artifacts/msprof/engram_decode_step_kernel_kdm1wj1a | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 3.218, 'max_latency_us': 3.218, 'min_latency_us': 2.657, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2, 't_op_breakdown': {'engram_decode_step_kernel': 2.657, 'engram_project_kernel': 3.218}, 't_op_us': 5.875} | msprof | 1 | 4 | 12 | 256 | 512 | 3 | 1e-06 | 12 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
| serving-b4 | 4.6820 | 1.8022 | 0.4877 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_092758_551541_EngramGateConvBwdOp/artifacts/msprof/engram_decode_step_kernel_1kzbbm9r | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 4.682, 'max_latency_us': 4.682, 'min_latency_us': 3.172, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2, 't_op_breakdown': {'engram_decode_step_kernel': 3.172, 'engram_project_kernel': 4.682}, 't_op_us': 7.854000000000001} | msprof | 4 | 4 | 20 | 512 | 1024 | 5 | 1e-06 | 20 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
| serving-b8 | 6.4780 | 0.6551 | 0.1067 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_092758_551541_EngramGateConvBwdOp/artifacts/msprof/engram_decode_step_kernel_wqkuh8hh | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 6.478, 'max_latency_us': 6.478, 'min_latency_us': 2.895, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2, 't_op_breakdown': {'engram_decode_step_kernel': 2.895, 'engram_project_kernel': 6.478}, 't_op_us': 9.373} | msprof | 8 | 4 | 18 | 256 | 512 | 3 | 1e-06 | 18 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | bfloat16 |
| decode-partial-cache | 2.8320 | 0.3746 | 0.1938 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_092758_551541_EngramGateConvBwdOp/artifacts/msprof/PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021_8kq8a4pz | {'captured_op_count': 3, 'captured_op_names': ['PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021', 'engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 2.832, 'max_latency_us': 3.062, 'min_latency_us': 2.829, 'op_name': 'PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021', 'sample_count': 3, 't_op_breakdown': {'PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021': 2.832, 'engram_decode_step_kernel': 2.829, 'engram_project_kernel': 3.062}, 't_op_us': 8.722999999999999} | msprof | 2 | 4 | 5 | 256 | 512 | 3 | 1e-06 | 12 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
