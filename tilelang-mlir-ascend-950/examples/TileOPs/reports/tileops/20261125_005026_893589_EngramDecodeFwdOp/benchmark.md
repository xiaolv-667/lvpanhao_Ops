# NPU Benchmark Report
Generated: 2026-11-25 00:51:43

## Environment

- **Torch version**: 2.13.0+cpu
- **torch_npu version**: 2.13.0.rc1
- **NPU model**: Ascend950PR_9579
- **NPU count**: 1
- **Profiling mode**: msprof

## EngramDecodeFwdOp

### tileops

| Label | Latency (us) | Derived TFLOP/s | Derived TB/s | artifact_dir | msprof | prof_mode | batch | conv_kernel_size | conv_len | d | d_mem | dilation | eps | max_conv_len | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| serving-b1 | 3.0670 | 0.1730 | 0.1766 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261125_005026_893589_EngramDecodeFwdOp/artifacts/msprof/engram_decode_step_kernel_4f6moi33 | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 3.067, 'max_latency_us': 3.067, 'min_latency_us': 2.674, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2, 't_op_breakdown': {'engram_decode_step_kernel': 2.674, 'engram_project_kernel': 3.067}, 't_op_us': 5.741} | msprof | 1 | 4 | 12 | 256 | 512 | 3 | 1e-06 | 12 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
| serving-b4 | 4.6050 | 1.8323 | 0.4959 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261125_005026_893589_EngramDecodeFwdOp/artifacts/msprof/engram_decode_step_kernel_hiuzw47g | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 4.605, 'max_latency_us': 4.605, 'min_latency_us': 3.56, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2, 't_op_breakdown': {'engram_decode_step_kernel': 3.56, 'engram_project_kernel': 4.605}, 't_op_us': 8.165000000000001} | msprof | 4 | 4 | 20 | 512 | 1024 | 5 | 1e-06 | 20 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
| serving-b8 | 3.1950 | 1.3282 | 0.2163 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261125_005026_893589_EngramDecodeFwdOp/artifacts/msprof/engram_decode_step_kernel_01c802ft | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 3.195, 'max_latency_us': 3.195, 'min_latency_us': 2.899, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2, 't_op_breakdown': {'engram_decode_step_kernel': 2.899, 'engram_project_kernel': 3.195}, 't_op_us': 6.093999999999999} | msprof | 8 | 4 | 18 | 256 | 512 | 3 | 1e-06 | 18 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | bfloat16 |
| decode-partial-cache | 2.9110 | 0.3644 | 0.1885 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261125_005026_893589_EngramDecodeFwdOp/artifacts/msprof/PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021_7b7r82me | {'captured_op_count': 3, 'captured_op_names': ['PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021', 'engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 2.911, 'max_latency_us': 3.093, 'min_latency_us': 2.887, 'op_name': 'PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021', 'sample_count': 3, 't_op_breakdown': {'PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021': 2.887, 'engram_decode_step_kernel': 2.911, 'engram_project_kernel': 3.093}, 't_op_us': 8.891} | msprof | 2 | 4 | 5 | 256 | 512 | 3 | 1e-06 | 12 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
