# NPU Benchmark Report
Generated: 2026-11-23 14:42:07

## Environment

- **Torch version**: 2.13.0+cpu
- **torch_npu version**: 2.13.0.rc1
- **NPU model**: Ascend950PR_9579
- **NPU count**: 1
- **Profiling mode**: msprof

## EngramGateConvFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | artifact_dir | bandwidth_tbs | msprof | prof_mode | tflops | M | d | eps | seq_len | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train-s64 | 8.2300 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143920_807616_EngramDecodeFwdOp/artifacts/msprof/main_kernel_umye5xje | 0.0806 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 8.23, 'max_latency_us': 8.23, 'min_latency_us': 8.23, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | 0.2297 | 2 | 512 | 1e-06 | 64 | float16 |
| train-s128 | 7.6190 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143920_807616_EngramDecodeFwdOp/artifacts/msprof/main_kernel_54skrivu | 0.0437 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 7.619, 'max_latency_us': 7.619, 'min_latency_us': 7.619, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | 0.1246 | 1 | 256 | 1e-06 | 128 | bfloat16 |
| train-s96 | 11.0490 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143920_807616_EngramDecodeFwdOp/artifacts/msprof/main_kernel_wtrwxmmb | 0.2014 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 11.049, 'max_latency_us': 11.049, 'min_latency_us': 11.049, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | 0.5784 | 3 | 768 | 1e-06 | 96 | bfloat16 |

## EngramGateConvBwdOp

### tileops

| Label | Latency (us) | Ratio (%) | artifact_dir | bandwidth_tbs | msprof | prof_mode | tflops | M | d | eps | seq_len | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train-s64 | 6.9970 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143920_807616_EngramDecodeFwdOp/artifacts/msprof/main_kernel_bgxkz01g | 0.1528 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.997, 'max_latency_us': 6.997, 'min_latency_us': 3.35, 'op_name': 'main_kernel', 'sample_count': 2} | msprof | 0.6412 | 2 | 512 | 1e-06 | 64 | float16 |
| train-s128 | 6.2380 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143920_807616_EngramDecodeFwdOp/artifacts/msprof/main_kernel_33xvybwt | 0.0859 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.238, 'max_latency_us': 6.238, 'min_latency_us': 2.38, 'op_name': 'main_kernel', 'sample_count': 2} | msprof | 0.3612 | 1 | 256 | 1e-06 | 128 | bfloat16 |
| train-s96 | 11.4020 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143920_807616_EngramDecodeFwdOp/artifacts/msprof/main_kernel_42akeofc | 0.3132 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 11.402, 'max_latency_us': 11.402, 'min_latency_us': 3.152, 'op_name': 'main_kernel', 'sample_count': 2} | msprof | 1.3315 | 3 | 768 | 1e-06 | 96 | bfloat16 |

## EngramDecodeFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | conv_kernel_size | conv_len | d | d_mem | dilation | eps | max_conv_len | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| serving-b1 | 3.1410 | 12.2432 | 13.6335 | 1.6000 | 378.4704 | 2.6707 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143920_807616_EngramDecodeFwdOp/artifacts/msprof/engram_decode_step_kernel_4ky70y4l | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 3.141, 'max_latency_us': 3.141, 'min_latency_us': 3.117, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2} | msprof | 1 | 4 | 12 | 256 | 512 | 3 | 1e-06 | 12 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
| serving-b4 | 4.5790 | 35.1256 | 13.0387 | 1.6000 | 378.4704 | 7.3279 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143920_807616_EngramDecodeFwdOp/artifacts/msprof/engram_decode_step_kernel_9o9ze2lu | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 4.579, 'max_latency_us': 4.579, 'min_latency_us': 3.183, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2} | msprof | 4 | 4 | 20 | 512 | 1024 | 5 | 1e-06 | 20 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
| serving-b8 | 6.3850 | 6.8360 | 12.0117 | 1.6000 | 378.4704 | 1.3138 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143920_807616_EngramDecodeFwdOp/artifacts/msprof/engram_decode_step_kernel_w0nczlj6 | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 6.385, 'max_latency_us': 6.385, 'min_latency_us': 2.824, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2} | msprof | 8 | 4 | 18 | 256 | 512 | 3 | 1e-06 | 18 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | bfloat16 |
| decode-partial-cache | 3.0920 | 12.4191 | 13.6533 | 1.6000 | 378.4704 | 2.7130 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143920_807616_EngramDecodeFwdOp/artifacts/msprof/PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021_t9yi0eam | {'captured_op_count': 3, 'captured_op_names': ['PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021', 'engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 3.092, 'max_latency_us': 3.294, 'min_latency_us': 3.089, 'op_name': 'PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021', 'sample_count': 3} | msprof | 2 | 4 | 5 | 256 | 512 | 3 | 1e-06 | 12 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
