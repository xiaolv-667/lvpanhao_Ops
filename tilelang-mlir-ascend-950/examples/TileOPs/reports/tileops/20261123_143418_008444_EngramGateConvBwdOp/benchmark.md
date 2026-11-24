# NPU Benchmark Report
Generated: 2026-11-23 14:37:06

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
| train-s64 | 8.0900 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143418_008444_EngramGateConvBwdOp/artifacts/msprof/main_kernel_s6unmnnv | 0.0820 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 8.09, 'max_latency_us': 8.09, 'min_latency_us': 8.09, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | 0.2336 | 2 | 512 | 1e-06 | 64 | float16 |
| train-s128 | 7.4580 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143418_008444_EngramGateConvBwdOp/artifacts/msprof/main_kernel_67bss1i0 | 0.0446 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 7.458, 'max_latency_us': 7.458, 'min_latency_us': 7.458, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | 0.1272 | 1 | 256 | 1e-06 | 128 | bfloat16 |
| train-s96 | 10.6970 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143418_008444_EngramGateConvBwdOp/artifacts/msprof/main_kernel_r4ommew_ | 0.2081 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 10.697, 'max_latency_us': 10.697, 'min_latency_us': 10.697, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | 0.5974 | 3 | 768 | 1e-06 | 96 | bfloat16 |

## EngramGateConvBwdOp

### tileops

| Label | Latency (us) | Ratio (%) | artifact_dir | bandwidth_tbs | msprof | prof_mode | tflops | M | d | eps | seq_len | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train-s64 | 7.5420 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143418_008444_EngramGateConvBwdOp/artifacts/msprof/main_kernel_ea728iis | 0.1417 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 7.542, 'max_latency_us': 7.542, 'min_latency_us': 3.102, 'op_name': 'main_kernel', 'sample_count': 2} | msprof | 0.5949 | 2 | 512 | 1e-06 | 64 | float16 |
| train-s128 | 6.2150 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143418_008444_EngramGateConvBwdOp/artifacts/msprof/main_kernel_0n7ckqx7 | 0.0862 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.215, 'max_latency_us': 6.215, 'min_latency_us': 2.397, 'op_name': 'main_kernel', 'sample_count': 2} | msprof | 0.3625 | 1 | 256 | 1e-06 | 128 | bfloat16 |
| train-s96 | 11.6780 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143418_008444_EngramGateConvBwdOp/artifacts/msprof/main_kernel_jqrht2q3 | 0.3058 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 11.678, 'max_latency_us': 11.678, 'min_latency_us': 3.523, 'op_name': 'main_kernel', 'sample_count': 2} | msprof | 1.3000 | 3 | 768 | 1e-06 | 96 | bfloat16 |

## EngramDecodeFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | conv_kernel_size | conv_len | d | d_mem | dilation | eps | max_conv_len | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| serving-b1 | 3.0000 | 0.2613 | 0.9592 | 1.6000 | 11.8272 | 0.0040 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143418_008444_EngramGateConvBwdOp/artifacts/msprof/engram_decode_step_kernel_kn2hiu6o | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 3.0, 'max_latency_us': 3.0, 'min_latency_us': 2.936, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2} | msprof | 1 | 4 | 12 | 256 | 512 | 3 | 1e-06 | 12 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
| serving-b4 | 4.7770 | 33.0802 | 13.2711 | 1.6000 | 378.4704 | 7.0242 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143418_008444_EngramGateConvBwdOp/artifacts/msprof/engram_decode_step_kernel_74yznrab | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 4.777, 'max_latency_us': 4.777, 'min_latency_us': 3.64, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2} | msprof | 4 | 4 | 20 | 512 | 1024 | 5 | 1e-06 | 20 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
| serving-b8 | 6.4130 | 6.7987 | 12.0250 | 1.6000 | 378.4704 | 1.3081 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143418_008444_EngramGateConvBwdOp/artifacts/msprof/engram_decode_step_kernel_322d0l80 | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 6.413, 'max_latency_us': 6.413, 'min_latency_us': 2.944, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2} | msprof | 8 | 4 | 18 | 256 | 512 | 3 | 1e-06 | 18 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | bfloat16 |
| decode-partial-cache | 3.0650 | 0.4463 | 1.0994 | 1.6000 | 11.8272 | 0.0079 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_143418_008444_EngramGateConvBwdOp/artifacts/msprof/PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021_1urdi0mp | {'captured_op_count': 3, 'captured_op_names': ['PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021', 'engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 3.065, 'max_latency_us': 3.165, 'min_latency_us': 2.996, 'op_name': 'PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021', 'sample_count': 3} | msprof | 2 | 4 | 5 | 256 | 512 | 3 | 1e-06 | 12 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
