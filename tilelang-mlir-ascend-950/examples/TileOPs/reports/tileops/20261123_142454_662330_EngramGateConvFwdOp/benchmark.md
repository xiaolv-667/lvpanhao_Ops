# NPU Benchmark Report
Generated: 2026-11-23 14:28:07

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
| train-s64 | 8.4660 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_142454_662330_EngramGateConvFwdOp/artifacts/msprof/main_kernel_ol_25qzv | 0.0784 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 8.466, 'max_latency_us': 8.466, 'min_latency_us': 8.466, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | 0.2233 | 2 | 512 | 1e-06 | 64 | float16 |
| train-s128 | 7.3480 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_142454_662330_EngramGateConvFwdOp/artifacts/msprof/main_kernel_i2j3roh7 | 0.0453 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 7.348, 'max_latency_us': 7.348, 'min_latency_us': 7.348, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | 0.1291 | 1 | 256 | 1e-06 | 128 | bfloat16 |
| train-s96 | 11.1170 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_142454_662330_EngramGateConvFwdOp/artifacts/msprof/main_kernel_4jmsi4fk | 0.2002 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 11.117, 'max_latency_us': 11.117, 'min_latency_us': 11.117, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | 0.5749 | 3 | 768 | 1e-06 | 96 | bfloat16 |

## EngramGateConvBwdOp

### tileops

| Label | Latency (us) | Ratio (%) | artifact_dir | bandwidth_tbs | msprof | prof_mode | tflops | M | d | eps | seq_len | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train-s64 | 7.5220 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_142454_662330_EngramGateConvFwdOp/artifacts/msprof/main_kernel_1njlhwzl | 0.1421 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 7.522, 'max_latency_us': 7.522, 'min_latency_us': 3.162, 'op_name': 'main_kernel', 'sample_count': 2} | msprof | 0.5964 | 2 | 512 | 1e-06 | 64 | float16 |
| train-s128 | 6.2810 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_142454_662330_EngramGateConvFwdOp/artifacts/msprof/main_kernel_3uu6hxws | 0.0853 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.281, 'max_latency_us': 6.281, 'min_latency_us': 2.531, 'op_name': 'main_kernel', 'sample_count': 2} | msprof | 0.3587 | 1 | 256 | 1e-06 | 128 | bfloat16 |
| train-s96 | 11.2260 | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_142454_662330_EngramGateConvFwdOp/artifacts/msprof/main_kernel_lp3w9gwn | 0.3181 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 11.226, 'max_latency_us': 11.226, 'min_latency_us': 3.371, 'op_name': 'main_kernel', 'sample_count': 2} | msprof | 1.3524 | 3 | 768 | 1e-06 | 96 | bfloat16 |

## EngramDecodeFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | bandwidth_tbs | tflops | batch | conv_kernel_size | conv_len | d | d_mem | dilation | eps | max_conv_len | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| serving-b1 | 3.2640 | 0.2525 | 0.9126 | 1.6000 | 11.8272 | 0.0037 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_142454_662330_EngramGateConvFwdOp/artifacts/msprof/engram_decode_step_kernel_c7ektz7g | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 3.264, 'max_latency_us': 3.264, 'min_latency_us': 2.931, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2} | msprof | N/A | N/A | 1 | 4 | 12 | 256 | 512 | 3 | 1e-06 | 12 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
| serving-b4 | 5.0920 | 30.9128 | 13.3230 | 1.6000 | 378.4704 | 6.5896 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_142454_662330_EngramGateConvFwdOp/artifacts/msprof/engram_decode_step_kernel_4qcgxh28 | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 5.092, 'max_latency_us': 5.092, 'min_latency_us': 3.234, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2} | msprof | N/A | N/A | 4 | 4 | 20 | 512 | 1024 | 5 | 1e-06 | 20 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
| serving-b8 | 6.2890 | 7.0027 | 11.9048 | 1.6000 | 378.4704 | 1.3339 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_142454_662330_EngramGateConvFwdOp/artifacts/msprof/engram_decode_step_kernel_lit02a2j | {'captured_op_count': 2, 'captured_op_names': ['engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 6.289, 'max_latency_us': 6.289, 'min_latency_us': 3.058, 'op_name': 'engram_decode_step_kernel', 'sample_count': 2} | msprof | N/A | N/A | 8 | 4 | 18 | 256 | 512 | 3 | 1e-06 | 18 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | bfloat16 |
| decode-partial-cache | 3.0520 | N/A | N/A | N/A | N/A | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_142454_662330_EngramGateConvFwdOp/artifacts/msprof/PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021_h7hxps8x | {'captured_op_count': 3, 'captured_op_names': ['PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021', 'engram_decode_step_kernel', 'engram_project_kernel'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 3.052, 'max_latency_us': 3.129, 'min_latency_us': 2.747, 'op_name': 'PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021', 'sample_count': 3} | msprof | 0.1798 | 0.3476 | 2 | 4 | 5 | 256 | 512 | 3 | 1e-06 | 12 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
