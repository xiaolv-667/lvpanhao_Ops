# NPU Benchmark Report
Generated: 2026-11-23 16:24:18

## Environment

- **Torch version**: 2.13.0+cpu
- **torch_npu version**: 2.13.0.rc1
- **NPU model**: Ascend950PR_9579
- **NPU count**: 1
- **Profiling mode**: msprof

## ChainSpeculativeSamplingFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | bandwidth_tbs | tflops | batch | num_draft | vocab |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-8b-b16-n4-float32 | 1247.4030 | 0.9747 | 1.0151 | 1.6000 | 11.8272 | 0.0158 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_162216_099261_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_b6e_iprr | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 1247.402954, 'max_latency_us': 1247.402954, 'min_latency_us': 1247.402954, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | N/A | N/A | 16 | 4 | 128256 |
| llama-8b-b64-n4-float32 | 2470.4250 | 2.0247 | 0.9942 | 1.6000 | 11.8272 | 0.0322 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_162216_099261_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel__i6mn77j | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2470.425049, 'max_latency_us': 2470.425049, 'min_latency_us': 2470.425049, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | N/A | N/A | 64 | 4 | 128256 |
| ds-v3-mtp-n1-float32 | 2538.1062 | 1.5294 | 1.1564 | 1.6000 | 11.8272 | 0.0283 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_162216_099261_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_rv5q2ilx | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2538.106201, 'max_latency_us': 2538.106201, 'min_latency_us': 2538.106201, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | N/A | N/A | 64 | 1 | 129280 |
| ds-v3-mtp-n3-float32 | 2533.7170 | 1.9154 | 1.0117 | 1.6000 | 11.8272 | 0.0310 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_162216_099261_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_qn0lhivc | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2533.717041, 'max_latency_us': 2533.717041, 'min_latency_us': 2533.717041, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | N/A | N/A | 64 | 3 | 129280 |
| qwen3-235b-eagle-n5-float32 | 1421.7251 | 2.1045 | 0.9626 | 1.6000 | 11.8272 | 0.0324 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_162216_099261_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_a5_ohgld | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 1421.725098, 'max_latency_us': 1421.725098, 'min_latency_us': 1421.725098, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | N/A | N/A | 32 | 5 | 151936 |
| llama-8b-b17-n8-float32 | 1255.9879 | 1.0632 | 0.9987 | 1.6000 | 11.8272 | 0.0170 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_162216_099261_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_5vml6y8b | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 1255.987915, 'max_latency_us': 1255.987915, 'min_latency_us': 1255.987915, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | N/A | N/A | 17 | 8 | 128256 |
| llama2-7b-b256-n1-float32 | 1684.6969 | N/A | N/A | N/A | N/A | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_162216_099261_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_k480cq5_ | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 1684.696899, 'max_latency_us': 1684.696899, 'min_latency_us': 1684.696899, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | 0.0195 | 0.0146 | 256 | 1 | 32000 |
