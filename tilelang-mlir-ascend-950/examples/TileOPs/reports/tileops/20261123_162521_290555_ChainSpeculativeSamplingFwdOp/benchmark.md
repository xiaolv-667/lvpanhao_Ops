# NPU Benchmark Report
Generated: 2026-11-23 16:27:20

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
| llama-8b-b16-n4-float32 | 1245.8091 | 0.9763 | 1.0147 | 1.6000 | 11.8272 | 0.0159 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_162521_290555_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_aa6sfz6s | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 1245.809082, 'max_latency_us': 1245.809082, 'min_latency_us': 1245.809082, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | N/A | N/A | 16 | 4 | 128256 |
| llama-8b-b64-n4-float32 | 2472.0908 | 2.0230 | 0.9943 | 1.6000 | 11.8272 | 0.0322 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_162521_290555_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_radyzpz5 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2472.09082, 'max_latency_us': 2472.09082, 'min_latency_us': 2472.09082, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | N/A | N/A | 64 | 4 | 128256 |
| ds-v3-mtp-n1-float32 | 2529.2458 | 1.5356 | 1.1558 | 1.6000 | 11.8272 | 0.0284 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_162521_290555_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_fooyjju5 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2529.24585, 'max_latency_us': 2529.24585, 'min_latency_us': 2529.24585, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | N/A | N/A | 64 | 1 | 129280 |
| ds-v3-mtp-n3-float32 | 2542.7910 | 1.9087 | 1.0117 | 1.6000 | 11.8272 | 0.0309 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_162521_290555_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_aweydobr | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2542.791016, 'max_latency_us': 2542.791016, 'min_latency_us': 2542.791016, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | N/A | N/A | 64 | 3 | 129280 |
| qwen3-235b-eagle-n5-float32 | 1427.4860 | 2.0947 | 0.9632 | 1.6000 | 11.8272 | 0.0323 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_162521_290555_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_sh6chkir | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 1427.485962, 'max_latency_us': 1427.485962, 'min_latency_us': 1427.485962, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | N/A | N/A | 32 | 5 | 151936 |
| llama-8b-b17-n8-float32 | 1259.7440 | 1.0608 | 0.9980 | 1.6000 | 11.8272 | 0.0169 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_162521_290555_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_5rxskl02 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 1259.744019, 'max_latency_us': 1259.744019, 'min_latency_us': 1259.744019, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | N/A | N/A | 17 | 8 | 128256 |
| llama2-7b-b256-n1-float32 | 1666.3409 | N/A | N/A | N/A | N/A | N/A | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261123_162521_290555_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_3a8jx6kn | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 1666.340942, 'max_latency_us': 1666.340942, 'min_latency_us': 1666.340942, 'op_name': 'main_kernel', 'sample_count': 1} | msprof | 0.0197 | 0.0147 | 256 | 1 | 32000 |
