# NPU Benchmark Report
Generated: 2026-11-25 17:18:00

## Environment

- **Torch version**: 2.13.0+cpu
- **torch_npu version**: 2.13.0.rc1
- **NPU model**: Ascend950PR_9579
- **NPU count**: 1
- **Profiling mode**: msprof

## ChainSpeculativeSamplingFwdOp

### tileops

| Label | Latency (us) | Derived TFLOP/s | Derived TB/s | artifact_dir | msprof | prof_mode | batch | num_draft | vocab |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-8b-b16-n4-float32 | 43.5180 | 0.1415 | 0.1886 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261125_171428_406993_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_m3fkz7g4 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 43.518002, 'max_latency_us': 87.406998, 'min_latency_us': 9.268, 'op_name': 'main_kernel', 'sample_count': 3, 't_op_breakdown': {'main_kernel': 140.193}, 't_op_us': 140.193} | msprof | 16 | 4 | 128256 |
| llama-8b-b64-n4-float32 | 84.8480 | 0.2902 | 0.3870 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261125_171428_406993_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_r4g5hmga | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 84.848, 'max_latency_us': 292.019989, 'min_latency_us': 16.75, 'op_name': 'main_kernel', 'sample_count': 3, 't_op_breakdown': {'main_kernel': 393.617989}, 't_op_us': 393.617989} | msprof | 64 | 4 | 128256 |
| ds-v3-mtp-n1-float32 | 83.0640 | 0.2988 | 0.3985 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261125_171428_406993_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_1ak__phk | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 83.064003, 'max_latency_us': 221.676987, 'min_latency_us': 8.365, 'op_name': 'main_kernel', 'sample_count': 3, 't_op_breakdown': {'main_kernel': 313.10599}, 't_op_us': 313.10599} | msprof | 64 | 1 | 129280 |
| ds-v3-mtp-n3-float32 | 86.2490 | 0.2878 | 0.3838 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261125_171428_406993_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_oma6kut4 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 86.249001, 'max_latency_us': 246.296997, 'min_latency_us': 13.547, 'op_name': 'main_kernel', 'sample_count': 3, 't_op_breakdown': {'main_kernel': 346.092998}, 't_op_us': 346.092998} | msprof | 64 | 3 | 129280 |
| qwen3-235b-eagle-n5-float32 | 47.2840 | 0.3085 | 0.4114 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261125_171428_406993_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_9mj6dy6z | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 47.284, 'max_latency_us': 314.444977, 'min_latency_us': 12.608, 'op_name': 'main_kernel', 'sample_count': 3, 't_op_breakdown': {'main_kernel': 374.336977}, 't_op_us': 374.336977} | msprof | 32 | 5 | 151936 |
| llama-8b-b17-n8-float32 | 44.2250 | 0.1479 | 0.1973 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261125_171428_406993_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_sdheoif7 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 44.224998, 'max_latency_us': 96.153999, 'min_latency_us': 15.005, 'op_name': 'main_kernel', 'sample_count': 3, 't_op_breakdown': {'main_kernel': 155.383997}, 't_op_us': 155.383997} | msprof | 17 | 8 | 128256 |
| llama2-7b-b256-n1-float32 | 191.3710 | 0.1284 | 0.1713 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261125_171428_406993_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_rq93u58c | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 191.371002, 'max_latency_us': 191.376999, 'min_latency_us': 16.843, 'op_name': 'main_kernel', 'sample_count': 3, 't_op_breakdown': {'main_kernel': 399.591001}, 't_op_us': 399.591001} | msprof | 256 | 1 | 32000 |
