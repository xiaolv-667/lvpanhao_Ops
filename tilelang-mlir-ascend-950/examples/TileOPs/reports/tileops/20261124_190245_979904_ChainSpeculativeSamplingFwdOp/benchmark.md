# NPU Benchmark Report
Generated: 2026-11-24 19:04:50

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
| llama-8b-b16-n4-float32 | 338.7830 | 0.0182 | 0.0242 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_190245_979904_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_d1w8x98x | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 338.78302, 'max_latency_us': 338.78302, 'min_latency_us': 338.78302, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 338.78302}, 't_op_us': 338.78302} | msprof | 16 | 4 | 128256 |
| llama-8b-b64-n4-float32 | 702.1810 | 0.0351 | 0.0468 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_190245_979904_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_ekxjbey8 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 702.180969, 'max_latency_us': 702.180969, 'min_latency_us': 702.180969, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 702.180969}, 't_op_us': 702.180969} | msprof | 64 | 4 | 128256 |
| ds-v3-mtp-n1-float32 | 661.3881 | 0.0375 | 0.0500 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_190245_979904_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_atax27v4 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 661.388062, 'max_latency_us': 661.388062, 'min_latency_us': 661.388062, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 661.388062}, 't_op_us': 661.388062} | msprof | 64 | 1 | 129280 |
| ds-v3-mtp-n3-float32 | 714.6411 | 0.0347 | 0.0463 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_190245_979904_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_nq5xi0j3 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 714.641052, 'max_latency_us': 714.641052, 'min_latency_us': 714.641052, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 714.641052}, 't_op_us': 714.641052} | msprof | 64 | 3 | 129280 |
| qwen3-235b-eagle-n5-float32 | 405.6610 | 0.0360 | 0.0479 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_190245_979904_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_77hp3f5h | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 405.66098, 'max_latency_us': 405.66098, 'min_latency_us': 405.66098, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 405.66098}, 't_op_us': 405.66098} | msprof | 32 | 5 | 151936 |
| llama-8b-b17-n8-float32 | 347.9170 | 0.0188 | 0.0251 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_190245_979904_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_t9nw4wj8 | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 347.916992, 'max_latency_us': 347.916992, 'min_latency_us': 347.916992, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 347.916992}, 't_op_us': 347.916992} | msprof | 17 | 8 | 128256 |
| llama2-7b-b256-n1-float32 | 629.3960 | 0.0390 | 0.0521 | /workspace/tilelang-mlir-ascend-950/examples/TileOPs/reports/tileops/20261124_190245_979904_ChainSpeculativeSamplingFwdOp/artifacts/msprof/main_kernel_haeslhin | {'captured_op_count': 1, 'captured_op_names': ['main_kernel'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 629.395996, 'max_latency_us': 629.395996, 'min_latency_us': 629.395996, 'op_name': 'main_kernel', 'sample_count': 1, 't_op_breakdown': {'main_kernel': 629.395996}, 't_op_us': 629.395996} | msprof | 256 | 1 | 32000 |
