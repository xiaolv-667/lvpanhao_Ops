# NPU Benchmark Report
Generated: 2026-11-23 14:42:53

## Environment

- **Torch version**: 2.13.0+cpu
- **torch_npu version**: 2.13.0.rc1
- **NPU model**: Ascend950PR_9579
- **NPU count**: 1
- **Profiling mode**: events

## EngramGateConvFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | tflops | bandwidth_tbs | prof_mode | M | d | eps | seq_len | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train-s64 | 48.8560 | N/A | 0.0387 | 0.0136 | events | 2 | 512 | 1e-06 | 64 | float16 |
| train-s128 | 47.1990 | N/A | 0.0201 | 0.0071 | events | 1 | 256 | 1e-06 | 128 | bfloat16 |
| train-s96 | 47.4448 | N/A | 0.1347 | 0.0469 | events | 3 | 768 | 1e-06 | 96 | bfloat16 |

## EngramGateConvBwdOp

### tileops

| Label | Latency (us) | Ratio (%) | tflops | bandwidth_tbs | prof_mode | M | d | eps | seq_len | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train-s64 | 59.5870 | N/A | 0.0753 | 0.0179 | events | 2 | 512 | 1e-06 | 64 | float16 |
| train-s128 | 59.6399 | N/A | 0.0378 | 0.0090 | events | 1 | 256 | 1e-06 | 128 | bfloat16 |
| train-s96 | 64.8757 | N/A | 0.2340 | 0.0550 | events | 3 | 768 | 1e-06 | 96 | bfloat16 |

## EngramDecodeFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | tflops | bandwidth_tbs | prof_mode | batch | conv_kernel_size | conv_len | d | d_mem | dilation | eps | max_conv_len | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| serving-b1 | 45.9035 | N/A | 0.0116 | 0.0118 | events | 1 | 4 | 12 | 256 | 512 | 3 | 1e-06 | 12 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
| serving-b4 | 46.6311 | N/A | 0.1809 | 0.0490 | events | 4 | 4 | 20 | 512 | 1024 | 5 | 1e-06 | 20 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
| serving-b8 | 46.1641 | N/A | 0.0919 | 0.0150 | events | 8 | 4 | 18 | 256 | 512 | 3 | 1e-06 | 18 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | bfloat16 |
| decode-partial-cache | 54.9491 | N/A | 0.0193 | 0.0100 | events | 2 | 4 | 5 | 256 | 512 | 3 | 1e-06 | 12 | {'block_k': 256, 'block_m': 16, 'block_n': 64, 'num_stages': 2} | float16 |
