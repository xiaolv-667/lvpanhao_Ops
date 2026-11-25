# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""`EngramDecodeFwdOp` — Stage 4 tuned drop-in for the two-launch NPU decode step.

One decode step of the Engram stack, end to end (same operator contract as the
frozen Stage-3 module this replaces):

    k = e_t @ W_K;  v = e_t @ W_V                            # launch 1 (Cube)
    new_conv_state[:, :mcl-1, :] = conv_state[:, 1:, :]      # launch 1 (AIV copy)
    h_norm = RMSNorm(h_t, rms_w_h);  k_norm = RMSNorm(k, rms_w_h)
    alpha  = sigmoid(dot(h_norm, k_norm) / sqrt(d));  v_hat = alpha * v
    vhat_n = RMSNorm(v_hat, rms_w_v)
    new_conv_state[:, mcl-1, :] = vhat_n                     # launch 2 (AIV)
    conv_out = sum_p conv_w[p] * new_conv_state_taps
    y_t = SiLU(conv_out) + v_hat

Stage 4 result (msopprof kernel-only `Task Duration(us)`, T_op = sum of the two
launches, median of 3 independent capture processes, ascend-950pr-9579):

    workload              baseline   this file
    serving-b8  bf16        9.466  ->  6.385   (-32.5%, the cache-shift rewrite)
    serving-b4  fp16        8.297  ->  8.515   (  0.0%, device code byte-identical)
    serving-b1  fp16        5.989  ->  6.094   (  0.0%, device code byte-identical)
    decode-partial-cache    6.186  ->  6.258   (  0.0%, device code byte-identical)

The +1..3% shown for the three unchanged shapes is measurement drift: their emitted
`device_kernel.asc` equals the Stage-3 baseline byte for byte (verified shape by
shape in `perf_opt/experiments/verify_codegen.py`), and a same-session paired A/B
flips the sign on serving-b4 (project launch -1.7%, T_op -3.7%).  The run-to-run
band on this host is per-launch +-8..12% / T_op +-4%.

The only optimised point is the cache shift (see `SHIFT_*` below): the baseline
serialised `batch` copy pairs inside `n_tiles` blocks, which cost 3.67 us on
serving-b8 (project launch 6.672 -> 2.999 us with the shift removed).  Everything
else measured at its floor:

* project launch = launch floor 2.89 us (same-geometry probe, grid 4/8/16 = 2.69-2.96 us) + 1.31 us weight
  stream (2 MB over 16 blocks = 98 GB/s/core, the documented MTE2 rate for this
  burst pattern) + 0.49 us mix(AIC+AIV) promotion + 0.19 us shift bytes;
* step launch = launch floor 2.69 us (probe) + 0.7 us work, and it is insensitive
  to the lane count {128, 256, 512} and to the tap descriptor count.

The stage-3 modules' semantics are untouched: same fp32 staging buffer
`kv (2, batch, d_padded)`, same arithmetic order, same dtype rounding points, and
the history shift is still a bit-exact dtype copy.  `python <this file> --level
all` runs the operator-level L0/L1/boundary harness against the repository torch
reference.

run: python EngramDecodeFwdOp.py --level L0|L1|boundary|all
"""
