# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Repro: an illegal ``max_conv_len < dilation*(conv_kernel_size-1)`` shape makes the
step kernel read a *negative* cache row instead of failing.

Target: ascend-950pr-9579 / dav-3510, tilelang 0.1.15 + CANN 9.2.1.

Found while writing the Stage 3 boundary tests of ``_engram_step_kernel``: the kernel
carries no tap-index guard by design (DESIGN.md 1.6.2 #6), because the operator's
shape rule ``max_conv_len >= dilation*(w-1)`` (checked in the upstream ``__init__``,
decode.py:293-298) keeps ``max_conv_len - (w-1-p)*dilation`` inside
``[0, max_conv_len-1]``.  Violate the rule and nothing crashes:

    the kernel reads conv_state[bx, negative_row, :], which is the *previous batch
    row* of the same tensor (or memory before it for bx == 0), while the torch
    reference silently skips that tap (``if 0 <= state_idx < max_conv_len``).
    No runtime error, and the magnitude is not reliably out of range: on the shape
    below batch 0 reads before the tensor (NaN output) while batch 1 is off by
    ~3e-3 -- *inside* the fp16 tolerance 0.05, i.e. a silent semantic error.

Evidence (asserted below), shape b=2, d=256, max_conv_len=7, w=4, dilation=3 (needs
max_conv_len >= 9):

  A  the illegal launch returns; batch 1's y_t differs from the reference by far more
     than rounding (and, at this scale, still less than the fp16 tolerance, so the
     deviation is silent), while batch 0 reads memory before the tensor and yields
     NaN;
  B  batch 1 read ``conv_state[0, 5]`` (its own row -2) for tap p=0: feeding that row
     through the reference as a legal single-row problem (taps at cache rows 0/3/6)
     reproduces the kernel output to the tolerance -- so the corruption is exactly
     the missing guard, not an arithmetic difference;
  C  the legal control (max_conv_len=10) matches the reference: the kernel is only
     wrong when the shape rule is broken.

Consequence for the port: keep the upstream shape check, or re-add the guard.

run:
  cd examples/engram_decode/_engram_step_kernel
  source <repo>/.agents/platforms/toolchains/ascend/env-950pr.sh
  ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python \
      repro/TRAP-950-step-tap-guard.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _engram_step_kernel as M  # noqa: E402

D = 256
ATOL, RTOL = M.TOL["float16"]


def _launch(batch, mcl, w, dilation, dtype, tensors):
    """Run the step kernel on already-padded tensors; returns (y, new_conv_state)."""
    d_pad = M._align_up(D)
    kernel = M._build_step(
        batch, D, d_pad, mcl, w, dilation, 1e-6, M._dtype_name(dtype),
        M.default_config(d_pad)["threads"],
    )
    y = torch.full((batch, d_pad), float("nan"), dtype=dtype, device="npu")
    state = torch.zeros(batch, mcl, d_pad, dtype=dtype, device="npu")
    kernel(
        tensors["h"].to("npu"), tensors["kv"].to("npu"), tensors["state"].to("npu"),
        tensors["wh"].to("npu"), tensors["wv"].to("npu"), tensors["cw"].to("npu"),
        y, state,
    )
    torch.npu.synchronize()
    return y.cpu(), state.cpu()


def main():
    batch, d, w, dilation = 2, D, 4, 3
    dtype = torch.float16
    torch.manual_seed(0)
    e_t = torch.randn(batch, d, dtype=dtype) * 0.1
    h_t = torch.randn(batch, d, dtype=dtype)
    w_k = torch.randn(d, d, dtype=dtype) * 0.02
    w_v = torch.randn(d, d, dtype=dtype) * 0.02
    rms_w_h = (1.0 + 0.1 * torch.randn(d)).to(dtype)
    rms_w_v = (1.0 + 0.1 * torch.randn(d)).to(dtype)
    conv_w = torch.randn(w, d, dtype=dtype) * 0.02

    # --- C: legal control, max_conv_len = dilation*(w-1) + 1 -------------------
    mcl_ok = 10
    conv_state = torch.randn(batch, mcl_ok, d, dtype=dtype) * 0.1
    y_ref, state_ref = M.golden_engram_decode(
        e_t, h_t, conv_state, w_k, w_v, rms_w_h, rms_w_v, conv_w, mcl_ok, dilation
    )
    tensors = {
        "h": h_t, "state": conv_state,
        "kv": M.host_kv(e_t, w_k, w_v, D),
        "wh": rms_w_h, "wv": rms_w_v, "cw": conv_w,
    }
    y_ok, state_ok = _launch(batch, mcl_ok, w, dilation, dtype, tensors)
    ok_y = (y_ok[:, :d].float() - y_ref.float()).abs().max().item()
    ok_s = (state_ok[:, mcl_ok - 1, :d].float()
            - state_ref[:, mcl_ok - 1, :d].float()).abs().max().item()
    torch.testing.assert_close(y_ok[:, :d].float(), y_ref.float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        state_ok[:, mcl_ok - 1, :d].float(), state_ref[:, mcl_ok - 1, :d].float(),
        atol=ATOL, rtol=RTOL,
    )
    print(f"[C legal mcl={mcl_ok}] PASS y_diff={ok_y:.3e} state_diff={ok_s:.3e}")

    # --- A: the same data at max_conv_len = 7 < 9 ------------------------------
    mcl_bad = 7
    short = conv_state[:, mcl_ok - mcl_bad:, :].contiguous()  # last 7 rows
    y_ref_bad, _ = M.golden_engram_decode(
        e_t, h_t, short, w_k, w_v, rms_w_h, rms_w_v, conv_w, mcl_bad, dilation
    )
    tensors["state"] = short
    y_bad, _ = _launch(batch, mcl_bad, w, dilation, dtype, tensors)
    ref_f = y_ref_bad.float()
    bad_y = (y_bad[:, :d].float() - ref_f).abs()
    tol = ATOL + RTOL * ref_f.abs()
    # A NaN (or a wild value) is also a mismatch: comparison operators make it one.
    exceed = int((~(bad_y <= tol)).sum().item())
    nan0 = int(torch.isnan(y_bad[0, :d]).sum())
    worst1 = bad_y[1:2].max().item()
    print(
        f"[A illegal mcl={mcl_bad}] returned without error, batch-1 y_diff={worst1:.3e} "
        f"(tolerance {ATOL}), mismatched elements={exceed}, "
        f"NaN in batch 0 (reads before the tensor)={nan0}"
    )
    # Far above the ~1 ulp an equivalent expression would produce, yet below the fp16
    # tolerance: the wrong row is invisible to the op-level gate.
    assert worst1 > 1e-4, "expected the missing tap guard to change the result"
    assert worst1 < ATOL, "expected the corruption to stay inside the fp16 tolerance"
    assert exceed > 0
    assert nan0 > 0, "batch 0 was expected to read memory before conv_state"

    # --- B: name the row that was read: conv_state[1, -2] == conv_state[0, 5] ---
    # A legal single-row problem at max_conv_len=9 reads taps at rows 0 / 3 / 6, so
    # putting batch 0's row 5 where tap p=0 lands reproduces the kernel's batch 1.
    witness = torch.zeros(1, 9, d, dtype=dtype)
    witness[0, 0] = short[0, 5]
    witness[0, 3] = short[1, 1]
    witness[0, 6] = short[1, 4]
    y_wit, _ = M.golden_engram_decode(
        e_t[1:], h_t[1:], witness, w_k, w_v, rms_w_h, rms_w_v, conv_w, 9, dilation
    )
    wit = (y_bad[1:2, :d].float() - y_wit.float()).abs().max().item()
    torch.testing.assert_close(
        y_bad[1:2, :d].float(), y_wit.float(), atol=ATOL, rtol=RTOL
    )
    print(
        f"[B witness] batch 1 matches the reference when tap p=0 is fed "
        f"conv_state[0, 5] (row -2): diff={wit:.3e}"
    )
    print("RESULT: TRAP CONFIRMED (illegal max_conv_len is a silent wrong answer)")


if __name__ == "__main__":
    main()
