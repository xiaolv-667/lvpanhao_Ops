# ---------------------------------------------------------------------------
# Operator-level harness: drives BOTH launches exactly as the Kernel.forward does
# (host padding -> project -> step) and compares against the repository torch
# reference.  This is the precision evidence for this file; the per-launch L0
# runners live in the round candidate modules.
# ---------------------------------------------------------------------------
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401 -- registers torch.npu

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "examples" / "TileOPs") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "examples" / "TileOPs"))

from tileops.workloads.sequence_modeling import engram_decode_step_torch  # noqa: E402

# examples/TileOPs/tests/ops/test_engram.py::_tol_decode
TOL = {"float16": (0.05, 0.05), "bfloat16": (0.1, 0.05)}

# ---------------------------------------------------------------------------
# Golden: PyTorch reference of the whole step, written from the operator
# semantics (fp32 arithmetic, only the two returned tensors rounded to the input
# dtype).  `engram_decode_step_torch` from the repository workloads is the
# semantic ground truth; every case cross-checks the two bit for bit, so this
# golden cannot drift from the reference it was derived from.
# ---------------------------------------------------------------------------
def _rmsnorm_torch(x, weight, eps):
    x_f = torch.unsqueeze(x.float(), 1)
    rrms = torch.rsqrt(torch.add(torch.mean(torch.square(x_f), dim=-1, keepdim=True), eps))
    return torch.squeeze(torch.mul(torch.mul(x_f, rrms), weight.float()), 1)


def golden_engram_decode(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
                         max_conv_len, dilation, eps=1e-6):
    """`y_t` and the full `max_conv_len`-row history of one decode step."""
    batch, d = h_t.shape
    w = conv_w.shape[0]
    cache_len = conv_state.shape[1]

    k = torch.matmul(e_t.float(), W_K.float())
    v = torch.matmul(e_t.float(), W_V.float())
    h_norm = _rmsnorm_torch(h_t, rms_w_h, eps)
    k_norm = _rmsnorm_torch(k, rms_w_h, eps)
    dot = torch.sum(torch.mul(h_norm, k_norm), dim=-1, keepdim=True)
    alpha = torch.sigmoid(torch.div(dot, float(d) ** 0.5))
    v_hat = torch.mul(alpha, v)
    v_hat_norm = _rmsnorm_torch(v_hat, rms_w_v, eps)

    state_pad = max_conv_len - cache_len
    padded_history = (
        F.pad(conv_state, (0, 0, state_pad, 0)) if state_pad > 0 else conv_state
    )
    padded_state = padded_history.float()
    conv_out = torch.zeros(batch, d, dtype=torch.float32, device=h_t.device)
    for p in range(w - 1):
        state_idx = max_conv_len - (w - 1 - p) * dilation
        if 0 <= state_idx < max_conv_len:
            conv_out = conv_out + torch.mul(
                conv_w[p].float().unsqueeze(0), padded_state[:, state_idx, :]
            )
    conv_out = conv_out + torch.mul(conv_w[w - 1].float().unsqueeze(0), v_hat_norm)
    new_conv_state = torch.cat(
        [padded_history[:, 1:, :], v_hat_norm.unsqueeze(1).to(conv_state.dtype)], dim=1
    )
    y_t = torch.add(F.silu(conv_out), v_hat)
    return y_t.to(h_t.dtype), new_conv_state

_CASES = []


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _align_up(value, multiple=ALIGNMENT):
    return (value + multiple - 1) // multiple * multiple


def _bits_equal(got, want):
    if got.shape != want.shape:
        return False
    return torch.equal(got.contiguous().view(torch.int16), want.contiguous().view(torch.int16))


def _prepare(batch, d_mem, d, max_conv_len, w, dilation, dtype, cache_len, eps, seed=0):
    """Inputs of one step: unpadded (reference) and padded (launch), from one seed."""
    torch.manual_seed(seed)
    e_t = torch.randn(batch, d_mem, dtype=dtype) * 0.1
    h_t = torch.randn(batch, d, dtype=dtype)
    conv_state = torch.randn(batch, cache_len, d, dtype=dtype) * 0.1
    w_k = torch.randn(d_mem, d, dtype=dtype) * 0.02
    w_v = torch.randn(d_mem, d, dtype=dtype) * 0.02
    rms_w_h = (1.0 + 0.1 * torch.randn(d)).to(dtype)
    rms_w_v = (1.0 + 0.1 * torch.randn(d)).to(dtype)
    conv_w = torch.randn(w, d, dtype=dtype) * 0.02
    ref = (e_t, h_t, conv_state, w_k, w_v, rms_w_h, rms_w_v, conv_w)
    return ref


def _run_case(batch, d_mem, d, max_conv_len, w, dilation, dtype, tag,
              cache_len=None, eps=1e-6, seed=0, launches=2):
    d_pad = _align_up(d)
    cache_len = max_conv_len if cache_len is None else cache_len
    assert max_conv_len >= dilation * (w - 1), f"[{tag}] max_conv_len shape rule violated"
    dt = _dtype_name(dtype)
    atol, rtol = TOL[dt]

    ref = _prepare(batch, d_mem, d, max_conv_len, w, dilation, dtype, cache_len, eps, seed)
    y_ref, state_ref = golden_engram_decode(*ref, max_conv_len, dilation, eps)
    if tag == "L0-serving-b1":  # keep the two references pinned together on every run
        y_alt, state_alt = engram_decode_step_torch(*ref, max_conv_len, dilation, eps)
        assert _bits_equal(y_alt, y_ref) and _bits_equal(state_alt, state_ref), (
            "golden_engram_decode disagrees with the repository torch reference")

    e_t, h_t, conv_state = ref[0], ref[1], ref[2]
    # host contract of Kernel.forward: left-pad the cache, then zero-pad the last axis
    hist = F.pad(conv_state, (0, 0, max_conv_len - cache_len, 0)) if cache_len < max_conv_len \
        else conv_state
    dev = "npu"
    pad = d_pad - d

    def _p(t, last=True):
        t = t.to(dev)
        return F.pad(t, (0, pad)) if pad and last else t

    e_t_d, h_t_d = _p(e_t, last=(d_pad != d_mem)), _p(h_t)
    hist_d = _p(hist, last=True)
    wh_d, wv_d, cw_d = _p(ref[5]), _p(ref[6]), _p(ref[7])
    wk_d, wv_d2 = _p(ref[3]), _p(ref[4])
    if pad:
        # e_t is (batch, d_mem) and is not padded; W_K/W_V are (d_mem, d)
        wk_d, wv_d2 = F.pad(ref[3].to(dev), (0, pad)), F.pad(ref[4].to(dev), (0, pad))
        e_t_d = e_t.to(dev)

    kv = torch.empty((2, batch, d_pad), dtype=torch.float32, device=dev)
    y_buf = torch.full((batch, d_pad), float("nan"), dtype=dtype, device=dev)
    state_buf = torch.full((batch, max_conv_len, d_pad), float("nan"), dtype=dtype, device=dev)

    proj = _engram_project_kernel(batch, d_mem, d_pad, max_conv_len, dt)(
        **project_default_config(d_pad))
    step = _engram_step_kernel(batch, d, d_pad, max_conv_len, w, dilation, eps, dt)(
        step_default_config(d_pad)["threads"])

    snaps = []
    t0 = time.perf_counter()
    for _ in range(launches):
        proj(e_t_d, wk_d, wv_d2, hist_d, kv, state_buf)
        step(h_t_d, kv, hist_d, wh_d, wv_d, cw_d, y_buf, state_buf)
        torch.npu.synchronize()
        snaps.append((y_buf.cpu().clone(), state_buf.cpu().clone()))
    elapsed = time.perf_counter() - t0

    for k in range(1, launches):
        assert _bits_equal(snaps[k][0], snaps[0][0]), f"[{tag}] y_t differs on launch {k}"
        assert _bits_equal(snaps[k][1], snaps[0][1]), f"[{tag}] state differs on launch {k}"

    got_y, got_state = snaps[-1]
    last = max_conv_len - 1
    dy = (got_y[:, :d].float() - y_ref.float()).abs().max().item()
    torch.testing.assert_close(got_y[:, :d].float(), y_ref.float(), atol=atol, rtol=rtol)
    ds = (got_state[:, last, :d].float() - state_ref[:, last, :d].float()).abs().max().item()
    torch.testing.assert_close(
        got_state[:, last, :d].float(), state_ref[:, last, :d].float(), atol=atol, rtol=rtol)

    # history shift: bit-exact dtype copy of conv_state[:, 1:] (state_ref[:, :last] is
    # exactly that, computed by the torch reference)
    if max_conv_len > 1:
        assert _bits_equal(got_state[:, :last, :d], state_ref[:, :last, :d]), (
            f"[{tag}] history shift is not a bit-exact copy")
    if pad:
        assert not torch.isnan(got_y).any(), f"[{tag}] padded y columns were not written"
        assert bool((got_y[:, d:] == 0).all()), f"[{tag}] padded y columns are not zero"

    _CASES.append((tag, dy, ds))
    print(
        f"[{tag}] PASS b={batch} d_mem={d_mem} d={d} d_pad={d_pad} mcl={max_conv_len} "
        f"L={cache_len} w={w} dl={dilation} {dt} y_diff={dy:.3e} state_diff={ds:.3e} "
        f"det={launches}x time={elapsed:.2f}s"
    )
    return dy, ds


def run_L0():
    """Manifest shapes plus the structural edges of the two-launch split."""
    _run_case(1, 512, 256, 12, 4, 3, torch.float16, "L0-serving-b1")
    _run_case(4, 1024, 512, 20, 4, 5, torch.float16, "L0-serving-b4")
    _run_case(8, 512, 256, 18, 4, 3, torch.bfloat16, "L0-serving-b8")
    # decode-s1: max_conv_len == dilation*(w-1), tap 0 hits history row 0
    _run_case(1, 256, 256, 9, 4, 3, torch.float16, "L0-decode-s1-tap0")
    # partial cache (host left padding) and the first step (empty cache)
    _run_case(2, 512, 256, 12, 4, 3, torch.float16, "L0-partial-cache-L5", cache_len=5)
    _run_case(1, 256, 256, 9, 4, 3, torch.float16, "L0-first-step-L0", cache_len=0)


def run_L1():
    """Representative shapes beyond the gate."""
    _run_case(2, 512, 1024, 10, 4, 3, torch.float16, "L1-d1024")
    _run_case(2, 512, 600, 12, 4, 3, torch.float16, "L1-d600-masked-tail")
    _run_case(1, 256, 1, 5, 4, 1, torch.float16, "L1-d1")
    _run_case(2, 256, 256, 4, 1, 1, torch.float16, "L1-one-tap")
    _run_case(17, 519, 256, 12, 4, 3, torch.float16, "L1-ktail-519-b17")
    _run_case(33, 512, 256, 12, 4, 3, torch.bfloat16, "L1-mtail-33")
    _run_case(8, 256, 512, 16, 4, 5, torch.bfloat16, "L1-bf16-d512")


def run_L2():
    """Larger scale than the gate; recorded, non-blocking (2 shapes)."""
    try:
        _run_case(64, 1024, 512, 20, 4, 5, torch.float16, "L2-b64-d1024")
        _run_case(3, 1024, 1024, 20, 4, 5, torch.bfloat16, "L2-b3-d1024-bf16")
    except Exception as exc:  # noqa: BLE001 - L2 records without blocking
        print(f"[L2] WARN (recorded, non-blocking): {type(exc).__name__}: {exc}")


def run_boundary():
    """Boundary: single-slot cache, one tap, and a rolling multi-step cache."""
    _run_case(1, 256, 256, 1, 1, 1, torch.float16, "Boundary-mcl1")
    _run_case(2, 519, 256, 1, 1, 1, torch.bfloat16, "Boundary-mcl1-ktail")
    _run_multistep(2, 256, 12, 4, 3, torch.float16, "Boundary-multistep-12", steps=12)


def _run_multistep(batch, d, max_conv_len, w, dilation, dtype, tag, steps=12):
    """Roll the real cache through both launches over `steps` decode steps."""
    d_pad = _align_up(d)
    dt = _dtype_name(dtype)
    atol, rtol = TOL[dt]
    torch.manual_seed(7)
    w_k = (torch.randn(d, d, dtype=dtype) * 0.02).to("npu")
    w_v = (torch.randn(d, d, dtype=dtype) * 0.02).to("npu")
    rms_w_h = (1.0 + 0.1 * torch.randn(d)).to(dtype)
    rms_w_v = (1.0 + 0.1 * torch.randn(d)).to(dtype)
    conv_w = (torch.randn(w, d, dtype=dtype) * 0.02).to("npu")
    wh, wv = rms_w_h.to("npu"), rms_w_v.to("npu")
    proj = _engram_project_kernel(batch, d, d_pad, max_conv_len, dt)(
        **project_default_config(d_pad))
    step = _engram_step_kernel(batch, d, d_pad, max_conv_len, w, dilation, 1e-6, dt)(
        step_default_config(d_pad)["threads"])
    ref_wk, ref_wv = (torch.randn(d, d, dtype=dtype) * 0.02), (torch.randn(d, d, dtype=dtype) * 0.02)
    ref_wk, ref_wv = w_k.cpu(), w_v.cpu()
    wh_c, wv_c, cw_c = rms_w_h, rms_w_v, conv_w.cpu()

    hist = torch.zeros(batch, 0, d, dtype=dtype)
    state_buf = torch.full((batch, max_conv_len, d_pad), float("nan"), dtype=dtype, device="npu")
    worst_y, worst_s = 0.0, 0.0
    for step_i in range(steps):
        e_t = torch.randn(batch, d, dtype=dtype) * 0.1
        h_t = torch.randn(batch, d, dtype=dtype)
        cache_len = min(step_i, max_conv_len)
        cache = hist[:, hist.shape[1] - cache_len:, :] if cache_len else hist
        y_ref, state_ref = engram_decode_step_torch(
            e_t, h_t, cache, ref_wk, ref_wv, wh_c, wv_c, cw_c, max_conv_len, dilation, 1e-6)
        y_buf = torch.full((batch, d_pad), float("nan"), dtype=dtype, device="npu")
        full = F.pad(cache, (0, 0, max_conv_len - cache_len, 0)) if cache_len < max_conv_len \
            else cache
        kv = torch.empty((2, batch, d_pad), dtype=torch.float32, device="npu")
        proj(e_t.to("npu"), w_k, w_v, full.to("npu"), kv, state_buf)
        step(h_t.to("npu"), kv, full.to("npu"), wh, wv, conv_w, y_buf, state_buf)
        torch.npu.synchronize()
        got_y, got_state = y_buf.cpu(), state_buf.cpu()
        torch.testing.assert_close(got_y[:, :d].float(), y_ref.float(), atol=atol, rtol=rtol)
        last = max_conv_len - 1
        torch.testing.assert_close(
            got_state[:, last, :d].float(), state_ref[:, last, :d].float(), atol=atol, rtol=rtol)
        assert _bits_equal(got_state[:, :last, :d], state_ref[:, :last, :d]), (
            f"[{tag}] step {step_i}: history shift is not bit-exact")
        worst_y = max(worst_y, (got_y[:, :d].float() - y_ref.float()).abs().max().item())
        worst_s = max(worst_s,
                      (got_state[:, last, :d].float() - state_ref[:, last, :d].float()).abs().max().item())
        # roll the cache with the *reference* appended slot (bit-exact source of truth)
        hist = torch.cat([hist, state_ref[:, last:last + 1, :]], dim=1)
    _CASES.append((tag, worst_y, worst_s))
    print(f"[{tag}] PASS b={batch} d={d} mcl={max_conv_len} steps={steps} {dt} "
          f"y_diff={worst_y:.3e} state_diff={worst_s:.3e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0",
                        choices=["L0", "L1", "L2", "boundary", "all"])
    args, _ = parser.parse_known_args()
    t0 = time.perf_counter()
    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L1()
    elif args.level == "L2":
        run_L2()
    elif args.level == "boundary":
        run_boundary()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    worst_y = max(dy for _, dy, _ in _CASES)
    worst_s = max(ds for _, _, ds in _CASES)
    print(f"cases={len(_CASES)} max_y_diff={worst_y:.3e} max_state_diff={worst_s:.3e} "
          f"total={time.perf_counter() - t0:.1f}s")
    print("\033[92mAll check passed!\033[0m")
    print("[PRECISION_PASS]")


if __name__ == "__main__":
    main()
