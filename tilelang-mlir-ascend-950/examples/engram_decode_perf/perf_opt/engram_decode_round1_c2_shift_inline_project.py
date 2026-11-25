# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 4 r1_c2 (perf_opt): cache shift folded into the first batch blocks.

Launch 1 of 2 of ``EngramDecodeFwdOp``.  One decode step projects the memory
embedding twice (``kv[0] = e_t @ W_K``, ``kv[1] = e_t @ W_V``, fp32 output) and
shifts the convolution cache one slot to the left
(``new_conv_state[:, :max_conv_len-1, :] = conv_state[:, 1:, :]``, a pure dtype
copy).  The last cache slot belongs to ``_engram_step_kernel`` (launch 2) and is
never touched here.

Implementation follows DESIGN.md (frozen, REVIEW.md passed):

* R1 -- platform ``T.Kernel`` takes a 1-D grid only, so the three output axes
  (batch tile x K-or-V x d tile) are packed into one integer block index.
* R2/R3/R5 -- the last tile of each axis carries the real remainder
  (``am``/``an``/``ak``) into the copy regions and the ``asc_mmad`` parameters,
  so unaligned ``d_mem`` (519) and a batch below the Cube M granularity need no
  host-side padding and no elementwise predicate.
* R4 -- the cache shift rides in the same launch as ``GM -> UB -> GM`` copies
  issued from the AIV region; a ``T.Kernel`` that holds both L1/L0 GEMM code and
  UB code is auto-promoted to ``__mix__(1, 2)`` by the compiler.

Cross-launch contract: ``kv`` is fp32 ``(2, batch, d_padded)``; the shift rows are
a bit-exact copy; ``d_padded = align_up(d, 256)`` on the host side.

run: python _engram_project_kernel.py --level L0|all
"""

import argparse
import functools
import time

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401 -- registers torch.npu
import tilelang
import tilelang.ascend.language as T

# The host rounds the hidden width up to this many elements before the launch;
# 256 keeps the trailing axis a multiple of the 16-element vector granularity.
ALIGNMENT = 256

# Stage 4 O-1 knobs: how many blocks run the cache shift (one batch row each) and
# the UB budget that keeps the full-width row form legal.
SHIFT_BLOCKS_MAX = 16
SHIFT_UB_BUDGET = 98304

# Frozen tiling (DESIGN.md 5.6).  ``block_k`` is the L0B reverse solve
# ``min(65536 // (block_n * itemsize * num_stages), 256)``; at
# block_n=64/fp16/2 stages it is exactly the 64 KiB L0B capacity.
BLOCK_M = 16
BLOCK_N = 64
BLOCK_K = 256
NUM_STAGES = 2

# DESIGN.md 8.2: fp32 accumulator, only the K-block grouping order can differ.
KV_ATOL = 1e-3
KV_RTOL = 1e-3

_DIFFS = []


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference, from the operator semantics only)
# ---------------------------------------------------------------------------
def golden_engram_project(e_t, W_K, W_V, conv_state, max_conv_len):
    """Bit-level reference of launch 1, ported from the source torch reference.

    Returns ``(kv, new_conv_state)``:

    * ``kv[0] = e_t.float() @ W_K.float()``, ``kv[1] = e_t.float() @ W_V.float()``
      -- fp32, no rounding back to the input dtype (the step kernel rounds);
    * ``new_conv_state[:, :max_conv_len-1, :] = state[:, 1:, :]`` where ``state``
      is ``conv_state`` left-padded to ``max_conv_len`` rows (the host contract of
      the source ``forward``), and the last row is a NaN sentinel because this
      launch must not write it.

    Deliberately free of kernel-side devices: no region truncation, no
    ``clear_accum``, no UB staging.
    """
    batch, d_mem = e_t.shape
    d_pad = W_K.shape[1]
    mcl = int(max_conv_len)
    cache_len = conv_state.shape[1]

    k = e_t.float() @ W_K.float()
    v = e_t.float() @ W_V.float()
    kv = torch.stack([k, v], dim=0)

    if mcl > cache_len:
        state = F.pad(conv_state, (0, 0, mcl - cache_len, 0))
    else:
        state = conv_state
    new_state = torch.empty(batch, mcl, d_pad, dtype=state.dtype, device=state.device)
    new_state[:, : mcl - 1, :] = state[:, 1:, :]
    new_state[:, mcl - 1, :] = float("nan")
    return kv, new_state


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _build_project(batch, d_mem, d_pad, max_conv_len, dtype_str,
                   block_m=BLOCK_M, block_n=BLOCK_N, block_k=BLOCK_K,
                   num_stages=NUM_STAGES):
    """Both projections of one decode step, tiled over ``d`` and the batch.

    One block owns one ``(batch tile, K-or-V, d tile)`` output tile, so each
    weight tile streams through L1 once and the Cube cores run the whole
    ``e_t @ W`` product.  The tile extents stay symbolic so the last tile of each
    axis carries the real remainder.
    """
    m_tiles = -(-batch // block_m)
    n_tiles = -(-d_pad // block_n)
    k_tiles = -(-d_mem // block_k)
    nshift = max_conv_len - 1
    grid = m_tiles * 2 * n_tiles
    # Stage 4 O-1b: the cache shift is spread over the first `nb_shift` blocks, one
    # batch row per block, as a single full-width copy pair (`conv_state[b, 1:]` is
    # contiguous).  The baseline ran it as `for b in T.serial(batch)` x
    # (copy + sync + copy) inside the `which == 0` blocks: serving-b8 paid 3.67 us
    # for 8 serialised pairs in 4 blocks.  Falls back to the per-n-tile slice form
    # when a full row does not fit the UB budget.
    nb_shift = min(batch, grid, SHIFT_BLOCKS_MAX) if nshift > 0 else 0
    shift_reps = -(-batch // nb_shift) if nb_shift > 0 else 0
    shift_wide = nshift * d_pad * 2 <= SHIFT_UB_BUDGET
    shift_width = d_pad if shift_wide else block_n

    @T.macro
    def _projection(E, W, kv, out_idx, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an):
        """One ``(batch tile, d tile)`` of ``kv[out_idx] = E @ W``, K-blocked."""
        for kt in T.Pipelined(k_tiles, num_stages=num_stages):
            k0 = kt * block_k
            ak = T.min(d_mem - k0, block_k)
            T.copy(E[m0 : m0 + am, k0 : k0 + ak], e_l1[0:am, 0:ak])
            T.copy(
                W[k0 : k0 + ak, n0 : n0 + an],
                w_l1[0:an, 0:ak],
                transpose=True,
            )
            T.copy(e_l1[0:am, 0:ak], e_l0[0:am, 0:ak])
            T.copy(w_l1[0:an, 0:ak], w_l0[0:an, 0:ak])
            T.gemm(
                e_l0[0:am, 0:ak],
                w_l0[0:an, 0:ak],
                res[0:am, 0:an],
                transpose_B=True,
                clear_accum=kt == 0,
            )
        T.copy(res[0:am, 0:an], kv[out_idx, m0 : m0 + am, n0 : n0 + an])

    @T.macro
    def _shift_cache(conv_state, new_conv_state, shift_ub, n0, an):
        """Left-shift one d column slice of the cache through UB, one batch row
        per 2-D descriptor (a >2-D strided copy is not supported)."""
        for b in T.serial(batch):
            T.copy(conv_state[b, 1:max_conv_len, n0 : n0 + an], shift_ub[0:nshift, 0:an])
            T.copy(shift_ub[0:nshift, 0:an], new_conv_state[b, 0:nshift, n0 : n0 + an])

    @T.prim_func
    def engram_project(
        e_t: T.Tensor((batch, d_mem), dtype_str),
        W_K: T.Tensor((d_mem, d_pad), dtype_str),
        W_V: T.Tensor((d_mem, d_pad), dtype_str),
        conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype_str),
        kv: T.Tensor((2, batch, d_pad), "float32"),
        new_conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype_str),
    ):
        with T.Kernel(grid) as bx:
            e_l1 = T.alloc_l1((block_m, block_k), dtype_str)
            w_l1 = T.alloc_l1((block_n, block_k), dtype_str)
            e_l0 = T.alloc_l0a((block_m, block_k), dtype_str)
            w_l0 = T.alloc_l0b((block_n, block_k), dtype_str)
            res = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions(
                {e_l1: num_stages, w_l1: num_stages, e_l0: num_stages, w_l0: num_stages}
            )
            m0 = (bx // (2 * n_tiles)) * block_m
            which = (bx // n_tiles) % 2
            n0 = (bx % n_tiles) * block_n
            am = T.min(batch - m0, block_m)
            an = T.min(d_pad - n0, block_n)
            if which == 0:
                _projection(e_t, W_K, kv, 0, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
            else:
                _projection(e_t, W_V, kv, 1, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
            if nshift > 0:
                if bx < nb_shift:
                    shift_ub = T.alloc_shared((nshift, shift_width), dtype_str)
                    for r in T.serial(shift_reps):
                        b = bx + r * nb_shift
                        if b < batch:
                            if shift_wide:
                                T.copy(conv_state[b, 1:max_conv_len, :], shift_ub[0:nshift, :])
                                T.copy(shift_ub[0:nshift, :], new_conv_state[b, 0:nshift, :])
                            else:
                                for nt in T.serial(n_tiles):
                                    cn0 = nt * block_n
                                    can = T.min(d_pad - cn0, block_n)
                                    T.copy(
                                        conv_state[b, 1:max_conv_len, cn0 : cn0 + can],
                                        shift_ub[0:nshift, 0:can],
                                    )
                                    T.copy(
                                        shift_ub[0:nshift, 0:can],
                                        new_conv_state[b, 0:nshift, cn0 : cn0 + can],
                                    )

    return tilelang.compile(engram_project, target="ascend", out_idx=[])


def _engram_project_kernel(batch, d_mem, d_padded, max_conv_len, dtype):
    """Harness-shaped entry (no ``threads`` parameter on this backend)."""

    def _func(block_m, block_n, block_k, num_stages):
        return _build_project(
            batch, d_mem, d_padded, max_conv_len, dtype,
            block_m, block_n, block_k, num_stages,
        )

    return _func


def default_config(d_pad):
    """Frozen Stage 3 tiling (DESIGN.md 5.6)."""
    return {
        "block_m": BLOCK_M,
        "block_n": min(BLOCK_N, d_pad),
        "block_k": BLOCK_K,
        "num_stages": NUM_STAGES,
    }


# ---------------------------------------------------------------------------
# Precision comparing
# ---------------------------------------------------------------------------
def _align_up(value, multiple=ALIGNMENT):
    return (value + multiple - 1) // multiple * multiple


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _run_case(batch, d_mem, d, max_conv_len, dtype, tag, cache_len=None, d_pad=None):
    """Compile, launch and check one shape; returns the per-case max diff."""
    d_pad = _align_up(d) if d_pad is None else d_pad
    cache_len = max_conv_len if cache_len is None else cache_len
    cfg = default_config(d_pad)

    torch.manual_seed(0)
    e_t = torch.randn(batch, d_mem, dtype=dtype)
    w_k = torch.randn(d_mem, d_pad, dtype=dtype)
    w_v = torch.randn(d_mem, d_pad, dtype=dtype)
    conv_state = torch.randn(batch, cache_len, d_pad, dtype=dtype)

    # Golden: unpadded history in, host left-pad happens inside the reference.
    gold_kv, gold_state = golden_engram_project(e_t, w_k, w_v, conv_state, max_conv_len)

    # The launch always sees a full-length cache (source ``forward`` pads first).
    if cache_len < max_conv_len:
        padded = F.pad(conv_state, (0, 0, max_conv_len - cache_len, 0))
    else:
        padded = conv_state

    kv = torch.zeros(2, batch, d_pad, dtype=torch.float32, device="npu")
    new_conv_state = torch.full(
        (batch, max_conv_len, d_pad), float("nan"), dtype=dtype, device="npu"
    )

    t0 = time.perf_counter()
    kernel = _build_project(
        batch, d_mem, d_pad, max_conv_len, _dtype_name(dtype),
        cfg["block_m"], cfg["block_n"], cfg["block_k"], cfg["num_stages"],
    )
    kernel(
        e_t.to("npu"), w_k.to("npu"), w_v.to("npu"), padded.to("npu"),
        kv, new_conv_state,
    )
    torch.npu.synchronize()
    elapsed = time.perf_counter() - t0

    got_kv = kv.cpu()
    got_state = new_conv_state.cpu()

    # 1) projections: fp32 values, atol/rtol from DESIGN.md 8.2.
    diff = (got_kv - gold_kv).abs().max().item()
    torch.testing.assert_close(got_kv, gold_kv, atol=KV_ATOL, rtol=KV_RTOL)

    # 2) shifted rows: pure dtype copy, bit-exact (covers -0.0 / inf / NaN).
    if max_conv_len > 1:
        want = padded[:, 1:, :]
        got = got_state[:, : max_conv_len - 1, :]
        assert got.shape == want.shape, (tag, got.shape, want.shape)
        bit_ok = torch.equal(
            got.contiguous().view(torch.int16), want.contiguous().view(torch.int16)
        )
        assert bit_ok, f"[{tag}] shift is not bit-exact"

    # 3) last slot: the sentinel must survive, proving no out-of-range write.
    assert torch.isnan(got_state[:, max_conv_len - 1, :]).all(), (
        f"[{tag}] last cache slot was written by the project launch"
    )

    _DIFFS.append((tag, diff))
    print(
        f"[{tag}] PASS batch={batch} d_mem={d_mem} d_pad={d_pad} "
        f"mcl={max_conv_len} cache_len={cache_len} dtype={dtype} "
        f"grid={(-(-batch // cfg['block_m'])) * 2 * (-(-d_pad // cfg['block_n']))} "
        f"max_diff={diff:.3e} time={elapsed:.2f}s"
    )
    return diff


# ---------------------------------------------------------------------------
# Hierarchical testing
# ---------------------------------------------------------------------------
def run_L0():
    """Gate level: manifest shapes, K tail, both cache-length extremes, 3 widths."""
    # serving-b1 (K blocks = 2) and serving-b8 (bf16, am = 8)
    _run_case(1, 512, 256, 12, torch.float16, "L0-serving-b1")
    _run_case(8, 512, 256, 18, torch.bfloat16, "L0-serving-b8")
    # serving-b4: 4 K blocks, nt = 8, largest cache
    _run_case(4, 1024, 512, 20, torch.float16, "L0-serving-b4")
    # K tail: d_mem = 519 -> ak = 7 in the last K block; M tail: am = 1
    _run_case(17, 519, 256, 12, torch.float16, "L0-ktail-519")
    # first step: empty cache, host pads it to max_conv_len
    _run_case(1, 256, 256, 9, torch.float16, "L0-first-step-L0", cache_len=0)
    # partial cache: L < max_conv_len
    _run_case(2, 512, 256, 12, torch.float16, "L0-partial-cache", cache_len=5)
    # d = 512 (d_padded = 512, nt = 8)
    _run_case(2, 256, 512, 12, torch.float16, "L0-d512")
    # d = 1024 (d_padded = 1024, nt = 16)
    _run_case(2, 512, 1024, 10, torch.float16, "L0-d1024")


def run_L1():
    """Representative shapes beyond the gate: deeper batch tail, N tail, bf16 K tail."""
    # M tail with three batch tiles (mt = 3, am = 1 in the last one)
    _run_case(33, 512, 256, 12, torch.float16, "L1-mtail-33")
    # N tail: d_padded = 272 -> an in {16, 32, 48, 64}
    _run_case(1, 512, 256, 12, torch.float16, "L1-ntail-272", d_pad=272)
    _run_case(17, 519, 256, 12, torch.bfloat16, "L1-ktail-519-bf16")
    # single K block, d_mem below block_k
    _run_case(4, 128, 512, 20, torch.float16, "L1-dmem-below-blockk")
    # cache of length 1 (no shift at source level either)
    _run_case(2, 512, 256, 6, torch.bfloat16, "L1-bf16-b2", cache_len=1)


def run_L2():
    """Larger scale than the gate; recorded, non-blocking."""
    try:
        _run_case(64, 1024, 512, 20, torch.float16, "L2-b64-d1024")
        _run_case(3, 1024, 1024, 20, torch.bfloat16, "L2-b3-d1024-bf16")
    except Exception as exc:  # noqa: BLE001 - L2 records without blocking
        print(f"[L2] WARN (recorded, non-blocking): {type(exc).__name__}: {exc}")


def run_boundary():
    """Boundary: no shift, unaligned K, and non-finite cache payloads."""
    # max_conv_len == 1: the shift disappears, every cache slot stays a sentinel.
    _run_case(1, 256, 256, 1, torch.float16, "Boundary-mcl1")
    _run_case(4, 519, 256, 1, torch.bfloat16, "Boundary-mcl1-ktail")
    # K narrower than one Cube step and not a multiple of 16
    _run_case(2, 17, 256, 9, torch.float16, "Boundary-dmem17")
    _run_case(2, 1, 256, 9, torch.float16, "Boundary-dmem1")

    # Non-finite / signed-zero cache payloads must survive the shift bit-exactly.
    batch, d_mem, d_pad, mcl = 2, 256, 256, 7
    torch.manual_seed(0)
    e_t = torch.randn(batch, d_mem, dtype=torch.float16)
    w_k = torch.randn(d_mem, d_pad, dtype=torch.float16)
    w_v = torch.randn(d_mem, d_pad, dtype=torch.float16)
    conv_state = torch.randn(batch, mcl, d_pad, dtype=torch.float16)
    conv_state[0, 0, 0] = float("nan")
    conv_state[0, 1, 1] = float("inf")
    conv_state[1, 2, 2] = float("-inf")
    conv_state[1, 3, 3] = -0.0

    gold_kv, _ = golden_engram_project(e_t, w_k, w_v, conv_state, mcl)
    kv = torch.zeros(2, batch, d_pad, dtype=torch.float32, device="npu")
    new_conv_state = torch.full(
        (batch, mcl, d_pad), float("nan"), dtype=torch.float16, device="npu"
    )
    cfg = default_config(d_pad)
    kernel = _build_project(
        batch, d_mem, d_pad, mcl, "float16",
        cfg["block_m"], cfg["block_n"], cfg["block_k"], cfg["num_stages"],
    )
    kernel(
        e_t.to("npu"), w_k.to("npu"), w_v.to("npu"),
        conv_state.to("npu"), kv, new_conv_state,
    )
    torch.npu.synchronize()
    diff = (kv.cpu() - gold_kv).abs().max().item()
    torch.testing.assert_close(kv.cpu(), gold_kv, atol=KV_ATOL, rtol=KV_RTOL)
    got = new_conv_state.cpu()[:, : mcl - 1, :]
    want = conv_state[:, 1:, :]
    assert torch.equal(got.contiguous().view(torch.int16), want.contiguous().view(torch.int16))
    assert torch.isnan(new_conv_state.cpu()[:, mcl - 1, :]).all()
    _DIFFS.append(("Boundary-nonfinite-shift", diff))
    print(f"[Boundary-nonfinite-shift] PASS max_diff={diff:.3e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
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

    worst = max(diff for _, diff in _DIFFS) if _DIFFS else 0.0
    print(f"cases={len(_DIFFS)} max_diff={worst:.3e} total={time.perf_counter() - t0:.1f}s")
    print("\033[92mAll check passed!\033[0m")
    print("[PRECISION_PASS]")


if __name__ == "__main__":
    main()
