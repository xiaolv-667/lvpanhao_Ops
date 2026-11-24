"""Design-time compile probe (D-3) for `_engram_project_kernel` on Ascend950PR_9579.

Compile-only: this script creates no NPU tensor and executes no kernel, so it does
not occupy a device.  Run from the task directory:

    source .agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        history_version/design_probe_compile.py

Each variant answers one open design question (see DESIGN.md §11.1):

  P1  cube-only projection (L1 GEMM NT, 1-D grid)            -> launch kind / GEMM template
  P2  P1 + cache-shift through UB in the *same* T.Kernel     -> is UB legal in a cube launch?
  P3  T.MixedKernel: T.Cube() GEMM + T.Vector() shift        -> mixed launch shape
  P4  K tail (d_mem=519, block_k=256, pad_value=0)           -> AscendInsertOOBPadding
  P5  3-D region shift copy (whole batch stack, one copy)    -> region rank accepted
  P6  block_n / block_k variants (32/64/128, 128/256)        -> tile menu compiles
"""

import sys
import traceback

import tilelang
import tilelang.ascend.language as T


def _ceil(a, b):
    return -(-a // b)


def _marks(compiled):
    src = ""
    for attr in ("get_kernel_source", "kernel_source"):
        try:
            src = getattr(compiled, attr)
            src = src() if callable(src) else src
            if src:
                break
        except Exception:  # noqa: BLE001 - debug aid only
            continue
    marks = [t for t in ("__cube__", "__mix__", "__aicore__", "ascend_gemm_l1",
                         "asc_copy_gm2ub", "asc_copy_ub2gm", "asc_fill_l1",
                         "asc_set_copy_pad_val") if t in src]
    return marks


# ---------------------------------------------------------------------------
# P1: cube-only projection (both weights on the grid's second axis)
# ---------------------------------------------------------------------------
def build_p1(batch=8, d_mem=512, d_pad=256, dtype="float16", block_m=16, block_n=64,
             block_k=256, num_stages=2):
    n_tiles = _ceil(d_pad, block_n)
    k_tiles = _ceil(d_mem, block_k)
    k_exact = d_mem % block_k == 0

    @T.macro
    def _one(E, W, kv, e_l1, w_l1, res, out_idx, n0, am, an):
        for kt in T.Pipelined(k_tiles, num_stages=num_stages):
            k0 = kt * block_k
            T.copy(E[0:am, k0 : k0 + block_k], e_l1[0:am, 0:block_k])
            T.copy(
                W[k0 : k0 + block_k, n0 : n0 + an],
                w_l1[0:an, 0:block_k],
                transpose=True,
            )
            T.gemm(e_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
        T.copy(res[0:am, 0:an], kv[out_idx, 0:am, n0 : n0 + an])

    @T.prim_func
    def main(
        E: T.Tensor((batch, d_mem), dtype),
        W_K: T.Tensor((d_mem, d_pad), dtype),
        W_V: T.Tensor((d_mem, d_pad), dtype),
        kv: T.Tensor((2, batch, d_pad), "float32"),
    ):
        with T.Kernel(2 * n_tiles) as bx:
            e_l1 = T.alloc_l1((block_m, block_k), dtype)
            w_l1 = T.alloc_l1((block_n, block_k), dtype)
            res = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions({e_l1: num_stages, w_l1: num_stages})
            which = bx // n_tiles
            n0 = (bx % n_tiles) * block_n
            am = T.min(batch, block_m)
            an = T.min(d_pad - n0, block_n)
            if which == 0:
                _one(E, W_K, kv, e_l1, w_l1, res, 0, n0, am, an)
            else:
                _one(E, W_V, kv, e_l1, w_l1, res, 1, n0, am, an)

    return tilelang.compile(main, target="ascend", out_idx=[])


# ---------------------------------------------------------------------------
# P2: cube GEMM + UB-staged cache shift in the same plain T.Kernel
# ---------------------------------------------------------------------------
def build_p2(batch=8, d_mem=512, d_pad=256, max_conv_len=18, dtype="float16",
             block_m=16, block_n=64, block_k=256, num_stages=2, shift_grid=True):
    n_tiles = _ceil(d_pad, block_n)
    k_tiles = _ceil(d_mem, block_k)
    nshift = max_conv_len - 1
    grid = (3 if shift_grid else 2) * n_tiles

    @T.macro
    def _one(E, W, kv, e_l1, w_l1, res, out_idx, n0, am, an):
        for kt in T.Pipelined(k_tiles, num_stages=num_stages):
            k0 = kt * block_k
            T.copy(E[0:am, k0 : k0 + block_k], e_l1[0:am, 0:block_k])
            T.copy(
                W[k0 : k0 + block_k, n0 : n0 + an],
                w_l1[0:an, 0:block_k],
                transpose=True,
            )
            T.gemm(e_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
        T.copy(res[0:am, 0:an], kv[out_idx, 0:am, n0 : n0 + an])

    @T.macro
    def _shift(conv_state, new_conv_state, shift_ub, n0, an):
        for b in T.serial(batch):
            T.copy(conv_state[b, 1:max_conv_len, n0 : n0 + an], shift_ub[0:nshift, 0:an])
            T.copy(shift_ub[0:nshift, 0:an], new_conv_state[b, 0:nshift, n0 : n0 + an])

    @T.prim_func
    def main(
        E: T.Tensor((batch, d_mem), dtype),
        W_K: T.Tensor((d_mem, d_pad), dtype),
        W_V: T.Tensor((d_mem, d_pad), dtype),
        conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype),
        kv: T.Tensor((2, batch, d_pad), "float32"),
        new_conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype),
    ):
        with T.Kernel(grid) as bx:
            e_l1 = T.alloc_l1((block_m, block_k), dtype)
            w_l1 = T.alloc_l1((block_n, block_k), dtype)
            res = T.alloc_l0c((block_m, block_n), "float32")
            shift_ub = T.alloc_shared((max(1, nshift), block_n), dtype)
            T.annotate_buffer_versions({e_l1: num_stages, w_l1: num_stages})
            which = bx // n_tiles
            n0 = (bx % n_tiles) * block_n
            am = T.min(batch, block_m)
            an = T.min(d_pad - n0, block_n)
            if which == 0:
                _one(E, W_K, kv, e_l1, w_l1, res, 0, n0, am, an)
            elif which == 1:
                _one(E, W_V, kv, e_l1, w_l1, res, 1, n0, am, an)
            else:
                _shift(conv_state, new_conv_state, shift_ub, n0, an)

    return tilelang.compile(main, target="ascend", out_idx=[])


# ---------------------------------------------------------------------------
# P3: mixed kernel, Cube region = GEMM, Vector region = shift
# ---------------------------------------------------------------------------
def build_p3(batch=8, d_mem=512, d_pad=256, max_conv_len=18, dtype="float16",
             block_m=16, block_n=64, block_k=256, num_stages=2):
    n_tiles = _ceil(d_pad, block_n)
    k_tiles = _ceil(d_mem, block_k)
    nshift = max_conv_len - 1
    half = nshift // 2

    @T.macro
    def _one(E, W, kv, e_l1, w_l1, res, out_idx, n0, am, an):
        for kt in T.Pipelined(k_tiles, num_stages=num_stages):
            k0 = kt * block_k
            T.copy(E[0:am, k0 : k0 + block_k], e_l1[0:am, 0:block_k])
            T.copy(
                W[k0 : k0 + block_k, n0 : n0 + an],
                w_l1[0:an, 0:block_k],
                transpose=True,
            )
            T.gemm(e_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
        T.copy(res[0:am, 0:an], kv[out_idx, 0:am, n0 : n0 + an])

    @T.prim_func
    def main(
        E: T.Tensor((batch, d_mem), dtype),
        W_K: T.Tensor((d_mem, d_pad), dtype),
        W_V: T.Tensor((d_mem, d_pad), dtype),
        conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype),
        kv: T.Tensor((2, batch, d_pad), "float32"),
        new_conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype),
    ):
        with T.MixedKernel(2 * n_tiles, sids=2) as (bx, sid):
            e_l1 = T.alloc_l1((block_m, block_k), dtype)
            w_l1 = T.alloc_l1((block_n, block_k), dtype)
            res = T.alloc_l0c((block_m, block_n), "float32")
            shift_ub = T.alloc_shared((max(1, half), block_n), dtype)
            T.annotate_buffer_versions({e_l1: num_stages, w_l1: num_stages})
            which = bx // n_tiles
            n0 = (bx % n_tiles) * block_n
            am = T.min(batch, block_m)
            an = T.min(d_pad - n0, block_n)
            with T.Cube():
                if which == 0:
                    _one(E, W_K, kv, e_l1, w_l1, res, 0, n0, am, an)
                else:
                    _one(E, W_V, kv, e_l1, w_l1, res, 1, n0, am, an)
            with T.Vector(vector=2):
                for b in T.serial(batch):
                    if sid == 0:
                        T.copy(
                            conv_state[b, 1 : 1 + half, n0 : n0 + an],
                            shift_ub[0:half, 0:an],
                        )
                        T.copy(
                            shift_ub[0:half, 0:an],
                            new_conv_state[b, 0:half, n0 : n0 + an],
                        )
                    else:
                        T.copy(
                            conv_state[b, 1 + half : max_conv_len, n0 : n0 + an],
                            shift_ub[0 : nshift - half, 0:an],
                        )
                        T.copy(
                            shift_ub[0 : nshift - half, 0:an],
                            new_conv_state[b, half:nshift, n0 : n0 + an],
                        )

    return tilelang.compile(main, target="ascend", out_idx=[])


# ---------------------------------------------------------------------------
# P4: K tail (d_mem not a multiple of block_k) with/without OOB padding
# ---------------------------------------------------------------------------
def build_p4(batch=17, d_mem=519, d_pad=256, dtype="float16", block_m=16, block_n=64,
             block_k=256, num_stages=2, pad=True):
    n_tiles = _ceil(d_pad, block_n)
    k_tiles = _ceil(d_mem, block_k)

    @T.macro
    def _one(E, W, kv, e_l1, w_l1, res, e_m0, out_idx, m0, n0, am, an):
        for kt in T.Pipelined(k_tiles, num_stages=num_stages):
            k0 = kt * block_k
            if pad:
                T.copy(
                    E[m0 : m0 + am, k0 : k0 + block_k],
                    e_l1[0:am, 0:block_k],
                    pad_value=0,
                )
                T.copy(
                    W[k0 : k0 + block_k, n0 : n0 + an],
                    w_l1[0:an, 0:block_k],
                    transpose=True,
                    pad_value=0,
                )
            else:
                T.copy(E[m0 : m0 + am, k0 : k0 + block_k], e_l1[0:am, 0:block_k])
                T.copy(
                    W[k0 : k0 + block_k, n0 : n0 + an],
                    w_l1[0:an, 0:block_k],
                    transpose=True,
                )
            T.gemm(e_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
        T.copy(res[0:am, 0:an], kv[out_idx, m0 : m0 + am, n0 : n0 + an])

    @T.prim_func
    def main(
        E: T.Tensor((batch, d_mem), dtype),
        W_K: T.Tensor((d_mem, d_pad), dtype),
        W_V: T.Tensor((d_mem, d_pad), dtype),
        kv: T.Tensor((2, batch, d_pad), "float32"),
    ):
        with T.Kernel(2 * n_tiles) as bx:
            e_l1 = T.alloc_l1((block_m, block_k), dtype)
            w_l1 = T.alloc_l1((block_n, block_k), dtype)
            res = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions({e_l1: num_stages, w_l1: num_stages})
            which = bx // n_tiles
            n0 = (bx % n_tiles) * block_n
            am = T.min(batch, block_m)
            an = T.min(d_pad - n0, block_n)
            if which == 0:
                _one(E, W_K, kv, e_l1, w_l1, res, E, 0, 0, n0, am, an)
            else:
                _one(E, W_V, kv, e_l1, w_l1, res, E, 1, 0, n0, am, an)

    return tilelang.compile(main, target="ascend", out_idx=[])


# ---------------------------------------------------------------------------
# P5: 3-D region shift copy (whole batch stack in one copy)
# ---------------------------------------------------------------------------
def build_p5(batch=8, d_pad=256, max_conv_len=18, dtype="float16", block_n=64):
    n_tiles = _ceil(d_pad, block_n)
    nshift = max_conv_len - 1

    @T.macro
    def _shift(conv_state, new_conv_state, shift_ub, n0, an):
        T.copy(
            conv_state[0:batch, 1:max_conv_len, n0 : n0 + an],
            shift_ub[0:batch, 0:nshift, 0:an],
        )
        T.copy(
            shift_ub[0:batch, 0:nshift, 0:an],
            new_conv_state[0:batch, 0:nshift, n0 : n0 + an],
        )

    @T.prim_func
    def main(
        conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype),
        new_conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype),
    ):
        with T.Kernel(n_tiles) as bx:
            shift_ub = T.alloc_shared((batch, max(1, nshift), block_n), dtype)
            n0 = bx * block_n
            an = T.min(d_pad - n0, block_n)
            _shift(conv_state, new_conv_state, shift_ub, n0, an)

    return tilelang.compile(main, target="ascend", out_idx=[])


# ---------------------------------------------------------------------------
# P6: K tail handled by an explicit T.clear of the L1 tail (no pad_value)
# ---------------------------------------------------------------------------
def build_p6(batch=17, d_mem=519, d_pad=256, dtype="float16", block_m=16, block_n=64,
             block_k=256, num_stages=2):
    n_tiles = _ceil(d_pad, block_n)
    k_tiles = _ceil(d_mem, block_k)

    @T.macro
    def _one(E, W, kv, e_l1, w_l1, res, out_idx, n0, am, an):
        for kt in T.Pipelined(k_tiles, num_stages=num_stages):
            k0 = kt * block_k
            ak = T.min(d_mem - k0, block_k)
            T.copy(E[0:am, k0 : k0 + ak], e_l1[0:am, 0:ak])
            T.copy(W[k0 : k0 + ak, n0 : n0 + an], w_l1[0:an, 0:ak], transpose=True)
            if d_mem % block_k != 0:
                T.clear(e_l1[0:am, ak:block_k])
                T.clear(w_l1[0:an, ak:block_k])
            T.gemm(e_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
        T.copy(res[0:am, 0:an], kv[out_idx, 0:am, n0 : n0 + an])

    @T.prim_func
    def main(
        E: T.Tensor((batch, d_mem), dtype),
        W_K: T.Tensor((d_mem, d_pad), dtype),
        W_V: T.Tensor((d_mem, d_pad), dtype),
        kv: T.Tensor((2, batch, d_pad), "float32"),
    ):
        with T.Kernel(2 * n_tiles) as bx:
            e_l1 = T.alloc_l1((block_m, block_k), dtype)
            w_l1 = T.alloc_l1((block_n, block_k), dtype)
            res = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions({e_l1: num_stages, w_l1: num_stages})
            which = bx // n_tiles
            n0 = (bx % n_tiles) * block_n
            am = T.min(batch, block_m)
            an = T.min(d_pad - n0, block_n)
            if which == 0:
                _one(E, W_K, kv, e_l1, w_l1, res, 0, n0, am, an)
            else:
                _one(E, W_V, kv, e_l1, w_l1, res, 1, n0, am, an)

    return tilelang.compile(main, target="ascend", out_idx=[])


# ---------------------------------------------------------------------------
# P7: full design with the shift (3-way grid), parameterised for the menu
#     max_conv_len == 1 -> no shift blocks at all
# ---------------------------------------------------------------------------
def build_p7(batch=1, d_mem=512, d_pad=256, max_conv_len=12, dtype="float16",
             block_m=16, block_n=64, block_k=256, num_stages=2):
    n_tiles = _ceil(d_pad, block_n)
    k_tiles = _ceil(d_mem, block_k)
    nshift = max_conv_len - 1
    slices = 3 if nshift > 0 else 2

    @T.macro
    def _one(E, W, kv, e_l1, w_l1, res, out_idx, n0, am, an):
        for kt in T.Pipelined(k_tiles, num_stages=num_stages):
            k0 = kt * block_k
            T.copy(E[0:am, k0 : k0 + block_k], e_l1[0:am, 0:block_k])
            T.copy(
                W[k0 : k0 + block_k, n0 : n0 + an],
                w_l1[0:an, 0:block_k],
                transpose=True,
            )
            T.gemm(e_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
        T.copy(res[0:am, 0:an], kv[out_idx, 0:am, n0 : n0 + an])

    @T.macro
    def _shift(conv_state, new_conv_state, shift_ub, n0, an):
        for b in T.serial(batch):
            T.copy(conv_state[b, 1:max_conv_len, n0 : n0 + an], shift_ub[0:nshift, 0:an])
            T.copy(shift_ub[0:nshift, 0:an], new_conv_state[b, 0:nshift, n0 : n0 + an])

    @T.prim_func
    def main(
        E: T.Tensor((batch, d_mem), dtype),
        W_K: T.Tensor((d_mem, d_pad), dtype),
        W_V: T.Tensor((d_mem, d_pad), dtype),
        conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype),
        kv: T.Tensor((2, batch, d_pad), "float32"),
        new_conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype),
    ):
        with T.Kernel(slices * n_tiles) as bx:
            e_l1 = T.alloc_l1((block_m, block_k), dtype)
            w_l1 = T.alloc_l1((block_n, block_k), dtype)
            res = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions({e_l1: num_stages, w_l1: num_stages})
            which = bx // n_tiles
            n0 = (bx % n_tiles) * block_n
            am = T.min(batch, block_m)
            an = T.min(d_pad - n0, block_n)
            if which == 0:
                _one(E, W_K, kv, e_l1, w_l1, res, 0, n0, am, an)
            elif which == 1:
                _one(E, W_V, kv, e_l1, w_l1, res, 1, n0, am, an)
            else:
                shift_ub = T.alloc_shared((nshift, block_n), dtype)
                _shift(conv_state, new_conv_state, shift_ub, n0, an)

    return tilelang.compile(main, target="ascend", out_idx=[])


# ---------------------------------------------------------------------------
# P8: K tail handled by clearing the whole L1 tile (C0-aligned) before the copy,
#     single-buffered (num_stages = 1) so the fill's buffer version is unambiguous
# ---------------------------------------------------------------------------
def build_p8(batch=17, d_mem=519, d_pad=256, dtype="float16", block_m=16, block_n=64,
             block_k=256):
    n_tiles = _ceil(d_pad, block_n)
    k_tiles = _ceil(d_mem, block_k)

    @T.macro
    def _one(E, W, kv, e_l1, w_l1, res, out_idx, n0, am, an):
        for kt in T.Pipelined(k_tiles, num_stages=1):
            k0 = kt * block_k
            ak = T.min(d_mem - k0, block_k)
            T.clear(e_l1)
            T.clear(w_l1)
            T.copy(E[0:am, k0 : k0 + ak], e_l1[0:am, 0:ak])
            T.copy(W[k0 : k0 + ak, n0 : n0 + an], w_l1[0:an, 0:ak], transpose=True)
            T.gemm(e_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
        T.copy(res[0:am, 0:an], kv[out_idx, 0:am, n0 : n0 + an])

    @T.prim_func
    def main(
        E: T.Tensor((batch, d_mem), dtype),
        W_K: T.Tensor((d_mem, d_pad), dtype),
        W_V: T.Tensor((d_mem, d_pad), dtype),
        kv: T.Tensor((2, batch, d_pad), "float32"),
    ):
        with T.Kernel(2 * n_tiles) as bx:
            e_l1 = T.alloc_l1((block_m, block_k), dtype)
            w_l1 = T.alloc_l1((block_n, block_k), dtype)
            res = T.alloc_l0c((block_m, block_n), "float32")
            which = bx // n_tiles
            n0 = (bx % n_tiles) * block_n
            am = T.min(batch, block_m)
            an = T.min(d_pad - n0, block_n)
            if which == 0:
                _one(E, W_K, kv, e_l1, w_l1, res, 0, n0, am, an)
            else:
                _one(E, W_V, kv, e_l1, w_l1, res, 1, n0, am, an)

    return tilelang.compile(main, target="ascend", out_idx=[])


# ---------------------------------------------------------------------------
# P9: N tail (d_pad not a multiple of block_n) + full design with shift
# ---------------------------------------------------------------------------
def build_p9(batch=3, d_mem=512, d_pad=272, max_conv_len=12, dtype="float16",
             block_m=16, block_n=64, block_k=256, num_stages=2):
    n_tiles = _ceil(d_pad, block_n)
    k_tiles = _ceil(d_mem, block_k)
    nshift = max_conv_len - 1

    @T.macro
    def _one(E, W, kv, e_l1, w_l1, res, out_idx, n0, am, an):
        for kt in T.Pipelined(k_tiles, num_stages=num_stages):
            k0 = kt * block_k
            T.copy(E[0:am, k0 : k0 + block_k], e_l1[0:am, 0:block_k])
            T.copy(
                W[k0 : k0 + block_k, n0 : n0 + an],
                w_l1[0:an, 0:block_k],
                transpose=True,
            )
            T.gemm(e_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
        T.copy(res[0:am, 0:an], kv[out_idx, 0:am, n0 : n0 + an])

    @T.macro
    def _shift(conv_state, new_conv_state, shift_ub, n0, an):
        for b in T.serial(batch):
            T.copy(conv_state[b, 1:max_conv_len, n0 : n0 + an], shift_ub[0:nshift, 0:an])
            T.copy(shift_ub[0:nshift, 0:an], new_conv_state[b, 0:nshift, n0 : n0 + an])

    @T.prim_func
    def main(
        E: T.Tensor((batch, d_mem), dtype),
        W_K: T.Tensor((d_mem, d_pad), dtype),
        W_V: T.Tensor((d_mem, d_pad), dtype),
        conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype),
        kv: T.Tensor((2, batch, d_pad), "float32"),
        new_conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype),
    ):
        with T.Kernel(3 * n_tiles) as bx:
            e_l1 = T.alloc_l1((block_m, block_k), dtype)
            w_l1 = T.alloc_l1((block_n, block_k), dtype)
            res = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions({e_l1: num_stages, w_l1: num_stages})
            which = bx // n_tiles
            n0 = (bx % n_tiles) * block_n
            am = T.min(batch, block_m)
            an = T.min(d_pad - n0, block_n)
            if which == 0:
                _one(E, W_K, kv, e_l1, w_l1, res, 0, n0, am, an)
            elif which == 1:
                _one(E, W_V, kv, e_l1, w_l1, res, 1, n0, am, an)
            else:
                shift_ub = T.alloc_shared((nshift, block_n), dtype)
                _shift(conv_state, new_conv_state, shift_ub, n0, an)

    return tilelang.compile(main, target="ascend", out_idx=[])


VARIANTS = {
    "P1_cube_only_64x256": lambda: build_p1(block_n=64, block_k=256),
    "P1_cube_only_32x256": lambda: build_p1(block_n=32, block_k=256),
    "P1_cube_only_128x128": lambda: build_p1(block_n=128, block_k=128),
    "P1_cube_only_b4_d512": lambda: build_p1(batch=4, d_mem=1024, d_pad=512, block_n=128,
                                             block_k=256),
    "P2_shift_in_cube_kernel": lambda: build_p2(),
    "P2_shift_attached_which0": lambda: build_p2(shift_grid=False),
    "P3_mixed_cube_vector": lambda: build_p3(),
    "P4_ktail_519_pad": lambda: build_p4(pad=True),
    "P4_ktail_519_nopad": lambda: build_p4(pad=False),
    "P5_shift_3d_region": lambda: build_p5(),
    "P6_ktail_explicit_clear": lambda: build_p6(),
    "P7_full_b1_d256_bn32": lambda: build_p7(batch=1, d_mem=512, d_pad=256, block_n=32,
                                            block_k=256),
    "P7_full_b8_d256_bn32_bf16": lambda: build_p7(batch=8, d_mem=512, d_pad=256,
                                                  dtype="bfloat16", block_n=32,
                                                  block_k=256, num_stages=2),
    "P7_full_b4_d512_bn64": lambda: build_p7(batch=4, d_mem=1024, d_pad=512, block_n=64,
                                             block_k=256, num_stages=2),
    "P7_full_maxconv1_noshift": lambda: build_p7(batch=1, d_mem=512, d_pad=256,
                                                 max_conv_len=1, block_n=32),
    "P7_full_b1_d256_bn32_ns1": lambda: build_p7(batch=1, d_mem=256, d_pad=256,
                                                 block_n=32, block_k=256, num_stages=1),
    "P8_ktail_clear_whole_tile": lambda: build_p8(),
    "P9_ntail_dpad272": lambda: build_p9(),
}


def main():
    only = sys.argv[1:] or list(VARIANTS)
    ok = fail = 0
    for name in only:
        fn = VARIANTS[name]
        try:
            compiled = fn()
            print(f"[OK]   {name}: compile passed; marks = {_marks(compiled)}")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
            fail += 1
    print(f"\nRESULT: {ok} compiled, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
