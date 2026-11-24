"""Design-time compile probe (D-3) #2: where the cache shift lives.

Compile-only: no NPU tensor is created and no kernel is launched, so no device
is occupied.  Run from the task directory:

    source .agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        history_version/design_probe_shift_placement.py

Questions (DESIGN.md §0.5 / §5 / §11):

  X1/X2  explicit ``T.MixedKernel(grid, sids=2)`` with a Cube GEMM region and a
         Vector shift region -- the obvious way to fuse both units in one
         launch.  Recorded here because it *fails* in this toolchain:
         ``Need input to be in SSA form dup sid`` (with and without the user
         touching ``sid``), so it is not a usable design.
  A_*    plain ``T.Kernel(grid)`` holding both the L1/L0 Cube GEMM and the
         UB-staged shift; the compiler promotes it to ``__mix__(1, 2)`` on its
         own.  The menu covers every manifest workload plus the M/K/N tails
         and ``max_conv_len == 1``.
  A_l1shift  the same kernel with the shift staged through L1 instead of UB
             (would keep the shift on the AIC and avoid the duplicated AIV
             store) -- checked for comparison.
"""

import sys
import traceback

import tilelang
import tilelang.ascend.language as T


def _ceil(a, b):
    return -(-a // b)


def build_mixed(batch=8, d_mem=512, d_pad=256, max_conv_len=18, dtype="float16",
                block_m=16, block_n=64, block_k=256, num_stages=2):
    """X1/X2: explicit MixedKernel (expected to fail -- evidence only)."""
    m_tiles = _ceil(batch, block_m)
    n_tiles = _ceil(d_pad, block_n)
    k_tiles = _ceil(d_mem, block_k)
    nshift = max_conv_len - 1
    grid = m_tiles * 2 * n_tiles

    @T.macro
    def _proj(E, W, kv, out_idx, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an):
        for kt in T.Pipelined(k_tiles, num_stages=num_stages):
            k0 = kt * block_k
            ak = T.min(d_mem - k0, block_k)
            T.copy(E[m0 : m0 + am, k0 : k0 + ak], e_l1[0:am, 0:ak])
            T.copy(W[k0 : k0 + ak, n0 : n0 + an], w_l1[0:an, 0:ak], transpose=True)
            T.copy(e_l1[0:am, 0:ak], e_l0[0:am, 0:ak])
            T.copy(w_l1[0:an, 0:ak], w_l0[0:an, 0:ak])
            T.gemm(
                e_l0[0:am, 0:ak],
                w_l0[0:an, 0:ak],
                res[0:am, 0:an],
                transpose_B=True,
                clear_accum=(kt == 0),
            )
        T.copy(res[0:am, 0:an], kv[out_idx, m0 : m0 + am, n0 : n0 + an])

    @T.prim_func
    def main(
        E: T.Tensor((batch, d_mem), dtype),
        W_K: T.Tensor((d_mem, d_pad), dtype),
        W_V: T.Tensor((d_mem, d_pad), dtype),
        conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype),
        kv: T.Tensor((2, batch, d_pad), "float32"),
        new_conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype),
    ):
        with T.MixedKernel(grid, sids=2) as (bx, sid):
            e_l1 = T.alloc_l1((block_m, block_k), dtype)
            w_l1 = T.alloc_l1((block_n, block_k), dtype)
            e_l0 = T.alloc_l0a((block_m, block_k), dtype)
            w_l0 = T.alloc_l0b((block_n, block_k), dtype)
            res = T.alloc_l0c((block_m, block_n), "float32")
            shift_ub = T.alloc_shared((max(1, nshift), block_n), dtype)
            T.annotate_buffer_versions(
                {e_l1: num_stages, w_l1: num_stages, e_l0: num_stages, w_l0: num_stages}
            )
            m0 = (bx // (2 * n_tiles)) * block_m
            which = (bx // n_tiles) % 2
            n0 = (bx % n_tiles) * block_n
            am = T.min(batch - m0, block_m)
            an = T.min(d_pad - n0, block_n)
            with T.Cube():
                if which == 0:
                    _proj(E, W_K, kv, 0, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
                else:
                    _proj(E, W_V, kv, 1, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
            with T.Vector(vector=2):
                if nshift > 0:
                    if which == 0 and m0 == 0:
                        if sid == 0:
                            for b in T.serial(batch):
                                T.copy(
                                    conv_state[b, 1:max_conv_len, n0 : n0 + an],
                                    shift_ub[0:nshift, 0:an],
                                )
                                T.copy(
                                    shift_ub[0:nshift, 0:an],
                                    new_conv_state[b, 0:nshift, n0 : n0 + an],
                                )

    return tilelang.compile(main, target="ascend", out_idx=[])


def build_plain(batch=8, d_mem=512, d_pad=256, max_conv_len=18, dtype="float16",
                block_m=16, block_n=64, block_k=256, num_stages=2, shift_stage="ub"):
    """A_*: plain T.Kernel; the compiler promotes it to __mix__(1, 2) itself."""
    m_tiles = _ceil(batch, block_m)
    n_tiles = _ceil(d_pad, block_n)
    k_tiles = _ceil(d_mem, block_k)
    nshift = max_conv_len - 1
    grid = m_tiles * 2 * n_tiles

    @T.macro
    def _proj(E, W, kv, out_idx, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an):
        for kt in T.Pipelined(k_tiles, num_stages=num_stages):
            k0 = kt * block_k
            ak = T.min(d_mem - k0, block_k)
            T.copy(E[m0 : m0 + am, k0 : k0 + ak], e_l1[0:am, 0:ak])
            T.copy(W[k0 : k0 + ak, n0 : n0 + an], w_l1[0:an, 0:ak], transpose=True)
            T.copy(e_l1[0:am, 0:ak], e_l0[0:am, 0:ak])
            T.copy(w_l1[0:an, 0:ak], w_l0[0:an, 0:ak])
            T.gemm(
                e_l0[0:am, 0:ak],
                w_l0[0:an, 0:ak],
                res[0:am, 0:an],
                transpose_B=True,
                clear_accum=(kt == 0),
            )
        T.copy(res[0:am, 0:an], kv[out_idx, m0 : m0 + am, n0 : n0 + an])

    @T.macro
    def _shift(conv_state, new_conv_state, buf, n0, an):
        for b in T.serial(batch):
            T.copy(conv_state[b, 1:max_conv_len, n0 : n0 + an], buf[0:nshift, 0:an])
            T.copy(buf[0:nshift, 0:an], new_conv_state[b, 0:nshift, n0 : n0 + an])

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
            e_l0 = T.alloc_l0a((block_m, block_k), dtype)
            w_l0 = T.alloc_l0b((block_n, block_k), dtype)
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
                _proj(E, W_K, kv, 0, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
                if nshift > 0 and m0 == 0:
                    if shift_stage == "ub":
                        buf = T.alloc_shared((nshift, block_n), dtype)
                        _shift(conv_state, new_conv_state, buf, n0, an)
                    elif shift_stage == "l1":
                        buf = T.alloc_l1((nshift, block_n), dtype)
                        _shift(conv_state, new_conv_state, buf, n0, an)
            else:
                _proj(E, W_V, kv, 1, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)

    return tilelang.compile(main, target="ascend", out_idx=[])


def build_plain_sidguard(batch=8, d_mem=512, d_pad=256, max_conv_len=18,
                         dtype="float16", block_m=16, block_n=64, block_k=256,
                         num_stages=2):
    """A_sid: plain T.Kernel + an explicit ``T.Vector`` region binding ``sid``,
    so the two AIV sub-blocks split the shift rows instead of both storing the
    same bytes."""
    m_tiles = _ceil(batch, block_m)
    n_tiles = _ceil(d_pad, block_n)
    k_tiles = _ceil(d_mem, block_k)
    nshift = max_conv_len - 1
    grid = m_tiles * 2 * n_tiles
    rps = max(1, _ceil(batch, 2))

    @T.macro
    def _proj(E, W, kv, out_idx, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an):
        for kt in T.Pipelined(k_tiles, num_stages=num_stages):
            k0 = kt * block_k
            ak = T.min(d_mem - k0, block_k)
            T.copy(E[m0 : m0 + am, k0 : k0 + ak], e_l1[0:am, 0:ak])
            T.copy(W[k0 : k0 + ak, n0 : n0 + an], w_l1[0:an, 0:ak], transpose=True)
            T.copy(e_l1[0:am, 0:ak], e_l0[0:am, 0:ak])
            T.copy(w_l1[0:an, 0:ak], w_l0[0:an, 0:ak])
            T.gemm(
                e_l0[0:am, 0:ak],
                w_l0[0:an, 0:ak],
                res[0:am, 0:an],
                transpose_B=True,
                clear_accum=(kt == 0),
            )
        T.copy(res[0:am, 0:an], kv[out_idx, m0 : m0 + am, n0 : n0 + an])

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
            e_l0 = T.alloc_l0a((block_m, block_k), dtype)
            w_l0 = T.alloc_l0b((block_n, block_k), dtype)
            res = T.alloc_l0c((block_m, block_n), "float32")
            shift_ub = T.alloc_shared((max(1, nshift), block_n), dtype)
            T.annotate_buffer_versions(
                {e_l1: num_stages, w_l1: num_stages, e_l0: num_stages, w_l0: num_stages}
            )
            m0 = (bx // (2 * n_tiles)) * block_m
            which = (bx // n_tiles) % 2
            n0 = (bx % n_tiles) * block_n
            am = T.min(batch - m0, block_m)
            an = T.min(d_pad - n0, block_n)
            if which == 0:
                _proj(E, W_K, kv, 0, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
            else:
                _proj(E, W_V, kv, 1, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
            if nshift > 0:
                with T.Vector(vector=2) as sid:
                    if which == 0 and m0 == 0:
                        if sid == 0:
                            for b in T.serial(rps):
                                T.copy(
                                    conv_state[b, 1:max_conv_len, n0 : n0 + an],
                                    shift_ub[0:nshift, 0:an],
                                )
                                T.copy(
                                    shift_ub[0:nshift, 0:an],
                                    new_conv_state[b, 0:nshift, n0 : n0 + an],
                                )
                        else:
                            for i in T.serial(rps):
                                if rps + i < batch:
                                    T.copy(
                                        conv_state[rps + i, 1:max_conv_len, n0 : n0 + an],
                                        shift_ub[0:nshift, 0:an],
                                    )
                                    T.copy(
                                        shift_ub[0:nshift, 0:an],
                                        new_conv_state[rps + i, 0:nshift, n0 : n0 + an],
                                    )

    return tilelang.compile(main, target="ascend", out_idx=[])


VARIANTS = {
    # explicit MixedKernel -- expected to fail (evidence)
    "X1_mixed_b8": lambda: build_mixed(),
    "X2_mixed_b1_bn32": lambda: build_mixed(batch=1, max_conv_len=12, block_n=32),
    # plain kernel promoted to __mix__(1, 2)
    "A_b8_bf16": lambda: build_plain(dtype="bfloat16"),
    "A_b8_fp16": lambda: build_plain(),
    "A_b1_bn32": lambda: build_plain(batch=1, max_conv_len=12, block_n=32),
    "A_b4_d512_bn64": lambda: build_plain(batch=4, d_mem=1024, d_pad=512, max_conv_len=20),
    "A_mtail_ktail_17x519": lambda: build_plain(batch=17, d_mem=519, max_conv_len=12),
    "A_noshift_mcl1": lambda: build_plain(max_conv_len=1),
    "A_ntail_dpad272": lambda: build_plain(batch=2, d_pad=272, max_conv_len=12),
    "A_sidguard_b8": lambda: build_plain_sidguard(),
    "A_sidguard_b1": lambda: build_plain_sidguard(batch=1, max_conv_len=12, block_n=32),
    "A_sidguard_mtail": lambda: build_plain_sidguard(batch=17, d_mem=519, max_conv_len=12),
    "A_l1shift": lambda: build_plain(shift_stage="l1"),
}


def main():
    only = sys.argv[1:] or list(VARIANTS)
    ok = fail = 0
    for name in only:
        try:
            VARIANTS[name]()
            print(f"[OK]   {name}: compile passed")
            ok += 1
        except Exception as exc:  # noqa: BLE001 - probe
            print(f"[FAIL] {name}: {type(exc).__name__}: {str(exc)[:300]}")
            fail += 1
    print(f"\nRESULT: {ok} compiled, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
