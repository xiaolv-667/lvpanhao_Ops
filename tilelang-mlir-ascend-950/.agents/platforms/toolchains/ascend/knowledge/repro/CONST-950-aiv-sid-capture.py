"""CONST-950-aiv-sid-capture: how to read the AIV sub-block id inside user code.
First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-23; origin_task batch2-20261123.

Found while implementing ``_bwd_kernel`` (stage 3, probe P1d).

Context
-------
Once the L0C -> UB crossing is a ``T.dual_copy`` 2:1 split (see
``TRAP-950-l0c2ub-single-aiv.py``) each AIV owns *half* of the tile, so any
per-element expression that needs the global index of the split axis (here the
key row of an attention block, for the causal mask) becomes AIV dependent.

The obvious spelling ``with T.Vector(vector=2) as sid:`` is rejected by this
toolchain whenever it wraps the auto-scheduled body:

    InternalError: Check failed: (!var_idmap_.count(v)) is false:
    Need input to be in SSA form dup sid

``asc_get_sub_block_id()`` is also not callable *inside* a ``T.SimtVF`` body
(the device compiler has no matching overload there: the ``__simt_vf__``
function is emitted outside the AIV prologue).

What works
----------
Call it once in the kernel body as an extern scalar and let the auto-scheduler
hoist it into the AIV branch; it is then captured into every ``T.SimtVF`` as an
argument:

    sid = T.call_extern("int64", "asc_get_sub_block_id")
    with T.SimtVF(threads=128):
        for i, j in T.Parallel(32, 64):
            ... gi = n0 + T.cast(sid, "int32") * 32 + i ...

and the value equals the id that ``dual_copy`` split with (each AIV sees its own
half).  This script asserts exactly that: the half tile is offset by
``sid * 1000`` and each GM row group must carry its own offset.

Run:

    source .agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python \
        examples/mla_dense_bwd/_bwd_kernel/repro/CONST-950-aiv-sid-capture.py
"""

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

BN = 64
BM = 64
DQK = 192
HN = BN // 2
DTYPE = "float16"


def build_sid():
    @T.prim_func
    def main(
        k: T.Tensor((BN, DQK), DTYPE),
        q: T.Tensor((BM, DQK), DTYPE),
        out: T.Tensor((BN, BM), "float32"),
    ):
        with T.Kernel(1):
            k_s = T.alloc_l1((BN, DQK), DTYPE)
            q_s = T.alloc_l1((BM, DQK), DTYPE)
            c = T.alloc_l0c((BN, BM), "float32")
            ub = T.alloc_shared((HN, BM), "float32")
            sid = T.call_extern("int64", "asc_get_sub_block_id")
            T.copy(k, k_s)
            T.copy(q, q_s)
            T.gemm(k_s, q_s, c, transpose_B=True, clear_accum=True)
            T.dual_copy(c, ub)
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(HN, BM):
                    ub[i, j] = ub[i, j] + T.cast(T.cast(sid, "int32") * 1000, "float32")
            T.dual_copy(ub, out)

    return tilelang.compile(main, target="ascend")


def build_vector_frame():
    """The rejected spelling, kept to pin the error message."""

    @T.prim_func
    def main(
        k: T.Tensor((BN, DQK), DTYPE),
        q: T.Tensor((BM, DQK), DTYPE),
        out: T.Tensor((BN, BM), "float32"),
    ):
        with T.Kernel(1):
            k_s = T.alloc_l1((BN, DQK), DTYPE)
            q_s = T.alloc_l1((BM, DQK), DTYPE)
            c = T.alloc_l0c((BN, BM), "float32")
            ub = T.alloc_shared((HN, BM), "float32")
            T.copy(k, k_s)
            T.copy(q, q_s)
            T.gemm(k_s, q_s, c, transpose_B=True, clear_accum=True)
            T.dual_copy(c, ub)
            with T.Vector(vector=2) as sid:  # "dup sid"
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(HN, BM):
                        ub[i, j] = ub[i, j] + T.cast(sid * 1000, "float32")
            T.dual_copy(ub, out)

    return tilelang.compile(main, target="ascend")


def main():
    torch.manual_seed(0)
    dt = getattr(torch, DTYPE)
    k = torch.randn(BN, DQK).to(dt)
    q = torch.randn(BM, DQK).to(dt)
    ref = k.float() @ q.float().T

    out = torch.full((BN, BM), -777.0, device="npu", dtype=torch.float32)
    build_sid()(k.to("npu"), q.to("npu"), out)
    got = out.cpu()
    torch.testing.assert_close(got[:HN], ref[:HN], atol=1e-2, rtol=1e-2)  # sid == 0
    torch.testing.assert_close(got[HN:], ref[HN:] + 1000.0, atol=1e-2, rtol=1e-2)  # sid == 1
    print("captured sid: rows [0,32) carry sid*1000 = 0, rows [32,64) carry 1000")

    try:
        build_vector_frame()
    except Exception as exc:  # noqa: BLE001 - the message is the assertion
        text = f"{type(exc).__name__}: {exc}"
        print("T.Vector(vector=2) as sid ->", text.splitlines()[0][:160])
        assert "dup sid" in text, "expected the documented SSA 'dup sid' rejection"
    else:
        raise AssertionError("T.Vector(vector=2) as sid was expected to be rejected")
    print("CONST-950-aiv-sid-capture: PASS")


if __name__ == "__main__":
    main()
