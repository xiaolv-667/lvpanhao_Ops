# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Repro: a row scalar carried through ``T.Parallel(1)`` is per-lane code, so it can be
broadcast to the elementwise pass without UB staging and without a memory barrier.

Target: ascend-950pr-9579 / dav-3510, tilelang 0.1.15 + CANN 9.2.1.

Context: a row-reduction kernel (RMSNorm and friends) finishes a row statistic with
``T.finalize_reducer``, then needs one scalar -- ``rsqrt(mean + eps)``, a sigmoid gate --
in the following elementwise pass.  Two ways to express it:

  V1  keep it in a ``(1,)`` fragment written inside ``T.Parallel(1)`` (the shape most
      design documents draw);
  V2  keep it as a plain Python PrimExpr built from the finalized reducer fragment and
      inline it into the elementwise expression.

Question this answers: does V1 write the value on lane 0 only (a per-thread register
would then leave every other lane reading garbage)?  Generated code says no -- the
single-iteration loop is emitted unconditionally in every lane, so ``alpha[0]`` is a
replicated scalar:

    a[0] = rsqrtf(...);            # no `if (threadIdx.x == 0)` guard

Assertions below, on device, for a 4-row / 256-wide RMSNorm:

  A  V1 matches a torch fp32 reference within 1e-5 relative;
  B  V2 matches the same reference;
  C  V1 and V2 agree bit for bit, i.e. the two forms are interchangeable;
  D  the V1 VF body contains the scalar expression and no thread-index guard on it.

run:
  cd examples/engram_decode/_engram_step_kernel
  source <repo>/.agents/platforms/toolchains/ascend/env-950pr.sh
  ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python \
      repro/PL-950-simt-scalar-broadcast.py
"""

import torch
import torch_npu  # noqa: F401 -- registers torch.npu
import tilelang
import tilelang.ascend.language as T

ROWS, D, TILE = 4, 256, 256
EPS = 1e-6
INV_D = 1.0 / D


def build(v):
    @T.prim_func
    def main(
        X: T.Tensor((ROWS, D), "float32"),
        W: T.Tensor((D,), "float32"),
        Y: T.Tensor((ROWS, D), "float32"),
    ):
        with T.Kernel(ROWS) as bx:
            x_ub = T.alloc_shared((D,), "float32")
            w_ub = T.alloc_shared((D,), "float32")
            y_ub = T.alloc_shared((D,), "float32")
            T.copy(X[bx, :], x_ub)
            T.copy(W, w_ub)
            with T.SimtVF(threads=TILE):
                xf = T.alloc_fragment((TILE,), "float32")
                acc = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(acc)
                for i in T.Parallel(TILE, coalesced_width=T.int32(1)):
                    xf[i] = x_ub[i]
                    T.reducer_update(acc[0], x_ub[i] * x_ub[i])
                s = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc, s)
                if v == 1:
                    a = T.alloc_fragment((1,), "float32")
                    for i in T.Parallel(1):
                        a[i] = T.rsqrt(s[0] * INV_D + EPS)
                    for i in T.Parallel(TILE, coalesced_width=T.int32(1)):
                        y_ub[i] = xf[i] * a[0] * w_ub[i]
                else:
                    a = T.rsqrt(s[0] * INV_D + EPS)
                    for i in T.Parallel(TILE, coalesced_width=T.int32(1)):
                        y_ub[i] = xf[i] * a * w_ub[i]
            T.copy(y_ub, Y[bx, :])

    return tilelang.compile(main, target="ascend", out_idx=[])


def golden(x, w):
    rrms = (x.float() ** 2).mean(dim=-1, keepdim=True).add(EPS).rsqrt()
    return x.float() * rrms * w.float()


def main():
    torch.manual_seed(0)
    x = torch.randn(ROWS, D)
    w = 1.0 + 0.1 * torch.randn(D)
    ref = golden(x, w)

    outs = {}
    for v in (1, 2):
        y = torch.zeros(ROWS, D, device="npu")
        ker = build(v)
        ker(x.to("npu"), w.to("npu"), y)
        torch.npu.synchronize()
        outs[v] = y.cpu()
        diff = (outs[v] - ref).abs().max().item()
        assert diff < 1e-5, f"V{v} differs from the fp32 reference by {diff:.3e}"
        print(f"[{'AB'[v - 1]}] V{v} max_diff={diff:.3e}")

    same = torch.equal(outs[1].view(torch.int32), outs[2].view(torch.int32))
    assert same, "the two scalar-carry forms are not bit identical"
    print("[C] V1 and V2 are bit identical")

    src = build(1).get_kernel_source()
    body = src.split("__simt_vf__")[1].split("}")[0]
    assert "rsqrtf" in body, "the scalar must be computed inside the VF"
    assert "threadIdx.x == 0" not in body and "threadIdx.x < 1" not in body, (
        "the scalar is guarded to one lane"
    )
    print("[D] the (1,) fragment write is emitted unguarded in every lane")
    print("RESULT: scalar broadcast via T.Parallel(1) is per-lane and needs no barrier")


if __name__ == "__main__":
    main()
