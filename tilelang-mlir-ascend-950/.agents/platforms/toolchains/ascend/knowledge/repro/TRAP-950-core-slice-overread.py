"""TRAP-950-core-slice-overread. First verified: TileLang ascend 0.1.15 / CANN 9.2.1 / Ascend950PR_9579.

A multi-core partial-histogram launch whose per-chunk length is clamped only by
the tensor end --

    n = min(CHUNK, numel - base)          # base = g * TPC

-- silently *double counts* whenever the per-chunk tile is larger than the
per-core slice ``TPC``: core ``g`` then reads ``CHUNK - TPC`` tokens that belong
to core ``g+1``, so the merged histogram is too large and every downstream
output (prefix sum, block map) is wrong.  ``CHUNK = align_up(TPC, 64)`` -- the
natural 64-lane tile -- is always > ``TPC`` unless ``TPC`` is a multiple of 64,
so this bites exactly the ragged geometries and passes on the round ones.

Measured on Ascend950PR_9579 (real device, int32, G=4, TPC=250, CHUNK=256):

  * clamp by the tensor end only: rows 0..2 over-count (sum 256 instead of 250)
  * clamp by the core slice as well: every row matches ``torch.bincount``
      n = max(0, min(min(TPC - c*CHUNK, CHUNK), numel - base))

Run (device, from the task directory):
    source .agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python \
        repro/TRAP-950-core-slice-overread.py

Exit code 0 = the over-read still reproduces and the clamped form is still
exact. A failing assertion means the behaviour changed: re-verify before
trusting either claim.
"""

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

G = 4
TPC = 250          # tokens per core, not a multiple of 64
CHUNK = 256        # = align_up(TPC, 64) > TPC
N = G * TPC        # 1000 tokens in total
E_PAD = 64


def build(clamped):
    n_chunks = (TPC + CHUNK - 1) // CHUNK

    @T.prim_func
    def main(X: T.Tensor((N,), "int32"), PART: T.Tensor((G, E_PAD), "int32")):
        with T.Kernel(G) as g:
            s_x = T.alloc_shared((CHUNK,), "int32")
            s_hist = T.alloc_shared((E_PAD,), "int32")
            with T.SimtVF(threads=E_PAD):
                for e in T.Parallel(E_PAD):
                    s_hist[e] = 0
            for c in T.serial(n_chunks):
                base = g * TPC + c * CHUNK
                if clamped:
                    # the core's own slice AND the tensor end
                    n = T.max(0, T.min(T.min(TPC - c * CHUNK, CHUNK), N - base))
                else:
                    # tensor end only: reads into the next core's slice
                    n = T.min(CHUNK, N - base)
                if n > 0:
                    T.copy(X[base : base + n], s_x[0:n])
                    with T.SimtVF(threads=E_PAD):
                        for e in T.Parallel(E_PAD):
                            acc = T.alloc_local([1], "int32")
                            acc[0] = 0
                            for i in T.serial(n):
                                acc[0] = acc[0] + T.if_then_else(s_x[i] == e, 1, 0)
                            s_hist[e] = s_hist[e] + acc[0]
            T.copy(s_hist, PART[g, :])

    return tilelang.compile(main, target="ascend", out_idx=[])


def run(clamped):
    x = (torch.arange(N, dtype=torch.int32) % 7)
    part = torch.zeros((G, E_PAD), dtype=torch.int32, device="npu")
    build(clamped)(x.npu(), part)
    torch.npu.synchronize()
    got = part.cpu()
    ref = torch.stack(
        [
            torch.bincount(x[g * TPC : min((g + 1) * TPC, N)], minlength=E_PAD)[:E_PAD]
            for g in range(G)
        ]
    ).to(torch.int32)
    return got, ref


def main():
    bad, ref = run(clamped=False)
    over = [g for g in range(G) if int(bad[g].sum()) != TPC]
    print(f"[tensor-end-only clamp] rows over-counting: {over} "
          f"(row sums {[int(bad[g].sum()) for g in range(G)]}, TPC={TPC})")
    assert over, "the over-read no longer reproduces: re-verify the histogram tiling"

    good, _ = run(clamped=True)
    assert torch.equal(good, ref), "the core-slice clamp is not exact any more"
    print("[core-slice clamp] all rows match torch.bincount")
    print("RESULT: over-read reproduced, core-slice clamp exact (Ascend950PR_9579, int32)")


if __name__ == "__main__":
    main()
