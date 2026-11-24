# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""TRAP-950-scatter-eid-upper-bound: an ``eid == num_experts`` route bumps ``cumsum[E]``.
First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-23; origin_task batch2-20261123.

Context.  A per-route bump cursor in a SIMT VF kernel is written

    if eid >= 0: slot = asc_atomic_add(&cumsum[eid], 1); sorted[slot] = idx

which is exactly the source GPU kernel's guard.  The compiler inserts an
automatic value-range guard on the index expression from the buffer extent, so
``cumsum[eid]`` is guarded by ``eid < num_experts + 1`` -- i.e. the extent of the
``cumsum`` buffer, *not* the number of experts.  An id equal to ``num_experts``
therefore passes that safety net and atomically bumps ``cumsum[num_experts]``,
which is the align stage's ``num_tokens_post_pad``, and the matching store lands
in the padding region.  Both failures are silent: the kernel returns normally.

This repro compiles the same kernel twice (unguarded vs ``eid < num_experts``),
feeds one out-of-range id, and asserts the difference.  It is the measured
justification for the explicit upper bound in ``_make_scatter_kernel.py``.

Run (from the task directory, after sourcing the toolchain environment):

    ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python \
        repro/TRAP-950-scatter-eid-upper-bound.py
"""

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401  -- registers torch.npu

THREADS = 128
NUMEL = 8  # one route per lane; lanes past NUMEL stay idle
NUM_EXPERTS = 8
BLOCK_SIZE = 4
CAPACITY = NUMEL + (NUM_EXPERTS + 1) * (BLOCK_SIZE - 1)  # 35


def _build(guard_upper):
    @tilelang.jit(out_idx=[], target="ascend")
    def _factory():
        @T.prim_func
        def main(
            flat: T.Tensor([NUMEL], "int32"),
            sorted_token_ids: T.Tensor([CAPACITY], "int32"),
            cumsum: T.Tensor([NUM_EXPERTS + 1], "int32"),
        ):
            with T.Kernel(1) as _bid, T.SimtVF(threads=THREADS):
                for j in T.Parallel(NUMEL):
                    eid = flat[j]
                    # The source kernel's guard is ``eid >= 0`` only; the trap
                    # is that the compiler's own range guard uses cumsum's
                    # extent (num_experts + 1), so ``eid == num_experts`` passes.
                    valid = eid >= 0 and eid < NUM_EXPERTS if guard_upper else eid >= 0
                    if valid:
                        slot = T.atomic_add(cumsum[eid], T.int32(1), return_prev=True)
                        sorted_token_ids[slot] = j

        return main

    return _factory()


def _pre_state(flat):
    """Align-stage pre-state: padded starts in ``cumsum``, a sentinel everywhere."""
    counts = torch.bincount(
        flat[flat < NUM_EXPERTS].to(torch.int64), minlength=NUM_EXPERTS
    )
    padded = (counts + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    cumsum = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32)
    cumsum[1:] = torch.cumsum(padded, 0).to(torch.int32)
    cumsum[NUM_EXPERTS] = int(padded.sum())
    return cumsum, counts


def _run(program, flat, cumsum):
    dflat = flat.to("npu")
    dsorted = torch.full((CAPACITY,), NUMEL, dtype=torch.int32, device="npu")
    dcumsum = cumsum.clone().to("npu")
    program(dflat, dsorted, dcumsum)
    torch.npu.synchronize()
    return dsorted.cpu(), dcumsum.cpu()


def main():
    # Two routes name expert 7 (in range), two name id 8 (== NUM_EXPERTS).
    ids = [3, NUM_EXPERTS, 7, 7, 0, NUM_EXPERTS, 2, NUM_EXPERTS]
    flat = torch.tensor(ids, dtype=torch.int32)
    cumsum_in, counts = _pre_state(flat)
    post_pad = int(cumsum_in[NUM_EXPERTS])
    out_of_range = int((flat == NUM_EXPERTS).sum())
    print(f"ids={ids} num_experts={NUM_EXPERTS} num_tokens_post_pad={post_pad}")

    # 1. Unguarded (source form): the out-of-range routes bump num_tokens_post_pad
    #    and write route indices into the padding region.
    got, got_cumsum = _run(_build(guard_upper=False), flat, cumsum_in)
    print(
        f"unguarded: cumsum[{NUM_EXPERTS}]={int(got_cumsum[NUM_EXPERTS])} "
        f"(expected {post_pad} without the bug)"
    )
    assert int(got_cumsum[NUM_EXPERTS]) == post_pad + out_of_range, (
        "expected the unguarded kernel to bump num_tokens_post_pad by "
        f"{out_of_range}, got {int(got_cumsum[NUM_EXPERTS])} vs {post_pad}"
    )
    used = torch.zeros(CAPACITY, dtype=torch.bool)
    for expert in range(NUM_EXPERTS):
        lo = int(cumsum_in[expert])
        used[lo : lo + int(counts[expert])] = True
    written_outside = int(((got != NUMEL) & ~used).sum())
    print(f"unguarded: {written_outside} sentinel slots overwritten with route ids")
    assert written_outside == out_of_range, (
        "expected one padding write per out-of-range route, got "
        f"{written_outside} vs {out_of_range}"
    )

    # 2. Guarded: nothing outside the valid ranges changes.
    got, got_cumsum = _run(_build(guard_upper=True), flat, cumsum_in)
    print(f"guarded:   cumsum[{NUM_EXPERTS}]={int(got_cumsum[NUM_EXPERTS])}")
    assert int(got_cumsum[NUM_EXPERTS]) == post_pad, (
        f"guarded kernel changed num_tokens_post_pad: "
        f"{int(got_cumsum[NUM_EXPERTS])} != {post_pad}"
    )
    assert int(((got != NUMEL) & ~used).sum()) == 0, (
        "guarded kernel wrote outside the valid ranges"
    )
    for expert in range(NUM_EXPERTS):
        lo = int(cumsum_in[expert])
        n = int(counts[expert])
        seg = torch.sort(got[lo : lo + n]).values
        want = (flat == expert).nonzero().squeeze(1)
        assert torch.equal(seg, want), f"expert {expert} multiset differs"
    print(
        "TRAP CONFIRMED: the source-form guard bumps cumsum[num_experts]; "
        "the explicit upper bound is required."
    )


if __name__ == "__main__":
    main()
