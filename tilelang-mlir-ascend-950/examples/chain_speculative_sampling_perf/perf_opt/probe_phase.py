"""Stage 4 probe: where does the baseline kernel's time go?

Times, with wall clock on device (diagnostic only -- the recorded metric is the
platform's msprof Task Duration), a set of structural variants of the Stage 3
kernel for one shape:

  copy      : the residual row read as a plain vectorized copy (bandwidth ref)
  chain     : stage A only (Philox + acceptance test)
  fold      : stage B only (residual fold over ntile tiles, no locate)
  fold_nost : stage B only, without the r_ub store/read round trip
  full      : the Stage 3 kernel verbatim

Usage (task dir):
  python perf_opt/probe_phase.py --batch 16 --num-draft 4 --vocab 128256
"""

import argparse
import functools
import importlib.util
import os
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401  -- registers torch.npu

F32 = "float32"
M0, M1 = 0xD2511F53, 0xCD9E8D57
W0, W1 = 0x9E3779B9, 0xBB67AE85
ROUNDS = 10
UBITS = 24
USCALE = 2.0**-UBITS
THREADS = 128
TILE_CAP = 8192


def _mulhilo(a, m):
    ah = a >> T.uint32(16)
    al = a & T.uint32(0xFFFF)
    mh, ml = m >> 16, m & 0xFFFF
    p0 = al * T.uint32(ml)
    p1 = ah * T.uint32(ml)
    p2 = al * T.uint32(mh)
    p3 = ah * T.uint32(mh)
    t = (p1 & T.uint32(0xFFFF)) + (p2 & T.uint32(0xFFFF)) + (p0 >> T.uint32(16))
    lo = (p0 & T.uint32(0xFFFF)) | ((t & T.uint32(0xFFFF)) << T.uint32(16))
    hi = p3 + (p1 >> T.uint32(16)) + (p2 >> T.uint32(16)) + (t >> T.uint32(16))
    return hi, lo


def _plan(batch, vocab, blocks_cap=56, tile_cap=TILE_CAP):
    tile = min(tile_cap, -(-vocab // THREADS) * THREADS)
    return tile, THREADS, min(batch, blocks_cap)


def _build(batch, num_draft, vocab, tile, threads, blocks, phase, tile_mult=1):
    """phase in {chain, fold, fold_nost, full, copy}."""
    ntile = -(-vocab // tile)
    per_core = -(-batch // blocks)
    per_lane = tile // threads
    ndraw = num_draft + 1
    rows = batch * ndraw

    @T.prim_func
    def main(
        draft_probs: T.Tensor((batch * num_draft * vocab,), F32),
        draft_token_ids: T.Tensor((batch * num_draft,), "int32"),
        target_probs: T.Tensor((rows * vocab,), F32),
        seed_w: T.Tensor((2,), "int32"),
        offset_w: T.Tensor((2,), "int32"),
        output_token_ids: T.Tensor((rows,), "int32"),
        num_accepted: T.Tensor((batch,), "int32"),
    ) -> None:
        with T.Kernel(blocks) as bx:
            r_ub = T.alloc_shared((tile * tile_mult,), F32)
            lane_sum = T.alloc_shared((threads,), F32)
            tile_sum = T.alloc_shared((ntile,), F32)
            ids = T.alloc_shared((num_draft,), "int32")
            n_s = T.alloc_shared((1,), "int32")
            at_t = T.alloc_shared((1,), "int32")
            at_d = T.alloc_shared((1,), "int32")
            u_tok = T.alloc_shared((1,), F32)
            tok = T.alloc_shared((1,), "int32")
            cw = T.alloc_shared((4,), "uint32")
            kw = T.alloc_shared((2,), "uint32")
            bw = T.alloc_shared((2,), "uint32")
            point = T.alloc_shared((1,), F32)
            cum = T.alloc_shared((1,), F32)
            left = T.alloc_shared((1,), F32)
            rem = T.alloc_shared((1,), F32)
            pbase = T.alloc_shared((1,), "int32")
            pick = T.alloc_shared((1,), "int32")
            lpick = T.alloc_shared((1,), "int32")
            hit = T.alloc_shared((1,), "int32")
            lastt = T.alloc_shared((1,), "int32")

            for rr in T.serial(per_core):
                line = bx + rr * blocks
                if line < batch:
                    with T.SimtVF(threads=threads):
                        if phase in ("fold", "fold_nost", "full"):
                            can = T.alloc_reducer((1,), F32, op="sum")
                            s = T.alloc_fragment((1,), F32)
                        if phase in ("fold", "fold_nost", "full"):
                            mxr = T.alloc_reducer((1,), F32, op="max")
                            m = T.alloc_fragment((1,), F32)
                        for j in T.Parallel(num_draft):
                            ids[j] = draft_token_ids[line * num_draft + j]
                        n_s[0] = num_draft
                        if phase in ("chain", "full"):
                            for j in T.serial(ndraw):
                                kw[0] = T.reinterpret(seed_w[0], "uint32")
                                kw[1] = T.reinterpret(seed_w[1], "uint32")
                                cw[0] = T.cast(j, "uint32")
                                cw[1] = T.cast(line, "uint32")
                                cw[2] = T.reinterpret(offset_w[0], "uint32")
                                cw[3] = T.reinterpret(offset_w[1], "uint32")
                                for step in T.serial(ROUNDS):
                                    if step > 0:
                                        kw[0] = kw[0] + T.uint32(W0)
                                        kw[1] = kw[1] + T.uint32(W1)
                                    hi0, lo0 = _mulhilo(cw[0], M0)
                                    hi1, lo1 = _mulhilo(cw[2], M1)
                                    bw[0] = hi1 ^ cw[1] ^ kw[0]
                                    bw[1] = hi0 ^ cw[3] ^ kw[1]
                                    cw[1] = lo1
                                    cw[3] = lo0
                                    cw[0] = bw[0]
                                    cw[2] = bw[1]
                                if j == num_draft:
                                    u_tok[0] = T.cast(
                                        cw[0] >> T.uint32(32 - UBITS), F32
                                    ) * T.float32(USCALE)
                                else:
                                    if (
                                        T.cast(cw[0] >> T.uint32(32 - UBITS), F32)
                                        * T.float32(USCALE)
                                        * draft_probs[(line * num_draft + j) * vocab + ids[j]]
                                        >= target_probs[(line * ndraw + j) * vocab + ids[j]]
                                    ):
                                        n_s[0] = T.min(n_s[0], j)
                        at_t[0] = (line * ndraw + n_s[0]) * vocab
                        at_d[0] = (line * num_draft + T.min(n_s[0], num_draft - 1)) * vocab

                        if phase == "copy":
                            for t in T.serial(ntile):
                                for i in T.Parallel(tile):
                                    r_ub[i] = target_probs[at_t[0] + t * tile + i]
                        elif phase in ("fold", "full"):
                            for t in T.serial(ntile):
                                T.reducer_init(can)
                                for i in T.Parallel(tile):
                                    i0 = t * tile + i
                                    r_ub[i] = T.if_then_else(
                                        i0 < vocab,
                                        T.max(
                                            target_probs[at_t[0] + i0]
                                            - T.if_then_else(
                                                n_s[0] < num_draft,
                                                draft_probs[at_d[0] + i0],
                                                T.float32(0),
                                            ),
                                            T.float32(0),
                                        ),
                                        T.float32(0),
                                    )
                                    T.reducer_update(can[0], r_ub[i])
                                T.finalize_reducer(can, s)
                                if T.get_thread_binding() == 0:
                                    tile_sum[t] = s[0]
                        elif phase == "fold_nost":
                            for t in T.serial(ntile):
                                T.reducer_init(can)
                                for i in T.Parallel(tile):
                                    i0 = t * tile + i
                                    T.reducer_update(
                                        can[0],
                                        T.if_then_else(
                                            i0 < vocab,
                                            T.max(
                                                target_probs[at_t[0] + i0]
                                                - T.if_then_else(
                                                    n_s[0] < num_draft,
                                                    draft_probs[at_d[0] + i0],
                                                    T.float32(0),
                                                ),
                                                T.float32(0),
                                            ),
                                            T.float32(0),
                                        ),
                                    )
                                T.finalize_reducer(can, s)
                                if T.get_thread_binding() == 0:
                                    tile_sum[t] = s[0]

                        if phase in ("full", "fold", "fold_nost"):
                            point[0] = T.float32(0)
                            for t in T.serial(ntile):
                                point[0] = point[0] + tile_sum[t]
                            point[0] = u_tok[0] * point[0]
                            lastt[0] = 0
                            for t in T.serial(ntile):
                                if tile_sum[t] > T.float32(0):
                                    lastt[0] = t
                            pick[0] = ntile - 1
                            left[0] = point[0]
                            cum[0] = T.float32(0)
                            hit[0] = 0
                            for t in T.serial(ntile):
                                if hit[0] == 0:
                                    if cum[0] + tile_sum[t] > point[0]:
                                        pick[0] = t
                                        hit[0] = 1
                                        left[0] = point[0] - cum[0]
                                    cum[0] = cum[0] + tile_sum[t]
                            if hit[0] == 0:
                                pick[0] = lastt[0]
                                left[0] = cum[0] + T.float32(1.0)

                            for t in T.serial(ntile):
                                if t == pick[0]:
                                    for i in T.Parallel(tile):
                                        i0 = t * tile + i
                                        r_ub[i] = T.if_then_else(
                                            i0 < vocab,
                                            T.max(
                                                target_probs[at_t[0] + i0]
                                                - T.if_then_else(
                                                    n_s[0] < num_draft,
                                                    draft_probs[at_d[0] + i0],
                                                    T.float32(0),
                                                ),
                                                T.float32(0),
                                            ),
                                            T.float32(0),
                                        )
                            T.reducer_init(mxr)
                            for i in T.Parallel(tile):
                                T.reducer_update(
                                    mxr[0],
                                    T.if_then_else(
                                        r_ub[i] > T.float32(0),
                                        T.cast(pick[0] * tile + i, F32),
                                        T.float32(-1),
                                    ),
                                )
                            T.finalize_reducer(mxr, m)
                            for l in T.Parallel(threads):
                                part = T.alloc_local((1,), F32)
                                part[0] = T.float32(0)
                                for k in T.serial(per_lane):
                                    part[0] = part[0] + r_ub[l * per_lane + k]
                                lane_sum[l] = part[0]
                            lpick[0] = threads - 1
                            cum[0] = T.float32(0)
                            rem[0] = left[0]
                            hit[0] = 0
                            for l in T.serial(threads):
                                if hit[0] == 0:
                                    if cum[0] + lane_sum[l] > left[0]:
                                        lpick[0] = l
                                        rem[0] = left[0] - cum[0]
                                        hit[0] = 1
                                    cum[0] = cum[0] + lane_sum[l]
                            pbase[0] = lpick[0] * per_lane
                            tok[0] = -1
                            for k in T.serial(per_lane):
                                if (tok[0] < 0) & (r_ub[pbase[0] + k] > rem[0]):
                                    tok[0] = pick[0] * tile + pbase[0] + k
                                rem[0] = rem[0] - r_ub[pbase[0] + k]
                            if tok[0] < 0:
                                tok[0] = T.cast(m[0], "int32")
                                if tok[0] < 0:
                                    tok[0] = vocab - 1
                            if tok[0] >= vocab:
                                tok[0] = vocab - 1

                            if T.get_thread_binding() == 0:
                                for j in T.serial(ndraw):
                                    output_token_ids[line * ndraw + j] = T.if_then_else(
                                        j < n_s[0],
                                        draft_token_ids[line * num_draft + T.min(j, num_draft - 1)],
                                        T.if_then_else(j == n_s[0], tok[0], T.int32(-1)),
                                    )
                                num_accepted[line] = n_s[0]

    return main


@functools.lru_cache(maxsize=32)
def build_variant(batch, num_draft, vocab, phase, tile_cap=TILE_CAP, blocks_cap=56):
    tile, threads, blocks = _plan(batch, vocab, blocks_cap, tile_cap)
    return tilelang.compile(
        _build(batch, num_draft, vocab, tile, threads, blocks, phase),
        target="ascend",
        out_idx=[],
    )


def load_baseline():
    path = (
        "/workspace/tilelang-mlir-ascend-950/examples/TileOPs/tileops/kernels/sampling/"
        "chain_speculative_sampling/chain_speculative_sampling_kernel/"
        "_chain_speculative_sampling_kernel.py"
    )
    spec = importlib.util.spec_from_file_location("css_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_inputs(batch, num_draft, vocab, gen_seed=24):
    gen = torch.Generator().manual_seed(gen_seed)
    draft = torch.randn(batch, num_draft, vocab, generator=gen, dtype=torch.float32).softmax(-1)
    target = torch.randn(batch, num_draft + 1, vocab, generator=gen, dtype=torch.float32).softmax(-1)
    ids = (
        torch.multinomial(draft.reshape(-1, vocab), 1, generator=gen)
        .view(batch, num_draft)
        .to(torch.int32)
    )
    seed = torch.tensor([1234], dtype=torch.int64)
    offset = torch.tensor([7], dtype=torch.int64)
    return draft.contiguous(), ids.contiguous(), target.contiguous(), seed, offset


def timeit(fn, args, warmup=5, iters=20):
    for _ in range(warmup):
        fn(*args)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--num-draft", type=int, default=4)
    p.add_argument("--vocab", type=int, default=128256)
    p.add_argument("--phases", default="copy,chain,fold,fold_nost,full")
    p.add_argument("--tile-cap", type=int, default=TILE_CAP)
    p.add_argument("--blocks-cap", type=int, default=56)
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args()

    batch, n, vocab = args.batch, args.num_draft, args.vocab
    draft, ids, target, seed, offset = make_inputs(batch, n, vocab)
    dev = "npu"
    dd, di, dt = draft.to(dev), ids.to(dev), target.to(dev)
    ds, do = seed.to(dev), offset.to(dev)
    out_t = torch.empty(batch, n + 1, dtype=torch.int32, device=dev)
    out_a = torch.empty(batch, dtype=torch.int32, device=dev)
    call_args = (
        dd.view(-1), di.view(-1), dt.view(-1),
        ds.view(torch.int32).reshape(-1), do.view(torch.int32).reshape(-1),
        out_t.view(-1), out_a.view(-1),
    )
    print(f"shape B={batch} N={n} V={vocab} rows*V={batch*vocab/1e6:.2f}M elems", flush=True)

    for phase in args.phases.split(","):
        fn = build_variant(batch, n, vocab, phase, args.tile_cap, args.blocks_cap)
        us = timeit(fn, call_args)
        print(f"  {phase:10s} tile_cap={args.tile_cap} blocks_cap={args.blocks_cap}: {us:9.1f} us", flush=True)
        del fn

    if args.baseline:
        mod = load_baseline()
        fn = mod._chain_speculative_sampling_kernel(batch, n, vocab)
        us = timeit(fn, call_args)
        print(f"  {'baseline':10s}: {us:9.1f} us", flush=True)


if __name__ == "__main__":
    main()
