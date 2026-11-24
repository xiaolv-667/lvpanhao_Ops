"""Extracted TileLang kernel implementation (curated reference).

Op: ``ChainSpeculativeSamplingFwdOp`` -- verifies each request's ``N`` draft
tokens in order and draws the token after the accepted prefix.

Provenance: ``extract_tl_kernel.py --op-name ChainSpeculativeSamplingFwdOp
--gpu-repo-root /workspace/gpu_mirror_950pr`` (resolution pattern A) emitted
``_chain_speculative_sampling_kernel`` verbatim from
``tileops/_kernels/sampling/chain_speculative_sampling.py``.  That factory is
the whole TileLang program: the ``T.macro`` helpers ``stage`` / ``fold`` /
``locate`` and the ``@tilelang.jit`` builder
``_chain_speculative_sampling_func`` it returns are defined inside it, so one
extracted function covers the op.

Appended (S3 Part A, "Helper function extraction (mandatory)"): the names the
factory body calls that the script does not emit (it emits only
``@tilelang.jit`` builders and the module constants their bodies read), copied
verbatim from their GPU modules so this file is self-contained:

  - ``csrc_include`` / ``csrc_path`` / ``_csrc_digest`` -- from
    ``tileops/_csrc.py``: the nvcc flags that pre-include
    ``streaming_load.h``, which defines the 16-byte ``load16`` /
    ``load16_evict_first`` device helpers ``load_vector`` calls.  A CUDA flag;
    the NPU component drops it with the decorator rewrite (K1).
  - ``mix`` -- from ``tileops/_kernels/sampling/philox.py``: the ten
    Philox4x32-10 rounds, with the round multipliers and Weyl keys it reads.
  - ``load_vector`` -- from ``tileops/_kernels/sampling/row_tiles.py``: the
    ``vec``-element (one 16-byte vector, or one element when the row is not a
    whole number of them) row read.

``UNIFORM_BITS`` (from ``philox.py``), ``WARP_LANES`` and
``WARP_SHUFFLE_STAGES`` (from ``tileops/_utils/utils.py``) were emitted by the
script as the module constants the body references.

These are the GPU (CUDA-target) implementations.  The NPU kernel component
reimplements them for the selected NPU target: K1 (decorator / target), K2
(grid + sync -- ``T.sync_grid`` needs a whole-grid-resident launch), K3
(``threads``; here a factory argument, not the returned callable's argument)
and K4 (padding / masking) apply.  The ``T.call_extern`` targets
(``tileops::load16*``, ``__umulhi``) have no Ascend equivalent and are part of
what the rewrite replaces.
"""

# --- external imports (preserved) ---
import hashlib
from pathlib import Path

import tilelang.language as T
import tilelang

# --- module-level constants referenced by the kernels ---
UNIFORM_BITS = 24
WARP_LANES: int = 32
WARP_SHUFFLE_STAGES: int = WARP_LANES.bit_length() - 1

# --- helper functions called by the kernel (copied verbatim from the GPU repo) ---
# ``_CSRC`` / ``csrc_path`` / ``_csrc_digest`` / ``csrc_include`` -- tileops/_csrc.py
_CSRC = Path(__file__).resolve().parent / "csrc"


def csrc_path(name: str) -> str:
    """Return the absolute path of ``name`` under ``tileops/csrc``."""
    return str(_CSRC / name)


def _csrc_digest() -> str:
    """Hash the name and content of every file under ``tileops/csrc``."""
    digest = hashlib.sha256()
    for path in sorted(p for p in _CSRC.rglob("*") if p.is_file()):
        digest.update(path.relative_to(_CSRC).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def csrc_include(*names: str) -> list[str]:
    """Return the nvcc flags that pre-include ``names`` from ``tileops/csrc``.

    TileLang keys its kernel and binary caches on the flag strings, not on the
    files they name, so an edited header would otherwise reuse a binary built
    from the old one. The trailing ``-D`` define carries a hash of every csrc
    file; no source reads it, but it changes the flags whenever a header does.
    An edit takes effect in the next process: kernel builders keep the flags
    they were built with for the life of one.

    Args:
        names: Header file names under ``tileops/csrc``, included in order.

    Returns:
        ``["-include", <path>, ..., "-DTILEOPS_CSRC_SHA256=<hash>"]``.
    """
    flags = [flag for name in names for flag in ("-include", csrc_path(name))]
    return [*flags, f"-DTILEOPS_CSRC_SHA256={_csrc_digest()}"]


# Philox4x32-10 round multipliers and Weyl key increments (Salmon et al., SC'11)
# -- tileops/_kernels/sampling/philox.py (``UNIFORM_BITS`` above came from it).
_EVEN_MUL, _ODD_MUL = 0xD2511F53, 0xCD9E8D57
_EVEN_WEYL, _ODD_WEYL = 0x9E3779B9, 0xBB67AE85
_ROUNDS = 10


@T.macro
def mix(counter, key, bump):
    """Run the ten rounds in place over *counter*, leaving the uniform's word in ``counter[0]``.

    *counter* and *key* are the caller's, already set; *bump* is two words of scratch.
    """
    for step in T.serial(_ROUNDS):
        if step > 0:
            key[0] = key[0] + T.uint32(_EVEN_WEYL)
            key[1] = key[1] + T.uint32(_ODD_WEYL)
        bump[0] = (
            T.call_extern("uint32", "__umulhi", counter[2], T.uint32(_ODD_MUL))
            ^ counter[1]
            ^ key[0]
        )
        bump[1] = (
            T.call_extern("uint32", "__umulhi", counter[0], T.uint32(_EVEN_MUL))
            ^ counter[3]
            ^ key[1]
        )
        counter[1] = counter[2] * T.uint32(_ODD_MUL)
        counter[3] = counter[0] * T.uint32(_EVEN_MUL)
        counter[0] = bump[0]
        counter[2] = bump[1]


# -- tileops/_kernels/sampling/row_tiles.py
@T.macro
def load_vector(dst, slot, src, at, vec: int, evict_first: bool):
    """Read one ``vec``-element vector of *src* at *at* into slot *slot* of *dst*."""
    if vec == 1:
        dst[slot, 0] = src[at]
    elif evict_first:
        T.call_extern(
            "handle",
            "tileops::load16_evict_first",
            T.address_of(dst[slot, 0]),
            T.address_of(src[at]),
        )
    else:
        T.call_extern(
            "handle",
            "tileops::load16",
            T.address_of(dst[slot, 0]),
            T.address_of(src[at]),
        )


# --- TileLang kernel functions (@tilelang.jit) ---
def _chain_speculative_sampling_kernel(
    batch: int, num_draft: int, vocab: int, vec: int, threads: int, parts: int, pace: int
):
    """Build the verification of ``batch`` chains of ``num_draft`` drafts over ``vocab`` tokens.

    The chain test reads ``2 * num_draft`` scalars, so the whole launch moves the one
    residual row ``max(0, target - draft)`` the accepted prefix stops at. A thread folds the
    ``vec`` weights of one 16-byte vector and a warp folds its 32 threads, leaving one entry
    of ``seg`` per (slot, warp), a slot being one vector per thread. Those entries are the
    leaves the draw descends: the row's blocks, then the block's ``seg`` entries, then the
    winning warp's lanes, then that lane's weights. Only the last of those re-reads the row,
    and it re-reads 32 vectors of it.

    The two folds a warp does are float32 over groups that ``vocab`` and ``threads`` fix
    between them; every sum above them is float64. A launch that splits the row differently
    therefore regroups float64 sums only.
    """
    full = vocab // vec
    row_tiles = -(-full // threads)
    chunk = -(-row_tiles // parts)
    warps = threads // WARP_LANES
    grid = batch * parts
    # Entries of ``seg``, and how many of them one lane of a descent walks.
    leaves = chunk * warps
    leaf_span = -(-leaves // WARP_LANES)
    rounds = -(-chunk // pace)
    draw_rounds = -(-(num_draft + 1) // threads)
    part_span = -(-parts // WARP_LANES)

    @tilelang.jit(compile_flags=csrc_include("streaming_load.h"))
    def _chain_speculative_sampling_func():
        @T.macro
        def stage(held, other, target_probs, draft_probs, slot, at_t, at_d, inside, drafted):
            """Read the ``vec`` target and draft weights at element offset ``at_t`` of the
            rows into ``held[slot]`` and ``other[slot]``, zero where the vector is past the row.

            ``drafted`` false reads no draft row, which is the bonus position, where the whole
            chain was accepted.
            """
            for c in T.unroll(vec):
                held[slot, c] = T.float32(0)
                other[slot, c] = T.float32(0)
            if inside:
                load_vector(held, slot, target_probs, at_t, vec, True)
                if drafted:
                    load_vector(other, slot, draft_probs, at_d, vec, True)

        @T.macro
        def fold(dst, held, other, slot):
            """Leave in ``dst[0]`` the float32 sum of the residual ``max(0, target - draft)``
            over the ``vec`` weights ``stage`` read into slot ``slot``."""
            for c in T.unroll(vec):
                held[slot, c] = T.max(held[slot, c] - other[slot, c], T.float32(0))
            for c in T.serial(1, vec):
                held[slot, 0] = held[slot, 0] + held[slot, c]
            dst[0] = held[slot, 0]

        @T.macro
        def locate(src, n, span, frac, base, lane, pick, prefix, total, left, acc, idx):
            """Leave in ``pick[0]`` the entry of ``src[0:n]`` the draw lands on and in
            ``left[0]`` how much of the draw point is left inside that entry.

            Lane ``l`` of warp 0 owns ``src[l * span : (l + 1) * span)`` and sums it into
            ``lane[l]``; lane 0 turns those sums into the exclusive prefixes ``prefix``, which
            the descent reads by lane and so stays one buffer; every lane then walks its own
            entries against a draw point of
            ``frac * total + base``. The entry taken is the first whose inclusive prefix
            passes that point and whose own value is positive, and the last positive entry
            when none does. A zero-weight entry is therefore never taken, and a point that
            rounding leaves past the total still lands on weight the row carries.
            """
            tx = T.get_thread_binding()
            if tx < WARP_LANES:
                acc[0] = T.cast(0, "float64")
                for i in T.serial(span):
                    if tx * span + i < n:
                        acc[0] = acc[0] + src[tx * span + i]
                lane[tx] = acc[0]
            T.sync_threads()
            if tx == 0:
                acc[0] = T.cast(0, "float64")
                for l in T.serial(WARP_LANES):
                    prefix[l] = acc[0]
                    acc[0] = acc[0] + lane[l]
                total[0] = acc[0]
            T.sync_threads()
            if tx < WARP_LANES:
                acc[0] = prefix[tx]
                # The prefix that placed this level was summed in another order than these
                # entries, so the remainder can come out negative, and every entry would then
                # pass the point, a zero-weight one first.
                acc[1] = T.max(frac * total[0] + base, T.cast(0, "float64"))
                idx[0] = n
                idx[1] = -1
                for i in T.serial(span):
                    if tx * span + i < n:
                        if src[tx * span + i] > T.cast(0, "float64"):
                            idx[1] = tx * span + i
                            if (idx[0] == n) & (acc[0] + src[tx * span + i] > acc[1]):
                                idx[0] = tx * span + i
                        acc[0] = acc[0] + src[tx * span + i]
                for stage in T.unroll(WARP_SHUFFLE_STAGES):
                    reach = T.int32(WARP_LANES // 2) >> stage
                    idx[0] = T.min(idx[0], T.shfl_xor(idx[0], reach, width=WARP_LANES))
                    idx[1] = T.max(idx[1], T.shfl_xor(idx[1], reach, width=WARP_LANES))
                if tx == 0:
                    pick[0] = T.if_then_else(idx[0] < n, idx[0], T.max(idx[1], 0))
            T.sync_threads()
            if tx == 0:
                acc[1] = T.max(frac * total[0] + base, T.cast(0, "float64"))
                acc[0] = prefix[pick[0] // span]
                for i in T.serial(span):
                    if (pick[0] // span) * span + i < pick[0]:
                        acc[0] = acc[0] + src[(pick[0] // span) * span + i]
                left[0] = acc[1] - acc[0]
            T.sync_threads()

        @T.prim_func
        def _chain_speculative_sampling_main(
            draft_probs: T.Tensor((batch * num_draft * vocab,), "float32"),
            draft_token_ids: T.Tensor((batch * num_draft,), "int32"),
            target_probs: T.Tensor((batch * (num_draft + 1) * vocab,), "float32"),
            seed: T.Tensor((1,), "int64"),
            offset: T.Tensor((1,), "int64"),
            partial: T.Tensor((grid,), "float64"),
            output_token_ids: T.Tensor((batch * (num_draft + 1),), "int32"),
            num_accepted: T.Tensor((batch,), "int32"),
        ):
            with T.Kernel(grid, threads=threads) as bx:
                tx = T.get_thread_binding()
                seg = T.alloc_shared((leaves,), "float64")
                chunks = T.alloc_shared((parts,), "float64")
                tail = T.alloc_shared((WARP_LANES,), "float64")
                lane = T.alloc_shared((WARP_LANES,), "float64")
                # The exclusive prefix of each lane's entries, read by a computed lane, so
                # it stays one buffer.
                prefix = T.alloc_shared((WARP_LANES,), "float64")
                total = T.alloc_shared((1,), "float64")
                left = T.alloc_shared((1,), "float64")
                pick = T.alloc_shared((1,), "int32")
                # Per draft, the chain position it rejects at, or ``num_draft`` when accepted.
                verdict = T.alloc_shared((max(num_draft, 1),), "int32")
                token_uniform = T.alloc_shared((1,), "float32")
                owner_part = T.alloc_shared((1,), "int32")
                warp_at = T.alloc_shared((1,), "int32")
                held = T.alloc_local((pace, vec), "float32")
                other = T.alloc_local((pace, vec), "float32")
                warp = T.alloc_local((1,), "float32")
                acc = T.alloc_local((2,), "float64")
                idx = T.alloc_local((2,), "int32")
                counter = T.alloc_local((4,), "uint32")
                bump = T.alloc_local((2,), "uint32")
                key = T.alloc_local((2,), "uint32")
                draw_index = T.alloc_local((1,), "int32")
                element = T.alloc_local((1,), "int32")
                token = T.alloc_local((1,), "int32")
                remainder = T.alloc_local((1,), "float64")
                uniform = T.alloc_local((1,), "float64")
                accepted = T.alloc_local((1,), "int32")
                target_at = T.alloc_local((1,), "int32")
                draft_at = T.alloc_local((1,), "int32")

                line = bx // parts
                head = bx % parts * chunk

                # Philox4x32-10 keyed by the seed and counted by the draw and the row
                # alone, so a uniform is the same value however the launch splits the row.
                for d in T.serial(draw_rounds):
                    draw_index[0] = d * threads + tx
                    if draw_index[0] < num_draft + 1:
                        key[0] = T.cast(seed[0] & T.int64(0xFFFFFFFF), "uint32")
                        key[1] = T.cast((seed[0] >> T.int64(32)) & T.int64(0xFFFFFFFF), "uint32")
                        counter[0] = T.cast(draw_index[0], "uint32")
                        counter[1] = T.cast(line, "uint32")
                        counter[2] = T.cast(offset[0] & T.int64(0xFFFFFFFF), "uint32")
                        counter[3] = T.cast(
                            (offset[0] >> T.int64(32)) & T.int64(0xFFFFFFFF), "uint32"
                        )
                        mix(counter, key, bump)
                        warp[0] = T.cast(
                            counter[0] >> T.uint32(32 - UNIFORM_BITS), "float32"
                        ) * T.float32(2.0**-UNIFORM_BITS)
                        if draw_index[0] == num_draft:
                            token_uniform[0] = warp[0]
                        else:
                            # A draft is accepted while ``u * draft < target`` at its id.
                            idx[0] = draft_token_ids[line * num_draft + draw_index[0]]
                            verdict[draw_index[0]] = T.if_then_else(
                                warp[0]
                                * draft_probs[(line * num_draft + draw_index[0]) * vocab + idx[0]]
                                < target_probs[
                                    (line * (num_draft + 1) + draw_index[0]) * vocab + idx[0]
                                ],
                                num_draft,
                                draw_index[0],
                            )
                T.sync_threads()

                accepted[0] = num_draft
                for j in T.serial(num_draft):
                    accepted[0] = T.min(accepted[0], verdict[j])
                # The bonus position has no draft row, so its residual is the target row.
                target_at[0] = (line * (num_draft + 1) + accepted[0]) * vocab
                draft_at[0] = (line * num_draft + T.min(accepted[0], num_draft - 1)) * vocab
                uniform[0] = T.cast(token_uniform[0], "float64")

                for r in T.serial(rounds):
                    for j in T.unroll(pace):
                        element[0] = ((head + r * pace + j) * threads + tx) * vec
                        stage(
                            held,
                            other,
                            target_probs,
                            draft_probs,
                            j,
                            target_at[0] + element[0],
                            draft_at[0] + element[0],
                            (r * pace + j < chunk) & (element[0] < full * vec),
                            accepted[0] < num_draft,
                        )
                    for j in T.unroll(pace):
                        fold(warp, held, other, j)
                        for step in T.unroll(WARP_SHUFFLE_STAGES):
                            reach = T.int32(WARP_LANES // 2) >> step
                            warp[0] = warp[0] + T.shfl_xor(warp[0], reach, width=WARP_LANES)
                        if (tx % WARP_LANES == 0) & (r * pace + j < chunk):
                            seg[(r * pace + j) * warps + tx // WARP_LANES] = T.cast(
                                warp[0], "float64"
                            )
                T.sync_threads()

                if parts == 1:
                    if tx == 0:
                        owner_part[0] = 0
                    remainder[0] = T.cast(0, "float64")
                else:
                    # The chunk total, summed exactly as ``locate`` sums the same entries,
                    # so the prefix a block takes from the barrier and the prefix it then
                    # descends agree entry for entry.
                    if tx < WARP_LANES:
                        acc[0] = T.cast(0, "float64")
                        for i in T.serial(leaf_span):
                            if tx * leaf_span + i < leaves:
                                acc[0] = acc[0] + seg[tx * leaf_span + i]
                        lane[tx] = acc[0]
                    T.sync_threads()
                    if tx == 0:
                        acc[0] = T.cast(0, "float64")
                        for l in T.serial(WARP_LANES):
                            acc[0] = acc[0] + lane[l]
                        partial[bx] = acc[0]
                    T.sync_grid()
                    for i in T.serial(-(-parts // threads)):
                        if i * threads + tx < parts:
                            chunks[i * threads + tx] = partial[line * parts + i * threads + tx]
                    T.sync_threads()
                    locate(
                        chunks,
                        parts,
                        part_span,
                        uniform[0],
                        T.cast(0, "float64"),
                        lane,
                        pick,
                        prefix,
                        total,
                        left,
                        acc,
                        idx,
                    )
                    if tx == 0:
                        owner_part[0] = pick[0]
                    remainder[0] = left[0]
                    uniform[0] = T.cast(0, "float64")
                T.sync_threads()

                if owner_part[0] == bx % parts:
                    locate(
                        seg,
                        leaves,
                        leaf_span,
                        uniform[0],
                        remainder[0],
                        lane,
                        pick,
                        prefix,
                        total,
                        left,
                        acc,
                        idx,
                    )
                    remainder[0] = left[0]
                    if tx == 0:
                        warp_at[0] = (
                            (head + pick[0] // warps) * threads + pick[0] % warps * WARP_LANES
                        ) * vec
                    T.sync_threads()
                    if tx < WARP_LANES:
                        element[0] = warp_at[0] + tx * vec
                        stage(
                            held,
                            other,
                            target_probs,
                            draft_probs,
                            0,
                            target_at[0] + element[0],
                            draft_at[0] + element[0],
                            element[0] < full * vec,
                            accepted[0] < num_draft,
                        )
                        fold(warp, held, other, 0)
                        tail[tx] = T.cast(warp[0], "float64")
                    T.sync_threads()
                    locate(
                        tail,
                        WARP_LANES,
                        1,
                        T.cast(0, "float64"),
                        remainder[0],
                        lane,
                        pick,
                        prefix,
                        total,
                        left,
                        acc,
                        idx,
                    )
                    if tx == 0:
                        # The chosen lane's weights, walked in index order: the token is the
                        # first whose weight passes what is left of the draw point.
                        element[0] = warp_at[0] + pick[0] * vec
                        acc[0] = left[0]
                        idx[0] = -1
                        idx[1] = 0
                        for c in T.serial(vec):
                            if element[0] + c < full * vec:
                                acc[1] = T.cast(
                                    T.max(
                                        target_probs[target_at[0] + element[0] + c]
                                        - T.if_then_else(
                                            accepted[0] < num_draft,
                                            draft_probs[draft_at[0] + element[0] + c],
                                            T.float32(0),
                                        ),
                                        T.float32(0),
                                    ),
                                    "float64",
                                )
                                if acc[1] > T.cast(0, "float64"):
                                    idx[1] = c
                                    if (idx[0] < 0) & (acc[1] > acc[0]):
                                        idx[0] = c
                                acc[0] = acc[0] - acc[1]
                        token[0] = element[0] + T.if_then_else(idx[0] >= 0, idx[0], idx[1])
                        for j in T.serial(num_draft + 1):
                            output_token_ids[line * (num_draft + 1) + j] = T.if_then_else(
                                j < accepted[0],
                                draft_token_ids[line * num_draft + T.min(j, num_draft - 1)],
                                T.if_then_else(j == accepted[0], token[0], -1),
                            )
                        num_accepted[line] = accepted[0]

        return _chain_speculative_sampling_main

    return _chain_speculative_sampling_func

__all__ = ["_chain_speculative_sampling_kernel"]
