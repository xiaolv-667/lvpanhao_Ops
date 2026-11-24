"""Design-time probe (D-3): compile-only validation of the `_ssd_chunk_coupling_kernel`
NPU structure written into DESIGN.md §3.3 / §6.

No device is touched (``ASCEND_RT_VISIBLE_DEVICES=""``): every variant is only
compiled through ``tilelang.compile(..., target="ascend")`` so the frontend and
lowering accept the exact API combination of the design.  Numerical correctness
is Stage 3's job (DESIGN.md §8).

Unknowns this probe resolves (each one is a design decision that must be
compile-real before it enters DESIGN.md):

  A. strided GM->L1 region copy out of the 4-D ``[B, S, G, N]`` input with a
     fixed group index (``G > 1``) -- i.e. row stride ``G*N`` not ``N``.
  B. ``T.copy(l0c -> GM[region])`` direct fixpipe store with fp32 -> fp16/bf16
     quantisation (mask-free tiles).
  C. masked epilogue: ``T.dual_copy`` L0C->UB (fp32) + explicit
     ``T.MixedKernel(sids=2)`` Cube/Vector split where the Vector side needs the
     sub-block id ``sid`` for the causal row offset.
  D. auto-scheduled ``T.Kernel`` variant with ``with T.Vector(vector=2) as sid``
     (fallback if C does not lower).

Run (task dir, platform env sourced, card not required):

    cd examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel
    ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        history_version/design_probe_compile.py
"""

import json
import os
import re
import traceback

import tilelang
import tilelang.ascend.language as T

OUT = os.environ.get("PROBE_OUT", "probe_out")
os.makedirs(OUT, exist_ok=True)


def _tile_cfg(chunk_len, d_state):
    block_l = 128 if chunk_len % 128 == 0 else (64 if chunk_len % 64 == 0 else 32)
    block_n = 64 if d_state % 64 == 0 else 16
    return block_l, block_n


def build(body_kind, batch, num_chunks, n_groups, chunk_len, d_state, dtype,
          block_l=None, block_n=64, stages=2, reduced=False):
    """The designed program.  ``body_kind`` selects the epilogue variant."""
    Q, N, G = chunk_len, d_state, n_groups
    S = num_chunks * Q
    if block_l is None:
        block_l, block_n = _tile_cfg(Q, N)
    BL, BK = block_l, block_n
    nlt = Q // BL
    nst = Q // BL
    grid = batch * num_chunks * n_groups * nlt * nst
    k_trips = -(-N // BK)
    c_shape = (batch, S, G, N)
    cb_shape = (batch, num_chunks, G, Q, Q)

    def _decode(bid):
        bcg = bid // (nlt * nst)
        rem = bid % (nlt * nst)
        lt = rem // nst
        st = rem % nst
        bb = bcg // (num_chunks * G)
        cg = bcg % (num_chunks * G)
        return bb, cg // G, cg % G, lt * BL, st * BL

    # --- variant bodies (written out explicitly, no metaprogramming) ---
    if body_kind == "direct_store":

        @T.prim_func
        def main(
            C_mat: T.Tensor(c_shape, dtype),
            B_mat: T.Tensor(c_shape, dtype),
            cb: T.Tensor(cb_shape, dtype),
        ):
            with T.Kernel(grid) as bid:
                bb, bc, bg, l0, s0 = _decode(bid)
                x1 = T.alloc_l1((BL, BK), dtype)
                y1 = T.alloc_l1((BL, BK), dtype)
                x0 = T.alloc_l0a((BL, BK), dtype)
                y0 = T.alloc_l0b((BL, BK), dtype)
                acc = T.alloc_l0c((BL, BL), "float32")
                T.annotate_buffer_versions({x1: stages, y1: stages})
                for kk in T.Pipelined(k_trips, num_stages=stages):
                    n0 = kk * BK
                    nk = T.min(N - n0, BK)
                    T.copy(
                        C_mat[bb, bc * Q + l0 : bc * Q + l0 + BL, bg, n0 : n0 + nk],
                        x1[:, :nk],
                    )
                    T.copy(
                        B_mat[bb, bc * Q + s0 : bc * Q + s0 + BL, bg, n0 : n0 + nk],
                        y1[:, :nk],
                    )
                    T.copy(x1[:, :nk], x0[:, :nk])
                    T.copy(y1[:, :nk], y0[:, :nk])
                    T.gemm(
                        x0[:, :nk],
                        y0[:, :nk],
                        acc,
                        transpose_B=True,
                        clear_accum=kk == 0,
                    )
                T.copy(acc, cb[bb, bc, bg, l0 : l0 + BL, s0 : s0 + BL])

        return main

    if body_kind == "mixed_mask":

        @T.prim_func
        def main(
            C_mat: T.Tensor(c_shape, dtype),
            B_mat: T.Tensor(c_shape, dtype),
            cb: T.Tensor(cb_shape, dtype),
        ):
            with T.MixedKernel(grid, sids=2) as (bid, sid):
                bb, bc, bg, l0, s0 = _decode(bid)
                x1 = T.alloc_l1((BL, BK), dtype)
                y1 = T.alloc_l1((BL, BK), dtype)
                x0 = T.alloc_l0a((BL, BK), dtype)
                y0 = T.alloc_l0b((BL, BK), dtype)
                acc = T.alloc_l0c((BL, BL), "float32")
                cv = T.alloc_shared((BL // 2, BL), "float32")
                co = T.alloc_shared((BL // 2, BL), dtype)
                T.annotate_buffer_versions({x1: stages, y1: stages})
                with T.Cube():
                    for kk in T.Pipelined(k_trips, num_stages=stages):
                        n0 = kk * BK
                        nk = T.min(N - n0, BK)
                        T.copy(
                            C_mat[bb, bc * Q + l0 : bc * Q + l0 + BL, bg, n0 : n0 + nk],
                            x1[:, :nk],
                        )
                        T.copy(
                            B_mat[bb, bc * Q + s0 : bc * Q + s0 + BL, bg, n0 : n0 + nk],
                            y1[:, :nk],
                        )
                        T.copy(x1[:, :nk], x0[:, :nk])
                        T.copy(y1[:, :nk], y0[:, :nk])
                        T.gemm(
                            x0[:, :nk],
                            y0[:, :nk],
                            acc,
                            transpose_B=True,
                            clear_accum=kk == 0,
                        )
                    T.dual_copy(acc, cv)
                with T.Vector():
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(BL // 2, BL):
                            # C: sid resolves the AIV row half -> absolute row.
                            co[i, j] = T.cast(
                                T.if_then_else(
                                    s0 + j <= l0 + sid * (BL // 2) + i,
                                    cv[i, j],
                                    T.float32(0.0),
                                ),
                                dtype,
                            )
                    T.dual_copy(co, cb[bb, bc, bg, l0 : l0 + BL, s0 : s0 + BL])

        return main

    if body_kind in ("autosched_mask", "autosched_mask_skip_upper"):

        skip_upper = body_kind.endswith("skip_upper")

        @T.prim_func
        def main(
            C_mat: T.Tensor(c_shape, dtype),
            B_mat: T.Tensor(c_shape, dtype),
            cb: T.Tensor(cb_shape, dtype),
        ):
            with T.Kernel(grid) as bid:
                bb, bc, bg, l0, s0 = _decode(bid)
                x1 = T.alloc_l1((BL, BK), dtype)
                y1 = T.alloc_l1((BL, BK), dtype)
                x0 = T.alloc_l0a((BL, BK), dtype)
                y0 = T.alloc_l0b((BL, BK), dtype)
                acc = T.alloc_l0c((BL, BL), "float32")
                cv = T.alloc_shared((BL // 2, BL), "float32")
                co = T.alloc_shared((BL // 2, BL), dtype)
                T.annotate_buffer_versions({x1: stages, y1: stages})
                # fully-upper tiles keep the same code path but skip the K loop.
                trips = T.if_then_else(s0 <= l0 + BL - 1, k_trips, 0) if skip_upper else k_trips
                for kk in T.Pipelined(trips, num_stages=stages):
                    n0 = kk * BK
                    nk = T.min(N - n0, BK)
                    T.copy(
                        C_mat[bb, bc * Q + l0 : bc * Q + l0 + BL, bg, n0 : n0 + nk],
                        x1[:, :nk],
                    )
                    T.copy(
                        B_mat[bb, bc * Q + s0 : bc * Q + s0 + BL, bg, n0 : n0 + nk],
                        y1[:, :nk],
                    )
                    T.copy(x1[:, :nk], x0[:, :nk])
                    T.copy(y1[:, :nk], y0[:, :nk])
                    T.gemm(
                        x0[:, :nk],
                        y0[:, :nk],
                        acc,
                        transpose_B=True,
                        clear_accum=kk == 0,
                    )
                T.dual_copy(acc, cv)
                with T.Vector(vector=2) as sid:
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(BL // 2, BL):
                            if reduced:
                                # §1.6.1 O2 strength-reduced predicate.
                                co[i, j] = T.cast(
                                    T.if_then_else(
                                        (j - i) <= (l0 + sid * (BL // 2) - s0),
                                        cv[i, j],
                                        T.float32(0.0),
                                    ),
                                    dtype,
                                )
                            else:
                                co[i, j] = T.cast(
                                    T.if_then_else(
                                        s0 + j <= l0 + sid * (BL // 2) + i,
                                        cv[i, j],
                                        T.float32(0.0),
                                    ),
                                    dtype,
                                )
                    T.dual_copy(co, cb[bb, bc, bg, l0 : l0 + BL, s0 : s0 + BL])

        return main

    raise ValueError(body_kind)


CASES = [
    # A+B: main data path (strided 4-D GM->L1 with G=8) + direct fixpipe store.
    ("P1-direct-G8-fp16", "direct_store", dict(batch=3, num_chunks=12, n_groups=8,
                                              chunk_len=256, d_state=128,
                                              dtype="float16")),
    # A: G=1 workload (row stride == N), smoke shapes.
    ("P2-direct-smoke-fp16", "direct_store", dict(batch=1, num_chunks=2, n_groups=1,
                                                  chunk_len=64, d_state=64,
                                                  dtype="float16")),
    # C: masked epilogue, explicit mixed kernel with sid.
    ("P3-mixed-mask-fp16", "mixed_mask", dict(batch=1, num_chunks=16, n_groups=1,
                                              chunk_len=256, d_state=128,
                                              dtype="float16")),
    ("P4-mixed-mask-bf16-smoke", "mixed_mask", dict(batch=1, num_chunks=2, n_groups=1,
                                                    chunk_len=64, d_state=64,
                                                    dtype="bfloat16")),
    # D: fallback -- auto-scheduled T.Kernel with an explicit Vector frame.
    ("P5-autosched-mask-fp16", "autosched_mask", dict(batch=1, num_chunks=16, n_groups=1,
                                                      chunk_len=256, d_state=128,
                                                      dtype="float16")),
    # D on the smoke config (block 64, N=64) and on the G=8 strided workload.
    ("P6-autosched-mask-bf16-smoke", "autosched_mask", dict(batch=1, num_chunks=2, n_groups=1,
                                                            chunk_len=64, d_state=64,
                                                            dtype="bfloat16")),
    ("P7-autosched-mask-G8-fp16", "autosched_mask", dict(batch=3, num_chunks=12, n_groups=8,
                                                         chunk_len=256, d_state=128,
                                                         dtype="float16")),
    # E: same structure but fully-upper tiles skip the K loop (dynamic trip count).
    ("P8-skip-upper-fp16", "autosched_mask_skip_upper", dict(batch=1, num_chunks=16, n_groups=1,
                                                             chunk_len=256, d_state=128,
                                                             dtype="float16")),
    ("P9-skip-upper-G8-fp16", "autosched_mask_skip_upper", dict(batch=3, num_chunks=12, n_groups=8,
                                                               chunk_len=256, d_state=128,
                                                               dtype="float16")),
    # F: the exact DESIGN.md §3.3 pseudocode (skip-upper + strength-reduced
    #    predicate, §1.6.1 O2) on the two heaviest shapes.
    ("P10-final-fp16", "autosched_mask_skip_upper", dict(batch=1, num_chunks=16, n_groups=1,
                                                         chunk_len=256, d_state=128,
                                                         dtype="float16", reduced=True)),
    ("P11-final-G8-fp16", "autosched_mask_skip_upper", dict(batch=3, num_chunks=12, n_groups=8,
                                                            chunk_len=256, d_state=128,
                                                            dtype="float16", reduced=True)),
    ("P12-final-smoke-bf16", "autosched_mask_skip_upper", dict(batch=1, num_chunks=2, n_groups=1,
                                                               chunk_len=64, d_state=64,
                                                               dtype="bfloat16", reduced=True)),
]


def facts(source):
    return {
        "kernel_kind": (re.findall(r"__global__ (__cube__|__mix__\([^)]*\)|__vector__)", source) or [None])[0],
        "nd2nz": source.count("asc_copy_gm2l1_nd2nz"),
        "dn2nz": source.count("asc_copy_gm2l1_dn2nz"),
        "l0c2gm": "asc_copy_l0c2gm" in source,
        "l0c2ub": "asc_copy_l0c2ub" in source,
        "ub2gm": "asc_copy_ub2gm_align" in source,
        "qgmmad": "asc_mmad" in source,
        "quant_mode": (re.findall(r"asc_quant_mode>\((\d+)\)", source) or [None]),
        "dual_dst": (re.findall(r"asc_dual_dst_mode>\((\d+)\)", source) or [None]),
        "sub_block_id": "SubBlockIdx" in source or "sub_block" in source or "asc_get_sub_block" in source,
        "gemm": (re.findall(r"ascend_gemm\w*<[^>]*>", source) or [None])[0],
        "chars": len(source),
    }


def main():
    results = []
    for name, kind, kw in CASES:
        rec = {"probe": name, "kind": kind}
        try:
            kernel = tilelang.compile(build(kind, **kw), target="ascend", out_idx=-1)
            src = kernel.get_kernel_source()
            rec["ok"] = True
            rec.update(facts(src))
            with open(os.path.join(OUT, f"probe_src_{name}.cce"), "w") as fh:
                fh.write(src)
        except Exception as exc:  # noqa: BLE001 - the probe reports, never raises
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}".splitlines()[0][:500]
            with open(os.path.join(OUT, f"probe_fail_{name}.log"), "w") as fh:
                fh.write(traceback.format_exc())
        results.append(rec)
        print(json.dumps(rec), flush=True)
    ok = sum(1 for r in results if r["ok"])
    print(json.dumps({"summary": f"{ok}/{len(results)} compiled"}), flush=True)


if __name__ == "__main__":
    main()
