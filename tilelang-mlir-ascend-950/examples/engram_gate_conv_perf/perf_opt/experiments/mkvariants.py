"""Generate round-1 experiment branches from the frozen Stage-3 kernel source.

Each variant applies exactly one main optimisation point to the same baseline
(the read-only Stage-3 file).  Patching textually keeps the diff of every
branch auditable in one place.

    python mkvariants.py            # writes ../engram_gate_conv_round1_c*.py
"""

import os

BASE = (
    "/workspace/tilelang-mlir-ascend-950/examples/TileOPs/tileops/kernels/"
    "sequence_modeling/engram_gate_conv/engram_gate_conv_kernel/_engram_gate_conv_fwd_kernel.py"
)
OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sub(text, old, new, count=1):
    if text.count(old) != count:
        raise SystemExit(f"patch anchor found {text.count(old)}x (want {count}):\n{old[:200]}")
    return text.replace(old, new)


def header(name, note):
    return (
        f'"""Round-1 Stage-4 experiment branch `{name}` of _engram_gate_conv_fwd_kernel.\n'
        f"\n{note}\n"
        'Derived from the frozen Stage-3 source (single main optimisation point per branch).\n"""\n'
    )


def main():
    with open(BASE, encoding="utf-8") as f:
        base = f.read()

    variants = {}

    # ------------------------------------------------------------------ c1
    # Overlap the per-row GM->UB row loads with the row compute:
    # T.Pipelined(num_stages=2) + two versions for the three row buffers.
    t = base
    t = sub(
        t,
        "            for i in T.serial(rc + 3):\n",
        "            for i in T.Pipelined(rc + 3, num_stages=2):\n",
    )
    t = sub(
        t,
        "                    h_ub: 1,\n                    k_ub: 1,\n                    v_ub: 1,\n",
        "                    h_ub: 2,\n                    k_ub: 2,\n                    v_ub: 2,\n",
    )
    t = sub(
        t,
        "            # One version each: the ring window is carried across rows on purpose and\n"
        "            # the row buffers are written then read inside the same row.\n",
        "            # h/k/v rows are double buffered so the DMA of row i+1 overlaps the\n"
        "            # compute of row i (T.Pipelined num_stages=2).\n",
    )
    variants["round1_c1"] = header(
        "round1_c1",
        "O-1: software-pipeline the row loop (num_stages=2) so the GM->UB row DMA of\n"
        "row i+1 overlaps the vector compute of row i (the baseline issues, waits and\n"
        "consumes one row per iteration: 27 serialised DMA waits per block at d=768).",
    ) + t

    # ------------------------------------------------------------------ c2
    # Drop the r_v store -> broadcast-load round trip (and its VST_VLD barrier):
    # rv_v[0] is already a fully broadcast fp32 vector.
    t = base
    t = sub(
        t,
        "                        # r_v is re-read from the statistics tile as a broadcast\n"
        "                        # vector (the same idiom the 950DT port uses for its gate\n"
        "                        # scalar).  That read is a vector store -> vector load\n"
        "                        # dependency, which needs the explicit barrier below.\n"
        "                        T.simd.mem_bar(\"VST_VLD\")\n"
        "                        rvb = T.simd.vld(st[3, i, 0], dist=\"BRC_B32\")\n",
        "                        # r_v is already a lane-broadcast fp32 vector: use the\n"
        "                        # register directly instead of storing it and re-loading it\n"
        "                        # as BRC_B32 behind a VST_VLD barrier.\n"
        "                        rvb = rv_v[0]\n",
    )
    variants["round1_c2"] = header(
        "round1_c2",
        "O-2: use the r_v register directly in the g = vhat * r_v pass.  The baseline\n"
        "stores r_v to the statistics tile and reloads it with dist=BRC_B32, which needs\n"
        "a VST_VLD mem_bar per row (9 barriers per block at d=768).",
    ) + t

    # ------------------------------------------------------------------ c3
    # Batch the two per-owned-row UB->GM stores into one 2-D copy each.
    t = base
    t = sub(t, "            y_ub = T.alloc_shared((dw,), dtype)\n",
            "            y_ub = T.alloc_shared((rc + 3, dw), dtype)\n")
    t = sub(t, "            vh_ub = T.alloc_shared((dw,), dtype)\n",
            "            vh_ub = T.alloc_shared((rc + 3, dw), dtype)\n")
    t = sub(t, "T.simd.vsts(vh_ub[ch * VL], narrowed, full, dist=\"PK_B32\")",
            "T.simd.vsts(vh_ub[i, ch * VL], narrowed, full, dist=\"PK_B32\")")
    t = sub(t, "                                y_ub[ch * VL],", "                                y_ub[i, ch * VL],")
    t = sub(
        t,
        "                if i >= 3:  # owned row only: the halo rows are not written back\n"
        "                    T.copy(vh_ub[:d], VHAT[b, r, :])\n"
        "                    T.copy(y_ub[:d], Y[b, r, :])\n",
        "",
    )
    t = sub(
        t,
        "            # -------- four fp32 row scalars: one strided copy per member --------\n",
        "            # -------- owned rows: one 2-D store per output, not one per row -----\n"
        "            # The owned rows are contiguous in GM, so a single (rc, d) region copy\n"
        "            # replaces rc per-row copies (TRAP-950-rowstore-mte3-drain: a per-row\n"
        "            # UB->GM copy drains PIPE_MTE3 before the next store).\n"
        "            T.copy(y_ub[3 : 3 + rc, :d], Y[b, t0 : t0 + rc, :])\n"
        "            T.copy(vh_ub[3 : 3 + rc, :d], VHAT[b, t0 : t0 + rc, :])\n"
        "\n"
        "            # -------- four fp32 row scalars: one strided copy per member --------\n",
    )
    variants["round1_c3"] = header(
        "round1_c3",
        "O-3: stage the output rows in a (rc+3, d) UB tile and write Y / VHAT with one\n"
        "2-D region copy per block instead of one copy per owned row (12 stores and 12\n"
        "PIPE_MTE3 drains per block at d=768 become 2 copies).",
    ) + t

    for name, text in variants.items():
        path = os.path.join(OUT_DIR, f"engram_gate_conv_{name}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print("wrote", path, len(text.splitlines()), "lines")


if __name__ == "__main__":
    main()
