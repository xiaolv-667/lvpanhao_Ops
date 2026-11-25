"""Build round-3 branch c6 (= c5 + full-lane mask hoisting).

After c5 the DMA is no longer the limiter (mte2 busy ratio 0.31 -> 0.13, vector
wait ratio 0.88 -> 0.12) and the block is issue-bound: vec 0.65 + scalar 0.23 of
a 6.5 us block.  Every chunk iteration of every pass re-derives its lane mask
(`T.simd.update_mask(min(64, d - ch*64), 32)`, 5 passes x nchunks per row = 60
mask builds per row at d=768).  For `d % 64 == 0` -- all four benchmark
workloads -- every mask is the full 64-lane pattern, so the mask is just the
predicate register that `full`/`full_p` already holds; the update_mask call can
be folded away at build time.  The tail-lane path (d % 64 != 0, e.g. L0's d=196
and the d=200 boundary case) keeps the original expression.

    python mk_round3_c6.py
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "engram_gate_conv_round2_c5.py")
DST = os.path.join(os.path.dirname(HERE), "engram_gate_conv_round3_c6.py")


def main():
    with open(SRC, encoding="utf-8") as f:
        text = f.read()

    # build-time flag right after the geometry constants of the factory
    anchor = "    fp32 = \"float32\"\n"
    assert text.count(anchor) == 1
    text = text.replace(
        anchor,
        "    # d is a multiple of the fp32 vector width: every lane mask is the full\n"
        "    # 64-lane pattern, so the per-chunk mask build folds into the hoisted\n"
        "    # predicate register (build-time decision, tail path keeps update_mask).\n"
        "    full_lanes = (d % VL == 0)\n"
        "\n" + anchor,
    )

    # prologue frame (variable `full_p`) and row frames (variable `full`)
    head, sep, tail = text.partition("            # -------- row pipeline")
    assert sep

    pat = r"mask = T\.simd\.update_mask\(T\.min\(VL, d - ch \* VL\), 32\)"
    head, n_head = re.subn(
        pat, "mask = full_p if full_lanes else T.simd.update_mask(T.min(VL, d - ch * VL), 32)", head
    )
    tail, n_tail = re.subn(
        pat, "mask = full if full_lanes else T.simd.update_mask(T.min(VL, d - ch * VL), 32)", tail
    )
    print(f"mask sites: prologue {n_head}, row body {n_tail}")
    if n_head + n_tail != 6:
        raise SystemExit("unexpected mask site count")

    with open(DST, "w", encoding="utf-8") as f:
        f.write(head + sep + tail)
    print("wrote", DST)


if __name__ == "__main__":
    main()
