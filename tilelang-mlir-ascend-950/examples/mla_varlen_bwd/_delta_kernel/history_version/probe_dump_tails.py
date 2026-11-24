"""Dump the descriptor tails of the packed delta probe geometries (compile only)."""

import os
import sys

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import design_probe_packed as P  # noqa: E402
import tilelang  # noqa: E402


def dump(label, fn, nchar=140):
    k = tilelang.compile(fn, target="ascend", out_idx=[2])
    print("===== " + label + " descriptor tails =====", flush=True)
    for ln in k.get_kernel_source().splitlines():
        s = ln.strip()
        if "asc_copy_gm2ub_align" in s or "asc_copy_ub2gm_align" in s:
            print("  .." + s[-nchar:], flush=True)


dump("P2 D_v=64 DW=128 heads=4", P._build(4, 64))
dump("P5 heads=128 D_v=128", P._build(128, 128))
dump("P3 D_v=192 nvec=2 heads=4", P._build(4, 192))
