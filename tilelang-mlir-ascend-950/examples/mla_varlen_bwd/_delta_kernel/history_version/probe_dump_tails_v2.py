"""Dump the full descriptor lines of the packed delta probe geometries (compile only)."""

import os
import sys

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import design_probe_packed_v2 as P  # noqa: E402
import tilelang  # noqa: E402


def dump(label, fn):
    print("===== " + label + " =====", flush=True)
    k = tilelang.compile(fn, target="ascend", out_idx=[])
    for ln in k.get_kernel_source().splitlines():
        s = ln.strip()
        if "asc_copy_gm2ub_align" in s or "asc_copy_ub2gm_align" in s:
            print("  " + s, flush=True)


dump("P1 D_v=128 heads=8", P._build(8, 128))
dump("P2 D_v=64 heads=4 (pad)", P._build(4, 64))
dump("P3 D_v=192 heads=4 (nvec=2)", P._build(4, 192))
