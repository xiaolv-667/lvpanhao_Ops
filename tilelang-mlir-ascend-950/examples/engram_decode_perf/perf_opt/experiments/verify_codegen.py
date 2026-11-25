"""Codegen equivalence check: baseline vs perf_opt project launch (Stage 4).

The Stage-4 change to the project launch is *shape adaptive*: the wide per-batch-row
cache shift is only selected when the baseline would serialise more than one copy
pair per block (`batch > n_tiles`).  For every other shape the emitted device code
must be byte-identical to the frozen baseline, which is the strongest possible
"no regression on the other cases" evidence.

This script compiles both modules' project kernel at the benchmark shapes and
compares the generated `device_kernel.asc` byte for byte.

Usage (cwd = examples/TileOPs, no device work beyond compilation):

    python verify_codegen.py
"""

import difflib
import hashlib
import importlib.util
import os
import re
import sys

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
PERF_OPT = os.path.dirname(HERE)
BASE_DIR = os.path.abspath(os.path.join(
    PERF_OPT, "..", "..", "TileOPs", "tileops", "kernels", "sequence_modeling",
    "engram_decode", "engram_decode_kernel"))

# label -> (batch, d_mem, d_pad, max_conv_len, n_tiles, dtype)
SHAPES = {
    "decode-s1": (1, 256, 256, 9, 4, "float16"),
    "serving-b1": (1, 512, 256, 12, 4, "float16"),
    "serving-b4": (4, 1024, 512, 20, 8, "float16"),
    "serving-b8": (8, 512, 256, 18, 4, "bfloat16"),
    "decode-partial-cache": (2, 512, 256, 12, 4, "float16"),
    "L1-ktail-519-b17": (17, 519, 256, 12, 4, "float16"),
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build(mod, geom):
    batch, d_mem, d_pad, mcl, _nt, dt = geom
    return mod._engram_project_kernel(batch, d_mem, d_pad, mcl, dt)(
        **mod.default_config(d_pad))


def asc_of(kernel):
    """Path of the device source the JIT produced for this compiled kernel."""
    p = getattr(kernel, "_tilelang_cache_path", None)
    if isinstance(p, str):
        cand = os.path.join(p, "device_kernel.asc")
        if os.path.exists(cand):
            return cand
    for attr in ("artifact_path", "kernel_source", "source_path"):
        p = getattr(kernel, attr, None)
        if isinstance(p, str) and p.endswith(".asc"):
            return p
    for attr in ("libpath", "lib_path"):
        p = getattr(kernel, attr, None)
        if isinstance(p, str):
            cand = os.path.join(os.path.dirname(p), "device_kernel.asc")
            if os.path.exists(cand):
                return cand
    raise RuntimeError(f"cannot locate device_kernel.asc on {type(kernel)}; "
                       f"attrs={[a for a in dir(kernel) if 'path' in a or 'src' in a]}")


def main():
    base = load(os.path.join(BASE_DIR, "_engram_project_kernel.py"), "base_proj")
    opt = load(os.path.join(PERF_OPT, "EngramDecodeFwdOp.py"), "opt_proj")
    if hasattr(opt, "project_default_config"):
        opt.default_config = opt.project_default_config
    for label, geom in SHAPES.items():
        a = asc_of(build(base, geom))
        b = asc_of(build(opt, geom))
        # the only intrinsic difference between the two module layouts is the
        # compiler's per-module buffer-version counter suffix (`..._1` vs `..._2`)
        def norm(path):
            txt = open(path).read()
            return re.sub(r"version_counter_\d+", "version_counter_N", txt)
        ha = hashlib.sha256(norm(a).encode()).hexdigest()[:16]
        hb = hashlib.sha256(norm(b).encode()).hexdigest()[:16]
        ta, tb = norm(a), norm(b)
        same = ha == hb
        nt = geom[4]
        print(f"{label:24s} batch={geom[0]:3d} n_tiles={nt:2d} "
              f"wide_shift={'yes' if geom[0] > nt else 'no ':3s} "
              f"baseline={ha} tuned={hb} identical={same}")
        if not same:
            diff = list(difflib.unified_diff(ta.splitlines(), tb.splitlines(),
                                             "baseline", "tuned", lineterm="", n=1))
            print(f"  ({len(diff)} diff lines; shift region shown)")
            for line in diff[:40]:
                print("   " + line[:170])


if __name__ == "__main__":
    main()
