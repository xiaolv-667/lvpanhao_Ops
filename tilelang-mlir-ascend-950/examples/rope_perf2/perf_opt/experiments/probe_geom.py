"""Probe one geometry of a candidate module: does it build, and with which form?

    python probe_geom.py --module <variant.py> [--case tiny|odd24|llama]
"""

import argparse
import importlib.util
import os
import sys

import torch  # noqa: F401
import torch_npu  # noqa: F401


CASES = {
    # (query_heads, key_heads, head_size, rotary_dim, max_position, dtype, neox)
    "tiny": (64, 1, 64, 24, 2048, "float32", False),   # pytest smoke: copies columns
    "odd24": (4, 2, 64, 24, 2048, "float16", False),  # L0-4 style
    "llama": (32, 8, 128, 128, 8192, "float32", True),
    "phi2": (32, 0, 80, 32, 2048, "bfloat16", True),
}


def load(path):
    name = "probe_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--case", default="tiny", choices=sorted(CASES))
    args = ap.parse_args()
    mod = load(os.path.abspath(args.module))
    q, k, dim, rot, pos, dtype, neox = CASES[args.case]
    try:
        fn = mod._build(q, k, dim, rot, pos, dtype, "int64", None, neox,
                        mod._UNITS_PER_BLOCK, False)
        src = mod._rope_positions_kernel.__globals__  # noqa: F841
        print(f"[OK]   {args.case} built ({q}x{dim} q, {k}x{dim} k, R={rot}, {dtype})")
        return 0
    except Exception as exc:  # noqa: BLE001 - probe
        msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        print(f"[FAIL] {args.case}: {msg[:200]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
