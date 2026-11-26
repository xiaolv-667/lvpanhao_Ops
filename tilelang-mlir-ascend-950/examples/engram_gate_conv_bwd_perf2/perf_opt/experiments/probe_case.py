"""Run one or a few precision cases of a bwd variant module (fast iteration).

    python probe_case.py --module <variant.py> --case 3,96,768,bfloat16 [--case ...]

Prints the per-output max|diff| and the trailing ``[L0] k/n`` verdict of the
module's own runner.
"""

import argparse
import importlib.util
import os
import sys
import time


def load_module(path):
    name = "probe_variant_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--case", action="append", required=True,
                    help="M,seq_len,d,dtype[,repeat]")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    mod = load_module(os.path.abspath(args.module))
    t0 = time.perf_counter()
    for spec in args.case:
        parts = spec.split(",")
        M, seq_len, d, dtype = int(parts[0]), int(parts[1]), int(parts[2]), parts[3]
        rep = int(parts[4]) if len(parts) > 4 else args.repeat
        ok = mod.run_case("probe", spec, M, seq_len, d, dtype, seed=777, repeat=rep)
        print(f"  -> {spec}: {'PASS' if ok else 'FAIL'}", flush=True)
    mod._finish("probe", strict=True)
    print(f"[TIME] {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
