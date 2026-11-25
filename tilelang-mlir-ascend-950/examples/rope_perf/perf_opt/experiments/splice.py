"""Splice a new factory into a copy of the Stage 3 kernel module.

    python perf_opt/experiments/splice.py \
        --base examples/TileOPs/tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py \
        --factory perf_opt/experiments/factory_c1.py \
        --out perf_opt/rope_round1_c1.py

The base module is read-only (Stage 3 baseline); everything after the
``_rope_positions_kernel`` definition (``_build``, the golden, the layered tests,
the CLI) is kept verbatim, so each experiment branch keeps the full L0/L1/L2
precision suite of the baseline.
"""

import argparse

START = "def _rope_positions_kernel("
END = "@functools.lru_cache(maxsize=128)\ndef _build("


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--factory", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--constants", default=None,
                    help="optional python literal dict of module constants to override")
    args = ap.parse_args()

    base = open(args.base, encoding="utf-8").read()
    factory = open(args.factory, encoding="utf-8").read()
    i, j = base.index(START), base.index(END)
    out = base[:i] + factory + base[j:]
    if args.constants:
        overrides = eval(args.constants)  # noqa: S307 - local experiment tooling
        for name, value in overrides.items():
            out = _replace_constant(out, name, value)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"{args.out}: {len(out)} chars")


def _replace_constant(text, name, value):
    import re

    pattern = re.compile(rf"^{name} = .*$", re.M)
    if not pattern.search(text):
        raise SystemExit(f"constant {name} not found")
    return pattern.sub(f"{name} = {value!r}", text, count=1)


if __name__ == "__main__":
    main()
