"""Bit-exactness gate on the *deployed* pair: installed perf_opt kernel vs Stage 3 kernel.

`verify_exact.py` compares the tuning artifacts in this task directory; this
script compares the two modules the integration package actually exposes, i.e.
what the flipped wrapper imports

    perf_opt  : tileops.kernels.sampling.chain_speculative_sampling
                .chain_speculative_sampling_kernel.perf_opt._chain_speculative_sampling_kernel
    baseline  : tileops.kernels.sampling.chain_speculative_sampling
                .chain_speculative_sampling_kernel

against each other and against the torch reference, on the same 26 cases
(benchmark shapes, L0/L1/boundary shapes, the four control-flow paths, a
``storage_offset=1`` view, an odd vocabulary and a one-token vocabulary).

Usage (task dir):
  python perf_opt/verify_exact_prod.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TILEOPS_ROOT = Path(__file__).resolve().parents[2] / "TileOPs"
sys.path.insert(0, str(TILEOPS_ROOT))

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

import verify_exact as V  # noqa: E402

BASE_PKG = (
    "tileops.kernels.sampling.chain_speculative_sampling"
    ".chain_speculative_sampling_kernel"
)
PERF_MOD = BASE_PKG + ".perf_opt._chain_speculative_sampling_kernel"


def main():
    import importlib

    base_pkg = importlib.import_module(BASE_PKG)
    perf_mod = importlib.import_module(PERF_MOD)
    wrapper = importlib.import_module(
        "tileops.kernels.sampling.chain_speculative_sampling.chain_speculative_sampling"
    )
    deployed = wrapper._chain_speculative_sampling_kernel
    assert deployed is perf_mod._chain_speculative_sampling_kernel, (
        "the wrapper does not import the perf_opt kernel -- the switch block is not flipped"
    )
    print(f"deployed factory module: {deployed.__module__}", flush=True)
    print(f"baseline package: {base_pkg.__name__}", flush=True)

    V.cand = perf_mod
    V.load_baseline = lambda: base_pkg
    return V.main()


if __name__ == "__main__":
    raise SystemExit(main())
