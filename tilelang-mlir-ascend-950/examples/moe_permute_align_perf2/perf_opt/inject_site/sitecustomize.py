"""Stage 4 measurement shim: dispatch the frozen TileOPs wrapper to a candidate.

The frozen wrapper ``tileops/kernels/moe/moe_permute_align/moe_permute_align.py``
imports the three kernel factories at module load and looks them up as module
globals on every build, so a candidate file can be measured **without touching
any tracked file** by replacing those three globals.

Enabled only when ``MOE_ALIGN_CANDIDATE`` names a Python file that defines
``_make_align_kernel`` / ``_make_scatter_kernel`` / ``_make_small_batch_kernel``
with the baseline signatures.  Imported automatically by any interpreter that
finds this directory on ``PYTHONPATH`` (``site`` imports ``sitecustomize``),
which is also how ``tileops.benchmark.msprof`` propagates it into the profiled
subprocess (it forwards ``os.environ`` and prepends the cwd to ``PYTHONPATH``).

If ``MOE_ALIGN_CANDIDATE`` is unset the shim is a no-op, so the baseline path
(Stage 3 source) is measured with exactly the frozen wrapper.
"""

import os
import sys

TARGET_MODULE = "tileops.kernels.moe.moe_permute_align.moe_permute_align"
FACTORIES = ("_make_align_kernel", "_make_scatter_kernel", "_make_small_batch_kernel")


def _patch(module):
    import importlib.util

    path = os.environ.get("MOE_ALIGN_CANDIDATE")
    if not path:
        return
    spec = importlib.util.spec_from_file_location("moe_align_candidate", path)
    candidate = importlib.util.module_from_spec(spec)
    sys.modules["moe_align_candidate"] = candidate
    spec.loader.exec_module(candidate)
    for name in FACTORIES:
        setattr(module, name, getattr(candidate, name))
    module.MOE_ALIGN_CANDIDATE = path
    print(f"[inject] {TARGET_MODULE} -> {path}", file=sys.stderr)


def _install():
    if not os.environ.get("MOE_ALIGN_CANDIDATE"):
        return
    from importlib.machinery import PathFinder
    from importlib.abc import MetaPathFinder

    class _CandidateFinder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != TARGET_MODULE:
                return None
            spec = PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            original = spec.loader

            class _Loader:
                def create_module(self, spec_):
                    return original.create_module(spec_)

                def exec_module(self, module):
                    original.exec_module(module)
                    _patch(module)

            spec.loader = _Loader()
            return spec

    sys.meta_path.insert(0, _CandidateFinder())


_install()
