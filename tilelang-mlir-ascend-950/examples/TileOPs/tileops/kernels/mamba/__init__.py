"""Load only the requested operator; other toolchains remain isolated."""

from importlib import import_module

_EXPORTS = {
    "SSDChunkScanFwdKernel": ("tileops.kernels.mamba.ssd_chunk_scan", "SSDChunkScanFwdKernel"),
    "SSDChunkCouplingFwdKernel": (
        "tileops.kernels.mamba.ssd_chunk_coupling",
        "SSDChunkCouplingFwdKernel",
    ),
}

__all__ = ["SSDChunkCouplingFwdKernel", "SSDChunkScanFwdKernel"]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
