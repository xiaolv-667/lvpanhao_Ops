"""Load only the requested operator; other toolchains remain isolated."""

from importlib import import_module

_EXPORTS = {
    "SSDChunkScanFwdOp": ("tileops.ops.mamba.ssd_chunk_scan", "SSDChunkScanFwdOp"),
    "SSDChunkCouplingFwdOp": ("tileops.ops.mamba.ssd_chunk_coupling", "SSDChunkCouplingFwdOp"),
}

__all__ = ["SSDChunkCouplingFwdOp", "SSDChunkScanFwdOp"]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
