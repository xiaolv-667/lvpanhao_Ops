"""Load only the requested operator; other toolchains remain isolated."""

from importlib import import_module

_EXPORTS = {
    "GQAPrefillFwdKernel": (
        "tileops.kernels.attention.multi_head_attention",
        "GQAPrefillFwdKernel",
    ),
    "MLADenseBwdKernel": (
        "tileops.kernels.attention.mla_dense_bwd",
        "MLADenseBwdKernel",
    ),
    "MLAVarlenBwdKernel": (
        "tileops.kernels.attention.mla_varlen_bwd",
        "MLAVarlenBwdKernel",
    ),
    "PagedKVCacheWriteKernel": (
        "tileops.kernels.attention.paged_cache_write",
        "PagedKVCacheWriteKernel",
    ),
}

__all__ = [
    "GQAPrefillFwdKernel",
    "MLADenseBwdKernel",
    "MLAVarlenBwdKernel",
    "PagedKVCacheWriteKernel",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
