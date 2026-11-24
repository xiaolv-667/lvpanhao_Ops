"""Load only the requested operator; other toolchains remain isolated."""

from importlib import import_module

_EXPORTS = {
    "MLADenseBwdOp": (
        "tileops.ops.attention.mla_dense_bwd",
        "MLADenseBwdOp",
    ),
    "MLAVarlenBwdOp": (
        "tileops.ops.attention.mla_varlen_bwd",
        "MLAVarlenBwdOp",
    ),
    "MultiHeadAttentionFwdOp": (
        "tileops.ops.attention.multi_head_attention",
        "MultiHeadAttentionFwdOp",
    ),
    "PagedKVCacheWriteFwdOp": (
        "tileops.ops.attention.paged_cache_write",
        "PagedKVCacheWriteFwdOp",
    ),
}

__all__ = [
    "MLADenseBwdOp",
    "MLAVarlenBwdOp",
    "MultiHeadAttentionFwdOp",
    "PagedKVCacheWriteFwdOp",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
