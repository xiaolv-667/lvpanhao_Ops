"""Load only the requested operator; other toolchains remain isolated."""

from importlib import import_module

_EXPORTS = {"AdaLayerNormKernel": ("tileops.kernels.norm.ada_layer_norm", "AdaLayerNormKernel")}

__all__ = ["AdaLayerNormKernel"]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
