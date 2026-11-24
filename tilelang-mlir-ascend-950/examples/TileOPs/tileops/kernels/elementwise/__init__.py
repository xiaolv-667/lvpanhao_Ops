"""Load only the requested operator; other toolchains remain isolated."""

from importlib import import_module

_EXPORTS = {
    "LerpTensorFwdKernel": ("tileops.kernels.elementwise.lerp_tensor", "LerpTensorFwdKernel"),
    "MishFwdKernel": ("tileops.kernels.elementwise.mish", "MishFwdKernel"),
}

__all__ = ["LerpTensorFwdKernel", "MishFwdKernel"]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
