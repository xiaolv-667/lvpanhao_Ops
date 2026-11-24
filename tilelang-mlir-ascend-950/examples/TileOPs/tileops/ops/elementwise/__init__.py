"""Load only the requested operator; other toolchains remain isolated."""

from importlib import import_module

_EXPORTS = {
    "LerpTensorFwdOp": ("tileops.ops.elementwise.lerp_tensor", "LerpTensorFwdOp"),
    "MishFwdOp": ("tileops.ops.elementwise.mish", "MishFwdOp"),
}

__all__ = ["LerpTensorFwdOp", "MishFwdOp"]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
