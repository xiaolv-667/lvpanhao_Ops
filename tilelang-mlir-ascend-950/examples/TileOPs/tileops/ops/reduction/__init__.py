"""Load only the requested operator; other toolchains remain isolated."""

from importlib import import_module

_EXPORTS = {
    "ArgmaxFwdOp": ("tileops.ops.reduction.argmax", "ArgmaxFwdOp"),
    "LogSumExpFwdOp": ("tileops.ops.reduction.softmax", "LogSumExpFwdOp"),
}

__all__ = ["ArgmaxFwdOp", "LogSumExpFwdOp"]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
