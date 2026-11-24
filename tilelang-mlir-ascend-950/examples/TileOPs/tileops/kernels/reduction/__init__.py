"""Load only the requested operator; other toolchains remain isolated."""

from importlib import import_module

_EXPORTS = {
    "ArgreduceKernel": ("tileops.kernels.reduction.argmax", "ArgreduceKernel"),
    "LogSumExpKernel": ("tileops.kernels.reduction.logsumexp", "LogSumExpKernel"),
}

__all__ = ["ArgreduceKernel", "LogSumExpKernel"]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
