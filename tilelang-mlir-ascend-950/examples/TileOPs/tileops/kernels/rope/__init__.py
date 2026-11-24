"""Load only the requested operator; other toolchains remain isolated."""

from importlib import import_module

_EXPORTS = {
    "RoPENeoxPositionIdsKernel": (
        "tileops.kernels.rope.rope",
        "RoPENeoxPositionIdsKernel",
    ),
    "RoPENeoxStagedRowsKernel": (
        "tileops.kernels.rope.rope",
        "RoPENeoxStagedRowsKernel",
    ),
    "RoPENonNeoxKernel": (
        "tileops.kernels.rope.rope",
        "RoPENonNeoxKernel",
    ),
    "RoPENonNeoxStagedRowsKernel": (
        "tileops.kernels.rope.rope",
        "RoPENonNeoxStagedRowsKernel",
    ),
}

__all__ = [
    "RoPENeoxPositionIdsKernel",
    "RoPENeoxStagedRowsKernel",
    "RoPENonNeoxKernel",
    "RoPENonNeoxStagedRowsKernel",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
