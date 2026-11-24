"""Load only the requested operator; other toolchains remain isolated."""

from importlib import import_module

_EXPORTS = {
    "EngramDecodeKernel": (
        "tileops.kernels.sequence_modeling.engram_decode",
        "EngramDecodeKernel",
    ),
    "EngramGateConvBwdKernel": (
        "tileops.kernels.sequence_modeling.engram_gate_conv_bwd",
        "EngramGateConvBwdKernel",
    ),
    "EngramGateConvFwdKernel": (
        "tileops.kernels.sequence_modeling.engram_gate_conv",
        "EngramGateConvFwdKernel",
    ),
}

__all__ = ["EngramDecodeKernel", "EngramGateConvBwdKernel", "EngramGateConvFwdKernel"]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
