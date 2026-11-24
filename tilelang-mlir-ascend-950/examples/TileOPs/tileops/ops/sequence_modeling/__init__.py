"""Load only the requested operator; other toolchains remain isolated."""

from importlib import import_module

_EXPORTS = {
    "EngramDecodeFwdOp": (
        "tileops.ops.sequence_modeling.engram_decode",
        "EngramDecodeFwdOp",
    ),
    "EngramGateConvBwdOp": (
        "tileops.ops.sequence_modeling.engram_gate_conv_bwd",
        "EngramGateConvBwdOp",
    ),
    "EngramGateConvFwdOp": (
        "tileops.ops.sequence_modeling.engram_gate_conv",
        "EngramGateConvFwdOp",
    ),
}

__all__ = ["EngramDecodeFwdOp", "EngramGateConvBwdOp", "EngramGateConvFwdOp"]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
