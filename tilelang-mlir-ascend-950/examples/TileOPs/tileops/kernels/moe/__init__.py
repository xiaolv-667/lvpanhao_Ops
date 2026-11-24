"""Load only the requested operator; other toolchains remain isolated."""

from importlib import import_module

_EXPORTS = {
    "MoEPermuteAlignKernel": (
        "tileops.kernels.moe.moe_permute_align",
        "MoEPermuteAlignKernel",
    ),
    "SharedExpertMLPKernel": (
        "tileops.kernels.moe.shared_expert_mlp",
        "SharedExpertMLPKernel",
    ),
}

__all__ = ["MoEPermuteAlignKernel", "SharedExpertMLPKernel"]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
