def shared_memory_budget(device=None) -> int:
    """Return the actual Unified Buffer (UB) capacity in bytes.

    Queries the Ascend chip model at runtime via
    :class:`tilelang.utils.npu_arch.AscendArch` and returns the
    real UB capacity (e.g. 192 KiB for Ascend910B, 256 KiB for
    Ascend910A, 248 KiB for Ascend950).  Falls back to 48 KiB when
    the chip model cannot be detected (e.g. on a non-Ascend
    machine or when ``torch_npu`` is unavailable).

    The *device* argument is accepted for API parity with
    :meth:`DeviceBackend.shared_memory_budget` but is not used:
    UB capacity is a per-chip-model constant, not a per-device
    property, so the current device's chip model is always used.
    """
    try:
        from tilelang.utils.npu_arch import get_arch_obj

        arch = get_arch_obj()
        ub_cap = int(arch.ub_cap)
        if ub_cap > 0:
            return ub_cap
    except Exception:
        pass
    return 48 * 1024
