"""Compile-only smoke test for the Ascend NPU path.

Follows the framework's own kernel template
(.agents/platforms/toolchains/ascend/kernel_template.py):
    import tilelang.ascend.language as T
    with T.Kernel(grid) as bx: ...
    tilelang.compile(main, target="ascend", out_idx=-1)

The compiled kernel is never launched: this only proves that codegen plus the
configured CANN compiler produce a device binary. No card is reserved — callers
should run it with ASCEND_RT_VISIBLE_DEVICES set empty.
"""

import os
import sys
import time

import tilelang
import tilelang.ascend.language as T

N = 4096


def build(n):
    tile = 256
    full, tail = divmod(n, tile)
    grid = full + bool(tail)

    @T.prim_func
    def main(X: T.Tensor((n,), "float32"), Y: T.Tensor((n,), "float32")):
        with T.Kernel(grid) as bx:
            x = T.alloc_shared((tile,), "float32")
            y = T.alloc_shared((tile,), "float32")
            if bx < full:
                start = bx * tile
                T.copy(X[start : start + tile], x)
                with T.SimtVF(threads=256):
                    for i in T.Parallel(tile):
                        y[i] = T.if_then_else(x[i] < 0, -x[i], x[i])
                T.copy(y, Y[start : start + tile])
            if tail:
                if bx == full:
                    T.copy(X[full * tile : n], x[:tail])
                    with T.SimtVF(threads=256):
                        for i in T.Parallel(tail):
                            y[i] = T.if_then_else(x[i] < 0, -x[i], x[i])
                    T.copy(y[:tail], Y[full * tile : n])

    return main


def main() -> int:
    print(f"tilelang                 : {getattr(tilelang, '__version__', '?')}")
    print(f"ASCEND_HOME_PATH         : {os.environ.get('ASCEND_HOME_PATH')}")
    print(f"ASCEND_RUNTIME_ROOT      : {os.environ.get('ASCEND_RUNTIME_ROOT')}")
    print(f"ASCEND_RT_VISIBLE_DEVICES: {os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!r}")
    print(f"TILELANG_CACHE_DIR       : {os.environ.get('TILELANG_CACHE_DIR')}")
    print(f"CXX                      : {os.environ.get('CXX')}")

    func = build(N)
    started = time.time()
    try:
        kernel = tilelang.compile(func, target="ascend", out_idx=-1)
    except Exception as exc:  # noqa: BLE001 - smoke test wants the raw failure
        print(f"COMPILE_FAILED after {time.time() - started:.1f}s: {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.time() - started

    source_len = 0
    for attr in ("get_kernel_source", "get_source"):
        getter = getattr(kernel, attr, None)
        if callable(getter):
            try:
                source_len = len(getter() or "")
                break
            except Exception:  # noqa: BLE001
                pass

    print(f"COMPILE_OK in {elapsed:.1f}s")
    print(f"kernel type  : {type(kernel).__name__}")
    print(f"source chars : {source_len}")
    print("NOTE: kernel was NOT launched (no device execution).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
