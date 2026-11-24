"""PL-950-vector-execution: fixed SIMD width and GM tail copies.
First verified: 2026-11-18 TileLang ascend 0.1.15 + CANN 9.1 simulator.
"""

import functools
import json
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
import tilelang
import tilelang.ascend.language as T


@T.macro
def _body(x, y, width: int, dtype: str, floating: bool):
    if floating:
        with T.SimdVF():
            if dtype == "int16":
                mask = T.simd.pset(16)
                signless = T.simd.vdup(T.int16(32767), "int16")
                for i in range(width // 128):
                    value = T.simd.vld(x[i * 128])
                    T.simd.vsts(y[i * 128], T.simd.vand(value, signless, mask), mask)
            else:
                mask = T.simd.pset(32)
                signless = T.simd.vdup(T.int32(2147483647), "int32")
                for i in range(width // 64):
                    value = T.simd.vld(x[i * 64])
                    T.simd.vsts(y[i * 64], T.simd.vand(value, signless, mask), mask)
    elif dtype == "uint8":
        with T.SimdVF():
            mask = T.simd.pset(8)
            for i in range(width // 256):
                T.simd.vsts(y[i * 256], T.simd.vld(x[i * 256]), mask)
    else:
        with T.SimtVF(threads=256):
            for i in T.Parallel(width):
                y[i] = T.if_then_else(x[i] < 0, -x[i], x[i])


@functools.lru_cache(maxsize=256)
def build(n, dtype, floating, tile=16384, blocks=64, stages=3):
    full, tail = divmod(n, tile)
    grid = min(blocks, full) if full else 1
    padded_tail = (tail + 255) // 256 * 256

    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(grid) as bx:
            if full:
                x = T.alloc_shared((tile,), dtype)
                y = x
                T.annotate_buffer_versions({x: stages})
                for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                    block = step * grid + bx
                    if block < full:
                        begin = block * tile
                        T.copy(X[begin : begin + tile], x, l2_cache_ctrl=4)
                        _body(x, y, tile, dtype, floating)
                        T.copy(y, Y[begin : begin + tile], l2_cache_ctrl=0)
            if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                if bx == 0:
                    xt = T.alloc_shared((padded_tail,), dtype)
                    yt = xt
                    T.copy(X[full * tile : n], xt[:tail], l2_cache_ctrl=4)
                    _body(xt, yt, padded_tail, dtype, floating)
                    T.copy(yt[:tail], Y[full * tile : n], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=-1)


for n in (256, 257):
    cpu_x = torch.linspace(-3, 3, n, dtype=torch.float32)
    x = cpu_x.to("npu")
    fn = build(n, "int32", True, tile=256, blocks=1, stages=1)
    result = fn(x.view(torch.int32)).view(torch.float32).cpu()
    torch.testing.assert_close(result, cpu_x.abs(), atol=0, rtol=0)
    print(json.dumps({"case": "simd_abs", "n": n, "precision": "pass"}), flush=True)
print("[PRECISION_PASS]", flush=True)


@functools.lru_cache
def build_simt(n):
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
            if tail:  # noqa: SIM102 - keep the compile-time tail test separate
                if bx == full:
                    T.copy(X[full * tile : n], x[:tail])
                    with T.SimtVF(threads=256):
                        for i in T.Parallel(tail):
                            y[i] = T.if_then_else(x[i] < 0, -x[i], x[i])
                    T.copy(y[:tail], Y[full * tile : n])

    return tilelang.compile(main, target="ascend", out_idx=-1)


for n in (256, 257):
    x = torch.linspace(-3, 3, n, dtype=torch.float32).to("npu")
    torch.testing.assert_close(build_simt(n)(x).cpu(), x.cpu().abs(), atol=0, rtol=0)
    print(json.dumps({"case": "simt_abs", "n": n, "precision": "pass"}), flush=True)
print("[PRECISION_PASS]", flush=True)
