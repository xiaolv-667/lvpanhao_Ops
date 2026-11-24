"""Design probe #1 (Stage 1, _chain_speculative_sampling_kernel) -- how Philox is expressed.

The source draws its uniforms from Philox4x32-10, whose two round steps multiply a
counter word by a 32-bit constant and keep the **high** word: the GPU code calls the
CUDA intrinsic ``__umulhi``.  The Ascend AIV has no such intrinsic and no 64-bit
integer multiply evidence, so the design emulates ``mulhi`` with four 16x16-bit
products on 32-bit unsigned words.  This probe compiles the three candidate
expressions (no NPU, no launch) and reports which reach the device code:

  p1a  uint32 split-``mulhi`` emulation, scalar (kernel-scope ``T.serial``), seed
       and offset read as an ``int32`` bit-view of the two int64 words
  p1b  the same rounds with int64 arithmetic (``(int64(a) * m) >> 32``) and
       ``int64`` seed/offset shifts, i.e. the source's own wording
  p1c  the same uint32 emulation inside ``T.SimtVF`` with ``T.Parallel`` over the
       ``N + 1`` draws (one lane a draw)

Run (no NPU, no launch):
    cd examples/chain_speculative_sampling/_chain_speculative_sampling_kernel
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        history_version/design_probe_philox.py [p1a|p1b|p1c|all]
"""

import os
import sys
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

# Philox4x32-10 round multipliers and Weyl key increments (Salmon et al., SC'11).
M0 = 0xD2511F53
M1 = 0xCD9E8D57
W0 = 0x9E3779B9
W1 = 0xBB67AE85
ROUNDS = 10
UNIFORM_BITS = 24

_OPS = ("mul", "vshl", "vshr", "vmul", "vand", "vor", "vxor", "shl", "shr", "and", "or", "xor")


def _mulhilo(a, m):
    """``(hi, lo)`` of ``a * m`` in 32-bit words, from four 16x16 products."""
    ah = a >> T.uint32(16)
    al = a & T.uint32(0xFFFF)
    mh, ml = m >> 16, m & 0xFFFF
    p0 = al * T.uint32(ml)
    p1 = ah * T.uint32(ml)
    p2 = al * T.uint32(mh)
    p3 = ah * T.uint32(mh)
    t = (p1 & T.uint32(0xFFFF)) + (p2 & T.uint32(0xFFFF)) + (p0 >> T.uint32(16))
    lo = (p0 & T.uint32(0xFFFF)) | ((t & T.uint32(0xFFFF)) << T.uint32(16))
    hi = p3 + (p1 >> T.uint32(16)) + (p2 >> T.uint32(16)) + (t >> T.uint32(16))
    return hi, lo


@T.macro
def _rounds_split(c, k, b):
    """Ten rounds over the four counter words ``c`` and the two key words ``k``."""
    for step in T.serial(ROUNDS):
        if step > 0:
            k[0] = k[0] + T.uint32(W0)
            k[1] = k[1] + T.uint32(W1)
        hi0, lo0 = _mulhilo(c[0], M0)
        hi1, lo1 = _mulhilo(c[2], M1)
        b[0] = hi1 ^ c[1] ^ k[0]
        b[1] = hi0 ^ c[3] ^ k[1]
        c[1] = lo1
        c[3] = lo0
        c[0] = b[0]
        c[2] = b[1]


@T.macro
def _rounds_int64(c, k, b):
    """Ten rounds with 64-bit products and shifts (the source's own wording)."""
    m0, m1 = T.int64(M0), T.int64(M1)
    for step in T.serial(ROUNDS):
        if step > 0:
            k[0] = k[0] + T.uint32(W0)
            k[1] = k[1] + T.uint32(W1)
        hi0 = T.cast((T.cast(c[0], "int64") * m0) >> T.int64(32), "uint32")
        lo0 = T.cast(T.cast(c[0], "int64") * m0, "uint32")
        hi1 = T.cast((T.cast(c[2], "int64") * m1) >> T.int64(32), "uint32")
        lo1 = T.cast(T.cast(c[2], "int64") * m1, "uint32")
        b[0] = hi1 ^ c[1] ^ k[0]
        b[1] = hi0 ^ c[3] ^ k[1]
        c[1] = lo1
        c[3] = lo0
        c[0] = b[0]
        c[2] = b[1]


def _report(name, fn):
    print(f"\n===== {name} =====", flush=True)
    try:
        kernel = tilelang.compile(fn, target="ascend", out_idx=[])
        print("COMPILE: OK", flush=True)
    except Exception:  # noqa: BLE001
        print("COMPILE: FAIL", flush=True)
        traceback.print_exc(limit=4)
        return None
    src = kernel.get_kernel_source()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"probe_{name}_gen.txt")
    with open(path, "w") as fh:
        fh.write(src)
    hits = [ln.strip() for ln in src.splitlines() if any(o in ln for o in _OPS)]
    print(f"  generated-code lines with integer ops: {len(hits)}", flush=True)
    for ln in hits[:12]:
        print("   ", ln[:160], flush=True)
    return src


def make_p1a():
    """Scalar rounds on the split-``mulhi`` emulation, seed/offset as int32 words."""

    @T.prim_func
    def main(
        seed_w: T.Tensor((2,), "int32"),
        offset_w: T.Tensor((2,), "int32"),
        out: T.Tensor((9,), "float32"),
    ):
        with T.Kernel(1):
            c = T.alloc_local((4,), "uint32")
            k = T.alloc_local((2,), "uint32")
            b = T.alloc_local((2,), "uint32")
            for j in T.serial(9):
                k[0] = T.reinterpret(seed_w[0], "uint32")
                k[1] = T.reinterpret(seed_w[1], "uint32")
                c[0] = T.uint32(j)
                c[1] = T.uint32(3)
                c[2] = T.reinterpret(offset_w[0], "uint32")
                c[3] = T.reinterpret(offset_w[1], "uint32")
                _rounds_split(c, k, b)
                out[j] = T.cast(c[0] >> T.uint32(32 - UNIFORM_BITS), "float32") * T.float32(
                    2.0**-UNIFORM_BITS
                )

    return main


def make_p1b():
    """The source's own wording: int64 products and shifts, int64 seed/offset."""

    @T.prim_func
    def main(seed: T.Tensor((1,), "int64"), offset: T.Tensor((1,), "int64"), out: T.Tensor((9,), "float32")):
        with T.Kernel(1):
            c = T.alloc_local((4,), "uint32")
            k = T.alloc_local((2,), "uint32")
            b = T.alloc_local((2,), "uint32")
            for j in T.serial(9):
                k[0] = T.cast(seed[0] & T.int64(0xFFFFFFFF), "uint32")
                k[1] = T.cast((seed[0] >> T.int64(32)) & T.int64(0xFFFFFFFF), "uint32")
                c[0] = T.uint32(j)
                c[1] = T.uint32(3)
                c[2] = T.cast(offset[0] & T.int64(0xFFFFFFFF), "uint32")
                c[3] = T.cast((offset[0] >> T.int64(32)) & T.int64(0xFFFFFFFF), "uint32")
                _rounds_int64(c, k, b)
                out[j] = T.cast(c[0] >> T.uint32(32 - UNIFORM_BITS), "float32") * T.float32(
                    2.0**-UNIFORM_BITS
                )

    return main


def make_p1c():
    """The same emulation with a lane per draw inside ``T.SimtVF``."""

    @T.prim_func
    def main(
        seed_w: T.Tensor((2,), "int32"),
        offset_w: T.Tensor((2,), "int32"),
        out: T.Tensor((9,), "float32"),
    ):
        with T.Kernel(1) as bx, T.SimtVF(threads=32):
            cw = T.alloc_shared((9, 4), "uint32")
            kw = T.alloc_shared((2,), "uint32")
            bw = T.alloc_shared((9, 2), "uint32")
            kw[0] = T.reinterpret(seed_w[0], "uint32")
            kw[1] = T.reinterpret(seed_w[1], "uint32")
            for j, w in T.Parallel(9, 4):
                cw[j, w] = T.uint32(0)
            for j in T.Parallel(9):
                cw[j, 0] = T.uint32(j)
                cw[j, 1] = T.uint32(bx)
                cw[j, 2] = T.reinterpret(offset_w[0], "uint32")
                cw[j, 3] = T.reinterpret(offset_w[1], "uint32")
                bw[j, 0] = T.uint32(0)
                bw[j, 1] = T.uint32(0)
            for step in T.serial(ROUNDS):
                for j in T.Parallel(9):
                    if step > 0:
                        kw[0] = kw[0] + T.uint32(W0)
                        kw[1] = kw[1] + T.uint32(W1)
                    hi0, lo0 = _mulhilo(cw[j, 0], M0)
                    hi1, lo1 = _mulhilo(cw[j, 2], M1)
                    bw[j, 0] = hi1 ^ cw[j, 1] ^ kw[0]
                    bw[j, 1] = hi0 ^ cw[j, 3] ^ kw[1]
                    cw[j, 1] = lo1
                    cw[j, 3] = lo0
                    cw[j, 0] = bw[j, 0]
                    cw[j, 2] = bw[j, 1]
            for j in T.Parallel(9):
                out[j] = T.cast(cw[j, 0] >> T.uint32(32 - UNIFORM_BITS), "float32") * T.float32(
                    2.0**-UNIFORM_BITS
                )

    return main


_CASES = {"p1a": make_p1a, "p1b": make_p1b, "p1c": make_p1c}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "all"
    cases = _CASES if name == "all" else {name: _CASES[name]}
    for label, build in cases.items():
        _report(label, build())


if __name__ == "__main__":
    main()
