"""Design-phase compile probe (D-3) for ``_silu_mul_fused_kernel`` on Ascend950PR.

No device work: ``tilelang.compile(..., target="ascend")`` only lowers and calls
the BishengIR compiler, so this runs with the card hidden:

    cd examples/shared_expert_mlp/_silu_mul_fused_kernel
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES= /workspace/venv-ascend950/bin/python history_version/design_probe_compile.py

What it validates (the constructs DESIGN.md 3.3 / 6.2 depend on):

  P1  the main geometry (bm=16, bn=512, 2 column tiles per core,
      ``T.Pipelined(2)`` + ``T.annotate_buffer_versions``) for
      kimi-k2-t512's ds-v3-t4096-shaped tile: 2-D strided ``T.copy`` of the
      gate half and of the ``+N`` up half of one ``[M, 2N]`` tensor into
      separate UB tiles, static ``am``/``an`` extents;
  P2  the same chain in ``float16`` at the small geometry (bm=1, bn=256) --
      the ds-v3-t32 dtype case;
  P3  both dynamic tails at once (``am = 1 < bm``, ``an = 476 < bn``) with
      ``tpb == 1`` (no pipeline frame): the runtime-extent ``T.copy`` path;
  P4  the kimi-k2-t512 geometry (bm=16, bn=512, 4 tiles per core, grid 288).

Per 64-element chunk the SIMD chain is the one DESIGN.md 1.6.1 A1 selects:

    vld(UNPK_B16) -> vcvt(fp32, part=0)        # gate, up
    num = vmul(g, u)                            # g*u
    den = vadds(vexp(vneg(g)), 1.0)             # 1 + exp(-g)
    y   = vdiv(num, den)                        # g*u / (1+exp(-g))
    vcvt(y, dtype, part=0) -> vsts(dist="PK_B32")

which is 1 vector op fewer than the source's ``g*sigmoid(g)*u`` (5 -> 4) and
uses only constructs already exercised by completed ascend kernels
(``engram_gate_conv`` / ``mla_dense_bwd/_delta_kernel``): ``vld``+``vcvt
(part=0)``+``vmul``/``vdiv``/``vexp``+``vsts(PK_B32)`` for the 64-lane chain,
``T.Pipelined`` + ``annotate_buffer_versions`` for the tile loop.
"""

import functools
import sys
import traceback

import tilelang
import tilelang.ascend.language as T

VL = 64          # fp32 lanes per vector register; also the fp16 chunk (UNPK_B16/PK_B32)
UB_CAPACITY = 253952  # Ascend950PR_9579.ini [AICoreSpec] ub_size


def geometry(M, N, bm, bn, nsp):
    """Build-time geometry: column tiles per core and the launched grid."""
    mt = -(-M // bm)
    nt = -(-N // bn)
    if nt % nsp:
        raise ValueError(f"nsp={nsp} does not divide nt={nt}")
    return {
        "MT": mt,
        "NT": nt,
        "bm": bm,
        "bn": bn,
        "tpb": nt // nsp,          # column tiles each core walks
        "grid": mt * nsp,
        "ub": 2 * 3 * bm * bn * 2,  # 3 buffers x 2 versions
    }


@functools.lru_cache(maxsize=16)
def build(M, N, dtype, bm, bn, nsp):
    geo = geometry(M, N, bm, bn, nsp)
    if geo["ub"] > UB_CAPACITY:
        raise ValueError(f"UB {geo['ub']} B > {UB_CAPACITY} B")
    grid, tpb, mt = geo["grid"], geo["tpb"], geo["MT"]
    nchunk = bn // VL

    @T.prim_func
    def main(
        gate_up: T.Tensor((M, 2 * N), dtype),  # type: ignore
        out: T.Tensor((M, N), dtype),  # type: ignore
    ) -> None:
        with T.Kernel(grid) as bx:
            im = bx // (nt_per_core := (geo["NT"] // tpb))
            sp = bx % nt_per_core
            g_ub = T.alloc_shared((bm, bn), dtype)
            u_ub = T.alloc_shared((bm, bn), dtype)
            y_ub = T.alloc_shared((bm, bn), dtype)
            T.annotate_buffer_versions({g_ub: 2, u_ub: 2, y_ub: 2})
            for it in T.Pipelined(tpb, num_stages=2):
                tn = sp * tpb + it
                m0 = im * bm
                n0 = tn * bn
                am = T.min(bm, M - m0)
                an = T.min(bn, N - n0)
                T.copy(gate_up[m0 : m0 + am, n0 : n0 + an], g_ub[0:am, 0:an])
                T.copy(gate_up[m0 : m0 + am, N + n0 : N + n0 + an], u_ub[0:am, 0:an])
                with T.SimdVF():
                    full = T.simd.pset(32)
                    for r in T.serial(bm):
                        for c in T.serial(nchunk):
                            g = T.simd.vcvt(
                                T.simd.vld(g_ub[r, c * VL], dist="UNPK_B16"),
                                "float32",
                                part=0,
                            )
                            u = T.simd.vcvt(
                                T.simd.vld(u_ub[r, c * VL], dist="UNPK_B16"),
                                "float32",
                                part=0,
                            )
                            num = T.simd.vmul(g, u, full)
                            den = T.simd.vadds(T.simd.vexp(T.simd.vneg(g, full), full), 1.0, full)
                            y = T.simd.vdiv(num, den, full)
                            T.simd.vsts(
                                y_ub[r, c * VL],
                                T.simd.vcvt(y, dtype, round="ROUND_R", sat=False, part=0),
                                full,
                                dist="PK_B32",
                            )
                T.copy(y_ub[0:am, 0:an], out[m0 : m0 + am, n0 : n0 + an])

    return tilelang.compile(main, target="ascend", out_idx=[1])


CASES = [
    # name, M, N, dtype, bm, bn, nsp
    ("P1-t4096-bf16", 4096, 2048, "bfloat16", 16, 512, 2),
    ("P2-t32-fp16", 32, 1536, "float16", 1, 256, 3),
    ("P3-tails-bf16", 1, 1500, "bfloat16", 1, 512, 3),
    ("P4-t512-bf16", 512, 18432, "bfloat16", 16, 512, 9),
]


def main():
    bad = 0
    for name, M, N, dtype, bm, bn, nsp in CASES:
        try:
            build(M, N, dtype, bm, bn, nsp)
            print(
                f"[{name}] M={M} N={N} {dtype} bm={bm} bn={bn} nsp={nsp} "
                f"geo={geometry(M, N, bm, bn, nsp)} COMPILE OK",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - the probe reports and continues
            bad += 1
            print(f"[{name}] COMPILE FAIL: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=6)
    print("PROBE_RESULT:", "ALL_OK" if bad == 0 else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
