"""Design-time probe (D-3): compile-only validation of the `_gemm_basic_kernel`
NPU structure written into DESIGN.md §3.3.

No device is touched (``ASCEND_RT_VISIBLE_DEVICES=""``): every variant is only
compiled through ``tilelang.compile(..., target="ascend")``, so the frontend and
lowering accept the exact API combination of the design.  Numerical correctness
is Stage 3's job (DESIGN.md §8.3).

Run (task dir, platform env sourced, card not required):

    ASCEND_RT_VISIBLE_DEVICES="" python history_version/design_probe_compile.py

Each case prints one JSON line: ``ok`` plus the key codegen facts (L1 version
ring stride, emitted GEMM template, L0C->GM quant mode) so DESIGN.md can cite
the mechanism, not only "it compiled".
"""

import json
import os
import re
import traceback

import tilelang
import tilelang.ascend.language as T

OUT = os.environ.get("PROBE_OUT", ".")


def build(m, n, k, trans_a=False, trans_b=True, dtype="bfloat16",
          has_bias=False, out_dtype=None, block_m=128, block_n=256, block_k=64,
          num_stages=3, split_k=1):
    """The designed program: NT-normalized L1-operand GEMM, fused quantized store.

    - ``trans_a``/``trans_b`` only select the *load* path (nd2nz vs dn2nz); the
      GEMM itself is always the backend's NT form (asserted in
      ``ascend/op/gemm/gemm_mad.py``).
    - Grid is 1-D; ``bx`` (n tile) is the slow axis and ``by`` (m tile) the fast
      one, so consecutive blocks share the A panel in L2 (§1.6.3 D4).
    """
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)
    c_dtype = out_dtype or dtype
    if split_k > 1:
        block_m = min(block_m, m)          # decode band: caller default 128 shrinks
        assert m % block_m == 0 and n % block_n == 0 and k % block_k == 0
        assert (k // block_k) % split_k == 0
    assert block_m % 16 == 0 and block_n % 16 == 0 and block_k % 16 == 0
    mt = -(-m // block_m)
    nt = -(-n // block_n)
    k_tiles = -(-k // block_k)
    k_slice = k_tiles // split_k
    out_shape = (m, n) if split_k == 1 else (split_k, m, n)
    out_dtype_ = c_dtype if split_k == 1 else "float32"
    exact_m = m % block_m == 0
    exact_n = n % block_n == 0
    exact_k = k % block_k == 0

    @T.prim_func
    def main(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        bias: T.Tensor((n,), dtype),
        C: T.Tensor(out_shape, out_dtype_),
    ):
        with T.Kernel(nt * mt * split_k) as bid:
            bx = bid // (mt * split_k)          # n tile: slow axis
            by = bid % (mt * split_k) // split_k  # m tile: fast axis
            bz = bid % split_k                  # split-K slice: fastest
            m0 = by * block_m
            n0 = bx * block_n
            am = block_m if exact_m else T.min(m - m0, block_m)
            an = block_n if exact_n else T.min(n - n0, block_n)

            a1 = T.alloc_l1((block_m, block_k), dtype)
            b1 = T.alloc_l1((block_n, block_k), dtype)
            c0 = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions({a1: num_stages, b1: num_stages})

            for step in T.Pipelined(k_slice, num_stages=num_stages):
                k0 = (bz * k_slice + step) * block_k
                ak = block_k if exact_k else T.min(k - k0, block_k)
                # NT normalization: A -> (M,K) tile, B -> (N,K) tile in L1.
                if trans_a:
                    T.copy(A[k0 : k0 + ak, m0 : m0 + am], a1[:am, :ak], transpose=True)
                else:
                    T.copy(A[m0 : m0 + am, k0 : k0 + ak], a1[:am, :ak])
                if trans_b:
                    T.copy(B[n0 : n0 + an, k0 : k0 + ak], b1[:an, :ak])
                else:
                    T.copy(B[k0 : k0 + ak, n0 : n0 + an], b1[:an, :ak], transpose=True)
                T.gemm(a1, b1, c0, transpose_B=True, clear_accum=step == 0)

            # Epilogue.  has_bias takes the Cube+Vector path; otherwise the
            # fixpipe store quantizes fp32 -> c_dtype in hardware.
            if has_bias:
                cv = T.alloc_shared((block_m // 2, block_n), "float32")
                co = T.alloc_shared((block_m // 2, block_n), dtype)
                bv = T.alloc_shared((block_n,), dtype)
                T.copy(bias[n0 : n0 + block_n], bv)
                T.dual_copy(c0, cv)
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(block_m // 2, block_n):
                        co[i, j] = T.cast(cv[i, j] + T.cast(bv[j], "float32"), dtype)
                T.dual_copy(co, C[m0 : m0 + block_m, n0 : n0 + block_n])
            elif split_k == 1:
                if exact_m and exact_n:
                    T.copy(c0, C[m0 : m0 + block_m, n0 : n0 + block_n])
                else:
                    T.copy(c0[:am, :an], C[m0 : m0 + am, n0 : n0 + an])
            else:
                T.copy(c0[:am, :an], C[bz, m0 : m0 + am, n0 : n0 + an])

    return main


# (name, kwargs) -- one row per design decision that must be compile-real.
CASES = [
    # primary: prefill, B stored (N,K) -> direct L1 load, fused quantized store
    ("D1-prefill-t4096-NT-direct-bf16", dict(m=4096, n=4096, k=7168)),
    # primary with fp16 element type
    ("D2-prefill-fp16", dict(m=512, n=36864, k=7168, dtype="float16")),
    # all four (trans_a, trans_b) load layouts
    ("D3-TN-transA-transB", dict(m=512, n=4096, k=7168, trans_a=True, trans_b=True)),
    ("D4-NN-transB-false", dict(m=512, n=4096, k=7168, trans_b=False)),
    ("D5-TT-transA-transB-false", dict(m=512, n=4096, k=7168, trans_a=True, trans_b=False)),
    # decode band, split-K: partials fp32 [split_k, m, n], exact tiles
    ("D6-splitk4-m32", dict(m=32, n=7168, k=2048, block_m=32, split_k=4)),
    ("D7-splitk4-clamped-bm128-to-m32", dict(m=32, n=4096, k=7168, split_k=4)),
    # m=1: split_k impossible (m % block_m != 0), M-tail path, block_m=16
    ("D8-m1-tail-bm16", dict(m=1, n=36864, k=7168, block_m=16)),
    # tail on m and n simultaneously (general factory contract)
    ("D9-tail-mn", dict(m=100, n=1000, k=7168, block_m=32, block_n=128)),
    # has_bias path: Cube + Vector epilogue with bias broadcast
    ("D10-has-bias", dict(m=512, n=4096, k=7168, has_bias=True)),
    # stage depth knob
    ("D11-ns4-decode-bm32", dict(m=32, n=7168, k=2048, block_m=32, num_stages=4)),
    # out_dtype = fp32 (no quant), split_k == 1
    ("D12-out-fp32", dict(m=512, n=4096, k=7168, out_dtype="float32")),
]


def facts(source):
    """Codegen facts worth citing in DESIGN.md."""
    return {
        "l1_ring": sorted({(int(a), int(b)) for a, b in
                           re.findall(r"buf_dyn_l1 \+ \(\(step % (\d+)\) \* (\d+)\)", source)}),
        "gemm": (re.findall(r"ascend_gemm_l1<[^>]*>", source) or [None])[0],
        "quant_mode": (re.findall(r"asc_quant_mode>\((\d+)\)", source) or [None]),
        "dual_dst": (re.findall(r"asc_dual_dst_mode>\((\d+)\)", source) or [None]),
        "l0c2gm": "asc_copy_l0c2gm" in source,
        "l0c2ub": "asc_copy_l0c2ub" in source,
        "kernel_kind": (re.findall(r"__global__ (__cube__|__mix__\([^)]*\)|__vector__)", source) or [None])[0],
        "nd2nz": source.count("asc_copy_gm2l1_nd2nz"),
        "dn2nz": source.count("asc_copy_gm2l1_dn2nz"),
        "chars": len(source),
    }


def main():
    results = []
    for name, kw in CASES:
        rec = {"probe": name}
        try:
            kernel = tilelang.compile(build(**kw), target="ascend", out_idx=-1)
            src = kernel.get_kernel_source()
            rec["ok"] = True
            rec.update(facts(src))
            with open(os.path.join(OUT, f"probe_src_{name}.cce"), "w") as fh:
                fh.write(src)
        except Exception as exc:  # noqa: BLE001 - the probe reports, never raises
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}".splitlines()[0][:400]
            with open(os.path.join(OUT, f"probe_fail_{name}.log"), "w") as fh:
                fh.write(traceback.format_exc())
        results.append(rec)
        print(json.dumps(rec), flush=True)
    ok = sum(1 for r in results if r["ok"])
    print(json.dumps({"summary": f"{ok}/{len(results)} compiled"}), flush=True)


if __name__ == "__main__":
    main()
