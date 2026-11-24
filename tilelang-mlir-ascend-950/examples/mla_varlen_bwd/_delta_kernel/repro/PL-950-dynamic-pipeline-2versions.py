"""PL-950-dynamic-pipeline-2versions. First verified: TileLang ascend 0.1.15 / CANN 9.2.1 / ascend-950pr-9579.

Claim under test: a ``T.Pipelined(it0, it1, num_stages=2)`` loop whose *bounds are
run-time expressions* still gets two versions of each ``alloc_shared`` buffer
(``annotate_buffer_versions`` honoured), and a tile inside that loop can carry a
run-time extent ``n = min(R, total - t0)`` on the two loads and on the
write-back.

Why this is worth a regression test: on the dense sibling
(``examples/mla_dense_bwd/_delta_kernel``) an integer-tail segment written
*outside* the pipelined loop made the compiler drop multi-buffering with the
warning ``did not enable multi-buffering for buffer "o_ub"``.  The packed form has
no such segment (there is no static tail when the row count is dynamic), and the
question "does the dynamic loop bound cost the second version?" is answered here:
it does not.

Assertions
  1. the compiled device code contains the two-version offset ``(it - it0) & 1``
     applied to both operand buffers (stride 8192 half = one (64, 128) tile),
  2. both load descriptors and the write-back descriptor carry the run-time count
     ``min(64, total_q - t0)`` instead of a literal ``64``,
  3. the write-back base is head-major, i.e. it multiplies the head index by the
     run-time ``total_q``,
  4. on device, ragged row counts (65 and 200 rows, i.e. ``n = 1`` and ``n = 8``
     on the last tile) match the float32 torch reference bit-for-bit within
     tolerance, which is what a broken/absent clamp or a single-versioned buffer
     would violate.

Run (device required for assertion 4):

    source .agents/platforms/toolchains/ascend/env-950pr.sh
    cd /tmp && /workspace/venv-ascend950/bin/python \
        /workspace/tilelang-mlir-ascend-950/examples/mla_varlen_bwd/_delta_kernel/repro/PL-950-dynamic-pipeline-2versions.py
"""

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

VL = 128
R = 64
G = 56
DT = "bfloat16"


def _row_dot(o_ub, do_ub, i, nvec):
    part = None
    for v in range(nvec):
        vo = T.simd.vld(o_ub[i, v * VL])
        vd = T.simd.vld(do_ub[i, v * VL])
        cur = T.simd.vadd(
            T.simd.vmul(
                T.simd.vcvt(vo, "float32", part=0), T.simd.vcvt(vd, "float32", part=0)
            ),
            T.simd.vmul(
                T.simd.vcvt(vo, "float32", part=1), T.simd.vcvt(vd, "float32", part=1)
            ),
        )
        part = cur if part is None else T.simd.vadd(part, cur)
    return T.simd.vcadd(part)


def build(heads=8, dim_v=128, dtype=DT):
    dw = ((dim_v + VL - 1) // VL) * VL
    nvec = dw // VL
    total = T.dynamic("total_q")

    @T.prim_func
    def main(
        o: T.Tensor((total, heads, dim_v), dtype),  # type: ignore
        do: T.Tensor((total, heads, dim_v), dtype),  # type: ignore
        delta: T.Tensor((heads, total), "float32"),  # type: ignore
    ):
        ntiles = T.ceildiv(total, R) * heads
        with T.Kernel(G) as bx:
            o_ub = T.alloc_shared((R, dw), dtype)
            do_ub = T.alloc_shared((R, dw), dtype)
            d_ub = T.alloc_shared((R, 8), "float32")
            T.annotate_buffer_versions({o_ub: 2, do_ub: 2})
            it0 = (ntiles * bx) // G
            it1 = (ntiles * (bx + 1)) // G
            for it in T.Pipelined(it0, it1, num_stages=2):
                h = it % heads
                t0 = (it // heads) * R
                n = T.min(R, total - t0)
                T.copy(o[t0 : t0 + n, h, :], o_ub[0:n, 0:dim_v])
                T.copy(do[t0 : t0 + n, h, :], do_ub[0:n, 0:dim_v])
                with T.SimdVF():
                    one = T.simd.pset(32, "PAT_VL1")
                    for i in T.serial(R):
                        T.simd.vsts(d_ub[i, 0], _row_dot(o_ub, do_ub, i, nvec), one, extent=1)
                T.copy(d_ub[0:n, 0:1], delta[h, t0 : t0 + n])

    return tilelang.compile(main, target="ascend", out_idx=[2])


def check_source(src):
    loads = [ln for ln in src.splitlines() if "asc_copy_gm2ub_align" in ln]
    stores = [ln for ln in src.splitlines() if "asc_copy_ub2gm_align" in ln]
    assert len(loads) == 2 and len(stores) == 1, (len(loads), len(stores))
    # 1. two versions of both operand buffers (8192 half = one (64,128) 16-bit tile)
    assert loads[0].count("& 1) * 8192") == 1 and loads[1].count("& 1) * 8192") == 1, (
        "operand buffers are not double buffered under the dynamic loop bound"
    )
    assert "+ 16384" in loads[1], "the second operand does not sit one version later"
    # 2. run-time count on all three descriptors
    for ln in loads + stores:
        assert "min(64, (total_q -" in ln, f"descriptor lost its run-time clamp: {ln[:160]}"
        assert ", 64, 4," not in ln, "write-back still uses a static 64-element count"
    # 3. head-major base: the head index multiplies the run-time row count
    assert "* ((int64_t)total_q)" in stores[0], "write-back is not head-major"
    print("[OK] 2 versions + 3 dynamic extents + head-major base", flush=True)


def check_device(heads=8, dim_v=128, dtype=DT):
    fn = build(heads, dim_v, dtype)
    for total in (65, 200):
        gen = torch.Generator().manual_seed(total)
        o = torch.randn((total, heads, dim_v), generator=gen, dtype=torch.float32).to(
            getattr(torch, dtype)
        )
        do = torch.randn((total, heads, dim_v), generator=gen, dtype=torch.float32).to(
            getattr(torch, dtype)
        )
        gold = (o.float() * do.float()).sum(-1).transpose(0, 1).contiguous()
        got = fn(o.to("npu"), do.to("npu")).cpu()
        torch.testing.assert_close(got, gold, atol=1e-4, rtol=1e-4)
        print(
            f"[OK] T_q={total} last_n={total - (total // R) * R} "
            f"max_diff={(got - gold).abs().max().item():.3e}",
            flush=True,
        )


kernel = build()
check_source(kernel.get_kernel_source())
check_device()
print("[PRECISION_PASS]", flush=True)
