"""Design-time compile probe (D-3) for `_engram_step_kernel` on Ascend950PR_9579.

Compile-only: no NPU tensor is created and no kernel is executed, so the device
is not occupied.  Run from the task directory:

    source .agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        history_version/design_probe_step.py

Variants (each answers one open question, see DESIGN.md §11.1):

  S1  fused structure: 4 concurrent reducer epochs in ONE row pass +
      scalar hoisting (dot = rrms_h·rrms_k·Σ h k w²) + variance hoisting
      (Σ(αv)² = α²Σv²) + one fused phase-2 pass (vn write + dilated conv +
      SiLU + residual), taps read straight from UB.
  S2  conservative structure (mirror of the 950DT reference): one pass per
      statistic, 4 sequential epochs, per-tap passes.
  S3  S1 with the masked (`i < d_pad`) variant and tile = next_pow2 > d_pad,
      i.e. the non-power-of-two padded width path.
"""

import math
import sys
import traceback

import tilelang
import tilelang.ascend.language as T


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _marks(compiled):
    src = ""
    for attr in ("get_kernel_source", "kernel_source"):
        try:
            src = getattr(compiled, attr)
            src = src() if callable(src) else src
            if src:
                break
        except Exception:  # noqa: BLE001 - debug aid only
            continue
    marks = [
        t
        for t in (
            "__mix__",
            "__aicore__",
            "asc_copy_gm2ub",
            "asc_copy_ub2gm",
            "asc_copy_gm2l1",
            "asc_mmad",
            "asc_set_vector_mask",
        )
        if t in src
    ]
    return marks, len(src)


# ---------------------------------------------------------------------------
# S1: fused structure (candidate main design)
# ---------------------------------------------------------------------------
def build_s1(batch=4, d=512, d_pad=512, max_conv_len=20, w=4, dilation=5,
             eps=1e-6, dtype_str="float16", threads=512):
    tile = _next_pow2(d_pad)
    exact = tile == d_pad
    inv_d = 1.0 / float(d)
    inv_sqrt_d = 1.0 / math.sqrt(float(d))

    @T.prim_func
    def engram_decode_step(
        h_t: T.Tensor((batch, d_pad), dtype_str),
        kv: T.Tensor((2, batch, d_pad), "float32"),
        conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype_str),
        rms_w_h: T.Tensor((d_pad,), dtype_str),
        rms_w_v: T.Tensor((d_pad,), dtype_str),
        conv_w: T.Tensor((w, d_pad), dtype_str),
        y_t: T.Tensor((batch, d_pad), dtype_str),
        new_conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype_str),
    ):
        with T.Kernel(batch) as bx:
            h_ub = T.alloc_shared((d_pad,), dtype_str)
            k_ub = T.alloc_shared((d_pad,), "float32")
            v_ub = T.alloc_shared((d_pad,), "float32")
            wh_ub = T.alloc_shared((d_pad,), dtype_str)
            wv_ub = T.alloc_shared((d_pad,), dtype_str)
            cw_ub = T.alloc_shared((w, d_pad), dtype_str)
            # zero-extent allocation when conv_kernel_size == 1: legal, never read
            tap_ub = T.alloc_shared((w - 1, d_pad), dtype_str)
            y_ub = T.alloc_shared((d_pad,), dtype_str)
            vn_ub = T.alloc_shared((d_pad,), dtype_str)

            T.copy(h_t[bx, :], h_ub)
            T.copy(kv[0, bx, :], k_ub)
            T.copy(kv[1, bx, :], v_ub)
            T.copy(rms_w_h, wh_ub)
            T.copy(rms_w_v, wv_ub)
            T.copy(conv_w, cw_ub)
            for p in range(w - 1):
                T.copy(
                    conv_state[bx, max_conv_len - (w - 1 - p) * dilation, :],
                    tap_ub[p, :],
                )

            with T.SimtVF(threads=threads):
                vf = T.alloc_fragment((tile,), "float32")
                vnf = T.alloc_fragment((tile,), "float32")
                cof = T.alloc_fragment((tile,), "float32")

                acc_h = T.alloc_reducer((1,), "float32", op="sum")
                acc_k = T.alloc_reducer((1,), "float32", op="sum")
                acc_p = T.alloc_reducer((1,), "float32", op="sum")
                acc_v = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(acc_h)
                T.reducer_init(acc_k)
                T.reducer_init(acc_p)
                T.reducer_init(acc_v)

                if exact:
                    for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                        vf[i] = v_ub[i]
                        T.reducer_update(
                            acc_h[0],
                            T.cast(h_ub[i], "float32") * T.cast(h_ub[i], "float32"),
                        )
                        T.reducer_update(acc_k[0], k_ub[i] * k_ub[i])
                        T.reducer_update(
                            acc_p[0],
                            T.cast(h_ub[i], "float32")
                            * k_ub[i]
                            * T.cast(wh_ub[i], "float32")
                            * T.cast(wh_ub[i], "float32"),
                        )
                        T.reducer_update(acc_v[0], v_ub[i] * v_ub[i])
                else:
                    for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                        hv = T.if_then_else(
                            i < d_pad, T.cast(h_ub[i], "float32"), T.float32(0)
                        )
                        kk = T.if_then_else(i < d_pad, k_ub[i], T.float32(0))
                        vv = T.if_then_else(i < d_pad, v_ub[i], T.float32(0))
                        ww = T.if_then_else(
                            i < d_pad, T.cast(wh_ub[i], "float32"), T.float32(0)
                        )
                        vf[i] = vv
                        T.reducer_update(acc_h[0], hv * hv)
                        T.reducer_update(acc_k[0], kk * kk)
                        T.reducer_update(acc_p[0], hv * kk * ww * ww)
                        T.reducer_update(acc_v[0], vv * vv)

                sh = T.alloc_fragment((1,), "float32")
                sk = T.alloc_fragment((1,), "float32")
                sp = T.alloc_fragment((1,), "float32")
                sv = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc_h, sh)
                T.finalize_reducer(acc_k, sk)
                T.finalize_reducer(acc_p, sp)
                T.finalize_reducer(acc_v, sv)

                alpha = T.alloc_fragment((1,), "float32")
                rv = T.alloc_fragment((1,), "float32")
                for i in T.Parallel(1):
                    alpha[i] = 1.0 / (
                        1.0
                        + T.exp(
                            -(T.rsqrt(sh[0] * inv_d + eps) * T.rsqrt(sk[0] * inv_d + eps)
                              * sp[0] * inv_sqrt_d)
                        )
                    )
                    rv[i] = T.rsqrt(alpha[0] * alpha[0] * sv[0] * inv_d + eps)

                if exact:
                    for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                        vnf[i] = alpha[0] * vf[i] * rv[0] * T.cast(wv_ub[i], "float32")
                        cof[i] = T.cast(cw_ub[w - 1, i], "float32") * vnf[i]
                        for p in range(w - 1):
                            cof[i] = cof[i] + T.cast(cw_ub[p, i], "float32") * T.cast(
                                tap_ub[p, i], "float32"
                            )
                        y_ub[i] = T.cast(
                            cof[i] / (1.0 + T.exp(-cof[i])) + alpha[0] * vf[i],
                            dtype_str,
                        )
                        vn_ub[i] = T.cast(vnf[i], dtype_str)
                else:
                    for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                        if i < d_pad:
                            vnf[i] = alpha[0] * vf[i] * rv[0] * T.cast(
                                wv_ub[i], "float32"
                            )
                            cof[i] = T.cast(cw_ub[w - 1, i], "float32") * vnf[i]
                            for p in range(w - 1):
                                cof[i] = cof[i] + T.cast(
                                    cw_ub[p, i], "float32"
                                ) * T.cast(tap_ub[p, i], "float32")
                            y_ub[i] = T.cast(
                                cof[i] / (1.0 + T.exp(-cof[i])) + alpha[0] * vf[i],
                                dtype_str,
                            )
                            vn_ub[i] = T.cast(vnf[i], dtype_str)

            T.copy(y_ub, y_t[bx, :])
            T.copy(vn_ub, new_conv_state[bx, max_conv_len - 1, :])

    return tilelang.compile(engram_decode_step, target="ascend", out_idx=[])


# ---------------------------------------------------------------------------
# S2: conservative structure (950DT-reference-like, per-statistic passes)
# ---------------------------------------------------------------------------
def build_s2(batch=4, d=512, d_pad=512, max_conv_len=20, w=4, dilation=5,
             eps=1e-6, dtype_str="float16", threads=512):
    tile = _next_pow2(d_pad)

    @T.prim_func
    def engram_decode_step_s2(
        h_t: T.Tensor((batch, d_pad), dtype_str),
        kv: T.Tensor((2, batch, d_pad), "float32"),
        conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype_str),
        rms_w_h: T.Tensor((d_pad,), dtype_str),
        rms_w_v: T.Tensor((d_pad,), dtype_str),
        conv_w: T.Tensor((w, d_pad), dtype_str),
        y_t: T.Tensor((batch, d_pad), dtype_str),
        new_conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype_str),
    ):
        with T.Kernel(batch) as bx:
            h_ub = T.alloc_shared((d_pad,), dtype_str)
            k_ub = T.alloc_shared((d_pad,), "float32")
            v_ub = T.alloc_shared((d_pad,), "float32")
            wh_ub = T.alloc_shared((d_pad,), dtype_str)
            wv_ub = T.alloc_shared((d_pad,), dtype_str)
            cw_ub = T.alloc_shared((w, d_pad), dtype_str)
            tap_ub = T.alloc_shared((w - 1, d_pad), dtype_str)
            y_ub = T.alloc_shared((d_pad,), dtype_str)
            vn_ub = T.alloc_shared((d_pad,), dtype_str)
            T.copy(h_t[bx, :], h_ub)
            T.copy(kv[0, bx, :], k_ub)
            T.copy(kv[1, bx, :], v_ub)
            T.copy(rms_w_h, wh_ub)
            T.copy(rms_w_v, wv_ub)
            T.copy(conv_w, cw_ub)
            for p in range(w - 1):
                T.copy(
                    conv_state[bx, max_conv_len - (w - 1 - p) * dilation, :],
                    tap_ub[p, :],
                )
            with T.SimtVF(threads=threads):
                hf = T.alloc_fragment((tile,), "float32")
                kf = T.alloc_fragment((tile,), "float32")
                vf = T.alloc_fragment((tile,), "float32")
                wf = T.alloc_fragment((tile,), "float32")
                co = T.alloc_fragment((tile,), "float32")
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    hf[i] = T.if_then_else(i < d_pad, T.cast(h_ub[i], "float32"), T.float32(0))
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    kf[i] = T.if_then_else(i < d_pad, k_ub[i], T.float32(0))
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    vf[i] = T.if_then_else(i < d_pad, v_ub[i], T.float32(0))
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    wf[i] = T.if_then_else(i < d_pad, T.cast(wh_ub[i], "float32"), T.float32(0))

                acc = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(acc)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    T.reducer_update(acc[0], hf[i] * hf[i])
                s = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc, s)
                for i in T.Parallel(1):
                    s[i] = T.rsqrt(s[i] / d + eps)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    hf[i] = hf[i] * s[0] * wf[i]

                acc_k = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(acc_k)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    T.reducer_update(acc_k[0], kf[i] * kf[i])
                sk = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc_k, sk)
                for i in T.Parallel(1):
                    sk[i] = T.rsqrt(sk[i] / d + eps)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    kf[i] = kf[i] * sk[0] * wf[i]

                acc_d = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(acc_d)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    T.reducer_update(acc_d[0], hf[i] * kf[i])
                sd = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc_d, sd)
                alpha = T.alloc_fragment((1,), "float32")
                for i in T.Parallel(1):
                    alpha[i] = 1.0 / (1.0 + T.exp(-(sd[i] / T.sqrt(T.float32(d)))))

                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    vf[i] = alpha[0] * vf[i]
                sq = T.alloc_fragment((tile,), "float32")
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    sq[i] = T.if_then_else(i < d_pad, vf[i] * vf[i], T.float32(0))
                acc_v = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(acc_v)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    T.reducer_update(acc_v[0], sq[i])
                sv = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc_v, sv)
                for i in T.Parallel(1):
                    sv[i] = T.rsqrt(sv[i] / d + eps)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    wf[i] = T.if_then_else(i < d_pad, T.cast(wv_ub[i], "float32"), T.float32(0))
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    sq[i] = vf[i] * sv[0] * wf[i]
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    if i < d_pad:
                        vn_ub[i] = T.cast(sq[i], dtype_str)

                tapf = T.alloc_fragment((tile,), "float32")
                cwf = T.alloc_fragment((tile,), "float32")
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    cwf[i] = T.if_then_else(i < d_pad, T.cast(cw_ub[w - 1, i], "float32"), T.float32(0))
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    co[i] = cwf[i] * sq[i]
                for p in range(w - 1):
                    for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                        tapf[i] = T.if_then_else(i < d_pad, T.cast(tap_ub[p, i], "float32"), T.float32(0))
                    for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                        cwf[i] = T.if_then_else(i < d_pad, T.cast(cw_ub[p, i], "float32"), T.float32(0))
                    for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                        co[i] = co[i] + cwf[i] * tapf[i]
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    if i < d_pad:
                        y_ub[i] = T.cast(co[i] / (1.0 + T.exp(-co[i])) + vf[i], dtype_str)
            T.copy(y_ub, y_t[bx, :])
            T.copy(vn_ub, new_conv_state[bx, max_conv_len - 1, :])

    return tilelang.compile(engram_decode_step_s2, target="ascend", out_idx=[])


CASES = [
    ("S1 serving-b4   b=4 d_pad=512 mcl=20 w=4 dl=5 fp16", build_s1,
     dict(batch=4, d=512, d_pad=512, max_conv_len=20, w=4, dilation=5,
          dtype_str="float16", threads=512)),
    ("S1 decode-s1    b=1 d_pad=256 mcl=9  w=4 dl=3 fp16", build_s1,
     dict(batch=1, d=256, d_pad=256, max_conv_len=9, w=4, dilation=3,
          dtype_str="float16", threads=256)),
    ("S1 serving-b8   b=8 d_pad=256 mcl=18 w=4 dl=3 bf16", build_s1,
     dict(batch=8, d=256, d_pad=256, max_conv_len=18, w=4, dilation=3,
          dtype_str="bfloat16", threads=256)),
    ("S1 masked tail  b=2 d=600 d_pad=768 mcl=12 fp16 (tile=1024>d_pad)", build_s1,
     dict(batch=2, d=600, d_pad=768, max_conv_len=12, w=4, dilation=3,
          dtype_str="float16", threads=512)),
    ("S1 threads=128  b=4 d_pad=512 mcl=20 fp16", build_s1,
     dict(batch=4, d=512, d_pad=512, max_conv_len=20, w=4, dilation=5,
          dtype_str="float16", threads=128)),
    ("S1 w=1 mcl=1   b=1 d_pad=256 mcl=1  w=1 dl=1 fp16 (fixture row 4)", build_s1,
     dict(batch=1, d=256, d_pad=256, max_conv_len=1, w=1, dilation=1,
          dtype_str="float16", threads=256)),
    ("S2 conservative b=4 d_pad=512 mcl=20 fp16", build_s2,
     dict(batch=4, d=512, d_pad=512, max_conv_len=20, w=4, dilation=5,
          dtype_str="float16", threads=512)),
]


def main() -> int:
    failures = 0
    for label, builder, kwargs in CASES:
        print(f"\n=== {label}", flush=True)
        try:
            compiled = builder(**kwargs)
        except Exception:  # noqa: BLE001 - probe reports every failure
            failures += 1
            print("  COMPILE FAIL")
            traceback.print_exc(limit=4)
            continue
        marks, n = _marks(compiled)
        print(f"  COMPILE OK  marks={marks}  source_len={n}")
    print(f"\nRESULT: {len(CASES) - failures}/{len(CASES)} compiled")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
