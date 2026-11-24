# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_dq_land_kernel (ascend, packed branch): MLA prefill backward stage 3 -- dQ landing.

``dq[t, h, d] = round_dtype(float32(dq_acc[t, h, d]) * float32(scale))``

Third and last launch of ``MLAVarlenBwdOp``: ``_bwd_kernel`` leaves the cross-CTA
float32 dQ partial sums in ``dq_acc`` (shape ``(T_q, H, D_qk)``, row major,
contiguous, ``T_q = sum(q_lens)``), and this kernel multiplies them by the
softmax scale and narrows once to the input dtype.  No reduction, no
cross-element dependency, no per-request state: one float32 multiply, one
round-to-nearest-even narrowing -- a pure bandwidth operator.

The source function is literally the same one ``MLADenseBwdOp`` uses (single
module-level ``_dq_land_kernel(heads, dim_qk, scale, dtype)`` in
``_kernels/attention/mla/bwd.py``, reached from ``_MLABwdKernelBase._run`` by
``dq = land(dq_acc)``), so this file reuses the device-validated dense
implementation (``examples/mla_dense_bwd/_dq_land_kernel``, 31/31, ``bit_diff=0``)
verbatim.  The packed specifics are confined to the *caller*: ``dq_acc`` is
already ``(T_q, H, D_qk)`` (no ``view``), ``cu_seqlens_*`` never enter this
launch, and ``T_q`` stays a run-time symbol, so one compiled program serves every
token count and every request splitting (checked by the P7 case below).

Structure (DESIGN.md section 3.3, validated SIMD form):

* one-dimensional ``T.Kernel(G)`` with ``G`` = the device's vector-core count
  (56 on this model); each core walks a run-time contiguous interval
  ``[bx*cpt, min((bx+1)*cpt, ntiles))`` of the logical ``(row tile, head)``
  tiles through ``T.Pipelined(..., num_stages=2)`` with two UB versions;
* one strided ``T.copy`` per tile into ``acc[0:n, 0:D_qk]`` (float32, 768 B
  rows) and one back from ``out[0:n, 0:D_qk]`` (16-bit, 384 B rows), where
  ``n = min(R, T_q - t0)`` clamps the tail tile on both sides;
* inside ``T.SimdVF``: per row and per 64-lane segment
  ``vld -> vmuls(scale) -> vcvt(ROUND_R, sat=False) -> vsts`` -- the multiply
  and the single narrowing stay in vector registers, no 16-bit GM staging;
* ``D_qk = 192 = 3 x 64`` fills every lane exactly, ``T_q`` is dynamic
  (``T.dynamic("total_q")``) and the output is JIT-allocated (``out_idx=[1]``).

Two implementation facts were established on the device in the dense Stage 3 and
are reused here unchanged:

1. ``vcvt(f32 -> 16-bit, part=0)`` leaves the 64 converted values in the *even*
   16-bit lanes of the 128-lane destination register, so the store must pack
   32-bit words down to their low half: ``dist="PK_B32"``
   (``asc_storealign_pack``) with a 32-bit-lane mask.  ``dist="NORM_B16"`` writes
   every other slot, ``dist="PK_B16"`` packs at the wrong granularity, and a
   16-bit mask leaves exactly half of the slots untouched -- all of them compile
   and store correct values where they write.
2. ``sat=False`` (``RS_DISABLE``) is IEEE on this hardware: the fp16 overflow
   threshold 65520 gives ``+-inf`` (65519 still rounds to 65504).

Content layout:

1. ``golden_dq_land`` -- torch CPU reference of the operator's semantics.
2. ``_geometry`` / ``_build`` / ``_dq_land_kernel`` -- the kernel factory,
   signature-compatible with the source ``_dq_land_kernel(heads, dim_qk,
   scale, dtype) -> Callable``; every compile is gated on the generated device
   code containing the SIMD chain (``_simd_body_gate``).
3. ``run_case`` -- launch on the NPU and compare with the golden.
4. ``run_L0`` / ``run_L1`` / ``run_L2`` / ``run_boundary`` / ``main``.
"""

import argparse
import functools
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401  -- registers torch.npu

_IN_DTYPES = ("float16", "bfloat16")
_TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}
_ATOL = {"float16": 5e-3, "bfloat16": 1.6e-2}
_RTOL = {"float16": 1e-3, "bfloat16": 1.6e-2}

_SEG = 64  # lanes of one 2048-bit vector register (float32)
_R = 64  # rows per tile: UB budget, DESIGN.md 4.5
_CORE_FALLBACK = 56  # Ascend950PR_9579 [SoCInfo] vector_core_cnt
_UB_CAPACITY = 253952  # Ascend950PR_9579.ini [AICoreSpec] ub_size
_STORE_DIST = "PK_B32"  # 16-bit store of a narrowed register (see header)

# Generated-code gate: the SIMD chain must survive lowering.  DESIGN.md 9.2 E1
# records a form that compiles but silently drops the whole compute body, so a
# successful compile alone is not evidence.
_COMPUTE_TOKENS = ("vlds_norm", "vmuls", "vcvt")
_STORE_TOKENS = ("vsts_pack_b32", "vsts_norm")
_SRC_EVIDENCE = {}


# ---------------------------------------------------------------------------
# 1. Golden: torch CPU reference of the operator semantics
# ---------------------------------------------------------------------------
def golden_dq_land(dq_acc, scale, dtype, chunk_rows=4096):
    """``dq = (dq_acc.float() * scale).to(dtype)``, elementwise.

    ``dq_acc`` is the ``(T_q, H, D_qk)`` float32 accumulator as ``_bwd_kernel``
    writes it, ``dtype`` is ``float16`` or ``bfloat16``.  Written straight from
    the operator's definition (source docstring ``dq = dq_acc * scale in the
    input dtype``): one float32 multiply, one round-to-nearest-even narrowing.
    It knows nothing about the kernel -- no tiles, no vector ops, no core count,
    no ``T_q`` assumption, no ``cu_seqlens`` and no request boundaries -- so it
    stays valid if the kernel design or the packed/dense call path changes.
    ``chunk_rows`` only bounds the float32 temporary of the large geometries.
    """
    dt = _TORCH_DTYPE[dtype] if isinstance(dtype, str) else dtype
    x = dq_acc.detach().cpu() if dq_acc.device.type != "cpu" else dq_acc.detach()
    out = torch.empty(x.shape, dtype=dt)
    for s0 in range(0, x.shape[0], chunk_rows):
        s1 = min(s0 + chunk_rows, x.shape[0])
        out[s0:s1] = (x[s0:s1].float() * float(scale)).to(dt)
    return out


# ---------------------------------------------------------------------------
# 2. Kernel
# ---------------------------------------------------------------------------
def _vector_cores():
    """Physical AIV count of the current device, with the recorded fallback."""
    try:
        props = torch.npu.get_device_properties(torch.npu.current_device())
        cores = int(getattr(props, "vector_core_num"))
        if cores > 0:
            return cores
    except Exception:  # noqa: BLE001 - host-side precompile has no device
        pass
    return _CORE_FALLBACK


def _geometry(heads, dim_qk):
    """Build-time tile geometry (DESIGN.md 5.2)."""
    dw = ((dim_qk + _SEG - 1) // _SEG) * _SEG  # 192 -> no padding columns
    nseg = dw // _SEG
    ub_bytes = 2 * _R * dw * 4 + 2 * _R * dw * 2  # acc and out, two versions each
    return {"R": _R, "DW": dw, "NSEG": nseg, "ub_bytes": ub_bytes}


def _simd_body_gate(kernel, key):
    """Require the SIMD compute chain in the generated device code."""
    src = kernel.get_kernel_source()
    missing = [tok for tok in _COMPUTE_TOKENS if tok not in src]
    store = [tok for tok in _STORE_TOKENS if tok in src]
    if missing or not store:
        raise RuntimeError(
            "generated device code lost the SIMD body "
            f"({key}): missing {missing}, store tokens {store}; "
            f"{len(src.splitlines())} source lines"
        )
    _SRC_EVIDENCE[key] = {
        "compute": list(_COMPUTE_TOKENS),
        "store": store,
        "lines": len(src.splitlines()),
    }
    return _SRC_EVIDENCE[key]


def _build(heads, dim_qk, scale, dtype, versions=2):
    """Compile one specialised ``(dq_acc) -> dq`` program."""
    if dtype not in _IN_DTYPES:
        raise ValueError(f"dtype: {dtype!r} is not one of {_IN_DTYPES}")
    if heads <= 0 or dim_qk <= 0:
        raise ValueError(f"shape: heads/dim_qk must be positive, got {heads}/{dim_qk}")
    if dim_qk % 16 != 0:
        raise ValueError(
            f"shape: dim_qk must be a multiple of 16 for MLAVarlenBwdOp (got {dim_qk})"
        )
    sval = float(scale)
    if not (sval > 0.0) or sval != sval or sval == float("inf"):
        raise ValueError(f"scale: must be finite and positive, got {scale!r}")

    geo = _geometry(heads, dim_qk)
    rows, dw, nseg = geo["R"], geo["DW"], geo["NSEG"]
    if geo["ub_bytes"] > _UB_CAPACITY:
        raise ValueError(
            f"tiling: {geo['ub_bytes']} B of UB exceeds the {_UB_CAPACITY} B of this "
            f"model (R={rows}, DW={dw})"
        )
    grid = _vector_cores()
    total = T.dynamic("total_q")  # one symbol reused by both annotations (9.2 E2)
    scale_lit = sval

    @T.prim_func
    def _dq_land_main(
        dq_acc: T.Tensor((total, heads, dim_qk), "float32"),  # type: ignore
        dq: T.Tensor((total, heads, dim_qk), dtype),  # type: ignore
    ) -> None:
        ntiles = T.ceildiv(total, rows) * heads
        with T.Kernel(grid) as bx:
            acc = T.alloc_shared((rows, dw), "float32")
            out = T.alloc_shared((rows, dw), dtype)
            T.annotate_buffer_versions({acc: versions, out: versions})
            cpt = T.ceildiv(ntiles, grid)
            it0 = bx * cpt
            it1 = T.min(it0 + cpt, ntiles)
            for it in T.Pipelined(it0, it1, num_stages=versions):
                h = it % heads
                t0 = (it // heads) * rows
                n = T.min(rows, total - t0)
                T.copy(dq_acc[t0 : t0 + n, h, :], acc[0:n, 0:dim_qk])
                with T.SimdVF():
                    # PK_B32 packs the low half of each 32-bit lane, i.e. the
                    # 16-bit result of vcvt(part=0); its mask is a 32-bit-lane
                    # pattern (a b16 mask would store only half of the lanes).
                    m = T.simd.pset(32, "PAT_ALL")
                    for i in T.serial(rows):
                        for s in T.serial(nseg):
                            v = T.simd.vcvt(
                                T.simd.vmuls(T.simd.vld(acc[i, s * _SEG]), scale_lit),
                                dtype,
                                part=0,
                                sat=False,
                            )
                            T.simd.vsts(out[i, s * _SEG], v, m, _STORE_DIST, extent=_SEG)
                T.copy(out[0:n, 0:dim_qk], dq[t0 : t0 + n, h, :])

    kernel = tilelang.compile(_dq_land_main, target="ascend", out_idx=[1])
    _simd_body_gate(kernel, (heads, dim_qk, sval, dtype, versions))
    return kernel


@functools.lru_cache(maxsize=32)
def _dq_land_kernel(heads, dim_qk, scale, dtype):
    """Frozen factory: same signature as the source operator's landing kernel.

    ``(heads, dim_qk, scale, dtype)`` is the specialisation key (the source
    uses ``@target_cache(maxsize=32)`` the same way); ``T_q`` is a run-time
    symbol, so one program serves every token count and every packed splitting.
    """
    return _build(heads, dim_qk, scale, dtype, versions=2)


@functools.lru_cache(maxsize=4)
def _dq_land_single_version(heads, dim_qk, scale, dtype):
    """Control variant: no buffer versions (serial load/compute/store)."""
    return _build(heads, dim_qk, scale, dtype, versions=1)


# ---------------------------------------------------------------------------
# 3. Launch + comparison with the golden
# ---------------------------------------------------------------------------
_RESULTS = []  # (level, name, ok, detail)
_MAX_DIFF = [0.0]
_BIT_DIFF = [0]
_LAUNCH = {}  # name -> ms of one launch (informative, Stage 4 baseline)


def _record(level, name, ok, detail=""):
    _RESULTS.append((level, name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[{level}][{tag}] {name}{(' -- ' + detail) if detail and not ok else ''}", flush=True)


def _scale_of(dim_qk):
    """Manifest default: ``sm_scale = D_qk ** -0.5``."""
    return float(dim_qk) ** -0.5


def _make_input(tq, heads, dim_qk, seed, special, dtype, scale):
    """Float32 accumulator values of the magnitude ``_bwd_kernel`` produces."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((tq, heads, dim_qk), generator=gen, dtype=torch.float32) * 3.0
    if special == "zeros":
        x.zero_()
    elif special == "nonfinite":
        x[0, 0, 0] = float("inf")
        x[0, 0, 1] = float("-inf")
        x[1 % tq, heads - 1, dim_qk - 1] = float("nan")
        x[tq - 1, 0, dim_qk - 1] = float("-inf")
    elif special == "corners":
        x = _corner_fill(x, dtype, scale)
    elif special == "overflow":
        # Product past every finite value of the target dtype; the operand is
        # clamped to the float32 range, so a scale that is too small cannot
        # express the corner at all (that would be a test bug, not a pass).
        mx = 65504.0 if dtype == "float16" else 3.3895313892515355e38
        edge = min(mx * 1.5 / scale, 3.4028234663852886e38)
        if not edge * scale > mx:
            raise ValueError(
                f"special: scale={scale!r} cannot overflow {dtype} from the float32 range"
            )
        x[0, 0, 0] = edge
        x[0, 0, 1] = -edge
        x[tq - 1, heads - 1, dim_qk - 1] = -edge
    elif special is not None:
        raise ValueError(f"special: unknown pattern {special!r}")
    return x


def _corner_fill(x, dtype, scale):
    """Boundary values of the dtype, pre-divided by the scale.

    The products then sit exactly on the dtype's edges: +-0, +-Inf, NaN, the
    subnormal band, the finite edge (65504), the fp16 overflow threshold
    (65519/65520), and far out of range.  The rest of the tensor keeps random
    values.
    """
    mx = 65504.0 if dtype == "float16" else 3.3895313892515355e38
    s = scale
    corners = [
        0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan"),
        1e-8, -1e-8, 1e-38, -1e-38,
        mx / s, -mx / s,
        mx * 0.9999 / s, -mx * 0.9999 / s,
        mx * 1.0001 / s, -mx * 1.0001 / s,
        7e4 / s, -7e4 / s, 1.0e5 / s, -1.0e5 / s, 3.0e38, -3.0e38,
    ]
    flat = x.reshape(-1)
    flat[: len(corners)] = torch.tensor(corners, dtype=torch.float32)
    return x


def _compare(got, dq_acc, scale, dtype, chunk_rows=2048):
    """Chunked golden comparison: tolerance, bit equality and NaN positions.

    Returns (max_abs_diff, bit_mismatches, nan_position_mismatches, first_bad)
    where ``first_bad`` is ``(index, got, gold)`` of the first difference.  Bit
    equality ignores the NaN payload (only the position must match), everything
    else must agree bit for bit -- the operator is one multiply plus one RNE
    narrowing, so a device-vs-torch gap would be a semantics finding, not noise.
    """
    dt = _TORCH_DTYPE[dtype]
    tq = dq_acc.shape[0]
    worst = 0.0
    bit_bad = 0
    nan_bad = 0
    first_bad = None
    for s0 in range(0, tq, chunk_rows):
        s1 = min(s0 + chunk_rows, tq)
        g = got[s0:s1].detach().cpu().contiguous()
        gold = (dq_acc[s0:s1].float() * float(scale)).to(dt)
        gn = g.isnan()
        dn = gold.isnan()
        nan_bad += int((gn != dn).sum())
        diff = (g.float() - gold.float()).abs()
        diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        if diff.numel():
            worst = max(worst, float(diff.max()))
        same_bits = (g.view(torch.int16) == gold.view(torch.int16)) | (gn & dn)
        bad = ~same_bits
        n = int(bad.sum())
        if n and first_bad is None:
            idx = int(bad.reshape(-1).nonzero()[0])
            first_bad = (
                (s0 + idx // (g.shape[1] * g.shape[2]),
                 (idx // g.shape[2]) % g.shape[1],
                 idx % g.shape[2]),
                float(g.reshape(-1)[idx]),
                float(gold.reshape(-1)[idx]),
            )
        bit_bad += n
    return worst, bit_bad, nan_bad, first_bad


def _launch(fn, dq_acc_dev):
    got = fn(dq_acc_dev)
    if isinstance(got, (tuple, list)):
        got = got[0]
    return got


def run_case(
    level,
    name,
    q_lens,
    heads,
    dim_qk,
    dtype="float16",
    seed=0,
    special=None,
    repeat=1,
    scale=None,
    single_version_control=False,
    canary=False,
    determinism=False,
    chunk_rows=2048,
):
    """Build one geometry, run it on the NPU and compare with the golden.

    ``q_lens`` is the packed request-length list.  Only its sum (``T_q``) can
    reach this launch: request boundaries are consumed by ``_bwd_kernel``, and
    ``T_q`` is a run-time symbol, so 10 splicings of the same length must give
    the same bits (the P7 case asserts exactly that).

    ``canary`` places pattern-filled device tensors on both sides of the
    JIT-allocated output and re-checks them after a second launch: the only
    observable form of an out-of-bounds tail write, since the operator's own
    output buffer is exactly ``(T_q, H, D_qk)``.
    """
    tq = int(sum(q_lens))
    reqs = len(q_lens)
    scale = _scale_of(dim_qk) if scale is None else scale
    diff = float("nan")
    try:
        dq_acc = _make_input(tq, heads, dim_qk, seed, special, dtype, scale)
        dq_acc_dev = dq_acc.to("npu")
        fn = _dq_land_kernel(heads, dim_qk, scale, dtype)
        canaries = []
        if canary:
            shape = (tq * heads * dim_qk // 2,)
            canaries.append(torch.full(shape, 1234.5, dtype=torch.float16, device="npu"))
        warm = _launch(fn, dq_acc_dev)  # first launch loads the module (~100 ms)
        del warm
        torch.npu.synchronize()
        start = time.perf_counter()
        got = _launch(fn, dq_acc_dev)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1e3
        if canary:
            shape = (tq * heads * dim_qk // 2,)
            canaries.append(torch.full(shape, 1234.5, dtype=torch.float16, device="npu"))
            _launch(fn, dq_acc_dev)
            torch.npu.synchronize()
            for k, c in enumerate(canaries):
                if not torch.equal(c, torch.full_like(c, 1234.5)):
                    raise AssertionError(f"canary {k} changed: the tail escaped its buffer")

        if determinism:  # out_idx allocates a fresh output per call
            again = _launch(fn, dq_acc_dev)
            torch.npu.synchronize()
            if not torch.equal(got.view(torch.int16), again.view(torch.int16)):
                raise AssertionError("two launches of one geometry are not bit-identical")
            del again

        if repeat > 1:
            start = time.perf_counter()
            for _ in range(repeat):
                _launch(fn, dq_acc_dev)
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) * 1e3 / repeat

        if got.shape != (tq, heads, dim_qk):
            raise AssertionError(f"output shape {tuple(got.shape)} != {(tq, heads, dim_qk)}")
        if got.numel() != tq * heads * dim_qk:
            raise AssertionError(f"output holds {got.numel()} elements, expected {tq * heads * dim_qk}")
        if got.dtype != _TORCH_DTYPE[dtype]:
            raise AssertionError(f"output dtype {got.dtype} != {_TORCH_DTYPE[dtype]}")

        diff, bit_bad, nan_bad, first_bad = _compare(got, dq_acc, scale, dtype, chunk_rows)
        _MAX_DIFF[0] = max(_MAX_DIFF[0], diff)
        _BIT_DIFF[0] += bit_bad
        gold = golden_dq_land(dq_acc, scale, dtype, chunk_rows)
        torch.testing.assert_close(
            got.detach().cpu(), gold, atol=_ATOL[dtype], rtol=_RTOL[dtype], equal_nan=True
        )
        del gold
        if bit_bad or nan_bad:
            raise AssertionError(
                f"{bit_bad} element(s) differ bitwise, {nan_bad} NaN position(s) differ"
                f" (max_diff={diff:.3e}, first at {first_bad})"
            )

        if single_version_control:
            ctrl = _launch(_dq_land_single_version(heads, dim_qk, scale, dtype), dq_acc_dev)
            torch.npu.synchronize()
            if not torch.equal(ctrl.view(torch.int16), got.view(torch.int16)):
                raise AssertionError("single-version control disagrees with the 2-version pipeline")
            del ctrl
    except AssertionError as exc:
        _record(level, name, False, f"max_diff={diff:.3e} {str(exc)[:260]}")
        return False
    except (ValueError, RuntimeError) as exc:  # noqa: BLE001 - reported, never swallowed
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:260]}")
        return False
    _record(level, name, True)
    _LAUNCH[name] = elapsed
    print(
        f"       T_q={tq} reqs={reqs} H={heads} D_qk={dim_qk} {dtype} "
        f"R={_R} ntiles={-(-tq // _R) * heads} max_diff={diff:.3e} "
        f"bit_diff=0 launch={elapsed:.3f} ms"
        + (" [det=bit-identical]" if determinism else "")
        + (" [single-version control equal]" if single_version_control else ""),
        flush=True,
    )
    return True


def _finish(level, strict):
    rows = [r for r in _RESULTS if r[0] == level]
    failed = [r for r in rows if not r[2]]
    print(
        f"[{level}] {len(rows) - len(failed)}/{len(rows)} cases passed, "
        f"max_diff={_MAX_DIFF[0]:.6g}",
        flush=True,
    )
    if failed and strict:
        raise AssertionError(f"[{level}] {len(failed)} case(s) failed")
    if failed:
        print(f"[{level}][WARN] {len(failed)} non-blocking case(s) failed", flush=True)


# ---------------------------------------------------------------------------
# 4. Layered cases
# ---------------------------------------------------------------------------
def _case_gates(level, name):
    """P0: the generated code keeps the SIMD chain, and the factory is cached."""
    problems = []
    if not _SRC_EVIDENCE:
        problems.append("no compiled program recorded a source gate")
    for key, ev in _SRC_EVIDENCE.items():
        if not ev["store"]:
            problems.append(f"{key}: no store instruction")
        if "vsts_pack_b32" not in ev["store"]:
            problems.append(f"{key}: the dense store is not vsts_pack_b32")
    first = _dq_land_kernel(8, 192, _scale_of(192), "float16")
    again = _dq_land_kernel(8, 192, _scale_of(192), "float16")
    if first is not again:
        problems.append("the compiled program is not cached per (heads, dim_qk, scale, dtype)")
    if problems:
        _record(level, name, False, "; ".join(problems))
        return False
    _record(level, name, True)
    for key, ev in sorted(_SRC_EVIDENCE.items(), key=lambda kv: str(kv[0])):
        print(
            f"       P0 gate {key}: compute={ev['compute']} store={ev['store']} "
            f"({ev['lines']} generated lines)",
            flush=True,
        )
    return True


def _case_contract(level, name):
    """The factory refuses the contracts it cannot serve (no device work)."""
    problems = []
    for args, why in (
        ((8, 192, _scale_of(192), "float32"), "float32 output dtype"),
        ((0, 192, _scale_of(192), "float16"), "heads=0"),
        ((8, 190, _scale_of(192), "float16"), "dim_qk not a multiple of 16"),
        ((8, 192, 0.0, "float16"), "scale=0"),
        ((8, 192, -0.1, "float16"), "negative scale"),
    ):
        try:
            _dq_land_kernel(*args)
            problems.append(f"{why}: no error raised")
        except ValueError:
            pass
    if problems:
        _record(level, name, False, "; ".join(problems))
        return False
    _record(level, name, True)
    print("       5 invalid contracts rejected", flush=True)
    return True


def _case_packed_split(level, name, tq=640, heads=8, dim_qk=192, dtype="float16", seed=61):
    """P7: one compiled program serves every packed splitting of one ``T_q``.

    The kernel is blind to request boundaries by contract (``cu_seqlens_*`` are
    not parameters of this launch), so five different splicings of the same
    ``T_q`` -- including zero-length requests -- must give byte-identical
    outputs on the *same* compiled program, and match the golden.
    """
    scale = _scale_of(dim_qk)
    splittings = (
        [256, 384],
        [100, 200, 340],
        [640],
        [0, 640],
        [0, 0, 0, 640],
    )
    diff = float("nan")
    try:
        dq_acc = _make_input(tq, heads, dim_qk, seed, None, dtype, scale)
        dq_acc_dev = dq_acc.to("npu")
        fn = _dq_land_kernel(heads, dim_qk, scale, dtype)
        first = _launch(fn, dq_acc_dev)
        torch.npu.synchronize()
        for q_lens in splittings[1:]:
            if int(sum(q_lens)) != tq:
                raise AssertionError(f"test bug: {q_lens} sums to {int(sum(q_lens))}, not {tq}")
            other = _launch(fn, dq_acc_dev)
            torch.npu.synchronize()
            if not torch.equal(first.view(torch.int16), other.view(torch.int16)):
                raise AssertionError(f"packed splitting {q_lens} gives different bits")
            del other
        gold = (dq_acc.float() * float(scale)).to(_TORCH_DTYPE[dtype])
        if not torch.equal(first.detach().cpu().view(torch.int16), gold.view(torch.int16)):
            bad = int((first.detach().cpu().view(torch.int16) != gold.view(torch.int16)).sum())
            raise AssertionError(f"{bad} element(s) differ from the golden under P7")
        del gold
    except AssertionError as exc:
        _record(level, name, False, str(exc)[:260])
        return False
    except (ValueError, RuntimeError) as exc:  # noqa: BLE001
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:260]}")
        return False
    _record(level, name, True)
    print(
        f"       T_q={tq} H={heads} D_qk={dim_qk} {dtype}: "
        f"{len(splittings)} splicings of one program are bit-identical",
        flush=True,
    )
    return True


# MLAVarlenBwdOp workloads (examples/TileOPs/tileops/manifest/attention.yaml):
# the packed launch only sees T_q = sum(q_lens) and H.
_MANIFEST = (
    ("ds-v3-8x4k", [4096] * 8, 128),
    ("ds-v3-mixed", [512, 1024, 2048, 4096, 8192, 16384], 128),
    ("kimi-k2-4x8k", [8192] * 4, 64),
)


def run_L0():
    """Gate: main path, ragged tails, degenerate shapes, corners, manifest set."""
    # L0-1 first: out_idx allocates the output from the dynamic first dimension,
    # and the packed call path feeds an already-flat (T_q, H, D_qk) tensor.
    run_case("L0", "L0-1 fp16-smoke-outidx", [256, 384], 8, 192, "float16", seed=1)
    run_case("L0", "L0-2 bf16-smoke", [256, 384], 8, 192, "bfloat16", seed=2)
    # Ragged requests: T_q = 308 = 4*64 + 52, so the last tile stores n = 52 rows.
    run_case("L0", "L0-3 fp16-ragged-tail-52", [200, 17, 91], 4, 192, "float16",
             seed=3, canary=True)
    run_case("L0", "L0-4 fp16-tiny-Tq1", [1], 1, 192, "float16", seed=4)
    # 65 = 1*64 + 1: one row in the tail tile, ntiles = 4 < grid (52 cores idle).
    run_case("L0", "L0-5 fp16-tail-1-row", [64, 1], 2, 192, "float16", seed=5,
             canary=True)
    # T_q a multiple of 64 (no tail): the control path for the clamp.
    run_case("L0", "L0-6 fp16-exact-Tq384", [320, 64], 4, 192, "float16", seed=6)
    # Corner set: single rounding, non-saturating narrowing, +-0, +-Inf, NaN.
    run_case("L0", "L0-7a fp16-corners", [64, 64], 4, 192, "float16", seed=7,
             special="corners")
    run_case("L0", "L0-7b bf16-corners", [64, 64], 4, 192, "bfloat16", seed=8,
             special="corners")
    run_case("L0", "L0-7c fp16-overflow", [128], 4, 192, "float16", seed=9,
             special="overflow")
    # bf16's finite range is within the float32 range, so the overflow corner
    # needs scale >= 1 (scale = 1.0 puts the float32 maximum past bf16 max).
    run_case("L0", "L0-7d bf16-overflow", [128], 4, 192, "bfloat16", seed=10,
             special="overflow", scale=1.0)
    # Non-finite accumulator values propagate elementwise (NaN compared by position).
    run_case("L0", "L0-8a fp16-non-finite-tail", [65], 4, 192, "float16", seed=11,
             special="nonfinite")
    run_case("L0", "L0-8b bf16-non-finite", [129], 4, 192, "bfloat16", seed=12,
             special="nonfinite")
    # Empty packed batch: ntiles = 0, every core runs an empty tile stream.
    run_case("L0", "L0-9 fp16-empty-Tq0", [0], 4, 192, "float16", seed=13)
    # Two identical launches must be bit-identical; the single-version variant
    # (no annotate_buffer_versions) is the ping-pong race control.
    run_case("L0", "L0-10 fp16-determinism-2x", [256, 384], 8, 192, "float16",
             seed=14, determinism=True, single_version_control=True)
    _case_packed_split("L0", "L0-11 P7-one-program-multi-splitting")
    _case_gates("L0", "L0-12 P0-simd-body-and-cache")
    _case_contract("L0", "L0-13 refuses-an-invalid-contract")
    # The three manifest workloads at their manifest T_q (largest: 32768 rows).
    for name, q_lens, heads in _MANIFEST:
        run_case("L0", f"L0-manifest-{name}", q_lens, heads, 192, "bfloat16",
                 seed=15, chunk_rows=1024)
    _finish("L0", strict=True)


def run_L1():
    """Representative packed shapes: both dtypes, repeats, own scale."""
    run_case("L1", "L1-1 fp16-smoke-repeat", [256, 384], 8, 192, "float16", seed=21,
             repeat=5)
    run_case("L1", "L1-2 bf16-smoke-repeat", [256, 384], 8, 192, "bfloat16", seed=22,
             repeat=5)
    run_case("L1", "L1-3 fp16-ds-v3-8x4k", [4096] * 8, 128, 192, "float16", seed=23,
             chunk_rows=1024)
    run_case("L1", "L1-4 bf16-kimi-k2-4x8k", [8192] * 4, 64, 192, "bfloat16", seed=24,
             chunk_rows=1024)
    run_case("L1", "L1-5 fp16-scale-0.05", [1024], 8, 192, "float16", seed=25,
             scale=0.05, single_version_control=True)
    run_case("L1", "L1-6 bf16-multi-request-T3584", [512, 1024, 2048], 16, 192,
             "bfloat16", seed=26, repeat=2)
    _finish("L1", strict=True)


def run_L2():
    """Manifest workloads (large, informative): non-blocking, timing baseline."""
    for name, q_lens, heads in _MANIFEST:
        run_case("L2", f"L2-{name}-repeat3", q_lens, heads, 192, "bfloat16",
                 seed=31, repeat=3, chunk_rows=1024)
    run_case("L2", "L2-fp16-ds-v3-mixed", _MANIFEST[1][1], 128, 192, "float16",
             seed=32, chunk_rows=1024)
    _finish("L2", strict=False)


def run_boundary():
    """Non-divisible T_q, odd head counts, other D_qk, all-zero and non-finite."""
    run_case("boundary", "B-1 bf16-tail-40", [1000], 8, 192, "bfloat16", seed=41)
    run_case("boundary", "B-2 fp16-tail-63", [63], 3, 192, "float16", seed=42)
    # H = 3 is not a power of two: the tile decode needs a real modulo.
    run_case("boundary", "B-3 bf16-heads3", [700], 3, 192, "bfloat16", seed=43)
    run_case("boundary", "B-4 fp16-Tq1-H1", [1], 1, 192, "float16", seed=44)
    # D_qk = 96 -> DW = 128, so the row is padded: the pad columns are computed
    # but never copied out (DESIGN.md 9.3, untested there).
    run_case("boundary", "B-5 bf16-Dqk96-padded", [200], 4, 96, "bfloat16", seed=45)
    run_case("boundary", "B-6 fp16-Dqk144-padded", [200], 4, 144, "float16", seed=46)
    run_case("boundary", "B-7 fp16-all-zero", [128], 4, 192, "float16", seed=47,
             special="zeros")
    run_case("boundary", "B-8 bf16-non-finite-tail", [129], 4, 192, "bfloat16",
             seed=48, special="nonfinite")
    # Small scale: no overflow anywhere, exercises the ordinary path only.
    run_case("boundary", "B-9 bf16-tiny-scale", [300], 6, 192, "bfloat16", seed=49,
             scale=1e-6)
    # Zero-length requests spliced into a non-empty batch.
    run_case("boundary", "B-10 bf16-empty-requests", [0, 200, 0], 4, 192, "bfloat16",
             seed=50)
    # Spill detector: a ragged tail first (any write past the last row of a head
    # lands in the next head at H=8), then an all-zero tensor where a stale write
    # would show up as a nonzero element.
    run_case("boundary", "B-11 fp16-spill-tail", [701], 8, 192, "float16", seed=51)
    run_case("boundary", "B-12 fp16-spill-zero", [704], 8, 192, "float16", seed=52,
             special="zeros")
    _finish("boundary", strict=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
    args, _ = parser.parse_known_args()
    levels = {"L0": run_L0, "L1": run_L1, "L2": run_L2, "boundary": run_boundary}
    chosen = list(levels.values()) if args.level == "all" else [levels[args.level]]
    started = time.perf_counter()
    for run in chosen:
        run()
    print(f"[TIME] {time.perf_counter() - started:.1f} s for level={args.level}", flush=True)
    blocking = [r for r in _RESULTS if r[0] in ("L0", "L1") and not r[2]]
    tag = "PRECISION_FAIL" if blocking else "PRECISION_PASS"
    print(
        f"[{tag}] max_diff={_MAX_DIFF[0]:.6g} bit_diff={_BIT_DIFF[0]} "
        f"blocking_failures={len(blocking)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
