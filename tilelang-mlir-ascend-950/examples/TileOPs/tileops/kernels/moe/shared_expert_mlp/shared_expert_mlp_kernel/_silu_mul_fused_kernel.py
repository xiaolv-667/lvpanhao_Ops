# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""SharedExpertMLPFwdOp part 3: ``silu(gate) * up`` on Ascend950PR_9579 (SIMD / AIV).

Migration of ``_silu_mul_fused_kernel`` (GPU TileLang / CUDA target) from
``tileops/_kernels/moe/shared_expert_mlp.py``::

    out[m, j] = silu(gate_up[m, j]) * gate_up[m, N + j]      m in [0, M), j in [0, N)
    silu(x)   = x / (1 + exp(-x))

``gate_up`` is ``[M, 2N]`` (gate half first, up half second) and ``out`` is
``[M, N]`` -- the activation output is the A operand of the following down
GEMM.  The dtype contract is 16-bit input widened exactly to fp32, the whole
chain evaluated in fp32 and a single rounding at the end (DESIGN.md 0.1).

NPU shape (DESIGN.md 0.6 / 3.3 / 10.5):

* one-dimensional ``T.Kernel(grid)``: ``grid = mt * nsp`` blocks, each block
  walks ``tpb`` column tiles of ``(bm, bn)`` rows/columns;
* ``T.Pipelined(tpb, num_stages=2)`` plus ``annotate_buffer_versions`` for the
  GM -> UB -> GM tile flow; every ``T.copy`` stays outside ``T.SimdVF()``;
* per 64-lane chunk the SIMD chain is DESIGN.md 1.6.1 A1::

      vld(UNPK_B16) -> vcvt(fp32, part=0)        # gate, up
      num = vmul(g, u)
      den = vadds(vexp(vneg(g)), 1.0)
      y   = vdiv(num, den)                       # (g*u) / (1 + exp(-g))
      vcvt(y, dtype, ROUND_R, sat=False) -> vsts(dist="PK_B32")

* the factory clamps the caller's tiling hint (DESIGN.md 5.2): the wrapper
  passes ``block_m=128, block_n=256``, which would need 393 216 B of UB for
  two buffer versions and overflow the 253 952 B budget.  ``bm`` is derived
  from ``M`` and ``bn`` is capped at 512, so ``(128, 256)`` lands on
  ``(16, 256)`` = 49 152 B.

run: python _silu_mul_fused_kernel.py --level all
"""

import argparse
import functools
import time

import torch
import torch.nn.functional as F

import tilelang
import tilelang.ascend.language as T
import torch_npu  # noqa: F401  # registers torch.npu

# ---------------------------------------------------------------------------
# Constants (DESIGN.md 4.5 / 5.2 / 5.5 / 8.2)
# ---------------------------------------------------------------------------
VL = 64  # fp32 vector lanes per register; also the 16-bit chunk of UNPK_B16/PK_B32
CORES_FALLBACK = 56  # Ascend950PR_9579 vector cores (statectl observed_cores.vector)
BN_MAX = 512  # bn ceiling (DESIGN.md 5.2 rule 1)
UB_BYTES = 253952  # Ascend950PR_9579.ini [AICoreSpec] ub_size
VERSIONS = 2  # buffer versions per T.Pipelined(num_stages=2)
N_BUFFERS = 3  # g_ub, u_ub, y_ub
ELEM_BYTES = {"float16": 2, "bfloat16": 2}

ATOL = {"float16": 1e-2, "bfloat16": 1e-2}  # DESIGN.md 8.2 (moe_verification(1))
RTOL = {"float16": 1e-2, "bfloat16": 1e-2}
NORM_SQ_TOL = 1e-4  # moe_close -> _assert_normalized_error


# ---------------------------------------------------------------------------
# 1. Golden (PyTorch CPU reference, DESIGN.md 8.1)
#
# Written from the source operator's dtype semantics only (DESIGN.md 0.1):
# exact 16-bit -> fp32 widening, fp32 silu and multiply, one final rounding.
# It deliberately contains none of the NPU implementation's structure, so the
# A1 identity of DESIGN.md 1.6.1 is cross-checked rather than self-proved.
# ---------------------------------------------------------------------------
def golden_silu_mul(gate_up: torch.Tensor, n: int) -> torch.Tensor:
    """``out = silu(gate_up[:, :n]) * gate_up[:, n:]`` in fp32, then one cast."""
    gate = gate_up[:, :n].float()  # gate half first (source gate_up[:, :N])
    up = gate_up[:, n:].float()  # up half second
    return (F.silu(gate) * up).to(gate_up.dtype)  # single rounding


# Harness gate name for this operator (``golden_{op}``).
golden__silu_mul_fused_kernel = golden_silu_mul


# ---------------------------------------------------------------------------
# 2. Build-time geometry (DESIGN.md 5.2 / 5.5)
# ---------------------------------------------------------------------------
_VECTOR_CORES = None


def vector_cores() -> int:
    """Vector core count, queried from the device (platform rule), not assumed."""
    global _VECTOR_CORES
    if _VECTOR_CORES is None:
        _VECTOR_CORES = CORES_FALLBACK
        try:  # no device at build-only time: keep the platform-recorded value
            props = torch.npu.get_device_properties(torch.npu.current_device())
            n = int(getattr(props, "vector_core_num", 0) or 0)
            if n > 0:
                _VECTOR_CORES = n
        except Exception:  # noqa: BLE001 - absence of a device is not an error
            pass
    return _VECTOR_CORES


def _divisors(n: int):
    small, large = [], []
    d = 1
    while d * d <= n:
        if n % d == 0:
            small.append(d)
            if d != n // d:
                large.append(n // d)
        d += 1
    return small + large[::-1]


def geometry(M: int, N: int, block_m: int, block_n: int, dtype_str: str = "float16") -> dict:
    """Clamp the caller's tiling hint to a UB-safe tile (DESIGN.md 5.2)."""
    if M < 1 or N < 1:
        raise ValueError(f"empty problem: M={M} N={N}")
    if dtype_str not in ELEM_BYTES:
        raise ValueError(f"unsupported dtype {dtype_str!r}")
    # rule 1: bn in [VL, 512], a multiple of VL (the vector chunk width).
    bn = max(VL, min(int(block_n), BN_MAX))
    bn -= bn % VL
    # rule 2: two row-tile steps, chosen from M (the hint's block_m is dropped:
    # 128 rows would need 393 216 B of UB with two versions).
    bm = 16 if M >= 64 else 1
    bm = min(bm, M)
    mt = -(-M // bm)
    nt = -(-N // bn)
    # rule 3: nsp divides nt and puts grid = mt * nsp closest to 4 waves.
    target = 4 * vector_cores()
    nsp, best = 1, None
    for d in _divisors(nt):
        grid = mt * d
        key = (abs(grid - target), grid)  # ties pick the smaller grid
        if best is None or key < best:
            best, nsp = key, d
    tpb = nt // nsp
    grid = mt * nsp
    # rule 4: UB budget with two versions of the three buffers.
    ub = VERSIONS * N_BUFFERS * bm * bn * ELEM_BYTES[dtype_str]
    return {
        "bm": bm,
        "bn": bn,
        "mt": mt,
        "nt": nt,
        "nsp": nsp,
        "tpb": tpb,
        "grid": grid,
        "ub": ub,
        "nchunk": bn // VL,
    }


# ---------------------------------------------------------------------------
# 3. Kernel (DESIGN.md 3.2 / 3.3 / 6.2 / 7)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=16)
def _silu_mul_fused_kernel(M: int, N: int, dtype_str: str):
    """Factory kept source-compatible: ``f(M, N, dtype)(block_m, block_n)``.

    The Stage 5 wrapper calls ``_silu_mul_fused_kernel(m, s, dtype)(
    block_m, block_n)`` with ``block_m=128, block_n=256``; the geometry clamp
    inside makes that hint UB-safe without touching the wrapper.
    """

    def _func(block_m: int, block_n: int):
        return _build(M, N, dtype_str, int(block_m), int(block_n))

    return _func


@functools.lru_cache(maxsize=32)
def _build(M: int, N: int, dtype_str: str, block_m: int, block_n: int):
    geo = geometry(M, N, block_m, block_n, dtype_str)
    if geo["ub"] > UB_BYTES:
        raise ValueError(
            f"UB budget exceeded: {geo['ub']} B > {UB_BYTES} B "
            f"(bm={geo['bm']} bn={geo['bn']} {dtype_str})"
        )
    bm, bn = geo["bm"], geo["bn"]
    nsp, tpb, grid, nchunk = geo["nsp"], geo["tpb"], geo["grid"], geo["nchunk"]

    @T.prim_func
    def main(
        gate_up: T.Tensor((M, 2 * N), dtype_str),  # type: ignore[valid-type]
        out: T.Tensor((M, N), dtype_str),  # type: ignore[valid-type]
    ) -> None:
        with T.Kernel(grid) as bx:
            im = bx // nsp  # row block
            sp = bx % nsp  # first column tile of this core
            g_ub = T.alloc_shared((bm, bn), dtype_str)
            u_ub = T.alloc_shared((bm, bn), dtype_str)
            y_ub = T.alloc_shared((bm, bn), dtype_str)
            T.annotate_buffer_versions({g_ub: VERSIONS, u_ub: VERSIONS, y_ub: VERSIONS})
            for it in T.Pipelined(tpb, num_stages=VERSIONS):
                tn = sp * tpb + it
                m0 = im * bm
                n0 = tn * bn
                am = T.min(bm, M - m0)  # tail rows of the last row block
                an = T.min(bn, N - n0)  # tail columns of the last column tile
                T.copy(gate_up[m0 : m0 + am, n0 : n0 + an], g_ub[0:am, 0:an])
                T.copy(gate_up[m0 : m0 + am, N + n0 : N + n0 + an], u_ub[0:am, 0:an])
                with T.SimdVF():
                    # 32-bit lane pattern: the narrow store packs 32-bit words, so
                    # a 16-bit mask would silently write only half the slots.
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
                            den = T.simd.vadds(
                                T.simd.vexp(T.simd.vneg(g, full), full), 1.0, full
                            )
                            y = T.simd.vdiv(num, den, full)
                            T.simd.vsts(
                                y_ub[r, c * VL],
                                T.simd.vcvt(y, dtype_str, round="ROUND_R", sat=False, part=0),
                                full,
                                dist="PK_B32",
                            )
                T.copy(y_ub[0:am, 0:an], out[m0 : m0 + am, n0 : n0 + an])

    return tilelang.compile(main, target="ascend", out_idx=[1])


# ---------------------------------------------------------------------------
# 4. Precision comparison (DESIGN.md 8.2)
# ---------------------------------------------------------------------------
TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}

_RESULTS = []  # (level, name, ok, detail)
_MAX_ABS = {"float16": 0.0, "bfloat16": 0.0}


def _record(level, name, ok, detail=""):
    _RESULTS.append((level, name, ok, detail))


def _make_inputs(M, N, dtype_str, seed, special=None):
    """One ``[M, 2N]`` gate/up tensor in the requested 16-bit dtype."""
    gen = torch.Generator().manual_seed(seed)
    if special == "zeros":
        x = torch.zeros(M, 2 * N, dtype=torch.float32)
    elif special in ("wide-range", "manifest"):
        scale = 20.0 if special == "wide-range" else 4.0
        x = torch.randn(M, 2 * N, generator=gen, dtype=torch.float32) * scale
    else:
        x = torch.randn(M, 2 * N, generator=gen, dtype=torch.float32) * 1.5

    if special == "non-finite":
        # IEEE propagation, same classification as the golden (DESIGN.md 0.1):
        # gate = -inf -> num = -inf, den = inf -> NaN; gate = +inf -> +inf.
        rows, up = x.shape[0], N
        pairs = [
            (0, 0, float("inf")),
            (0, 1, float("-inf")),
            (0, 2, float("nan")),
            (0, up + 0, float("inf")),
            (1, 3, float("-inf")),
            (1, up + 1, float("-inf")),
            (2, 4, 0.0),
            (2, up + 2, float("nan")),
        ]
        for r, c, v in pairs:
            if r < rows and c < x.shape[1]:
                x[r, c] = v
    elif special == "fp16-overflow":
        # |silu(g) * u| > 65504: the golden rounds to inf, the narrow vcvt with
        # sat=False is the implementation's own overflow semantics (Boundary).
        x[:, 0] = 10.0
        x[:, N] = 65504.0
        x[:, 1] = 20.0
        x[:, N + 1] = 65504.0
    elif special == "bf16-overflow":
        x[:, 0] = 100.0
        x[:, N] = 3.0e38
    elif special not in (None, "zeros", "wide-range", "manifest"):
        raise ValueError(f"unknown special pattern {special!r}")
    return x.to(TORCH_DTYPE[dtype_str])


def _diff(got, gold):
    """Largest absolute difference in float space plus the NaN-mask mismatch."""
    a, b = got.float(), gold.float()
    d = (a - b).abs()
    d = torch.nan_to_num(d, nan=0.0)  # inf - inf and nan - nan are not errors
    nan_mismatch = int(((a != a) != (b != b)).sum())
    return (float(d.max()) if d.numel() else 0.0), nan_mismatch


def _norm_sq_err(got, gold):
    a, b = got.float(), gold.float()
    den = float((b * b).sum())
    if not torch.isfinite(b).all() or den == 0.0:
        return float("nan")
    return float(((a - b) ** 2).sum()) / den


def run_case(
    level,
    name,
    M,
    N,
    dtype_str,
    seed=0,
    special=None,
    repeat=1,
    block_m=128,
    block_n=256,
):
    """Build one geometry, run it on the NPU and compare with the golden.

    ``block_m`` / ``block_n`` default to the Stage 5 wrapper's
    ``default_config`` (128, 256) so every case exercises the UB clamp.
    """
    geo = geometry(M, N, block_m, block_n, dtype_str)
    tag = (
        f"bm={geo['bm']} bn={geo['bn']} nsp={geo['nsp']} tpb={geo['tpb']} "
        f"grid={geo['grid']} ub={geo['ub']}"
    )
    elapsed = float("nan")
    diff = float("nan")
    norm = float("nan")
    try:
        gate_up = _make_inputs(M, N, dtype_str, seed, special)
        gold = golden_silu_mul(gate_up, N)
        dev = gate_up.npu()
        fn = _build(M, N, dtype_str, block_m, block_n)

        start = time.perf_counter()
        got = fn(dev)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1e3

        det = True
        if repeat > 1:
            ref_bits = got.view(torch.int16)
            start = time.perf_counter()
            for _ in range(repeat - 1):
                again = fn(dev)
                det = det and torch.equal(again.view(torch.int16), ref_bits)
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) * 1e3 / (repeat - 1)
        got = got.cpu()

        problems = []
        if got.shape != gold.shape or got.dtype != gold.dtype:
            problems.append(f"got {tuple(got.shape)}/{got.dtype} want {tuple(gold.shape)}/{gold.dtype}")
        else:
            diff, nan_mismatch = _diff(got, gold)
            _MAX_ABS[dtype_str] = max(_MAX_ABS[dtype_str], diff)
            norm = _norm_sq_err(got, gold)
            if nan_mismatch:
                problems.append(f"{nan_mismatch} NaN positions differ")
            if not det:
                problems.append(f"non-deterministic over {repeat} launches")
            try:
                torch.testing.assert_close(
                    got,
                    gold,
                    atol=ATOL[dtype_str],
                    rtol=RTOL[dtype_str],
                    equal_nan=True,
                )
            except AssertionError:
                problems.append(
                    f"max_diff={diff:.3e} > atol={ATOL[dtype_str]:g} rtol={RTOL[dtype_str]:g}"
                )
            if norm == norm and norm > NORM_SQ_TOL:
                problems.append(f"normalized sq err {norm:.3e} > {NORM_SQ_TOL:g}")
        if problems:
            raise AssertionError("; ".join(problems))
    except AssertionError as exc:
        _record(level, name, False, f"{str(exc)[:400]}")
        print(f"[{level}][FAIL] {name}: {str(exc)[:400]} | {tag}", flush=True)
        return False
    except (ValueError, RuntimeError, TypeError) as exc:
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:400]}")
        print(f"[{level}][FAIL] {name}: {type(exc).__name__}: {str(exc)[:300]} | {tag}", flush=True)
        return False
    _record(level, name, True)
    print(
        f"       {name}: M={M} N={N} {dtype_str} diff={diff:.3e} norm={norm:.3e} "
        f"{tag} launch={elapsed:.3f} ms",
        flush=True,
    )
    return True


def _finish(level, strict):
    rows = [r for r in _RESULTS if r[0] == level]
    failed = [r for r in rows if not r[2]]
    print(
        f"[{level}] {len(rows) - len(failed)}/{len(rows)} cases passed | "
        f"max_abs fp16={_MAX_ABS['float16']:.3e} bf16={_MAX_ABS['bfloat16']:.3e}",
        flush=True,
    )
    if failed:
        for row in failed:
            print(f"[{level}][{'FAIL' if strict else 'WARN'}] {row[1]}: {row[3]}", flush=True)
        if strict:
            raise AssertionError(f"[{level}] {len(failed)} case(s) failed")


# ---------------------------------------------------------------------------
# 5. Layered cases
# ---------------------------------------------------------------------------
def run_L0():
    """Gate: the two workloads that actually call this kernel, the decode-width
    workloads, both 16-bit dtypes, the M tail block, non-finite propagation and
    three-launch determinism."""
    run_case("L0", "L0-1 kimi-k2-t512 bf16", 512, 18432, "bfloat16", seed=1)
    run_case("L0", "L0-2 ds-v3-t4096 bf16", 4096, 2048, "bfloat16", seed=2)
    run_case("L0", "L0-3 glm45-t32 bf16", 32, 1536, "bfloat16", seed=3)
    run_case("L0", "L0-4 glm45-t32 fp16", 32, 1536, "float16", seed=4)
    run_case("L0", "L0-5 ds-v3-t32 fp16", 32, 2048, "float16", seed=5)
    run_case("L0", "L0-6 ds-v3-t32 bf16", 32, 2048, "bfloat16", seed=6)
    run_case("L0", "L0-7 M-tail-100 bf16", 100, 2048, "bfloat16", seed=7)
    run_case("L0", "L0-8 non-finite bf16", 32, 2048, "bfloat16", seed=8, special="non-finite")
    run_case("L0", "L0-9 det3 kimi-k2-t512 bf16", 512, 18432, "bfloat16", seed=9, repeat=3)
    run_case("L0", "L0-10 det3 ds-v3-t32 fp16", 32, 2048, "float16", seed=10, repeat=3)
    _finish("L0", strict=True)


def run_L1():
    """Cross dtype / geometry coverage: fp16 on the large workloads, M = 1,
    intermediate row blocks, longer pipelines and repeat determinism."""
    run_case("L1", "L1-1 kimi-k2-t512 fp16", 512, 18432, "float16", seed=21, repeat=3)
    run_case("L1", "L1-2 ds-v3-t4096 fp16", 4096, 2048, "float16", seed=22, repeat=3)
    run_case("L1", "L1-3 kimi-k2-t1 bf16", 1, 18432, "bfloat16", seed=23)
    run_case("L1", "L1-4 rows-128 bf16", 128, 2048, "bfloat16", seed=24, repeat=3)
    run_case("L1", "L1-5 1024x3072 bf16", 1024, 3072, "bfloat16", seed=25, repeat=3)
    run_case("L1", "L1-6 kimi-k2-t1 fp16", 1, 4096, "float16", seed=26)
    run_case("L1", "L1-7 rows-256 fp16", 256, 18432, "float16", seed=27, repeat=3)
    _finish("L1", strict=True)


def run_L2():
    """The five manifest workloads with the manifest-style input recipe
    (informative: same code paths as L0/L1, larger traffic)."""
    run_case("L2", "L2-1 kimi-k2-t1", 1, 18432, "bfloat16", seed=31, special="manifest", repeat=3)
    run_case("L2", "L2-2 kimi-k2-t512", 512, 18432, "bfloat16", seed=32, special="manifest", repeat=3)
    run_case("L2", "L2-3 ds-v3-t32", 32, 2048, "bfloat16", seed=33, special="manifest", repeat=3)
    run_case("L2", "L2-4 ds-v3-t4096", 4096, 2048, "bfloat16", seed=34, special="manifest", repeat=3)
    run_case("L2", "L2-5 glm45-t32", 32, 1536, "bfloat16", seed=35, special="manifest", repeat=3)
    _finish("L2", strict=False)


def run_boundary():
    """Non-divisible N (tail lanes), N < bn, N == VL, M tail rows, zero and
    non-finite inputs, and the narrow-conversion overflow regime (informative:
    the golden rounds to inf while vcvt(sat=False) may wrap)."""
    run_case("boundary", "B-1 N1500-tail-lane bf16", 8, 1500, "bfloat16", seed=41)
    run_case("boundary", "B-2 N200-below-bn fp16", 8, 200, "float16", seed=42)
    run_case("boundary", "B-3 N100-single-row bf16", 1, 100, "bfloat16", seed=43)
    run_case("boundary", "B-4 N1537-one-lane-tail bf16", 8, 1537, "bfloat16", seed=44)
    run_case("boundary", "B-5 N64-single-chunk fp16", 8, 64, "float16", seed=45)
    run_case("boundary", "B-6 M17-tail-row bf16", 17, 2048, "bfloat16", seed=46)
    run_case("boundary", "B-7 zeros fp16", 8, 256, "float16", seed=47, special="zeros")
    run_case("boundary", "B-8 non-finite fp16", 8, 256, "float16", seed=48, special="non-finite")
    run_case("boundary", "B-9 wide-range bf16", 32, 2048, "bfloat16", seed=49, special="wide-range")
    run_case("boundary", "B-10 fp16-overflow", 8, 256, "float16", seed=50, special="fp16-overflow")
    run_case("boundary", "B-11 bf16-overflow", 8, 256, "bfloat16", seed=51, special="bf16-overflow")
    _finish("boundary", strict=False)


def _verdict():
    failed = [r for r in _RESULTS if not r[2] and r[0] in ("L0", "L1")]
    warns = [r for r in _RESULTS if not r[2] and r[0] not in ("L0", "L1")]
    if warns:
        print(f"[WARN] {len(warns)} non-blocking case(s) differed: "
              + ", ".join(r[1] for r in warns), flush=True)
    if failed:
        print(f"[PRECISION_FAIL] {len(failed)} blocking case(s) failed", flush=True)
    else:
        print(
            f"[PRECISION_PASS] max_abs fp16={_MAX_ABS['float16']:.3e} "
            f"bf16={_MAX_ABS['bfloat16']:.3e}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
    args, _ = parser.parse_known_args()
    levels = {"L0": run_L0, "L1": run_L1, "L2": run_L2, "boundary": run_boundary}
    chosen = list(levels.values()) if args.level == "all" else [levels[args.level]]
    started = time.perf_counter()
    print(f"[ENV] vector_cores={vector_cores()}", flush=True)
    for run in chosen:
        run()
    print(f"[TIME] {time.perf_counter() - started:.1f} s for level={args.level}", flush=True)
    _verdict()


if __name__ == "__main__":
    main()
