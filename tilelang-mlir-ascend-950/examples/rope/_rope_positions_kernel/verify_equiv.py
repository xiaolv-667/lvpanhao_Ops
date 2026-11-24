"""Equivalence machine-check for the `_rope_positions_kernel` design (D-1).

Torch CPU only (no device, no torch_npu). Every adopted optimization of
``DESIGN.md`` §1.6.1 is compared against the baseline formula on random tensors
plus corner values, with a float64 reference where the two formulas are not
provably bit-identical.

Items (adopted -- must be ``EQUIV_PASS``):
  A1  table-row hoisting: the cos/sin row of a token is read once and reused by
      every head, instead of re-reading the row per (head, pair). The two
      addressing schemes are implemented independently; a hoist that took the
      wrong ``k`` offset would show up as a mismatch.
  A2  clamp-then-narrow: the table row is ``clamp(position + offset, 0, P-1)``
      in the position's own dtype and only then narrowed to int32. Compared
      against the int64 row arithmetic over an exhaustive index domain
      (negative, past the table, int32 edges, beyond-int32 values).
  A3  tail pass-through: columns past ``rotary_dim`` travel through the staged
      tile (one bulk copy of the whole tile, then rotate in place) instead of a
      per-column copy loop. Bit-exact by construction; the check confirms the
      copy leaves every non-rotated column untouched, for both pairings and for
      odd head sizes (65 columns).

Items (rejected -- recorded with their numbers in ``DESIGN.md`` §11):
  X1  native-dtype arithmetic (round after every op in the caller's dtype)
      instead of one float32 transit with a single final rounding.
  X2  single-rounding fused multiply-add of ``x_lo*c - x_hi*s``.
  X3  computing cos/sin on the fly from ``position * base**(-2k/rotary_dim)``
      instead of reading the caller's dtype-rounded tables.
  G1  golden sanity: the §8.1 golden (float32 transit, one rounding) against an
      fp64 reference, so that §8.2 tolerances rest on a measured number.

Usage:  python examples/rope/_rope_positions_kernel/verify_equiv.py
Exit code 0 when every adopted item passes, 1 otherwise.
"""

from __future__ import annotations

import math

import torch

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MANTISSA_MASK = {2: 0x7FFF, 4: 0x7FFFFFFF}
_INT_VIEW = {2: torch.int16, 4: torch.int32}

ADOPTED: list[tuple] = []
REJECTED: list[tuple] = []


# ---------------------------------------------------------------------------
# numeric helpers
# ---------------------------------------------------------------------------


def _monotonic_key(t: torch.Tensor) -> torch.Tensor:
    """Map a float tensor's bits onto a monotonic integer key (ulp distance)."""
    itemsize = t.element_size()
    i = t.contiguous().view(_INT_VIEW[itemsize]).to(torch.int64)
    sign = (i >> (8 * itemsize - 1)) & 1
    return i ^ (sign * _MANTISSA_MASK[itemsize])


def _ulp_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max ulp distance between two same-dtype tensors (NaN==NaN when both NaN)."""
    both_nan = torch.isnan(a) & torch.isnan(b)
    ka, kb = _monotonic_key(a), _monotonic_key(b)
    diff = (ka - kb).abs()
    diff = torch.where(both_nan, torch.zeros_like(diff), diff)
    if torch.isnan(a).any() or torch.isnan(b).any():
        # one-sided NaN is an infinite distance
        one_sided = torch.isnan(a) ^ torch.isnan(b)
        if one_sided.any():
            return float("inf")
    return float(diff.max()) if diff.numel() else 0.0


def _mismatches(got: torch.Tensor, want: torch.Tensor) -> int:
    """Elements whose dtype bit pattern differs (NaN equal iff both NaN)."""
    same = got.view(torch.int16 if got.element_size() == 2 else torch.int32) == want.view(
        torch.int16 if want.element_size() == 2 else torch.int32
    )
    both_nan = torch.isnan(got) & torch.isnan(want)
    return int((~(same | both_nan)).sum())


def _record(name, kind, compared, violations, max_ulp, tol_violations, basis="bitwise"):
    """Record one comparison row.

    ``basis="bitwise"``: the claim is bit-exact equality (A1/A2/A3).
    ``basis="tolerance"``: the claim is equality within the contract tolerance
    (G1 -- the float32 transit differs from an fp64 reference by <=1 ulp, which
    is what §8.2's tolerances have to absorb).
    """
    failed = violations if basis == "bitwise" else tol_violations
    row = {
        "item": name,
        "kind": kind,
        "compared": compared,
        "violations": violations,
        "rate": violations / compared if compared else 0.0,
        "max_ulp": max_ulp,
        "tol_violations": tol_violations,
        "verdict": "EQUIV_PASS" if failed == 0 else "EQUIV_FAIL",
    }
    (ADOPTED if kind == "adopted" else REJECTED).append(row)


# ---------------------------------------------------------------------------
# the rotation, written four ways
# ---------------------------------------------------------------------------


def _table_step(cos, sin, pos, offsets, max_position, dtype):
    """Baseline table row: clamp in the position dtype, then narrow to int32."""
    seen = pos if offsets is None else pos + offsets
    seen = torch.clamp(seen.to(torch.int64), 0, max_position - 1)
    return seen.to(torch.int32)


def _rotate_f32(x, cos, sin, rows, neox, dtype):
    """§0.1 golden: float32 arithmetic for every op, one rounding to *dtype*.

    ``x`` is the dtype tensor, ``cos``/``sin`` the ``(P, rotary_dim/2)`` tables.
    Layout: ``x`` is ``(N, H, head_size)``; only the leading ``rotary_dim``
    columns rotate, the rest are copied.
    """
    n, h, head_size = x.shape
    rotary_dim = cos.shape[-1] * 2
    half = rotary_dim // 2
    xf = x.to(torch.float32)
    c = cos.to(torch.float32)[rows].unsqueeze(1)  # (N, 1, half)
    s = sin.to(torch.float32)[rows].unsqueeze(1)
    if neox:
        lo = xf[:, :, :half]
        hi = xf[:, :, half:rotary_dim]
    else:
        lo = xf[:, :, 0:rotary_dim:2]
        hi = xf[:, :, 1:rotary_dim:2]
    out_lo = lo * c - hi * s
    out_hi = hi * c + lo * s
    out = xf.clone()
    if neox:
        out[:, :, :half] = out_lo
        out[:, :, half:rotary_dim] = out_hi
    else:
        out[:, :, 0:rotary_dim:2] = out_lo
        out[:, :, 1:rotary_dim:2] = out_hi
    return out.to(dtype)


def _rotate_hoisted(x, cos, sin, pos, offsets, max_position, neox, dtype):
    """A1: the row is read once per token and re-addressed per head/pair.

    The table value a pair uses is fetched *inside* the head loop from the single
    staged row, exactly as the kernel does (`cs_ub[i, k]`), so a wrong ``k`` or
    head offset in the hoisted form shows up here.
    """
    rows = _table_step(cos, sin, pos, offsets, max_position, dtype)
    n, h, head_size = x.shape
    half = cos.shape[-1]
    rotary_dim = 2 * half
    staged_cos = cos[rows]  # (N, half) -- one row load per token
    staged_sin = sin[rows]
    xf = x.to(torch.float32)
    out = xf.clone()
    for head in range(h):
        for k in range(half):
            c = staged_cos[:, k].to(torch.float32).unsqueeze(1)
            s = staged_sin[:, k].to(torch.float32).unsqueeze(1)
            # within-head column offsets: dimension 1 of ``x`` is already the head
            if neox:
                lo_c, hi_c = k, half + k
            else:
                lo_c, hi_c = 2 * k, 2 * k + 1
            x_lo = xf[:, head, lo_c].unsqueeze(1)
            x_hi = xf[:, head, hi_c].unsqueeze(1)
            out[:, head, lo_c] = (x_lo * c - x_hi * s).squeeze(1)
            out[:, head, hi_c] = (x_hi * c + x_lo * s).squeeze(1)
    return out.to(dtype), rows


def _rotate_native(x, cos, sin, rows, neox, dtype):
    """X1: every arithmetic op rounded to *dtype* (no float32 transit)."""
    n, h, head_size = x.shape
    rotary_dim = cos.shape[-1] * 2
    half = rotary_dim // 2

    def r(t):
        return t.to(dtype).to(torch.float32)

    xf = x.to(torch.float32)
    c = r(cos[rows].unsqueeze(1))
    s = r(sin[rows].unsqueeze(1))
    if neox:
        lo, hi = xf[:, :, :half], xf[:, :, half:rotary_dim]
    else:
        lo, hi = xf[:, :, 0:rotary_dim:2], xf[:, :, 1:rotary_dim:2]
    out_lo = r(r(lo * c) - r(hi * s))
    out_hi = r(r(hi * c) + r(lo * s))
    out = xf.clone()
    if neox:
        out[:, :, :half], out[:, :, half:rotary_dim] = out_lo, out_hi
    else:
        out[:, :, 0:rotary_dim:2], out[:, :, 1:rotary_dim:2] = out_lo, out_hi
    return out.to(dtype)


def _rotate_fma(x, cos, sin, rows, neox, dtype):
    """X2: ``x_lo*c - x_hi*s`` fused to a single rounding (fp64 emulation)."""
    n, h, head_size = x.shape
    rotary_dim = cos.shape[-1] * 2
    half = rotary_dim // 2
    xd = x.to(torch.float64)
    c = cos.to(torch.float64)[rows].unsqueeze(1)
    s = sin.to(torch.float64)[rows].unsqueeze(1)
    if neox:
        lo, hi = xd[:, :, :half], xd[:, :, half:rotary_dim]
    else:
        lo, hi = xd[:, :, 0:rotary_dim:2], xd[:, :, 1:rotary_dim:2]
    # single rounding of the whole expression: fp64 product/sum, round to dtype
    out_lo = (lo * c - hi * s).to(dtype).to(torch.float64)
    out_hi = (hi * c + lo * s).to(dtype).to(torch.float64)
    out = xd.clone()
    if neox:
        out[:, :, :half], out[:, :, half:rotary_dim] = out_lo, out_hi
    else:
        out[:, :, 0:rotary_dim:2], out[:, :, 1:rotary_dim:2] = out_lo, out_hi
    return out.to(dtype)


def _rotate_onthefly(x, pos, offsets, base, rotary_dim, max_position, neox, dtype):
    """X3: tables recomputed in fp64 from the position instead of read."""
    rows = _table_step(cos=None, sin=None, pos=pos, offsets=offsets, max_position=max_position,
                       dtype=dtype)
    inv = 1.0 / torch.tensor(
        [float(base) ** (-2.0 * k / rotary_dim) for k in range(rotary_dim // 2)],
        dtype=torch.float64,
    )
    angles = torch.arange(max_position, dtype=torch.float64).unsqueeze(1) * inv.unsqueeze(0)
    cos64 = torch.cos(angles).to(dtype)
    sin64 = torch.sin(angles).to(dtype)
    return _rotate_f32(x, cos64, sin64, rows, neox, dtype)


def _rotate_fp64(x, cos, sin, rows, neox, dtype):
    """fp64 reference: same formula, one rounding at the very end."""
    n, h, head_size = x.shape
    rotary_dim = cos.shape[-1] * 2
    half = rotary_dim // 2
    xd = x.to(torch.float64)
    c = cos.to(torch.float64)[rows].unsqueeze(1)
    s = sin.to(torch.float64)[rows].unsqueeze(1)
    if neox:
        lo, hi = xd[:, :, :half], xd[:, :, half:rotary_dim]
    else:
        lo, hi = xd[:, :, 0:rotary_dim:2], xd[:, :, 1:rotary_dim:2]
    out = xd.clone()
    if neox:
        out[:, :, :half] = lo * c - hi * s
        out[:, :, half:rotary_dim] = hi * c + lo * s
    else:
        out[:, :, 0:rotary_dim:2] = lo * c - hi * s
        out[:, :, 1:rotary_dim:2] = hi * c + lo * s
    return out.to(dtype)


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


def _tables(max_position, half, dtype, base=10000.0, rotary_dim=None):
    rotary_dim = rotary_dim or 2 * half
    t = torch.arange(max_position, dtype=torch.float32)
    inv = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    angles = torch.outer(t, inv)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _inputs(n, h, head_size, dtype, gen):
    if gen == "random":
        return torch.randn(n, h, head_size, dtype=torch.float32).to(dtype)
    corners = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan"),
         65504.0, -65504.0, 6.1e-5, -6.1e-5, 3.4e38, -3.4e38],
        dtype=torch.float32,
    )
    flat = corners.repeat((n * h * head_size) // corners.numel() + 1)
    return flat[: n * h * head_size].reshape(n, h, head_size).to(dtype)


PROBES = [
    # n, heads, head_size, rotary_dim, neox
    (24, 4, 64, 64, True),
    (24, 4, 64, 64, False),
    (24, 4, 80, 32, True),
    (24, 4, 65, 32, False),
    (24, 2, 256, 64, False),
]


def check_adopted():
    # ---- A1: table-row hoisting -------------------------------------------
    for dtype in DTYPES:
        for n, h, head_size, rotary_dim, neox in PROBES:
            torch.manual_seed(7000 + rotary_dim + int(neox))
            cos, sin = _tables(128, rotary_dim // 2, dtype)
            x = _inputs(n, h, head_size, dtype, "random")
            pos = torch.randint(-16, 160, (n,), dtype=torch.int64)
            rows64 = _table_step(cos, sin, pos, None, 128, dtype)
            base = _rotate_f32(x, cos, sin, rows64, neox, dtype)
            got, rows32 = _rotate_hoisted(x, cos, sin, pos, None, 128, neox, dtype)
            compared = got.numel()
            viol = _mismatches(got, base)
            ulp = _ulp_diff(got, base)
            _record(
                f"A1 表行复用 head_size={head_size} rot={rotary_dim} neox={neox} {str(dtype).split('.')[-1]}",
                "adopted", compared, viol, ulp, viol,
            )
            assert rows32.dtype == torch.int32
            assert int(_ulp_diff(rows32.to(torch.float32), rows64.to(torch.float32))) == 0

    # ---- A2: clamp-then-narrow -------------------------------------------
    max_position = 8192
    domain = list(range(-70, max_position + 70))
    domain += [
        -(2**31), 2**31 - 1, -(2**31) + 1, 2**31, 2**31 + 1,
        2**32 + 1, -(2**32) + 1, 2**40, -(2**40),
    ]
    pos64 = torch.tensor(domain, dtype=torch.int64)
    want = torch.clamp(pos64, 0, max_position - 1)
    # int32 carriers cannot hold out-of-int32 values: the clamp is done in the
    # position dtype's own arithmetic and only then narrowed, so the int32 case
    # is the identity on its own domain.
    got32 = torch.clamp(pos64.to(torch.int32).to(torch.int64), 0, max_position - 1)
    # narrow AFTER the int64 clamp (the design's order)
    got_narrow = torch.clamp(pos64, 0, max_position - 1).to(torch.int32).to(torch.int64)
    viol = int((got_narrow != want).sum())
    _record(
        "A2 clamp 后窄化 int64→int32（域 -70..P+70 全枚举 + int32/int64 极端值）",
        "adopted", want.numel(), viol, 0.0, viol,
    )
    # the same clamp expressed in int32 without the value check is only safe
    # because positions are clamped first: show the failure mode of narrowing
    # first, as the guard's evidence.
    narrow_first = torch.clamp(pos64.to(torch.int32).to(torch.int64), 0, max_position - 1)
    narrow_first_viol = int((narrow_first != want).sum())
    REJECTED.append({
        "item": "A2 反例：先窄化后 clamp（若窄化前不 clamp，超出 int32 的 position 回绕）",
        "kind": "guard", "compared": want.numel(), "violations": narrow_first_viol,
        "rate": narrow_first_viol / want.numel(), "max_ulp": float(narrow_first_viol > 0),
        "tol_violations": narrow_first_viol,
        "verdict": "EQUIV_FAIL" if narrow_first_viol else "EQUIV_PASS",
    })

    # ---- A3: tail columns pass through the staged tile --------------------
    for dtype in DTYPES:
        for n, h, head_size, rotary_dim, neox in PROBES:
            if head_size == rotary_dim:
                continue
            torch.manual_seed(9100 + head_size + int(neox))
            cos, sin = _tables(128, rotary_dim // 2, dtype)
            x = _inputs(n, h, head_size, dtype, "random")
            pos = torch.randint(0, 128, (n,), dtype=torch.int64)
            rows = _table_step(cos, sin, pos, None, 128, dtype)
            base = _rotate_f32(x, cos, sin, rows, neox, dtype)
            # staged scheme: whole tile copied (x -> y), then the rotary region
            # overwritten in place; the copied columns are never touched again.
            y = x.clone()
            rotated = _rotate_f32(x, cos, sin, rows, neox, dtype)
            if neox:
                y[:, :, :rotary_dim] = rotated[:, :, :rotary_dim]
            else:
                y[:, :, :rotary_dim] = rotated[:, :, :rotary_dim]
            compared = y.numel()
            viol = _mismatches(y, base)
            _record(
                f"A3 尾列直通（整块拷贝 + 旋转区覆写）head_size={head_size} rot={rotary_dim} neox={neox} {str(dtype).split('.')[-1]}",
                "adopted", compared, viol, _ulp_diff(y, base), viol,
            )

    # ---- G1: golden vs fp64 reference ------------------------------------
    for dtype in DTYPES:
        for n, h, head_size, rotary_dim, neox in PROBES:
            torch.manual_seed(1300 + rotary_dim)
            cos, sin = _tables(128, rotary_dim // 2, dtype)
            x = _inputs(n, h, head_size, dtype, "random")
            pos = torch.randint(0, 128, (n,), dtype=torch.int64)
            rows = _table_step(cos, sin, pos, None, 128, dtype)
            got = _rotate_f32(x, cos, sin, rows, neox, dtype)
            ref = _rotate_fp64(x, cos, sin, rows, neox, dtype)
            ulp = _ulp_diff(got, ref)
            viol = _mismatches(got, ref)
            tol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1e-5}[dtype]
            tol_viol = int(
                (~torch.isclose(got.float(), ref.float(), atol=tol, rtol=tol)).sum()
            )
            _record(
                f"G1 golden(f32 单次舍入) vs fp64 参照 head_size={head_size} rot={rotary_dim} neox={neox} {str(dtype).split('.')[-1]}",
                "adopted", got.numel(), viol, ulp, tol_viol, basis="tolerance",
            )


def check_rejected():
    for dtype in DTYPES:
        tol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1e-5}[dtype]
        for n, h, head_size, rotary_dim, neox in PROBES:
            torch.manual_seed(4400 + rotary_dim)
            cos, sin = _tables(128, rotary_dim // 2, dtype)
            x = _inputs(n, h, head_size, dtype, "random")
            pos = torch.randint(0, 128, (n,), dtype=torch.int64)
            rows = _table_step(cos, sin, pos, None, 128, dtype)
            base = _rotate_f32(x, cos, sin, rows, neox, dtype)
            native = _rotate_native(x, cos, sin, rows, neox, dtype)
            fma = _rotate_fma(x, cos, sin, rows, neox, dtype)
            tol_viol = int((~torch.isclose(native.float(), base.float(), atol=tol, rtol=tol)).sum())
            _record(
                f"X1 原生 dtype 逐 op 舍入 head_size={head_size} rot={rotary_dim} neox={neox} {str(dtype).split('.')[-1]}",
                "rejected", base.numel(), _mismatches(native, base), _ulp_diff(native, base),
                tol_viol,
            )
            _record(
                f"X2 单次舍入 FMA head_size={head_size} rot={rotary_dim} neox={neox} {str(dtype).split('.')[-1]}",
                "rejected", base.numel(), _mismatches(fma, base), _ulp_diff(fma, base),
                int((~torch.isclose(fma.float(), base.float(), atol=tol, rtol=tol)).sum()),
            )

    # X3 needs the table contract (fp32 angle -> dtype) as its baseline
    for dtype in DTYPES:
        n, h, head_size, rotary_dim, neox = 64, 4, 64, 64, True
        max_position = 4096
        base_freq = 500000.0
        cos, sin = _tables(max_position, rotary_dim // 2, dtype, base=base_freq)
        torch.manual_seed(5200)
        x = _inputs(n, h, head_size, dtype, "random")
        pos = torch.randint(0, max_position, (n,), dtype=torch.int64)
        rows = _table_step(cos, sin, pos, None, max_position, dtype)
        base = _rotate_f32(x, cos, sin, rows, neox, dtype)
        fly = _rotate_onthefly(x, pos, None, base_freq, rotary_dim, max_position, neox, dtype)
        tol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1e-5}[dtype]
        _record(
            f"X3 表即时重算（fp64 角度 → dtype）base=500000 {str(dtype).split('.')[-1]}",
            "rejected", base.numel(), _mismatches(fly, base), _ulp_diff(fly, base),
            int((~torch.isclose(fly.float(), base.float(), atol=tol, rtol=tol)).sum()),
        )


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _fmt(row):
    return (
        f"| {row['item']} | {row['kind']} | {row['compared']} | {row['violations']} | "
        f"{row['rate']:.3e} | {row['tol_violations']} | {row['max_ulp']:.3f} | "
        f"**{row['verdict']}** |"
    )


def main():
    torch.manual_seed(0)
    check_adopted()
    check_rejected()

    header = (
        "| 项 | 类型 | 对照元素数 | 逐位违反数 | 逐位违反率 | 容差违反数 | 最大 ulp 差 | 结论 |\n"
        "|----|------|-----------|-----------|-----------|-----------|------------|------|"
    )
    print("### 采纳项（adopted）\n")
    print(header)
    for row in ADOPTED:
        print(_fmt(row))
    print("\n### 被否决项 / guard 证据（rejected）\n")
    print(header)
    for row in REJECTED:
        print(_fmt(row))

    failed = [r for r in ADOPTED if r["verdict"] != "EQUIV_PASS"]
    print(f"\nadopted items: {len(ADOPTED)}, failed: {len(failed)}")
    print("ADOPTED_VERDICT: " + ("ALL_PASS" if not failed else "HAS_FAILURE"))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
