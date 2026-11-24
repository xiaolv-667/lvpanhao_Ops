"""Ascend complex-to-complex power-of-two FFT.

Power-of-two lengths up to what the Unified Buffer holds run as one Stockham autosort
radix-2 decimation-in-frequency transform: every stage is a pair of elementwise phases in
the Unified Buffer (butterfly, then the block-interleave read-index mapping), so no bit
reversal and no intermediate global traffic is needed.  Longer lengths take a two-factor
four-step decomposition, each factor a Stockham transform of its own and the cross term a
single twiddle multiplication.

Complex values are interleaved ``(batch, n, 2)`` float tensors; the twiddle table is the
full-circle ``exp(-2*pi*i*e/n)`` table built once on the host in float64 and cast to the
real dtype.

Ascend 950 has no float64 path (neither the vector core nor aclnn's FFT), so complex128
is refused as a declared gap, matching the manifest's excluded-by-gap decision for it.  A
complex64 length past the two-factor limit falls back to the vendor library transform.
"""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.fft import FFTC2CCall, FFTC2CFwdInterface
from tileops.kernels.kernel_base import Kernel

__all__ = ["AscendFFTC2CKernel"]

# Unified-buffer transforms: 16 * nf bytes of float32 ping-pong, or 32 * nf for float64.
_MAX_FLOAT32_LEN = 1 << 13
_MAX_FLOAT64_LEN = 0  # Ascend 950 has no float64 path; complex128 is refused.
_BLOCKS = 64


def _pad(length: int, real_dtype: str) -> int:
    """Round a Unified Buffer length up to its 32-byte vector."""
    align = 8 if real_dtype == "float32" else 4
    return (length + align - 1) // align * align


def _split_factors(n: int, max_len: int) -> tuple[int, int]:
    """``(n1, n2)`` with ``n1 * n2 == n`` and both at most *max_len*."""
    log2n = n.bit_length() - 1
    half = (log2n + 1) // 2
    n1 = 1 << min(half, max_len.bit_length() - 1)
    if n1 > n:
        n1 = n
    n2 = n // n1
    if n2 > max_len:
        raise ValueError(f"n = {n} needs a factor above the {max_len} Unified Buffer limit")
    return n1, n2


def _make_transform(
    nf: int,
    inner: int,
    table_period: int,
    real_dtype: str,
    four_step: bool,
):
    """One launch: a length-``nf`` transform of ``inner`` strided vectors per row.

    With ``four_step`` the input for vector ``b`` is ``x[row, a*inner + b]``, the output is
    ``t[row, b*nf + k]``, and each output is multiplied by ``W_table_period^(b*k)``.
    Without it ``inner == 1`` and the transform is over one contiguous row.
    """
    L = nf.bit_length() - 1
    nh = nf // 2
    stride = table_period // nf
    pad = _pad(nf, real_dtype)
    threads = min(256, max(32, pad))
    grid = _BLOCKS

    batch = T.dynamic("batch")

    @T.prim_func
    def main(
        x: T.Tensor((batch, table_period, 2), real_dtype),
        table: T.Tensor((table_period, 2), real_dtype),
        y: T.Tensor((batch, table_period, 2), real_dtype),
    ):
        with T.Kernel(grid) as bx:
            u_r = T.alloc_shared((pad,), real_dtype)
            u_i = T.alloc_shared((pad,), real_dtype)
            v_r = T.alloc_shared((pad,), real_dtype)
            v_i = T.alloc_shared((pad,), real_dtype)
            tasks = batch * inner
            for step in T.serial(T.ceildiv(tasks, grid)):
                task = step * grid + bx
                if task < tasks:
                    row = task // inner
                    b = task % inner
                    with T.SimtVF(threads=threads):
                        for a in T.Parallel(nf):
                            u_r[a] = x[row, a * inner + b, 0]
                            u_i[a] = x[row, a * inner + b, 1]
                    # Mid stages: butterfly, twiddle, block-interleave back into u.
                    for q in T.serial(L - 1):
                        with T.SimtVF(threads=threads):
                            for j in T.Parallel(nh):
                                ar = u_r[j]
                                ai = u_i[j]
                                br = u_r[j + nh]
                                bi = u_i[j + nh]
                                v_r[j] = ar + br
                                v_i[j] = ai + bi
                                # Twiddle of the difference half: W_nf^((j) * table_period/nf).
                                s = 1 << q
                                e = (j // s) * s * stride
                                dr = ar - br
                                di = ai - bi
                                wr = table[e, 0]
                                wi = table[e, 1]
                                v_r[j + nh] = dr * wr - di * wi
                                v_i[j + nh] = dr * wi + di * wr
                        with T.SimtVF(threads=threads):
                            for j in T.Parallel(nf):
                                s = 1 << q
                                x_ = j // s
                                k_ = j // (2 * s)
                                r_ = j % s
                                u_r[j] = v_r[(x_ % 2) * nh + k_ * s + r_]
                                u_i[j] = v_i[(x_ % 2) * nh + k_ * s + r_]
                    # Final stage: the two halves are the natural-order result.
                    with T.SimtVF(threads=threads):
                        for j in T.Parallel(nh):
                            ar = u_r[j]
                            ai = u_i[j]
                            br = u_r[j + nh]
                            bi = u_i[j + nh]
                            v_r[j] = ar + br
                            v_i[j] = ai + bi
                            v_r[j + nh] = ar - br
                            v_i[j + nh] = ai - bi
                    with T.SimtVF(threads=threads):
                        for k in T.Parallel(nf):
                            if four_step:
                                e = b * k
                                wr = table[e, 0]
                                wi = table[e, 1]
                                y[row, b * nf + k, 0] = v_r[k] * wr - v_i[k] * wi
                                y[row, b * nf + k, 1] = v_r[k] * wi + v_i[k] * wr
                            else:
                                # Fixed-index (kernel B / one-CTA) output: k1 + n1 * k2.
                                y[row, b + inner * k, 0] = v_r[k]
                                y[row, b + inner * k, 1] = v_i[k]

    return main


@functools.lru_cache(maxsize=64)
def _compile_transform(nf: int, inner: int, table_period: int, real_dtype: str, four_step: bool):
    return tilelang.compile(
        _make_transform(nf, inner, table_period, real_dtype, four_step),
        target="ascend",
    )


@functools.lru_cache(maxsize=16)
def _table(n: int, dtype: torch.dtype):
    """Full-circle ``exp(-2*pi*i*e/n)`` table, float64 built then cast."""
    real = torch.float32 if dtype == torch.complex64 else torch.float64
    idx = torch.arange(n, dtype=torch.float64)
    angle = -2.0 * torch.pi * idx / n
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=1).to(real)


class AscendFFTC2CKernel(Kernel, FFTC2CFwdInterface):
    """One Stockham transform, or a two-factor four-step decomposition, on Ascend 950."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: FFTC2CCall) -> Optional[str]:
        if call.dtype == torch.complex128:
            # Neither the vector core nor aclnn's FFT has a float64 path on Ascend 950.
            return "complex128 needs a float64 path, which Ascend 950 does not provide"
        if call.dtype != torch.complex64:
            return "requires complex64 input"
        if call.n <= 0 or call.n & (call.n - 1):
            return "requires a positive power-of-two length"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: FFTC2CCall):
        index = call.device.index if call.device is not None else None
        return (call.n, call.dtype, index), lambda: cls(
            call.n, call.dtype, device_index=index
        )

    def __init__(self, n: int, dtype: torch.dtype, device_index: Optional[int] = None) -> None:
        super().__init__(device_index=device_index)
        self.n = n
        self.dtype = dtype
        self.real = "float32" if dtype == torch.complex64 else "float64"
        max_len = _MAX_FLOAT32_LEN if dtype == torch.complex64 else _MAX_FLOAT64_LEN
        self.fallback = max_len == 0 or n > max_len * max_len
        if self.fallback:
            self.kernels = ()
            self.factors = ()
        elif n <= max_len:
            self.kernels = (_compile_transform(n, 1, n, self.real, False),)
            self.factors = (n,)
        else:
            n1, n2 = _split_factors(n, max_len)
            self.factors = (n1, n2)
            self.kernels = (
                _compile_transform(n1, n2, n, self.real, True),
                _compile_transform(n2, n1, n, self.real, False),
            )
        self._tables: dict = {}

    def _table_for(self, device: torch.device):
        index = device.index
        key = (self.n, self.dtype, index)
        table = self._tables.get(key)
        if table is None:
            table = _table(self.n, self.dtype).to(device).contiguous()
            self._tables[key] = table
        return table

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_devices(input=x)
        if self.fallback:
            # A complex64 length past the two-factor limit: the vendor library transform
            # stands in for the TileLang kernel.
            return torch.fft.fft(x, dim=-1)
        x = x.resolve_conj()
        n = self.n
        out_shape = x.shape
        real = torch.view_as_real(x.contiguous()).reshape(-1, n, 2)
        batch = real.shape[0]
        table = self._table_for(x.device)
        y = torch.empty_like(real)
        if len(self.kernels) == 1:
            self.kernels[0](real, table, y)
        else:
            t = torch.empty_like(real)
            self.kernels[0](real, table, t)
            self.kernels[1](t, table, y)
        return torch.view_as_complex(y.reshape(*out_shape, 2))


register_kernel_type("FFTC2CFwdOp", "ascend", AscendFFTC2CKernel)
