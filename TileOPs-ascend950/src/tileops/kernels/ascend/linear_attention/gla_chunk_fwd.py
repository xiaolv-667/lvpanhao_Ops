"""Ascend Gated Linear Attention (GLA) chunked forward.

The chunked GLA forward is mathematically the per-token gated recurrence, so this
implementation runs that recurrence directly: one AI core owns a ``(batch, head)``
pair, keeps the ``[dim_k, dim_v]`` state in a shared buffer across the whole
sequence, and walks the tokens.  The state never returns to global memory until
the last token, which is what makes a sequential kernel viable here.
"""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.linear_attention.call_spec import GLAChunkCall, GLAChunkFwdInterface

__all__ = ["AscendGLAChunkFwdKernel"]

# The recurrent state is ``dim_k * dim_v`` float32 elements per core.  A core's
# unified buffer holds the state, the gate row and the output/reduction scratch.
_STATE_BYTES_BUDGET = 128 * 1024
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@functools.lru_cache(maxsize=32)
def _gla_fwd_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    scale: float,
    dtype: str,
    has_initial_state: bool,
):
    """Build the per-token GLA recurrence for one call shape."""
    threads = 128
    init_shape = (batch, heads, dim_k, dim_v) if has_initial_state else (1,)

    @T.prim_func
    def main(
        q: T.Tensor((batch, seq_len, heads, dim_k), dtype),
        k: T.Tensor((batch, seq_len, heads, dim_k), dtype),
        v: T.Tensor((batch, seq_len, heads, dim_v), dtype),
        g: T.Tensor((batch, seq_len, heads, dim_k), dtype),
        initial_state: T.Tensor(init_shape, "float32"),
        o: T.Tensor((batch, seq_len, heads, dim_v), dtype),
        final_state: T.Tensor((batch, heads, dim_k, dim_v), "float32"),
    ):
        with T.Kernel(batch * heads) as bx:
            b = bx // heads
            h = bx % heads
            # State and per-token gate stay in shared memory across the walk.
            state = T.alloc_shared((dim_k, dim_v), "float32")
            gate = T.alloc_shared((dim_k,), "float32")

            with T.SimtVF(threads=threads):
                for kk, vv in T.Parallel(dim_k, dim_v):
                    if has_initial_state:
                        state[kk, vv] = initial_state[b, h, kk, vv]
                    else:
                        state[kk, vv] = 0.0

            for t in T.serial(seq_len):
                with T.SimtVF(threads=threads):
                    for kk in T.Parallel(dim_k):
                        gate[kk] = T.exp(T.cast(g[b, t, h, kk], "float32"))
                    # S <- exp(g) * S + k (x) v
                    for kk, vv in T.Parallel(dim_k, dim_v):
                        state[kk, vv] = state[kk, vv] * gate[kk] + T.cast(
                            k[b, t, h, kk], "float32"
                        ) * T.cast(v[b, t, h, vv], "float32")
                    # o = scale * q^T S
                    prod = T.alloc_fragment((dim_k, dim_v), "float32")
                    reduced = T.alloc_fragment((dim_v,), "float32")
                    for kk, vv in T.Parallel(dim_k, dim_v):
                        prod[kk, vv] = T.cast(q[b, t, h, kk], "float32") * state[kk, vv]
                    T.reduce_sum(prod, reduced, dim=0)
                    for vv in T.Parallel(dim_v):
                        o[b, t, h, vv] = T.cast(reduced[vv] * scale, dtype)

            with T.SimtVF(threads=threads):
                for kk, vv in T.Parallel(dim_k, dim_v):
                    final_state[b, h, kk, vv] = state[kk, vv]

    return tilelang.compile(main, target="ascend")


class AscendGLAChunkFwdKernel(Kernel, GLAChunkFwdInterface):
    """GLA chunked forward as the per-token recurrence, one core a (batch, head)."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: GLAChunkCall) -> Optional[str]:
        if call.dtype not in _SUPPORTED_DTYPES:
            return f"requires float16, bfloat16 or float32, got {call.dtype}"
        if call.batch <= 0 or call.seq_len <= 0 or call.heads <= 0:
            return "requires positive batch, sequence and head counts"
        if call.dim_k <= 0 or call.dim_v <= 0:
            return "requires positive head dimensions"
        if call.dim_k * call.dim_v * 4 > _STATE_BYTES_BUDGET:
            return (
                f"state {call.dim_k} x {call.dim_v} exceeds the per-core "
                f"{_STATE_BYTES_BUDGET}-byte state budget"
            )
        return super().refusal(call)

    def __init__(self, call: GLAChunkCall) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        # A non-positive scale means ``dim_k ** -0.5``, as the op documents.
        self.scale = call.scale if call.scale > 0 else call.dim_k**-0.5
        self._compiled = _gla_fwd_kernel(
            call.batch,
            call.seq_len,
            call.heads,
            call.dim_k,
            call.dim_v,
            self.scale,
            self.dtype_to_str(call.dtype),
            call.has_initial_state,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        call = self.call
        o = torch.empty_like(v)
        final_state = torch.empty(
            (call.batch, call.heads, call.dim_k, call.dim_v),
            dtype=torch.float32,
            device=v.device,
        )
        state = (
            initial_state.to(torch.float32).contiguous()
            if initial_state is not None
            else torch.empty(1, dtype=torch.float32, device=v.device)
        )
        self._compiled(q, k, v, g, state, o, final_state)
        return o, final_state


register_kernel_type("GLAChunkFwdOp", "ascend", AscendGLAChunkFwdKernel)
