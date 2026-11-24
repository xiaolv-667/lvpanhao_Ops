"""Ascend Gated DeltaNet (GDN) inference forward.

The gated delta rule is evaluated as the token recurrence it defines, which is
what the op's chunked solver computes in closed form: one AI core owns a
``(sequence, value head)`` pair, keeps the ``[dim_k, dim_v]`` state in shared
memory, and walks the sequence.  Equal-length and packed calls share the walk;
a packed core reads its token span from ``cu_seqlens`` and guards the tail.
The optional per-token transforms (Q/K L2 normalisation, the raw gate and the
beta sigmoid) run in the same tile, before the recurrence step.
"""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.linear_attention.call_spec import GDNCall, GDNFwdInterface

__all__ = ["AscendGDNFwdKernel"]

# The recurrent state is ``dim_k * dim_v`` float32 elements held per core.
_STATE_BYTES_BUDGET = 128 * 1024
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


@functools.lru_cache(maxsize=32)
def _gdn_fwd_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    value_heads: int,
    dim_k: int,
    dim_v: int,
    scale: float,
    num_sequences: int,
    dtype: str,
    has_initial_state: bool,
    varlen: bool,
    state_v_first: bool,
    l2norm: bool,
    raw_gate: bool,
    beta_sigmoid: bool,
    neg_eigval: bool,
):
    """Build the token recurrence for one call shape and transform set."""
    threads = 128
    group = value_heads // heads
    cu_shape = (num_sequences + 1,) if varlen else (1,)
    init_shape = (
        (num_sequences, value_heads, dim_v, dim_k)
        if has_initial_state and state_v_first
        else (num_sequences, value_heads, dim_k, dim_v)
        if has_initial_state
        else (1,)
    )
    state_shape = (
        (num_sequences, value_heads, dim_v, dim_k)
        if state_v_first
        else (num_sequences, value_heads, dim_k, dim_v)
    )
    beta_scale = 2.0 if neg_eigval else 1.0
    # A one-token call leaves a degenerate one-iteration body beside the wide
    # reductions; the span guard skips the padded extra step without touching a
    # tensor, so the walk runs at least two iterations.
    walk = max(seq_len, 2)
    # The lowering keeps a reduced scalar beside the wide reductions only for a
    # multi-token walk; a single-token call reads it from the reduction fragment.
    norm_through_shared = seq_len > 1

    @T.prim_func
    def main(
        q: T.Tensor((batch, seq_len, heads, dim_k), dtype),
        k: T.Tensor((batch, seq_len, heads, dim_k), dtype),
        v: T.Tensor((batch, seq_len, value_heads, dim_v), dtype),
        g: T.Tensor((batch, seq_len, value_heads), dtype),
        beta: T.Tensor((batch, seq_len, value_heads), dtype),
        initial_state: T.Tensor(init_shape, "float32"),
        cu_seqlens: T.Tensor(cu_shape, "int64"),
        A_log: T.Tensor((value_heads,), "float32"),
        dt_bias: T.Tensor((value_heads,), "float32"),
        o: T.Tensor((batch, seq_len, value_heads, dim_v), dtype),
        final_state: T.Tensor(state_shape, "float32"),
    ):
        with T.Kernel(num_sequences * value_heads) as bx:
            seq = bx // value_heads
            hv = bx % value_heads
            hk = hv // group
            state = T.alloc_shared((dim_k, dim_v), "float32")
            step = T.alloc_shared((dim_v,), "float32")
            # exp(g), beta and the two inverse norms of the current token, padded
            # to the 32-byte shared-allocation alignment.
            scalars = T.alloc_shared((8,), "float32")
            span = T.alloc_shared((8,), "int32")
            if raw_gate:
                # The per-head rate and time bias, staged once: a global scalar read
                # inside the wide token tile miscompiles in the current lowering.
                params = T.alloc_shared((8,), "float32")

            with T.SimtVF(threads=threads):
                if varlen:
                    span[0] = T.cast(cu_seqlens[seq], "int32")
                    span[1] = T.cast(cu_seqlens[seq + 1], "int32")
                else:
                    # Each batch row is its own sequence, so the token index is
                    # sequence-local and the guard is trivially true.
                    span[0] = 0
                    span[1] = seq_len
                if raw_gate:
                    params[0] = T.exp(A_log[hv])
                    params[1] = dt_bias[hv]
                for kk, vv in T.Parallel(dim_k, dim_v):
                    if has_initial_state:
                        if state_v_first:
                            state[kk, vv] = initial_state[seq, hv, vv, kk]
                        else:
                            state[kk, vv] = initial_state[seq, hv, kk, vv]
                    else:
                        state[kk, vv] = 0.0

            for t in T.serial(walk):
                with T.SimtVF(threads=threads):
                    pos = span[0] + t
                    if pos < span[1]:
                        b = 0 if varlen else seq
                        if raw_gate:
                            shifted = T.cast(g[b, pos, hv], "float32") + params[1]
                            softplus = T.if_then_else(
                                shifted > 20,
                                shifted,
                                T.log(1.0 + T.exp(T.min(shifted, 20.0))),
                            )
                            scalars[0] = T.exp(-params[0] * softplus)
                        else:
                            scalars[0] = T.exp(T.cast(g[b, pos, hv], "float32"))
                        if beta_sigmoid:
                            scalars[1] = (
                                1.0
                                / (1.0 + T.exp(-T.cast(beta[b, pos, hv], "float32")))
                                * beta_scale
                            )
                        else:
                            scalars[1] = T.cast(beta[b, pos, hv], "float32")
                        if l2norm:
                            knorm = T.alloc_fragment((dim_k, 1), "float32")
                            kred = T.alloc_fragment((1,), "float32")
                            for kk in T.Parallel(dim_k):
                                kv = T.cast(k[b, pos, hk, kk], "float32")
                                knorm[kk, 0] = kv * kv
                            T.reduce_sum(knorm, kred, dim=0)
                            qnorm = T.alloc_fragment((dim_k, 1), "float32")
                            qred = T.alloc_fragment((1,), "float32")
                            for kk in T.Parallel(dim_k):
                                qv = T.cast(q[b, pos, hk, kk], "float32")
                                qnorm[kk, 0] = qv * qv
                            T.reduce_sum(qnorm, qred, dim=0)
                            if norm_through_shared:
                                scalars[2] = 1.0 / T.sqrt(kred[0] + 1e-6)
                                scalars[3] = 1.0 / T.sqrt(qred[0] + 1e-6)
                                k_norm = scalars[2]
                                q_norm = scalars[3]
                            else:
                                k_norm = 1.0 / T.sqrt(kred[0] + 1e-6)
                                q_norm = 1.0 / T.sqrt(qred[0] + 1e-6)
                        else:
                            k_norm = 1.0
                            q_norm = 1.0

                        # old = k^T S; the value written is beta * (v - exp(g) * old).
                        outer = T.alloc_fragment((dim_k, dim_v), "float32")
                        old = T.alloc_fragment((dim_v,), "float32")
                        for kk, vv in T.Parallel(dim_k, dim_v):
                            outer[kk, vv] = (
                                k_norm * T.cast(k[b, pos, hk, kk], "float32") * state[kk, vv]
                            )
                        T.reduce_sum(outer, old, dim=0)
                        for vv in T.Parallel(dim_v):
                            step[vv] = scalars[1] * (
                                T.cast(v[b, pos, hv, vv], "float32") - scalars[0] * old[vv]
                            )
                        # S = exp(g) * S + k (x) step
                        for kk, vv in T.Parallel(dim_k, dim_v):
                            state[kk, vv] = (
                                scalars[0] * state[kk, vv]
                                + k_norm * T.cast(k[b, pos, hk, kk], "float32") * step[vv]
                            )
                        # o = scale * q^T S
                        prod = T.alloc_fragment((dim_k, dim_v), "float32")
                        reduced = T.alloc_fragment((dim_v,), "float32")
                        for kk, vv in T.Parallel(dim_k, dim_v):
                            prod[kk, vv] = (
                                q_norm * T.cast(q[b, pos, hk, kk], "float32") * state[kk, vv]
                            )
                        T.reduce_sum(prod, reduced, dim=0)
                        for vv in T.Parallel(dim_v):
                            o[b, pos, hv, vv] = T.cast(reduced[vv] * scale, dtype)

            with T.SimtVF(threads=threads):
                for kk, vv in T.Parallel(dim_k, dim_v):
                    if state_v_first:
                        final_state[seq, hv, vv, kk] = state[kk, vv]
                    else:
                        final_state[seq, hv, kk, vv] = state[kk, vv]

    return tilelang.compile(main, target="ascend")


class AscendGDNFwdKernel(Kernel, GDNFwdInterface):
    """Gated delta rule as its token recurrence, one core a (sequence, value head)."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: GDNCall) -> Optional[str]:
        if call.dtype not in _SUPPORTED_DTYPES:
            return f"requires float16 or bfloat16, got {call.dtype}"
        if call.batch <= 0 or call.seq_len <= 0:
            return "requires a positive batch and token count"
        if call.heads <= 0 or call.value_heads <= 0:
            return "requires positive key- and value-head counts"
        if call.value_heads % call.heads != 0:
            return (
                f"requires value heads a multiple of key heads, got "
                f"{call.value_heads} and {call.heads}"
            )
        if call.dim_k <= 0 or call.dim_v <= 0:
            return "requires positive head dimensions"
        if call.value_heads == 1 and call.dim_k == 128:
            # The 128-wide state beside a single-element head axis miscompiles in
            # the current Ascend tile lowering (device error 507035).
            return "does not serve a single value head over a 128-wide state"
        if call.dim_k * call.dim_v * 4 > _STATE_BYTES_BUDGET:
            return (
                f"state {call.dim_k} x {call.dim_v} exceeds the per-core "
                f"{_STATE_BYTES_BUDGET}-byte state budget"
            )
        if call.varlen and call.batch != 1:
            return "requires a packed call with one token axis"
        return super().refusal(call)

    def __init__(self, call: GDNCall) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self._compiled = _gdn_fwd_kernel(
            call.batch,
            call.seq_len,
            call.heads,
            call.value_heads,
            call.dim_k,
            call.dim_v,
            call.scale,
            call.num_sequences,
            self.dtype_to_str(call.dtype),
            call.has_initial_state,
            call.varlen,
            call.state_v_first,
            call.l2norm,
            call.gate_in_kernel,
            call.beta_sigmoid,
            call.allow_neg_eigval,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        call = self.call
        del cu_seqlens_cpu
        o = torch.empty_like(v)
        final_state = torch.empty(
            (call.num_sequences, call.value_heads, call.dim_v, call.dim_k)
            if call.state_v_first
            else (call.num_sequences, call.value_heads, call.dim_k, call.dim_v),
            dtype=torch.float32,
            device=v.device,
        )
        state = (
            initial_state.to(torch.float32).contiguous()
            if initial_state is not None
            else torch.empty(1, dtype=torch.float32, device=v.device)
        )
        offsets = (
            cu_seqlens.contiguous()
            if cu_seqlens is not None
            else torch.empty(1, dtype=torch.int64, device=v.device)
        )
        rates = (
            A_log.to(torch.float32).contiguous()
            if A_log is not None
            else torch.zeros(call.value_heads, dtype=torch.float32, device=v.device)
        )
        steps = (
            dt_bias.to(torch.float32).contiguous()
            if dt_bias is not None
            else torch.zeros(call.value_heads, dtype=torch.float32, device=v.device)
        )
        self._compiled(q, k, v, g, beta, state, offsets, rates, steps, o, final_state)
        return o, final_state


register_kernel_type("GDNFwdOp", "ascend", AscendGDNFwdKernel)
