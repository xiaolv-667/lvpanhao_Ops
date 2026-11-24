"""MLAVarlenBwdOp (spec-conformant interface).

Multi-Head Latent Attention (MLA) prefill backward over packed requests, after
the latent is decompressed.  Layout: THD.

Every head has its own ``D_qk``-wide query and key and its own ``D_v``-wide
value: the caller has expanded the shared rope part of the key to every head.
``o`` and ``lse`` are the forward's, ``lse`` the natural-log log-sum-exp of each
query row's scaled scores, head-major ``(H, T_q)``.  Request ``b`` owns rows
``cu_seqlens_q[b]:cu_seqlens_q[b + 1]`` of ``q``/``do`` and rows
``cu_seqlens_kv[b]:cu_seqlens_kv[b + 1]`` of ``k``/``v``; query ``i`` of a
request sits at key position ``i + kv_len - q_len``, and a query that sees no key
has ``lse`` ``-inf`` and a zero gradient.  ``dq``/``dk``/``dv`` are written back
at those same packed offsets.

This is :class:`MLADenseBwdOp`'s packed form: both ops share the GPU kernel
factory set (``_bwd_kernel`` / ``_delta_kernel`` / ``_dq_land_kernel``) and
differ only in the scheduler (``GroupTiling`` per request), the ``lse`` layout
(head-major ``(H, T_q)`` here, ``(B, H, S_q)`` there) and the ``cu_seqlens``
offset arithmetic on every load and store.

Adaptation from GPU (TileOPs) to NPU:

- O1: the device check is ``get_device_backend().is_device_tensor(x)`` (the GPU
  op relied on its kernel-selection protocol for that).
- O3: the ``tune`` parameter is removed together with the GPU ``target``
  argument; the kernel constructor takes ``(batch, heads, dim_qk, dim_v,
  is_causal, dtype, sm_scale=None, config=None, device_index=None)``.  The GPU
  op defers the geometry to each call (``kernel_for`` with ``MLABwdCall``); the
  NPU kernel class needs it up front, so it is resolved from the inputs on the
  first call and the kernels are cached per geometry in ``_kernel_cache``.
- O6: op-specific flow preserved unchanged -- contiguous input normalization,
  the ``(q, k, v, o, do, lse, cu_seqlens_q, cu_seqlens_kv)`` order handed to
  the kernel (the op's own argument order is the manifest's ``do, q, k, v, o,
  lse, cu_seqlens_q, cu_seqlens_kv``), the ``(dq, dk, dv)`` return contract.
  The GPU validity checks come from the manifest ``signature``/``shape_rules``
  codegen; the NPU project has no manifest codegen, so ``_validate`` reproduces
  them explicitly (including the head-major ``lse`` layout, which the GPU
  codegen also enforced).
- ``validate_inputs`` is preserved from the GPU op: when set, each offset vector
  is checked on the host (starts at 0, never decreases, ends at its packed
  length).  Also used for the roofline source -- the per-request lengths are
  read back from ``cu_seqlens`` lazily in ``eval_roofline``, never on the hot
  path of ``forward``.
- K5: the GPU ``kernel_types`` declares ``mla_varlen_bwd``
  (``MLAVarlenBwdKernel``) and ``mla_varlen_bwd_ws``
  (``MLAVarlenBwdWSKernel``, SM90 warp-specialized); NPU collapses to the
  architecture-generic kernel.
- ``eval_roofline`` is ported from the GPU manifest's
  ``tileops._perf/formulas.py:mla_varlen_bwd_roofline`` (the NPU project has no
  ``tileops.perf`` module; roofline arithmetic is device-agnostic): the dense
  formula summed over the requests the offsets carry.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.attention.mla_varlen_bwd import MLAVarlenBwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["MLAVarlenBwdOp"]

_INPUT_DTYPES = (torch.float16, torch.bfloat16)


def _is_prefix_offsets(offsets: list[int], total: int) -> bool:
    """Whether *offsets* starts at 0, never decreases and ends at *total*."""
    if not offsets or offsets[0] != 0 or offsets[-1] != total:
        return False
    return all(lo <= hi for lo, hi in zip(offsets, offsets[1:], strict=False))


class MLAVarlenBwdOp(Op):
    """MLA prefill backward over packed requests (THD), decompressed latent.

    Args:
        is_causal: Whether a query row reads only keys at or before its own
            position under bottom-right alignment.  Default ``False``, matching
            the forward.
        sm_scale: Score scale, or ``None`` for ``D_qk ** -0.5`` (the query-key
            head dimension including the rope part).
        validate_inputs: Check on the host that each offset vector starts at 0,
            never decreases and ends at its packed length.
        kernel_map: Optional override for kernel dispatch.

    Raises:
        ValueError: If ``sm_scale`` is given and is not finite and positive.
    """

    def __init__(
        self,
        is_causal: bool = False,
        sm_scale: Optional[float] = None,
        *,
        validate_inputs: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        # shape_rules: "not present(sm_scale) or 0 < sm_scale < inf".
        if sm_scale is not None and not (0.0 < sm_scale < float("inf")):
            raise ValueError(f"sm_scale must satisfy 0 < sm_scale < inf; got {sm_scale}")
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        self.validate_inputs = validate_inputs
        self.dtype: Optional[torch.dtype] = None
        self._geometry: Optional[tuple[int, int, int]] = None
        self._offsets: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        # K5: GPU ``kernel_types`` has ``mla_varlen_bwd`` (generic) and
        # ``mla_varlen_bwd_ws`` (SM90 warp-specialized); NPU keeps the generic
        # packed kernel only.
        return {"mla_varlen_bwd": MLAVarlenBwdKernel}

    def _validate(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        lse: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ) -> None:
        """Reproduce the manifest contract (devices, dtypes, shapes, offsets).

        GPU parity: the GPU op inherits these checks from the manifest
        ``signature``/``shape_rules`` codegen; the NPU project has no manifest
        codegen, so they are checked here.  The offset *values* are only read
        when ``validate_inputs`` is set (the GPU op's own opt-in check), so a
        plain call never syncs the device.
        """
        backend = get_device_backend()
        tensors = (do, q, k, v, o, lse, cu_seqlens_q, cu_seqlens_kv)
        if not all(backend.is_device_tensor(t) for t in tensors):
            raise ValueError(
                f"{type(self).__name__} expects {backend.name} tensors; "
                "move every input to the device backend first"
            )

        if q.dtype not in _INPUT_DTYPES:
            raise ValueError(
                f"unsupported input dtype {q.dtype}; "
                f"expected one of {', '.join(str(d) for d in _INPUT_DTYPES)}"
            )
        for name, tensor in (("do", do), ("k", k), ("v", v), ("o", o)):
            if tensor.dtype is not q.dtype:
                raise ValueError(f"{name} dtype {tensor.dtype} must match q dtype {q.dtype}")
        if lse.dtype is not torch.float32:
            raise ValueError(f"lse dtype {lse.dtype} must be float32")
        for name, tensor in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_kv", cu_seqlens_kv)):
            if tensor.dtype is not torch.int32:
                raise ValueError(f"{name} dtype {tensor.dtype} must be int32")

        for name, tensor in (("do", do), ("q", q), ("k", k), ("v", v), ("o", o)):
            if tensor.dim() != 3:
                raise ValueError(f"{name} must be 3-D (THD); got shape {tuple(tensor.shape)}")
        total_q, heads, dim_qk = q.shape
        total_kv = k.shape[0]
        if (k.shape[1], k.shape[2]) != (heads, dim_qk):
            raise ValueError(
                f"k shape {tuple(k.shape)} must be (T_kv, H, D_qk) = "
                f"{(total_kv, heads, dim_qk)}"
            )
        if (v.shape[1],) != (heads,):
            raise ValueError(
                f"v shape {tuple(v.shape)} must be (T_kv, H, D_v) = "
                f"{(total_kv, heads, v.shape[2])}"
            )
        dim_v = v.shape[2]
        if total_kv <= 0:
            raise ValueError(f"packed key length must be positive; got T_kv {total_kv}")
        for name, tensor in (("do", do), ("o", o)):
            expected = (total_q, heads, dim_v)
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} must be (T_q, H, D_v) = {expected}"
                )
        expected_lse = (heads, total_q)
        if tuple(lse.shape) != expected_lse:
            raise ValueError(
                f"lse must be (H, T_q) = {expected_lse}, the head-major layout FA3 returns "
                f"for packed requests, not token-major (T_q, H); got {tuple(lse.shape)}"
            )

        if cu_seqlens_q.dim() != 1 or cu_seqlens_kv.dim() != 1:
            raise ValueError(
                "cu_seqlens_q and cu_seqlens_kv must be 1-D (B + 1,) offset vectors; got "
                f"{tuple(cu_seqlens_q.shape)} and {tuple(cu_seqlens_kv.shape)}"
            )
        if cu_seqlens_q.shape != cu_seqlens_kv.shape:
            raise ValueError(
                f"cu_seqlens_q shape {tuple(cu_seqlens_q.shape)} and cu_seqlens_kv shape "
                f"{tuple(cu_seqlens_kv.shape)} must both be (B + 1,)"
            )
        if cu_seqlens_q.shape[0] < 2:
            raise ValueError(
                f"cu_seqlens_q must hold at least 2 offsets (B + 1, B >= 1); got "
                f"{cu_seqlens_q.shape[0]}"
            )

        if self.validate_inputs:
            q_offsets = [int(x) for x in cu_seqlens_q.tolist()]
            kv_offsets = [int(x) for x in cu_seqlens_kv.tolist()]
            if not _is_prefix_offsets(q_offsets, total_q):
                raise ValueError(
                    f"cu_seqlens_q {q_offsets} must start at 0, never decrease and end at "
                    f"T_q = {total_q}"
                )
            if not _is_prefix_offsets(kv_offsets, total_kv):
                raise ValueError(
                    f"cu_seqlens_kv {kv_offsets} must start at 0, never decrease and end at "
                    f"T_kv = {total_kv}"
                )

    def _get_or_create_kernel(self, geometry: tuple) -> Kernel:
        """Return the kernel for *geometry*, building and caching it on first use."""
        kernel = self._kernel_cache.get(geometry)
        if kernel is None:
            batch, heads, dim_qk, dim_v, dtype = geometry
            kernel_cls = self.kernel_map["mla_varlen_bwd"]
            kernel = kernel_cls(
                batch,
                heads,
                dim_qk,
                dim_v,
                self.is_causal,
                dtype,
                sm_scale=self.sm_scale,
            )
            self._kernel_cache[geometry] = kernel
        self.kernel = kernel
        return kernel

    def forward(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        lse: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            do: The gradient of the loss with respect to ``o``, same dtype as ``q``.
            q: Input tensor, dtype ``float16 | bfloat16``.
            k: Input tensor, same dtype as ``q``.
            v: Input tensor, same dtype as ``q``.
            o: The forward output, same dtype as ``q``.
            lse: The forward's natural-log log-sum-exp, ``(H, T_q)``, float32.
            cu_seqlens_q: Packed request offsets into ``q``, int32 ``(B + 1,)``.
            cu_seqlens_kv: Packed request offsets into ``k`` and ``v``, int32 ``(B + 1,)``.

        Returns:
            ``dq``, ``dk``, ``dv``, as the manifest declares: ``dq`` shaped like
            ``q``, ``dk`` like ``k``, ``dv`` like ``v``.
        """
        self._validate(do, q, k, v, o, lse, cu_seqlens_q, cu_seqlens_kv)
        q, k, v, o, do, lse, cu_seqlens_q, cu_seqlens_kv = (
            t.contiguous()
            for t in (q, k, v, o, do, lse, cu_seqlens_q, cu_seqlens_kv)
        )
        batch = cu_seqlens_q.shape[0] - 1
        heads, dim_qk, dim_v = q.shape[1], q.shape[2], v.shape[2]
        geometry = (batch, heads, dim_qk, dim_v, q.dtype)
        kernel = self._get_or_create_kernel(geometry)
        self._geometry = (heads, dim_qk, dim_v)
        self._offsets = (cu_seqlens_q, cu_seqlens_kv)
        self.dtype = q.dtype
        return kernel(q, k, v, o, do, lse, cu_seqlens_q, cu_seqlens_kv)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the last call's packed geometry.

        Ported from the GPU ``mla_varlen_bwd_roofline``: the dense backward
        arithmetic summed over the requests the offsets carry.  Per score five
        contractions (2 FLOPs per multiply-add -- the scores again, ``dQ`` and
        ``dK`` over ``D_qk``, ``dP`` and ``dV`` over ``D_v``) plus 5 elementwise
        ops for ``P`` and ``P * (dP - delta)``; per query row
        ``delta = rowsum(o * do)`` over ``D_v``.  Only the scores a request's
        mask admits are counted (bottom-right alignment).  ``lse`` is float32,
        both offset vectors are int32, every other tensor the input dtype, and
        each moves once.
        """
        if self._geometry is None or self._offsets is None:
            raise RuntimeError("eval_roofline needs the call geometry (call forward first)")
        heads, dim_qk, dim_v = self._geometry
        q_offsets = [int(x) for x in self._offsets[0].tolist()]
        kv_offsets = [int(x) for x in self._offsets[1].tolist()]
        q_lens = [hi - lo for lo, hi in zip(q_offsets, q_offsets[1:], strict=False)]
        kv_lens = [hi - lo for lo, hi in zip(kv_offsets, kv_offsets[1:], strict=False)]

        scores = rows = 0
        for q_len, kv_len in zip(q_lens, kv_lens, strict=False):
            if kv_len <= 0:
                continue
            if self.is_causal:
                visible = min(q_len, kv_len)
                scores += visible * kv_len - visible * (visible - 1) // 2
                rows += visible
            else:
                scores += q_len * kv_len
                rows += q_len
        per_score = 2 * (3 * dim_qk + 2 * dim_v) + 5
        flops = heads * (scores * per_score + rows * 2 * dim_v)

        total_q, total_kv = sum(q_lens), sum(kv_lens)
        elem_bytes = self.dtype.itemsize
        q_elems = total_q * heads * dim_qk
        qv_elems = total_q * heads * dim_v
        kv_elems = total_kv * heads * dim_qk
        kvv_elems = total_kv * heads * dim_v
        moved = 2 * (q_elems + qv_elems + kv_elems + kvv_elems) * elem_bytes
        moved += heads * total_q * 4  # lse, float32
        moved += 2 * (len(q_lens) + 1) * 4  # both cu_seqlens, int32
        return int(flops), int(moved)
