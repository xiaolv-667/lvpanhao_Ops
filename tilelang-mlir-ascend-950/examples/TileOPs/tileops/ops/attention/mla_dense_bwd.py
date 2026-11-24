"""MLADenseBwdOp (spec-conformant interface).

Multi-Head Latent Attention (MLA) prefill backward over fixed-length requests,
after the latent is decompressed.  Layout: BSHD.

Every head has its own ``D_qk``-wide query and key and its own ``D_v``-wide
value: the caller has expanded the shared rope part of the key to every head.
``o`` and ``lse`` are the forward's, ``lse`` the natural-log log-sum-exp of each
query row's scaled scores.  Query ``i`` sits at key position
``i + S_kv - S_q``; a query that sees no key has ``lse`` ``-inf`` and a zero
gradient.

Adaptation from GPU (TileOPs) to NPU:

- O1: the device check is ``get_device_backend().is_device_tensor(x)`` (the GPU
  op relied on its kernel-selection protocol for that).
- O3: the ``tune`` parameter is removed together with the GPU ``target``
  argument; the kernel constructor takes ``(batch, heads, dim_qk, dim_v,
  seqlen_q, seqlen_kv, is_causal, dtype, sm_scale=None, config=None,
  device_index=None)``.  The GPU op defers the geometry to each call
  (``kernel_for`` with ``MLABwdCall``); the NPU kernel class needs it up
  front, so it is resolved from the inputs on the first call and the kernels
  are cached per geometry in ``_kernel_cache``.
- O6: op-specific flow preserved unchanged -- contiguous input normalization,
  the ``(q, k, v, o, do, lse)`` order handed to the kernel (the op's own
  argument order is the manifest's ``do, q, k, v, o, lse``), the
  ``(dq, dk, dv)`` return contract.  The GPU validity checks come from the
  manifest ``signature``/``shape_rules`` codegen; the NPU project has no
  manifest codegen, so ``_validate`` reproduces them explicitly (including
  the ``lse`` layout, which the GPU codegen also enforced).
- K5: the GPU ``kernel_types`` declares ``mla_dense_bwd``
  (``MLADenseBwdKernel``) and ``mla_dense_bwd_ws``
  (``MLADenseBwdWSKernel``, SM90 warp-specialized); NPU collapses to the
  architecture-generic dense kernel.
- ``eval_roofline`` is ported from the GPU manifest's
  ``tileops._perf/formulas.py:mla_dense_bwd_roofline`` (the NPU project has no
  ``tileops.perf`` module; roofline arithmetic is device-agnostic).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.attention.mla_dense_bwd import MLADenseBwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["MLADenseBwdOp"]

_INPUT_DTYPES = (torch.float16, torch.bfloat16)


class MLADenseBwdOp(Op):
    """MLA prefill backward over fixed-length requests (BSHD), decompressed latent.

    Args:
        is_causal: Whether a query row reads only keys at or before its own
            position under bottom-right alignment.  Default ``False``, matching
            the forward.
        sm_scale: Score scale, or ``None`` for ``D_qk ** -0.5`` (the query-key
            head dimension including the rope part).
        kernel_map: Optional override for kernel dispatch.

    Raises:
        ValueError: If ``sm_scale`` is given and is not finite and positive.
    """

    def __init__(
        self,
        is_causal: bool = False,
        sm_scale: Optional[float] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        # shape_rules: "not present(sm_scale) or 0 < sm_scale < inf".
        if sm_scale is not None and not (0.0 < sm_scale < float("inf")):
            raise ValueError(f"sm_scale must satisfy 0 < sm_scale < inf; got {sm_scale}")
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        self.dtype: Optional[torch.dtype] = None
        self._geometry: Optional[tuple[int, int, int, int, int, int]] = None
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        # K5: GPU ``kernel_types`` has ``mla_dense_bwd`` (generic) and
        # ``mla_dense_bwd_ws`` (SM90 warp-specialized); NPU keeps the generic
        # dense kernel only.
        return {"mla_dense_bwd": MLADenseBwdKernel}

    def _validate(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        lse: torch.Tensor,
    ) -> None:
        """Reproduce the manifest contract (devices, dtypes, shapes).

        GPU parity: the GPU op inherits these checks from the manifest
        ``signature``/``shape_rules`` codegen; the NPU project has no manifest
        codegen, so they are checked here.
        """
        backend = get_device_backend()
        tensors = (do, q, k, v, o, lse)
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

        for name, tensor in (("do", do), ("q", q), ("k", k), ("v", v), ("o", o)):
            if tensor.dim() != 4:
                raise ValueError(f"{name} must be 4-D (BSHD); got shape {tuple(tensor.shape)}")
        batch, seqlen_q, heads, dim_qk = q.shape
        seqlen_kv = k.shape[1]
        if (k.shape[0], k.shape[2], k.shape[3]) != (batch, heads, dim_qk):
            raise ValueError(
                f"k shape {tuple(k.shape)} must be (B, S_kv, H, D_qk) = "
                f"{(batch, seqlen_kv, heads, dim_qk)}"
            )
        if (v.shape[0], v.shape[2]) != (batch, heads):
            raise ValueError(
                f"v shape {tuple(v.shape)} must be (B, S_kv, H, D_v) = "
                f"{(batch, seqlen_kv, heads, v.shape[3])}"
            )
        dim_v = v.shape[3]
        if seqlen_kv <= 0:
            raise ValueError(f"shape_rules: S_kv > 0; got {seqlen_kv}")
        for name, tensor in (("do", do), ("o", o)):
            expected = (batch, seqlen_q, heads, dim_v)
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} must be (B, S_q, H, D_v) = {expected}"
                )
        expected_lse = (batch, heads, seqlen_q)
        if tuple(lse.shape) != expected_lse:
            raise ValueError(
                f"lse must be (B, H, S_q) = {expected_lse}, the head-major layout FA3 "
                f"returns, not token-major; got {tuple(lse.shape)}"
            )

    def _get_or_create_kernel(self, geometry: tuple) -> Kernel:
        """Return the kernel for *geometry*, building and caching it on first use."""
        kernel = self._kernel_cache.get(geometry)
        if kernel is None:
            batch, heads, dim_qk, dim_v, seqlen_q, seqlen_kv, dtype = geometry
            kernel_cls = self.kernel_map["mla_dense_bwd"]
            kernel = kernel_cls(
                batch,
                heads,
                dim_qk,
                dim_v,
                seqlen_q,
                seqlen_kv,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            do: The gradient of the loss with respect to ``o``, same dtype as ``q``.
            q: Input tensor, dtype ``float16 | bfloat16``.
            k: Input tensor, same dtype as ``q``.
            v: Input tensor, same dtype as ``q``.
            o: The forward output, same dtype as ``q``.
            lse: The forward's natural-log log-sum-exp, ``(B, H, S_q)``, float32.

        Returns:
            ``dq``, ``dk``, ``dv``, as the manifest declares: ``dq`` shaped like
            ``q``, ``dk`` like ``k``, ``dv`` like ``v``.
        """
        self._validate(do, q, k, v, o, lse)
        q, k, v, o, do, lse = (t.contiguous() for t in (q, k, v, o, do, lse))
        batch, seqlen_q, heads, dim_qk = q.shape
        seqlen_kv = k.shape[1]
        geometry = (batch, heads, dim_qk, v.shape[3], seqlen_q, seqlen_kv, q.dtype)
        kernel = self._get_or_create_kernel(geometry)
        self._geometry = geometry[:6]
        self.dtype = q.dtype
        return kernel(q, k, v, o, do, lse)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the last call's geometry.

        Ported from the GPU ``mla_dense_bwd_roofline``: the backward arithmetic
        of the scores each query row sees under bottom-right alignment and the
        causal mask, and every tensor moves once.  Per score five contractions
        (2 FLOPs per multiply-add -- the scores again, ``dQ`` and ``dK`` over
        ``D_qk``, ``dP`` and ``dV`` over ``D_v``) plus 5 elementwise ops for
        ``P`` and ``P * (dP - delta)``; per query row ``delta = rowsum(o * do)``
        over ``D_v``.  ``lse`` is float32, every other tensor the input dtype.
        """
        if self._geometry is None:
            raise RuntimeError("eval_roofline needs the call geometry (call forward first)")
        batch, heads, dim_qk, dim_v, seqlen_q, seqlen_kv = self._geometry
        rows = min(seqlen_q, seqlen_kv) if self.is_causal else seqlen_q
        if self.is_causal:
            scores = rows * seqlen_kv - rows * (rows - 1) // 2
        else:
            scores = seqlen_q * seqlen_kv
        per_score = 2 * (3 * dim_qk + 2 * dim_v) + 5
        flops = batch * heads * (scores * per_score + rows * 2 * dim_v)
        elem_bytes = self.dtype.itemsize
        q_elems = batch * seqlen_q * heads * dim_qk
        qv_elems = batch * seqlen_q * heads * dim_v
        kv_elems = batch * seqlen_kv * heads * dim_qk
        kvv_elems = batch * seqlen_kv * heads * dim_v
        moved = 2 * (q_elems + qv_elems + kv_elems + kvv_elems) * elem_bytes
        moved += batch * heads * seqlen_q * 4  # lse, float32
        return int(flops), int(moved)
