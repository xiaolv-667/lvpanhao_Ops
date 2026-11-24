"""PagedKVCacheWriteFwdOp (spec-conformant interface).

Scatters new K/V rows into the page pools in place, the write half of a
paged KV cache.

Adaptation from GPU (TileOPs) to NPU:

- O1: CUDA device checks (``x.is_cuda``) become
  ``get_device_backend().is_device_tensor(x)``.
- O3: the ``tune`` parameter is removed; the kernel constructor takes
  ``(heads_kv, dim, num_pages, page_size, dtype, cache_dtype, config=None)``.
  The GPU op defers the cache geometry to each call (``kernel_for`` with a
  call-spec); the NPU kernel class needs it up front, so the geometry is
  resolved from the inputs on the first call and the kernels are cached
  per geometry in ``_kernel_cache``.  The same geometry may be supplied
  eagerly through the constructor keywords when an ``eval_roofline()``
  call is expected before the first ``forward`` (benchmarks).
- O6: op-specific flow preserved unchanged -- validity checks, contiguous
  input normalization, the flat-row view handed to the kernel, the skip of
  an empty call, and the ``None`` return (the op mutates both pools).  The
  GPU validity checks come from the manifest ``shape_rules``; the NPU
  project has no manifest codegen, so ``_validate`` reproduces them
  explicitly.
- ``eval_roofline`` is ported from the GPU manifest's
  ``tileops._perf/formulas.py:paged_kv_cache_write_roofline`` (device-
  agnostic arithmetic): only the tokens whose slot is not ``-1`` are read
  and written, and an FP8 cache scales and saturates each stored value.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.attention.paged_cache_write import PagedKVCacheWriteKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["PagedKVCacheWriteFwdOp"]

_INPUT_DTYPES = (torch.float16, torch.bfloat16)
_CACHE_DTYPES = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)


class PagedKVCacheWriteFwdOp(Op):
    """Scatter new K/V rows into the page pools in place, the write half of a paged cache.

    The slots come from the caller as one flattened row index per token, ``-1`` skipping
    the token, so the pages a call writes need not be consecutive, need not be new, and
    need not belong to one request: a prefix-caching server reuses a page it already holds
    and a chunked prefill writes the pages it was just given. Distinct non-negative slots
    are the caller's obligation, and a call writes nothing outside the slots it names.
    With an FP8 cache the rows are divided by their scale and cast to E4M3FN. NaN is
    preserved; finite overflow and infinities saturate to the largest finite magnitude
    with the same sign.

    Args:
        heads_kv: Optional eager geometry -- key/value heads one token's row
            holds.  Omitted, the geometry is taken from each call.
        dim: Optional eager geometry -- elements per head in a row.
        num_pages: Optional eager geometry -- pages in the pool.
        page_size: Optional eager geometry -- rows in a page.
        dtype: Optional eager geometry -- dtype of the rows the call passes in.
        cache_dtype: Optional eager geometry -- dtype of the page pool.
        kernel_map: Optional override for kernel dispatch.
    """

    def __init__(
        self,
        *,
        heads_kv: Optional[int] = None,
        dim: Optional[int] = None,
        num_pages: Optional[int] = None,
        page_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        cache_dtype: Optional[torch.dtype] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.heads_kv = heads_kv
        self.dim = dim
        self.num_pages = num_pages
        self.page_size = page_size
        self.dtype = dtype
        self.cache_dtype = cache_dtype
        self._tokens: Optional[int] = None
        self._slot_mapping: Optional[torch.Tensor] = None
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"paged_cache_write_kernel": PagedKVCacheWriteKernel}

    @property
    def quantized(self) -> bool:
        return self.cache_dtype is torch.float8_e4m3fn

    def _validate(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
    ) -> None:
        """Reproduce the manifest contract (dtypes, shapes, scale presence).

        GPU parity: the GPU op inherits these checks from the manifest
        ``shape_rules`` (``present(v_scale) == present(k_scale)``,
        ``PS > 0``, ``(KV == 'float8_e4m3fn') == present(k_scale)``) plus the
        declared input/cache dtype sets; the NPU project has no manifest
        codegen, so they are checked here.
        """
        backend = get_device_backend()
        tensors = (k, v, k_cache, v_cache, slot_mapping)
        if not all(backend.is_device_tensor(t) for t in tensors):
            raise ValueError(
                f"{type(self).__name__} expects {backend.name} tensors; "
                "move every input to the device backend first"
            )

        if k.dtype not in _INPUT_DTYPES:
            raise ValueError(
                f"unsupported input dtype {k.dtype}; "
                f"expected one of {', '.join(str(d) for d in _INPUT_DTYPES)}"
            )
        if v.dtype is not k.dtype:
            raise ValueError(f"v dtype {v.dtype} must match k dtype {k.dtype}")
        if k_cache.dtype not in _CACHE_DTYPES:
            raise ValueError(
                f"unsupported cache dtype {k_cache.dtype}; "
                f"expected one of {', '.join(str(d) for d in _CACHE_DTYPES)}"
            )
        if v_cache.dtype is not k_cache.dtype:
            raise ValueError(
                f"v_cache dtype {v_cache.dtype} must match k_cache dtype {k_cache.dtype}"
            )
        if slot_mapping.dtype is not torch.int64:
            raise ValueError(f"slot_mapping dtype {slot_mapping.dtype} must be int64")

        if k.dim() != 3:
            raise ValueError(f"k must be (N, H_kv, D); got shape {tuple(k.shape)}")
        if v.shape != k.shape:
            raise ValueError(f"v shape {tuple(v.shape)} must match k shape {tuple(k.shape)}")
        if k_cache.dim() != 4:
            raise ValueError(
                f"k_cache must be (NP, PS, H_kv, D); got shape {tuple(k_cache.shape)}"
            )
        if v_cache.shape != k_cache.shape:
            raise ValueError(
                f"v_cache shape {tuple(v_cache.shape)} must match "
                f"k_cache shape {tuple(k_cache.shape)}"
            )
        expected = (k_cache.shape[0], k_cache.shape[1], k.shape[1], k.shape[2])
        if k_cache.shape != expected:
            raise ValueError(
                f"k_cache shape {tuple(k_cache.shape)} must be (NP, PS, H_kv, D) = {expected}"
            )
        if k_cache.shape[1] <= 0:
            raise ValueError(f"page_size must be positive; got {k_cache.shape[1]}")
        if slot_mapping.dim() != 1 or slot_mapping.shape[0] != k.shape[0]:
            raise ValueError(
                f"slot_mapping must be ({k.shape[0]},); got shape {tuple(slot_mapping.shape)}"
            )

        scales = (k_scale, v_scale)
        if (k_scale is None) != (v_scale is None):
            raise ValueError("present(v_scale) must equal present(k_scale)")
        quantized = k_cache.dtype is torch.float8_e4m3fn
        if quantized != (k_scale is not None):
            raise ValueError(
                "k_scale/v_scale are required exactly when the cache dtype is "
                f"float8_e4m3fn (cache dtype {k_cache.dtype})"
            )
        for name, scale in (("k_scale", k_scale), ("v_scale", v_scale)):
            if scale is None:
                continue
            if scale.dtype is not torch.float32:
                raise ValueError(f"{name} dtype {scale.dtype} must be float32")
            if tuple(scale.shape) != (1,):
                raise ValueError(f"{name} must have shape (1,); got {tuple(scale.shape)}")
            if not backend.is_device_tensor(scale):
                raise ValueError(f"{name} must live on the device backend")

    def _get_or_create_kernel(
        self,
        geometry: tuple[int, int, int, int, torch.dtype, torch.dtype],
    ) -> Kernel:
        """Return the kernel for *geometry*, building and caching it on first use."""
        kernel = self._kernel_cache.get(geometry)
        if kernel is None:
            heads_kv, dim, num_pages, page_size, dtype, cache_dtype = geometry
            kernel_cls = self.kernel_map["paged_cache_write_kernel"]
            kernel = kernel_cls(heads_kv, dim, num_pages, page_size, dtype, cache_dtype)
            self._kernel_cache[geometry] = kernel
        self.kernel = kernel
        return kernel

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
    ) -> None:
        """Write each token's key and value row into the page-pool row its slot names.

        Args:
            k: New keys, ``(tokens, heads_kv, dim)`` in float16 or bfloat16.
            v: New values, the same shape and dtype.
            k_cache: The key page pool, ``(num_pages, page_size, heads_kv, dim)``,
                written in place. Its dtype is the cache dtype: float16,
                bfloat16, or float8_e4m3fn.
            v_cache: The value page pool, the same shape and dtype, written in place.
            slot_mapping: ``int64`` ``(tokens,)``; entry ``n`` is
                ``page * page_size + offset``, the flattened row token ``n``
                writes, or ``-1`` to skip it.
            k_scale: ``float32`` ``(1,)`` divisor of the stored keys, required
                exactly when the cache is float8_e4m3fn.
            v_scale: The same divisor for the stored values.

        Returns:
            None; both page pools are mutated and no other tensor is touched.
        """
        self._validate(k, v, k_cache, v_cache, slot_mapping, k_scale, v_scale)
        k = k.contiguous()
        v = v.contiguous()
        slot_mapping = slot_mapping.contiguous()
        k_scale = None if k_scale is None else k_scale.contiguous()
        v_scale = None if v_scale is None else v_scale.contiguous()
        # k_cache and v_cache are declared contiguous and mutated, so they are
        # handed over as they arrived: a copy would take the write with it.
        geometry = (
            k.shape[1],
            k.shape[2],
            k_cache.shape[0],
            k_cache.shape[1],
            k.dtype,
            k_cache.dtype,
        )
        kernel = self._get_or_create_kernel(geometry)
        # Remember the geometry and the slot reference for ``eval_roofline``;
        # computing the written count here would sync the device on every call.
        self.heads_kv, self.dim, self.num_pages, self.page_size = geometry[:4]
        self.dtype, self.cache_dtype = k.dtype, k_cache.dtype
        self._tokens, self._slot_mapping = k.shape[0], slot_mapping
        kernel(k, v, k_cache, v_cache, slot_mapping, k_scale, v_scale)
        return None

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the last call's shape and slots.

        Ported from the GPU ``paged_kv_cache_write_roofline``: only the
        tokens whose slot is not ``-1`` are read and written.  Each such
        token reads its k and v rows and writes them into both caches; an
        FP8 cache scales and saturates each stored value (2 FLOPs per
        value, 4 per token-element across k and v).  ``slot_mapping``
        (int64, ``N`` entries) is read once and both scales (float32, one
        element each) only when at least one row is written.

        Before the first ``forward`` the eager constructor geometry is
        used and every token counts as written (the only upper bound
        available without the slots).  The written count is resolved here,
        outside the timed region, so ``forward`` never syncs on it.
        """
        if self.dtype is None or self.cache_dtype is None:
            raise RuntimeError("eval_roofline needs the cache geometry (call forward first)")
        row = self.heads_kv * self.dim
        tokens = self._tokens if self._tokens is not None else 0
        if self._slot_mapping is None:
            written = tokens
        else:
            written = int((self._slot_mapping >= 0).sum().item())
        in_elem_bytes = self.dtype.itemsize
        cache_elem_bytes = self.cache_dtype.itemsize
        flops = 4 * written * row if self.quantized else 0
        moved = 2 * written * row * (in_elem_bytes + cache_elem_bytes)
        moved += tokens * 8  # slot_mapping is read once (int64)
        if self.quantized and written * row:
            moved += 8  # k_scale + v_scale, one float32 element each
        return int(flops), int(moved)
