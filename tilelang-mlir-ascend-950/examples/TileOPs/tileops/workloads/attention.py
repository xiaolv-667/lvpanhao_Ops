"""Workload definitions for the attention op family (NPU).

Ported from the GPU TileOPs workloads:

- ``workloads/attention/mha.py`` -- the ``MhaFwdWorkload`` class only
  (``MhaBwdWorkload`` / ``MhaDecodeWorkload`` / ``MhaDecodePagedWorkload``
  belong to ops that are not migrated yet, and the Bwd workload
  additionally depends on the unmigrated GQA square-LSE helper).
- ``workloads/attention/paged_cache_write.py`` -- the
  ``PagedKVCacheWriteWorkload`` class and its ``paged_cache_write_result``
  helper (the ``PagedKVCacheWriteCall`` manifest-call adapter has no NPU
  counterpart: the NPU manifest is data-only, and the tests parametrize
  the geometry directly).
- ``workloads/attention/mla.py`` -- the ``MLADenseBwdWorkload`` and
  ``MLAVarlenBwdWorkload`` classes (the decode/paged MLA workloads and the
  ``MLADenseBwdCall`` / ``MLAVarlenBwdCall`` manifest-call adapters belong to
  ops that are not migrated yet), plus the shared float32 helpers they need:
  ``_request_slabs``, ``_mla_forward`` and ``mla_bwd_reference``.
  ``gen_inputs`` runs that float32 forward to produce ``o`` and ``lse``, and
  ``ref_program`` recomputes the gradients from ``q``/``k``/``v``/``do`` alone,
  so the oracle does not depend on the NPU kernel or on FA3.  The packed
  workload additionally builds its ``cu_seqlens`` offset vectors in
  ``_make_cu_seqlens`` (the GPU ``workloads/sequence_metadata.make_cu_seqlens``
  helper, kept here so the NPU package stays self-contained).

The only adaptation (W1) is device placement: ``device='cuda'`` is
resolved through :func:`tileops.device.get_device_backend` instead of the
hard-coded ``"cuda"``.  The GPU ``self.rng(...)`` generator helper is not
part of the NPU workload base; input draws use the ambient torch RNG
(seeded by the test/benchmark ``setup`` fixtures).
"""

from itertools import accumulate

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import WorkloadBase

__all__ = [
    "MLADenseBwdWorkload",
    "MLAVarlenBwdWorkload",
    "MhaFwdWorkload",
    "PagedKVCacheWriteWorkload",
    "mla_bwd_reference",
    "paged_cache_write_result",
]


class MhaFwdWorkload(WorkloadBase):
    """Workload for MultiHeadAttentionFwdOp.

    Generates the q/k/v triple in BSHD layout:
    ``(batch, seq_len, heads, dim)`` per tensor.
    """

    def __init__(
        self, batch: int, heads: int, seq_len: int, dim: int, is_causal: bool, dtype: torch.dtype
    ):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # W1: device resolved through the device backend (GPU used "cuda").
        device = get_device_backend().name
        q = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device=device, dtype=self.dtype
        )
        k = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device=device, dtype=self.dtype
        )
        v = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device=device, dtype=self.dtype
        )
        return q, k, v


def paged_cache_write_result(subject, *inputs):
    """Expose the two declared cache writes as the numeric result of an in-place op.

    The op returns ``None`` (it mutates both page pools), so tests and the
    benchmark reference read the written pools back through this helper.
    """
    subject(*inputs)
    return inputs[2], inputs[3]


class PagedKVCacheWriteWorkload(WorkloadBase):
    """New rows scattered into a paged cache by slot, the pool's other rows left alone.

    ``slots[n]`` is the pool row token ``n`` writes, or ``-1`` for a token that writes
    nothing. Omitted, the slots are distinct random rows, which is what a serving step
    hands over; a caller passing its own list exercises the skip path, a slot at either
    end of a page, and slots out of order.

    The stored bytes are the whole result: a named row is written exactly and a row no
    slot names must still hold what it held, so the tests compare with no tolerance
    (the GPU ``verification()`` hook returned ``Exact(rtol=0, atol=0)``; the NPU
    ``TestBase`` has no verification hook, so the test module asserts it directly).
    """

    def __init__(
        self,
        tokens: int,
        heads_kv: int,
        dim: int,
        page_size: int,
        num_pages: int,
        dtype: torch.dtype,
        cache_dtype: torch.dtype | None = None,
        slots: list[int] | None = None,
    ) -> None:
        self.tokens, self.heads_kv, self.dim = tokens, heads_kv, dim
        self.page_size, self.num_pages = page_size, num_pages
        self.dtype = dtype
        self.cache_dtype = dtype if cache_dtype is None else cache_dtype
        self.slots = slots

    @property
    def row_width(self) -> int:
        return self.heads_kv * self.dim

    @property
    def physical_tokens(self) -> int:
        return self.num_pages * self.page_size

    @property
    def quantized(self) -> bool:
        return self.cache_dtype == torch.float8_e4m3fn

    def gen_inputs(self):
        if self.quantized and self.tokens > 0 and self.physical_tokens <= 0:
            raise ValueError("a quantized cache needs at least one row")
        # W1: device resolved through the device backend (GPU used "cuda").
        device = get_device_backend().name
        row = (self.heads_kv, self.dim)
        k = torch.randn(
            (self.tokens, *row),
            dtype=torch.float32,
            device=device,
        ).to(self.dtype)
        v = torch.randn(
            (self.tokens, *row),
            dtype=torch.float32,
            device=device,
        ).to(self.dtype)
        pool = torch.randn(
            (self.num_pages, self.page_size, *row),
            dtype=torch.float32,
            device=device,
        )
        # The two pools differ, so a kernel writing one row to both cannot pass.
        k_cache = pool.to(self.cache_dtype)
        v_cache = (-pool).to(self.cache_dtype)
        if self.slots is None:
            if self.tokens > self.physical_tokens:
                raise ValueError(
                    f"{self.tokens} tokens need {self.tokens} distinct slots but the pool "
                    f"holds {self.physical_tokens} rows"
                )
            drawn = torch.randperm(self.physical_tokens)[: self.tokens]
            slots = drawn.to(device=device, dtype=torch.int64)
        else:
            if len(self.slots) != self.tokens:
                raise ValueError(f"slots holds {len(self.slots)} rows for {self.tokens} tokens")
            slots = torch.tensor(self.slots, dtype=torch.int64, device=device)
        if self.quantized:
            # Distinct, non-unit divisors: a kernel that drops one, or swaps the two,
            # stores something the reference does not.
            k_scale = torch.tensor([1.25], dtype=torch.float32, device=device)
            v_scale = torch.tensor([0.75], dtype=torch.float32, device=device)
        else:
            k_scale = v_scale = None
        return k, v, k_cache, v_cache, slots, k_scale, v_scale

    def ref_program(self, k, v, k_cache, v_cache, slot_mapping, k_scale=None, v_scale=None):
        """Scatter by index, with saturating FP8 conversion that preserves NaN.

        The pages are cloned before they are written: the reference runs on the shared
        inputs, and a call that wrote them would leave nothing to compare against. A
        token whose slot is ``-1`` writes nothing, and a row no token names keeps the
        value it was handed, which the caller's comparison reads back.
        """
        k_cache, v_cache = k_cache.clone(), v_cache.clone()
        written = (slot_mapping >= 0).nonzero().squeeze(1)
        rows = slot_mapping[written]
        for pages, new, scale in (
            (k_cache, k, k_scale),
            (v_cache, v, v_scale),
        ):
            flat = pages.view(-1, *pages.shape[2:])
            stored = new[written]
            if scale is not None:
                # Express saturation explicitly: some torch FP8 casts turn overflow
                # into NaN. torch.clamp preserves NaN while bounding other values.
                limit = torch.finfo(pages.dtype).max
                stored = (stored.float() / scale[0]).clamp(-limit, limit).to(pages.dtype)
            if pages.dtype is torch.float8_e4m3fn:
                # aclnnIndexPut does not implement float8_e4m3fn on this stack
                # (error 161002: the dtype support list excludes it), so the same
                # scatter runs on the host and the finished pool is copied back.
                host = pages.cpu()
                host.view(-1, *pages.shape[2:])[rows.cpu()] = stored.cpu()
                pages.copy_(host)
            else:
                flat[rows] = stored
        return k_cache, v_cache


def _request_slabs(q, k, *, is_causal, scale):
    """Yield ``(heads, rows, probs, lse)`` over one request's float32 attention, slab by slab.

    *q* is ``(q_len, heads, dim)`` and *k* ``(kv_len, heads, dim)``. ``probs`` is the
    ``(heads, rows, kv_len)`` softmax and ``lse`` its ``(heads, rows, 1)`` natural-log
    log-sum-exp. Query ``i`` sits at key position ``i + kv_len - q_len``, FA3's bottom-right
    alignment. A query that sees no key has zero probabilities and a ``-inf`` ``lse``.
    """
    q_len, num_heads, _ = q.shape
    kv_len = k.shape[0]
    key = k.float()
    position = torch.arange(q_len, device=q.device) + (kv_len - q_len)
    key_index = torch.arange(kv_len, device=q.device)
    # Bounds the (heads, rows, kv_len) float32 scores; each row still sees every key.
    heads_per_slab, rows_per_slab = 8, 1024
    for h0 in range(0, num_heads, heads_per_slab):
        for r0 in range(0, q_len, rows_per_slab):
            heads, rows = slice(h0, h0 + heads_per_slab), slice(r0, r0 + rows_per_slab)
            scores = torch.einsum("shd,nhd->hsn", q[rows, heads].float(), key[:, heads]) * scale
            if is_causal:
                hidden = key_index[None, :] > position[rows, None]
                scores = scores.masked_fill(hidden[None], float("-inf"))
            lse = scores.logsumexp(-1, keepdim=True)
            # A row of all -inf scores gives exp(-inf - -inf) = NaN; its probabilities are 0.
            yield heads, rows, torch.exp(scores - lse).nan_to_num(0.0), lse


def _mla_forward(q, k, v, bounds_q, bounds_kv, *, is_causal, scale):
    """Float32 ``o`` ``(total_q, heads, dim_v)`` and head-major ``lse`` ``(heads, total_q)``.

    Request ``b`` holds rows ``bounds_q[b]:bounds_q[b + 1]`` of *q* and rows
    ``bounds_kv[b]:bounds_kv[b + 1]`` of *k* and *v*.
    """
    o = torch.zeros(*q.shape[:2], v.shape[-1], dtype=torch.float32, device=q.device)
    lse = torch.empty(q.shape[1], q.shape[0], dtype=torch.float32, device=q.device)
    for q0, q1, k0, k1 in zip(bounds_q, bounds_q[1:], bounds_kv, bounds_kv[1:], strict=False):
        value = v[k0:k1].float()
        slabs = _request_slabs(q[q0:q1], k[k0:k1], is_causal=is_causal, scale=scale)
        for heads, rows, probs, row_lse in slabs:
            o[q0:q1][rows, heads] = torch.einsum("hsn,nhd->shd", probs, value[:, heads])
            lse[heads, q0:q1][:, rows] = row_lse[..., 0]
    return o, lse


def mla_bwd_reference(q, k, v, do, bounds_q, bounds_kv, *, is_causal, sm_scale=None):
    """Float32 ``(dq, dk, dv)`` of decompressed MLA attention from *q*, *k*, *v* and *do* alone.

    The probabilities are recomputed rather than read from the op's inputs ``o`` and ``lse``.
    Request ``b`` holds rows ``bounds_q[b]:bounds_q[b + 1]`` of *q* and *do* and rows
    ``bounds_kv[b]:bounds_kv[b + 1]`` of *k* and *v*; the tensors are ``(rows, heads, dim)``.
    """
    scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
    dq, dk, dv = (torch.zeros(t.shape, dtype=torch.float32, device=t.device) for t in (q, k, v))
    for q0, q1, k0, k1 in zip(bounds_q, bounds_q[1:], bounds_kv, bounds_kv[1:], strict=False):
        key, value = k[k0:k1].float(), v[k0:k1].float()
        slabs = _request_slabs(q[q0:q1], k[k0:k1], is_causal=is_causal, scale=scale)
        for heads, rows, probs, _ in slabs:
            grad_out = do[q0:q1][rows, heads].float()
            dprobs = torch.einsum("shd,nhd->hsn", grad_out, value[:, heads])
            dscores = probs * (dprobs - (probs * dprobs).sum(-1, keepdim=True))
            dv[k0:k1, heads] += torch.einsum("hsn,shd->nhd", probs, grad_out)
            dk[k0:k1, heads] += torch.einsum(
                "hsn,shd->nhd", dscores, q[q0:q1][rows, heads].float()
            )
            dq[q0:q1][rows, heads] = torch.einsum("hsn,nhd->shd", dscores, key[:, heads])
    return dq * scale, dk * scale, dv


class MLADenseBwdWorkload(WorkloadBase):
    """Fixed-length MLA prefill backward; ``o`` and ``lse`` are the float32 forward's.

    Ported from the GPU ``MLADenseBwdWorkload`` (the call adapter
    ``MLADenseBwdCall`` is not ported: the NPU manifest is data-only).  The
    inputs are ``(do, q, k, v, o, lse)``, ``lse`` laid out ``(B, H, S_q)``
    in float32, as FA3 returns it; ``o`` is the forward output rounded once
    to the input dtype.  ``ref_program`` returns float32 gradients rounded
    once to the input dtype.
    """

    def __init__(
        self,
        batch,
        seqlen_q,
        seqlen_kv,
        heads,
        dim_qk,
        dim_v,
        dtype,
        is_causal=True,
        sm_scale=None,
    ):
        self.batch, self.seqlen_q, self.seqlen_kv, self.heads = batch, seqlen_q, seqlen_kv, heads
        self.dim_qk, self.dim_v = dim_qk, dim_v
        self.dtype, self.is_causal, self.sm_scale = dtype, is_causal, sm_scale

    def gen_inputs(self):
        """``(do, q, k, v, o, lse)``: random q, k, v and do, then the forward of q, k, v."""
        # W1: device resolved through the device backend (GPU used ``run_device()``).
        device = get_device_backend().name

        def randn(*shape):
            return torch.randn(*shape, dtype=self.dtype, device=device)

        q = randn(self.batch, self.seqlen_q, self.heads, self.dim_qk)
        k = randn(self.batch, self.seqlen_kv, self.heads, self.dim_qk)
        v = randn(self.batch, self.seqlen_kv, self.heads, self.dim_v)
        do = randn(self.batch, self.seqlen_q, self.heads, self.dim_v)
        o, lse = _mla_forward(
            q.flatten(0, 1),
            k.flatten(0, 1),
            v.flatten(0, 1),
            range(0, self.batch * self.seqlen_q + 1, self.seqlen_q),
            range(0, self.batch * self.seqlen_kv + 1, self.seqlen_kv),
            is_causal=self.is_causal,
            scale=self.sm_scale if self.sm_scale is not None else self.dim_qk**-0.5,
        )
        o = o.view_as(do).to(self.dtype)
        lse = lse.view(self.heads, self.batch, self.seqlen_q).transpose(0, 1).contiguous()
        return do, q, k, v, o, lse

    def ref_program(self, do, q, k, v, o, lse):
        """Float32 gradients, rounded once to the input dtype."""
        batch, seqlen_q = q.shape[:2]
        seqlen_kv = k.shape[1]
        grads = mla_bwd_reference(
            q.flatten(0, 1),
            k.flatten(0, 1),
            v.flatten(0, 1),
            do.flatten(0, 1),
            range(0, batch * seqlen_q + 1, seqlen_q),
            range(0, batch * seqlen_kv + 1, seqlen_kv),
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
        )
        return tuple(g.reshape(t.shape).to(q.dtype) for g, t in zip(grads, (q, k, v), strict=True))


def _make_cu_seqlens(lengths) -> torch.Tensor:
    """Exclusive prefix sum of *lengths*, the packed-varlen offset vector (int32).

    The GPU ``workloads/sequence_metadata.make_cu_seqlens`` helper with two NPU
    adaptations: the device comes from the device backend (W1) and the prefix sum
    is built with :func:`itertools.accumulate` so no extra import is needed.
    """
    # W1: device resolved through the device backend (GPU used ``run_device()``).
    device = get_device_backend().name
    return torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device)


class MLAVarlenBwdWorkload(WorkloadBase):
    """Packed MLA prefill backward; ``o`` and ``lse`` are the float32 forward's.

    Ported from the GPU ``MLAVarlenBwdWorkload`` (the call adapter
    ``MLAVarlenBwdCall`` is not ported: the NPU manifest is data-only).  Request
    ``b`` owns ``q_lens[b]`` query rows and ``kv_lens[b]`` key rows of the packed
    tensors; query ``i`` of a request sits at key position
    ``i + kv_len - q_len``, and a query that sees no key has zero probabilities
    and a ``-inf`` ``lse`` (hence a zero gradient).

    The inputs are ``(do, q, k, v, o, lse, cu_seqlens_q, cu_seqlens_kv)``:
    ``lse`` is head-major ``(H, T_q)`` in float32, as FA3 returns it for packed
    requests (and as ``MLAVarlenFwdOp`` returns it on GPU) -- *not* the dense
    ``(B, H, S_q)`` layout; ``cu_seqlens_q``/``cu_seqlens_kv`` are the int32
    ``(B + 1)`` offset vectors.  ``o`` is the forward output rounded once to the
    input dtype, and ``ref_program`` returns float32 gradients rounded once to
    the input dtype.
    """

    def __init__(
        self,
        q_lens,
        kv_lens,
        heads,
        dim_qk,
        dim_v,
        dtype,
        is_causal=True,
        sm_scale=None,
    ):
        self.q_lens, self.kv_lens, self.heads = list(q_lens), list(kv_lens), heads
        self.dim_qk, self.dim_v = dim_qk, dim_v
        self.dtype, self.is_causal, self.sm_scale = dtype, is_causal, sm_scale

    @property
    def total_q(self) -> int:
        """Packed query length ``T_q``: the rows of ``q``, ``do``, ``o`` and ``dq``."""
        return sum(self.q_lens)

    @property
    def total_kv(self) -> int:
        """Packed key/value length ``T_kv``: the rows of ``k``, ``v``, ``dk`` and ``dv``."""
        return sum(self.kv_lens)

    def gen_inputs(self):
        """``(do, q, k, v, o, lse, cu_seqlens_q, cu_seqlens_kv)``, ``lse`` head-major."""
        # W1: device resolved through the device backend (GPU used ``run_device()``).
        device = get_device_backend().name
        total_q, total_kv = self.total_q, self.total_kv

        def randn(*shape):
            return torch.randn(*shape, dtype=self.dtype, device=device)

        q = randn(total_q, self.heads, self.dim_qk)
        k = randn(total_kv, self.heads, self.dim_qk)
        v = randn(total_kv, self.heads, self.dim_v)
        do = randn(total_q, self.heads, self.dim_v)
        cu_seqlens_q, cu_seqlens_kv = (
            _make_cu_seqlens(self.q_lens),
            _make_cu_seqlens(self.kv_lens),
        )
        # ``_mla_forward`` already returns the head-major ``(H, T_q)`` log-sum-exp:
        # the packed backward takes it with no transpose (GPU parity).
        o, lse = _mla_forward(
            q,
            k,
            v,
            cu_seqlens_q.tolist(),
            cu_seqlens_kv.tolist(),
            is_causal=self.is_causal,
            scale=self.sm_scale if self.sm_scale is not None else self.dim_qk**-0.5,
        )
        return do, q, k, v, o.to(self.dtype), lse, cu_seqlens_q, cu_seqlens_kv

    def ref_program(self, do, q, k, v, o, lse, cu_seqlens_q, cu_seqlens_kv):
        """Float32 gradients, rounded once to the input dtype.

        The probabilities are recomputed per request from ``q``/``k``/``v``
        alone (the op's ``o`` and ``lse`` are not read), and each request's
        gradients land at its own ``cu_seqlens`` offsets.
        """
        grads = mla_bwd_reference(
            q,
            k,
            v,
            do,
            cu_seqlens_q.tolist(),
            cu_seqlens_kv.tolist(),
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
        )
        return tuple(g.to(q.dtype) for g in grads)
