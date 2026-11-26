"""MLA packed (varlen) prefill backward kernel (NPU-adapted wrapper + Kernel class).

MLA prefill backward over packed requests, after the latent is decompressed
(THD layout).  Every head has its own ``dim_qk``-wide query and key and its own
``dim_v``-wide value, so the contractions run at their own widths: the scores,
``dQ`` and ``dK`` over ``dim_qk``, ``dP`` and ``dV`` over ``dim_v``.  Three
launches serve one call, exactly as on GPU:

1. ``delta = rowsum(o * do)``, head-major like ``lse``: ``(H, T_q)``.
2. One CTA per (request, key block, head) keeps its keys and values in shared
   memory and ``dK`` / ``dV`` in registers, walks the query blocks its mask
   admits, and adds each step's ``dQ`` partial to a float32 accumulator.
3. The accumulator is scaled and cast to the input dtype.

Each request attends only to its own keys: request ``b`` holds rows
``cu_seqlens_q[b]:cu_seqlens_q[b + 1]`` of ``q``/``do`` and rows
``cu_seqlens_kv[b]:cu_seqlens_kv[b + 1]`` of ``k``/``v``, and query ``i`` of a
request sits at key position ``i + kv_len - q_len`` (bottom-right alignment).
A query that sees no key has ``lse = -inf`` and a zero gradient.  ``dq``/``dk``/
``dv`` are written back at those same packed offsets.

This is ``MLADenseBwdOp``'s packed form: the GPU side shares **one** kernel
factory set (``_bwd_kernel`` / ``_delta_kernel`` / ``_dq_land_kernel`` behind
``_MLABwdKernelBase``) and selects the branch with ``dense_seqlen_q == 0``.
Only two things differ from the dense call:

* the scheduler -- ``GroupTiling`` enumerates key blocks per request and maps a
  flat tile id back to ``(request, first key row)``, instead of the dense
  ``tile // blocks_per_request`` arithmetic;
* the ``lse``/``delta`` layout -- head-major ``(H, T_q)`` instead of
  ``(B, H, S_q)`` -- and the packed ``cu_seqlens`` offsets every read/write of
  ``q``/``k``/``v``/``do``/``dq``/``dk``/``dv`` goes through.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    ``_bwd_kernel`` is extracted from the GPU repo by
    ``extract_tl_kernel.py``; ``_delta_kernel``, ``_dq_land_kernel`` and
    ``GroupTiling`` (with its ``_tile_prefix_sum`` macro factory) are appended
    verbatim by the scaffold under the skill's "helper function extraction"
    rule, because the extractor follows only the ``self.kernel`` member.  All
    are the GPU (CUDA-target) implementations imported as-is and serve as the
    reference the NPU kernel component rewrites for ``target="npuir"``.
    K1-K4 adaptations (decorator, grid/sync semantics, ``threads`` removal,
    boundary-mask strategy) are handled by the NPU component during
    re-implementation.

  **Part B -- Kernel class** (fully ported, no custom_op wrapper):
    E10-like: the GPU kernel class has no ``@torch.library.custom_op``
    wrapper -- its ``_run`` calls the three compiled programs directly --
    so neither does the NPU port.
    K5: ``supported_archs = None`` (was ``[90]``; the backward kernels are
        SM90 only on GPU).
    K8: no autotune on either side; heuristic ``default_config`` only.
    K9: ``threads`` removed from ``default_config`` and from the factory
        call; the GPU default was ``threads=256``.
    The GPU ``_MLABwdKernelBase`` selection protocol (``refusal`` /
        ``entry_for`` / ``MLABwdCall`` / ``CompiledProgram`` /
        ``program(self._compile)``) is GPU target plumbing: its refusal
        checks are inlined in ``__init__``, and the compiled program set is
        cached per config in ``_programs_for`` instead.
    The GPU ``dense_seqlen_q``/``dense_seqlen_kv`` pair is passed as ``0, 0``:
        ``MLAVarlenBwdKernel`` is the packed call, and ``0`` selects the
        packed branch of all three factories (the same argument the GPU
        ``_MLABwdKernelBase.__init__`` receives from the varlen path, where
        ``MLABwdCall`` carries no ``seqlen_*``).

  **Scope note (K5)**: the GPU op declares two kernel variants --
  ``mla_varlen_bwd`` (``MLAVarlenBwdKernel``, the architecture-generic
  key-stationary kernel ported here) and ``mla_varlen_bwd_ws``
  (``MLAVarlenBwdWSKernel``, the SM90 warp-specialized variant).  NPU
  collapses to the generic kernel; the warp-specialized variant is not
  migrated.

  ``msprof_kernel_name`` is left undeclared (default): this op launches three
  kernels, so a single ``msprof op --kernel-name`` filter cannot name the
  whole call.  Stage 4/5 profiling decides whether to declare the dominant
  ``_bwd_main`` launch name.

  Stage 5 state: the imported factories are the integrated NPU
  (``target="ascend"``) implementations in ``mla_varlen_bwd_kernel/``.  The
  dispatch passes the NPU-shaped argument list (no ``threads``), which is
  what those factories accept (K3).
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: the three NPU TileLang kernel factories, integrated per function into
# ``mla_varlen_bwd_kernel/`` (Stage 5).  The GPU extracted file
# ``_mla_varlen_bwd_kernels.py`` is retained only as the migration record; the
# dispatch below imports the integrated NPU sources.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .mla_varlen_bwd_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .mla_varlen_bwd_kernel import _bwd_kernel, _delta_kernel, _dq_land_kernel
# The launch plan and the factory builder of the slabbed main pass (DESIGN.md
# 5.6 g): the wrapper must plan before every launch and pass one ``h_off`` per
# head slab, because the plan depends on the **run-time** token count.
from .mla_varlen_bwd_kernel._bwd_kernel import (
    build_kernel,
    describe,
    factory_config,
    plan_launch,
)
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .mla_varlen_bwd_kernel.perf_opt._bwd_kernel import _bwd_kernel
# from .mla_varlen_bwd_kernel.perf_opt._delta_kernel import _delta_kernel
# from .mla_varlen_bwd_kernel.perf_opt._dq_land_kernel import _dq_land_kernel

__all__ = ["MLAVarlenBwdKernel"]

_STEP = 16


class MLAVarlenBwdKernel(Kernel):
    """The three launches of one packed MLA backward call.

    Args:
        batch: Number of packed requests (``len(cu_seqlens_q) - 1``).
        heads: Number of query heads.
        dim_qk: Query/key head width (``128 nope + 64 rope`` for DeepSeek-V3,
            i.e. not a power of two).
        dim_v: Value head width.
        is_causal: Whether a query row reads only keys at or before its own
            position under bottom-right alignment.
        dtype: Data type (float16 or bfloat16).
        sm_scale: Score scale, or ``None`` for ``dim_qk ** -0.5``.
        config: Optional dict with ``block_n``, ``block_m``, ``num_stages``.
        device_index: Device the kernel is built for (unused by the NPU
            kernels; kept for the shared kernel-constructor contract).

    Raises:
        ValueError: If ``dtype`` is not float16/bfloat16, or a head dimension
            is not a multiple of 16 (the GPU ``refusal`` contract, inlined).
    """

    # K5: [90] (CUDA SM, backward kernels are SM90 only) -> None (all archs).
    supported_archs: Optional[list] = None

    # The NPU source form (Cube ``T.gemm`` + vector epilogue, memory scopes) is
    # settled by the NPU kernel component during re-implementation; defer to
    # the ambient mode until then.
    ascend_mode = None

    def __init__(
        self,
        batch: int,
        heads: int,
        dim_qk: int,
        dim_v: int,
        is_causal: bool,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        config: Optional[dict] = None,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        # The GPU ``_MLABwdKernelBase.refusal`` contract, inlined (the NPU
        # project does not port the kernel-selection protocol).
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"runs float16 and bfloat16, got {dtype}")
        if dim_qk % _STEP != 0 or dim_v % _STEP != 0:
            raise ValueError(
                f"contracts every head dimension in {_STEP}-wide steps, got {dim_qk}, {dim_v}"
            )
        self.batch, self.heads = batch, heads
        self.dim_qk, self.dim_v = dim_qk, dim_v
        self.is_causal = is_causal
        self.dtype = dtype
        self.device_index = device_index
        self.scale = dim_qk**-0.5 if sm_scale is None else sm_scale

        # No single ``self.kernel``: the main pass is specialized on the
        # **compile-time head slab**, which is only known once the run-time
        # ``total_kv`` of a call fixes the launch plan (DESIGN.md 5.6 b/g), so the
        # compiled programs are cached per ``(geometry, slab)`` in
        # ``_main_programs`` and built on the first call of each slab.  ``delta``
        # and the dQ landing kernel do not depend on the token count and are
        # compiled here.
        self.kernel = None
        self._programs: Optional[tuple] = None
        self._programs_config: Optional[tuple] = None
        self._main_programs: dict = {}
        self._main_plans: dict = {}

        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """No tunable knobs: each integrated factory is frozen per geometry.

        The Stage 3 products compile every shape into the program (the source
        specialises through ``@target_cache`` the same way), so the GPU tiling
        knobs (``block_n`` / ``block_m`` / ``num_stages``) no longer exist and
        the empty config is the only valid one.
        """
        return {}

    def _compile(self, cfg: dict) -> tuple:
        """The two token-count independent programs: the delta and the dQ landing.

        The main pass is *not* built here: its head slab depends on the run-time
        ``total_kv`` (``_main_for``), so it lives in its own ``(geometry, slab)``
        cache.
        """
        prep = _delta_kernel(self.batch, self.heads, self.dim_v, 0, self.dtype_str)
        land = _dq_land_kernel(self.heads, self.dim_qk, self.scale, self.dtype_str)
        return prep, land

    def _programs_for(self, cfg: dict) -> tuple:
        """Return the compiled program set for *cfg*, (re)compiling on change.

        The GPU side used ``self.program(self._compile)`` for the same cache.
        """
        key = tuple(sorted(cfg.items()))
        if self._programs is None or self._programs_config != key:
            self._programs = self._compile(cfg)
            self._programs_config = key
        return self._programs

    def _main_for(self, total_kv: int) -> tuple:
        """The main pass and its launch plan for this call's token count.

        The plan is computed from the run-time ``total_kv`` and the slab it returns
        is part of the cache key, so a call with a longer token vector builds (and
        reuses) the program compiled for the *smaller* slab the 65,535-block cap
        demands.  ``build_kernel`` re-plans and post-asserts plan == artifact
        (``assert_launch_consistency``), so a stale cache entry cannot launch a grid
        that silently truncates.
        """
        total_kv = int(total_kv)
        cfg = factory_config()
        plan = plan_launch(self.batch, total_kv, self.heads, block_n=cfg["block_n"])
        key = (self.batch, self.heads, self.dim_qk, self.dim_v, self.is_causal,
               self.dtype_str, cfg["block_n"], plan.slab)
        fn = self._main_programs.get(key)
        if fn is None:
            fn, plan, _ = build_kernel(self.batch, self.heads, self.dim_qk, self.dim_v,
                                       self.is_causal, self.scale, self.dtype_str,
                                       total_kv)
            self._main_programs[key] = fn
            self._main_plans[key] = plan
            print(f"MLAVarlenBwdKernel main pass for total_kv={total_kv}: "
                  f"{describe(plan)}")
        return fn, self._main_plans[key]

    def _run(self, q, k, v, o, do, lse, cu_seqlens_q, cu_seqlens_kv):
        """Run the launches over packed ``(T, H, D)`` tensors.

        One launch per head slab: ``dq_acc`` is a store-atomic accumulator over
        disjoint head ranges, so it is zeroed **once** before the first launch
        (a per-launch clear would drop earlier slabs), and the dQ landing kernel
        runs after the last slab (DESIGN.md 5.6 d item 4).
        """
        prep, land = self._programs_for(self.config)
        delta = prep(o, do)
        main, plan = self._main_for(int(k.shape[0]))
        device = q.device
        dq_acc = torch.zeros(q.shape, dtype=torch.float32, device=device)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        h_off = torch.tensor([launch.head_off for launch in plan.launches],
                             dtype=torch.int32, device=device)
        for index in range(len(plan.launches)):
            main(q, k, v, do, lse, delta, cu_seqlens_q, cu_seqlens_kv, dq_acc, dk, dv,
                 h_off[index:index + 1])
        dq = land(dq_acc)
        return dq, dk, dv

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        do: torch.Tensor,
        lse: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the three launches on THD-layout q/k/v/o/do and ``(H, T_q)`` lse.

        The tensors are already packed, so no reshape happens here; every
        kernel indexes them through ``cu_seqlens_q`` / ``cu_seqlens_kv``.
        Returns ``(dq, dk, dv)`` shaped like ``q``, ``k`` and ``v``.  The
        device check lives in the Op layer (O1).
        """
        return self._run(q, k, v, o, do, lse, cu_seqlens_q, cu_seqlens_kv)
