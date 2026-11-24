"""Ascend pre-layer Manifold-Constrained Hyper-Connections (mHC) step.

One AI core owns one batch row: it projects the width-expanded stream through phi,
normalises the resulting Sinkhorn matrix, and mixes the stream down to the layer
input.  Correctness-first implementation; the projection is a scalar loop.
"""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sequence_modeling.mhc.call_spec import MHCPreCall, MHCPreFwdInterface

__all__ = ["AscendMHCPreKernel"]

SUM_TILE = 128


def _make_mhc_pre_kernel(
    batch: int,
    n_expand: int,
    c_x: int,
    x_dtype: str,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    sinkhorn_repeat: int,
    sinkhorn_eps: float,
):
    x_dim = n_expand * c_x
    phi_dim = n_expand * n_expand + 2 * n_expand
    sum_tiles = (x_dim + SUM_TILE - 1) // SUM_TILE

    @T.prim_func
    def main(
        phi: T.Tensor((x_dim, phi_dim), "float32"),
        x: T.Tensor((batch, x_dim), x_dtype),
        b: T.Tensor((phi_dim,), "float32"),
        x_res: T.Tensor((batch, x_dim), x_dtype),
        x_layer: T.Tensor((batch, c_x), x_dtype),
        h_post: T.Tensor((batch, n_expand), "float32"),
    ):
        with T.Kernel(batch) as bx:
            h = T.alloc_shared((phi_dim,), "float32")
            hres = T.alloc_shared((n_expand, n_expand), "float32")
            hpre = T.alloc_shared((n_expand,), "float32")
            s2 = T.alloc_shared((1,), "float32")

            # Projection: h[j] = sum_c x[b, c] * phi[c, j] and sum_c x^2.
            with T.SimtVF(threads=64):
                hj = T.alloc_fragment((phi_dim,), "float32")
                sq = T.alloc_fragment((SUM_TILE,), "float32")
                s2f = T.alloc_fragment((1,), "float32")
                sred = T.alloc_reducer((1,), "float32", op="sum")
                T.clear(hj)
                for c in T.serial(x_dim):
                    xv = T.cast(x[bx, c], "float32")
                    for j in T.Parallel(phi_dim):
                        hj[j] = hj[j] + xv * phi[c, j]
                for j in T.Parallel(phi_dim):
                    h[j] = hj[j]
                T.reducer_init(sred)
                for t in T.serial(sum_tiles):
                    for i in T.Parallel(SUM_TILE):
                        col = T.min(t * SUM_TILE + i, x_dim - 1)
                        v = T.if_then_else(t * SUM_TILE + i < x_dim, T.cast(x[bx, col], "float32"), 0.0)
                        sq[i] = v * v
                        T.reducer_update(sred[0], sq[i])
                T.finalize_reducer(sred, s2f)
                for i in T.Parallel(1):
                    s2[i] = s2f[i]

            # Weights, row softmax and Sinkhorn normalisation of the residual matrix.
            with T.SimtVF(threads=64):
                hr = T.alloc_fragment((n_expand, n_expand), "float32")
                rowsum = T.alloc_fragment((n_expand,), "float32")
                colsum = T.alloc_fragment((n_expand,), "float32")
                rowmax = T.alloc_fragment((n_expand,), "float32")
                inv_r = T.alloc_fragment((1,), "float32")
                inv_r[0] = 1.0 / (T.sqrt(s2[0]) / x_dim**0.5 + 0.0001)
                for j in T.Parallel(n_expand):
                    hpre[j] = 1.0 / (
                        1.0 + T.exp(-(alpha_pre * inv_r[0] * h[j] + b[j]))
                    )
                    h_post[bx, j] = 2.0 / (
                        1.0 + T.exp(-(alpha_post * inv_r[0] * h[n_expand + j] + b[n_expand + j]))
                    )
                for i, k in T.Parallel(n_expand, n_expand):
                    hr[i, k] = (
                        alpha_res * inv_r[0] * h[2 * n_expand + i * n_expand + k]
                        + b[2 * n_expand + i * n_expand + k]
                    )
                T.reduce_max(hr, rowmax, dim=1)
                for i, k in T.Parallel(n_expand, n_expand):
                    hres[i, k] = T.exp(hr[i, k] - rowmax[i])
                for _ in T.serial(sinkhorn_repeat):
                    T.reduce_sum(hres, rowsum, dim=1)
                    for i, k in T.Parallel(n_expand, n_expand):
                        hres[i, k] = hres[i, k] / (rowsum[i] + sinkhorn_eps)
                    T.reduce_sum(hres, colsum, dim=0)
                    for i, k in T.Parallel(n_expand, n_expand):
                        hres[i, k] = hres[i, k] / (colsum[k] + sinkhorn_eps)

            # Mix down to the layer input and the residual stream.
            with T.SimtVF(threads=64):
                for c in T.Parallel(c_x):
                    layer = T.alloc_var("float32")
                    layer = 0.0
                    for j in T.serial(n_expand):
                        layer = layer + hpre[j] * T.cast(x[bx, j * c_x + c], "float32")
                    x_layer[bx, c] = T.cast(layer, x_dtype)
                    for i in T.serial(n_expand):
                        res = T.alloc_var("float32")
                        res = 0.0
                        for k in T.serial(n_expand):
                            res = res + hres[i, k] * T.cast(x[bx, k * c_x + c], "float32")
                        x_res[bx, i * c_x + c] = T.cast(res, x_dtype)

    return tilelang.compile(main, target="ascend")


class AscendMHCPreKernel(Kernel, MHCPreFwdInterface):
    """Ascend mHC pre-layer step, one batch row per AI core."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: MHCPreCall) -> Optional[str]:
        if call.dtype != torch.bfloat16:
            return "requires bfloat16 inputs"
        if call.n_expand <= 0 or call.c_x <= 0 or call.batch <= 0:
            return "requires positive shapes"
        return None

    @classmethod
    def entry_for(cls, call: MHCPreCall):
        args = (
            call.batch,
            call.n_expand,
            call.c_x,
            call.alpha_pre,
            call.alpha_post,
            call.alpha_res,
            call.sinkhorn_repeat,
            call.sinkhorn_eps,
        )
        return (args, call.dtype), lambda: cls(*args, dtype=call.dtype)

    def __init__(
        self,
        batch: int,
        n_expand: int,
        c_x: int,
        alpha_pre: float,
        alpha_post: float,
        alpha_res: float,
        sinkhorn_repeat: int,
        sinkhorn_eps: float,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype
        self.alpha_pre = alpha_pre
        self.alpha_post = alpha_post
        self.alpha_res = alpha_res
        self.sinkhorn_repeat = sinkhorn_repeat
        self.sinkhorn_eps = sinkhorn_eps
        self._kernel = None

    def forward(self, phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat, sinkhorn_eps):
        if self._kernel is None:
            self._kernel = _make_mhc_pre_kernel(
                self.batch,
                self.n_expand,
                self.c_x,
                "bfloat16" if self.dtype == torch.bfloat16 else "float16",
                self.alpha_pre,
                self.alpha_post,
                self.alpha_res,
                self.sinkhorn_repeat,
                self.sinkhorn_eps,
            )
        x_res = torch.empty_like(x)
        x_layer = torch.empty((self.batch, self.c_x), device=x.device, dtype=x.dtype)
        h_post = torch.empty((self.batch, self.n_expand), device=x.device, dtype=torch.float32)
        self._kernel(phi, x, b, x_res, x_layer, h_post)
        return x_res, x_layer, h_post


register_kernel_type("MHCPreFwdOp", "ascend", AscendMHCPreKernel)
