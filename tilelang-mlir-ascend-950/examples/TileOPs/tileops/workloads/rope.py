"""Workload definitions for the rope (rotary position embedding) op family.

Ported from TileOPs ``workloads/rope.py`` -- the ``RoPEFwdOp`` subset only (the
shared GPU file also defines the Llama-3.1 / YaRN / LongRoPE cache formulas,
the ``RoPECosSinFwdOp`` call classes and their references, which are not
migrated; the name-dispatched branches of :func:`rope_cache` therefore collapse
to RoPEFwdOp's standard frequencies).

The reference side is kept verbatim from the GPU file:

- :func:`rope_cache` -- the float32 cos/sin tables of ``base ** (-2k /
  rotary_dim)`` over ``max_position_embeddings`` rows, and the magnitude (1.0
  for RoPEFwdOp);
- :func:`rope_tables` -- the tables as the op holds them: scaled by the
  magnitude, rounded to the ``dtype`` parameter, then to the query's dtype and
  device;
- :func:`rope_reference` -- ``(query_out, key_out)`` rotating each head's
  leading ``rotary_dim`` columns at row ``positions + offsets``; the rotation is
  float32 and rounds once.

Adaptation point W1: ``device`` is resolved from :func:`get_device_backend`
instead of the hard-coded ``"cuda"`` in the original TileOPs code.
"""

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import FixtureBase, WorkloadBase


def rope_cache(params: dict) -> tuple[torch.Tensor, torch.Tensor, float]:
    """The float32 cos and sin caches of a RoPE op with *params*, and their magnitude.

    The caches are ``(max_position_embeddings, rotary_dim // 2)``, by the
    formula of the vLLM class the op follows (RoPEFwdOp: standard frequencies
    ``base ** (-2k / rotary_dim)``, magnitude 1).
    """
    rotary, base = params["rotary_dim"], params["base"]
    pos_freqs = base ** (torch.arange(0, rotary, 2, dtype=torch.float32) / rotary)
    inv = 1.0 / pos_freqs
    rows, mscale = params["max_position_embeddings"], 1.0
    angles = torch.outer(torch.arange(rows, dtype=torch.float32), inv)
    return angles.cos(), angles.sin(), mscale


def rope_tables(params: dict, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The cos and sin caches of a RoPE op with *params*, as the op holds them.

    Each is scaled by the magnitude, rounded to the ``dtype`` parameter, then to
    the query's dtype, on the query's device.
    """
    angles_cos, angles_sin, mscale = rope_cache(params)
    return tuple(
        (t * mscale).to(params["dtype"]).to(device=query.device, dtype=query.dtype)
        for t in (angles_cos, angles_sin)
    )


def _rotate_pairs(part: torch.Tensor, c: torch.Tensor, s: torch.Tensor, neox: bool) -> torch.Tensor:
    """*part* rotated by cosines *c* and sines *s*: its halves against each other with *neox*,
    else each adjacent pair."""
    if neox:
        x1, x2 = part.chunk(2, dim=-1)
        return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
    x1, x2 = part[..., 0::2], part[..., 1::2]
    return torch.stack((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).flatten(-2)


def _rotate(
    x: torch.Tensor,
    index: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    neox: bool,
) -> torch.Tensor:
    """*x* ``[*S, head_size * H]`` with each head's first *rotary_dim* columns rotated.

    Each token reads cache row *index*; the rotation is float32, rounded once.
    """
    tokens = index.numel()
    heads = x.reshape(tokens, -1, head_size)
    c, s = cos[index.reshape(-1)].unsqueeze(1).float(), sin[index.reshape(-1)].unsqueeze(1).float()
    rotated = _rotate_pairs(heads[..., :rotary_dim].float(), c, s, neox)
    out = torch.cat((rotated.to(x.dtype), heads[..., rotary_dim:]), dim=-1)
    return out.reshape(x.shape)


def rope_reference(
    params: dict,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(query_out, key_out)`` of a RoPE op with *params*, reading *cos* and *sin* (from
    :func:`rope_tables`) at row ``positions + offsets`` (``positions`` without offsets)."""
    index = positions if offsets is None else positions + offsets
    hs, rd, neox = params["head_size"], params["rotary_dim"], params["is_neox_style"]
    query_out = _rotate(query, index, cos, sin, hs, rd, neox)
    return query_out, None if key is None else _rotate(key, index, cos, sin, hs, rd, neox)


class RoPEFwdWorkload(WorkloadBase):
    """A manifest call of ``RoPEFwdOp`` checked against :func:`rope_reference`.

    ``query_shape`` / ``key_shape`` hold each token's heads flattened,
    ``head_size`` columns a head; ``key_shape`` is None for a query-only call.
    ``position_dtype`` is the dtype of ``positions`` (int32 or int64).
    """

    def __init__(
        self,
        positions_shape: tuple[int, ...],
        query_shape: tuple[int, ...],
        key_shape: tuple[int, ...] | None,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        position_dtype: torch.dtype = torch.int64,
    ):
        self.positions_shape = tuple(positions_shape)
        self.query_shape = tuple(query_shape)
        self.key_shape = None if key_shape is None else tuple(key_shape)
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.position_dtype = position_dtype

    def params(self) -> dict:
        """The op's construction parameters, as the GPU manifest call holds them."""
        return {
            "head_size": self.head_size,
            "rotary_dim": self.rotary_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "base": self.base,
            "is_neox_style": self.is_neox_style,
            "dtype": self.dtype,
        }

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # W1: device via the device backend (was ``device="cuda"``).
        device = get_device_backend().name
        positions = torch.randint(
            0,
            self.max_position_embeddings,
            self.positions_shape,
            dtype=self.position_dtype,
            device=device,
        )
        query = torch.randn(*self.query_shape, dtype=self.dtype, device=device)
        key = (
            None
            if self.key_shape is None
            else torch.randn(*self.key_shape, dtype=self.dtype, device=device)
        )
        return positions, query, key

    def ref_program(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cos, sin = rope_tables(self.params(), query)
        return rope_reference(self.params(), cos, sin, positions, query, key, offsets)


class RoPEFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest

        return [
            (
                "positions_shape, query_shape, key_shape, head_size, rotary_dim, "
                "is_neox_style, dtype",
                [
                    # First three: smoke, fp32 / fp16 / bf16 (one also packaging).
                    pytest.param(
                        (64,),
                        (64, 64 * 4),
                        (64, 64 * 2),
                        64,
                        64,
                        True,
                        torch.float32,
                        marks=[pytest.mark.smoke, pytest.mark.packaging],
                    ),
                    pytest.param(
                        (64,),
                        (64, 64 * 4),
                        (64, 64 * 2),
                        64,
                        64,
                        True,
                        torch.float16,
                        marks=pytest.mark.smoke,
                    ),
                    pytest.param(
                        (64,),
                        (64, 64 * 4),
                        (64, 64 * 2),
                        64,
                        64,
                        True,
                        torch.bfloat16,
                        marks=pytest.mark.smoke,
                    ),
                    # Full: Llama-3-8B prefill, NeoX pairs, heads copying no column.
                    pytest.param(
                        (2048,),
                        (2048, 128 * 8),
                        (2048, 128 * 2),
                        128,
                        128,
                        True,
                        torch.bfloat16,
                        marks=pytest.mark.full,
                    ),
                    # Full: GPT-J adjacent pairs, the first 64 of 256 columns rotate.
                    pytest.param(
                        (512,),
                        (512, 256 * 4),
                        (512, 256 * 4),
                        256,
                        64,
                        False,
                        torch.float16,
                        marks=pytest.mark.full,
                    ),
                    # Full: partial rotation in NeoX style, query and key.
                    pytest.param(
                        (256,),
                        (256, 64 * 8),
                        (256, 64 * 2),
                        64,
                        24,
                        False,
                        torch.float32,
                        marks=pytest.mark.full,
                    ),
                    # Full: query only, no key (Phi-2 cross-layer KV sharing).
                    pytest.param(
                        (1000,),
                        (1000, 80 * 32),
                        None,
                        80,
                        32,
                        True,
                        torch.bfloat16,
                        marks=pytest.mark.full,
                    ),
                ],
            ),
        ]
