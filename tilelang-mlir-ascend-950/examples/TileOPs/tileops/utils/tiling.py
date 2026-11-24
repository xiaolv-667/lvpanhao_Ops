"""Pure tiling arithmetic with explicit platform constraints."""

from tileops.utils.task_platform import shared_memory_budget


def device_smem_budget(device_index=None):
    """Read the current task model budget through the device backend."""
    return shared_memory_budget(device_index)


def align_up(n: int, alignment: int) -> int:
    """Round *n* up to the nearest multiple of *alignment*.

    Args:
        n: Value to align.
        alignment: Alignment boundary (must be positive).

    Returns:
        Smallest multiple of *alignment* that is >= *n*.

    Raises:
        ValueError: If *alignment* is not positive.
    """
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((n + alignment - 1) // alignment) * alignment


def compute_tile_n(
    block_m: int,
    elem_bytes: int,
    N_padded: int,
    alignment: int,
    budget: int,
    num_buffers: int = 1,
) -> int:
    """Compute the tile_n (column chunk) for shared memory, preferring divisibility.

    The budget-derived cap (``tile_n_max``) is the largest multiple of
    *alignment* such that
    ``num_buffers * block_m * tile_n_max * elem_bytes <= budget``.
    The return value may be a smaller divisor of *N_padded* when that
    divisor does not increase the number of N-tiles, because an exact
    division eliminates the nearly-empty remainder tile and reduces
    wasted memory traffic.  For example, N_padded=32768 with
    tile_n_max=32512 gives 2 tiles where tile 2 has only 256 valid
    columns (99.2% waste), while divisor=16384 also gives 2 tiles with
    zero waste.

    When every divisor requires strictly more tiles, the full
    budget-derived cap is returned and the tiled kernel handles the
    single remainder tile via masked loads.

    If N_padded already fits (with *num_buffers* copies), returns N_padded
    (no tiling needed).

    Args:
        block_m: Number of rows per thread block.
        elem_bytes: Bytes per element (e.g. 2 for fp16/bf16, 4 for fp32).
        N_padded: Padded hidden dimension (already aligned to *alignment*).
        alignment: Column alignment boundary (explicit platform value).
        budget: Shared memory budget in bytes (explicit platform value).
        num_buffers: Number of shared memory buffers of shape
            ``(block_m, tile_n)`` that must fit simultaneously (default 1).
            Softmax/log_softmax tiled kernels use 2 (one per pass) due to
            TileLang allocator aliasing constraints.

    Returns:
        tile_n: column tile size, a multiple of *alignment*, or N_padded
        if it fits entirely.

    Raises:
        ValueError: If even a single alignment-width slice cannot fit.
    """
    per_buffer = block_m * elem_bytes
    if num_buffers * per_buffer * N_padded <= budget:
        return N_padded

    # Largest multiple of alignment that fits in shared memory
    max_cols = budget // (num_buffers * per_buffer)
    tile_n_max = (max_cols // alignment) * alignment
    if tile_n_max == 0:
        raise ValueError(
            f"Cannot fit even {alignment} columns in {budget} bytes "
            f"with block_m={block_m}, elem_bytes={elem_bytes}, "
            f"num_buffers={num_buffers}."
        )

    # Prefer the largest tile_n that evenly divides N_padded, so that
    # num_tiles * tile_n == N_padded and no remainder tile is needed.
    # Search downward from tile_n_max in alignment-sized steps.
    best_dividing = 0
    for candidate in range(tile_n_max, 0, -alignment):
        if N_padded % candidate == 0:
            best_dividing = candidate
            break

    # Accept the divisor when it does not increase the number of
    # N-tiles.  Each tile incurs a global-memory pass in both passes of
    # the 2-pass softmax, so fewer tiles is always cheaper.  When the
    # divisor gives the same tile count as tile_n_max, prefer the
    # divisor because it eliminates the nearly-empty remainder tile
    # (e.g. N_padded=32768, tile_n_max=32512 -> 2 tiles with a 256-col
    # remainder vs divisor=16384 -> 2 even tiles with zero waste).
    #
    # When the divisor requires strictly more tiles (e.g. smaller
    # divisors of N_padded), stick with tile_n_max and handle the
    # single remainder tile via masked loads.
    if best_dividing > 0:
        div_tiles = N_padded // best_dividing  # exact division
        max_tiles = (N_padded + tile_n_max - 1) // tile_n_max
        if div_tiles <= max_tiles:
            return best_dividing
    return tile_n_max
