import sys
sys.path.insert(0, "history_version")
import tilelang, tilelang.ascend.language as T

def build(batch=8, d_pad=256, max_conv_len=18, dtype="float16", block_n=64, m_tiles=1):
    n_tiles = -(-d_pad // block_n); nshift = max_conv_len - 1
    grid = m_tiles * 2 * n_tiles

    @T.prim_func
    def main(conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype),
             new_conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype)):
        with T.Kernel(grid) as bx:
            buf = T.alloc_shared((batch, nshift, block_n), dtype)
            n0 = (bx % n_tiles) * block_n
            an = T.min(d_pad - n0, block_n)
            if (bx // n_tiles) % 2 == 0:
                T.copy(conv_state[0:batch, 1:max_conv_len, n0:n0+an], buf[0:batch, 0:nshift, 0:an])
                T.copy(buf[0:batch, 0:nshift, 0:an], new_conv_state[0:batch, 0:nshift, n0:n0+an])
    return tilelang.compile(main, target="ascend", out_idx=[])

try:
    c = build()
    s = c.get_kernel_source() if callable(getattr(c,"get_kernel_source",None)) else c.kernel_source
    print("COMPILED")
    for ln in s.splitlines():
        if any(t in ln for t in ("asc_copy_gm2ub", "asc_copy_ub2gm", "for (int32_t", "ASC_IS")):
            print(ln.strip()[:200])
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:300])
