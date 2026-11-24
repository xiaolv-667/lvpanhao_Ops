import sys
sys.path.insert(0, "history_version")
import tilelang, tilelang.ascend.language as T
from design_probe_shift_placement import _ceil

def build(batch=17, d_mem=519, d_pad=256, max_conv_len=12, dtype="float16",
          block_m=16, block_n=64, block_k=256, num_stages=2):
    m_tiles=_ceil(batch,block_m); n_tiles=_ceil(d_pad,block_n); k_tiles=_ceil(d_mem,block_k)
    nshift=max_conv_len-1; grid=m_tiles*2*n_tiles
    tail = d_mem % block_k != 0

    @T.macro
    def _proj(E, W, kv, out_idx, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an):
        for kt in T.Pipelined(k_tiles, num_stages=num_stages):
            k0 = kt * block_k
            ak = T.min(d_mem - k0, block_k)
            T.copy(E[m0:m0+am, k0:k0+ak], e_l1[0:am, 0:ak])
            T.copy(W[k0:k0+ak, n0:n0+an], w_l1[0:an, 0:ak], transpose=True)
            if tail:
                if ak < block_k:
                    T.clear(e_l1[0:am, ak:block_k])
                    T.clear(w_l1[0:an, ak:block_k])
            T.copy(e_l1[0:am, 0:block_k], e_l0[0:am, 0:block_k])
            T.copy(w_l1[0:an, 0:block_k], w_l0[0:an, 0:block_k])
            T.gemm(e_l0, w_l0, res, transpose_B=True, clear_accum=(kt == 0))
        T.copy(res[0:am, 0:an], kv[out_idx, m0:m0+am, n0:n0+an])

    @T.prim_func
    def main(E: T.Tensor((batch,d_mem), dtype), W_K: T.Tensor((d_mem,d_pad), dtype),
             W_V: T.Tensor((d_mem,d_pad), dtype), conv_state: T.Tensor((batch,max_conv_len,d_pad), dtype),
             kv: T.Tensor((2,batch,d_pad),"float32"), new_conv_state: T.Tensor((batch,max_conv_len,d_pad), dtype)):
        with T.Kernel(grid) as bx:
            e_l1 = T.alloc_l1((block_m, block_k), dtype); w_l1 = T.alloc_l1((block_n, block_k), dtype)
            e_l0 = T.alloc_l0a((block_m, block_k), dtype); w_l0 = T.alloc_l0b((block_n, block_k), dtype)
            res = T.alloc_l0c((block_m, block_n), "float32")
            shift_ub = T.alloc_shared((nshift, block_n), dtype)
            T.annotate_buffer_versions({e_l1: num_stages, w_l1: num_stages, e_l0: num_stages, w_l0: num_stages})
            m0=(bx//(2*n_tiles))*block_m; which=(bx//n_tiles)%2; n0=(bx%n_tiles)*block_n
            am=T.min(batch-m0, block_m); an=T.min(d_pad-n0, block_n)
            if which == 0:
                _proj(E, W_K, kv, 0, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
                if m0 == 0:
                    for b in T.serial(batch):
                        T.copy(conv_state[b, 1:max_conv_len, n0:n0+an], shift_ub[0:nshift, 0:an])
                        T.copy(shift_ub[0:nshift, 0:an], new_conv_state[b, 0:nshift, n0:n0+an])
            else:
                _proj(E, W_V, kv, 1, e_l1, w_l1, e_l0, w_l0, res, m0, n0, am, an)
    return tilelang.compile(main, target="ascend", out_idx=[])

try:
    c = build()
    s = c.get_kernel_source() if callable(getattr(c,"get_kernel_source",None)) else c.kernel_source
    print("COMPILED")
    for ln in s.splitlines():
        if any(t in ln for t in ("asc_mmad(", "asc_fill", "asc_copy_l12l0a(", "asc_copy_gm2l1_nd2nz")):
            print(ln.strip()[:220])
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:300])
