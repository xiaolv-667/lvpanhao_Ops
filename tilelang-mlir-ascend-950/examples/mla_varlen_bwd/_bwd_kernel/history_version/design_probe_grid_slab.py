"""Stage 1 设计期编译探针 P6（v1 修订：单次 launch 的 grid ≤ 65535）。

无设备（仅前端 + 编译）：`ASCEND_RT_VISIBLE_DEVICES=""`。按平台 execution 上下文
在任务目录内运行（仓根有 dev-root `tilelang/`，会遮蔽 site-packages 的 0.1.15）：

    cd examples/mla_varlen_bwd/_bwd_kernel
    TMPDIR=/workspace ASCEND_RT_VISIBLE_DEVICES="" \
        /workspace/venv-ascend950/bin/python history_version/design_probe_grid_slab.py

探针对象（DESIGN.md §5.6 / §3.5.4 P6）
--------------------------------------
把 v0 的 `tile = bid // heads, h = bid % heads`（grid = tiles*heads，实测会越过
65535 静默截断）改成 **head slab 多次 launch**：

    grid   = (total_kv // 64 + batch) * SLAB          # SLAB 编译期常量
    tile   = bid // SLAB                              # 静态除数
    h      = h_off[0] + (bid % SLAB)                  # h_off: int32[1] 运行期标量
    if h < heads:  <decode + live + body>              # 尾 slab 守卫，空 block 立即返回

P6 用**完整 P2 计算体**（5 个 NT GEMM + dual_copy 半区 + 运行期掩码 + store-atomic
+ 块尾双分支），只替换任务映射；几何取 `heads=3, SLAB=2`，故意不整除以覆盖尾
slab（launch 1 有 1 个活跃 head + 1 个空 block）。判定：
  * 编译通过；
  * 设备码里 `block_idx` 的除/模按编译期常量 SLAB 出现；
  * `h_off` 以 GM 标量读出现，且进入全部 GM region（q/k/v/do/lse/delta/dq_acc/dk/dv）；
  * 尾 slab 守卫进入设备码（`h < heads` 分支包住整个 body）；
  * 原子 store 仍夹在 set_atomic / set_atomic_none 之间。
"""

import sys

import tilelang
import tilelang.ascend.language as T

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import design_probe_compile as base  # noqa: E402  复用 P2 的计算体（同一文件内 @T.macro）

B = base.B
H = 3  # 故意让 SLAB 不整除：SLAB=2 -> 2 次 launch，尾 slab 只有 1 个 head
SLAB = 2
BN, BM, HN, DQK, DV, DT, SCALE = (
    base.BN, base.BM, base.HN, base.DQK, base.DV, base.DT, base.SCALE,
)


def build_slab_grid():
    total_q = T.dynamic("total_q")
    total_kv = T.dynamic("total_kv")
    ntiles = T.floordiv(total_kv, BN) + B
    grid = ntiles * SLAB  # 单次 launch 的 grid：<= 65535 由 host 侧 plan_launch 保证

    @T.prim_func
    def main(
        q: T.Tensor((total_q, H, DQK), DT),
        k: T.Tensor((total_kv, H, DQK), DT),
        v: T.Tensor((total_kv, H, DV), DT),
        dout: T.Tensor((total_q, H, DV), DT),
        lse: T.Tensor((H, total_q), "float32"),
        delta: T.Tensor((H, total_q), "float32"),
        cu_q: T.Tensor((B + 1,), "int32"),
        cu_kv: T.Tensor((B + 1,), "int32"),
        h_off: T.Tensor((1,), "int32"),
        dq_acc: T.Tensor((total_q, H, DQK), "float32"),
        dk: T.Tensor((total_kv, H, DQK), DT),
        dv: T.Tensor((total_kv, H, DV), DT),
    ):
        with T.Kernel(grid) as bid:
            tile = bid // SLAB  # 静态除数（SLAB 编译期常量）
            h = h_off[0] + bid % SLAB  # 运行期 head 原点 + 静态槽位
            sid = T.call_extern("int64", "asc_get_sub_block_id")
            d = base._alloc()
            req = T.alloc_local([1], "int32")
            n0l = T.alloc_local([1], "int32")
            cnt = T.alloc_local([1], "int32")
            live = T.alloc_local([1], "int32")
            # 尾 slab 守卫：h >= heads 的 block 只算两个标量即返回（无 DMA、无标量读）
            if h < H:
                base._decode(cu_kv, tile, req, n0l, cnt, live)
                if live[0] == T.int32(1):
                    q_start = cu_q[req[0]]
                    kv_start = cu_kv[req[0]]
                    lq = cu_q[req[0] + 1] - q_start
                    lkv = cu_kv[req[0] + 1] - kv_start
                    base._body(
                        q, k, v, dout, lse, delta, dq_acc, dk, dv, *base._args(d),
                        h, tile, sid, n0l[0], lq, lkv, lkv - lq, kv_start, q_start,
                        False, False,
                    )

    return main


MARKS = (
    "ascend_gemm_l1",
    "asc_copy_l0c2ub",
    "ascend_nd2nz_scatter",
    "asc_copy_l0c2gm",
    "asc_set_atomic_add_float",
    "asc_set_atomic_none",
    "__mix__",
    "asc_get_sub_block_id",
    "asc_fill_l1",
    "asc_copy_gm2l1_nd2nz",
    "asc_copy_gm2l1_dn2nz",
    "asc_copy_ub2gm",
    "asc_sync_",
)


def main():
    fn = build_slab_grid()
    src = tilelang.compile(fn, target="ascend").get_kernel_source()
    marks = sorted({kw for kw in MARKS if kw in src})
    print(f"[COMPILE_OK] P6 slab_grid -- source {len(src)} B")
    print(f"  marks: {marks}")
    # 任务映射证据：静态 SLAB 的除法/取模 + 运行期 h_off 标量读 + 尾 slab 守卫
    probes = {
        "static_div_by_slab": f"/ {SLAB}" in src or f">> {SLAB.bit_length() - 1}" in src,
        "h_off_scalar_read": "h_off" in src,
        "head_guard": "< 3" in src or "<3" in src,
        "grid_expr": "total_kv" in src,
    }
    for name, ok in probes.items():
        print(f"  [{'OK' if ok else 'MISS'}] {name}")
    for line in src.splitlines():
        if "h_off" in line:
            print("  | " + line.strip()[:160])
            break
    print("[PROBE_SUMMARY] 1/1 COMPILE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
