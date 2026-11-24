# NPUIR 专用约束：tileops-adaptation

以下保留原平台的 API、命令和参数说明；公共适配与采集流程仍按共享文档执行。

| O1 | `_logsumexp_kernel_single(M, N, dtype)` — `@tilelang.jit` + `@T.prim_func` 构建 CUDA 内核；返回的 callable 接受 `(block_m, threads)` | NPU 实现：`@tilelang.jit(target="npuir")` + `@T.prim_func` 构建 NPUIR 内核；callable 只接受 `block_m`（无 `threads`）；不做 alignment padding，直接用原始 N | `kernels/reduction/logsumexp/logsumexp.py` | NPU 无 `threads` 概念；NPUIR `vsel` 在 mask 与 tensor 形状不一致时报错，故不做 padding |
| O2 | `_logsumexp_kernel_tiled(M, N, dtype, tile_n)` — 多 tile 路径；返回的 callable 接受 `(block_m, threads)` | NPU 实现：`@tilelang.jit(target="npuir")`；callable 只接受 `block_m`（`tile_n` 已在闭包固化）；不做 alignment padding | 同上 | 同上 |
