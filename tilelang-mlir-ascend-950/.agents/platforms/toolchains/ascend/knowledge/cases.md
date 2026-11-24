# 950 案例索引（batch2 迁移）

本文件是算子目录级案例索引：只记录路径、一句话与触发条件，不复制内容。索引项对应的实现与
复盘仍留在其任务目录（`origin_task` 记录归属），条目讨论的事实已内嵌在各主题文件中。

---
id: CASE-batch2-moe-permute-align
kind: case
status: verified
origin_task: batch2-20261123 (moe_permute_align, 3 kernels)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: none
family: moe
mode: simt
dtype: [int32, float16, bfloat16]
provenance: 溯源见 origin_task；目录含 3 个逐函数 RETROSPECTIVE（整数路由/前缀扫描/小批融合）、5 个 repro 与 12 条 manifest workload
---

# moe_permute_align（SIMT 整数路由三 kernel 正例）

路径 `examples/moe_permute_align`（`_make_align_kernel` / `_make_scatter_kernel` /
`_make_small_batch_kernel`）：无 scatter-add / 无 warp shuffle / 无跨核 barrier 平台上，用
「expert-lane 比较累加 + 局部直方图 GM 归并」与「原子 bump allocator」实现 permute-align，全部
整数算子 `max|diff| = 0`。

触发：MoE permute-align、直方图 + 前缀 + 散射类迁移；需要「双 buffer 零 halo 前缀扫描」或
「SIMT 线程宽度上限」的可用形态时。反例价值：同一目录记录了 UB→UB 移位 copy 与每核越界读两类
静默错（TRAP-950-shared-shift-copy / TRAP-950-core-slice-overread）。

---
id: CASE-batch2-engram-gate-conv
kind: case
status: verified
origin_task: batch2-20261123 (engram_gate_conv + engram_gate_conv_bwd)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: none
family: sequence_modeling
mode: simd
dtype: [float16, bfloat16]
provenance: 溯源见 origin_task；同族前向/反向各 1 个函数目录，Stage 5 均为 0 次修复 attempt，含 `repro/TRAP-950-simd-ub-store-load.py` 与 `TRAP-950-simd-strip-pitch.py`
---

# engram gate-conv 前向/反向（SIMD 滚动窗口 + 确定性归约正例）

路径 `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel` 与
`examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel`：单 launch 融合（`RC+3` 行仿射窗口 +
3 行 halo 重算）与「per-core fp32 部分和 + 第二 launch 按核号有序求和」的确定性权重梯度归约，
方向 25 例 / 前向 15 例真机全过且 repeat 逐位一致。

触发：带跨行窗口依赖的卷积类迁移、需要避开跨核原子而保持确定性的归约设计、SUB 32 B 行距的
标量条带暂存。反例价值：同目录固化 `TRAP-950-simd-ub-store-load`（VST→VLD 非确定性）与
`TRAP-950-simd-strip-pitch`（507035 行距）。

---
id: CASE-batch2-shared-expert-mlp
kind: case
status: verified
origin_task: batch2-20261123 (shared_expert_mlp, 3 kernels)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: none
family: moe
mode: cube
dtype: [float16, bfloat16]
provenance: 溯源见 origin_task；目录含 `_gemm_basic_kernel`（L1 操作数 GEMM 1.21 ms ≈ 198 TF/s）、`splitk_reduce_kernel` 与 `_silu_mul_fused_kernel`，3 个 repro
---

# shared_expert_mlp（Cube GEMM 分解 + host 契约夹取正例）

路径 `examples/shared_expert_mlp`：一个 Op 分解为 gate/up GEMM + gated activation + down GEMM 的
多个 TileLang 子 kernel；host 侧按「`block_m := min(block_m, align16(m))`、split-K 精确 tile 前置
断言、`block_k` 64/32/16 阶梯 + 零 pad K」夹取，使 GPU 时代默认配置在 decode 段（`T=1/32`）可用。

触发：MoE MLP / 多级 GEMM 分解迁移、wrapper 需按子 kernel 构建期前置条件翻译配置、需要
「L1 操作数 GEMM + fixpipe 量化」性能锚点时。反例价值：Stage 5 的 5 次集成修复（配置前置条件、
两级/一级工厂、重复选择块、`block_k | k`）均源自此处。

---
id: CASE-batch2-mla-varlen-reuse
kind: case
status: verified
origin_task: batch2-20261123 (mla_varlen_bwd / _bwd_kernel)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: none
family: attention
mode: hybrid
dtype: [float16, bfloat16]
provenance: 溯源见 origin_task；§0.8 复用/差异分区、R1–R7 运行期探针与 packed 三件套（融合前缀解码 / head-major 统计量 / dk/dv 双分支）记录在 mla_varlen_bwd/_bwd_kernel/RETROSPECTIVE.md
---

# mla_varlen_bwd packed（同源工厂复用 dense 已验证核心正例）

路径 `examples/mla_varlen_bwd/_bwd_kernel`：与 dense 版共用工厂，只重推三处 packed 差异
（融合式 running-prefix decode 取代 warp-shuffle 前缀扫描、head-major `lse/delta` 保持 256 B 连续段、
逐 request 边界收口）；动态 grid `(total_kv//64 + B) * H` 在混合 kernel 上真机成立（9 / 2 / 8160 /
16640 块），OOB 哨兵 0 次触碰。

触发：同源 dense/packed 双变体迁移、动态 grid + `__mix__` 混合结构、request 级 `dk/dv`（赋值型）
与 `dq`（累加型）语义分家。反例价值：平台 OOB clamp 只认张量 extent，request 边界必须自己收口。

---
id: CASE-batch2-paged-cache-write
kind: case
status: verified
origin_task: batch2-20261123 (paged_cache_write / _paged_cache_write_kernel)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: none
family: attention
mode: simt
dtype: [float16, bfloat16, fp8_e4m3]
provenance: 溯源见 origin_task；两次 repro（fp8 标量 store、NaN 符号位）与「未命中行逐字节不变」集合断言记录在 paged_cache_write/_paged_cache_write_kernel/RETROSPECTIVE.md Stage 3
---

# paged_cache_write（逐元素 scatter 写 + fp8 路径正例/反例）

路径 `examples/paged_cache_write/_paged_cache_write_kernel`：行索引 × 列段折叠进一维
`T.Kernel(nblocks)`、`slot == -1` 用 lane 级谓词跳过、fp8 路径 K/V 拆两个 `T.Parallel` 并加
`per_thread >= 2` 编译期 guard；L0 10/10、L1 2/2、L2 3/3、boundary 10/10 逐字节通过。

触发：分页 KV 写 / 稀疏行列 scatter / fp8 GM 直写路径；需要「原地写算子的精确验证套路」
（`atol=rtol=0` + 原始字节比对 + 未命中行不变）时。反例价值：单循环双 fp8 store 与每线程 <2 元素
都会退化为标量 e4m3 store 并写 0（见 TRAP-950-fp8-scalar-store）。
