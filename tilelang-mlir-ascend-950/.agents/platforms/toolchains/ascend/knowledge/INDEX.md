# Ascend 950 知识索引

条目规范见 [.agents/skills/_shared/standards/knowledge-entry-schema.md](../../../../skills/_shared/standards/knowledge-entry-schema.md)。

适用于 Ascend950DT_9582 与 ascend 工具链。阶段技能按此索引读取已有条目，并核对来源及任务工具链版本。

- [PL-950-vector-execution：950 向量执行和尾块](PL-950-vector-execution.md)

- [PL-950-row-primitives：单行转换、归约与广播](PL-950-row-primitives.md)
- [PL-950-cube-tile：显式 L0 操作数的单 tile GEMM](PL-950-cube-tile.md)
- [ALG-950-formulas：CPU 数学参考，非 950 kernel 认证](ALG-950-formulas.md)
- [接口与资源规则](../references/programming.md)
- [迁移与诊断](../references/migration-diagnostics.md)
- [核验来源与范围](../references/evidence.md)

- [算法候选](algorithm-candidates.md)
- [瓶颈模式](bottlenecks.md)
- [能力缺口](capability-gaps.md)
- [PL-950-normalization：带权重的 LayerNorm 与 RMSNorm](operator-patterns.md)
- [PL-950-attention：单头 dense attention 的 SIMT 正确性参考](operator-patterns.md)
- [PL-950-gemm-tiles：GEMM 输入布局、K 分块与尾块](gemm-tiles.md)
- [PL-950-gemm-l1-fill：L1 填充带宽判据（block_k burst + tile 顶 L0C 上限）](gemm-tiles.md)
- [CONST-950-gemm-mte2-burst：MTE2/L2/HBM/Cube 实测常数](constants.md)
- [PL-950-mixed-epilogue：Cube GEMM 与 Vector ReLU 的混合 kernel](execution-patterns.md)
- [PL-950-manual-sync：关闭自动调度后的单 tile Cube 管线同步](execution-patterns.md)
- [PL-950-atomic-splitk：两个 Cube 核的 split-K 原子累加](execution-patterns.md)
- [PL-950-conversion-layout：SIMT 转换特殊值与带广播的转置](numerical-boundaries.md)
- [PL-950-simd-reduce：显式 SIMD 寄存器行归约](numerical-boundaries.md)
- [PL-950-softmax-boundaries：FP32 softmax 的有限极值与非有限输入](numerical-boundaries.md)
- [TRAP-950-simd-shared-reduce：高层 SIMD 二维 shared→shared 归约的输出差异](numerical-boundaries.md)

## batch2 实测集群（2026-11-23，11 算子 harness 迁移）

- [GM↔UB 存储与搬运陷阱](traps-950pr-store-copy.md)：**TRAP-950-rowload-dma-chain（逐行 GM→UB copy 链按描述符数计费，
  27 条单行 6.0 µs vs 6 条整窗 3.2 µs；行级预取反向 +16.5%）** / TRAP-950-fp8-scalar-store（标量 e4m3 写 0）/
  TRAP-950-rowstore-mte3-drain（落盘侧对偶） /
  CONST-950-vcvt-narrow-store-pack-b32（窄化落盘 PK_B32）/ TRAP-950-shared-shift-copy（UB→UB 移位丢项）/
  TRAP-950-l0c2ub-single-aiv（整 tile 只落一个 AIV）/ TRAP-950-l0c-region-store-stride（源截断行距）/
  TRAP-950-partial-gm2ub-load（32 B 上取整覆盖）/ TRAP-950-simd-strip-pitch（2D 行距 32 B → 507035）/
  TRAP-950-simd-ub-store-load（VST→VLD 需 mem_bar）/ TRAP-950-rowstore-mte3-drain（散写逐行
  `T.copy` 每行排空 MTE3 → 16-bit 走逐 lane 落盘，实测 80.4→48.2 µs）
- [kernel 结构与执行陷阱](traps-950pr-execution.md)：TRAP-950-simt-thread-cap（threads ≤ 2048）/
  TRAP-950-grid-over-65535（grid 上限）/ TRAP-950-aiv-sid-vector-frame（`T.Vector as sid` 值错）/
  TRAP-950-prim-func-helper-body（body 不可抽 helper）/ TRAP-950-scatter-eid-upper-bound（自动守卫按 extent）/
  TRAP-950-core-slice-overread（chunk 双夹紧）
- [转换/舍入语义](numerics-950pr-conversion.md)：CONST-950-vcvt-rs-disable-ieee（sat 无效且 IEEE）/
  CONST-950-vcvt-sat-false-overflow-inf（128/128 位等价）/ CONST-950-fp8-nan-sign（NaN 符号位规范化）/
  CONST-950-silu-mul-a1-one-ulp（5 算符等价 0 ulp）
- [已实测可用形态](patterns-950pr-verified-forms.md)：**PL-950-window-bulk-copy（整窗 2-D copy 取代逐行装载，-19.9%）** /
  CONST-950-kernel-launch-floor-48blk（48 块 1-D grid 的 launch 地板 ≈2.7 µs）/ CONST-950-aiv-sid-capture /
  CONST-950-ub-value-as-gm-row-index / PL-950-simt-scalar-broadcast / PL-950-dynamic-pipeline-2versions /
  CONST-950-mmad-ktail-nonaligned / CONST-950-clear-pad-all-versions / CONST-950-tail-no-multibuffer /
  CONST-950-dma-vs-lane-gm / CONST-950-simt-lane-vec-8b
- [案例索引](cases.md)：CASE-batch2-moe-permute-align / CASE-batch2-engram-gate-conv /
  CASE-batch2-shared-expert-mlp / CASE-batch2-mla-varlen-reuse / CASE-batch2-paged-cache-write
