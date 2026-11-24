# NPUIR 迁移实现映射

以下保留原迁移参考中的 NPUIR 实现映射与约束。通用迁移流程仍由共享 migration-analysis.md 定义；具体 API 使用前按当前 NPUIR 版本核验。

### 4.1 GPU TileLang 常见优化手段识别清单

| 优化手段 | 典型代码特征 | 目的 | 依赖的源硬件特性 | 初判 |
|----------|-------------|------|-----------------|------|
| SMEM tiling | `alloc_shared` + 分块 `T.copy` | 减少 GM 重复访问 | shared memory | 硬件强相关 |
| coalesced 访问 | 尾轴连续读写 | 访存带宽 | memory coalescing | 硬件强相关 |
| Tensor Core / mma | `T.gemm` / mma fragment | 矩阵算力 | tensor core | 硬件强相关（NPU 有 Cube 等价） |
| warp shuffle 归约 | shuffle / intra-warp reduce | 低延迟跨线程通信 | warp（32 线程） | 硬件强相关 |
| 异步搬运 / 多级流水 | `cp.async`、`T.Pipelined` | 访存计算重叠 | 异步拷贝引擎 | 硬件强相关（NPU 有 MTE 等价） |
| double buffer | 双 buffer 交替 | 流水不 stall | — | 模式可移植 |
| swizzle / 布局重排 | index 重排、pad | 消 bank conflict | SMEM bank 结构 | 硬件强相关（NPU 不适用） |
| register tiling / 累加 | fragment 累加、寄存器分块 | 数据驻留寄存器 | 大寄存器堆 | 硬件强相关 |
| persistent kernel | block 常驻循环处理任务 | 消 kernel 启动开销 | — | 模式可移植 |
| fused epilogue | 归约/激活融合进主循环 | 消中间写回 | — | 模式可移植 |
| online 算法（如 online softmax） | 分块局部 max/sum 增量更新 | 消 O(N) 中间缓冲 | **无**（纯算法层） | **可移植** |
| 向量化访存 | 连续多元素一次读写 | 带宽 | — | 模式可移植 |

### 5.3 GPU → Ascend NPU 能力映射表（等价替换速查）

| GPU / 通用概念 | Ascend NPU 等价能力 | 处置建议 | 注意事项 |
|----------------|--------------------|---------|----------|
| shared memory | L1（Cube 输入）/ UB（Vector 数据） | 等价替换 | 容量与对齐不同：按目标型号配置和工具链规则核对 L1、UB 容量及对齐 |
| Tensor Core mma | Cube 单元（`T.gemm` + L0A/L0B/L0C） | 等价替换 | 按工具链规则核对分形与 NZ 布局，L0C 容量读取目标型号预算 |
| warp shuffle 归约 | Vector 归约（`T.reduce_sum/max/min`） | **重新设计** | 并行层级不同：warp 32 线程内 shuffle → 向量整行归约，需重排归约策略 |
| cp.async / 多级异步流水 | `T.Pipelined` + MTE 异步搬运 | 等价替换 | num_stages 需按 UB/L1 容量重新计算 |
| double buffer | 多 buffer 交替 | 等价替换 | — |
| swizzle 防 bank conflict | 目标平台支持的布局及装载方式 | 按平台核对后舍弃或替换 | 核对目标内存组织，不保留仅适用于源硬件的处理 |
| block × thread 两级并行 | block 级任务划分 + 向量化元素并行 | **重新设计** | 本项目 `T.Kernel` 只接受一维 block 数 |
| register 累加 | L0C 累加器（fp32）/ UB 中间缓冲 | 等价替换 | 累加顺序变化需评估精度影响 |
| persistent kernel | `T.Persistent` | 等价替换 | — |
| fused epilogue | 按目标平台核对矩阵出数后的后处理或融合 | 等价替换或重新设计 | 取决于融合点在 Cube 侧还是 Vector 侧 |
| grid-stride loop | block 内循环处理多任务 | **重新设计** | 一维 block 任务映射 |
| 三维 Kernel（`T.Kernel(m,n,b)`） | 三维乘积展开为一维 block 数 | **重新设计** | 本项目不支持三维 Kernel（见 ascend-constraints.md） |
| online 算法 / 分块累加 | 同款算法结构 | **保留** | 纯算法层，与硬件无关 |

## NPUIR 重设计约束与示例

- NPU 硬件约束（[decision-tree.md](decision-tree.md) §3：分形限制 / 对齐 / 各级存储容量 / 一维 Kernel）。

| 模式 | 适用场景 | 设计要点 |
|------|---------|---------|
| **并行结构重设计** | block×thread 两级并行、grid-stride | 一维 `T.Kernel(total_blocks)`；元素级并行交给 Vector 向量化（`T.Parallel`）；矩阵计算交给 Cube 分形；block 数与任务映射重排 |
| **数据搬运重设计** | cp.async + SMEM 多级流水 | 分级搬运 `GM → L1 → L0A/L0B`（Cube）或 `GM → UB`（Vector）+ `T.Pipelined`；stage 数按 L1/UB 容量与 buffer 数重新计算 |

## NPUIR 存储访问错误识别

| 子类型 | 识别信号 | 路由策略 |
|--------|---------|----------|
| 内存层级越级 | stderr 提示 GM/L1/UB/L0 访问违规 | Stage 3 内重试，提示 Developer 复核 AGENTS.md 原则 4（硬件内存层级） |
