---
id: PL-950-gemm-tiles
kind: pattern
status: verified
origin_task: 20261119-ascend950-knowledge-extended
toolchain: TileLang ascend 0.1.15; CANN 9.2.0 SDK / 9.1.0 simulator; 2026-11-19
toolchain_stamp: fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d
toolchain_id: ascend
hardware_id: ascend-950dt-9582
repro: repro/PL-950-gemm-tiles.py
---

# GEMM 输入布局、K 分块与尾块

FP16/BF16 输入、FP32 输出，分别验证 B 按 (K,N) 和 (N,K) 存放。每种组合运行两组 M/N/K：
- (16,16,64)：K 切成两个 32 元素块，单阶段配置。
- (17,19,35)：M/N 分成 16×16 tile，K 切成 32+3；两阶段配置和显式两版本 L1/L0 操作数。

共八项，阈值 atol=rtol=1e-4，参考为量化输入转 FP32 后的矩阵乘。随机种子 2。

B=(N,K) 时按 region 搬入；B=(K,N) 时 GM→L1 copy 使用 transpose=True，使后续 B 缓冲保持 (N,K) 布局。L0 GEMM 使用 transpose_B=True。A 保持 (M,K)，本例没有测试 transpose_A=True。

每次 copy 和 gemm 的 region 使用当前有效 M/N/K；物理分配仍按 16×32 操作数与 16×16 L0C。clear_accum 仅在第一个 K 块为 True；输出只写当前有效 M/N。跨 K 块保留累加器，避免每块清零丢失前面的部分和。

两阶段配置通过正确性验证，不代表发生了理想重叠或性能提升。配置参数与最终调度结果是不同证据。本条目不覆盖量化 GEMM、FP32 Cube 输入及所有分块组合。

适用环境与启动方法见 [核验来源](../references/evidence.md)。此条目的正确性结论限定于所列用例，不包含真机性能。
