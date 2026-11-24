---
id: PL-950-cube-tile
kind: pattern
status: verified
origin_task: 20261119-ascend950-knowledge
toolchain: TileLang ascend 0.1.15; CANN 9.2.0 SDK / 9.1.0 simulator; 2026-11-19
toolchain_stamp: fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d
toolchain_id: ascend
hardware_id: ascend-950dt-9582
repro: repro/PL-950-cube-tile.py
apis: [T.alloc_l1, T.alloc_l0a, T.alloc_l0b, T.alloc_l0c, T.copy, T.gemm]
---

# 950 显式 L0 操作数的单 tile GEMM

## 验证范围

CANN 9.1 仿真器 Ascend950DT 模式，矩阵形状 A=(16,32)、B=(32,16)、C=(16,16)，一个 kernel、一个核。输入分别验证 FP16 和 BF16，L0C 与输出为 FP32，无转置。

输入为 CPU 固定随机种子 0 生成的正态数据，再量化到相应输入类型。参考为量化输入转 FP32 后的矩阵乘，阈值 `atol=1e-4, rtol=1e-4`。

## 已验证路径

`GM → L1 → L0A/L0B → T.gemm → L0C → GM`。

L1 和 L0 操作数显式分配。第一次 GEMM 使用 `clear_accum=True`，通过默认编译流水线完成调度与同步。脚本没有手动设置布局或同步事件。

对应源代码分配的逻辑占用为：L1 两个操作数合计 2048 B，L0A/L0B 各 1024 B，L0C 为 1024 B。该计算不替代最终编译分配检查。

## 适用边界

只认证上述单 tile、输入类型、输出类型和布局。K 维多轮累加、转置、非整齐尺寸、低精度输出、量化 GEMM、Cube/Vector 混合流水及性能收益均未由本例验证。

接口与来源见 [编程规则](../references/programming.md) 和 [核验来源](../references/evidence.md)。执行使用任务平台 `execution.prototype_command` 与本条目的自包含脚本。
