---
id: PL-950-row-primitives
kind: pattern
status: verified
origin_task: 20261119-ascend950-knowledge
toolchain: TileLang ascend 0.1.15; CANN 9.2.0 SDK / 9.1.0 simulator; 2026-11-19
toolchain_stamp: fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d
toolchain_id: ascend
hardware_id: ascend-950dt-9582
repro: repro/PL-950-row-primitives.py
apis: [T.SimtVF, T.copy, T.cast, T.alloc_reducer, T.reducer_init, T.reducer_update, T.finalize_reducer]
---

# 950 单行转换、归约与广播

## 验证范围

CANN 9.1 仿真器的 Ascend950DT 模式，锁定环境通过检查。该证据为仿真正确性，不是真机性能。

输入为 `torch.linspace(-3, 2, N)` 再量化到 FP32、FP16 或 BF16，N 分别为 64 和 63。输出均为 FP32；每行使用一个核、64 个 SIMT 线程、64 元素 UB 分配。下列四种计算各覆盖 3 种输入类型 × 2 个长度，共 24 项。

| 计算 | 实现与参考 |
|---|---|
| 行最大值广播相加 | FP32 max reducer，结果广播到每个有效元素，输出 `x + max(x)` |
| 行求和 | FP32 sum reducer，结果广播写到每个有效元素 |
| softmax | 先减有效行最大值，再 exp、sum、除法归一化 |
| 无权重 RMSNorm | FP32 平方和除以有效长度，`eps=1e-5`，再 sqrt 和除法 |

误差阈值为 `atol=1e-5, rtol=1e-5`。参考使用量化后的输入转成 FP32 计算。

## 实现要点

源码中 GM↔UB 搬运同时限定输入和输出有效 region。N=63 时，FP32 fragment 的无效 lane 初始化为 0；max reducer 的无效贡献为负无穷；softmax 的无效 exp 贡献为 0。归一化只处理有效元素，RMSNorm 分母使用 N，不使用填充宽度。

SIMT reducer 按 `alloc_reducer → reducer_init → reducer_update → finalize_reducer` 使用。每个输出元素读取同一已完成的行统计量，从而表达广播。该脚本没有手动跨核同步，也没有使用 SIMD shared→shared 的 `T.reduce_sum` 路径。

## 适用边界

这是算法表达和尾块处理的最小正确性示例，保留了便于统一检查的辅助统计量，不是优化模板。不覆盖大行分块、多行分核、低精度输出、特殊值/极值、任意 dtype 转换或跨核归约；不包含速度和最佳参数结论。

环境与接口来源见 [evidence.md](../references/evidence.md)。执行时使用任务平台 `execution.prototype_command`，将脚本路径替换为本条目的自包含脚本绝对路径。`--mode` 可选单个计算以缩小验证范围。
