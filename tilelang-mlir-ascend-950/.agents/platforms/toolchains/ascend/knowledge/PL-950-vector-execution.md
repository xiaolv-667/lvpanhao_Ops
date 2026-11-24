---
id: PL-950-vector-execution
kind: pattern
status: verified
origin_task: 20261118-platform-abstraction simulator verification; upstream TileOPs reference
toolchain: TileLang ascend 0.1.15 + CANN 9.1 simulator, 2026-11-18
toolchain_stamp: fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d
repro: repro/PL-950-vector-execution.py
toolchain_id: ascend
hardware_id: ascend-950dt-9582
device: Ascend950DT_9582
apis: [T.SimdVF, T.SimtVF, T.copy]
---

# 950 向量执行和尾块

TileOPs 的 Ascend Abs 实现对固定 SIMD 向量宽度的路径在 UB 内做尾块填充，并按有效元素数搬入、写回 GM。该实现也包含 SIMT 分支。根据 dtype 与数据访问方式选择路径，记录填充后的 UB 占用。

证据见 [TileOPs Ascend Abs](https://github.com/yyttt6/TileOPs/blob/ascend950/src/tileops/kernels/ascend/elementwise/abs.py)。

本地验证：950 DT 仿真器上，FP32 的 N=256、257 分别通过 SIMD 与 SIMT 精度检查，输出与 torch.abs 完全一致。该验证覆盖向量执行与尾块读写；不包含真机时延或吞吐结论。
