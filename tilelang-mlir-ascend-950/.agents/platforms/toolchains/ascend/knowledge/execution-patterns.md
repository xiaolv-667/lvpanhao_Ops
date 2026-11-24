---
id: PL-950-mixed-epilogue
kind: pattern
status: verified
origin_task: 20261119-ascend950-knowledge-extended
toolchain: TileLang ascend 0.1.15; CANN 9.2.0 SDK / 9.1.0 simulator; 2026-11-19
toolchain_stamp: fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d
toolchain_id: ascend
hardware_id: ascend-950dt-9582
repro: repro/PL-950-mixed-epilogue.py
---

# Cube GEMM 与 Vector ReLU 的混合 kernel

FP16/BF16 输入 A=(16,32)、B=(32,16)，FP32 输出 C=(16,16)。Cube 完成 GEMM；同一个 kernel 内通过 dual_copy 将 FP32 L0C 的 (16,16) tile 分给两个 AIV 的 (8,16) UB，SIMT 完成 ReLU，再通过 dual_copy 写回完整输出。

两项通过，参考为 (A.float() @ B.float()).relu()，atol=rtol=1e-4，随机种子 4。代码采用默认自动调度，没有手写跨核 flag。

此用例确认 M 方向双 AIV 分片、同 dtype L0C→UB 和 UB→GM 拼接在该形状下的正确性。FP32 在 L0C→UB 路径中保持不变；不把它推广为该路径可同时转换 dtype，也不推广到 N 分片、任意尾块或多阶段混合流水。

补充边界（2026-11-23 实测，见 TRAP-950-l0c2ub-single-aiv）：本用例走的是 `dual_copy` 2:1 分片；整 tile 的 plain `T.copy`(L0C→UB) 在同一混合结构下只写一个 AIV 的 UB（`dual_dst_mode=0`），因此「L0C→UB 可用」的适用形态是 `dual_copy` 2:1，而不是任意整 tile copy。

适用环境与启动方法见 [核验来源](../references/evidence.md)。此条目的正确性结论限定于所列用例，不包含真机性能。

---
id: PL-950-manual-sync
kind: pattern
status: verified
origin_task: 20261119-ascend950-knowledge-extended
toolchain: TileLang ascend 0.1.15; CANN 9.2.0 SDK / 9.1.0 simulator; 2026-11-19
toolchain_stamp: fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d
toolchain_id: ascend
hardware_id: ascend-950dt-9582
repro: repro/PL-950-manual-sync.py
---

# 关闭自动调度后的单 tile Cube 管线同步

输入为 FP16 A=(16,32)、B=(32,16)，FP32 输出。compile 显式设置 tl.enable_auto_schedule=False，计算位于 T.Cube 区域。

本例在两次 GM→L1 搬运后设置并等待 MTE2_MTE1；L1→L0A/B 后设置并等待 MTE1_M；GEMM 后设置并等待 M_FIX；L0C→GM 后执行 PIPE_ALL 屏障。三个事件分别使用其事件命名空间中的 ID 0，不据此推断总可用事件数。

一次完整正确性检查通过，随机种子 5，参考为量化输入转 FP32 的矩阵乘，atol=rtol=1e-4。实际脚本包含所述配置与每个 set/wait 配对。

这是单核、单 tile、无缓冲复用循环的配对示例。没有证明跨核 flag 的使用方法，也没有认证加入循环后的写后读/读后写依赖或并发流水。

适用环境与启动方法见 [核验来源](../references/evidence.md)。此条目的正确性结论限定于所列用例，不包含真机性能。

---
id: PL-950-atomic-splitk
kind: pattern
status: verified
origin_task: 20261119-ascend950-knowledge-extended
toolchain: TileLang ascend 0.1.15; CANN 9.2.0 SDK / 9.1.0 simulator; 2026-11-19
toolchain_stamp: fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d
toolchain_id: ascend
hardware_id: ascend-950dt-9582
repro: repro/PL-950-atomic-splitk.py
---

# 两个 Cube 核的 split-K 原子累加

A=(16,64)、B=(64,16)，两个核各处理 K=32，分别在 FP32 L0C 中生成部分和。写回同一 GM 输出前设置 set_atomic("add","float32")，写回后调用 set_atomic_none()。FP16/BF16 输入各一项。

调用者先分配全零 FP32 输出，再将它作为普通输入输出参数传给 kernel；kernel 不使用 out_idx 自动分配未初始化输出。两项以 atol=rtol=1e-4 对照完整矩阵乘，随机种子 6。

这是通过原子写回完成的单次 kernel 跨核求和，不是全局 barrier 示例。结果在 kernel 完成并返回主机后检查；不保证同一 kernel 内某个消费者可以在两个核写完前读取最终值。每次独立调用前需要满足输出初始化契约。本例不评估原子争用、可重复性或性能。

适用环境与启动方法见 [核验来源](../references/evidence.md)。此条目的正确性结论限定于所列用例，不包含真机性能。
