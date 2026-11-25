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

---
id: PL-950-deterministic-slot-alloc
kind: pattern
status: verified
origin_task: moe_permute_align-moe_permute_align-20261124T171256Z (Stage 4)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-25
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/PL-950-deterministic-slot-alloc.py
provenance: 溯源见 origin_task；`examples/moe_permute_align_perf/perf_opt/opt_log.md` §3/§5、
  `perf_opt/moe_permute_align.py`（最终实现）、原始采集 `perf_opt/profiles/r{0_base,3_final}/msprof/*`
apis: [T.atomic_add, T.SimtVF, T.Parallel, T.copy]
dtype: [int32]
---

# 计数散射的槽位改为「可推导 + 片上游标」：替换每元素一次 GM 原子

**慢 → 快的关键代码更改**（计数排序 / bump allocator 类，slot allocator）：

```python
# before: 每路由一次 GM 原子（GM 游标 = cumsum[e]），586.0 us @ 32768 路由 / 56 核
with T.Kernel(nblocks) as bid, T.SimtVF(threads=128):
    for i in T.serial(ept):
        for j in T.Parallel(128):
            idx = (bid * ept + i) * 128 + j
            eid = flat[T.min(idx, numel - 1)]
            if idx < numel:
                slot = T.atomic_add(cumsum[eid], T.int32(1), return_prev=True)
                sorted_token_ids[slot] = idx

# after: 槽位由「上游已有的每核直方图 + 本核排他偏移」推导，原子只落在 UB（7.2 us）
#   上游（直方图 launch）产出 partial[core, e]；本 launch 每核把
#   cursor[e] = cumsum[e] + Σ_{r < core} partial[r, e] 搬进 UB 后自增
with T.Kernel(g) as a, T.SimtVF(threads=128):
    s_cur = T.alloc_shared((e_pad,), "int32")
    eid = T.alloc_local([1], "int32")
    slot = T.alloc_local([1], "int32")
    for k in T.serial(e_pad // 128):                     # cursor row -> UB
        for j in T.Parallel(128):
            s_cur[k * 128 + j] = cursor[a, k * 128 + j]
    for r in T.serial(rounds):                           # 本核的路由切片
        for j in T.Parallel(128):
            idx = a * tpc + r * 128 + j
            if r * 128 + j < tpc and idx < numel:
                eid[0] = flat[idx]
                if eid[0] < E:                           # 显式上界（cumsum 有 E+1 元）
                    slot[0] = T.atomic_add(s_cur[eid[0]], T.int32(1), return_prev=True)
                    sorted_token_ids[slot[0]] = idx
```

**机制归因**：`atomic_add` 落在 GM 时是逐 lane 串行往返（约 1465 cyc/元素，
`CONST-950-atomic-tier-cost`），落 UB 时约 5 cyc；而逐 lane 的**数据相关 GM 写**只要 14 cyc。
所以「分配槽位」这个动作必须搬下 GM：每个核的排他起点由上游直方图的一次 exclusive 前缀给出
（`cumsum[e] + Σ_{r<core} partial[r,e]`，整数加法可结合 ⇒ 各核冗余重算仍逐位相同），核内再用 UB
游标分配子区间。槽位互异且都落在 expert 的补齐区间内，**expert 内顺序仍是自由项**（op 契约明示）。

**不要**为躲避散写而先写 UB 再整块 DMA 出：实测 15.54 us vs 直写 GM 6.70 us（UB 窗口 + 一次
228 KB 的 UB→GM 回写反而更贵）。

适用前提与边界：映射必须是「每元素一次落位」的计数散射（MoE permute-align、分组 GEMM 的索引
准备、bucket/radix 的一趟落桶）；需要上游已有（或可廉价算出）每核 × 每桶的计数矩阵。若消费方要求
元素顺序严格等于索引升序，则本形态只保证「多重集相同」，需另加一趟排序。
int32、1 维 grid、`T.SimtVF(threads=128)` 下实测；未测 Cube 侧、未测跨核 flag 的变体。
口径是 kernel-only `Task Duration`（逐 launch），不是端到端时延。
