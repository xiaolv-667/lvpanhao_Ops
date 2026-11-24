# NPUIR 专用约束：iteration-diagnosis

以下保留从公共文档抽出的原平台条款。阶段、门禁和重试仍按公共文档执行。

本文件不把 kernel 强行归到某个固定瓶颈大类，而是把 `msprof op` 数据、kernel 代码结构和历史迭代结果转换成：

- [hardware-context.md](../../../../skills/tilelang-op-optimize/references/hardware-context.md) 中的硬件上下文。
- 按 [profile-collection.md](../../../../skills/tilelang-op-optimize/references/profile-collection.md) 从 benchmark 的非 smoke workload 逐 kernel 采集到的 profiling 结果（`msprof op` 是唯一时延口径）。

同一轮采集未出现并发 msprof op 或输出目录覆盖

如果当前现象需要模式参考，读取 [bottleneck-patterns.md](../../../../skills/tilelang-op-optimize/references/bottleneck-patterns.md)，用本轮诊断上下文辅助生成候选优化点。

结构性优化分支通过正确性且方向有效后，不能只保留单个手写配置。必须围绕该结构暴露的关键参数做一轮 coarse autotune 或等价手动粗搜；例如 `T.serial` 多块迭代至少搜索 `block_size × num_cores`。autotune winner 不是最终结论，还要检查 top-k 配置和 winner 邻域，必要时手动/脚本精搜，再用最终 winner进入 `msprof op` 复测。

每个配置都记录 `msprof Task Duration` 和 imbalance。若甜点区内 `Task Duration` 差异小于噪声阈值，先用更大 `launch-count` 复测；复测后仍打平时用 imbalance、整除性和资源占用决胜，不要用单次 msprof 结果对平区内候选排序。

msprof op 性能采集

- 有一个或多个分支提升时，先按本轮主指标（`msprof op Task Duration(us)`）选择候选 winner；Task Duration 打平时，用 imbalance、UB/L0 余量和代码复杂度决胜。

- 小 kernel（Task Duration <20us）或逼近带宽/搬运地板的 workload，<5% 量级的候选差异须先经**交错 A/B/A/B 多 run 协议**裁决：≥3 次独立 `msprof op` run、候选与基线交替次序、合并中位数比较——同 kernel 跨 run 存在 ±3–5% 快/慢双态（BP_run_state_bimodality），单 run median-of-15 可能把双态膨胀误判为增益或回退（lerp_tensor 实证：单轮 -5.2% 复测收缩为 -3.1%）。协议在首次出现 <5% 差异时即前置使用，而非采纳后复核。工具化执行：`python3 .agents/tools/ab_test.py --a <baseline_kernel> --b <candidate_kernel> ...`（交错多 run + 合并中位差 + 配对方向一致性判定）。

- 每个有效实验分支都必须用 `msprof op` 重新采集目标 kernel；候选差异落入噪声阈值时先用更大 `launch-count` 复测，不要用单次 msprof 结果对平区内候选排序。
