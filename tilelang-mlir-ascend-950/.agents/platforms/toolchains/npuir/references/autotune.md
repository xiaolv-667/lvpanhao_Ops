# NPUIR 专用约束：autotune

以下保留从公共文档抽出的原平台条款。阶段、门禁和重试仍按公共文档执行。

- coarse search 结果只用于筛候选，最终 winner 必须单独用 `msprof op` 复测。

- 若 autotune latency 与 `msprof Task Duration` 排序冲突，以独立复测的 `msprof op` 决胜。

- 对 winner 单独跑 `msprof op`。
- 最终性能报告使用单独复测的 `msprof op`，不直接使用 autotune latency 作为最终结果。

如果所有 config 的 `msprof Task Duration` 接近，先用更大 `launch-count` 复测。复测后仍打平时，使用 imbalance、整除性和资源余量决胜；若仍无法区分，回到 Phase 2 重新分析当前现象，而不是继续扩大搜索空间。
