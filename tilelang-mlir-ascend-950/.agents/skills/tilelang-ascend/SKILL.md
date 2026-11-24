---
name: tilelang-ascend
description: 面向 Ascend950DT_9582 的 TileLang ascend 设计、检视、开发、调优、诊断和 TileOPs 集成。
---

读取 [平台规则](../../platforms/toolchains/ascend/workflow.md) 和平台配置给出的容量与证据。沿用调用方定义的阶段产物、章节和退出信号。

各阶段完整执行下列现有技能的流程、引用、工件模板和退出条件。其中的 NPUIR API、编程模式、硬件知识和性能工具，按本平台配置与规则替换。

- Stage 0：使用项目内原有 add-npu-op 技能及七文件脚手架，保留原有迁移输入、目录和元数据；kernel 按本平台 ascend API 实现。
- Stage 1：先完整读取并执行[原有设计技能](../tilelang-op-design/SKILL.md)。从算子语义、shape/dtype 和参考实现推导算法、搬运与分核。比较支持的编程模式，按任务容量核算 UB/L1/L0C 与多版本缓冲。在 DESIGN.md 记录物理核数查询和逻辑到物理核的映射。
- Stage 2：先完整读取并执行[原有检视技能](../tilelang-design-review/SKILL.md)。按调用方规定的维度检视 DESIGN.md。优先检查语义、资源预算、尾块、写入范围及同步，返回 REVIEW.md 和检视结论。
- Stage 3：先完整读取并执行[原有开发技能](../tilelang-op-develop/SKILL.md)。从平台 kernel 模板实现算子和独立 PyTorch golden，执行 L0、代表 shape、多 dtype 和边界用例。失败时保留日志和最小复现，按环境、编译、运行或精度分类。
- Stage 4：先完整读取并执行[原有调优技能](../tilelang-op-optimize/SKILL.md)。按平台规则枚举 workload，先验精度，再通过原有 A/B 与实验 runner 对比基线与候选，采用 Task Duration(us) 和 msprof_raw_path。按 Stage 4 契约保存 workload inventory、原始采集、性能记录和最终版本，使用调用方给出的实验预算。
- Stage 5：使用项目内原有集成脚本和单算子报告入口，传递保存的平台选择；沿用原有 wrapper 切换、run.json 和 partial 验收条件。
