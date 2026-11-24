# NPUIR 专用约束：skill-retrospective

以下保留从公共文档抽出的原平台条款。阶段、门禁和重试仍按公共文档执行。

- baseline 与 final 的 `msprof op` 数据

- **结构优化点（P，正向/反向）**：**慢→快的关键代码更改**——从实验分支/探针/参考 kernel 裁出的最小 diff（完整 before/after 对照，或效应只在完整 kernel 规模显现时的 delta 骨架）**+ 优化见解摘要**（LLM 提炼的一般化洞察：为何该更改慢→快，机制归因到硬件常数/编译器行为）——目标形态见 pattern-library [repro/README.md](../../../../skills/tilelang-op-optimize/references/pattern-library/repro/README.md)（PATT-twophase-restructure.py 为范例）；无法当场最小化的在 repro 字段标 `repro-missing`。

- 现有 [bottleneck-patterns.md](../../../../skills/tilelang-op-optimize/references/bottleneck-patterns.md) 不能很好覆盖。
