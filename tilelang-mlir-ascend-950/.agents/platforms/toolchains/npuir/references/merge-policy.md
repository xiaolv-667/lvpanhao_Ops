# NPUIR 专用约束：merge-policy

以下保留从公共文档抽出的原平台条款。阶段、门禁和重试仍按公共文档执行。

> 2026-09-10 修订（conductor 改进报告 ED-A/C、E-3/E-4、K-2 落地）：证据三件套语义按 ED-A 重定义（provenance / 版本戳 / repro 分离，见 distillation-rules.md §5）；D 类合入门槛与 repro 责任前移见 §1/§2；预算从行数改为**字节预算**；apply 预验证见 §8。

| `.agents/skills/tilelang-op-optimize/references/pattern-library/` 主题文件（layout/elementwise/attention/traps-compiler/traps-runtime/constants/cases.md） | Tier 0：add/update/negate/deprecate + 预算触发的 consolidate（条目 front-matter 按 INDEX.md §5 schema） |
| `.agents/skills/tilelang-op-optimize/references/pattern-library/repro/` | **repro 的转正与校验（ED-C，机械动作）**：把 Stage 3 任务内产出的 `examples/{op}/repro/*.py` 机械拷入、核验头部规范（条目 ID + 版本戳 + 断言说明）与 `py_compile`、在条目 front-matter 登记 `repro:` 路径——**不编写、不裁剪代码**（编写与首次运行发生在有环境的任务内 Subagent） |
| `.agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md` | Tier 1：P 类合入（计数达 2/2） + consolidate |
| `.agents/skills/tilelang-op-design/references/algorithm-candidates.md` | Tier 1：P 类合入（算法候选库——"算法族 X 的候选 Y 实测收益 Z"类发现，条目含 `known_impl`/`kb_links` 指针） |
