# 平台知识条目规范

各平台知识入口采用同一条目格式；目录与具体事实由任务平台提供。原 NPUIR INDEX.md §5 的规范继续适用于旧条目。

- 必填：id、kind、status、origin_task、toolchain、repro。kind/status 枚举与 kb_lint.py 一致。
- repro 为知识域内自包含脚本，或如实标记 repro-missing；缺少实测证据的参考内容保留在知识演进队列（pending），验证后再按条目状态枚举入库。
- 新条目记录 hardware_id/toolchain_id（与事实有关时），以及实际核验工具链的 toolchain_stamp 原始指纹。toolchain 保留便于阅读的版本与来源说明。
- 无可比较版本信息时，标记版本戳缺失并保持待核验；不能把 unknown 当成 fresh。
- 历史 device/toolchain 文本保留，由所选工具链的兼容比较规则解释，不改写成伪造的新指纹。
- 复现路径相对于本平台 knowledge.patterns；来源与出处允许指向历史任务，但复现脚本不能依赖任务目录存活。
