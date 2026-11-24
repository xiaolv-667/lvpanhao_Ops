# NPUIR 工具链规则

API 规范、文档路由和专用阶段参考由 [NPUIR 编程与资料规则](development.md) 提供。通用技能中的 API、模式和约束检查使用此入口。

## 环境与设计

开发前加载 CANN 环境并确认 TileLang 和 torch_npu 可导入。新任务用 `statectl platform --task-dir <任务目录> --probe` 核对设备型号、运行时报告的核数及工具链版本。设计 §5 记录 `hardware_id`、核数与来源。离线设计可用型号配置中已有的核数证据，真机执行前重新查询。实际核数按运行时分别返回的 Cube 和 Vector 数记录。

设计预算和硬件对齐约束从选定型号读取。片上容量缺失时写明未核验，保留可复算的缓冲用量和核数计算。分形和 API 使用约束按当前工具链的阶段技能核对。逻辑核数超过物理核数时按共享分核标准评估整数倍对齐与核内串行。NPUIR 使用一维 Kernel，按 Cube/Vector/DMA 与 L1/UB/L0 分级重设计；分形、对齐和容量须满足当前型号。核内串行使用 `T.serial`，任务分配可写为 `num_local_tasks = T.ceildiv(num_logical_kernels - kernel_id, num_physical_kernels)`，循环边界必须为静态值；K 维分块循环可用 `T.serial(K // block_K)`。

设计期用同一代表 workload 的流量、发射和容量三项估算，取最大值作为下界。流量项按读写字节与匹配的有效带宽计算，发射项按操作数与匹配的发射代价计算，容量项按容量约束引入的循环次数与每轮额外搬运代价计算。新任务只使用平台上下文 `cost_measurements` 中与目标型号、工具链版本及本次 workload 条件匹配的记录。缺少记录时列出待核验项，假设值须标注来源与适用边界。

## 阶段技能

设计、检视、开发和调优分别加载 `stage_skills.design`、`stage_skills.review`、`stage_skills.develop` 和 `stage_skills.optimize`。这些技能及其引用提供现有的 NPUIR 编程规范、技术限制、模板、案例和调优规则。

## 开发与测量

生成 kernel 时使用 `target="npuir"`，沿用 Developer／Expert 编程约定和 v-prefix API。精度执行 `python {kernel}.py --level L0` 或 `python {kernel}.py --level all`。主 Agent 对已通过的实现复跑全量。TileOPs harness 的最终 kernel 在模块顶层声明字面量 `ASCEND_MODE`，取 `Developer` 或 `Expert`，独立测试时的 `TILELANG_ASCEND_MODE` 与之保持一致，集成时同步 wrapper 的 `Kernel.ascend_mode`。诊断沿用配置 `diagnostics` 指定的 NPUIR 调试技能。性能使用 msprof op 的 Task Duration(us)，继续以 `msprof_raw_path` 保存原始采集证据，主指标标识为 `msprof_task_duration`。新任务写性能记录时附上同一份平台选择和真机证据。正式调优按 kernel-only 时延判定，保持全局串行采集与现有实验协议。接近测量噪声时增加 `launch-count` 后比较多次独立结果。

## 工程迁移

源工程仍以 TileOPs manifest、测试和 benchmark 目录判定。脚手架、wrapper 和编程模式检查沿用现有契约。集成时在 `examples/TileOPs/` 下执行 `python <integration.script> --meta <meta_path> --task-dir <聚合任务绝对目录>`，使用最终 Stage 3 kernel 的字面量 `ASCEND_MODE`，已有产物也可读取字面量 `TILELANG_ASCEND_MODE`。每次 report 前同步 `Kernel.ascend_mode`，重复集成也需核对。缺声明或声明冲突时修正产物后再试。同一 Kernel 类混用 Developer/Expert 时，报告 `[INTEGRATE_FAIL]`，交回设计逐调用模式作用域。NPUIR vsel shape mismatch 可尝试去掉 padding 用 raw N，或选整除 N 的 tile 参数。从 `examples/TileOPs/` 运行 `TILEOPS_TASK_DIR=<聚合任务绝对目录> python -m tileops.reporting.cli run --op {op_name} --prof-mode msprof`，使用 manifest 的 PascalCase 键。写入会话耗时 Markdown 后，从同一工作目录运行 `python -m tileops.reporting.cli render {run_json} --with-session-timing` 更新本次报告。已有算子调优后按原流程切换 wrapper，报告正确性失败时切回基线。

NPUIR 阶段的具体 API、构建和迁移提示保留在 `.agents/platforms/toolchains/npuir/stage-examples.md`；原采集命令与回退参数见 `.agents/platforms/toolchains/npuir/references/profile-collection.md`。工程适配中的 NPUIR 专用约束见同目录 `tileops-adaptation.md`。这些专用内容不再作为其他平台的共用要求。

迁移能力映射、并行/搬运重设计示例和存储访问错误识别见 [NPUIR 迁移实现映射](references/migration-analysis.md)。
