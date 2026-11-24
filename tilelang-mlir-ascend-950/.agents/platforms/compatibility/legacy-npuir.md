# 现有 NPUIR 平台规则

API 规范、文档路由和专用阶段参考由 [NPUIR 编程与资料规则](../toolchains/npuir/development.md) 提供。通用技能中的 API、模式和约束检查使用此入口。

## 环境与设计

开发前加载现有 CANN 环境并确认 TileLang 和 torch_npu 可导入。用 `python3 .agents/tools/statectl.py platform --task-dir <任务目录> --probe-cores` 实查并保存结果。该入口调用 `NPUUtils.get().get_aicore_num()`，设计文档继续记录该查询依据。Cube 和混合算子使用返回值，纯 Vector 使用返回值的两倍。记录查询代码及结果。逻辑核数超过物理核数时分波串行调度，按共享分核标准评估整数倍对齐与核内串行。尾轴按 32B 对齐，分形宽度至少 16，并核对向量宽度。容量与多缓冲预算读取平台配置，910B2C 成本常数与调优经验继续使用原知识库。

## 设计成本估算

**估算下界 = max(三项)**，全部按 dispatch 代表 workload 计算：

1. **流量项** = GM 读写总字节 ÷ 有效带宽（MTE2 曲线按总流量档位取值，`CONST-mte2-degradation`；混合读写流量禁用峰值参考值——3:1 R/W 地板见该条目）；
2. **发射项** = 向量 op 数 × ~0.5µs/op（`CONST-vector-launch-overhead`；Cube 段另按 L1 端口带宽折算操作数流地板，`CONST-L1-port-bw`）；
3. **容量项** = 容量约束决定的 tile 循环数 × 每轮跨引擎往返代价（`CONST-store-fixpipe-gm-only`：S/P 类中间矩阵无法直写 L1/UB，跨引擎传输强制 GM 往返）。

## 阶段技能

设计、检视、开发和调优分别加载 `stage_skills.design`、`stage_skills.review`、`stage_skills.develop` 和 `stage_skills.optimize`。这些技能及其引用提供现有的 NPUIR 编程规范、技术限制、模板、案例和调优规则。

## 开发与测量

生成 kernel 时使用 `target="npuir"`，沿用 Developer／Expert 编程约定和 v-prefix API。精度执行 `python {kernel}.py --level L0` 或 `python {kernel}.py --level all`。主 Agent 对已通过的实现复跑全量。TileOPs harness 的最终 kernel 在模块顶层声明字面量 `ASCEND_MODE`，取 `Developer` 或 `Expert`，独立测试时的 `TILELANG_ASCEND_MODE` 与之保持一致，集成时同步 wrapper 的 `Kernel.ascend_mode`。诊断沿用配置 `diagnostics` 指定的 NPUIR 调试技能。性能使用 msprof op 的 Task Duration(us)，继续以 `msprof_raw_path` 保存原始采集证据，主指标标识为 `msprof_task_duration`。正式调优按 kernel-only 时延判定，保持全局串行采集与现有实验协议。接近测量噪声时增加 `launch-count` 后比较多次独立结果。

## 工程迁移

源工程仍以 TileOPs manifest、测试和 benchmark 目录判定。脚手架、wrapper 和编程模式检查沿用现有契约。集成时在 `examples/TileOPs/` 下执行 `python <integration.script> --meta <meta_path> --task-dir <聚合任务绝对目录>`，使用最终 Stage 3 kernel 的字面量 `ASCEND_MODE`，已有产物也可读取字面量 `TILELANG_ASCEND_MODE`。每次 report 前同步 `Kernel.ascend_mode`，重复集成也需核对。缺声明或声明冲突时修正产物后再试。同一 Kernel 类混用 Developer/Expert 时，报告 `[INTEGRATE_FAIL]`，交回设计逐调用模式作用域。NPUIR vsel shape mismatch 可尝试去掉 padding 用 raw N，或选整除 N 的 tile 参数。从 `examples/TileOPs/` 运行 `TILEOPS_TASK_DIR=<聚合任务绝对目录> python -m tileops.reporting.cli run --op {op_name} --prof-mode msprof`，使用 manifest 的 PascalCase 键。写入会话耗时 Markdown 后，从同一工作目录运行 `python -m tileops.reporting.cli render {run_json} --with-session-timing` 更新本次报告。已有算子调优后按原流程切换 wrapper，报告正确性失败时切回基线。

NPUIR 阶段的具体 API、构建和迁移提示保留在 `.agents/platforms/toolchains/npuir/stage-examples.md`；原采集命令与回退参数见 `.agents/platforms/toolchains/npuir/references/profile-collection.md`。工程适配中的 NPUIR 专用约束见同目录 `tileops-adaptation.md`。这些专用内容不再作为其他平台的共用要求。
