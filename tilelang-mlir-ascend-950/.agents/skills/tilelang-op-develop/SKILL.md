---
name: tilelang-op-develop
description: "根据冻结的 DESIGN.md 生成算子实现（{op}.py：kernel + golden），执行测试并返回四出口判定。触发：实现算子、生成 kernel、算子开发、跑精度。"
---

# TileLang 算子开发与验证

## 1. 目标

根据 Stage 1 冻结的 `DESIGN.md` 与 Stage 2 通过的 `REVIEW.md`，生成算子实现文件 `{op}.py`（含平台规定的 kernel 构建入口 + 内嵌 PyTorch golden + main 入口），执行测试，并返回四出口判定（[PRECISION_PASS] / [PRECISION_FAIL] / [DESIGN_ERROR] / RUNTIME_FAIL）供 conductor 路由。

> **环境前提**：按任务平台规则加载执行环境并确认依赖。kernel 编译和精度校验在目标设备上实际执行。

---

## 2. 输入

| 字段 | 说明 |
|------|------|
| `design_md_path` | 冻结的 `DESIGN.md`（含 L0 测试计划） |
| `review_md_path` | Stage 2 通过的 `REVIEW.md`（设计已检视通过） |
| `mode` | 调度模式，由 conductor 传入；枚举唯一出处 `_shared/standards/signal-registry.md` §2 |
| `attempt_index` | 当前 Stage 3 attempt 序号 |
| `last_failure_summary` | 重试时传入的失败信息（stderr 摘要 / 精度失败详情） |
| `design_revision_count` | 设计修订次数（用于回退后清零判断） |

---

## 3. 工作流程

### Phase 1：读取设计
1. Read `DESIGN.md` 全文，提取：算子名、I/O 规格、编程模式、API 映射、Tiling、内存层级、同步策略、L0 测试计划、精度标准。
2. Read `REVIEW.md`，确认检视已通过（如有 warn 项记录但不阻塞）。
3. 读取任务平台配置 `knowledge.patterns` 下的 INDEX.md，按检索命中精读布局模式、性能经验或编译与运行时陷阱。实现与调试前检查知识条目的版本戳，调试命中在 `skills_consulted` 中注明。

### Phase 2：生成 kernel
1. 按 DESIGN.md §3 API 映射、§6 循环结构及任务平台规则生成 kernel。
2. API 选型按任务平台规则及其编程指南。
3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。

### Phase 3：生成 golden
1. 按 DESIGN.md §8.1 生成 PyTorch CPU 参考实现 `golden_{op}(...)`。
2. Golden 必须在 CPU 上可独立运行（独立于设备运行时）。

### Phase 4：执行测试
1. 按任务平台规则运行 L0。
2. L0 通过后扩展 L1/L2/Boundary 并按任务平台规则跑全量。
3. 收集结果：max_diff、失败用例 shape、层级。

### Phase 5：四出口判定与返回

| 条件 | 返回标记 |
|------|----------|
| 输出与 golden 函数输出对比精度正常 | `[PRECISION_PASS]` |
| 输出与 golden 函数输出对比精度未过 | `[PRECISION_FAIL]` |
| 发现设计层错误（API 不可用、L0C 溢出、内存层级冲突等实现层无法修复） | `[DESIGN_ERROR]` + 原因 |
| 无标记且 exit code ≠ 0 | 运行失败（conductor 按 retry_impl 路由） |

### Phase 6：任务复盘（Retrospective，自进化钩子）⭐

返回 `[PRECISION_PASS]` 或 `[DESIGN_ERROR]` 前（`[PRECISION_FAIL]` / 运行失败**不写**——失败摘要已由 conductor 路由），向 `examples/{project}/{op}/RETROSPECTIVE.md` **追加**复盘章节（先 Read 既有内容，整文件写回，只追加不覆盖历史章节）：

- 章节模板与字段规范（canonical）：`tilelang-skill-evolution` skill 的 [references/retrospective-schema.md](../tilelang-skill-evolution/references/retrospective-schema.md)。两个标准表（Skill Flow Issues / Value Point Proposals）+ Transferable Lessons 小节。
- **PASS 复盘覆盖整个 attempt 链**（含中间失败中的发现）：
  - API 实际行为实证（D 类，须带三件套）：如某 API 的隐藏限制、对齐触发条件、静默 dtype 转换、合法形态边界；
  - 有效的调试手法（P 类）：如精度问题的定位顺序、IR dump 关键点；
  - `[DESIGN_ERROR]` 的根因与设计判断教训（P/R 类）：哪类设计判断错了、正确依据是什么。
- **探针转正（ED-C，D 类发现的 repro 责任前移）**：调试中发现 API 陷阱/行为实证时，探针不再只写 `/tmp`——有价值的最小复现**转正**为 `examples/{project}/{op}/repro/<TRAP-or-CONST-id>.py`（自包含 + 自描述头 + 断言，ED-B 规范见 平台 knowledge.patterns 中的 repro 规范），RETROSPECTIVE 的 Value Point Proposals 以该路径为 `repro` 字段（不再引用 `/tmp` session-local 路径）；evolver 终态蒸馏时机械拷入知识域 repro/ 并登记。本 skill 有 NPU 环境，转正前**实际跑一遍**确认断言成立。
- **Transferable Lessons**（迁移多函数任务必写，其他场景可 none）：写给本迁移任务**后续函数**实施者的跨函数教训，一条一行，自包含（不依赖本函数上下文可理解）。
- 质量红线：无则如实写 `none`；单任务偶然现象不得写成通用规则；不得硬凑。

---

## 4. `{op}.py` 结构规范

TileOPs harness 最终 kernel 按任务平台规则声明已验证的编程模式，供 Stage 5 集成使用。Stage 3 调度模式与 kernel 编程模式分别记录。

生成的文件必须包含以下组成部分（顺序）：
注意：*.py 中的注释只能使用英文

```python
# 1. Copyright (c) Huawei Technologies Co., Ltd. 2026.
# 2. platform and reference imports
# 2. golden_{op}(...) function
# 3. kernel construction from the selected platform template
# 4. precision comparing func：run_case()
# 5. main()
```

完整可运行模板由任务平台配置的 `templates.kernel` 提供。

### main 块结构

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    if args.level == "L0":
        run_L0()
    else:
        run_L0(); run_L1(); run_L2(); run_boundary()
```

每个 `run_LX()` 内部在 NPU 上创建张量、调用 kernel 执行、与 golden 对比精度。详见模板。

---

## 5. 失败处理

| 失败类型 | 识别 | 处理 |
|---------|------|------|
| 编译错误（实现层） | stderr 含 lowering/codegen 错误 | 返回运行失败 + stderr 摘要，conductor 走 retry_impl |
| API 不存在 | `AttributeError` / 设计用 API 无导出 | 返回 `[DESIGN_ERROR]` + 原因 |
| L0C/UB 溢出 | 编译期或运行期报容量超限 | 返回 `[DESIGN_ERROR]` + 原因 |
| 精度不达标 | `assert_close` 失败 | 返回 `[PRECISION_FAIL]` + max_diff/失败 shape |
| 内存层级越级 | stderr 提示 GM/L1/UB/L0 访问违规 | 返回 `[DESIGN_ERROR]` + 原因 |
| 环境问题 | `ImportError` 指向平台依赖未安装或环境未激活 | 返回运行失败，提示检查环境 |

---

## 6. 备份规则

`precision_fix` 模式每次修改 `{op}.py` 前，必须先备份：
```bash
cp {op}.py history_version/{op}_impl_s3_attempt{N}.py
```
（`{N}` = 当前 attempt_index）

---

## 7. 完成报告

返回结构化摘要：

```markdown
## Stage Result
- stage: 3
- mode: first_impl / retry_impl / precision_fix
- programming_mode: <最终通过验证的 kernel 模式，按平台规则记录>
- project: {project}
- operator: {op}
- output: examples/{project}/{op}/{op}.py
- verdict: [PRECISION_PASS] / [PRECISION_FAIL] / [DESIGN_ERROR] / RUNTIME_FAIL
- test_results:
  - L0: pass / fail (N cases)
  - L1: pass / fail (N cases)
  - L2: pass / warn (N cases, 不阻塞)
  - Boundary: pass / warn (N cases, 不阻塞)
- max_diff: <精度数值>
- design_error_summary: <仅 DESIGN_ERROR 时填>
- skills_consulted: <引用的 skill 路径>
- summary: <一句话>
- issues: <若无则 none>
```
