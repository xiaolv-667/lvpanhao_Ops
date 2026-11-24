# TileLang npuir Agent Guide

This repository uses AGENTS skills for TileLang NPUIR development.

## Scope

For conductor tasks, resolve hardware and toolchain with `python3 .agents/tools/statectl.py platform --hardware <model> --toolchain <toolchain>` or use `--task-dir <task>` to read the saved selection and follow its rules and skill references. The implementation guidance below applies to the NPUIR backend.

The skills in .agents/skills are designed for target="npuir" workflows.
They prioritize v-prefix vector APIs such as vadd, vmul, vexp, vcast, vbrc.
Legacy npuir_xxx APIs remain valid as compatibility aliases.

## API Convention (Mandatory)

- Prefer v-prefix APIs in new examples and generated code.
- Keep compatibility with npuir_xxx when reading existing code.
- If both forms are available, output should default to v-prefix.

Examples:
- Prefer: T.vmul(A, B, C)
- Compatible: T.npuir_mul(A, B, C)

## Skill Index

1. tilelang-npuir-overview
Purpose: architecture and compile pipeline for npuir branch.

2. tilelang-vector-skill
Purpose: vector operator generation with v-prefix API style.

3. tilelang-cube-skill
Purpose: cube operator generation with load_nd2nz and store_fixpipe.

4. tilelang-mixcv-skill
Purpose: mixed Cube+Vector kernels such as flash attention pipelines.

5. tilelang-mlir-skill
Purpose: TileLangIR and MLIR pass workflow and debugging.

6. tilelang-debug-helper
Purpose: GDB + IR dump + pass-level debug workflow for npuir.

7. tilelang-error-fixer
Purpose: diagnosis and repair workflow for compile/runtime/pass failures.

8. tilelang-review-skill
Purpose: risk-first code review and CI-aligned format checks.

9. tilelang-github-operations
Purpose: npuir branch commit/rebase/PR/issue workflow.

10. tilelang-skill-evolution
Purpose: self-evolution distillation for Tilelang-Op-Conductor. Distills value points (D measured data / P patterns / R process rules / C case index) from task artifacts at task end, merges by tier governance into pattern-library or enqueues to .agents/evolution/queue.md. Invoked by tilelang-skill-evolver only.

## Trigger Guidance

Use the matching skill whenever the user asks for:
- npuir kernel writing, performance tuning, vector math, cube gemm, mixed kernels
- pass debugging, IR dump, MLIR transform troubleshooting
- compile/runtime error analysis on npuir branch
- code review, lint/format checks, PR readiness
- commit/push/rebase/upstream sync, PR or issue workflow
- reviewing or merging evolution proposals (.agents/evolution/queue.md), value-point distillation, self-evolution of conductor skills

Developer-mode MixCV trigger rule:
- If one kernel contains Cube-side T.gemm and Vector-side at least one v-prefix op (such as T.vadd/T.vmul/T.vexp/T.vcast/T.vbrc), treat it as MixCV and use tilelang-mixcv-skill.

## Operator Implementation Baseline (Mandatory)

For operator-writing tasks, always start from existing examples and tests:

- First consult examples/ and testing/npuir/ for the closest existing pattern.
- Prefer modifying an existing operator case instead of generating a brand-new kernel from scratch.
- If no close template exists, explicitly state that and then build the minimal new kernel.

## Compiler Capability Gap Rule (Mandatory)

When any task stage (design algorithm research / API mapping, develop, optimize, review) concludes that current npuir capability is insufficient to implement the chosen algorithm (cannot implement / forced algorithm swap / performance ceiling):

1. Identify the specific missing capability and its compiler layer (Frontend API / TileLangIR pass / BishengIR / tladapter / runtime).
2. Record it in `.agents/evolution/capability-gaps.md`: dedupe first, then append a new entry or increment `occurrences` of the existing one, following the schema in that file.
3. Evidence rule is identical to negative claims in Docs Auto Routing: cite concrete doc path and limiting clause, or explicitly mark as undocumented assumption with estimation basis.
4. Escalation (compiler-level self-evolution): a gap independently identified in 2 or more different tasks becomes `recurring` and must produce a capability-fill proposal (target layer, proposed change, quantified benefit, affected operators) surfaced to the user at task end. Filing an external issue (with `[npuir]` prefix) requires user approval.
5. Gap conclusions bind to toolchain version stamps; re-verify before reusing them after toolchain upgrades (mechanized: `.agents/tools/kb_stale_check.py` detects stale entries; `.agents/tools/repro_runner.py --filter stale` re-verifies repro-backed ones).

## Pre-PR Formatting Rule (Mandatory)

Before creating or updating a PR, run format validation for changed files from repository root:

- bash format.sh --files changed_files

Notes:
- This is a required self-check for clean code and style consistency.
- The changed_files placeholder represents the file list modified in the current branch.

## Docs Auto Routing Rules (Mandatory)

When any skill answers technical questions, it must route references by docs directory first.

### Routing Priority

1. docs/Tilelang.language/ (API semantics and signatures)
2. docs/Tilelang算子调试指南.md (debug and issue localization)
3. docs/developer/ (runtime and environment variables)
4. docs/开发指南.md and docs/快速入门.md (workflow and onboarding)
5. docs/Tilelang-Ascend贡献指南.md (PR, issue, contribution process)

### Keyword to Docs Mapping

- Vector ops (vadd/vmul/vexp/vcast/vbrc/reduce/sigmoid/rmsnorm):
    docs/Tilelang.language/数学操作/
    docs/Tilelang.language/数据类型转换操作/
    docs/Tilelang.language/shape操作/
    docs/Tilelang.language/规约操作/

- Cube ops (gemm/load_nd2nz/store_fixpipe/L1/L0C/NZ):
    docs/Tilelang.language/线性代数操作/
    docs/Tilelang.language/内存操作/

- Layout/creation ops (transpose/arange/infinity/broadcast):
    docs/Tilelang.language/创建操作/
    docs/Tilelang.language/shape操作/

- Comparison and selection ops (vcmp/T.min/T.max/if_then_else/select):
    docs/Tilelang.language/比较操作/
    docs/Tilelang.language/条件操作/

- Pipeline and sync (sync_block_set/wait/pipe_barrier/set_flag/wait_flag):
    docs/Tilelang.language/同步管道操作/

- Debug, compile failure, runtime failure, precision issue:
    docs/Tilelang算子调试指南.md
    docs/Tilelang.language/调试操作/

- MLIR, pass, tilelangir, bishengir-compile:
    docs/Tilelang算子调试指南.md
    docs/developer/EnvironmentVariables.md

- Runtime target, mode switch, env setup:
    docs/developer/npu runtime.md
    docs/developer/EnvironmentVariables.md
    docs/安装指南.md

- PR, rebase, commit, issue, CI workflow:
    docs/Tilelang-Ascend贡献指南.md

### Conflict Resolution

- If multiple mappings match, select by priority and keep at most 3 primary doc references.
- Always include at least 1 concrete API doc under docs/Tilelang.language/ when the question is API-related.
- If API docs and examples differ, API docs are source of truth and examples are secondary.
- If a question involves an API not covered by any mapping above, enumerate `ls docs/Tilelang.language/` and check ALL subdirectories (including 创建操作/索引与元素操作/条件操作/排序操作/逻辑操作/原子操作/编译器提示操作) before concluding the API is unsupported or undocumented.
- Negative claims (API does not exist / not supported / too costly / no precedent) must cite the concrete doc path and the limiting clause read from it, or be explicitly marked as "undocumented assumption" with the estimation basis. Never reject a design candidate from memory or GPU prior before the retrieval.

## 设计与 API 资料

- 设计约束：[ascend-constraints.md](references/ascend-constraints.md)。
- 编程模式和 API 映射：[decision-tree.md](references/decision-tree.md)。其中容量数值是历史示例，实际预算读取任务选定型号。
- 本工具链信息源：[info-sources.md](references/info-sources.md)。
- API 命名空间为 tilelang.language。Developer 使用自动同步，Expert 核对显式层级与手动同步点；v-prefix 优先规则仅在本工具链内适用。
- GEMM 的分形与对齐按上述约束核对。内存层级和容量从任务平台配置读取。

## NPUIR 设计伪代码与签名示例

如果迁移前的`TIR原语函数`签名中`T.Tensor`通过关键字参数`dtype=`传递数据类型，改为将数据类型直接作为第二个位置参数传入，不再使用`dtype=`关键字。
这是为了避免产生类似`error: name 'in_dtype' is not defined`报错信息但编译执行正常的`false alarm`。
举例来说, 原来写法：
```python
@T.prim_func
def tilelang_kernel(
    input: T.Tensor((M, N), dtype=in_dtype),
    result: T.Tensor((M, N), dtype=out_dtype),
)
```
修改后：
```
@T.prim_func
def tilelang_kernel(
    input: T.Tensor((M, N), in_dtype),
    result: T.Tensor((M, N), out_dtype),
)
```
注意，这个修改并不违反"迁移执行规则"中的第3条。

```python
# 基于 TileLang API 的计算流程伪代码
with T.Kernel(block_num, is_npu=True) as (cid, _):
    # 1. 分配 buffer
    {buffer 分配代码}

    # 2. 数据搬入
    {T.copy 搬入代码}

    # 3. 计算
    {核心计算代码}

    # 4. 数据搬出
    {T.copy 搬出代码}
```


## NPUIR 搬运路径示例

示例（纯 Vector）:
GM[A] --T.copy--> UB[a_ub] --计算--> UB[c_ub] --T.copy--> GM[C]

示例（Cube + Vector）:
GM[A] --T.copy--> L1[a_l1] --T.copy--> L0A[a_l0a]
GM[B] --T.copy--> L1[b_l1] --T.copy--> L0B[b_l0b]
L0A + L0B --T.gemm--> L0C[c_l0c] --T.copy--> UB[c_ub]
UB[c_ub] --后处理--> UB[c_ub] --T.copy--> GM[C]

## NPUIR Autotune 资料

本文件不复制 TileLang Ascend autotune 的完整 API 和示例，避免 `tilelang-mlir-ascend` 更新后两边内容不同步。具体用法、参数、示例和实现细节，以 `tilelang-mlir-ascend` 源库为准。

## 先读这些源文件

源码根目录：

```text
tilelang-mlir-ascend/
```

优先读取官方/源码侧文档：

1. `docs/Autotune使用指南.md`
2. `docs/developer/EnvironmentVariables.md`

需要确认实现细节时读取：

1. `tilelang/autotuner/tuner.py`
2. `tilelang/autotuner/param.py`
3. `tilelang/profiler/bench.py`

需要参考可运行示例时读取：

1. `testing/autotune/example_gemm_autotune.py`
2. `testing/autotune/example_gemm_carver.py`
3. `testing/autotune/example_elementwise_add_autotune.py`
4. `testing/autotune/example_elementwise_add_carver.py`
5. `testing/autotune/gemv_autotune.py`
6. `testing/autotune/gemv_carver.py`
7. `testing/autotune/reduce_min_autotune.py`
8. `testing/autotune/reduce_min_carver.py`
9. `testing/autotune/flash_attn_npuir_autotune.py`

需要参考结构性优化模式时，优先搜索并阅读：

1. `examples/elementwise/example_elementwise_exp2.py`（`T.serial` 多块迭代模式）
2. `examples/elementwise/vec_add_2d_multi_buffer.py`（multi-buffer / 多缓冲模式）
3. `examples/elementwise/` 下其它 activation、unary、binary 示例
4. 历史优化目录或同类算子的 `perf_opt/opt_log.md`

如果历史目录存在 `Optimize.md`，只能把它当作快速摘要；完整过程仍以 `perf_opt/opt_log.md` 为准。
