# NPUIR 阶段专用写法

以下内容保留原 NPUIR 提示及示例，仅供 NPUIR 任务使用。

## Output Prompt

After all 7 files are created and **Tier 1 structural verification passes**,
this skill emits **one migration prompt per extracted TileLang kernel
function** for the **TileLang 算子端到端开发编排 Agent**
(`tilelang-op-conductor`). If the op has N kernel functions extracted in
S3 Part A, the skill emits N prompts — one per function.

### Purpose

Each prompt hands off the NPU TileLang kernel *re-implementation* work
for a **single** extracted kernel function (adapting it to
`the task-selected target`) to the conductor agent, which then drives design →
develop → review → optimize stages end-to-end. Each prompt conveys:

- Which op is being migrated and its manifest identity.
- The **single** TileLang kernel function name to reimplement.
- Where the GPU reference implementations were extracted to (the
  single output file from the extraction script, shared across all
  per-kernel prompts for this op).
- Where the op spec (workloads), correctness tests, and benchmarks live
  in the NPU project.
- The programming mode to use (`developer`).

### When to emit

Emit the prompts **once**, immediately after Tier 1 (S7) passes. Emit
**one prompt per extracted kernel function** — if the script extracted N
functions (reported in its stdout), emit N prompts. Do **not** emit if
Tier 1 fails — fix structural issues first. The prompts are not needed
for Tier 2 (runtime verification), which is the conductor's
responsibility after it rewrites the kernel functions for the selected backend.

**Mandatory constraint**: Regardless of whether the TileLang kernel
functions in S3 Part A have already been reimplemented for `the task-selected target`
(by a prior conductor run) or are still GPU implementations, **every
execution of this skill must emit the full set of per-kernel prompts for
the op**. Implementation status never suppresses the prompts — the skill
always outputs them as the final messages so the conductor can (re-)run
the design → develop → review → optimize pipeline on demand for each
kernel independently.

### Template

Fill the placeholders below and output **one text block per extracted
kernel function** as the final messages of the skill invocation (one
block per function, in the order the script reported them). The prompt is
written in Chinese to match the conductor agent's working language.

```
迁移 {op} 算子 {op_name}。 其 TileLang 实现函数 {kernel_function_name} 在 {extracted_file}。
{kernel_function_name} 是对外接口，不要改变接口的参数(除非该参数跟后端实现相关，在迁移时需要适配)。
{op_name} 的规格见 examples/TileOPs/tileops/manifest/{family}.yaml中 {op_name} 的 workloads 部分。
精度用例见 examples/TileOPs/tests/ops/test_{test_slug}.py，
性能用例见 examples/TileOPs/benchmarks/ops/bench_{bench_slug}.py
用 developer 模式实现。
```

### Placeholder resolution

Each placeholder is resolved from values gathered during Phase 0 and S1-S7.
Do not ask the caller for additional input — all values are already known by
the end of S7. The `{kernel_function_name}` placeholder is **per-kernel** —
it takes a different value in each emitted prompt block (one per extracted
function). All other placeholders, including `{extracted_file}`, are
shared across all prompts for the same op.

| Placeholder                | Scope                | Source                                                                                                                                                                                                                                                                                     | Example                                                   |
| -------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- |
| `{op}`                   | shared               | `op_slug` derived in S1 (snake_case of the op)                                                                                                                                                                                                                                           | `logsumexp`                                             |
| `{op_name}`              | shared               | PascalCase manifest key from Inputs                                                                                                                                                                                                                                                        | `LogSumExpFwdOp`                                        |
| `{kernel_function_name}` | **per-kernel** | A single function name from the extraction script's reported function list in S3 Part A                                                                                                                                                                                                    | `_logsumexp_kernel_single`                              |
| `{extracted_file}`       | shared               | The output file path printed by the extraction script in S3 Part A (the`[ok] wrote <path>` line). When `--out` is a directory, the script auto-generates the filename inside it; when `--out` is a file path, it is used directly. All extracted functions live in this single file. | `tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py` |
| `{family}`               | shared               | Op family directory from Inputs                                                                                                                                                                                                                                                            | `reduction`                                             |
| `{test_slug}`            | shared               | Test file slug extracted from manifest`source.test` (strip `tests/ops/test_` prefix and `.py` suffix)                                                                                                                                                                                | `softmax`                                               |
| `{bench_slug}`           | shared               | Bench file slug extracted from manifest`source.bench` (strip `benchmarks/ops/bench_` prefix and `.py` suffix)                                                                                                                                                                        | `softmax`                                               |

**Notes on resolution**:

- `{kernel_function_name}` is a **single** function name — one of the
  names reported by the extraction script. Each emitted prompt block
  uses exactly one function name. The full set of prompts covers
  **all** variants (e.g. `_{op_slug}_kernel_single`,
  `_{op_slug}_kernel_tiled`) and the dispatcher (e.g. `_{op_slug}_kernel`)
  if it is among the extracted names — each gets its own prompt.
- `{extracted_file}` is the **shared** output file path printed by the
  extraction script (the `[ok] wrote <path>` line). When `--out` is a
  directory (e.g. `tileops/kernels/{family}/{op_slug}`), the script
  auto-generates the filename inside it; when `--out` is a file path, it
  is used directly. This is the single file where all GPU kernel
  functions now live in the NPU project (extracted from the GPU repo by
  the script). Every per-kernel prompt for the same op references this
  same file.
- `{test_slug}` and `{bench_slug}` come from the **NPU** manifest's
  `source.test` / `source.bench` fields (written in S1). They may differ
  from `{op}` when the family groups multiple ops in one test/bench file
  (e.g. `logsumexp` op → `softmax` test file). Always use the manifest
  value, not the op slug.

### Reference example

For `LogSumExpFwdOp` (family `reduction`, GPU repo root
`/home/tilelang/zuochuanuong/TileOPs-fork`), after running:

```bash
python scripts/extract_tl_kernel.py \
  --op-name LogSumExpFwdOp \
  --gpu-repo-root /home/tilelang/zuochuanuong/TileOPs-fork \
  --out tileops/kernels/reduction/logsumexp
```

the script auto-generates the filename and prints
`[ok] wrote tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py`.
The script extracts `_logsumexp_kernel_single` and
`_logsumexp_kernel_tiled`. Two prompts are emitted — one per kernel
function, both referencing the same extracted file:

**Prompt 1** (for `_logsumexp_kernel_single`):

```
迁移 logsumexp 算子 LogSumExpFwdOp。 其 TileLang 实现函数 _logsumexp_kernel_single 在 tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py。
_logsumexp_kernel_single 是对外接口，不要改变接口的参数(除非该参数跟后端实现相关，在迁移时需要适配)。
LogSumExpFwdOp 的规格见 examples/TileOPs/tileops/manifest/reduction.yaml中 LogSumExpFwdOp 的 workloads 部分。
精度用例见 examples/TileOPs/tests/ops/test_softmax.py，
性能用例见 examples/TileOPs/benchmarks/ops/bench_softmax.py
用 developer 模式实现。
```

**Prompt 2** (for `_logsumexp_kernel_tiled`):

```
迁移 logsumexp 算子 LogSumExpFwdOp。 其 TileLang 实现函数 _logsumexp_kernel_tiled 在 tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py。
_logsumexp_kernel_tiled 是对外接口，不要改变接口的参数(除非该参数跟后端实现相关，在迁移时需要适配)。
LogSumExpFwdOp 的规格见 examples/TileOPs/tileops/manifest/reduction.yaml中 LogSumExpFwdOp 的 workloads 部分。
精度用例见 examples/TileOPs/tests/ops/test_softmax.py，
性能用例见 examples/TileOPs/benchmarks/ops/bench_softmax.py
用 developer 模式实现。
```

### Verification of the prompts

Before emitting, sanity-check **each** prompt against the NPU files just
created:

1. `{kernel_function_name}` exists as a `def` statement in the
   extracted file (`{extracted_file}`).
2. `{extracted_file}` exists and is the extracted kernel file containing
   all extracted functions.
3. The number of emitted prompts equals the number of TileLang kernel
   functions reported by the extraction script.
4. `examples/TileOPs/tileops/manifest/{family}.yaml` contains a
   `{op_name}:` key with a `workloads:` section.
5. `examples/TileOPs/tests/ops/test_{test_slug}.py` exists.
6. `examples/TileOPs/benchmarks/ops/bench_{bench_slug}.py` exists.

If any check fails, correct the placeholder value (or the underlying file)
before emitting — the conductor agent will act on each prompt verbatim.


## 开发与修复示例

原开发规范的构建段为 `@tilelang.jit` kernel。NPUIR 的实际实现及完整模板仍由 profile 的 templates.kernel 提供。

原设计模板的向量化示例为 `T.vexp（有 examples/docs 佐证）`；同步示例为 `T.barrier_all()`。使用前仍须按 NPUIR 的实际版本查证，历史示例不构成 API 已通过验证的声明。

原集成修复建议：kernel 内加 fp32 中间量（vcast 上抛 → 计算 → vcast 回写）。本建议的具体 API 仅适用于 NPUIR，仍按其 dtype 能力核验。

## 分核查询与编译限制

原分核依据见仓库 `docs/开发指南.md` §3.3：`NPUUtils.get().get_aicore_num()`；历史 Vector 换算取两倍。新任务按本工具链 workflow 记录实际 Cube/Vector 核数。原核内串行静态边界限制与相应示例也仅作为本工具链的编译约束核验。

## 原迁移入口约定

1. **严格**使用 `@tilelang.jit()` 所装饰的函数名作为算子名，不擅自裁剪变换。
2. `@tilelang.jit()` 装饰的函数（TileLang 内核函数）声明在迁移前后保持不变。

## 原平台示例

原队列版本戳示例：`- toolchain_stamp: tilelang@2026-08-28 build, Ascend910B2C, CANN 8.x`。

原启动示例：请帮我开发一个 softmax 算子，target=npuir。

## 原平台辅助技能目录

| 辅助类 | `tilelang-mlir-skill` / `tilelang-debug-helper` / `tilelang-error-fixer` / `tilelang-review-skill` / `tilelang-github-operations` | pass 调试 / GDB+IR / 错误修复 / 审查 / GitHub 工作流 |


## 原 NPUIR 启动示例与文档

前置：NPU 设备可用、`source set_env.sh`、OpenCode 会话。向 conductor 发送需求即可（预检、调度、状态维护均自动）：

```
# 新算子（5 必需字段：算子名 / 公式 / 输入规格 / 输出规格 / 编程模式）
请帮我开发一个 softmax 算子，hardware=<已注册型号>，toolchain=<已注册工具链>。数学公式：softmax(x_i)=exp(x_i-max)/sum(exp(x_j-max))
输入：[B, N] float16，B 动态；输出：同输入；编程模式：Developer

# 迁移（plain：给出 GPU 源文件；harness：给出 TileOPs 同构仓路径）
请把这个 TileLang GPU 算子迁移到 npuir：文件 examples/gemm/matmul.py，输出 shape (M, N)

# 优化（指向已存在算子产物）
请优化 examples/norm/layer_norm/layer_norm.py 的性能，目标：latency < 100us

# 续跑/恢复
继续开发 examples/norm/layer_norm/ 的算子
```


| 文档 | 路径 |
| ---- | ---- |
| 快速入门 / 开发指南 / 调试指南 | `docs/快速入门.md` / `docs/开发指南.md` / `docs/Tilelang算子调试指南.md` |
| 环境变量 / NPU Runtime / 安装 | `docs/developer/EnvironmentVariables.md` / `docs/developer/npu runtime.md` / `docs/安装指南.md` |
| 模式对比 / 贡献指南 | `docs/Developer_Expert_Mode对比.md` / `docs/Tilelang-Ascend贡献指南.md` |
