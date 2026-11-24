# RETROSPECTIVE -- mla_dense_bwd (migration-harness)

## Stage 5 (integrate) Retrospective — harness / attempt 1 — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py::rewrite_wrapper_import` 只替换 `from .{extracted_module} import ...` 这一行，不清理 scaffold 在 import 前写的 “Part A: the GPU TileLang kernel factories …” 注释块，也不处理模块 docstring 里 “Pre-Stage-3 note: the imported factories are still the GPU (CUDA target) implementations … Running forward before that rewrite fails”。集成后 wrapper 顶部的两处文字与新切换块、与真实 import 目标自相矛盾，后续 Stage 4/6 读 wrapper 会被误导。 | `examples/TileOPs/tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd.py#45-70`（原文见 `history_version/mla_dense_bwd_s5_attempt0.py`）；integrate_kernel.py 本次输出 `[wrapper] ... rewritten`，`[warn]` 无任何提示 | 让 `rewrite_wrapper_import` 在替换 import 时一并定位 scaffold 的 “Part A / Pre-Stage-3 note” 段落，标记为「已集成」或删除；至少在 `[warn]` 中提示残留注释需人工清理 | R |
| Integrate-flow | 已知修复目录（agent 内嵌表）缺一条高频项：「Kernel expected N inputs, but M are provided」= wrapper 仍按 GPU 二级 tiling 调用（`self.kernel(block_n, block_m, num_stages)`），而 Stage 3 冻结工厂已返回编译好的程序。该症状首次 report 即出现，8/10 用例同一条报错，但从报错文本无法直接看出是 wrapper 而非 kernel。 | `reports/tileops/20261123_113734_679663_MLADenseBwdOp/report.md#Failed Cases`；修复见 `mla_dense_bwd_kernel/integration_log.md#调试历史 attempt 1` | 在 integrator agent 的「已知修复目录」增加一行：`Kernel expected N inputs, but M are provided` → 检查 wrapper `_compile` 是否多套了一层 config 调用 / `default_config` 是否残留 GPU tiling 键 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 迁移 op 的 wrapper 必须按“集成后工厂返回形态”校准调用层级：Stage 3 冻结工厂返回 `tilelang.compile(...)` 编译好的程序（`_bwd_kernel` 11 输入、`_delta_kernel` `out_idx=[2]`、`_dq_land_kernel` `out_idx=[1]`），GPU wrapper 的 `self.kernel(block_n, block_m, num_stages)` 二级调用会把 3 个 int 当输入实参，报 `Kernel expected 11 inputs, but 3 are provided`；修法是 `default_config={}` + `_compile` 直接返回 `self.kernel` | P | `mla_dense_bwd_kernel/integration_log.md#调试历史 attempt 1`；`history_version/mla_dense_bwd_s5_attempt1.py` vs `mla_dense_bwd.py#_compile` | `pytest tests/ops/test_mla_dense_bwd.py -k smoke-bwd-bf16`（修前 8/10 fail，修后 10/10 pass） | tilelang 0.1.15, CANN9.2.1, Ascend950PR_9579 | pattern-library（migration-integration） |
| 多 launch（一次调用启动 3 个 kernel）算子的 TileOPs msprof 采集失效模式：`msprof op --kernel-name` 无法用单一名字过滤整个调用，未声明 `msprof_kernel_name` 时 capture 含 5 个 op（`Range_*`、`ZerosLike_*`、`_bwd_main_kernel`、`_delta_main_kernel`、`_dq_land_main_kernel`），报告把 `Range_` 的 208 us 记成算子延迟（真实 `events` 口径 ds-v3-b2-8k 为 455066 us），并标 `invalid tileops profile: unfiltered capture contains 5 distinct ops` | D | `examples/TileOPs/reports/tileops/20261123_114002_550215_MLADenseBwdOp/run.json#performance.cases[0..1]` 与 `#warnings` | `cd examples/TileOPs && TILEOPS_TASK_DIR=<task_dir> python -m tileops.reporting.cli run --op MLADenseBwdOp --prof-mode msprof`（5 case 中 2 条 msprof 无效、3 条回落 events） | tilelang 0.1.15, CANN9.2.1, Ascend950PR_9579, msprof 25.7.rc1.10 | pattern-library / bottleneck-patterns |
| 反例（C）：把 benchmark 的 “partial + CLI 非零退出码” 当集成失败会误判——本次 `status=partial` 纯由 profiling 口径造成（correctness 10/10、benchmark pytest 5/5 passed、0 failed），必须读 `run.json` 的 `summary.correctness_passed` 与 `invalid_profile_count` 区分 | C | `reports/tileops/20261123_114002_550215_MLADenseBwdOp/run.json#summary`（`correctness_passed=true`、`benchmark_passed=false`、`invalid_profile_count=2`） | 同上 report 命令，看 CLI `status=partial` 与 run.json summary 的对比 | tilelang 0.1.15, CANN9.2.1, Ascend950PR_9579 | 待定（integrator agent 判定规则已含此条，可提升为案例索引） |

### Transferable Lessons（可迁移教训）

- 迁移算子的 wrapper 里，凡是 GPU 侧存在 `self.kernel(block_n, block_m, num_stages)` 这类“二级 tiling 调用”的，集成后必须核对 Stage 3 工厂是否已把 tiling 冻结进 `tilelang.compile(...)`；残留二级调用会把 int 当 tensor 实参，报 `Kernel expected N inputs, but M are provided`，且 `default_config` 的 GPU tiling 键应清空。
- 三段式（delta → main → land）这一类多 launch 算子的 msprof 采集天然无效：要么按每个 `*_main_kernel` 分别采集后求和，要么在 kernel 侧合并；不要相信 report 里 msprof 行的延迟数字，先看 `run.json` 的 `profiler_valid`。
- 集成脚本改写 wrapper import 后，scaffold 留下的“仍是 GPU 实现 / forward 尚不可用”等历史注释会与新状态冲突，集成时顺手清理（只改注释，不动行为），否则后续 Stage 会误判当前实现状态。
