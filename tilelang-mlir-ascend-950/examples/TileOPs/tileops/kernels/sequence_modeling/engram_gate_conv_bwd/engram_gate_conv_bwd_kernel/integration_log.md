# Integration Log -- EngramGateConvBwdOp (engram_gate_conv_bwd, family=sequence_modeling)

- meta: `examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/.migration_meta.json`
  (`op_name=EngramGateConvBwdOp`, `kernel_class_name=EngramGateConvBwdKernel`,
  `test_path=tests/ops/test_engram.py`, `bench_path=benchmarks/ops/bench_engram.py`)
- Stage 3 product: `examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py`
  (L0 10/10, L1 6/6, L2 4/4, Boundary 5/5; second-pass verification passed)
- Stage 1 deliverable: `examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/DESIGN.md` (snapshot only, not edited)
- platform: ascend / ascend-950pr-9579 (dav-3510, 28 Cube / 56 Vector; CANN 9.2.1, compiler 12.3.0)
- task dir (`TILEOPS_TASK_DIR`): `/workspace/tilelang-mlir-ascend-950/examples/engram_gate_conv_bwd`
- interpreter: `/workspace/venv-ascend950/bin/python`; integration script
  `examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py`, run with cwd `examples/TileOPs` after
  `source .agents/platforms/toolchains/ascend/env-950pr.sh`
- **scope: Stage 5 integration (step 1) + single-op report (step 2), both complete.** The report ran on an idle
  NPU (`ASCEND_RT_VISIBLE_DEVICES=0`, `--prof-mode msprof`); see "验证结果" below.
- attempts: 0 debug-loop attempts. 1 integration-time glue fix (wrapper dispatch + refusal, see 调试历史 attempt 0),
  and 0 fixes after the report (correctness passed on the first run).

## 本轮产物

- integrated: `_engram_gate_conv_bwd_kernel` -> `engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py`
  (copied by the script from the Stage 3 product; byte-identical, `md5 bfc3be54...`, 56265 B). **Not reformatted**
  -- the file is kept as the verbatim Stage 3 deliverable for provenance; its lint findings are listed under Issues.
- design_docs: `_engram_gate_conv_bwd_kernel` ->
  `engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel_DESIGN.md`
  (byte-identical to the Stage 1 deliverable, `md5 fe993967...`, 106800 B; snapshot only, not edited)
- aggregated `__init__.py`: written by `integrate_kernel.py` (`md5 67e4b7d7...`)
  `from ._engram_gate_conv_bwd_kernel import _engram_gate_conv_bwd_kernel`;
  `__all__ = ["_engram_gate_conv_bwd_kernel"]`; no `[warn]` lines (the first pass had two -- see attempt 0)
- wrapper: `tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd.py` (`md5 0408f355...`)
  - import rewritten by the script into the baseline/perf_opt kernel-source selection block; baseline
    `from .engram_gate_conv_bwd_kernel import _engram_gate_conv_bwd_kernel` active, the
    `perf_opt._engram_gate_conv_bwd_kernel` import commented for Stage 4 (`wrapper_rewritten: true` on the first
    run). Import names corrected by hand from `ALIGNMENT, CONV_KERNEL_SIZE, _engram_gate_conv_bwd_kernel` to the
    single factory the integrated package defines (attempt 0).
  - `ascend_mode = None` -- plain assignment, already correct, script reported `unchanged` both runs; exactly one
    `Kernel` subclass in the file (`EngramGateConvBwdKernel`), matching `adapter.integration_mode -> None`
  - `supported_archs: Optional[list] = None`, `default_config == {}` (verified by import, no construction)
  - history: `history_version/engram_gate_conv_bwd_s5_attempt0_preintegrate.py` (`md5 5e1fd7af...`, the scaffold
    wrapper before any Stage 5 edit; backed up before the first script run)

## 声明式 refusal 上界（显式口径记录，conductor 需知悉）

- **本迁移声称的上界是 `d <= 1024`，不是 manifest 的 `d <= 4096`。**
- 依据: Stage 3 kernel `_check()` 对 `d > MAX_HIDDEN_DIM = 1024` 抛
  `ValueError("d=... exceeds the hidden-dim refusal of 1024 ...")`；融合结构每 SIMD-padded channel 占 152 B，
  `d <= 1024` 在 `seq_len = 512` 时仍余 >= 35 % 的 248 KiB UB（DESIGN.md 9.2 R-1）。manifest 的 4096 需要
  DESIGN 1.6.4 的 **备选 B1（分阶段 fallback）**，本次**未实现**。
- wrapper 侧落点: `MAX_HIDDEN_DIM = 1024` + `_refusal(dtype, M, seq_len, d)`（GPU `refusal()` 契约的内联版），
  在 `Kernel.__init__` 构造/编译之前抛出；实测 `d=1024` 放行、`d=1025`/`d=4096` 报
  `d must be <= 1024`、fp32 报 `runs float16 and bfloat16`。
- **残留口径不一致（不可在本阶段修复）**: `tileops/ops/sequence_modeling/engram_gate_conv_bwd.py` 仍声明
  `MAX_D = 4096` 并在 `forward` 里先做 `d > 4096` 检查（`tileops/ops/` 不在 Stage 5 可改范围）。因此端到端
  有效上界 = `min(4096, 1024) = 1024`：`d in (1024, 4096]` 由本 wrapper 拒绝，`d > 4096` 由 Op 先拒绝。
  两个错误消息文案不同（Op: `d must be <= 4096`；Kernel: `d must be <= 1024`）。
- 若用户后续需要 4096，应回设计修订实现 DESIGN 1.6.4 B1，并同步 manifest shape_rule 与 `ops/` 的 `MAX_D`。

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261123_143418_008444_EngramGateConvBwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261123_143418_008444_EngramGateConvBwdOp/report.md`
- command: `python -m tileops.reporting.cli run --op EngramGateConvBwdOp --prof-mode msprof`
  (cwd `examples/TileOPs`, `TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/engram_gate_conv_bwd`,
  `ASCEND_RT_VISIBLE_DEVICES=0`, idle NPU 0, no residual pytest/msprof processes after the run)
- run status: **`partial`** (CLI exits non-zero) -- **sole cause is `EngramDecodeFwdOp`'s 4 invalid benchmark
  profiles** (multi-kernel capture that single-name filtering cannot split: `unfiltered capture contains 2/3
  distinct ops`); run.json `warnings` lists exactly those 4 Decode cases plus "correctness passed, but the
  benchmark stage did not produce valid records". **All 3 of this operator's own benchmark cases are valid.**
- correctness: **pass, 46/46** (`summary.correctness_passed=true`, `correctness_failed=0`, `passed_cases=46`,
  `pass_rate=1.0`; `errors=0`, `skipped=0`). `tests/ops/test_engram.py` is shared by the three Engram ops, so the
  full file ran. **This operator's own cases: 17/17 passed** -- 8 workload params
  `test_engram_gate_conv_bwd[1-32-256-fp16]`, `[1-32-256-bf16]`, `[2-64-512-fp16]`, `[2-16-256-bf16]`,
  `[2-512-256-fp16]`, `[3-401-256-bf16]`, `[1-8-256-fp16]`, `[2-1-256-fp16]`; 2 `_roofline`; 7 contract cases
  (`rejects_unsupported_dtype`, `rejects_non_fp32_saved_scalars`, `noncontiguous`, `weight_grads_are_fp32`,
  `dconv_w_keeps_nonfinite_like_torch[inf/-inf]`, `matches_each_output`). Worst recorded `max_abs_err` = 1.06e-02
  (`[3-401-256-bf16]`, tolerance 0.3/0.2); report.md operator-level Avg Max Abs Error = 2.05e-03.
  Note on report.md's "100.0% (30/30)" row: that is this operator's *scope attribution* (17 explicitly tagged Bwd
  cases + 13 untagged shared cases), not a contradiction of run.json's 46/46.
- benchmark: **partial** (`summary.benchmark_passed=false`, `invalid_profile_count=4`); pytest side
  `benchmark_tests` = 10/10 passed. msprof side for this operator = **3/3 valid**:

  | case | params | dtype | latency (µs) | BW (TB/s) | perf (TOPS) |
  |---|---:|---|---:|---:|---:|
  | train-s64 | M=2, seq=64, d=512 | float16 | 7.542 | 0.1417 | 0.5949 |
  | train-s128 | M=1, seq=128, d=256 | bfloat16 | 6.215 | 0.0862 | 0.3625 |
  | train-s96 | M=3, seq=96, d=768 | bfloat16 | 11.678 | 0.3058 | 1.3000 |

  The 4 invalid records are all `EngramDecodeFwdOp` (`serving-b1`, `serving-b4`, `serving-b8`,
  `decode-partial-cache`). Benchmark is report-only: no tuning, no retry, no code change.
- profile 口径说明（只记录，不修复）: one `forward` of this operator launches **2 kernels** (fused `main` +
  reduction `main`). With `--kernel-name=main` msprof reports `captured_op_count=1` / `sample_count=2`, and the
  recorded latency is the **max over the two launches** (`max_latency_us == latency_us`; the other launch is
  3.102 / 2.397 / 3.523 µs respectively). So the table above is per-launch peak, not the summed end-to-end time of
  the op; the 2-kernel single-name filter limitation is inherent to the harness (K9/`msprof_kernel_name` is left
  undeclared), not a defect of this kernel.
- benchmark coverage: the harness runs benchmarks with `-m not smoke`, so the smoke-marked first workload
  (train-s32, fp16+bf16) is not profiled; that is why 3 of the 5 parametrized Bwd bench cases appear.
- `integration_report.json`: replaced with the real `{"run_json": ...}` pointer to this run.
- import smoke (`integrate_kernel.py`, final run): `smoke-ok ['_engram_gate_conv_bwd_kernel']`
- extra static/import checks (same interpreter, cwd `examples/TileOPs`, no device touched, nothing constructed --
  construction compiles both launches):
  - `tileops.kernels.sequence_modeling.engram_gate_conv_bwd.engram_gate_conv_bwd_kernel` -> `__all__` ==
    `['_engram_gate_conv_bwd_kernel']`
  - `EngramGateConvBwdKernel.ascend_mode is None`, `.supported_archs is None`, `.default_config == {}`
  - `_refusal` table (fp16/bf16 pass at d=256/1024; 1025/4096 refused; fp32 refused; M=0 refused)
  - AST check of the dispatch: `_engram_gate_conv_bwd_kernel` is called with **4** positional args and its result is
    called with the twelve tensors (one call level) at both sites (custom_op body, `Kernel.__init__`)
  - `tileops.kernels.sequence_modeling` -> `['EngramDecodeKernel', 'EngramGateConvBwdKernel',
    'EngramGateConvFwdKernel']`; `tileops.ops.sequence_modeling.engram_gate_conv_bwd.EngramGateConvBwdOp`
    importable, its `MAX_D` still 4096 (unmodified, out of scope)
- manifest cross-check: `tileops/manifest/sequence_modeling.yaml::EngramGateConvBwdOp`
  `source.test = tests/ops/test_engram.py`, `source.bench = benchmarks/ops/bench_engram.py`,
  `source.kernel = .../engram_gate_conv_bwd.py`, `source.kernel_map = {engram_gate_conv_bwd:
  EngramGateConvBwdKernel}`, `source.bench_manifest_driven = true` -- all agree with `.migration_meta.json`.
  `status` is still `spec-only` (manifest is not Stage 5 scope).

### Report scope note

`tests/ops/test_engram.py` and `benchmarks/ops/bench_engram.py` are shared by `EngramGateConvFwdOp`,
`EngramGateConvBwdOp` and `EngramDecodeFwdOp`, so one report run always executes all three operators' cases:
46 correctness tests (46/46 passed) and 10 benchmark tests (10/10 pytest-passed; 4 Decode profiles invalid).
The `--op EngramGateConvBwdOp` scope selects this operator's rows/warnings attribution; the numbers above isolate
its own 17 correctness cases and 3 valid benchmark cases.

## 调试历史

### attempt 0 -- integration-time glue fix (not a report failure; no attempt budget consumed)

- 症状 (first `integrate_kernel.py` run, import smoke FAILED):
  `ImportError: cannot import name 'ALIGNMENT' from ...engram_gate_conv_bwd_kernel` at
  `engram_gate_conv_bwd.py:92`. Static cross-check of the wrapper against the Stage 3 contract found three
  mismatches, all in the wrapper glue:
  1. **imported names**: scaffold imports `ALIGNMENT, CONV_KERNEL_SIZE, _engram_gate_conv_bwd_kernel` from the
     *extracted GPU* module; the Stage 3 module defines none of the first two (it keeps `KS = 4` and derives
     `DW = -(-d // VL) * VL` itself). The script therefore generated `# [warn] ALIGNMENT: not found` /
     `# [warn] CONV_KERNEL_SIZE: not found` plus `__all__` entries that do not exist.
  2. **dispatch 层数**: scaffold called `_engram_gate_conv_bwd_kernel(M, seq_len, d, dtype, run_len)()(12 tensors)`
     (two levels: the GPU factory's launch-parameter level), while the Stage 3 factory
     `_engram_gate_conv_bwd_kernel(M, seq_len, d, dtype)` compiles both launches and returns
     `run(*inputs) -> 6 gradients` directly -- one level, **4 args** (evidence: kernel source §5 `return run`;
     `_build_fused`/`_build_reduce` both `tilelang.compile(...)` and are called inside the factory; the in-file
     L0/L1 runners call `fn = factory(...); fn(*args)`; `tilelang/jit/kernel.py JITKernel.__call__`).
  3. **host padding**: scaffold imported `ALIGNMENT` and did `F.pad` on eight tensors + trimmed the outputs back to
     `d`. The Stage 3 prim_funcs declare the *unpadded* shapes (`dY: (M, seq_len, d)`, out rows `DH[b, tp, :]` with
     `T.copy(dhrow[:d], ...)`, reduction clamping `cols = T.min(width, d - off)`) and pad only inside the UB
     (`DW = align_up(d, 64)`), matching the manifest signature -- the wrapper's pad path would have fed
     `d_padded != d`, i.e. a shape mismatch.
- 分类: wrapper 胶水错误（import 名 / dispatch 层数 / 张量契约），只改 wrapper 允许范围内的文件。
- 定位证据: Stage 3 源文件 `_check()` / `geometry()` / §5 工厂 / §6 `run_case()` 调用形式、
  `tilelang.compile(..., target="ascend", out_idx=[12, 13, 14, 15])` 与 `out_idx=[1, 2, 3]`、
  `tileops/ops/.../engram_gate_conv_bwd.py` 的 `_get_kernel` 构造签名、同族 fwd 集成包
  (`engram_gate_conv_kernel/`) 的同形修法。
- 修复（备份 `history_version/engram_gate_conv_bwd_s5_attempt0_preintegrate.py` 后）:
  1. import 改为单一 `from .engram_gate_conv_bwd_kernel import _engram_gate_conv_bwd_kernel`；删除
     `import torch.nn.functional as F`、`from tileops.utils.tiling import align_up`、
     `from tileops.device import get_device_backend`（三个消费者都已消失）；perf_opt 注释行同步为单函数形态；
  2. custom_op 与 `Kernel.__init__` 的 dispatch 改为 `factory(M, seq_len, d, dtype_str)(12 tensors)`；custom_op 与
     `register_fake` 签名删除 `run_len`；
  3. 删除 `_core_count()` 与 `Kernel._run_len()`（K6：run 长度与 launch grid 由 kernel 按 shape + AI core 数自行
     推导），`forward` 删除 `F.pad` 与输出 trim，`self.d_padded` 一并删除；
  4. 新增 refusal：kernel 真实上界 `MAX_HIDDEN_DIM = 1024` + `_refusal()`，`__init__` 构造前抛出（见上节）；
  5. docstring/注释同步：Part A 改为"抽取件仅作迁移记录、实际使用 NPU 重实现"，新增 Stage-5 note（单级调用、
     无 `run_len`、无 host padding、refusal 1024 口径与 `ops/` 4096 的残留差异），K6/K9 说明更新，
     `out_idx` 注释从错误的"8 outputs / dvhat scratch"改为融合 4 输出 + 规约 3 输出。
- 结果: 重跑 `integrate_kernel.py`（幂等）-> `__init__.py` 重生成且无 `[warn]`、`[smoke] smoke-ok
  ['_engram_gate_conv_bwd_kernel']`、`wrapper_rewritten: false`、`mode: None (unchanged)`。第三次运行同样通过
  （最终记录）。

## Lint

- `engram_gate_conv_bwd.py`（wrapper，本阶段手改文件）：`ruff check` / `ruff format --check` 均通过
  （`examples/TileOPs/pyproject.toml` rule set，venv 内 ruff）。
- `engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py`（Stage 3 产物，逐字节拷贝）**不**通过
  `ruff check`：`I001`（import 块排序，可自动修）、`B002`（`-(-x // y)` 被当作一元 `--`，天花板除法的误报）、
  `SIM102`（嵌套 `if`）；`ruff format --check` 亦会重排。本阶段**故意不**格式化该文件以保持与 Stage 3 交付件
  逐字节一致（md5 可核对）；见 Issues 2。同族 fwd 集成包选择了格式化，本包不跟随。
- `engram_gate_conv_bwd_kernel/__init__.py` 生成即干净。`history_version/` 内的备份文件不干净（全仓一致现象）。

## Issues

1. `integrate_kernel.py::parse_wrapper_imports` 从 **extracted（GPU）模块** 收集 wrapper 的 import 名，生成
   `__init__.py` 的 `__all__` 时并不知道 Stage 3 模块定义了哪些名字：本次生成过
   `# [warn] ALIGNMENT/CONV_KERNEL_SIZE: not found` 且 `__all__` 含不存在符号（首次运行 import smoke 因此失败）。
   建议脚本在 `defining_module()` 找不到时不要写入 `__all__`、并对 `[warn]` 直接 `[error]` 早退。流程侧（R 类）。
2. Stage 3 产物不满足 TileOPs rule set B（`I001`/`B002`/`SIM102` + format），而集成是整文件拷贝，每条迁移链路
   都要在 Stage 5 再格式化 30 KB kernel；且 `per-file-ignores` 只覆盖 `_*_kernels.py`（复数），未覆盖
   `_*_kernel.py`（单数）。另有 `B002` 对 `-(-a // b)` 的误报，建议加 `# noqa` 或扩展 ignore。
3. wrapper 文档/工具链遗留的 GPU 时代 "pad d 到 256" 叙述已在本 wrapper 内改正；`tileops/ops/` 的
   `MAX_D = 4096` 与注释仍在（不在本阶段可改范围），造成端到端上界与声明不一致（见 refusal 上界一节）。
4. 从**仓库根目录**运行 python 会命中本地 `tilelang/` 源码树（缺私有 TVM）；本轮所有 python 命令均在
   `examples/TileOPs` 下执行。

## 遗留风险（report 阶段新发现，只记录）

1. **本算子 latency 口径**: 每次 forward 启动 2 个 kernel，二者 prim_func 同名 `main`，msprof 以
   `--kernel-name=main` 过滤后拿到 `sample_count=2`，报告取两次的**最大值**（`max_latency_us == latency_us`）。
   若后续需要端到端时间或逐 launch 归因，需要在 `Kernel.msprof_kernel_name` 或报告侧做多 kernel 处理
   （本阶段不改）。这是 harness 的单名过滤限制（与 Decode 的 4 条 invalid 同源），不影响正确性。
2. **benchmark 不含 smoke workload**: harness 以 `-m not smoke` 跑 benchmark，mark 为 smoke 的首个 workload
   （train-s32 fp16/bf16）不进 profile，故本算子只 profile 了 3/5 条参数化 case（其余三个 workload 全覆盖）。
3. **`run.json` 整轮 `partial`**: 由 `EngramDecodeFwdOp` 的 4 条无效 profile 引起；本算子 17/17 正确性、
   3/3 有效 bench case 均达标。conductor 若按整轮 `partial` 判定，请以上述算子内口径为准。
4. **端到端 refusal 上界仍是 1024**（见 "声明式 refusal 上界" 一节）：manifest `d <= 4096` 与 `ops/MAX_D = 4096`
   未同步；需要 4096 时须先实现 DESIGN 1.6.4 B1。
