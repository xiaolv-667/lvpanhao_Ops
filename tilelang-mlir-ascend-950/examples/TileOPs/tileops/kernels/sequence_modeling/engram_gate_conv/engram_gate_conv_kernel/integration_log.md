# Integration Log -- EngramGateConvFwdOp (engram_gate_conv, family=sequence_modeling)

- meta: `examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv/.migration_meta.json`
  (`op_name=EngramGateConvFwdOp`, `kernel_class_name=EngramGateConvFwdKernel`, `test_path=tests/ops/test_engram.py`,
  `bench_path=benchmarks/ops/bench_engram.py`)
- platform: ascend / ascend-950pr-9579 (dav-3510, 28 Cube / 56 Vector; CANN 9.2.1, compiler 12.3.0)
- task dir (`TILEOPS_TASK_DIR`): `/workspace/tilelang-mlir-ascend-950/examples/engram_gate_conv`
- interpreter: `/workspace/venv-ascend950/bin/python` (Python 3.12.15, torch 2.13.0+cpu, torch_npu 2.13.0rc1,
  tilelang 0.1.15 from `site-packages`)
- integration script: `examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py`, run with cwd
  `examples/TileOPs` after `source .agents/platforms/toolchains/ascend/env-950pr.sh` (CPU-side only: copies +
  import smoke; **no device work, no card held this round**)
- **scope: Stage 5 complete.** Round 1 = deterministic integration only (copies + import smoke, CPU side);
  Round 2 (this entry) = single-op report `EngramGateConvFwdOp` on the free device 0, after the Bwd/Decode
  kernels landed, with the whole shared `tests/ops/test_engram.py` run in one report.
- attempts: 0 debug-loop attempts (report passed correctness on the first run); 1 integration-time glue fix
  (wrapper dispatch/padding, see 调试历史 attempt 0); 1 report run, 0 re-runs
- report command: `cd examples/TileOPs && source ../../.agents/platforms/toolchains/ascend/env-950pr.sh &&
  TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/engram_gate_conv ASCEND_RT_VISIBLE_DEVICES=0
  /workspace/venv-ascend950/bin/python -m tileops.reporting.cli run --op EngramGateConvFwdOp --prof-mode msprof`
  (CLI exit 1 -- `status=partial`, see 验证结果; a non-zero exit on `partial` is not an integration failure)

## 本轮产物

- integrated: `_engram_gate_conv_fwd_kernel` -> `engram_gate_conv_kernel/_engram_gate_conv_fwd_kernel.py`
  (copied by the script from `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/_engram_gate_conv_fwd_kernel.py`;
  byte-identical at copy time, `md5 348eea51...`; then ruff-format applied -> `md5 3b14a5b9...`, AST-identical,
  see "Lint" below)
- design_docs: `_engram_gate_conv_fwd_kernel` -> `engram_gate_conv_kernel/_engram_gate_conv_fwd_kernel_DESIGN.md`
  (byte-identical to `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/DESIGN.md`, `md5 e0386daa...`;
  snapshot only, not edited)
- aggregated `__init__.py`: written by `integrate_kernel.py`;
  `from ._engram_gate_conv_fwd_kernel import _engram_gate_conv_fwd_kernel`, `__all__ = ["_engram_gate_conv_fwd_kernel"]`
- wrapper: `tileops/kernels/sequence_modeling/engram_gate_conv/engram_gate_conv.py`
  - import rewritten to the baseline/perf_opt kernel-source selection block; baseline
    `from .engram_gate_conv_kernel import _engram_gate_conv_fwd_kernel` active, the
    `perf_opt/_engram_gate_conv_fwd_kernel` line commented for Stage 4 (`wrapper_rewritten: true`)
  - `ascend_mode = None` -- plain assignment, already correct, script reported `unchanged`; exactly one
    `Kernel` subclass in the file (`EngramGateConvFwdKernel`), matching `adapter.integration_mode -> None`
  - history: `history_version/engram_gate_conv_s5_attempt0.py` (scaffold wrapper before any Stage 5 edit),
    `history_version/engram_gate_conv_s5_attempt0_prelint.py` (post-integration, pre-ruff)

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261123_142454_662330_EngramGateConvFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261123_142454_662330_EngramGateConvFwdOp/report.md`
  (`report.html`, `benchmark.json`, `benchmark.md`, `pytest/correctness.xml`, `pytest/benchmark.xml`,
  `artifacts/msprof/*` in the same run dir; `reports/tileops/latest.json` points here too)
- report run: `20261123_142454_662330_EngramGateConvFwdOp`, `metadata.test_file = tests/ops/test_engram.py`,
  `metadata.benchmark_file = benchmarks/ops/bench_engram.py`, `prof_mode_requested = msprof`
- **run status: `partial`** (CLI exit 1). 门禁口径 `S5-REPORT-BENCH` 的判定是「正确性通过、benchmark 未得到
  有效结果 -> 仅记录、不阻断集成」；`S5-REPORT-STATUS/OP/TARGET/PROF/TEST` 均满足（`run.operator =
  EngramGateConvFwdOp`，`correctness_tests = 46 >= 1`）
- correctness: **pass, 46/46** (`summary.correctness_passed = true`, `correctness_tests = 46`,
  `correctness_failed = 0`, `errors = 0`, `skipped = 0`, `pass_rate = 1.0`; `pytest/correctness.xml` status
  `passed`)。共享文件里三个算子的用例全部通过：
  - `EngramGateConvFwdOp`（本算子）**13/13**：`test_engram_gate_conv_fwd[1-32-256-dtype0/1, 2-64-512-dtype2,
    2-16-256-dtype3, 1-128-256-dtype4, 3-96-768-dtype5]`（6 个 fixture shape，含 manifest train-s32/s128/s96）、
    `..._outputs_are_fresh`、`..._matches_each_output`（六输出逐个比）、`..._noncontiguous`、
    `..._rejects_unsupported_dtype`、`..._rejects_wide_d`、`..._roofline[dtype0/dtype1]`；
    六个有比对的 shape 的 `max_abs_err` = 1.83e-04 / 1.90e-03 / 2.44e-04 / 1.90e-03 / 1.95e-03 / 9.77e-04，
    远在 test 容差内（fp16 atol=rtol=0.1，bf16 atol=0.2/rtol=0.1）
  - `EngramGateConvBwdOp` 17/17、`EngramDecodeFwdOp` 16/16（同一轮全过；它们的集成没有问题）
  - 注：`report.md` 的 `Operator Analysis` 行写 `100.0% (28/28)`，是 reporting 层「单算子归因」口径
    （8 个带 `properties.op` 的用例 + 20 个未标注 op 的记录归给被请求算子），**不是** Fwd 自身用例数；
    pytest 口径的 Fwd 用例数是 13
- benchmark: **Fwd 3/3 有效 profile**；整轮 `benchmark_passed = false` 只因 **Decode** 的 4 条 profile 无效
  （见下），与本算子无关
- import smoke (`integrate_kernel.py`, round 1): `smoke-ok ['_engram_gate_conv_fwd_kernel']`

### Benchmark（msprof Task Duration，`run.json.performance.cases`，只记录不修复）

| operator | label | dtype | shape (M, seq_len, d) | latency (us) | valid | bandwidth (TB/s) |
|---|---|---|---:|---:|---|---:|
| EngramGateConvFwdOp | train-s64 | float16 | (2, 64, 512) | **8.466** | yes | 0.0784 |
| EngramGateConvFwdOp | train-s128 | bfloat16 | (1, 128, 256) | **7.348** | yes | 0.0453 |
| EngramGateConvFwdOp | train-s96 | bfloat16 | (3, 96, 768) | **11.117** | yes | 0.2002 |
| EngramGateConvBwdOp | train-s64 | float16 | (2, 64, 512) | 7.522 | yes | 0.1421 |
| EngramGateConvBwdOp | train-s128 | bfloat16 | (1, 128, 256) | 6.281 | yes | 0.0853 |
| EngramGateConvBwdOp | train-s96 | bfloat16 | (3, 96, 768) | 11.226 | yes | 0.3181 |
| EngramDecodeFwdOp | serving-b1 | float16 | — | 3.264 | **no** | — |
| EngramDecodeFwdOp | serving-b4 | float16 | — | 5.092 | **no** | — |
| EngramDecodeFwdOp | serving-b8 | bfloat16 | — | 6.289 | **no** | — |
| EngramDecodeFwdOp | decode-partial-cache | float16 | — | 3.052 | **no** | — |

- 整轮 `summary`: `benchmark_requested=true`, `benchmark_passed=false`, `benchmark_tests=10`,
  `benchmark_failed=0`, `msprof_case_count=10`, `profiler_fallback_count=0`,
  `kernel_filter_fallback_count=0`, `invalid_profile_count=4`。
- 4 条无效记录全部属于 `EngramDecodeFwdOp`，`invalid_reason` 分别为
  `unfiltered capture contains 2 distinct ops`（serving-b1/b4/b8：`engram_project_kernel` +
  `engram_decode_step_kernel`）与 `3 distinct ops`（decode-partial-cache 额外多一个
  `PadV3_e1345c336ff11c15a25aa8fa9e41b47b_high_performance_30021`）；即一次调用启动多个 kernel 而
  `msprof` 侧没有 kernel-name 过滤（`kernel_name_requested = null`），采集到整段 2-3 个 op 的叠加时长。
  这是 **Decode 算子 / benchmark harness** 侧的缺口：`tileops/benchmark/msprof.py` 的 kernel-name 解析链
  （`functor.msprof_kernel_name` -> `functor.kernel.msprof_kernel_name` -> env -> 从 `JitKernel_NPU` 图派生）
  本轮全部落空返回 `None`，且 `--kernel-name` 只能给**单个**名字，多 kernel op 无法一次过滤到两个 launch
  （即使给 step kernel 声明名字，也会漏掉 project kernel 的时长）。不属于本算子，benchmark 只记录不修复 ->
  原样记录，交 conductor / Decode 集成方与 harness 维护者处理。
- 本算子 latency 7.348-11.117 us，三个 shape 均为单 `main_kernel` 采集（`captured_op_count=1`、`sample_count=1`；
  `msopprof ... --kernel-name=main --aic-metrics=Roofline`）。`roofline_utilization_percent` 为 `null`
  （`report.md`/`benchmark.md` 的 `Ratio` 列 = `N/A`）：这三个 case 走的是
  `benchmark_base._build_result` 的「msprof roofline 解析失败 -> 退回理论指标」分支，只留下
  `tflops`（0.2233 / 0.1291 / 0.5749）与 `bandwidth_tbs`（0.0784 / 0.0453 / 0.2002），没有
  `Ratio/AI/Computility`；`pytest/benchmark.log` 里 Fwd/Bwd 六例均无 `=== [msprof] ... roofline:` 行，而
  Decode 四例有（证明是该 kernel 的 roofline bin 未被解析出来，不是 harness 全局失效）。故本算子不给利用率
  结论，只给 latency/带宽；数值为未调优（Stage 4 前）基线。
- 附注（不计入本算子）: Decode 的 `Ratio` 虽存在，但其 `Bandwidth = 1.6 TB/s` 是平台峰值常量（非实测），
  且其 profile 本身无效（见上），不可作为性能结论引用。

- `integration_report.json`: 已从占位替换为
  `{"run_json": "examples/TileOPs/reports/tileops/20261123_142454_662330_EngramGateConvFwdOp/run.json"}`
- manifest 一致性: `tileops/manifest/sequence_modeling.yaml::EngramGateConvFwdOp` 的
  `source.test = tests/ops/test_engram.py` / `source.bench = benchmarks/ops/bench_engram.py` 与
  `.migration_meta.json`、与 run.json `metadata` 三处一致；`bench_manifest_driven: true`

### 结论

本算子（`EngramGateConvFwdOp`）正确性 13/13 全过、benchmark 3/3 profile 有效；整轮 `partial` 仅由
Decode 的 4 条无效 profile 触发，按平台口径「正确性通过 + benchmark 未得到有效结果 = 仅记录、不阻断集成」，
Stage 5 判定 `INTEGRATE_COMPLETED`（附 `partial` 披露）。本轮无 kernel/wrapper 改动，故 report 不会过期
（wrapper / `__init__.py` / 集成 kernel 的 mtime 均早于本次 `run.json`）。

## 调试历史

### attempt 0 -- integration-time glue fix (not a report failure; no attempt budget consumed)

- 症状 (found by静态核对 while integrating, before any run): wrapper 与 Stage 3 契约不匹配两处 --
  (a) dispatch 形态: scaffold 为两级调用 `_engram_gate_conv_fwd_kernel(...)()(H, k, v, ...)`（第二级是 GPU 时代
  的空 launch 参数层），而 Stage 3 工厂 `tilelang.compile(...)` 直接返回 JIT kernel，一级调用
  `factory(...)(H, k, v, ...)`（铁证: kernel 内嵌 L0 runner `fn = factory(M, seq_len, d, 1e-6, dtype); fn(*args)`，
  11/11 通过；`tilelang/jit/kernel.py:206 JITKernel.__call__ -> torch_function(*args)`；DESIGN §10.3 工厂签名
  `_engram_gate_conv_fwd_kernel(M, seq_len, d, eps, dtype)`）；
  (b) host padding: scaffold import 了抽取件的 `ALIGNMENT` 并做 `F.pad` 六张量 + 输出裁回，而 Stage 3 的
  prim_func 声明 `H: T.Tensor((M, seq_len, d), ...)`、`rms_w_h: (d,)`、`conv_w: (taps, d)`（未 pad 原形状），
  内部用 `DW = align_up(d, 64)` + 尾 lane mask 自理对齐（DESIGN §0.6 R2 / §8 签名表："kernel 入参即未 pad 的原形状"）；
  manifest 签名同为 `[M, seq_len, d]` / `[d]` / `[4, d]`。若照 scaffold 传递，喂给 kernel 的最后一维会是
  `d_padded != d`，shape 不匹配。
- 分类: wrapper 胶水错误（dispatch 层数 / 张量契约），只改 wrapper 允许范围内文件。
- 定位证据: Stage 3 源文件 `run_case()` 的调用形式、`tilelang.compile` 返回 `JITKernel` 的 `__call__`、
  DESIGN §0.6 R2 / §8 / §10.3、manifest `EngramGateConvFwdOp.signature`。
- 修复（备份 `history_version/engram_gate_conv_s5_attempt0.py` 后）:
  1. `import torch.nn.functional as F`、`from tileops.utils.tiling import align_up` 删除（唯一消费者是 padding 路径）；
  2. `ALIGNMENT, _engram_gate_conv_fwd_kernel` -> `_engram_gate_conv_fwd_kernel`（集成包的聚合导出因此只含该工厂；
     若保留 `ALIGNMENT`，脚本会生成 `from .engram_gate_conv_kernel import ALIGNMENT, ...` 而集成产物无此名 ->
     `[warn] ALIGNMENT: not found` + import smoke 失败，与 ada_layer_norm 曾踩的同类坑一致）；
  3. custom_op 内 `factory(...)()(tensors)` -> `factory(...)(tensors)`；
  4. `Kernel.__init__` 删除 `self.d_padded = align_up(d, ALIGNMENT)`；`forward` 删除 `F.pad` 六张量 + `Y/vhat`
     裁回，直接 `return _engram_gate_conv_fwd_wrapped(M, seq_len, d, eps, dtype_str, H, k, v, w_h, w_v, conv_w)`；
  5. docstring/注释同步（K9 说明改为一级调用；新增 Stage-3 note 记录 padding 契约内移）。
- 结果: `integrate_kernel.py` 的 import smoke 通过；wrapper `ascend_mode = None` 未被脚本改写（`unchanged`）。

## Lint（TileOPs rule set B）

`format.sh` 无法使用（工作区不是 git 仓库，脚本依赖 `git rev-parse`），改为直接用同一 venv 的 ruff 0.17.6 与
`examples/TileOPs/pyproject.toml`（`select = E,F,B,SIM,I,N,RUF022`, `line-length = 100`）对集成包内文件执行
`ruff check --fix` + `ruff format`；改动前均备份到 `history_version/`（`*_s5_attempt0_prelint.*`）。

- `engram_gate_conv.py`: `I001`（生成的 selection block 里 baseline import 与 perf_opt 注释之间缺空行）-> 补 1 行空行；
  `ruff check` / `ruff format --check` 均通过。diff 仅此 1 行。
- `engram_gate_conv_kernel/_engram_gate_conv_fwd_kernel.py`: `I001`（`tilelang` / `torch` 导入顺序）+
  `ruff format` 行折叠（长 `raise ValueError(...)` 与多行 SIMD 调用）；formatted 后 `ruff check` / `ruff format --check`
  均通过。**语义等价已核**：对 pre-lint 备份与现行文件做 `ast.parse` + `ast.dump(include_attributes=False)` 比对，
  36 个顶层语句逐条相同、import 语句集合相同 -> 纯格式变化。参考已集成算子（rope / ssd_chunk_scan /
  ada_layer_norm 的集成包文件）同样 ruff-check + ruff-format 干净，本次对齐该约定。
- `engram_gate_conv_kernel/__init__.py`: 生成即干净，未改动。
- 注意: `integrate_kernel.py` 幂等重跑会**整文件覆盖** `_engram_gate_conv_fwd_kernel.py`，本次 format 结果会丢；
  重跑后需按本条复放（先备份再格式），或等 conductor 在三个算子全部集成后只跑一次集成。

## Issues

1. `integrate_kernel.py::gen_kernel_source_block` 生成的 import 块在 baseline import 与 perf_opt 注释之间不满足
   ruff `I001`（需要一行空行），每个算子集成后都要手工补 1 行 -- 流程侧（R 类）缺陷，建议生成器直接产出空行。
2. Stage 3 产物文件本身不满足 TileOPs rule set B（`I001` + format），而集成是整文件拷贝，于是每条迁移链路都要在
   Stage 5 再格式化一次 30 KB kernel；且集成重跑会覆盖（见上）。per-file-ignores 只覆盖 `_*_kernels.py`（复数），
   未覆盖 `_*_kernel.py`（单数）。
3. 从**仓库根目录**运行 python 会命中本地 `tilelang/` 源码树（缺私有 TVM，`ImportError: TileLang's private TVM
   Python package is missing`）；所有 python 命令必须在 `examples/TileOPs` 下执行（本次冒烟均如此）。
4. wrapper/scaffold 遗留的 GPU 时代 "pad d 到 256" 叙述仍留在 `tileops/ops/sequence_modeling/engram_gate_conv.py`
   的注释/refusal 说明里（`MAX_D = 4096` 的依据文字）；`tileops/ops/` 不在本阶段可改范围，仅登记，不回改。
5. **（report 轮发现，非本算子）`EngramDecodeFwdOp` 的 4 条 benchmark profile 全部无效**：
   `unfiltered capture contains 2 distinct ops`（`engram_project_kernel` + `engram_decode_step_kernel`）
   / `3 distinct ops`（再叠一个 `PadV3_...`），`kernel_name_requested = null`。根因是 msprof 的 kernel-name
   解析链在「一次 forward 启动多个 kernel」的 op 上全部落空，且 `--kernel-name` 只支持单名（给 step kernel
   声明名字也会漏 project kernel 的时长）。后果：整轮 `status=partial`、`benchmark_passed=false`
   （`invalid_profile_count=4`）。benchmark 只记录不修复，且该 op 不属本 Stage 5 范围 -> 交 conductor /
   Decode 集成方与 harness 维护者。证据: `run.json.warnings` 4 条 + `performance.cases[*].invalid_reason` +
   `tileops/benchmark/msprof.py` L380-400 的解析链文档与 L471-480 的实现。
6. **（reporting 口径）** `report.md` 的 `Operator Analysis` 行把「未声明 `properties.op` 的用例」归给唯一被请求的
   算子（`analyzer._case_operator` 的 `default`），于是共享测试文件下 `EngramGateConvFwdOp` 显示 `28/28`
   （= 8 个声明 op 的用例 + 20 个 Bwd/Decode 用例），与 pytest 口径的 Fwd 13 例不一致；读报告时须以
   `run.json.correctness.cases` 的 `properties.op` / nodeid 为准。多算子共享测试文件时应改用
   `run --op all` 或让每个测试都 `record_check_result(op, ...)`。
