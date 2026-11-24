# Integration Log -- EngramDecodeFwdOp

- meta: `examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/.migration_meta.json`
- integrated:
  - `_engram_project_kernel` -> `engram_decode_kernel/_engram_project_kernel.py`
    (from `examples/engram_decode/_engram_project_kernel/_engram_project_kernel.py`, 20 cases, max_diff 9.2e-05)
  - `_engram_step_kernel` -> `engram_decode_kernel/_engram_step_kernel.py`
    (from `examples/engram_decode/_engram_step_kernel/_engram_step_kernel.py`, 27 cases, 3 launches bit-identical)
- design_docs:
  - `_engram_project_kernel` -> `engram_decode_kernel/_engram_project_kernel_DESIGN.md`
  - `_engram_step_kernel` -> `engram_decode_kernel/_engram_step_kernel_DESIGN.md`
  - both copied verbatim from `examples/engram_decode/{func}/DESIGN.md` (no `[warn]`, no missing doc)
- attempts: 0 (deterministic integration + glue alignment only; no failure loop entered)
- scope: **round 1** = deterministic integration only (no device work, device busy);
  **round 2** = single-op report on the freed device — see 「验证结果」: the msprof report is
  the gate-referenced record (`status=partial`, 4 decode workloads without a valid profile),
  the `events` rerun is supplementary latency evidence only.

## 集成步骤（命令行）

```bash
cd /workspace/tilelang-mlir-ascend-950 && source .agents/platforms/toolchains/ascend/env-950pr.sh
cd examples/TileOPs
TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/engram_decode \
  /workspace/venv-ascend950/bin/python .agents/skills/add-npu-op/scripts/integrate_kernel.py \
  --meta tileops/kernels/sequence_modeling/engram_decode/.migration_meta.json \
  --task-dir /workspace/tilelang-mlir-ascend-950/examples/engram_decode
```

Script facts (platform `integration.script` = `.agents/skills/add-npu-op/scripts/integrate_kernel.py`):

- `[plan] op_slug=engram_decode family=sequence_modeling`
- 2 kernels copied, 2 DESIGN snapshots copied, aggregation `__init__.py` generated:
  `from ._engram_project_kernel import ALIGNMENT`, `... import _engram_project_kernel`,
  `from ._engram_step_kernel import _engram_step_kernel`;
  `__all__ = ["ALIGNMENT", "_engram_project_kernel", "_engram_step_kernel"]`
- `[wrapper] rewritten -> baseline/perf_opt kernel-source selection block` (baseline active,
  3 commented `perf_opt` lines)
- `[mode] ...: None (unchanged)` (wrapper already declared `ascend_mode = None`)
- `[smoke] smoke-ok ['ALIGNMENT', '_engram_project_kernel', '_engram_step_kernel']`
- no `[error]`, no `[warn]`

## dispatch / config 对齐（wrapper 胶水层）

The scaffold wrapper still carried the two GPU-derived launch choices; both were corrected
in `engram_decode.py` only (backup:
`engram_decode_kernel/history_version/engram_decode_s5_pre_integrate.py`).

| piece | scaffold (GPU) | integrated (Stage 3) |
|-------|----------------|----------------------|
| project config | `block_k=_projection_block_k(d_mem)` (64/32/16), `num_stages=4` | `block_k=256`, `num_stages=2`, `block_m=16`, `block_n=min(64,d_pad)` |
| step launch | `_func()` with thread count stripped (K9) | `_func(threads)` with `threads=min(512,next_pow2(d_pad))` |
| `_projection_block_k` | static helper kept | removed (unused; its only caller was the GPU `default_config`) |

Implementation choice: `default_config` and the step lane count now **delegate to the
integrated modules** instead of duplicating constants —

```python
from .engram_decode_kernel._engram_project_kernel import default_config as _project_default_config
from .engram_decode_kernel._engram_step_kernel import default_config as _step_default_config
...
return _project_default_config(self.d_padded)                     # Kernel.default_config
)...(_step_default_config(d_padded)["threads"])                   # step launch
```

so the wrapper cannot drift from the Stage 3 constants it was verified with, and a Stage 4
perf_opt drop-in has exactly one place to repoint (comment in the source-selection block).
`block_n` stays `min(64, d_pad)` and `d_padded = align_up(d, ALIGNMENT=256)`, both unchanged
from the scaffold (`ALIGNMENT` now re-exported from the integrated project kernel, value 256
in both modules).

Also verified/updated in the same file (glue only):

- `ascend_mode = None` is a plain class-literal assignment and the module defines exactly
  one Kernel class (`EngramDecodeKernel`) -- `ast` check + script `[mode]` report agree.
  The ascend backend has no Developer/Expert mode declaration, so `None` is final, not a
  pre-Stage-3 placeholder.
- module docstring: Part A now describes the integrated Stage 3 NPU factories (it still
  described the GPU `_engram_decode_fwd_kernels.py` reference and a "Pre-Stage-3 note"
  claiming `forward` fails before rewrite); K9 bullet and the launch comments updated.
- `ruff check` / `ruff format --check` clean on the wrapper after the edits (import block
  re-ordered by ruff's isort fix; comment blocks preserved).

## 集成侧参考实现核对

`tileops/workloads/sequence_modeling.py::engram_decode_step_torch` is the corrected
version: it left-pads `conv_state` (L rows) to `max_conv_len` once and returns
`cat([padded_history[:, 1:, :], vhat_n])`, i.e. **always `max_conv_len` rows**, matching the
manifest shape rule (`new_conv_state: [batch, max_conv_len, d]`) and the kernel contract.
Read-only confirmation; file untouched.

Manifest cross-check (read-only): `EngramDecodeFwdOp.source.test =
tests/ops/test_engram.py`, `source.bench = benchmarks/ops/bench_engram.py` -- identical to
`meta.test_path` / `meta.bench_path`.

## 静态验证（无 device 计算）

Import smoke (re-run after the glue edits):

```
smoke-ok ['ALIGNMENT', '_engram_project_kernel', '_engram_step_kernel']
```

Config resolution (kernel objects only; factories return closures, nothing is compiled and
no device tensor is allocated):

| d | d_padded | `Kernel.config` | step_threads |
|---|----------|-----------------|--------------|
| 512 | 512 | `{block_m:16, block_n:64, block_k:256, num_stages:2}` | 512 |
| 600 | 768 | `{block_m:16, block_n:64, block_k:256, num_stages:2}` | 512 |
| 256 | 256 | `{block_m:16, block_n:64, block_k:256, num_stages:2}` | 256 |

Values equal the integrated `_engram_project_kernel.default_config(d_pad)` and
`_engram_step_kernel.default_config(d_padded)["threads"] = min(512, tile)` for every row; a
user `config={"block_k": 128}` override still merges (Stage 4 A/B path intact).
Byte-identity: both copied kernels and both DESIGN snapshots `cmp`-clean against
`examples/engram_decode/...`.

## 验证结果

门禁（`S5-REPORT-PROF`）引用的是**请求 msprof 模式**的报告，因此正式记录 = msprof 报告
（`status=partial`）；`events` 补跑只作为有效 latency 的补充证据，不当门禁引用。

### 正式记录：msprof 报告（gate 引用，`integration_report.json.run_json`）

```bash
cd examples/TileOPs
TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/engram_decode \
ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python -m tileops.reporting.cli run \
  --op EngramDecodeFwdOp --prof-mode msprof
```

- report_run_json: `examples/TileOPs/reports/tileops/20261123_143920_807616_EngramDecodeFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261123_143920_807616_EngramDecodeFwdOp/report.md`
  （同目录 `report.html`）
- prof-mode: `msprof`（`metadata.prof_mode_requested=msprof`）；`status=partial`，CLI exit 1
- correctness: **pass, 46/46**（共享 `tests/ops/test_engram.py`；本算子 `EngramDecodeFwdOp`
  16/16：8 条参数化主用例 + 8 条结构/契约用例，decode 参数化用例 `max_abs_err` 最大 1.22e-04）
- benchmark: benchmark 已实际运行（10 case），**本算子 4 条全部 `profiler_valid=false`**，
  `invalid_profile_count=4`：

  | case | captured ops | reason | 记录值 (us) |
  |---|---|---|---:|
  | serving-b1 | project + step (2) | unfiltered capture contains 2 distinct ops | 3.141 |
  | serving-b4 | project + step (2) | 同上 | 4.579 |
  | serving-b8 | project + step (2) | 同上 | 6.385 |
  | decode-partial-cache | `PadV3_e1345c33…` + project + step (3) | unfiltered capture contains 3 distinct ops | 3.092 |

  根因：一次 decode forward 固定发 2 个 launch（`_engram_project_kernel` +
  `_engram_step_kernel`），msprof 的单 kernel-name 过滤无法隔离其一的耗时；`L < max_conv_len`
  的 partial-cache workload 还多一个 host `F.pad` 生成的 `PadV3` 设备算子，故 3 个 op。
  上述 4 个数值是未过滤整段 capture 的时间，**不作为性能结论**（仅记录）。另外 6 条
  `train-*` case（共享 bench 的其它 Engram 算子）msprof 有效，6.2–11.4 us。
- 结论：正确性通过、benchmark 已运行但本算子无有效 profile → `status=partial`，平台口径
  "只记录不修复"，门禁允许 `S5-REPORT-BENCH` 告警。

### 补充证据：events 报告（有效 latency，非门禁引用）

- events_run_json: `examples/TileOPs/reports/tileops/20261123_144235_457422_EngramDecodeFwdOp/run.json`
- events_report_md: 同目录 `report.md`；prof-mode `events`，`status=passed`，CLI exit 0
- correctness: **pass, 46/46**（本算子 16/16，与 msprof 轮一致）
- benchmark: **10/10 case 有效，`invalid_profile_count=0`**，本算子 4 个 workload：

  | label | shape | dtype | latency (us) | bandwidth (TB/s) |
  |---|---|---|---:|---:|
  | serving-b1 | batch 1, d 256, d_mem 512, max_conv_len 12 | float16 | 45.90 | 0.0118 |
  | serving-b4 | batch 4, d 512, d_mem 1024, max_conv_len 20 | float16 | 46.63 | 0.0490 |
  | serving-b8 | batch 8, d 256, d_mem 512, max_conv_len 18 | bfloat16 | 46.16 | 0.0150 |
  | decode-partial-cache | batch 2, d 256, max_conv_len 12 (L=5) | float16 | 54.95 | 0.0100 |

- 口径差异：`events` 计的是整次算子调用的 host 提交 + 设备执行（两次 launch 顺序下发），
  故为 45–55 us；msprof 计单 kernel 设备时间（3–6 us，且本算子因双 kernel 而无效）。
  **两者不可直接比较**，也不构成 Stage 4 优化基线；需单 kernel 归因时应先在 benchmark 侧
  支持多 kernel-name 聚合。

## 调试历史

none -- **0 attempt** 花在 kernel/wrapper 修复上：

- round 1（集成）：脚本无 `[error]`/`[warn]`；仅两处计划的胶水对齐（config/dispatch）与
  docstring 更新。
- round 2（report）：msprof 的 4 条无效 profile 按平台口径**只记录不修复**（根因是
  benchmark 侧单 kernel-name 过滤与"一次 forward 两 launch"的架构不匹配，属 report 工具
  能力边界，不在 kernel/wrapper 可修范围）；补 `events` 运行后 `status=passed`，正确性
  46/46 一次通过，未进入任何修复闭环。

## issues / observations

1. **msprof 与两 launch 算子的结构性不兼容（遗留风险，report 侧）**：`EngramDecodeFwdOp`
   每次 forward 发 `project` + `step` 两个 kernel，msprof 的单 kernel-name 过滤只能给出
   "2 distinct ops" 的整段 capture → 本算子 4 个 workload 的 msprof 数值永远无效。本次以
   `events` 模式取得有效 latency；若后续 Stage 4 需要单 kernel 级归因，得在 benchmark 侧
   支持多 kernel-name 聚合（tests/benchmarks 不在本阶段写权限内）。
2. Stale scaffold reference `tileops/kernels/sequence_modeling/engram_decode/_engram_decode_fwd_kernels.py`
   (GPU-extracted factories) remains in the op package, now unreferenced by the wrapper. It is
   outside the Stage 5 edit scope (pre-existing scaffold artifact) and the wrapper docstring
   cites it as the porting reference; conductor may decide whether to delete it.
3. `tileops/manifest/sequence_modeling.yaml::EngramDecodeFwdOp` still carries
   `status: spec-only` (same for the other two Engram entries). The reporting CLI does not
   read `status` (no reference in `tileops/reporting/`), so it should not block the report;
   manifest is out of Stage 5 write scope, reported only.
4. The integrated step module keeps its harness-side `sys.path` bootstrap
   (`_REPO_ROOT = Path(__file__).resolve().parents[3]`, then
   `from tileops.workloads.sequence_modeling import engram_decode_step_torch`). Inside the
   package that path computes to `tileops/kernels/examples/TileOPs` (a no-op insert) and the
   import resolves through the installed package -- verified by the import smoke, so no
   edit was needed.
5. `_engram_project_kernel.py` and `_engram_step_kernel.py` each define top-level helpers
   with the same names (`default_config`, `_align_up`, `_run_case`, `main`, ...); the package
   `__init__.py` re-exports explicitly, so there is no shadowing, but anyone importing
   `default_config` must use the module path (the wrapper does).
6. 报告阅读提示：本次 `report.md` 的算子表显示 `Correctness 100.0% (28/28)`，而 `run.json`
   `summary` 是 46/46。两者不矛盾——analyzer 先按 JUnit 用例的 `op` property 归属（本算子 16
   条），其余无 `op` property 的共享结构用例因三个 Engram 算子在 `tests/ops/test_engram.py`
   中同时命中 `_test_key` 而回落到默认算子（`requested_operator`），于是计入本算子的 28 条。
   验收以 `run.json` 的 46/46（本算子 16/16）为准。
