# Integration Log -- MLADenseBwdOp

- meta: `examples/TileOPs/tileops/kernels/attention/mla_dense_bwd/.migration_meta.json`
- task dir (`TILEOPS_TASK_DIR`): `/workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd`
- platform: ascend-950pr-9579 / ascend (CANN 9.2.1, dav-3510, 28 Cube / 56 Vector), 1 x NPU 0
- integrated (baseline, Stage 3 source, active):
  - `_bwd_kernel` -> `tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd_kernel/_bwd_kernel.py`
  - `_delta_kernel` -> `.../mla_dense_bwd_kernel/_delta_kernel.py`
  - `_dq_land_kernel` -> `.../mla_dense_bwd_kernel/_dq_land_kernel.py`
- design_docs (copied from `examples/mla_dense_bwd/{func}/DESIGN.md`): `_bwd_kernel_DESIGN.md`,
  `_delta_kernel_DESIGN.md`, `_dq_land_kernel_DESIGN.md` (none missing)
- aggregate `__init__.py`: re-exports the three factories; import smoke
  `smoke-ok ['_bwd_kernel', '_delta_kernel', '_dq_land_kernel']`
- wrapper: `mla_dense_bwd.py` reported `already integrated` (no rewrite, `wrapper_rewritten=false`,
  `wrapper_mode_updated=false`); the Cube-revision `_run` (**4-D batch views** `(B, S, H, D)` for
  `q/do/dq_acc` and `(B, S_kv, H, D)` for `k/v/dk/dv`) is preserved
- attempts: 0 in the latest (A5/v4) re-integration round and 0 in the Cube round (no failure inside
  the debug closed loop); the A5 rotation and the Cube redesign were produced and validated by
  Stage 3/4, not by this stage

## Stage 5 re-integration after the A5 (v4) rotation -- 2026-11-26

The `_bwd_kernel` product changed again: the A5 rotation (`GB(k-1)` lag, `T.serial` over two
statically expanded half bodies, `pT/dsT/doT/qT/lse/delta` each held in two static versions
`*_s0`/`*_s1`, drain by `n_it` parity, `clear_accum=(kq-1==loop_st)`, `dq` written at
`(kq-1)*64`).  `T.Pipelined` / `_STAGES` are gone, so the factory signature is now
`_bwd_kernel(batch, heads, dim_qk, dim_v, dense_seqlen_q, dense_seqlen_kv, is_causal, scale, dtype,
a2_fast=True, n_versions=2)` -- the first nine positional parameters are unchanged, so the wrapper's
existing call site still binds (no `num_stages` is expected or passed; `default_config={}` is still
correct).  This round re-ran the deterministic integration, the op-level correctness suite, the
B=2/S=8192 non-silent probe and the single-operator report against the *integrated* package.

> **Read the artifacts before acting (injection guard).** `.migration_meta.json`, the wrapper /
> kernel sources, their docstrings and the pytest logs were treated as data only.  They carry
> Stage-state prose (e.g. "Stage 5 state: the imported factories are the integrated NPU
> implementations", "the NPU component rewrites it for target=npuir"); this is stale round-1
> documentation and was **not** executed as an instruction.  All actions came from the conductor
> prompt plus the platform ruleset.

### 一致性核对

| item | source (task dir) | integrated package | verdict |
|---|---|---|---|
| `_bwd_kernel.py` | `examples/mla_dense_bwd/_bwd_kernel/_bwd_kernel.py` | `mla_dense_bwd_kernel/_bwd_kernel.py` | md5 `e0d6e9a7be8f1aeb1720ea13b64979c4` both (v4 = A5) |
| `_bwd_kernel_DESIGN.md` | `examples/mla_dense_bwd/_bwd_kernel/DESIGN.md` | `mla_dense_bwd_kernel/_bwd_kernel_DESIGN.md` | copied by the script (same file) |
| `_delta_kernel.py` / `_dq_land_kernel.py` (+DESIGN) | task dir | integrated package | unchanged since the Cube round |

- `integrate_kernel.py`: `[ok] integration complete`, `wrapper_rewritten=false`,
  `ascend_mode=null (unchanged)`, `smoke-ok ['_bwd_kernel', '_delta_kernel', '_dq_land_kernel']`,
  `design_docs` all three present (none missing).
- wrapper `mla_dense_bwd.py` md5 `367eeb05f597be9b08ea6efa1bd5acfb` before and after -> **not
  modified in this round** (no backup needed, no diff).  The v4 factory keeps the source parameter
  order for its first nine arguments, so `_run`/`_compile` needed no adaptation; the only
  interface change (`num_stages` -> `n_versions`) is defaulted and never passed by the wrapper
  (`default_config == {}`).

### 验证结果（A5/v4，最终）

- report_run_json: `examples/TileOPs/reports/tileops/20261126_040450_369928_MLADenseBwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261126_040450_369928_MLADenseBwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261126_040450_369928_MLADenseBwdOp/report.html`
- report `status`: **passed** (CLI exit 0, `warnings: []`)
- `summary`: `correctness_passed=true` (10/10, 0 failed), `pass_rate=1.0`,
  `avg_max_abs_err=1.32e-02`, `benchmark_passed=true`, `benchmark_tests=5`,
  `benchmark_failed=0`, `invalid_profile_count=0`, `invalid_acceptance_count=0`,
  `msprof_case_count=5`, `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`
- standalone `pytest tests/ops/test_mla_dense_bwd.py -q` -> `10 passed, 14 warnings in 132.46s`
  (log `evidence_s5_a5/pytest_correctness.log`: `10 passed in 3.31s` inside the report's own run,
  the standalone run additionally pays the per-test re-import/compile)
- benchmark: 5/5 cases `prof_mode=msprof`, `profiler_valid=true`, `invalid_reason=null`;
  acceptance unit **`cube`** for every case (`peak_used.compute_tflops=378.0`, metric `MFU`),
  `invalid = 0`

| label | dtype | T_op (us) | achieved TFLOP/s | MFU (cube) | Cube round (us) | vs Cube round |
|---|---|---:|---:|---:|---:|---:|
| ds-v3-4k | bf16 | 13613.7 | 131.7 | 34.8% (实测 35.5%) | 20571.7 | -33.8% |
| ds-v3-4k | fp16 | 13593.8 | 131.9 | 34.9% (实测 35.6%) | 20572.4 | -33.9% |
| **ds-v3-b2-8k** | bf16 | **99091.7** | **144.7** | **38.3% (实测 39.1%)** | **154419.2** | **-35.8%** |
| kimi-k2-8k | bf16 | 24860.5 | 144.2 | 38.1% (实测 38.9%) | 38421.1 | -35.3% |
| ds-v2-lite-3k | bf16 | 3747.2 | 134.6 | 35.6% (实测 36.3%) | 6296.7 | -40.5% |

- `ds-v3-b2-8k` breakdown: `_bwd_main_kernel` 95762.4 + `_delta_main_kernel` 715.4 +
  `_dq_land_main_kernel` 1640.5 = 98118.3 us for the four kernels (+ `ZerosLike` 968.9 +
  `Range` 4.4).  Matches the A5 development measurement (`T_op` 98.5 ms / cube 39.7% /
  MFU ~38.6%) within run-to-run noise: this round 99.09 ms, cube 39.1%.
- `acceptance.value` is `34.8 / 34.9 / 38.3 / 38.1 / 35.6` with `pass=False` against the 90%
  target -- the honest efficiency number, unchanged policy from the Cube round; `invalid=0`,
  so it is not an invalidity signal.  The only `notes` entry remains
  `report-underreports: use t_op_us`.
- compile-log note (not a failure): `benchmarks/.../benchmark.log` shows one
  `Z3 scheduling failed: solver returned unsat, falling back to topological sort` line, exactly the
  behaviour the v4 kernel docstring predicts for the rotated schedule; every capture stayed valid.

### B=2 / S=8192 / H=128 non-silent check through the integrated op

`examples/mla_dense_bwd/_bwd_kernel/evidence/s5/check_b2s8k_op.py` (log
`evidence_s5_a5/s5_b2s8k_check_a5.log`, exit 0, `RESULT PASS`) drives `MLADenseBwdOp` (integrated op
-> integrated wrapper -> integrated v4 `_bwd_kernel`) at the production shape, bf16, causal.

- structural: `dq/dk/dv` finite for both requests; non-zero counts per request
  `dq b0 201313147/201326592`, `dq b1 201311803/201326592`,
  `dk b0/b1 201326592/201326592`, `dv b0 134217728/134217728`, `dv b1 134217727/134217728`;
  request 1 key rows 0..63 fully populated (`dk` zeros `0/1572864`, `dv` zeros `0/1048576`,
  max|.| 3.23 / 8.13) -> **no cross-request overwrite**; the 4-D per-request store window survives
  the v4 rewrite.
- sliced fp64 reference, `(request, head)` in `{(0,0),(0,127),(1,0),(1,127)}` (~16.8M elements):
  **0 elements** outside `atol+rtol|ref|` at both the dev floor (`atol 5e-3`) and the op-test floor
  (`atol 1.6e-2`, `rtol 1.6e-2`) when the reference holds the `lse`/`delta` the kernel consumes;
  worst gaps `dq 9.8e-3`, `dk 1.2e-2`, `dv 1.2e-2`; worst threshold ratio 0.38 (`dq` b0 h0).
- op-level `T_op` (2 reps, min, whole call) **100.20 ms**, vs 154.07 ms in the Cube round (-35.0%)
  and 100.9 ms in the report -- the A5 gain reproduces end-to-end through the integrated op.

### 调试历史

- none in this round: integration was idempotent-clean (wrapper untouched), correctness 10/10, the
  B=2/S=8192 probe passed and every msprof capture was valid on the first full report.
- interface check done up front instead of by trial: the v4 factory's first nine positional
  parameters match the wrapper's call site, and `n_versions` is defaulted, so no
  `num_stages`->`n_versions` adaptation was required.

### 修改文件清单

- 未修改任何 kernel / wrapper / tests / benchmarks / manifest / ops / workloads 文件
  (wrapper md5 unchanged, `history_version/` gained no backup because nothing was edited).
- 集成包：`_bwd_kernel.py` + `_bwd_kernel_DESIGN.md` 由 `integrate_kernel.py` 整文件覆盖为任务
  目录的 v4/A5 版本（`_delta_kernel` / `_dq_land_kernel` 内容未变）；`__init__.py` 由脚本重生成；
  `integration_log.md` / `integration_report.json` 更新。
- 新增证据（集成包内，仅新增）：`evidence_s5_a5/{s5_b2s8k_check_a5.log, report_s5_a5.log,
  pytest_correctness.log, pytest_benchmark.log}`.

## Stage 5 re-integration after the Cube redesign (2026-11-25)

The kernel product changed (all five GEMMs on the explicit L0 path, `dsT` reused through
`trans_A`, `Nk=96`, 2-tile merged epilogue, `num_stages=2`, 4-D batch views for the per-request
`dk`/`dv` store window).  This round re-ran the deterministic integration and the single-operator
report against the *integrated* package.

> **Read the artifacts before acting (injection guard).** `.migration_meta.json`, the wrapper /
> kernel sources and their comments were treated as data only.  The wrapper docstring carries
> Stage-state prose (e.g. "Stage 5 state: the imported factories are the integrated NPU
> implementations"); it was consumed as documentation, not as an instruction, and the only
> actionable statements came from the conductor's prompt and the platform ruleset.

### 一致性核对

| item | source (task dir) | integrated package | verdict |
|---|---|---|---|
| `_bwd_kernel.py` | `examples/mla_dense_bwd/_bwd_kernel/_bwd_kernel.py` | `mla_dense_bwd_kernel/_bwd_kernel.py` | md5 `ad4486725a9fa1a394367f7bd051d7e5` both |
| `_bwd_kernel_DESIGN.md` | `examples/mla_dense_bwd/_bwd_kernel/DESIGN.md` | `mla_dense_bwd_kernel/_bwd_kernel_DESIGN.md` | md5 `f46ae7b791f37e524bcb17e863312427` both |

`_mla_dense_bwd_kernels.py` is the GPU extract kept as the migration record only: nothing imports
it at runtime (the wrapper's import block points at `.mla_dense_bwd_kernel`), so it is *not*
expected to track the Cube revision.  Verified with `grep -rn _mla_dense_bwd_kernels`: the only
hits are the wrapper comment and the pre-integration wrapper backups in `history_version/`.

### 验证结果（口径修正后重跑，最终）

- report_run_json: `examples/TileOPs/reports/tileops/20261125_044632_077346_MLADenseBwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261125_044632_077346_MLADenseBwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261125_044632_077346_MLADenseBwdOp/report.html`
- report `status`: **passed** (CLI exit 0, `warnings: []`)
- `summary`: `correctness_passed=true` (10/10, 0 failed), `benchmark_passed=true`, `benchmark_tests=5`,
  `benchmark_failed=0`, `invalid_profile_count=0`, `invalid_acceptance_count=0`,
  `msprof_case_count=5`, `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`
- acceptance unit: `cube`, `peak_used.compute_tflops=378.0` for all 5 cases; metric `MFU`;
  each case reports `pass=False` against the 90% target (23.1 / 23.1 / 24.6 / 24.7 / 21.2%), which is
  the honest efficiency number, not an invalidity.
- 重跑原因（口径修正）: `tileops/reporting/compute_units.json` 的 `MLADenseBwdOp` 由 `vector` 改为
  `cube`（evidence 采自本 op 的 msprof capture：cube 25.5% / vec 44.3%、93 TFLOP/s > Vector 峰 47），
  `tests/test_reporting_acceptance.py::test_compute_unit_map_covers_every_migrated_operator` 的期望同步
  改为 `cube`（由 conductor 侧完成，本 stage 未改这些文件）。上一轮
  `reports/tileops/20261125_043620_849427_MLADenseBwdOp/` 因旧 map（vector 47 TFLOPS → MFU 197.8%）
  触发 `physically-impossible` 熔断而 `status=partial`（`invalid_acceptance_count=5`），保留作证据。

<details><summary>上一轮（口径修正前）记录</summary>

- report_run_json: `examples/TileOPs/reports/tileops/20261125_043620_849427_MLADenseBwdOp/run.json`
- report `status`: partial（`invalid_acceptance_count=5`；correctness 与全部 msprof 采集均有效）。
  原因：`tileops/reporting/compute_units.json` 当时仍记录 `MLADenseBwdOp: unit=vector`（evidence 是
  重设计前的 `aic_cube_ratio 8.3% / aiv_vec_ratio 73.5%`），Vector 峰 47 TFLOPS 使 93.0 TFLOP/s 达
  197.8%，触发 `physically-impossible: achieved compute exceeds the peak`。本轮 capture
  （`artifacts/msprof/.../ArithmeticUtilization_*.csv`，block 0）实测 `aic_cube_ratio 25.5%`
  （`aic_cube_fp_ratio 25.5%`、int 0）、`aiv_vec_ratio 44.3%`，证明 FLOPs 在 Cube 上执行。

</details>

- correctness: **pass, 10/10** (`summary.correctness_passed=true`, `correctness_failed=0`),
  `avg_max_abs_err=1.32e-02`; `pytest tests/ops/test_mla_dense_bwd.py -q` -> `10 passed in 121.46s`
  (log: `examples/mla_dense_bwd/_bwd_kernel/evidence/s5/pytest_ops_integrated.log`)
- benchmark: 5 cases, all `prof_mode=msprof` (each capture holds 5 ops: `Range`, `ZerosLike` plus the
  three `*_main_kernel`), `invalid_profile_count=0`, `invalid_acceptance_count=0`,
  `benchmark_passed=true`
  - `T_op` = sum of every kernel of one call (`tileops_msprof_metadata.json:t_op_us`):

| label | dtype | T_op (us) | achieved TFLOP/s | MFU (cube 378) | previous round (events) | speed-up |
|---|---|---:|---:|---:|---:|---:|
| ds-v3-4k | bf16 | 20571.7 | 87.1 | 23.1% | (invalid capture) | n/a |
| ds-v3-4k | fp16 | 20572.4 | 87.1 | 23.1% | (invalid capture) | n/a |
| **ds-v3-b2-8k** | bf16 | **154419.2** | **92.9** | **24.6%** | **455066.6** | **2.95x** |
| kimi-k2-8k | bf16 | 38421.1 | 93.3 | 24.7% | 113774.4 | 2.96x |
| ds-v2-lite-3k | bf16 | 6296.7 | 80.1 | 21.2% | 16621.7 | 2.64x |

  - `ds-v3-b2-8k` breakdown (`_bwd_main_kernel` 151099.0 + `_delta_main_kernel` 715.0 +
    `_dq_land_main_kernel` 1642.3 = **153456.3 us for the 4 kernels**, plus the framework helpers
    `ZerosLike` 958.3 and `Range` 4.6 -> 154419.2 us).  The Cube redesign's `_bwd_main_kernel`
    therefore carries 97.9% of the whole call; the dev-stage figure (150.7 ms, 3.02x) reproduces.
  - acceptance per case: `metric=MFU`, `unit=cube`, `peak_used.compute_tflops=378.0`,
    `status=measured`, `value` 23.05 / 23.05 / 24.57 / 24.68 / 21.18 %, `pass=False` against the 90 %
    target (the honest efficiency, not an invalidity); the only `notes` entry is
    `report-underreports: use t_op_us`.
  - `latency_us` in the report (208.17 us for ds-v3-4k, 958.32 us for ds-v3-b2-8k) is the median of
    the *filtered* op capture (the `Range_*` helper), which is why `T_op` -- not `latency_us` -- is
    the number quoted above.

### B=2 / S=8192 / H=128 non-silent check through the integrated op

`examples/mla_dense_bwd/_bwd_kernel/evidence/s5/check_b2s8k_op.py --heads 0,127`
(log `.../evidence/s5/check_b2s8k_op.log`, exit 0) drives `MLADenseBwdOp` (integrated op ->
integrated wrapper -> integrated `_bwd_kernel`) at the production shape, bf16, causal.  It reuses
the development `--level big` ideas only: the chunked float32 forward for `o`/`lse` (the
`_stats_npu` loop, extended to also return `o`, because the op recomputes
`delta = rowsum(o * do)`) and `bk.golden_head` for the sliced fp64 reference; no large golden was
rewritten.

- structural: `dq/dk/dv` finite for both requests; non-zero counts per request
  `dq b0 201314301/201326592`, `dq b1 201314684/201326592`,
  `dk b0/b1 201326592/201326592`, `dv b0 134217728/134217728`, `dv b1 134217727/134217728`;
  request 1 key rows 0..63 are fully populated (`dk` zeros `0/1572864`, `dv` zeros `0/1048576`,
  max|.| 3.27 / 6.84) -> **no cross-request overwrite**, i.e. the 4-D per-request store window the
  wrapper passes survives integration (a 3-D packed view would have zeroed that region).
- sliced reference, `(request, head)` in `{(0,0), (0,127), (1,0), (1,127)}`, ~16.8M elements:
  **0 elements** outside `atol + rtol|ref|` (both the dev floor `atol 5e-3` and the op-test floor
  `atol 1.6e-2`, `rtol 1.6e-2`); worst gaps `dq 4.9e-3`, `dk 1.0e-2`, `dv 1.8e-2`; the worst
  ratio to the threshold anywhere is 0.51 (`dq`, b0 h0), 0.30-0.32 for `|ref| >= 0.1`.
- op-level `T_op` (2 reps, min, whole call) 154.07 ms -- matches the msprof `T_op` 154.25 ms and the
  dev-stage 150.7 ms; the baseline was 455.3 ms.
- probe artifact worth recording: comparing the same output against an fp64 golden that
  *re-derives* `lse`/`delta` (as `golden_head` does) leaves a handful of elements over the dev
  floor (`dq 5 / dk 17 / dv 9` of 16.8M, max gap 1.8e-2).  Holding `lse`/`delta` at the values the
  kernel actually consumes (the op's own) drops all three counts to 0, so the residual is the
  fp32-from-16-bit forward statistics, not the kernel -- the same discriminator the development
  `run_big` gets by overwriting `lse`/`delta` with the fp64 statistics.

### 调试历史

- none in this round: integration was idempotent-clean (wrapper untouched), correctness 10/10 and
  every msprof capture valid on the first full report.  The only non-pass item of the first report
  was the acceptance tripwire (`compute_units.json` still `vector`), a reporting-metadata staleness;
  it was **not** fixed here (out of scope) but resolved conductor-side, and this round's re-run
  ("口径修正后重跑") confirms `invalid_acceptance_count=0` / `status=passed` with better throughput
  numbers than the stale-unit case (154419.2 vs 154251.2 us is run-to-run noise).

### 修改文件清单

- 未修改任何 kernel / wrapper / tests / benchmarks / manifest / ops / workloads 文件.
- 集成包：`_bwd_kernel.py` / `_delta_kernel.py` / `_dq_land_kernel.py` + 三份 `*_DESIGN.md` +
  `__init__.py` 由 `integrate_kernel.py` 整文件覆盖为任务目录的当前（Cube）版本；
  `integration_log.md` / `integration_report.json` 更新；`history_version/` 未新增备份（无改动）。
- 新增证据（任务目录，仅新增不改既有）：
  `examples/mla_dense_bwd/_bwd_kernel/evidence/s5/{check_b2s8k_op.py, check_b2s8k_op.log,
  pytest_ops_integrated.log, report_s5_cube.log, report_s5_cube_rerun.log}`
- conductor 侧（非本 stage）：`tileops/reporting/compute_units.json`（`MLADenseBwdOp` -> `cube`）与
  `tests/test_reporting_acceptance.py` 的期望同步；本 stage 只重跑报告并更新本日志

## 验证命令

```
source .agents/platforms/toolchains/ascend/env-950pr.sh
export ASCEND_RT_VISIBLE_DEVICES=0 TMPDIR=/workspace \
       TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd
cd examples/TileOPs
/workspace/venv-ascend950/bin/python .agents/skills/add-npu-op/scripts/integrate_kernel.py \
  --meta tileops/kernels/attention/mla_dense_bwd/.migration_meta.json \
  --task-dir /workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd
/workspace/venv-ascend950/bin/python -m pytest tests/ops/test_mla_dense_bwd.py -q
/workspace/venv-ascend950/bin/python ../../mla_dense_bwd/_bwd_kernel/evidence/s5/check_b2s8k_op.py
/workspace/venv-ascend950/bin/python -m tileops.reporting.cli run --op MLADenseBwdOp --prof-mode msprof
# 口径修正（compute_units.json MLADenseBwdOp -> cube）后的最终重跑：
rm -rf /tmp/tileops_msprof_*
/workspace/venv-ascend950/bin/python -m tileops.reporting.cli run --op MLADenseBwdOp --prof-mode msprof
```

## Round 1 (pre-Cube baseline, 2026-11-23) -- retained history

- report: `reports/tileops/20261123_114002_550215_MLADenseBwdOp/` (`status=partial`,
  correctness 10/10, `invalid_profile_count=2`, `ds-v3-b2-8k` 455066.6 us via the `events` fallback)
- attempt 1 (wrapper glue): `Kernel expected 11 inputs, but 3 are provided` -> the Stage 3 factory
  already returns the compiled 11-input program, so `default_config={}` and `_compile` returns
  `self.kernel` directly (backup `history_version/mla_dense_bwd_s5_attempt1.py`).
- Round 1 also recorded that a multi-launch op cannot be filtered by one `--kernel-name`; the
  framework now sums every launched kernel into `t_op_us`, which is why this round's captures are
  all valid (`invalid_profile_count=0`).
