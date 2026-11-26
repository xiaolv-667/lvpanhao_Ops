# Integration Log -- MLAVarlenBwdOp

- meta: `examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/.migration_meta.json`
- task dir (`TILEOPS_TASK_DIR`): `/workspace/tilelang-mlir-ascend-950/examples/mla_varlen_bwd`
- platform: ascend-950pr-9579 / ascend (CANN 9.2.1, dav-3510, 28 Cube / 56 Vector), 1 x NPU 0, serial
- **this run (2026-11-26)**: re-integration + re-acceptance after the **v5 redesign (A5, multi-buffer
  epilogue)** of `_bwd_kernel`: the query loop is `T.serial` over two statically expanded half bodies
  `F(k) -> GA(k) -> EP(k) -> GB(k-1)`, i.e. `GB` runs one iteration late beside the *next* iteration's
  `EP` and reads the **other** half's buffers (`qT/doT/pT/dsT` exist twice, `q/do/lse/delta` twice,
  `k/v/kT/L0A/L0B/L0C` and the UB half tiles stay single-version), with an odd/even drain that replays
  `GB(loop_ed-1)`. `num_stages` is now a **contract (= 2)** of the source, not a scheduler hint.
  **0 fix attempts**: no failure was encountered in this session.

## 集成（deterministic）

`integrate_kernel.py --meta <meta> --task-dir examples/mla_varlen_bwd` -> `[ok] integration complete`,
`wrapper_rewritten: false`, `wrapper_mode_updated: false`, `ascend_mode: null (unchanged)`,
`smoke-ok ['_bwd_kernel', '_delta_kernel', '_dq_land_kernel']`.

| function | integrated file | result |
|---|---|---|
| `_bwd_kernel` | `.../mla_varlen_bwd_kernel/_bwd_kernel.py` (v5, md5 `4df7899b64fdad2b5ceb0328c8d1b0e8`) | `cmp` identical to task dir |
| `_delta_kernel` | `.../_delta_kernel.py` | `cmp` identical |
| `_dq_land_kernel` | `.../_dq_land_kernel.py` | `cmp` identical |

`design_docs` (copied by the script, verified `cmp` against `examples/mla_varlen_bwd/{func}/DESIGN.md`):

- `_bwd_kernel_DESIGN.md` — refreshed from the v5 design doc (224,671 B, identical)
- `_delta_kernel_DESIGN.md`, `_dq_land_kernel_DESIGN.md` — identical
- none missing, no `[warn]` from the script

### Wrapper: **source activation flipped to baseline** (only edit made by this session)

The script left the wrapper untouched (`wrapper_rewritten: false`), but its **active** block was still
the previous generation's Stage 4 drop-in:

- `..._kernel/perf_opt/_bwd_kernel.py` = v4 Cube kernel, `_STAGES = 3`, md5 `6e2e693565b2...`
  (the v4 Stage-4 winner, produced 2026-11-25 08:21 — *before* the v5 redesign);
- `task_timeline.jsonl` shows `migration-func _bwd_kernel -> DEV_LOOP done @ 2026-11-26T02:52:46Z`,
  i.e. A5 is a **Stage 3 (design v5) product**, not a Stage 4 drop-in.

Leaving `perf_opt` active would have measured the **superseded v4 kernel** (~168.5 ms / 563.7 ms) and
silently reported the v5 gain as absent. Per the wrapper's own policy comment ("the perf_opt source
becomes active once tuned drop-in kernels land ...; the baseline (Stage 3) source is active
otherwise"), the baseline block is now active and the v4 `perf_opt` block is commented out with a note
that it is superseded until a **v5** Stage 4 winner lands. Backup:
`history_version/mla_varlen_bwd_s5_preA5.py`.

- glue contract re-checked by `inspect.signature` binding (not just the import smoke, which only
  imports): `factory_config() -> {block_n:96, block_m:64, num_stages:2, head_slab:32}`;
  `build_kernel(batch, heads, dim_qk, dim_v, is_causal, scale, dtype, total_kv, a2_fast, head_slab)`
  binds the wrapper's 8 positional args; `plan_launch(batch, total_kv, heads, *, block_n=)` binds;
  the v5 guard `cfg["num_stages"] != 2 -> ValueError` is satisfied (`2`).
- **multi-launch retained**: `prep` (delta) -> `_main_for(total_kv)` = `plan_launch(...)` +
  `build_kernel(...)` cached per `(geometry, block_n, plan.slab)` -> `dq_acc` zeroed **once** ->
  one `main(...)` per `plan.launches` entry with `h_off[index:index+1]` (int32) ->
  `land(dq_acc)` after the last slab. `ascend_mode = None`.

Launch plans of the reported workloads (`block_n=96`, `HEAD_SLAB_DEFAULT=32`; `blocks == useful`,
`idle == 0`, `launches * max_grid <= 65535`):

| workload | tiles | slab | launches | max grid | blocks | useful | idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| ds-v3-8x4k | 349 | 32 | 4 | 11,168 | 44,672 | 44,672 | 0 |
| ds-v3-mixed | 342 | 32 | 4 | 10,944 | 43,776 | 43,776 | 0 |
| kimi-k2-4x8k | 345 | 32 | 4 | 11,040 | 44,160 | 44,160 | 0 |
| smoke | 8 | 8 | 1 | 64 | 64 | 64 | 0 |

## 验证结果

- report run dir: `examples/TileOPs/reports/tileops/20261126_025737_402618_MLAVarlenBwdOp/`
- report_run_json: `examples/TileOPs/reports/tileops/20261126_025737_402618_MLAVarlenBwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261126_025737_402618_MLAVarlenBwdOp/report.md`
- report.html / benchmark.md / benchmark.json in the same directory
- `run.json#status` = **passed**; `summary`: `correctness_passed=true` (12/12),
  `benchmark_passed=true` (4/4), `invalid_acceptance_count=0`, `invalid_profile_count=0`,
  `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`, `msprof_case_count=4`
- standalone `pytest tests/ops/test_mla_varlen_bwd.py -q` -> **12 passed** (137.3 s), through the
  A5 baseline source; the report re-runs the same 12 cases (`passed_cases=12`)

### performance（msprof, kernel-only `t_op_us`；报告表 `Latency` 列是多 launch 的单次样本中位数，不用它判性能）

| workload | dtype | T_op (us) | ms | TFLOP/s | MFU (cube, 378 peak) | MFU (measured 370.5) |
|---|---|---:|---:|---:|---:|---:|
| ds-v3-8x4k | bfloat16 | 138,353.98 | 138.35 | 103.66 | 27.42% | 27.98% |
| ds-v3-8x4k | float16 | 138,298.76 | 138.30 | 103.70 | 27.43% | 27.99% |
| ds-v3-mixed | bfloat16 | 441,697.67 | 441.70 | 86.54 | 22.90% | 23.36% |
| kimi-k2-4x8k | bfloat16 | 140,302.38 | 140.30 | 102.20 | 27.04% | 27.58% |

- `acceptance.unit = cube` for all four cases; `invalid = 0`, `blocked = 0`
  (`acceptance_summary.MLAVarlenBwdOp.metrics.MFU`: best 27.43% / worst 22.90%).
- vs the Stage 3 dev measurements (A5): 8x4k 138.35 vs 139.5 (-0.8%), mixed 441.70 vs 440.2 (+0.3%),
  kimi 140.30 vs 140.4 (-0.1%) — reproduced, single-run variance.
- vs the last integrated report (v4 baseline `_STAGES=2`, 2026-11-25 06:19): 8x4k 203.55 -> 138.35
  (**-32.0%**), mixed 689.09 -> 441.70 (**-35.9%**), kimi 194.26 -> 140.30 (**-27.8%**).
  vs the v4 **adopted** `perf_opt` (`_STAGES=3`: 168.5 / 563.7 / 158.3): -17.9% / -21.6% / -11.4%,
  matching the dev's -17.2% / -21.9% / -11.3% (the caliber difference is the baseline chosen).
- breakdown (8x4k bf16): `_bwd_main_kernel` 131,692.75 us + `_dq_land_main_kernel` 3,280.24 +
  `ZerosLike` (dq_acc clear) 1,972.64 + `_delta_main_kernel` 1,408.36 = 138,353.98 us.
  mixed: `_bwd_main_kernel` 435,142 us of 441,698.
- **口径**: the acceptance rule requires 90% of peak, so `0P/4F` on MFU is expected for this op
  (previously `0P/4F` at 18.6%); `report.md` also carries `report-underreports: use t_op_us`.
- compile note: the v5 body emits `Python Z3 scheduler failed ... falling back to topological sort`
  — expected, the loop is `T.serial` over two static halves by design.

## 调试历史（若有）

- attempt 0 (this session): no failure -> no fix, no attempt consumed. Integration was byte-identical
  to the task-dir products on the first run (`wrapper_rewritten: false`), correctness 12/12 standalone
  and in the report, 8x4k non-silent check PASS, benchmark produced `t_op_us` for all four cases with
  `invalid_*` all 0. The single code edit (wrapper activation flip) is a *correctness-of-measurement*
  fix, not a failure fix: it prevents measuring the superseded v4 kernel.
- Earlier sessions (kept for history): attempt 1 fixed the scaffold's two-level factory call in the
  wrapper glue (backups `history_version/mla_varlen_bwd_s5_attempt0.py` / `_attempt1.py`); the
  head-slab grid bug (66,560 > 65,535 silently truncated on 8x4k) was routed to design/kernel rework,
  then re-integrated.

### 8x4k non-silent check (integration-side, no large reference recomputed)

`MLAVarlenBwdOp` over `MLAVarlenBwdWorkload([4096]*8, [4096]*8, H=128, 192, 128, bf16, causal=True)`
(plan: `tiles=349 slab=32 launches=4`, 4 x 11,168 blocks, 0 idle):

```
dq: finite=True per-request-nonzero=[100650231, 100648505, 100650619, 100649276,
                                    100650040, 100648888, 100650999, 100647931]
    zero-heads=[] heads[0,125,126,127]=[6290688, 6291072, 6290880, 6290496] absmax=3.96875
dk: finite=True per-request-nonzero=[100663296, 100663295, 100663295, 100663296,
                                    100663296, 100663296, 100663296, 100663296]
    zero-heads=[] heads[0,125,126,127]=[6291456, 6291456, 6291456, 6291456] absmax=3.6875
dv: finite=True per-request-nonzero=[67108864, 67108863, 67108864, 67108864,
                                    67108863, 67108862, 67108862, 67108864]
    zero-heads=[] heads[0,125,126,127]=[4194304, 4194304, 4194304, 4194304] absmax=6.5625
head128-nonzero: PASS
```

All 8 requests and all 128 heads (including the 125/126/127 tail heads and the 4th head slab) receive
non-zero, finite gradients -> the lagged-`GB` + drain structure did not drop any iteration's `dq`
atomic, and the grid/head-slab mapping still covers the whole problem.

## 修改文件清单（本 session）

- 集成包（脚本写入，允许范围）：`_bwd_kernel.py` / `_bwd_kernel_DESIGN.md` /
  `_delta_kernel.py` / `_delta_kernel_DESIGN.md` / `_dq_land_kernel.py` / `_dq_land_kernel_DESIGN.md` /
  `__init__.py`
- wrapper（本 session 手工改，胶水层 import 选择块）：`mla_varlen_bwd.py` — baseline active,
  v4 `perf_opt` commented out with rationale; **no** logic / launch / dtype change
- 本次写入：`integration_report.json` + 本 log（重写）
- `history_version/` 新增：`integration_log_s5_prev3.md`、`integration_report_s5_prev3.json`、
  `_bwd_kernel_s5_preA5.py`、`_delta_kernel_s5_preA5.py`、`_dq_land_kernel_s5_preA5.py`、
  `../history_version/mla_varlen_bwd_s5_preA5.py`
- **未修改**：`tests/`、`benchmarks/`、`tileops/manifest/`、`tileops/ops/`、`tileops/workloads/`、
  `tileops/reporting/compute_units.json`（仍为 `cube`）、
  `examples/mla_varlen_bwd/` 既有产物（仅在 `RETROSPECTIVE.md` 追加 Stage 5 章节）、
  集成包内 `*_DESIGN.md` 的设计内容（仅由脚本按 Stage 1 产物刷新）、`perf_opt/` 内容（保留不删）

## 验证命令

```
export TMPDIR=/workspace ASCEND_RT_VISIBLE_DEVICES=0 \
       TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/mla_varlen_bwd
cd examples/TileOPs
/workspace/venv-ascend950/bin/python .agents/skills/add-npu-op/scripts/integrate_kernel.py \
  --meta tileops/kernels/attention/mla_varlen_bwd/.migration_meta.json \
  --task-dir /workspace/tilelang-mlir-ascend-950/examples/mla_varlen_bwd
/workspace/venv-ascend950/bin/python -m pytest tests/ops/test_mla_varlen_bwd.py -q
/workspace/venv-ascend950/bin/python /tmp/opencode/mla_s5_head128_check.py
/workspace/venv-ascend950/bin/python -m tileops.reporting.cli run --op MLAVarlenBwdOp --prof-mode msprof
```

- msprof temp dirs: none left under `/tmp/tileops_msprof_*`; artifacts live under
  `reports/tileops/20261126_025737_402618_MLAVarlenBwdOp/artifacts/msprof/` (2.3 GB, `/workspace`).
