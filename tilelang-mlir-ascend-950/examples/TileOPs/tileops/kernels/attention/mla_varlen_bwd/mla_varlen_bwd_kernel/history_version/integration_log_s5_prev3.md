# Integration Log -- MLAVarlenBwdOp

- meta: `examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/.migration_meta.json`
- task dir (`TILEOPS_TASK_DIR`): `/workspace/tilelang-mlir-ascend-950/examples/mla_varlen_bwd`
- platform: ascend-950pr-9579 / ascend (CANN 9.2.1, dav-3510, 28 Cube / 56 Vector), 1 x NPU 0
  (`ASCEND_RT_VISIBLE_DEVICES=0`), serial
- **this run (2026-11-25)**: re-integration + re-acceptance after the **Cube redesign** of
  `_bwd_kernel` (5 GEMMs on the explicit L0 path, `trans_A` reuse of `dsT_s`, `block_n=96`,
  2-tile merged epilogue writes, `num_stages=2`; head slab / `h_off` / tail-slab guard and the
  packed per-request guards preserved). Compute-unit calibration for this op changed
  `vector` -> `cube` (conductor, `tileops/reporting/compute_units.json`). **0 fix attempts**:
  no failure was encountered in this session.

## 集成（deterministic）

`integrate_kernel.py --meta <meta> --task-dir examples/mla_varlen_bwd` -> `[ok] integration complete`,
`wrapper_rewritten: false`, `smoke-ok ['_bwd_kernel', '_delta_kernel', '_dq_land_kernel']`.

| function | integrated file | result |
|---|---|---|
| `_bwd_kernel` | `.../mla_varlen_bwd_kernel/_bwd_kernel.py` (1943 lines, md5 `4262fe4b72cbf7c4106dbc40c1e91ee3`) | `cmp` identical to task dir |
| `_delta_kernel` | `.../_delta_kernel.py` | `cmp` identical |
| `_dq_land_kernel` | `.../_dq_land_kernel.py` | `cmp` identical |

`design_docs` (copied by the script, verified `cmp` against `examples/mla_varlen_bwd/{func}/DESIGN.md`):

- `_bwd_kernel_DESIGN.md` — refreshed from the Cube-redesign design doc (identical)
- `_delta_kernel_DESIGN.md`, `_dq_land_kernel_DESIGN.md` — identical
- none missing, no `[warn]` from the script

Wrapper (`tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd.py`) **not rewritten / not modified**
(the script short-circuits on the existing `from .mla_varlen_bwd_kernel ... import`):

- baseline/perf_opt selection block intact, baseline active; glue import
  `build_kernel, describe, factory_config, plan_launch` from `.mla_varlen_bwd_kernel._bwd_kernel`
  still matches the redesigned module's exports/signatures (`factory_config()` no-arg returns
  `block_n/block_m/num_stages/head_slab`; `plan_launch(batch, total_kv, heads, *, block_n=, head_slab=)`;
  `build_kernel(batch, heads, dim_qk, dim_v, is_causal, scale, dtype, total_kv, ...)`)
- **multi-launch retained**: `prep` (delta) -> `_main_for(total_kv)` = `plan_launch(...)` +
  `build_kernel(...)` cached per `(geometry, block_n, plan.slab)` -> `dq_acc` zeroed **once** ->
  one `main(...)` per `plan.launches` entry with `h_off[index:index+1]` (int32, device) ->
  `land(dq_acc)` after the last slab (read at `mla_varlen_bwd.py:261-282`)
- `ascend_mode = None` (`[mode] ... None (unchanged)`)

Launch plans of the reported workloads (`block_n=96`, `HEAD_SLAB_DEFAULT=32`):

| workload | tiles | slab | launches | max grid | idle |
|---|---:|---:|---:|---:|---:|
| ds-v3-8x4k | 349 | 32 | 4 | 11,168 | 0 |
| ds-v3-mixed | 342 | 32 | 4 | 10,944 | 0 |
| kimi-k2-4x8k | 345 | 32 | 2 | 11,040 | 0 |
| smoke | 8 | 8 | 1 | 64 | 0 |

## 验证结果

- report run dir: `examples/TileOPs/reports/tileops/20261125_061941_018389_MLAVarlenBwdOp/`
- report_run_json: `examples/TileOPs/reports/tileops/20261125_061941_018389_MLAVarlenBwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261125_061941_018389_MLAVarlenBwdOp/report.md`
- report.html: `.../report.html`, benchmark table `.../benchmark.md`
- `run.json#status` = **passed**; `summary`: `correctness_passed=true` (12/12),
  `benchmark_passed=true` (4/4), `invalid_acceptance_count=0`,
  `invalid_profile_count=0`, `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`
- standalone `pytest tests/ops/test_mla_varlen_bwd.py -q` -> **12 passed** (115.9 s), same 12 cases
  the report re-runs (5 param sets x dtypes + general kernel + 2 validation + roofline)

### performance（msprof, kernel-only `t_op_us`；报告表的 `Latency` 列对多 kernel/多 launch 算子偏低，见下）

| workload | dtype | T_op (us) | ms | TFLOP/s | MFU (cube, 378 peak) | MFU (measured 370.5) |
|---|---|---:|---:|---:|---:|---:|
| ds-v3-8x4k | bfloat16 | 203,548.70 | 203.55 | 70.46 | 18.64% | 19.02% |
| ds-v3-8x4k | float16 | 203,564.01 | 203.56 | 70.45 | 18.64% | 19.01% |
| ds-v3-mixed | bfloat16 | 689,089.97 | 689.09 | 55.51 | 14.68% | 14.97% |
| kimi-k2-4x8k | bfloat16 | 194,260.43 | 194.26 | 73.83 | 19.53% | 19.92% |

- `acceptance.unit = cube` for all four cases (`acceptance_summary.MLAVarlenBwdOp.metrics.MFU`,
  best 19.53% / worst 14.68%, 0 invalid, 0 blocked; the only note is the framework
  `report-underreports: use t_op_us`).
- breakdown (8x4k bf16): `_bwd_main_kernel` 196,908 us + `_dq_land_main_kernel` 3,280 us +
  `ZerosLike` (dq_acc clear) 1,955 us + `_delta_main_kernel` 1,405 us = 203,549 us.
  mixed: `_bwd_main_kernel` 682,542 us of 689,090 total.
- cube/vector pipe ratios from this run's artifacts (`PipeUtilization` over the `_bwd_main_kernel`
  blocks): 8x4k cube 20.1% / vec 38.1%, mixed 20.6% / 37.7%, kimi 20.6% / 37.2% — consistent with
  the dev-session 20.8% / 35.7% that motivated the `cube` calibration.
- vs the Stage 3/4 dev measurements: 8x4k 203.55 ms vs 201.6 ms (+1.0%), mixed 689.09 vs 681.5
  (+1.1%), kimi 194.26 vs 197.6 (−1.7%) — single-run variance, no regression signal.
- **口径说明**: the ~20.8% quoted by the redesign is the msprof `aic_cube_ratio` (pipe occupancy),
  not the harness MFU. The harness MFU divides roofline FLOPs by the 378 TFLOP/s cube peak, so the
  same 70.5 TFLOP/s reads as 18.6% MFU. Both refer to the same measurement; neither is a
  regression. The acceptance rule requires 90% of peak, hence `0P/4F` on MFU (expected for this op).

## 调试历史（若有）

- attempt 0 (this session): no failure -> no fix, no attempt consumed. Integration was byte-identical
  to the task-dir products on the first run (`wrapper_rewritten: false`), correctness 12/12 both
  standalone and in the report, 8x4k non-silent check PASS, benchmark produced `t_op_us` for all
  four cases with `invalid_*` all 0.
- Earlier sessions (kept for history): attempt 1 fixed the scaffold's two-level factory call in the
  wrapper glue (backups `history_version/mla_varlen_bwd_s5_attempt0.py` / `_attempt1.py`); the
  head-slab grid bug (66,560 > 65,535 silently truncated on 8x4k) was routed to design/kernel
  rework, then re-integrated.

### 8x4k non-silent check (integration-side, no large reference recomputed)

`MLAVarlenBwdOp` over `MLAVarlenBwdWorkload([4096]*8, [4096]*8, H=128, 192, 128, bf16, causal=True)`
(plan: `tiles=349 slab=32 launches=4`, 4 x 11,168 blocks, 0 idle):

```
dq: finite=True per-request-nonzero=[100650231, 100648505, 100650619, 100649276,
                                    100650040, 100648888, 100650999, 100647931]
    zero-heads=[] heads[0,125,126,127]=[6290688, 6291072, 6290880, 6290496] absmax=3.96875
dk: finite=True per-request-nonzero=[100663296, 100663295, 100663295, 100663296,
                                    100663296, 100663296, 100663296, 100663296]
    zero-heads=[] absmax=3.6875
dv: finite=True per-request-nonzero=[67108864, 67108863, 67108864, 67108864,
                                    67108863, 67108862, 67108862, 67108864]
    zero-heads=[] absmax=6.5625
head128-nonzero: PASS
```

All 8 requests and all 128 heads (including the 125/126/127 tail heads and the 4th head slab)
receive non-zero, finite gradients -> the grid/head-slab mapping still covers the whole problem
after the `block_n=96` re-tiling.

## 修改文件清单（本 session）

- 集成包（脚本写入，允许范围）：`_bwd_kernel.py` / `_bwd_kernel_DESIGN.md` /
  `_delta_kernel.py` / `_delta_kernel_DESIGN.md` / `_dq_land_kernel.py` / `_dq_land_kernel_DESIGN.md` /
  `__init__.py` / `integration_report.json`（本次写入）+ 本 log（重写）
- `history_version/` 新增：`integration_log_s5_prev2.md`、`integration_report_s5_prev2.json`、
  `_bwd_kernel_pre_cube_20261125.py`（本次重集成前的快照）；既有备份保留
- **未修改**：wrapper（脚本 `wrapper_rewritten: false`，本 session 未编辑）、`tests/`、
  `benchmarks/`、`tileops/manifest/`、`tileops/ops/`、`tileops/workloads/`、
  `examples/mla_varlen_bwd/` 既有产物（仅在 `RETROSPECTIVE.md` 追加 Stage 5 章节）、
  集成包内 `*_DESIGN.md` 的设计内容（仅由脚本按 Stage 1 产物刷新）

## 验证命令

```
source .agents/platforms/toolchains/ascend/env-950pr.sh
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
  `reports/tileops/20261125_061941_018389_MLAVarlenBwdOp/artifacts/msprof/` (2.3 GB, `/workspace`).
