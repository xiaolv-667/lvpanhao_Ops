# Integration Log -- MLAVarlenBwdOp

- meta: `examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/.migration_meta.json`
- task dir (`TILEOPS_TASK_DIR`): `/workspace/tilelang-mlir-ascend-950/examples/mla_varlen_bwd`
- platform: ascend-950pr-9579 / ascend (CANN 9.2.1, dav-3510, 28 Cube / 56 Vector), 1 x NPU 0
  (`ASCEND_RT_VISIBLE_DEVICES=0`), serial
- **this run**: re-integration + re-acceptance after the Stage 3 `_bwd_kernel` head-slab grid fix
  (`head_slab` compile-time param + `h_off int32[1]` run-time param + grid `(total_kv//64 + B) * slab`
  + tail-slab guard) and the wrapper multi-launch `_run`. No wrapper/kernel edit was needed in this
  session (0 fix attempts).

## 集成（deterministic）

`integrate_kernel.py --meta <meta> --task-dir examples/mla_varlen_bwd` -> `[ok] integration complete`,
`wrapper_rewritten: false`, `smoke-ok ['_bwd_kernel', '_delta_kernel', '_dq_land_kernel']`.

| function | integrated file | md5 (integrated == `examples/mla_varlen_bwd/{func}/{func}.py`) |
|---|---|---|
| `_bwd_kernel` | `.../mla_varlen_bwd_kernel/_bwd_kernel.py` | `625b7ec39a8313a172f4f150efb315ce` (92145 B) |
| `_delta_kernel` | `.../_delta_kernel.py` | `9ea12516fbd7491f133c89dfebdf6780` (identical, `cmp`) |
| `_dq_land_kernel` | `.../_dq_land_kernel.py` | `1fc22a38f9338522474f11fae70012bc` (identical, `cmp`) |

`design_docs` (copied by the script, verified `cmp` against the revised `examples/mla_varlen_bwd/{func}/DESIGN.md`):

- `_bwd_kernel_DESIGN.md` (222885 B, md5 `722d399c3d1866f23a89fd66d0132913`) -- refreshed from the
  revised design (the previous snapshot was the 144171 B pre-revision version)
- `_delta_kernel_DESIGN.md`, `_dq_land_kernel_DESIGN.md` (unchanged, identical)
- none missing, no `[warn]` from the script

Wrapper (`tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd.py`) **not rewritten** (the script
short-circuits on the existing `from .mla_varlen_bwd_kernel ... import`):

- baseline/perf_opt selection block intact, baseline active; extra glue import
  `build_kernel, describe, factory_config, plan_launch` from `.mla_varlen_bwd_kernel._bwd_kernel` kept
- `_run` is still the multi-launch version: `prep` (delta) -> `_main_for(total_kv)` returns
  `(main, plan)` cached per `(geometry, block_n, plan.slab)` -> `dq_acc` zeroed **once** ->
  one `main(...)` per `plan.launches` entry with `h_off[index:index+1]` (int32, device) ->
  `land(dq_acc)` after the last slab (verified by reading the file, lines 261-282)
- `ascend_mode = None` (`[mode] ... None (unchanged)`)

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261124_070330_693136_MLAVarlenBwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261124_070330_693136_MLAVarlenBwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261124_070330_693136_MLAVarlenBwdOp/report.html`
- report `status`: `partial` (CLI exit 1), reason from `run.json#summary`:
  `correctness_passed=true` (12/12, 0 failed, 0 errors), `benchmark_passed=false`,
  `msprof_case_count=4`, `profiler_fallback_count=0`, `invalid_acceptance_count=0`,
  `invalid_profile_count=4` (all four captures contain 4 distinct ops -> `profiler_valid=false`)
- correctness: **pass, 12/12**; `avg_max_abs_err=1.77e-02`; tolerances are the suite's own
  (`atol=max(5e-3, rtol)`, `rtol` 1e-3 fp16 / 1.6e-2 bf16), nothing widened.
  Also run standalone: `pytest tests/ops/test_mla_varlen_bwd.py -v` -> **12 passed in 83.71 s**
  (smoke 3 + full 5 + general_kernel + 3 contract tests).

### Extra check -- heads=128 is not silent (ds-v3-8x4k packed input, integrated op)

`/tmp/opencode/mla_s5_head128_check.py` (B=8, H=128, L=4096, D_qk=192, D_v=128, bf16; inputs from
`MLAVarlenBwdWorkload.gen_inputs`, no large reference recomputed):

```
MLAVarlenBwdKernel main pass for total_kv=32768: tiles=520 slab=32 launches=4 max_grid=16640
  blocks=66560 useful=66560 idle=0 | h[0:32) / h[32:64) / h[64:96) / h[96:128) grid=16640 idle=0
dq: finite=True per-request-nonzero=[100650231, ...] zero-heads=[] heads[0,125,126,127]=[6290688, 6291072, 6290880, 6290496]
dk: finite=True per-request-nonzero=[100663296, ...] zero-heads=[] heads[0,125,126,127]=[6291456, 6291456, 6291456, 6291456]
dv: finite=True per-request-nonzero=[67108864, ...] zero-heads=[] heads[0,125,126,127]=[4194304, 4194304, 4194304, 4194304]
head128-nonzero: PASS
```

-> all 8 requests and all 128 heads (including the tail slab 126..127) get non-zero, finite
`dq/dk/dv`; the silent-head path is gone. Plan: `slab=32`, 4 launches x 16640 blocks = 66560
blocks (<= 65535 per launch).

### Benchmark (msprof, per-workload `t_op_us` = sum of captured per-op times)

| label | dtype | `t_op_us` | `_bwd_main_kernel` | delta | dq_land | ZerosLike | achieved | MFU(vector) | acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ds-v3-8x4k | bfloat16 | 490616.4 | 483978.3 | 1405.3 | 3278.2 | 1954.5 | 29.23 TFLOP/s | 62.19% | measured, pass=false |
| ds-v3-8x4k | float16 | 490694.1 | 484048.7 | 1399.5 | 3289.1 | 1956.8 | 29.23 TFLOP/s | 62.18% | measured, pass=false |
| ds-v3-mixed | bfloat16 | 1728907.1 | 1722340.6 | 1396.0 | 3232.8 | 1937.8 | 22.11 TFLOP/s | 47.04% | measured, pass=false |
| kimi-k2-4x8k | bfloat16 | 470528.4 | 467186.4 | 725.8 | 1659.6 | 956.6 | 30.47 TFLOP/s | 64.84% | measured, pass=false |

- **`acceptance.status = "measured"` for all four cases -- none is `invalid`** (`invalid=0` in
  `acceptance_summary`). `pass=false` only because the 90% MFU target is unmet (62.2 / 47.0 / 64.8%);
  the op is vector-bound (`compute_unit=vector`, peak 47 TFLOPS, `aic 8.3% / aiv 73.5%`).
- **ds-v3-8x4k regression fixed**: 490.6 ms / 29.2 TFLOP/s now, versus 20.77 ms / 690.3 TFLOP/s in
  the previous report -- the old number exceeded the board's cube peak (378 TFLOPS) and was the
  silent-head artifact. The dev hot measurement (483.3 ms / 29.7 TFLOPS) is reproduced
  (`_bwd_main_kernel` 483.98 ms).
- ds-v3-mixed: 1728.9 ms / 22.11 TFLOP/s (before: 1727.1 ms / 22.13) -- expected, `total_kv=32256`
  gives `510 * 128 = 65280 <= 65535`, so this case was already a single legal launch before the fix;
  it is now 4 slab launches of 16320 blocks with the same work (~0.1% extra). Its per-FLOP gap vs
  8x4k (2.67x FLOPs -> 3.52x time) is request-length load imbalance (512 .. 16384 rows in one call),
  a Stage 4 topic.
- kimi-k2-4x8k: 470.5 ms / 30.47 TFLOP/s (before: 472.3 ms / 30.36) -- unchanged, as expected
  (`total_kv=32768` -> `516 * 64 = 33024`, single launch both before and after).
- **Report-table caveat (framework, record-only)**: the `Latency (us)` column shows
  `latency_us` = lower median over *all* captured per-op samples of the whole call
  (120958.6 for 8x4k = one launch sample, 1659.6 for kimi = the dq_land sample), i.e. it
  under-reports the op wall for multi-kernel/multi-launch ops. `acceptance` already derives from
  `t_op_us` and carries the note `report-underreports: use t_op_us`; read `t_op_us`
  (`run.json#performance.cases[*].msprof.t_op_us`), not `latency_us` / `result.tflops`.
- gate self-check: `statectl gate 5 --dir examples/mla_varlen_bwd --migration-dir
  examples/mla_varlen_bwd` -> `ok: true` (only the non-blocking `S5-REPORT-BENCH` warning).

## 调试历史

- attempt 0 (this session): no failure encountered -> no fix, no attempt consumed. Integration was
  byte-identical to the task-dir products on the first run, the wrapper already carried the
  multi-launch `_run`, correctness passed 12/12 on both the standalone pytest and the report, and the
  benchmark produced `t_op_us` for all four workloads.
- attempt 1 (previous session, kept for history): before the first report, a static factory-arity
  audit found the scaffold's two-level factory call (`self.kernel(block_n, block_m, num_stages)`)
  against Stage 3's one-level factories; fixed in the wrapper glue only
  (`default_config={}`, `_compile` returns `prep, self.kernel, land`, cache key
  `tuple(sorted(cfg.items()))`; backups `history_version/mla_varlen_bwd_s5_attempt0.py` /
  `_attempt1.py`), plus stale scaffold comments/docstring cleaned. Then
  `20.77 ms / 690 TFLOPS` on ds-v3-8x4k exposed the head-slab grid bug -> routed to design/kernel
  rework (not an integration fix), whose result is what this session re-integrated.

## 修改文件清单（本 session）

- 集成包（脚本写入，允许范围）：`_bwd_kernel.py` / `_bwd_kernel_DESIGN.md` /
  `_delta_kernel.py` / `_delta_kernel_DESIGN.md` / `_dq_land_kernel.py` / `_dq_land_kernel_DESIGN.md` /
  `__init__.py` / `integration_report.json`（本 log 重写）
- `history_version/`：新增 `integration_report_s5_prev.json`、`integration_log_s5_prev.md`；
  既有 `mla_varlen_bwd_s5_attempt0.py` / `_attempt1.py` / `_bwd_kernel_pre_slab_20261124.py` 保留
- 未修改：wrapper（`mla_varlen_bwd.py` 已是多 launch 版本，脚本 `wrapper_rewritten: false`）、
  `tests/`、`benchmarks/`、`tileops/manifest/`、`tileops/ops/`、`tileops/workloads/`、
  `examples/mla_varlen_bwd/` 既有产物（仅在 `RETROSPECTIVE.md` 追加 Stage 5 章节）、
  集成包内 `*_DESIGN.md` 设计内容（只被脚本按 Stage 1 产物刷新）

## 验证命令

```
source .agents/platforms/toolchains/ascend/env-950pr.sh
export TMPDIR=/workspace ASCEND_RT_VISIBLE_DEVICES=0 \
       TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/mla_varlen_bwd
cd examples/TileOPs
/workspace/venv-ascend950/bin/python .agents/skills/add-npu-op/scripts/integrate_kernel.py \
  --meta tileops/kernels/attention/mla_varlen_bwd/.migration_meta.json \
  --task-dir /workspace/tilelang-mlir-ascend-950/examples/mla_varlen_bwd
/workspace/venv-ascend950/bin/python -m pytest tests/ops/test_mla_varlen_bwd.py -v
/workspace/venv-ascend950/bin/python /tmp/opencode/mla_s5_head128_check.py
/workspace/venv-ascend950/bin/python -m tileops.reporting.cli run --op MLAVarlenBwdOp --prof-mode msprof
```

- msprof temp dirs: none left under `/tmp/tileops_msprof_*`; artifacts live under
  `reports/tileops/20261124_070330_693136_MLAVarlenBwdOp/artifacts/msprof/` (3.4 GB, `/workspace`).
