# Integration Log -- SharedExpertMLPFwdOp

- meta: `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/.migration_meta.json`
- op / slug / family: `SharedExpertMLPFwdOp` / `shared_expert_mlp` / `moe`
- TILEOPS_TASK_DIR: `/workspace/tilelang-mlir-ascend-950/examples/shared_expert_mlp`
- integrated (baseline source, Stage 3 products):
  - `_gemm_basic_kernel` -> `_gemm_basic_kernel.py` (from `examples/shared_expert_mlp/_gemm_basic_kernel/_gemm_basic_kernel.py`)
  - `splitk_reduce_kernel` -> `splitk_reduce_kernel.py` (from `examples/shared_expert_mlp/splitk_reduce_kernel/splitk_reduce_kernel.py`)
  - `_silu_mul_fused_kernel` -> `_silu_mul_fused_kernel.py` (from `examples/shared_expert_mlp/_silu_mul_fused_kernel/_silu_mul_fused_kernel.py`)
- design_docs (Stage 1 snapshots, copied unmodified by the integration script):
  - `_gemm_basic_kernel_DESIGN.md`
  - `splitk_reduce_kernel_DESIGN.md`
  - `_silu_mul_fused_kernel_DESIGN.md`
  - missing: none
- integration script: `.agents/skills/add-npu-op/scripts/integrate_kernel.py --meta ... --task-dir ...`
  - first run: `[ok] integration complete`, `ascend_mode = None` written, import smoke `smoke-ok ['_gemm_basic_kernel', '_silu_mul_fused_kernel', 'splitk_reduce_kernel']`
  - wrapper rewritten to the baseline/perf_opt selection block, but the pre-existing scaffold block was left behind -> duplicated header + duplicated `perf_opt` comment lines (known script defect, cleaned by hand in attempt 1; wrapper only)
  - `__init__.py` regenerated correctly (no bogus `__all__` names: all three are defined in their own module)
- attempts: 3 (of 5) -- attempt 3 is a verification re-run of the report after the conductor's Stage 0 fix of the ambiguous `[down]` test case; no new code change was needed in this package

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261123_151607_469010_SharedExpertMLPFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261123_151607_469010_SharedExpertMLPFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261123_151607_469010_SharedExpertMLPFwdOp/report.html`
- benchmark_json: `examples/TileOPs/reports/tileops/20261123_151607_469010_SharedExpertMLPFwdOp/benchmark.json`
- command: `tileops-report run --op SharedExpertMLPFwdOp --prof-mode msprof` (from `examples/TileOPs`, `TILEOPS_TASK_DIR=examples/shared_expert_mlp`, `ASCEND_RT_VISIBLE_DEVICES=0`)
- report status: **partial** (CLI exit 1) -- correctness is green; the benchmark stage produced 6 records but the reporter's profile-validity gate rejected all 6
- correctness: **PASS** -- 25/25 (`summary.correctness_passed=true, passed_cases=25, failed_cases=0, pass_rate=1.0`); report.md `Correctness 100.0% (25/25)`, `Avg Max Abs Error 1.97e-04`
  - every case green, including the previously blocking `test_shared_expert_mlp_rejects_an_invalid_tensor_contract[down]` (Stage 0 rewrote that case to `w_down = (2H, S)`, no assertion loosened) and all numerical rows: `matches_its_reference` (bf16 + fp16), `runs_its_shape_rows` (ragged / one-wave / separate / two-waves / narrow-expert / narrow-tiles), `splits_k_at_32_tokens` (wide-expert / narrow-model), `matches_the_manifest_workloads` (5 rows x dtypes), `keeps_the_gate_up_row_order`, `dispatches_to_its_kernel`, `eval_roofline`, `roofline_requires_a_call`
- benchmark: **ran, 6/6 cases executed, `summary.benchmark_passed=false`** -- `case_count=6, msprof_case_count=6, profiler_fallback_count=0, kernel_filter_fallback_count=0, invalid_profile_count=6`
  - rejection reason (identical for all 6): `invalid tileops profile: unfiltered capture contains 2 distinct ops`; root cause is structural, not numerical: this op is a **decomposition** (gate/up GEMM + gated activation + down GEMM, or the split-K gate/up GEMM + fused reduce), so one call launches kernels named `gemm_basic_kernel` and `main_kernel`. Without a `TILEOPS_MSPROF_KERNEL_NAME`/`--kernel-name` filter the reporter cannot attribute the capture to one op and marks the record invalid.
  - measured values (msprof, device Ascend950PR_9579; `latency_us` is the reporter's per-launch **median** over the captured launches, `min`/`max` are the shortest/longest captured launch -- i.e. **not** a chain sum):

    | label (T x H x S) | dtype | latency_us (median launch) | min / max launch us | samples | bandwidth TB/s | performance TOps/s | ratio % |
    |---|---|---|---|---|---|---|---|
    | 1x7168x18432 | bf16 | 227.4760 | 3.53 / 478.17 | 3 | 3.4850 | 3.49 | N/A |
    | 512x7168x18432 | bf16 | 716.7780 | 53.02 / 1380.39 | 3 | 1.6000 | 12.25 | 30.55 |
    | 32x7168x2048 | bf16 | 27.9040 | 8.71 / 57.60 | 4 | 3.1894 | 101.02 | N/A |
    | 32x7168x2048 | fp16 | 28.1790 | 8.64 / 57.67 | 4 | 3.1583 | 100.04 | N/A |
    | 4096x7168x2048 | bf16 | 608.4310 | 46.59 / 1222.91 | 3 | 0.3378 | 593.05 | N/A |
    | 32x5120x1536 | bf16 | 16.6750 | 4.57 / 30.69 | 4 | 2.8690 | 90.57 | N/A |

  - reading the numbers safely: `max` is the dominant cube GEMM of the call (e.g. 1222.91 us for the 4096 prefill gate/up GEMM ~240 GFLOP -> ~196 TOps/s, consistent with the Stage 3 kernel report), `min` is the elementwise activation; the median is whatever launch lands in the middle, so it must **not** be compared against the op's roofline as an op latency. `performance_tops`/`bandwidth_tb_s` are derived from the median, hence also launch-level.
  - recorded, not fixed (benchmark policy stays with the benchmark/report owner): a valid single-op msprof record for a decomposed op needs either an op-level kernel name declaration (the kernel class declares none; one name would capture only part of the chain) or sum-based aggregation across the decomposition. No source was changed for this.
- earlier failed run kept for the record: `reports/tileops/20261123_151219_181414_SharedExpertMLPFwdOp/` (24/25, the then-blocking `[down]` contract case).

## 调试历史

- attempt 1 (symptom: `[ragged]` / `[narrow-tiles]` / `[down]`, 22/25; plus wrapper glue defects found by inspection)
  - classification: wrapper 胶水层 (config/split-K/block_m 决策) + 集成脚本缺陷; one case (see below) out of Stage 5 scope
  - fixes (wrapper only, backed up to `history_version/shared_expert_mlp_s5_attempt1.py`):
    1. removed the duplicated kernel-source selection block the integration script left behind (duplicate header + duplicate `perf_opt` comment lines)
    2. `_split_k` now takes `(m, n, k, cfg)` and mirrors the GEMM's exact-tile precondition: `m < 16 -> 1`, `m > 32 -> 1`, `m % tile_m`, `n % block_n`, `k_tiles` checks; slice count kept at the Stage 3-validated 4 (whole K tiles) or 1
    3. `_gemm_program` now hands the GEMM the *effective* M tile (`_tile_m(m, block_m)` = `min(block_m, max(16, ceil16(m)))`, mirroring the kernel's `_clamp_block_m`), so the harness default `block_m = 128` works for `m = 1/32`; the split-K reduce call is still built from the very same `split_k`
    4. stale "Pre-Stage-3 scaffold" module docstring replaced with the Stage 5 statement
  - result: `[ragged]`/`[narrow-tiles]` still failed at this point -- the real cause is the K tile contract, not split-K (see attempt 2); `T = 1` (kimi-k2-t1) and the `T = 32` split-K rows started passing
- attempt 2 (symptom: `k must be divisible by block_k (a K tail corrupts live outputs): k=7384 block_k=64` and `k=1020 block_k=64`)
  - classification: wrapper 胶水层 (config 参数不适配: block_k / K padding), per `_gemm_basic_kernel_DESIGN.md` 5.4 rule 2 which delegates exactly this to the host: "必须由 host 换 block_k (64/32/16) ... 若连 16 都不能整除，则本 kernel 不适用（上游需 padding）"
  - fixes (wrapper only, backed up to `history_version/shared_expert_mlp_s5_attempt1.py` before the edit):
    1. new `_k_tile(cfg, k)`: first `block_k` on the `config / 64 / 32 / 16` ladder that divides `k`; when nothing divides it, pad the contraction to the next multiple of 16 and re-pick from the ladder (k=7384 -> 7392 with block_k=32, k=1020 -> 1024 with block_k=64)
    2. `_gemm_program` builds for `k_eff` and zero-extends both operands along K when padding is needed (`A[m, k]`, `B[n, k]`: K is the last axis); zeros contribute nothing, so this changes tile geometry only
    3. `_split_k` uses the same `_k_tile` block_k / `k_eff`
  - result: `[ragged]` and `[narrow-tiles]` pass (verified standalone and in the full report); 24/25 in the report, only `[down]` remains
- attempt 3 (conductor-ordered verification re-run, no code change in this package)
  - input: Stage 0 rewrote `tests/ops/test_fused_moe_shared_expert.py::…invalid_tensor_contract[down]` to `w_down = torch.randn(2 * hidden_size, ffn_size)` -- exactly the recommended minimal fix, no assertion loosened (the case now has only one offending tensor, so the message is unambiguous)
  - action: full single-op report re-run on the unchanged package state (wrapper + kernel copies byte-identical to attempt 2)
  - result: correctness **25/25 PASS** (`pass_rate=1.0`, `Avg Max Abs Error 1.97e-04`); benchmark executed 6/6 msprof cases but all six were rejected by the reporter's profile-validity gate (decomposed op -> 2 distinct captured kernel names), so the report is `partial`. No further code change made (benchmark is report-only at Stage 5).
- (historical) the `[down]` failure that blocked attempts 1-2, root cause OUTSIDE the Stage 5 edit scope (`tileops/ops/moe/shared_expert_mlp.py` / `tests/ops/test_fused_moe_shared_expert.py`, both Stage 0 products) -- resolved by Stage 0 in attempt 3:
  - the case set `w_down = (H, 2S)` while `w_gate_up = (2S, H)`; with `H = 2S = 64` the two shapes are square, so `SharedExpertMLPFwdOp._validate` derived `S = w_down.shape[1] = 64` and rejected `w_gate_up` first: `ValueError: w_gate_up must be [2 * S, H] = (128, 64) ... got (64, 64)`, while the test expected the regex `w_down must be`.
  - the two "rejects an invalid tensor contract" cases were contradictory under any single validation order (checking `w_gate_up` against `S` from `w_down` satisfied `[gate-up]` and broke `[down]`; checking `w_down` against `S = w_gate_up.shape[0] // 2` satisfied `[down]` and broke `[gate-up]`); the op refused the call with `ValueError` in both, only the message differed.
  - fix applied upstream: the `down` case now perturbs only `w_down`'s row count (`2H`), which trips the op's existing dedicated `w_down must be [H, S]` check. No Stage 5 file had to change.

## 遗留风险 / 备注

- Benchmark record invalidity (report-level, not numerical): this op decomposes into children named `gemm_basic_kernel` and `main_kernel`, so an unfiltered msprof capture holds 2 distinct op names and the reporter marks all 6 records invalid; the reported `latency_us` is therefore a per-launch *median*, not an op-level chain latency. A valid single-op record needs an op-level kernel-name declaration or sum-based aggregation across the decomposition -- owned by the benchmark/report side, untouched here. Nothing suggests a kernel-side numerical problem: the captured per-launch extremes agree with the Stage 3 kernel report (4096 prefill gate/up gemm 1222.91 us ~240 GFLOP -> ~196 TOps/s).
- `_split_k` deliberately never emits a 2- or 3-slice chain: the Stage 3 L1 matrix measured the split-K GEMM at 4 slices and at 1 only, so the smallest-divisor search of the GPU source was narrowed to "4 whole K tiles or none". All manifest rows are unaffected (their K tile counts are multiples of 4).
- K padding materialises zero-extended copies of both operands on the padded rows only; it is reachable only when `k % 16 != 0`, which no manifest row hits.
- `ruff check` on the wrapper reports one pre-existing `I001` (import block un-sorted) that is present in the untouched scaffold copy as well (`history_version/shared_expert_mlp_s5_pre_attempt1.py`); left alone so the selection-block comments keep their layout.
- No file under `tests/`, `benchmarks/`, `tileops/manifest/`, `tileops/ops/`, `tileops/workloads/` or `examples/shared_expert_mlp/` was modified; the only edits are `shared_expert_mlp.py` (wrapper glue) and the installed copies inside this package.
