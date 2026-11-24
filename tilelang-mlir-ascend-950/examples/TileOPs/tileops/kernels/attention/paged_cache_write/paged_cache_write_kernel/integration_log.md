# Integration Log -- PagedKVCacheWriteFwdOp

- meta: examples/TileOPs/tileops/kernels/attention/paged_cache_write/.migration_meta.json
- task dir (TILEOPS_TASK_DIR): /workspace/tilelang-mlir-ascend-950/examples/paged_cache_write
- platform: ascend-950pr-9579 / ascend (CANN 9.2.1, GCC 12.3.0, dav-3510, 28 Cube / 56 Vector, 1x NPU card 0)
- interpreter: /workspace/venv-ascend950/bin/python
- integrated: `_paged_cache_write_kernel` -> `paged_cache_write_kernel/_paged_cache_write_kernel.py`
  (byte-identical to `examples/paged_cache_write/_paged_cache_write_kernel/_paged_cache_write_kernel.py`, verified with `diff`)
- design_docs: `_paged_cache_write_kernel` -> `paged_cache_write_kernel/_paged_cache_write_kernel_DESIGN.md`
  (copy of `examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md`, verified identical; not edited)
- aggregated `__init__.py`: re-exports `_paged_cache_write_kernel` (`__all__` = same), written by `integrate_kernel.py`
- wrapper: `tileops/kernels/attention/paged_cache_write/paged_cache_write.py`
  - import rewritten to the baseline/perf_opt kernel-source selection block, baseline (`.paged_cache_write_kernel`) active,
    `# from .paged_cache_write_kernel.perf_opt._paged_cache_write_kernel import ...` kept commented for Stage 4
  - `ascend_mode = None` inserted by the script (`adapter.integration_mode` -> None for this source, i.e. ambient mode; the
    kernel uses `T.SimtVF(threads=128)`, which is the SIMT form and needs no scoped override)
- attempts: 1 (integration + report passed on the first run; no debug loop entered)

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261123_093312_542195_PagedKVCacheWriteFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261123_093312_542195_PagedKVCacheWriteFwdOp/report.md`
  (`report.html`, `benchmark.json`, `benchmark.md`, `pytest/*.xml`, `artifacts/msprof/*` alongside)
- command: `(cd examples/TileOPs && python -m tileops.reporting.cli run --op PagedKVCacheWriteFwdOp --prof-mode msprof)`, CLI exit 0
- report_status: `passed` (not partial; `run.json.summary.correctness_passed=true`, `benchmark_passed=true`, `warnings=[]`)
- correctness: **pass, 15/15** (`correctness_tests=15`, `correctness_failed=0`, `pass_rate=1.0`)
  - `test_paged_kv_cache_write_scatters_the_named_rows` x6 (fp16 odd-page/skip/page-edge, bf16 one-kv-head last row,
    fp8-bf16 scaled, fp8-fp16 page boundary, bf16 decode-step, fp16 prefill sweep)
  - `..._saturates_fp8_overflow`, `..._skips_every_token`, `..._of_no_tokens_writes_nothing`,
    `..._refuses_an_invalid_tensor_contract` x4, `..._dispatches_to_its_kernel`, `..._eval_roofline`
  - every comparison case reports `max_abs_err = 0.00e+00` (byte-exact, `atol=rtol=0`, `equal_nan=True`)
- benchmark: **pass, 7/7 valid msprof profiles** (`invalid_profile_count=0`, `profiler_fallback_count=0`,
  `kernel_filter_fallback_count=0`), measured latency per case:

  | label | input / cache dtype | tokens | latency (us) | roofline ratio |
  |---|---|---:|---:|---:|
  | odd-tokens-and-page-float16 | fp16 / fp16 | 7 | 6.463 | 0.05% |
  | llama-8b-decode-bfloat16 | bf16 / bf16 | 64 | 15.218 | 1.38% |
  | llama-8b-decode-float16 | fp16 / fp16 | 64 | 15.418 | 1.37% |
  | gpt-oss-120b-bfloat16 | bf16 / bf16 | 256 | 25.768 | 1.68% |
  | llama-8b-prefill-bfloat16 | bf16 / bf16 | 8192 | 884.653 | 3.17% |
  | qwen3-235b-fp8-fp16 | fp16 / fp8_e4m3fn | 1024 | 78.891 | 2.18% |
  | qwen3-235b-fp8-bf16 | bf16 / fp8_e4m3fn | 4096 | 267.690 | 2.48% |

  - profiled kernels captured correctly: `_16bit_kernel` (5 cases) / `_fp8_kernel` (2 cases), `sample_count=1` each.
  - reported `bandwidth_tb_s=1.6` is the platform-computed peak used for the ratio, not a measured value; the ratio column
    (0.05%–3.17%) is the only meaningful performance signal. Performance is recorded, not tuned, in this stage.

## 调试历史

- attempt 1: no failure -- integration script smoke `smoke-ok ['_paged_cache_write_kernel']`, first full report `status=passed`.
- attempt 1 (cosmetic, verified by a second full report): `integrate_kernel.py` replaced only the wrapper's
  `from ._paged_k_v_cache_write_fwd_kernels import ...` line with the full generated selection block, so the scaffold's
  pre-written header block and perf_opt comment line stayed behind and the wrapper ended up with a duplicated comment block
  (and two identical commented perf_opt imports). Removed the duplicate (backup:
  `paged_cache_write_kernel/history_version/paged_cache_write_s5_attempt1.py`) and re-ran the complete report:
  `20261123_093312_542195_PagedKVCacheWriteFwdOp`, still `status=passed`, correctness 15/15, benchmark 7/7 valid.
  The first run (`20261123_093016_709274_PagedKVCacheWriteFwdOp`) is kept as history; the final record points at the second.

## issues

- none functional. Design/implementation deviations carried over from Stage 3 (two `T.Parallel` fp8 loops, fp8
  `per_thread >= 2` compile-time guard, host-side zero-token skip, NaN sign bit normalised by the device e4m3 cast and
  compared with `equal_nan=True`) all hold under the framework tests and are exercised by `fp8-*`, `*_of_no_tokens_*` and
  `*_saturates_fp8_overflow` respectively.
- The wrapper module docstring still carries the scaffold's "Pre-Stage-3 note: the imported factory is still the GPU
  (CUDA target) implementation ... this is the expected scaffold state" paragraph, which is stale now that the NPU factory
  is integrated. It is documentation inside an otherwise-frozen wrapper scaffold text, so it was left untouched; only the
  import block was edited (glue layer).
