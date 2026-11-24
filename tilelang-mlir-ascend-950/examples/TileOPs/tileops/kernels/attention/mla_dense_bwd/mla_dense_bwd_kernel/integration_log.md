# Integration Log -- MLADenseBwdOp

- meta: `examples/TileOPs/tileops/kernels/attention/mla_dense_bwd/.migration_meta.json`
- task dir (`TILEOPS_TASK_DIR`): `/workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd`
- platform: ascend-950pr-9579 / ascend (CANN 9.2.1, dav-3510, 28 Cube / 56 Vector), 1 x NPU 0
- integrated (baseline, Stage 3 source, active):
  - `_bwd_kernel` -> `tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd_kernel/_bwd_kernel.py`
  - `_delta_kernel` -> `.../mla_dense_bwd_kernel/_delta_kernel.py`
  - `_dq_land_kernel` -> `.../mla_dense_bwd_kernel/_dq_land_kernel.py`
- design_docs (copied from `examples/mla_dense_bwd/{func}/DESIGN.md`):
  - `_bwd_kernel_DESIGN.md`, `_delta_kernel_DESIGN.md`, `_dq_land_kernel_DESIGN.md` (none missing)
- aggregate `__init__.py`: re-exports the three factories; import smoke `smoke-ok ['_bwd_kernel', '_delta_kernel', '_dq_land_kernel']`
- wrapper: `tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd.py` rewritten to the baseline/perf_opt
  selection block (baseline active, three `perf_opt` imports commented out); `MLADenseBwdKernel.ascend_mode = None`
  (ordinary assignment, single Kernel class -- S5-ASCEND-MODE satisfied; the three kernels' sources are all
  mode-`None` for this backend, so the hybrid/simd difference is a source-form detail, not a mode declaration)
- attempts: 1 (attempt 1 fixed the wrapper tiling-call glue; see Debug History)

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261123_114002_550215_MLADenseBwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261123_114002_550215_MLADenseBwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261123_114002_550215_MLADenseBwdOp/report.html`
- report `status`: `partial` (CLI exit code non-zero because the benchmark records are not all valid; correctness fully passed)
- correctness: **pass, 10/10** (`summary.correctness_passed=true`, `correctness_failed=0`), `avg_max_abs_err=1.32e-02`, `max_abs_err=3.12e-02` (bf16 smoke; passes the suite's `atol=max(5e-3, rtol)` + `rtol` check)
  - `smoke-bwd-bf16` 3.12e-02, `smoke-bwd-fp16` 3.91e-03, `full-bwd-s200` 1.56e-02, `full-bwd-q128-kv320` 3.91e-03,
    `full-bwd-q320-kv128` 1.56e-02, `full-bwd-noncausal` 3.91e-03, `full-bwd-sm-scale` 1.56e-02,
    `test_mla_dense_bwd_general_kernel` (dim_qk 128 / dim_v 64) 1.56e-02,
    `test_mla_dense_bwd_rejects_a_token_major_lse` pass (contract), `test_mla_dense_bwd_eval_roofline` pass (contract)
- benchmark: **partial** -- pytest benchmark 5/5 `passed`, 0 failed; `summary.benchmark_passed=false`,
  `invalid_profile_count=2`, `msprof_case_count=2`, `profiler_fallback_count=3`
  - `ds-v3-4k` bf16: msprof capture **invalid** -- `unfiltered capture contains 5 distinct ops`
    (`Range_...`, `ZerosLike_...`, `_bwd_main_kernel`, `_delta_main_kernel`, `_dq_land_main_kernel`);
    the recorded 208.0010 us is the latency of the captured `Range_` op, **not** this operator -> not usable
  - `ds-v3-4k` fp16: same invalidity, 208.3330 us -> not usable
  - `ds-v3-b2-8k` bf16: prof_mode `events` (msprof fell back), valid -> 455066.5833 us, 31.51 TFLOPS, 0.0118 TB/s
  - `kimi-k2-8k` bf16: `events`, valid -> 113774.4148 us, 31.51 TFLOPS, 0.0118 TB/s
  - `ds-v2-lite-3k` bf16: `events`, valid -> 16621.6948 us, 30.34 TFLOPS, 0.0303 TB/s
  - 遗留原因（只记录不修复，Stage 4 处理）：本 op 一次调用启动三个 kernel，`msprof_kernel_name` 未声明（wrapper
    文档已说明），msprof 无法用单一 `--kernel-name` 过滤整个调用；capture 里还含框架侧的 `Range`（构造 cu_seqlens）
    与 `ZerosLike`（dq_acc 预清零）。建议 Stage 4 要么按三个 `*_main_kernel` 分别采集求和，要么在 kernel 侧合并为单 kernel。

## 调试历史

- attempt 1
  - 症状：首次 report correctness 2/10（仅两条无设备的契约用例通过），8 条报
    `ValueError: Kernel expected 11 inputs, but 3 are provided.`
  - 分类：wrapper 胶水错误（config 传递），非 kernel / 非测试问题
  - 根因：wrapper 的 `_compile` 沿用 GPU 形态 `self.kernel(cfg["block_n"], cfg["block_m"], cfg["num_stages"])`，
    但 Stage 3 的 `_bwd_kernel(...)` 直接返回 `tilelang.compile(_bwd_main, target="ascend")` 得到的 11 输入程序
    （`q,k,v,do,lse,delta,cu_q,cu_kv,dq_acc,dk,dv`），没有第二级 tiling 调用；于是三个 int 被当成输入实参。
    `_delta_kernel`（`out_idx=[2]` -> 2 入 1 出）和 `_dq_land_kernel`（`out_idx=[1]` -> 1 入 1 出）的调用形态本来就正确。
  - 修复（仅 wrapper 胶水层，备份 `history_version/mla_dense_bwd_s5_attempt1.py`）：
    `default_config` 改为 `{}`（三个工厂的 tiling 在编译期冻结，GPU 的 block_n/block_m/num_stages 已不存在），
    `_compile` 直接用 `self.kernel` 作为主程序，`_programs_for` 改用 `tuple(sorted(cfg.items()))` 作缓存键。
  - 附带：清理 integrate_kernel.py 改写 import 后残留的 “Part A: GPU extracted factories …” 注释与模块 docstring
    的 “Pre-Stage-3 note”（与新切换块自相矛盾），只改注释/文档，无行为变化。
  - 结果：report correctness 10/10 通过；benchmark 进入 partial（见上）。

## 修改文件清单（均在允许范围内）

- `tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd.py`（wrapper 胶水：import 切换块、`default_config` / `_compile` / `_programs_for`、过期注释）
- 集成包新增：`_bwd_kernel.py` / `_delta_kernel.py` / `_dq_land_kernel.py` + 三份 `*_DESIGN.md` + `__init__.py`
  + `integration_log.md` + `integration_report.json` + `history_version/`
- 未修改：tests / benchmarks / manifest / ops / workloads / 集成包内 `*_DESIGN.md` / `examples/mla_dense_bwd/` 既有产物

## 验证命令

```
source .agents/platforms/toolchains/ascend/env-950pr.sh
export ASCEND_RT_VISIBLE_DEVICES=0 TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd
cd examples/TileOPs
/workspace/venv-ascend950/bin/python .agents/skills/add-npu-op/scripts/integrate_kernel.py \
  --meta tileops/kernels/attention/mla_dense_bwd/.migration_meta.json \
  --task-dir /workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd
/workspace/venv-ascend950/bin/python -m tileops.reporting.cli run --op MLADenseBwdOp --prof-mode msprof
```

- 首次失败的报告目录（保留）：`reports/tileops/20261123_113734_679663_MLADenseBwdOp/`
- 终态报告目录：`reports/tileops/20261123_114002_550215_MLADenseBwdOp/`
