# Integration Log -- MLAVarlenBwdOp

- meta: `examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/.migration_meta.json`
- task dir (`TILEOPS_TASK_DIR`): `/workspace/tilelang-mlir-ascend-950/examples/mla_varlen_bwd`
- platform: ascend-950pr-9579 / ascend (CANN 9.2.1, dav-3510, 28 Cube / 56 Vector), 1 x NPU 0 (`ASCEND_RT_VISIBLE_DEVICES=0`)
- integrated (baseline, Stage 3 source, active):
  - `_bwd_kernel` -> `tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd_kernel/_bwd_kernel.py`
  - `_delta_kernel` -> `.../mla_varlen_bwd_kernel/_delta_kernel.py`
  - `_dq_land_kernel` -> `.../mla_varlen_bwd_kernel/_dq_land_kernel.py`
- design_docs (copied byte-identically from `examples/mla_varlen_bwd/{func}/DESIGN.md`, verified with `cmp`):
  - `_bwd_kernel_DESIGN.md`, `_delta_kernel_DESIGN.md`, `_dq_land_kernel_DESIGN.md` (none missing, no `[warn]` from the script)
- aggregate `__init__.py`: re-exports the three factories; import smoke `smoke-ok ['_bwd_kernel', '_delta_kernel', '_dq_land_kernel']`
- wrapper: `tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd.py` rewritten to the baseline/perf_opt selection block
  (baseline active, three `perf_opt` imports commented out; exactly one occurrence of each block, no duplicated comment,
  `__all__` names all exist in the integrated modules -- the known `rewrite_wrapper_import` defect did not fire this time);
  `MLAVarlenBwdKernel.ascend_mode = None` as a plain single-line assignment in the single `Kernel` subclass
  (S5-ASCEND-MODE satisfied; `integrate_kernel.py` reported `mode None (unchanged)`, i.e. the wrapper declaration already
  matched the three integrated sources' mode, which the ascend adapter reports as `None` for all three -> hybrid
  `_bwd_kernel` (Cube `T.gemm` + vector epilogue) vs simd `_delta_kernel`/`_dq_land_kernel` is a source-form difference,
  not a mode declaration).
- attempts: 1 (a static factory-arity audit before the first report run produced one wrapper-glue fix; see Debug History)

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261123_180329_503237_MLAVarlenBwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261123_180329_503237_MLAVarlenBwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261123_180329_503237_MLAVarlenBwdOp/report.html`
  (a `latest.json` symlink/copy also exists but is not cited -- the timestamped `run.json` is authoritative)
- report `status`: `partial` (CLI exit code 1 because `summary.benchmark_passed=false`; correctness fully passed)
- correctness: **pass, 12/12** (`summary.correctness_passed=true`, `correctness_failed=0`, `errors=0`, `skipped=0`),
  `avg_max_abs_err=1.77e-02`
  - `smoke-varlen-bwd-bf16` 3.12e-02, `smoke-varlen-bwd-fp16` 1.95e-03, `smoke-varlen-bwd-single-request` 3.12e-02,
    `full-varlen-bwd-ragged` 1.56e-02, `full-varlen-bwd-q128-kv320` 1.56e-02, `full-varlen-bwd-q320-kv128` 1.56e-02,
    `full-varlen-bwd-noncausal` 3.91e-03, `full-varlen-bwd-sm-scale` 1.56e-02,
    `test_mla_varlen_bwd_general_kernel` (D_qk 192 / D_v 128) 1.56e-02,
    `test_mla_varlen_bwd_validate_inputs_accepts_prefix_offsets` 3.12e-02 (contract),
    `test_mla_varlen_bwd_rejects_a_token_major_lse` pass (contract),
    `test_mla_varlen_bwd_eval_roofline` pass (contract)
  - tolerances are the suite's own (`atol = max(5e-3, rtol)`, `rtol` = 1e-3 fp16 / 1.6e-2 bf16); nothing widened
- benchmark: **partial** -- pytest benchmark 4/4 `passed`, 0 failed; `summary.benchmark_passed=false`,
  `msprof_case_count=0`, `profiler_fallback_count=4`, `invalid_profile_count=0`
  (every case warns "requested msprof but recorded events", i.e. the anticipated single-`--kernel-name` filter failure:
  one call launches three kernels -- `_delta_main_kernel`, `_bwd_main_kernel`, `_dq_land_main_kernel` -- plus framework-side
  `Range` (cu_seqlens construction) and `ZerosLike` (dq_acc pre-zeroing). `msprof_kernel_name` is intentionally undeclared
  in the wrapper. All four events-mode records are `profiler_valid=true`, so the numbers below are usable as events-mode
  latency; they are **not** msprof kernel-scope numbers.

  | label | dtype | latency (us) | TOPS | TB/s |
  |---|---:|---:|---:|---:|
  | ds-v3-8x4k | bfloat16 | 20774.7768 | 690.32 | 0.5177 |
  | ds-v3-8x4k | float16 | 20776.9335 | 690.25 | 0.5176 |
  | ds-v3-mixed | bfloat16 | 1727128.5889 | 22.13 | 0.0061 |
  | kimi-k2-4x8k | bfloat16 | 472266.7908 | 30.36 | 0.0114 |

- 遗留原因（只记录不修复，Stage 4 处理）:
  1. msprof 无法用单一 `--kernel-name` 过滤整个调用（三 kernel + 框架侧 Range/ZerosLike）-> 全部回退 events。
     建议 Stage 4 按三个 `*_main_kernel` 分别采集求和，或在 kernel 侧合并为单 kernel。
  2. **性能异常待查（record only）**：`ds-v3-8x4k` 与 `kimi-k2-4x8k` 的 roofline FLOPs 几乎相同
     （690.32e12 x 20774.78e-6 = 1.434e13；30.36e12 x 0.4722668 = 1.434e13），但延迟相差 22.7x
     （20.77 ms vs 472.27 ms）；`ds-v3-mixed` 约 2.67x FLOPs 却慢 83x（1727.13 ms）。benchmark 协议的 warmup
     （`bench_kernel`: 5 次 untimed warmup + 3 trials x 10 repeats，取 median trial mean）已把首次编译排除在外，
     所以这不是编译耗时残留。物理上不可解释的同工作量 22x 差异是 Stage 4 的首要排查点，候选方向（未验证假设）：
     每个 request 的 key-tile 数与 grid 上界 `(total_kv // 64 + batch) * heads` 在长 request 下的负载均衡、
     `dq_acc` fp32 累加的原子写带宽、以及 events 模式在高延迟下的计时可靠性。

## 调试历史

- attempt 1（首次 report 前，静态核对工厂签名 -> 一次性修正，未进入失败-重跑循环）
  - 症状（预判，非本次 report 失败）：wrapper `_compile` 沿用 GPU 两级工厂形态
    `self.kernel(cfg["block_n"], cfg["block_m"], cfg["num_stages"])`，而 Stage 3 的 `_bwd_kernel(...)`
    已 `return tilelang.compile(_bwd_main, target="ascend")`（一级工厂，返回 11 输入程序
    `q,k,v,do,lse,delta,cu_q,cu_kv,dq_acc,dk,dv`），三个 int 会被当作输入实参 ->
    `Kernel expected 11 inputs, but 3 are provided`（与 dense 集成的已知症状同源）。
  - 分类：wrapper 胶水错误（config 传递），非 kernel / 非测试问题（按已知修复目录处理）
  - 核对证据：`_delta_kernel(batch, heads, dim_v, dense_seqlen, dtype)` -> `out_idx=[2]`（2 入 1 出）；
    `_bwd_kernel(batch, heads, dim_qk, dim_v, 0, 0, is_causal, scale, dtype, a2_fast=True)` -> 无 `out_idx` 的 11 入程序；
    `_dq_land_kernel(heads, dim_qk, scale, dtype)` -> `out_idx=[1]`（1 入 1 出）。三者均为一级工厂；
    wrapper 传给 `_delta_kernel` / `_dq_land_kernel` 的实参本来就与签名一致，只有主 kernel 多了一层调用。
  - 修复（仅 wrapper 胶水层，备份 `history_version/mla_varlen_bwd_s5_attempt0.py` 与 `_attempt1.py`）：
    `default_config` 改为 `{}`（三个工厂的 tiling 在编译期冻结，GPU 的 block_n/block_m/num_stages 已不存在），
    `_compile` 直接 `return prep, self.kernel, land`，`_programs_for` 缓存键改为 `tuple(sorted(cfg.items()))`。
  - 附带（仅注释/文档，无行为变化）：清理 `integrate_kernel.py` 改写后残留的自相矛盾注释 --
    模块 docstring 的 "Pre-Stage-3 note: ... imported factories are still the GPU (CUDA target) implementations ...
    Running forward before that rewrite fails" 改为 Stage 5 状态说明；import 块上方 "Part A: the GPU TileLang kernel
    factories extracted by extract_tl_kernel.py ..." 改为 "the three NPU TileLang kernel factories, integrated per
    function into `mla_varlen_bwd_kernel/` (Stage 5)"。
  - 结果：首次 report correctness 12/12 通过；benchmark 进入 partial（见上）。无第 2 次 attempt。
- `_dq_land_kernel` 的编译后生成码断言（`vlds_norm|vmuls|vcvt` 与 `vsts_pack_b32|vsts_norm`）随 Stage 3 源码原样集成，
  在 report 的 12 个用例中被实际触发并通过，未绕过、未放宽。

## 修改文件清单（均在允许范围内）

- `tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd.py`
  （wrapper 胶水：import 切换块、`default_config` / `_compile` / `_programs_for`、过期注释与 docstring）
- 集成包新增：`_bwd_kernel.py` / `_delta_kernel.py` / `_dq_land_kernel.py` + 三份 `*_DESIGN.md` + `__init__.py`
  + `integration_log.md` + `integration_report.json` + `history_version/`（两份 attempt 备份）
- 未修改：`tests/`、`benchmarks/`、`tileops/manifest/`、`tileops/ops/`、`tileops/workloads/`、
  集成包内 `*_DESIGN.md`（与 `examples/mla_varlen_bwd/{func}/DESIGN.md` `cmp` 一致）、
  `examples/mla_varlen_bwd/` 既有产物、`_mla_varlen_bwd_kernels.py`（GPU 抽取件，保留为迁移记录）

## 验证命令

```
source .agents/platforms/toolchains/ascend/env-950pr.sh
export ASCEND_RT_VISIBLE_DEVICES=0 TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/mla_varlen_bwd
cd examples/TileOPs
/workspace/venv-ascend950/bin/python .agents/skills/add-npu-op/scripts/integrate_kernel.py \
  --meta tileops/kernels/attention/mla_varlen_bwd/.migration_meta.json \
  --task-dir /workspace/tilelang-mlir-ascend-950/examples/mla_varlen_bwd
/workspace/venv-ascend950/bin/python -m tileops.reporting.cli run --op MLAVarlenBwdOp --prof-mode msprof
```

- 唯一一次 report 运行的目录即终态目录：`reports/tileops/20261123_180329_503237_MLAVarlenBwdOp/`
- `statectl gate 5 --dir examples/mla_varlen_bwd --migration-dir examples/mla_varlen_bwd` -> `ok: true`
  （首次自检仅报本 log / report 两份工件尚未写出，写出后通过）
