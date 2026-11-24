# Integration Log -- ChainSpeculativeSamplingFwdOp

- meta: examples/TileOPs/tileops/kernels/sampling/chain_speculative_sampling/.migration_meta.json
- integrated:
  - `_chain_speculative_sampling_kernel` -> `tileops/kernels/sampling/chain_speculative_sampling/chain_speculative_sampling_kernel/_chain_speculative_sampling_kernel.py` (Stage 3 product, 945 lines; kernel factory + torch golden + L0/L1 suites)
- design_docs:
  - `_chain_speculative_sampling_kernel` -> `tileops/kernels/sampling/chain_speculative_sampling/chain_speculative_sampling_kernel/_chain_speculative_sampling_kernel_DESIGN.md` (copy of `examples/chain_speculative_sampling/_chain_speculative_sampling_kernel/DESIGN.md`, unmodified)
- attempts: 1

## 集成过程

1. `integrate_kernel.py --meta ... --task-dir /workspace/tilelang-mlir-ascend-950/examples/chain_speculative_sampling`
   （cwd `examples/TileOPs/`，`/workspace/venv-ascend950/bin/python`，`TILEOPS_TASK_DIR` 已设）：
   - kernel + DESIGN.md 复制、聚合 `__init__.py` 生成、wrapper import 改写为 baseline/perf_opt 切换块、import smoke `smoke-ok`
   - `ascend_mode = None`（普通赋值，unchanged）；`wrapper_rewritten=true`；未出现重复注释或多余 `__all__` 名字
2. wrapper 胶水修正（唯一的代码修改，attempt 1，备份见 `history_version/chain_speculative_sampling_s5_attempt1.py`）：
   - Stage 0 scaffold 的 wrapper 仍按 GPU 工厂接线：`_chain_speculative_sampling_kernel(batch, num_draft, vocab, vec, threads, parts, pace)` + 调用时传 fp64 `partial` 跨块缓冲、int64 seed/offset、`self.kernel()` 二次调用。
   - 集成后的 NPU 工厂签名是 `(batch, num_draft, vocab) -> compiled callable`，`out_idx=[]`，调用参数为 7 个：flatten 概率行、int32 词组 seed/offset、调用方分配的 int32 输出。
   - 修改内容：工厂按三参调用；`self._compiled_fn = self.kernel`（compile 直接返回可调用对象，不再二次 `()`）；去掉 `partial` 分配；seed/offset 以 `.contiguous().view(torch.int32).view(-1)` 传两字；删除随之失效的 GPU launch 规划（`_plan`/`_refusal`/`_VECTOR_ACCESS_BYTES`/`_WARP_LANES`/`_vector_width`/`_THREADS` 等）与不再使用的 import。
   - 未改动 kernel 计算逻辑、`tests/`、`benchmarks/`、`manifest`、`ops`、`workloads`、DESIGN 快照。

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261123_162521_290555_ChainSpeculativeSamplingFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261123_162521_290555_ChainSpeculativeSamplingFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261123_162521_290555_ChainSpeculativeSamplingFwdOp/report.html`
- correctness: **pass**，9/9 用例（`tests/ops/test_sampling.py`，manifest workload 6 型 + 65536 行分布探针 + 非向量边界对齐 + int 概率拒绝），failed 0 / skipped 0
- report_status: `passed`（CLI exit 0）
- benchmark: **pass / msprof 有效**，7 个 workload 全部 `profiler_valid=true`，`invalid_profile_count=0`，`profiler_fallback_count=0`
  | workload (B,N,V) | latency (us) | roofline util (%) |
  |---|---|---|
  | llama-8b-b16-n4 (16,4,128256) | 1245.8 | 0.98 |
  | llama-8b-b64-n4 (64,4,128256) | 2472.1 | 2.02 |
  | ds-v3-mtp-n1 (64,1,129280) | 2529.2 | 1.54 |
  | ds-v3-mtp-n3 (64,3,129280) | 2542.8 | 1.91 |
  | qwen3-235b-eagle-n5 (32,5,151936) | 1427.5 | 2.09 |
  | llama-8b-b17-n8 (17,8,128256) | 1259.7 | 1.06 |
  | llama2-7b-b256-n1 (256,1,32000) | 1666.3 | N/A（report 未给 util，profile 本身有效） |
  - 有效数值口径：`performance.cases[].latency_us` 与 `roofline_utilization_percent`（峰值带宽 1.6 TB/s 的占比）。实测 0.98%-2.09%，与 Stage 3 已知"约带宽 2%"一致；性能问题归属 Stage 4，本次只记录不修复。
  - 最终 report 是在 wrapper 胶水定稿（含 docstring 与接线）后重新运行的一次，`integration_report.json` 指向该次 run.json。

## 调试历史

- attempt 1: 症状 = 集成脚本成功后 wrapper 仍为 Stage 0 GPU 接线（工厂 7 参 / `partial` fp64 / int64 seed），直接跑测试必然 TypeError 或参数不符；分类 = wrapper 胶水错误（非 kernel 设计问题）；修复 = 仅改 wrapper glue（工厂参数、view 与 dtype、去掉 `partial`、删除失效的 GPU launch 规划）；结果 = smoke 3 例通过后全量 report 9/9 correctness + 7/7 benchmark 通过，无需 attempt 2。随后对 wrapper docstring（Part A/B 与 K6/K9 说明）做纯文档定稿，并按要求用最终代码状态重跑一次完整 report（`20261123_162521_290555_...`），结果与首次一致（9/9 + 7/7）。

## issues

- 集成脚本 `rewrite_wrapper_import` 已知缺陷本次未触发（import 块单次替换，`__all__` 正确）。
- report 对 `llama2-7b-b256-n1-float32` 未计算 `roofline_utilization_percent`（其余 6 例有值）；benchmark 只报告不修复，交由 Stage 4 / report 链路关注。
- wrapper 改动属于 scaffold→Stage 3 产品接线的必要胶水，已在设计文档外单独记录；若 Stage 4 采用 `perf_opt`，需保持同一工厂签名 `(batch, num_draft, vocab)` 与 7 参调用契约。
