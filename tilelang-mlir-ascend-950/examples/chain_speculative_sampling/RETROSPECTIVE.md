# RETROSPECTIVE — chain_speculative_sampling (Stage 5 op 级集成)

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` 不做「wrapper 的工厂调用参数 vs Stage 3 工厂签名」一致性校验：Stage 0 scaffold 的 wrapper 仍按 GPU 工厂 7 参（`vec/threads/parts/pace`）接线、调用时传 fp64 `partial` 与 int64 seed，而 Stage 3 产品工厂是 `(batch, num_draft, vocab)` + 7 个调用实参。脚本的 import smoke 只验证名字可导入（`smoke-ok`），不调用 `forward`，于是集成「通过」但首个用例必失败，把接线工作推给 Stage 5 手工改 | `tileops/kernels/sampling/chain_speculative_sampling/chain_speculative_sampling_kernel/history_version/chain_speculative_sampling_s5_attempt1.py`（改动前：`_chain_speculative_sampling_kernel(batch, num_draft, vocab, vec, threads, parts, pace)` + `self.kernel()` + `partial` 分配）；改动后 `chain_speculative_sampling.py:156,192,198-199` | 在 `integrate_kernel.py` 的 `smoke_test` 加一条 AST/签名检查：对 `extracted_functions` 取其 `def` 形参个数，与 wrapper 中该名字的调用实参个数比对，不符时打 `[warn] factory arity mismatch: wrapper N vs kernel M`，并提示 Stage 5 需改胶水 | R |
| Info-source | report 链路对 `llama2-7b-b256-n1-float32`（B=256,N=1,V=32000）未给出 `roofline_utilization_percent`（其余 6 例均有值），`benchmark.md` 对应行的 Ratio/Bandwidth 显示 N/A，但 `profiler_valid=true`；读者无法判断是「roofline 分母缺失」还是「采集无效」 | `reports/tileops/20261123_162521_290555_ChainSpeculativeSamplingFwdOp/run.json#performance.cases[6]`（`roofline_utilization_percent: null`, `invalid_reason: null`, `profiler_valid: true`）与同目录 `benchmark.md` 该行 | 在 `tileops/reporting` 里对 `roofline_utilization_percent is None` 的 case 输出 `benchmark.md` 的 `ratio_missing_reason` 列（如 `eval_roofline` 未绑定 shape / 分母 0），避免 N/A 与「采集失败」混淆 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| Stage 0 scaffold wrapper → Stage 3 NPU 产品的固定接线差异四件套：① 工厂只按 `(batch, num_draft, vocab)` 调；② `tilelang.compile(...)` 直接返回可调用对象，不要再 `self.kernel()`；③ 输出由调用方分配（`out_idx=[]`），删掉 GPU 的 fp64 `partial` 跨块缓冲；④ int64 seed/offset 以 int32 词对传（`seed.view(torch.int32).view(-1)`）；随之删除 GPU launch 规划字段（`_plan/_refusal/_THREADS/_WARP_LANES`）与不再使用的 import | P | `tileops/kernels/sampling/chain_speculative_sampling/chain_speculative_sampling.py:156,192,198-199`；对照 `examples/chain_speculative_sampling/_chain_speculative_sampling_kernel/_chain_speculative_sampling_kernel.py:519-534`（工厂）与 `:580-594`（`_launch` 契约） | `cd examples/TileOPs && TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/chain_speculative_sampling /workspace/venv-ascend950/bin/python -m pytest tests/ops/test_sampling.py -m smoke -q`（3 passed） | TileLang ascend 0.1.15 / CANN 9.2.1 / Ascend950PR_9579 x1 / 2026-11-23 | `.agents/skills/tilelang-op-*` 集成检查单 / `examples/TileOPs/.agents/skills/add-npu-op/SKILL.md`（Stage 5 接线核对项） |
| 集成期 msprof 实测（7 workload 全有效、0 invalid/0 fallback）：latency 1245.8 / 2472.1 / 2529.2 / 2542.8 / 1427.5 / 1259.7 / 1666.3 us；roofline util 0.98 / 2.02 / 1.54 / 1.91 / 2.09 / 1.06 / N/A %（峰值 1.6 TB/s），与 Stage 3 的「约带宽 2%」自评一致，属 Stage 4 调优范围 | D | `reports/tileops/20261123_162521_290555_ChainSpeculativeSamplingFwdOp/run.json#performance.cases[*]`；`run.json#summary`（`msprof_case_count=7`, `invalid_profile_count=0`） | `cd examples/TileOPs && TILEOPS_TASK_DIR=<task> /workspace/venv-ascend950/bin/python -m tileops.reporting.cli run --op ChainSpeculativeSamplingFwdOp --prof-mode msprof` | TileLang ascend 0.1.15 / CANN 9.2.1 / Ascend950PR_9579 x1 / 2026-11-23 | `.agents/platforms/toolchains/ascend/knowledge/bottlenecks.md`（chain_speculative_sampling 集成期基线行） |
| 集成包 `_DESIGN.md` 快照与 kernel 同目录、由 `integrate_kernel.py` 从 `examples/{op_slug}/{func}/DESIGN.md` 整文件覆盖复制；wrapper 的 baseline/perf_opt 切换块首次改写即为干净单块（无重复注释、`__init__.__all__` 无多余名字），已知脚本缺陷本次未触发 | C | `chain_speculative_sampling_kernel/_chain_speculative_sampling_kernel_DESIGN.md`；`chain_speculative_sampling.py:79-92`；`chain_speculative_sampling_kernel/__init__.py` | `python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/sampling/chain_speculative_sampling/.migration_meta.json --task-dir <task>` | TileLang ascend 0.1.15 / Ascend950PR_9579 / 2026-11-23 | 待定 |

### Transferable Lessons（可迁移教训）

- **集成第一步先对齐 wrapper 与 Stage 3 工厂的调用契约**：scaffold wrapper 的工厂参数、输出缓冲、seed/offset dtype 都继承自 GPU 形态，而集成脚本只做 import 冒烟；先读 Stage 3 文件的工厂 `def` 与 `_launch`，再跑 report，能省一次必败的 9 例全量运行。
- **`tilelang.compile(...)` 在本工具链直接返回可调用对象**：wrapper 里写 `self.kernel()` 会把编译产物当 builder 再调用一次，必然报错；输出由调用方分配（`out_idx=[]`）时也不该再传 GPU 的跨块 fp64 缓冲。
- **int64 的 seed/offset 进 NPU kernel 前要转 int32 词对**（`.contiguous().view(torch.int32).view(-1)`），这是 Philox 状态在本工具链的入参契约；直接在 wrapper 里传 int64 会与 kernel 的 `(2,) int32` 形参不符。
- **集成期的性能数字只做基线锚点**：本次 util 0.97–2.10% 说明 kernel 与框架接线正确但远未调优，不要在 Stage 5 改 tiling；把它作为 Stage 4 的起点数值。
