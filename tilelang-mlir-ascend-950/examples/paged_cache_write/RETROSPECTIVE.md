## Stage 5 (integrate) Retrospective — attempt 1 — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` 的 wrapper 改写只替换 `from .{extracted_module} import ...` 这一条语句，脚手架的 scaffold 预先在 wrapper 里写好了同款 "Kernel source selection" 头注释 + `# --- perf_opt` 注释行，于是改写后文件里出现两份头注释块和两条相同的 perf_opt 注释 import，需要人工清理 | `examples/TileOPs/tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/history_version/paged_cache_write_s5_attempt1.py#L59-L87`（重复块）vs 现行 `paged_cache_write.py#L57-L72`（清理后） | 在 `examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py::rewrite_wrapper_import` 里，匹配时把紧随其前的 selection 头注释块与 `# --- perf_opt` 注释行一并吞掉（或先删除已有的 selection 块再插入） | R |
| Info-source | `run.json` 的 `performance.cases[].bandwidth_tb_s` 是 msprof roofline `GM/L2 - GM Read + Write` 条目的**分母**带宽（本机恒为 1.6 TB/s），不是实测访存带宽；把它当实测值会误判性能 | `examples/TileOPs/reports/tileops/20261123_093312_542195_PagedKVCacheWriteFwdOp/run.json`（7 个 case 的 bandwidth_tb_s 全为 1.6）；来源 `examples/TileOPs/tileops/benchmark/benchmark_base.py#L464-L470`（`"Bandwidth(TB/s)": gm["bw"]`） | 在 `examples/TileOPs/.agents/skills/add-npu-op/SKILL.md` 的 Tier 2 报告判读段注明：`bandwidth_tb_s` 是 roofline 分母，性能判读只看 `latency_us` 与 `roofline_utilization_percent` | R |
| Integrate-flow | 集成后 wrapper 模块 docstring 仍保留脚手架的 "Pre-Stage-3 note: the imported factory is still the GPU (CUDA target) implementation ... this is the expected scaffold state"，与集成后状态矛盾；integrator 的编辑范围（import/config/dtype 胶水）不覆盖 docstring，只能留待后续 | `examples/TileOPs/tileops/kernels/attention/paged_cache_write/paged_cache_write.py#L43-L49` | 在 scaffold 模板里把该段改成条件说明（"Stage 3 完成后由 integrator 删除"），或让 `integrate_kernel.py` 在改写 import 时同步删除该段落 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| paged_cache_write 端到端 msprof 基线（7 case：6.46–884.65 us，roofline 0.05%–3.17%），可作后续调优前后对比锚点 | D | `examples/TileOPs/reports/tileops/20261123_093312_542195_PagedKVCacheWriteFwdOp/run.json#performance.cases` | `cd examples/TileOPs && python -m tileops.reporting.cli run --op PagedKVCacheWriteFwdOp --prof-mode msprof` | tilelang 0.1.15 / CANN 9.2.1 / Ascend950PR_9579 x1 (dav-3510, 28 Cube / 56 Vector) | pattern-library / bottleneck-patterns |
| 纯 `T.SimtVF` 单模式 kernel 集成时 `ascend_mode=None` 即可（ambient mode）；只有同一 pytest 进程混用 Expert/Developer kernel 才需要在 Kernel 类上声明 `ascend_mode` | P | `examples/TileOPs/tileops/kernels/attention/paged_cache_write/paged_cache_write.py#L93`（`ascend_mode = None`，由 adapter 自动写入）vs `tileops/kernels/attention/multi_head_attention/multi_head_attention.py#L173`（`ascend_mode = "Expert"`）；本任务 15/15 正确性通过 | `python -m tileops.reporting.cli run --op PagedKVCacheWriteFwdOp --prof-mode msprof`（无 TILELANG_ASCEND_MODE 设置） | tilelang 0.1.15 / CANN 9.2.1 / Ascend950PR_9579 | pattern-library |
| 集成脚本改写 wrapper 时的重复注释块缺陷（R 类提案见上表第一行），出现条件是 scaffold 已预写 selection block | R | `paged_cache_write_kernel/history_version/paged_cache_write_s5_attempt1.py#L59-L87` | 对任一 scaffold 已预写 selection block 的 op 重跑 `integrate_kernel.py`，再 `grep -c "Kernel source selection" wrapper.py` 得 2 | tilelang 0.1.15 / CANN 9.2.1 | `examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py` |

### Transferable Lessons（可迁移教训）

- 集成脚本只替换 wrapper 里的那一条 `from ._gpu_extracted import ...` 语句行；若脚手架已预写 "Kernel source selection" 块，改写后会出现两份头注释与两条重复的 perf_opt 注释 import。集成后务必 `grep -c "Kernel source selection"` 核对，清理 wrapper 属于允许的胶水层编辑，但任何 wrapper 改动都必须重跑**完整** single-op report 并以新 `run.json` 作为结论。
- 判读 TileOPs 报告时：`bandwidth_tb_s` 是 msprof roofline 的分母带宽（本机恒 1.6 TB/s），不是实测值；性能只看 `latency_us` 与 `roofline_utilization_percent`。
- 只用 `T.SimtVF`（SIMT 向量化）写成的 kernel，Kernel 类保持 `ascend_mode = None` 即可；`ascend_mode` 只用于隔离同进程内的 Expert/Developer 模式冲突。
- `integrate_kernel.py` 的 `integration_mode()` 会对 Stage 3 源码做 S5 级 kernel 检查，源码不合规时脚本直接 `[error]` 退出；集成前不需要另跑静态检查。
- fp8 相关偏差（两个 `T.Parallel` 分别写 K/V、`per_thread >= 2` 编译期 guard、host 侧零 token 跳过、NaN 符号位差异按 `equal_nan=True` 比较）在框架层用例中被 `fp8-*`、`*_of_no_tokens_*`、`*_saturates_fp8_overflow` 三个用例分别覆盖并通过，说明这些偏差是语义等价的实现选择，不需要在集成期再改 kernel。
