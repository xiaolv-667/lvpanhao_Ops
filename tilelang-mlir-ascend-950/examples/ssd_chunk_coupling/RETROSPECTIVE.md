# RETROSPECTIVE -- ssd_chunk_coupling (migration-harness, op level)

> Stage 1/2/3 的逐函数复盘在 `_ssd_chunk_coupling_kernel/RETROSPECTIVE.md`；本文件只承载
> **op 级**的 Stage 5 集成复盘（集成问题不归属单个函数）。

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` 只做「复制 kernel/DESIGN + 生成 `__init__` + 重写 wrapper import + import 冒烟」，**不校验 wrapper 的 dispatch 调用形状是否与 Stage 3 工厂匹配**。scaffold 期 wrapper 按 GPU 两级工厂写死 `_k(...)(block_l, block_s, block_n)(C, B)`，而 Stage 3 把工厂收敛为一级（直接返回 `tilelang.compile(...)`），集成脚本照样报 `[ok] integration complete` + `smoke-ok`（import 冒烟不触发调用），集成态必然运行失败 | `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_coupling/ssd_chunk_coupling_kernel/integration_log.md#调试历史`；integrate 运行 stdout `[smoke] smoke-ok ['_ssd_chunk_coupling_kernel']`；修前文件 `ssd_chunk_coupling_kernel/history_version/ssd_chunk_coupling_s5_attempt1_pre_glue_fix.py` | `integrate_kernel.py` 增加一条 AST 检查：若集成 kernel 的顶层函数末尾 return 的是编译产物（无内层 `def`/闭包工厂）而 wrapper 仍有两级调用 `)(block_...)(`，则 `[error]` 或至少 `[warn]` 要求同步 wrapper dispatch | R |
| Integrate-flow | Stage 5 的 canonical 报告入口是**整文件** pytest（`metadata.test_file`），manifest 为多个算子共享同一 `tests/ops/test_mamba.py` 时，任一兄弟算子既存失败都会把本算子的 report 判 `failed` 并跳过 benchmark，算子质量与报告结论脱耦 | 未过滤 run `examples/TileOPs/reports/tileops/20261123_135355_670687_SSDChunkCouplingFwdOp/run.json`（12 例中 4 例 scan 失败、benchmark skipped）；本算子 8 例在其中逐例 passed | `tileops.reporting.cli run` 增加 `--op-filter`（按 `item.user_properties["op"]` 过滤 correctness/benchmark 用例）或规定 manifest 每算子独占测试文件；同时在 `_shared/standards/gate-and-retry.md` Stage 5 行注明共享文件的处置方式 | R |
| Integrate-flow | 集成对 **perf_opt 激活状态**没有校验：`ssd_chunk_scan` 的 wrapper 当前激活的是 `perf_opt` 导入（非 baseline），而 perf_opt 产物依赖 npuir 分支 pass key，在锁定 `ascend` 工具链上构造即崩。集成脚本只保证「有 baseline 块 + perf_opt 注释行」，不保证「baseline 处于激活态」 | `tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan.py`（`from ...perf_opt...`）；失败栈 `ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py:213` | `integrate_kernel.py` 增加 `S5-WRAPPER-SWITCH` 自检：baseline import 行不得被注释、perf_opt import 行必须是注释态；对既有算子做一次全仓巡检 | R |
| Constraint-doc | 平台未记录「同一仓库内 `ascend`（site-packages 发行版）与 `npuir`（dev 树）两套 TileLang 形态并存」的可判定边界：哪些 pass key / target 属于哪一套。集成期只能现场用 `dir(PassConfigKey)` 和 lock sha256 对照推断 | `tilelang 0.1.15` site-packages `transform/pass_config.py`（无 `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION`/`NPUIR_ENABLE_AUTO_MULTI_BUFFER`）vs 仓库 dev 树 `tilelang/transform/pass_config.py:45`（有）；`toolchain.lock.json` 5 个文件 sha256 全 OK | 在 `.agents/platforms/toolchains/ascend/knowledge/capability-gaps.md` 增条目：`ascend` 发行版缺 npuir pass key 清单 + 「dev 树无私有 TVM 时不可作为运行时」 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 已集成 SSDChunkCouplingFwdOp 的 msprof 端到端实测（cv 混合尾部 cube kernel，grid 256/128/1152）：780m fp16 12.64 µs、2p7b bf16 22.91 µs、grouped-prefill fp16 171.0 µs；roofline `GM Read+Write` 利用率 14.47 / 15.24 / 18.11 %；与 Stage 3 自测（20 / — / 154–155 µs）的差异来自 profiler 取稳态与自测含 launch+sync 的口径差 | D | `examples/TileOPs/reports/tileops/20261123_135638_847395_SSDChunkCouplingFwdOp/run.json#performance.cases` | `cd examples/TileOPs && TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/ssd_chunk_coupling /workspace/venv-ascend950/bin/python -m tileops.reporting.cli run --op SSDChunkCouplingFwdOp --prof-mode msprof` | TileLang 0.1.15 (site-packages, lock sha256 OK) / CANN 9.2.1 / Ascend950PR_9579 (dav-3510) | `.agents/platforms/toolchains/ascend/knowledge/bottlenecks.md`（与 Stage 3 的 kernel-only 常数并列，标注两种口径） |
| 集成期判定「共享测试文件的兄弟算子失败是否与本次改动有关」的三步法：①隔离复现（`pytest <file> -k <sibling> -x`，看失败点是否在本次产物之外且耗时秒级）；②工具链锁对账（`toolchain.lock.json` 逐文件 sha256）；③API 证据对账（`dir(PassConfigKey)` / 源码中 key 是否存在）。三步可把「本算子质量问题」与「环境/既存产物问题」分开，避免误消耗 attempt | P | `integration_log.md#未过滤（canonical）report 运行与既存阻塞`；隔离复现输出 `1 failed, 8 deselected in 2.53s` | `cd examples/TileOPs && /workspace/venv-ascend950/bin/python -m pytest tests/ops/test_mamba.py -k ssd_chunk_scan -x -q` | 同左 | `pattern-library`（集成/诊断方法：既存失败归因三步法） |
| 集成包 wrapper 的 `default_config` 可能是**惰性键**：本算子 `block_l/block_s/block_n` 保留为 custom_op 缓存键，但集成 kernel 用 `_tile_shape(chunk_len, d_state)` 内部定 tile（Q=256→BL=128/BK=64），改这三个键不改变性能。Stage 4 若把它们当可调 tile 会得到「调参无效」的假象 | P | `ssd_chunk_coupling_kernel/_ssd_chunk_coupling_kernel.py#_tile_shape`；`ssd_chunk_coupling.py#default_config`；`integration_log.md#口径说明` | `run.json#performance.cases[*].config` 恒为 64/64/64 而 latency 只随 shape 变化 | 同左 | `tilelang-op-optimize`（调优前置检查：先确认 config 键是否真的进入 kernel 参数） |

### Transferable Lessons（可迁移教训）

- harness 集成前先核对「Stage 3 工厂层数」与「scaffold wrapper 的 dispatch 层数」是否一致：scaffold 按 GPU 两级工厂 `_k(dims)(block_...)(inputs)` 生成，Stage 3 若收敛成一级（工厂内部定 tile、直接返回编译产物），wrapper 必须同步去掉内层 `(block_...)` 调用；import 冒烟不会暴露这类错误，必须跑一次真实 `forward` 冒烟。
- 集成期出现同文件兄弟算子失败时，先用 `-k <sibling> -x` 隔离复现（秒级）+ 对账 `toolchain.lock.json` 的 sha256，再判断归属；不要把「另一算子的 npuir 形态 pass key 在本机 `ascend` 发行版缺失」误当成本次 kernel 的精度/运行问题。
- 95x 上 `ascend`（site-packages 发行版，`target="ascend"`）与 `npuir`（dev 树，`target="npuir"`）是两套形态：`TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION` / `NPUIR_ENABLE_AUTO_MULTI_BUFFER` 只存在于 npuir 分支；在 `ascend` 环境跑包含这些 key 的 kernel 会在**构造期** `AttributeError`，与本算子数值无关。
- 报告 `Bandwidth(TB/s)` 列是 roofline 模型参考带宽常量（950PR GM 峰值 1.6 TB/s），不是实测带宽；引用性能结论时用 `latency_us` + `Ratio(%)`，需要有效带宽就自己用 `bytes/latency` 派生并标注「派生值」。
- 集成包 wrapper 的 `default_config` 不必然是 kernel 的真实可调参数：先读集成 kernel 的 tile 推导函数确认哪些键真正生效，再交给 Stage 4 调优，避免整轮实验落在惰性键上。
