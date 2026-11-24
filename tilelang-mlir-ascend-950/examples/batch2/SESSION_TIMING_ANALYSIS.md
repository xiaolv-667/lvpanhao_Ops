<!-- TILEOPS_SESSION_TIMING_V1
{"schema_version": 1, "operator": "BATCH-batch2-11ops", "total_duration_s": 32752.3, "breakdown": [{"name": "Stage 0 脚手架", "duration_s": 4198.0, "ratio_percent": 1.8, "detail": "11 个算子该 Stage 的累计耗时（并行执行，非墙钟）"}, {"name": "Stage 1 设计", "duration_s": 75300.0, "ratio_percent": 31.6, "detail": "11 个算子该 Stage 的累计耗时（并行执行，非墙钟）"}, {"name": "Stage 2 检视", "duration_s": 18151.0, "ratio_percent": 7.6, "detail": "11 个算子该 Stage 的累计耗时（并行执行，非墙钟）"}, {"name": "Stage 3 开发/真机", "duration_s": 140863.0, "ratio_percent": 59.1, "detail": "11 个算子该 Stage 的累计耗时（并行执行，非墙钟）"}]}
-->

# Session 耗时分析（第二批 11 算子）

- session id: `ses_df85ac510ffe7KzXRIE4m13Bin`；full.json: `examples/batch2/full.json`
- 开始: 2026-11-23 09:03:18；结束: 2026-11-23 18:09:11；总耗时: **32752.3s（9.1 h）**

## 阶段累计（并行工作量，非墙钟）

| # | 阶段 | 累计耗时(s) | 占比 |
|---|---|---:|---:|
| 1 | Stage 0 脚手架 | 4198.0 | 1.8% |
| 2 | Stage 1 设计 | 75300.0 | 31.6% |
| 3 | Stage 2 检视 | 18151.0 | 7.6% |
| 4 | Stage 3 开发/真机 | 140863.0 | 59.1% |

## 各算子活动区间

| 算子 | 开始 | 结束 | 跨度(s) |
|---|---|---|---:|
| paged_cache_write | 2026-11-23 04:52:29 | 2026-11-23 09:36:13 | 17024.0 |
| engram_gate_conv | 2026-11-23 09:12:18 | 2026-11-23 14:34:11 | 19313.0 |
| engram_gate_conv_bwd | 2026-11-23 09:12:19 | 2026-11-23 14:39:13 | 19614.0 |
| mla_varlen_bwd | 2026-11-23 09:12:19 | 2026-11-23 18:08:57 | 32198.0 |
| rope | 2026-11-23 09:12:19 | 2026-11-23 10:09:23 | 3424.0 |
| mla_dense_bwd | 2026-11-23 09:12:19 | 2026-11-23 11:43:53 | 9094.0 |
| ssd_chunk_coupling | 2026-11-23 09:12:19 | 2026-11-23 14:03:31 | 17472.0 |
| engram_decode | 2026-11-23 09:12:19 | 2026-11-23 14:45:19 | 19980.0 |
| shared_expert_mlp | 2026-11-23 09:12:19 | 2026-11-23 15:21:44 | 22165.0 |
| chain_speculative_sampling | 2026-11-23 09:12:19 | 2026-11-23 16:28:04 | 26145.0 |
| moe_permute_align | 2026-11-23 09:18:52 | 2026-11-23 17:00:56 | 27724.0 |

## 减少耗时方案

1. **设备是唯一硬瓶颈**：11 个算子共用 1 张卡，Stage 3/5 严格串行；本批 Stage 3 累计约占一半墙钟。若增至 2 卡，可把开发与集成分成两条设备泳道（需卡资源与 benchmark 互斥约定）。
2. **多函数算子按函数串行设计**：`moe_permute_align`(3)、`mla_dense_bwd`(3)、`shared_expert_mlp`(3)、`mla_varlen_bwd`(3)、`engram_decode`(2) 共 14 个函数各自跑 1→2→3；对共享同一工厂的多函数（varlen 复用 dense），可引入「同源设计一次产出、差异章节独立评审」的合并模式，预计省 2–3 轮设计/检视。
3. **脚手架与 wrapper 契约对齐**：本批 5 次集成修复（工厂层数、split_k、UB 夹取、重复注释、测试用例矛盾）都可在 Stage 0 用 AST 自检提前拦截；每次修复平均多耗 20–40 min。
4. **编译缓存复用**：Stage 3 与 Stage 5 各自重复编译相同几何；统一 `TILELANG_CACHE_DIR` 并在集成前预热可省 10–20%。
5. **report 的多 kernel 归因**：`mla_dense` / `mla_varlen` / `moe` / `shared_expert` / `engram_decode` 的 msprof 单名过滤失败，导致重复补跑 events/整文件；建议 benchmark 侧支持多 kernel 聚合（属框架改进）。

> 说明：`Stage 5 集成` 未出现在上表——集成期事件由 `statectl migration` 写入聚合状态，不产生带 `stage` 的 timeline 事件；集成耗时只能从报告目录时间戳（见各 `integration_log.md`）近似，故未纳入机械汇总。上表「累计耗时」是**并行工作量之和**（11 个算子 3–4 路并发），不是墙钟；墙钟以本文件顶部的总耗时为准。
