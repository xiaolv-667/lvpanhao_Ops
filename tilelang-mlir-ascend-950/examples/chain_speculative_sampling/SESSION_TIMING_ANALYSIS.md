<!-- TILEOPS_SESSION_TIMING_V1
{"schema_version": 1, "operator": "chain_speculative_sampling", "total_duration_s": 26145.0, "breakdown": [{"name": "Stage 0 脚手架", "duration_s": 514.0, "ratio_percent": 2.0, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 16:28:04；stage 0 累计 514.0s"}, {"name": "Stage 1 设计", "duration_s": 1315.0, "ratio_percent": 5.0, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 16:28:04；stage 1 累计 1315.0s"}, {"name": "Stage 2 检视", "duration_s": 466.0, "ratio_percent": 1.8, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 16:28:04；stage 2 累计 466.0s"}, {"name": "Stage 3 开发/真机", "duration_s": 17297.0, "ratio_percent": 66.2, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 16:28:04；stage 3 累计 17297.0s"}]}
-->

# Session 耗时分析：chain_speculative_sampling

- session id: `ses_df85ac510ffe7KzXRIE4m13Bin`（本批 11 算子共享同一会话）；full.json: `examples/batch2/full.json`
- 本算子活动区间: 2026-11-23 09:12:19 ~ 2026-11-23 16:28:04，跨度 **26145.0s**（含与其他算子并行等待的时间）

## 阶段耗时（本算子，来自 `.task_timeline.jsonl`）

| # | 阶段 | 耗时(s) | 占本算子跨度 |
|---|---|---:|---:|
| 1 | Stage 0 脚手架 | 514.0 | 2.0% |
| 2 | Stage 1 设计 | 1315.0 | 5.0% |
| 3 | Stage 2 检视 | 466.0 | 1.8% |
| 4 | Stage 3 开发/真机 | 17297.0 | 66.2% |

## 减少耗时方案

见 `examples/batch2/SESSION_TIMING_ANALYSIS.md` 的批级方案（设备串行、多函数串行、脚手架契约、编译缓存、多 kernel 归因）。
