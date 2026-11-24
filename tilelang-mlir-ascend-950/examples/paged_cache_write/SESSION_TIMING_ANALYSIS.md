<!-- TILEOPS_SESSION_TIMING_V1
{"schema_version": 1, "operator": "paged_cache_write", "total_duration_s": 17024.0, "breakdown": [{"name": "Stage 0 脚手架", "duration_s": 982.0, "ratio_percent": 5.8, "detail": "2026-11-23 04:52:29 ~ 2026-11-23 09:36:13；stage 0 累计 982.0s"}, {"name": "Stage 1 设计", "duration_s": 3399.0, "ratio_percent": 20.0, "detail": "2026-11-23 04:52:29 ~ 2026-11-23 09:36:13；stage 1 累计 3399.0s"}, {"name": "Stage 2 检视", "duration_s": 915.0, "ratio_percent": 5.4, "detail": "2026-11-23 04:52:29 ~ 2026-11-23 09:36:13；stage 2 累计 915.0s"}, {"name": "Stage 3 开发/真机", "duration_s": 941.0, "ratio_percent": 5.5, "detail": "2026-11-23 04:52:29 ~ 2026-11-23 09:36:13；stage 3 累计 941.0s"}]}
-->

# Session 耗时分析：paged_cache_write

- session id: `ses_df85ac510ffe7KzXRIE4m13Bin`（本批 11 算子共享同一会话）；full.json: `examples/batch2/full.json`
- 本算子活动区间: 2026-11-23 04:52:29 ~ 2026-11-23 09:36:13，跨度 **17024.0s**（含与其他算子并行等待的时间）

## 阶段耗时（本算子，来自 `.task_timeline.jsonl`）

| # | 阶段 | 耗时(s) | 占本算子跨度 |
|---|---|---:|---:|
| 1 | Stage 0 脚手架 | 982.0 | 5.8% |
| 2 | Stage 1 设计 | 3399.0 | 20.0% |
| 3 | Stage 2 检视 | 915.0 | 5.4% |
| 4 | Stage 3 开发/真机 | 941.0 | 5.5% |

## 减少耗时方案

见 `examples/batch2/SESSION_TIMING_ANALYSIS.md` 的批级方案（设备串行、多函数串行、脚手架契约、编译缓存、多 kernel 归因）。
