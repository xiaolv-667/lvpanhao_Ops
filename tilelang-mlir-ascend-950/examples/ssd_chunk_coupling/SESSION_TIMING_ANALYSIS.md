<!-- TILEOPS_SESSION_TIMING_V1
{"schema_version": 1, "operator": "ssd_chunk_coupling", "total_duration_s": 17472.0, "breakdown": [{"name": "Stage 0 脚手架", "duration_s": 271.0, "ratio_percent": 1.6, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 14:03:31；stage 0 累计 271.0s"}, {"name": "Stage 1 设计", "duration_s": 810.0, "ratio_percent": 4.6, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 14:03:31；stage 1 累计 810.0s"}, {"name": "Stage 2 检视", "duration_s": 413.0, "ratio_percent": 2.4, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 14:03:31；stage 2 累计 413.0s"}, {"name": "Stage 3 开发/真机", "duration_s": 10396.0, "ratio_percent": 59.5, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 14:03:31；stage 3 累计 10396.0s"}]}
-->

# Session 耗时分析：ssd_chunk_coupling

- session id: `ses_df85ac510ffe7KzXRIE4m13Bin`（本批 11 算子共享同一会话）；full.json: `examples/batch2/full.json`
- 本算子活动区间: 2026-11-23 09:12:19 ~ 2026-11-23 14:03:31，跨度 **17472.0s**（含与其他算子并行等待的时间）

## 阶段耗时（本算子，来自 `.task_timeline.jsonl`）

| # | 阶段 | 耗时(s) | 占本算子跨度 |
|---|---|---:|---:|
| 1 | Stage 0 脚手架 | 271.0 | 1.6% |
| 2 | Stage 1 设计 | 810.0 | 4.6% |
| 3 | Stage 2 检视 | 413.0 | 2.4% |
| 4 | Stage 3 开发/真机 | 10396.0 | 59.5% |

## 减少耗时方案

见 `examples/batch2/SESSION_TIMING_ANALYSIS.md` 的批级方案（设备串行、多函数串行、脚手架契约、编译缓存、多 kernel 归因）。
