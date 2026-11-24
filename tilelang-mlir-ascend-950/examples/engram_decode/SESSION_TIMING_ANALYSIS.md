<!-- TILEOPS_SESSION_TIMING_V1
{"schema_version": 1, "operator": "engram_decode", "total_duration_s": 19980.0, "breakdown": [{"name": "Stage 0 脚手架", "duration_s": 372.0, "ratio_percent": 1.9, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 14:45:19；stage 0 累计 372.0s"}, {"name": "Stage 1 设计", "duration_s": 12833.0, "ratio_percent": 64.2, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 14:45:19；stage 1 累计 12833.0s"}, {"name": "Stage 2 检视", "duration_s": 999.0, "ratio_percent": 5.0, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 14:45:19；stage 2 累计 999.0s"}, {"name": "Stage 3 开发/真机", "duration_s": 3166.0, "ratio_percent": 15.8, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 14:45:19；stage 3 累计 3166.0s"}]}
-->

# Session 耗时分析：engram_decode

- session id: `ses_df85ac510ffe7KzXRIE4m13Bin`（本批 11 算子共享同一会话）；full.json: `examples/batch2/full.json`
- 本算子活动区间: 2026-11-23 09:12:19 ~ 2026-11-23 14:45:19，跨度 **19980.0s**（含与其他算子并行等待的时间）

## 阶段耗时（本算子，来自 `.task_timeline.jsonl`）

| # | 阶段 | 耗时(s) | 占本算子跨度 |
|---|---|---:|---:|
| 1 | Stage 0 脚手架 | 372.0 | 1.9% |
| 2 | Stage 1 设计 | 12833.0 | 64.2% |
| 3 | Stage 2 检视 | 999.0 | 5.0% |
| 4 | Stage 3 开发/真机 | 3166.0 | 15.8% |

## 减少耗时方案

见 `examples/batch2/SESSION_TIMING_ANALYSIS.md` 的批级方案（设备串行、多函数串行、脚手架契约、编译缓存、多 kernel 归因）。
