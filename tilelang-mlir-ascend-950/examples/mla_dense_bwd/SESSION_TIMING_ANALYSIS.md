<!-- TILEOPS_SESSION_TIMING_V1
{"schema_version": 1, "operator": "mla_dense_bwd", "total_duration_s": 9094.0, "breakdown": [{"name": "Stage 0 脚手架", "duration_s": 445.0, "ratio_percent": 4.9, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 11:43:53；stage 0 累计 445.0s"}, {"name": "Stage 1 设计", "duration_s": 8401.0, "ratio_percent": 92.4, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 11:43:53；stage 1 累计 8401.0s"}, {"name": "Stage 2 检视", "duration_s": 2785.0, "ratio_percent": 30.6, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 11:43:53；stage 2 累计 2785.0s"}, {"name": "Stage 3 开发/真机", "duration_s": 5450.0, "ratio_percent": 59.9, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 11:43:53；stage 3 累计 5450.0s"}]}
-->

# Session 耗时分析：mla_dense_bwd

- session id: `ses_df85ac510ffe7KzXRIE4m13Bin`（本批 11 算子共享同一会话）；full.json: `examples/batch2/full.json`
- 本算子活动区间: 2026-11-23 09:12:19 ~ 2026-11-23 11:43:53，跨度 **9094.0s**（含与其他算子并行等待的时间）

## 阶段耗时（本算子，来自 `.task_timeline.jsonl`）

| # | 阶段 | 耗时(s) | 占本算子跨度 |
|---|---|---:|---:|
| 1 | Stage 0 脚手架 | 445.0 | 4.9% |
| 2 | Stage 1 设计 | 8401.0 | 92.4% |
| 3 | Stage 2 检视 | 2785.0 | 30.6% |
| 4 | Stage 3 开发/真机 | 5450.0 | 59.9% |

## 减少耗时方案

见 `examples/batch2/SESSION_TIMING_ANALYSIS.md` 的批级方案（设备串行、多函数串行、脚手架契约、编译缓存、多 kernel 归因）。
