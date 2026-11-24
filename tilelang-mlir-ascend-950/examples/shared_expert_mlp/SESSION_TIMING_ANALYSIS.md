<!-- TILEOPS_SESSION_TIMING_V1
{"schema_version": 1, "operator": "shared_expert_mlp", "total_duration_s": 22165.0, "breakdown": [{"name": "Stage 0 脚手架", "duration_s": 493.0, "ratio_percent": 2.2, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 15:21:44；stage 0 累计 493.0s"}, {"name": "Stage 1 设计", "duration_s": 25103.0, "ratio_percent": 113.3, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 15:21:44；stage 1 累计 25103.0s"}, {"name": "Stage 2 检视", "duration_s": 1180.0, "ratio_percent": 5.3, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 15:21:44；stage 2 累计 1180.0s"}, {"name": "Stage 3 开发/真机", "duration_s": 19641.0, "ratio_percent": 88.6, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 15:21:44；stage 3 累计 19641.0s"}]}
-->

# Session 耗时分析：shared_expert_mlp

- session id: `ses_df85ac510ffe7KzXRIE4m13Bin`（本批 11 算子共享同一会话）；full.json: `examples/batch2/full.json`
- 本算子活动区间: 2026-11-23 09:12:19 ~ 2026-11-23 15:21:44，跨度 **22165.0s**（含与其他算子并行等待的时间）

## 阶段耗时（本算子，来自 `.task_timeline.jsonl`）

| # | 阶段 | 耗时(s) | 占本算子跨度 |
|---|---|---:|---:|
| 1 | Stage 0 脚手架 | 493.0 | 2.2% |
| 2 | Stage 1 设计 | 25103.0 | 113.3% |
| 3 | Stage 2 检视 | 1180.0 | 5.3% |
| 4 | Stage 3 开发/真机 | 19641.0 | 88.6% |

## 减少耗时方案

见 `examples/batch2/SESSION_TIMING_ANALYSIS.md` 的批级方案（设备串行、多函数串行、脚手架契约、编译缓存、多 kernel 归因）。
