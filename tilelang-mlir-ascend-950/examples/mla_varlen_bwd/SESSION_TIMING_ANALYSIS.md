<!-- TILEOPS_SESSION_TIMING_V1
{"schema_version": 1, "operator": "mla_varlen_bwd", "total_duration_s": 32198.0, "breakdown": [{"name": "Stage 0 脚手架", "duration_s": 427.0, "ratio_percent": 1.3, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 18:08:57；stage 0 累计 427.0s"}, {"name": "Stage 1 设计", "duration_s": 14851.0, "ratio_percent": 46.1, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 18:08:57；stage 1 累计 14851.0s"}, {"name": "Stage 2 检视", "duration_s": 4260.0, "ratio_percent": 13.2, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 18:08:57；stage 2 累计 4260.0s"}, {"name": "Stage 3 开发/真机", "duration_s": 48153.0, "ratio_percent": 149.6, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 18:08:57；stage 3 累计 48153.0s"}]}
-->

# Session 耗时分析：mla_varlen_bwd

- session id: `ses_df85ac510ffe7KzXRIE4m13Bin`（本批 11 算子共享同一会话）；full.json: `examples/batch2/full.json`
- 本算子活动区间: 2026-11-23 09:12:19 ~ 2026-11-23 18:08:57，跨度 **32198.0s**（含与其他算子并行等待的时间）

## 阶段耗时（本算子，来自 `.task_timeline.jsonl`）

| # | 阶段 | 耗时(s) | 占本算子跨度 |
|---|---|---:|---:|
| 1 | Stage 0 脚手架 | 427.0 | 1.3% |
| 2 | Stage 1 设计 | 14851.0 | 46.1% |
| 3 | Stage 2 检视 | 4260.0 | 13.2% |
| 4 | Stage 3 开发/真机 | 48153.0 | 149.6% |

## 减少耗时方案

见 `examples/batch2/SESSION_TIMING_ANALYSIS.md` 的批级方案（设备串行、多函数串行、脚手架契约、编译缓存、多 kernel 归因）。
