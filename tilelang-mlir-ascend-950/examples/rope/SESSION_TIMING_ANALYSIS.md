<!-- TILEOPS_SESSION_TIMING_V1
{"schema_version": 1, "operator": "rope", "total_duration_s": 3424.0, "breakdown": [{"name": "Stage 1 设计", "duration_s": 820.0, "ratio_percent": 23.9, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 10:09:23；stage 1 累计 820.0s"}, {"name": "Stage 2 检视", "duration_s": 428.0, "ratio_percent": 12.5, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 10:09:23；stage 2 累计 428.0s"}, {"name": "Stage 3 开发/真机", "duration_s": 815.0, "ratio_percent": 23.8, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 10:09:23；stage 3 累计 815.0s"}]}
-->

# Session 耗时分析：rope

- session id: `ses_df85ac510ffe7KzXRIE4m13Bin`（本批 11 算子共享同一会话）；full.json: `examples/batch2/full.json`
- 本算子活动区间: 2026-11-23 09:12:19 ~ 2026-11-23 10:09:23，跨度 **3424.0s**（含与其他算子并行等待的时间）

## 阶段耗时（本算子，来自 `.task_timeline.jsonl`）

| # | 阶段 | 耗时(s) | 占本算子跨度 |
|---|---|---:|---:|
| 1 | Stage 1 设计 | 820.0 | 23.9% |
| 2 | Stage 2 检视 | 428.0 | 12.5% |
| 3 | Stage 3 开发/真机 | 815.0 | 23.8% |

## 减少耗时方案

见 `examples/batch2/SESSION_TIMING_ANALYSIS.md` 的批级方案（设备串行、多函数串行、脚手架契约、编译缓存、多 kernel 归因）。
