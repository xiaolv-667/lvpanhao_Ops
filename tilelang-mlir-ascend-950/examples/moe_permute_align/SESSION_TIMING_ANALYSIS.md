<!-- TILEOPS_SESSION_TIMING_V1
{"schema_version": 1, "operator": "moe_permute_align", "total_duration_s": 27724.0, "breakdown": [{"name": "Stage 1 设计", "duration_s": 5345.0, "ratio_percent": 19.3, "detail": "2026-11-23 09:18:52 ~ 2026-11-23 17:00:56；stage 1 累计 5345.0s"}, {"name": "Stage 2 检视", "duration_s": 5835.0, "ratio_percent": 21.0, "detail": "2026-11-23 09:18:52 ~ 2026-11-23 17:00:56；stage 2 累计 5835.0s"}, {"name": "Stage 3 开发/真机", "duration_s": 21436.0, "ratio_percent": 77.3, "detail": "2026-11-23 09:18:52 ~ 2026-11-23 17:00:56；stage 3 累计 21436.0s"}]}
-->

# Session 耗时分析：moe_permute_align

- session id: `ses_df85ac510ffe7KzXRIE4m13Bin`（本批 11 算子共享同一会话）；full.json: `examples/batch2/full.json`
- 本算子活动区间: 2026-11-23 09:18:52 ~ 2026-11-23 17:00:56，跨度 **27724.0s**（含与其他算子并行等待的时间）

## 阶段耗时（本算子，来自 `.task_timeline.jsonl`）

| # | 阶段 | 耗时(s) | 占本算子跨度 |
|---|---|---:|---:|
| 1 | Stage 1 设计 | 5345.0 | 19.3% |
| 2 | Stage 2 检视 | 5835.0 | 21.0% |
| 3 | Stage 3 开发/真机 | 21436.0 | 77.3% |

## 减少耗时方案

见 `examples/batch2/SESSION_TIMING_ANALYSIS.md` 的批级方案（设备串行、多函数串行、脚手架契约、编译缓存、多 kernel 归因）。
