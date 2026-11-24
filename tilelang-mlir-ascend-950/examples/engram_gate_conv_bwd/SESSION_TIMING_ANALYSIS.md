<!-- TILEOPS_SESSION_TIMING_V1
{"schema_version": 1, "operator": "engram_gate_conv_bwd", "total_duration_s": 19614.0, "breakdown": [{"name": "Stage 0 脚手架", "duration_s": 370.0, "ratio_percent": 1.9, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 14:39:13；stage 0 累计 370.0s"}, {"name": "Stage 1 设计", "duration_s": 1317.0, "ratio_percent": 6.7, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 14:39:13；stage 1 累计 1317.0s"}, {"name": "Stage 2 检视", "duration_s": 409.0, "ratio_percent": 2.1, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 14:39:13；stage 2 累计 409.0s"}, {"name": "Stage 3 开发/真机", "duration_s": 6884.0, "ratio_percent": 35.1, "detail": "2026-11-23 09:12:19 ~ 2026-11-23 14:39:13；stage 3 累计 6884.0s"}]}
-->

# Session 耗时分析：engram_gate_conv_bwd

- session id: `ses_df85ac510ffe7KzXRIE4m13Bin`（本批 11 算子共享同一会话）；full.json: `examples/batch2/full.json`
- 本算子活动区间: 2026-11-23 09:12:19 ~ 2026-11-23 14:39:13，跨度 **19614.0s**（含与其他算子并行等待的时间）

## 阶段耗时（本算子，来自 `.task_timeline.jsonl`）

| # | 阶段 | 耗时(s) | 占本算子跨度 |
|---|---|---:|---:|
| 1 | Stage 0 脚手架 | 370.0 | 1.9% |
| 2 | Stage 1 设计 | 1317.0 | 6.7% |
| 3 | Stage 2 检视 | 409.0 | 2.1% |
| 4 | Stage 3 开发/真机 | 6884.0 | 35.1% |

## 减少耗时方案

见 `examples/batch2/SESSION_TIMING_ANALYSIS.md` 的批级方案（设备串行、多函数串行、脚手架契约、编译缓存、多 kernel 归因）。
