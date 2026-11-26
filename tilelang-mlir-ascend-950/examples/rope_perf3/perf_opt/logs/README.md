# `perf_opt/logs/` — 实验过程日志说明

本次调优的 stdout/stderr 归属如下（顶层 `perf_opt/` 只放最终版本、分支版本、`opt_log.md`、
`perf_records.jsonl`、`workload_inventory.json`、`profiles/`、`logs/`、`experiments/`）：

| 内容 | 位置 |
|---|---|
| 每个候选文件的 **L0 精度** stdout/stderr | `logs/l0/<candidate>.log`（`experiments/bench_one.py` 抓取，未跑 L0 的纯性能分支在 `opt_log.md` 中注明） |
| 每次 **msprof 采集**的原始输出（`OPPROF_*/OpBasicInfo.csv`、`PipeUtilization.csv` 等） | `profiles/<phase>/<tag>_r<n>/`（即 `perf_records.jsonl` 的 `msprof_raw_path`） |
| 每次采集的 kernel 生成码（`.cce`） | `profiles/src/<module>_<workload>.cce` |
| 地板探针 / 代价分解探针的输出 | `profiles/experiments/{pcopy_*,p3_*}_r*/` |

采集命令（每次 = 一次算子调用，`--launch-count=1 --warm-up=0`）：

```bash
cd examples/rope_perf3                     # 任务目录（平台 execution.working_directory）
source .agents/platforms/toolchains/ascend/env-950pr.sh
export TMPDIR=/workspace TILELANG_CACHE_DIR=/tmp/opencode/tl-cache-opt-rope
export PYTHONPATH=<repo>/examples/TileOPs:$PYTHONPATH
/workspace/venv-ascend950/bin/python perf_opt/experiments/bench_one.py \
    --module <候选文件> --workload <workload_id> --reps <n> \
    --tag <tag> --out-root perf_opt/profiles/<phase>
# 内部：msopprof --output=<dir> --launch-count=1 --warm-up=0 --kernel-name=main
#                 --aic-metrics=BasicInfo[,PipeUtilization,...] python <child: 一次调用>
# child 与 warm-up 各自独立进程；未计时的 launch 不会落进采集窗口。
```

精度回归：

```bash
/workspace/venv-ascend950/bin/python perf_opt/rope.py --level L0|L1|L2|boundary
cd examples/TileOPs && /workspace/venv-ascend950/bin/python \
    ../rope_perf/perf_opt/experiments/wrapper_check.py --impl ../rope_perf/perf_opt/rope.py
```
