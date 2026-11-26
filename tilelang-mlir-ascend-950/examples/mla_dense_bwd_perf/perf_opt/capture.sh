#!/usr/bin/env bash
# Usage: capture.sh <variant-file> <tag> <case> [metrics]
# One platform capture of one candidate on one workload.
set -u
cd /workspace/tilelang-mlir-ascend-950/examples/TileOPs || exit 1
source ../mla_dense_bwd_perf/perf_opt/env.sh
impl=$1; tag=$2; case=$3; metrics=${4:-pipe}; round=${5:-round1}
"$PY" "$OPT_DIR/collect.py" --impl "$OPT_DIR/$impl" --case "$case" \
  --out "$OPT_DIR/profiles/$round/${case}_${tag}" \
  --json "$OPT_DIR/profiles/$round/${case}_${tag}.json" --metrics "$metrics" \
  > "$OPT_DIR/logs/${round}_${tag}_${case}.out" 2>&1
rc=$?
echo "[$tag/$case] rc=$rc $(grep -o '"task_duration_us": [0-9.]*' "$OPT_DIR/profiles/$round/${case}_${tag}.json" 2>/dev/null) $(grep -o '"window_ok": [a-z]*' "$OPT_DIR/profiles/$round/${case}_${tag}.json" 2>/dev/null)"
