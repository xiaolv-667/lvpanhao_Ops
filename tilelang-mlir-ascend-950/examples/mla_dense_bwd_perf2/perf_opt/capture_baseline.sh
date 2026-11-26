#!/usr/bin/env bash
# Baseline capture of every tune workload (run 0).  Rounds are serial: one card.
set -u
_self=$(cd -- "$(dirname -- "$0")" && pwd)
cd /workspace/tilelang-mlir-ascend-950/examples/TileOPs || exit 1
source "$_self/env.sh"
for case in "$@"; do
  echo "=== $case $(date -u +%H:%M:%S)"
  "$PY" "$OPT_DIR/collect.py" --impl baseline --case "$case" \
    --out "$OPT_DIR/profiles/round0/${case}_baseline" \
    --json "$OPT_DIR/profiles/round0/${case}_baseline.json" --metrics full \
    > "$OPT_DIR/logs/round0_${case}.out" 2>&1
  echo "rc=$? $(grep -o '"task_duration_us": [0-9.]*' "$OPT_DIR/profiles/round0/${case}_baseline.json" 2>/dev/null)"
done
