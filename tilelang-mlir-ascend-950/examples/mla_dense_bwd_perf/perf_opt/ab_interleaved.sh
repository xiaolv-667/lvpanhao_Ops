#!/usr/bin/env bash
# Interleaved A/B/A/B protocol for a <5% difference (skill T-2), on the platform
# metric T_op (sum of a call's launches) -- ab_test.py's single-kernel-name
# wrapper cannot sum a 5-launch op, so the same protocol is applied here.
set -u
cd /workspace/tilelang-mlir-ascend-950/examples/TileOPs || exit 1
source ../mla_dense_bwd_perf/perf_opt/env.sh
case_id=$1; shift
pairs=${PAIRS:-3}
for p in $(seq 1 $pairs); do
  if [ $((p % 2)) -eq 1 ]; then order="baseline c9"; else order="c9 baseline"; fi
  for arm in $order; do
    if [ "$arm" = baseline ]; then
      impl="$OPT_DIR/MLADenseBwdOp_baseline_stage3.py"
    else
      impl="$OPT_DIR/MLADenseBwdOp_round1_c9.py"
    fi
    out="$OPT_DIR/profiles/ab/${case_id}_${arm}_p${p}"
    "$PY" "$OPT_DIR/collect.py" --impl "$impl" --case "$case_id" \
      --out "$out" --json "$out.json" --metrics pipe > "$OPT_DIR/logs/ab_${case_id}_${arm}_p${p}.out" 2>&1
    echo "p$p $arm $(python3 -c "import json;d=json.load(open('$out.json'));print('%.1f'%d['task_duration_us'],d['window_ok'])" 2>/dev/null)"
  done
done
