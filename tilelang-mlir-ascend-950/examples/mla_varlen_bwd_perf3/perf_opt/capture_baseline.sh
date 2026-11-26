#!/bin/bash
# Baseline captures of the four tune workloads (one call each, full metric set).
set -u
cd "$(dirname "$0")/.." || exit 1
export TMPDIR=/workspace TILELANG_CACHE_DIR=/tmp/opencode/tl-cache-opt-mlav3
PY=/workspace/venv-ascend950/bin/python
stage=${1:-round0}
shift || true
mkdir -p "perf_opt/profiles/${stage}"
for case_id in "$@"; do
  out="perf_opt/profiles/${stage}/${case_id}_baseline"
  echo "=== ${stage} ${case_id}"
  ${PY} perf_opt/collect.py --impl baseline --case "${case_id}" --out "${out}" \
      --metrics full --warm-up 1 --json "${out}.json" > "${out}.stdout" 2>&1
  echo "rc=$? -> ${out}.json"
  grep -E "PROFILER_FAIL|NO_ROWS|Traceback" "${out}.stdout" | head -3
done
${PY} perf_opt/summarize.py perf_opt/profiles/"${stage}"/*_baseline.json
