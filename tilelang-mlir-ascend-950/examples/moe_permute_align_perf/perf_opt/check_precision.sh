#!/usr/bin/env bash
# Stage 4 precision regression for one candidate module.
#   bash check_precision.sh <candidate.py> [pytest -m expr, default: all]
# Runs the frozen op test suite with the candidate injected through
# inject_site/sitecustomize.py (no tracked file is modified) and logs to
# perf_opt/logs/precision/<candidate>.log
set -u
here=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(cd -- "$here/../../TileOPs" && pwd)
cand=${1:?usage: check_precision.sh <candidate.py> [markexpr]}
marks=${2:-all}
name=$(basename "$cand" .py)
logdir="$here/logs/precision"
mkdir -p "$logdir"
log="$logdir/${name}-$(echo "$marks" | tr ' ,' '__').log"
export PYTHONPATH="$here/inject_site:${PYTHONPATH:-}"
export MOE_ALIGN_CANDIDATE="$(cd "$(dirname "$cand")" && pwd)/$(basename "$cand")"
export TMPDIR=${TMPDIR:-/workspace}
export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-/tmp/opencode/tl-cache-opt-moe}
sel=()
[ "$marks" != "all" ] && sel=(-m "$marks")
cd "$repo" || exit 1
python_bin=${PYTHON_BIN:-/workspace/venv-ascend950/bin/python}
"$python_bin" -m pytest tests/ops/test_moe_permute_align.py -q "${sel[@]}" >"$log" 2>&1
rc=$?
tail -n 1 "$log"
if [ $rc -ne 0 ]; then
  echo "[precision] FAIL $name ($marks) -> $log"
  grep -E "^FAILED|^E   " "$log" | head -10
  exit 1
fi
echo "[precision] PASS $name ($marks) -> $log"
