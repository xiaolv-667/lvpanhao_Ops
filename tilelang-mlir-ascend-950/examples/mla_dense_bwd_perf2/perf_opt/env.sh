#!/usr/bin/env bash
# Task 4 (MLADenseBwdOp) execution context: source this, then run from examples/TileOPs.
#   source examples/mla_dense_bwd_perf/perf_opt/env.sh
_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
_repo=$(cd -- "$_here/../../.." && pwd)
source "$_repo/.agents/platforms/toolchains/ascend/env-950pr.sh" >/dev/null 2>&1
export TMPDIR=/workspace
export TILELANG_CACHE_DIR=/tmp/opencode/tl-cache-opt-mlad2
export PY=/workspace/venv-ascend950/bin/python
export PYTHONPATH="$_repo/examples/TileOPs:$_here:${PYTHONPATH}"
export OPT_DIR="$_here"
export TASK_DIR="$_here/.."
