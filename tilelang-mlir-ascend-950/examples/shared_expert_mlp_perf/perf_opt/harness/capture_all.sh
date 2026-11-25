#!/usr/bin/env bash
# Baseline capture for every tune workload (Stage 4, round 0).
set -u
cd /workspace/tilelang-mlir-ascend-950
source .agents/platforms/toolchains/ascend/env-950pr.sh >/dev/null 2>&1
export TMPDIR=/workspace
export TILELANG_CACHE_DIR=/tmp/opencode/tl-cache-opt-shared
mkdir -p "$TILELANG_CACHE_DIR"
PY=/workspace/venv-ascend950/bin/python
cd examples/shared_expert_mlp_perf/perf_opt/harness

TAG=${TAG:-round0-baseline}
IMPL=${IMPL:-baseline}
REPEATS=${REPEATS:-3}
WORKLOADS=${WORKLOADS:-"kimi-k2-t512 ds-v3-t4096 kimi-k2-t1 ds-v3-t32-bf16 ds-v3-t32-fp16 glm45-t32"}
CONFIG=${CONFIG:-}

for w in $WORKLOADS; do
  extra=""
  if [ -n "$CONFIG" ]; then extra="--config $CONFIG"; fi
  echo "=== $TAG / $w"
  $PY capture.py --impl "$IMPL" --workload "$w" --tag "$TAG" --repeats "$REPEATS" $extra 2>&1 | tail -12
done
echo "=== done $TAG"
