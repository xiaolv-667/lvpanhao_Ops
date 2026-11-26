#!/usr/bin/env bash
# Capture every tune workload for one candidate tag (self-locating: the harness
# directory and the repo root are derived from this file's path).
set -u
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=/workspace/tilelang-mlir-ascend-950
source "$REPO/.agents/platforms/toolchains/ascend/env-950pr.sh" >/dev/null 2>&1
export TMPDIR=/workspace
export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-/tmp/opencode/tl-cache-opt-shared2}
mkdir -p "$TILELANG_CACHE_DIR"
PY=/workspace/venv-ascend950/bin/python
cd "$HERE"

TAG=${TAG:-round0-baseline}
IMPL=${IMPL:-baseline}
REPEATS=${REPEATS:-3}
WORKLOADS=${WORKLOADS:-"kimi-k2-t512 ds-v3-t4096 kimi-k2-t1 ds-v3-t32-bf16 ds-v3-t32-fp16 glm45-t32"}
CONFIG=${CONFIG:-}

echo "[capture_all] here=$HERE impl=$IMPL tag=$TAG"
for w in $WORKLOADS; do
  extra=""
  if [ -n "$CONFIG" ]; then extra="--config $CONFIG"; fi
  echo "=== $TAG / $w"
  $PY capture.py --impl "$IMPL" --workload "$w" --tag "$TAG" --repeats "$REPEATS" $extra 2>&1 | tail -12
done
echo "=== done $TAG"
