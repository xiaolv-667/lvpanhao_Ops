#!/usr/bin/env bash
# Ascend950PR_9579 framework environment (x86_64).
#
#   source .agents/platforms/toolchains/ascend/env-950pr.sh
#
# Mirrors the canonical toolchain contract as closely as this host allows:
#   compiler SDK  = CANN 9.2.0 release   (/home/l00970450/zuo/Ascend/cann-9.2.0)
#   runtime / OPP = CANN 9.1.0           (/usr/local/Ascend/cann-9.1.0)
#   host C++      = GCC 12.3.0           (/home/l00970450/zuo/.environment/gcc-12.3.0)
# Deviations from the canonical profile are recorded in
# .agents/platforms/hardware/ascend-950pr-9579-notes.md ("站点 profile 偏离声明").
#
# Override any of these before sourcing if the layout differs.

_950pr_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
_950pr_ws=$(cd -- "$_950pr_root/.." && pwd)

export TILEOPS_950PR_REPO="$_950pr_root"
export TILEOPS_950PR_VENV="${TILEOPS_950PR_VENV:-$_950pr_ws/.venvs/ascend950-py31215}"
[ -d "$TILEOPS_950PR_VENV" ] || export TILEOPS_950PR_VENV="$_950pr_ws/venv-ascend950"
export TILEOPS_950PR_CACHE="${TILEOPS_950PR_CACHE:-$_950pr_ws/.cache/tilelang-950pr}"

# --- compiler SDK (9.2.0) and runtime (9.1.0) are deliberately split ---
export ASCEND_COMPILER_ROOT="${ASCEND_COMPILER_ROOT:-/home/l00970450/zuo/Ascend/cann-9.2.0}"
export ASCEND_RUNTIME_ROOT="${ASCEND_RUNTIME_ROOT:-/usr/local/Ascend/cann-9.1.0}"
# --- site fallback -----------------------------------------------------------
# The two paths above describe the reference machine. On a site that does not
# have them, resolve to the layout this host actually uses, so that a plain
# `source env-950pr.sh` works without per-shell overrides.
[ -d "$ASCEND_COMPILER_ROOT" ] || export ASCEND_COMPILER_ROOT=/usr/local/Ascend/cann-9.2.1
[ -d "$ASCEND_RUNTIME_ROOT" ] || export ASCEND_RUNTIME_ROOT=/usr/local/Ascend/cann-9.2.1
export ASCEND_HOME_PATH="$ASCEND_COMPILER_ROOT"
export ASCEND_TOOLKIT_HOME="$ASCEND_COMPILER_ROOT"
export ASCEND_OPP_PATH="$ASCEND_RUNTIME_ROOT/opp"
export ASCEND_AICPU_PATH="$ASCEND_RUNTIME_ROOT"

# Hardware identity: must match .agents/platforms/hardware/ascend-950pr-9579.json
export ASCEND_NPU_ARCH=dav-3510
export USE_ASCEND=ON USE_CUDA=OFF USE_ROCM=OFF

# Host C++ compiler pinned by the contract (system default is 13.3.0).
export CXX="${CXX:-/home/l00970450/zuo/.environment/gcc-12.3.0/bin/g++}"
[ -x "$CXX" ] || export CXX=/usr/bin/g++-12

# Cards: only 0-3 are permitted by team ACL; 3/4 currently report Alarm.
# Site fallback: a single-card host narrows the canonical 0,2 list to 0.
if [ -z "${TILEOPS_950PR_CARDS:-}" ]; then
  _npu_cnt=$(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\|')
  [ "${_npu_cnt:-0}" -eq 1 ] && TILEOPS_950PR_CARDS=0
fi
export ASCEND_RT_VISIBLE_DEVICES="${TILEOPS_950PR_CARDS:-0,2}"

# PATH: 9.2.0 compiler first. CANN 9.2.0 ships bishengir-compile/opt but NOT the
# `-a5` variants (only hivmc-a5); the A5 variants live in 9.1.0. Set
# TILEOPS_950PR_A5_SHIM=1 to put the repository's .shims-a5 first, which maps
# bishengir-compile/opt onto the 9.1.0 A5 binaries.
_950pr_path="$ASCEND_COMPILER_ROOT/bin:$ASCEND_COMPILER_ROOT/tools/bisheng_compiler/bin"
# Site fallback: the OpenCode CLI lives under the Node runtime installed on this
# host; keep it discoverable so `opencode debug agents` works from any shell.
case ":$PATH:" in
  *":/usr/local/node20/bin:"*) ;;
  *) [ -d /usr/local/node20/bin ] && _950pr_path="/usr/local/node20/bin:$_950pr_path" ;;
esac
if [ "${TILEOPS_950PR_A5_SHIM:-0}" = "1" ]; then
  _950pr_path="$_950pr_ws/.shims-a5:$_950pr_path"
fi
export PATH="$_950pr_path:${PATH}"

# Runtime libraries stay on 9.1.0; do not let the SDK libs shadow them.
_sp="$TILEOPS_950PR_VENV/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$ASCEND_RUNTIME_ROOT/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$_sp/torch_npu/lib:$_sp/torch/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="$_950pr_root/.agents/tools:${PYTHONPATH}"
export TILELANG_CACHE_DIR="$TILEOPS_950PR_CACHE"

_950pr_py="$_950pr_root/.agents/tools"

# Verify the platform resolves, the device matches the lock, and the agents load.
check_950pr() {
  local py="$TILEOPS_950PR_VENV/bin/python"
  echo "== toolchain =="
  echo "compiler: $ASCEND_COMPILER_ROOT"
  echo "runtime : $ASCEND_RUNTIME_ROOT"
  echo "CXX     : $("$CXX" -dumpfullversion 2>/dev/null || echo missing)"
  echo "== platform probe =="
  "$py" "$_950pr_py/statectl.py" platform \
    --hardware ascend-950pr-9579 --toolchain ascend --probe >/dev/null \
    && echo "ok: ascend-950pr-9579 / ascend" || return 1
  echo "== knowledge base =="
  "$py" "$_950pr_py/kb_lint.py" --toolchain ascend --hardware ascend-950pr-9579 | tail -1
  echo "== standards =="
  "$py" "$_950pr_py/standards_check.py" check | tail -1
  echo "== agents =="
  local n
  n=$( cd "$_950pr_root" && opencode debug agents 2>/dev/null | grep -c '"id": "tilelang-' )
  if [ "${n:-0}" -eq 0 ]; then
    # opencode >=1.19 dropped the plural `debug agents`; count the resolved
    # config's agent entries instead. The JSON is large enough that a pipe can
    # truncate it, so it is written to a file first.
    _oc_tmp=$(mktemp 2>/dev/null || echo /tmp/oc_cfg.$$)
    ( cd "$_950pr_root" && opencode debug config >"$_oc_tmp" 2>/dev/null )
    n=$("$py" -c 'import json,sys; print(sum(1 for k in json.load(open(sys.argv[1])).get("agent",{}) if k.startswith("tilelang-")))' "$_oc_tmp" 2>/dev/null)
    rm -f "$_oc_tmp"
  fi
  echo "tilelang agents registered: ${n:-0}"
}

echo "950PR env ready: compiler=$(basename "$ASCEND_COMPILER_ROOT") runtime=$(basename "$ASCEND_RUNTIME_ROOT") cards=$ASCEND_RT_VISIBLE_DEVICES"
echo "run 'check_950pr' to verify the platform, knowledge base and agents"
