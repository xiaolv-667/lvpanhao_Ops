#!/usr/bin/env bash
# 950PR 迁移验证环境一键安装（x86_64 + CANN 的 Ascend950PR 机器）
#
# 用法（与 compile_smoke_npu.py、wheels/ 放同一目录）：
#   bash setup.sh              # 安装依赖 + 无卡编译冒烟
#   bash setup.sh --no-install # 只做环境检查与编译冒烟
#
# 全程不占用卡：编译阶段 ASCEND_RT_VISIBLE_DEVICES 强制为空。

set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORK="${WORK:-/workspace}"
VENV="${VENV:-$WORK/venv-ascend950}"
CANN_ROOT="${CANN_ROOT:-}"
DO_INSTALL=1
[ "${1:-}" = "--no-install" ] && DO_INSTALL=0

step() { printf '\n===== %s =====\n' "$*"; }
ok()   { printf '  [OK] %s\n' "$*"; }
warn() { printf '  [WARN] %s\n' "$*"; }
die()  { printf '\n[ABORT] %s\n' "$*"; exit 1; }

step "1/6 检查机器与卡"
command -v npu-smi >/dev/null 2>&1 || die "找不到 npu-smi，这不是 NPU 机器"
npu-smi info -t board -i 0 2>/dev/null | grep -E 'NPU Name|Chip Name|Product Name' || warn "board 信息读取失败"
NPU_NAME=$(npu-smi info -t board -i 0 2>/dev/null | grep -m1 'NPU Name' | awk -F: '{print $2}' | tr -d ' ')
[ -n "${NPU_NAME:-}" ] && echo "  检测到型号: $NPU_NAME（期望 9579）"
[ "${NPU_NAME:-}" = "9579" ] || warn "型号不是 9579，硬件档案可能要新建"
command -v python3 >/dev/null 2>&1 || die "找不到 python3"
python3 -V

step "2/6 定位 CANN 并加载环境"
for c in "$CANN_ROOT" /usr/local/Ascend/cann /usr/local/Ascend/ascend-toolkit/latest; do
  [ -n "$c" ] && [ -d "$c" ] && { CANN_ROOT="$c"; break; }
done
[ -n "$CANN_ROOT" ] || die "找不到 CANN 安装目录"
echo "  CANN: $CANN_ROOT -> $(readlink -f "$CANN_ROOT")"
set +u
[ -f "$CANN_ROOT/set_env.sh" ] && { . "$CANN_ROOT/set_env.sh" >/dev/null 2>&1; ok "已 source set_env.sh"; }
set -u
export ASCEND_HOME_PATH="$CANN_ROOT"
export ASCEND_RUNTIME_ROOT="$CANN_ROOT"
export ASCEND_NPU_ARCH="${ASCEND_NPU_ARCH:-dav-3510}"
export PATH="$CANN_ROOT/bin:$CANN_ROOT/tools/bishengir/bin:$PATH"
echo "  ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
command -v bishengir-compile >/dev/null 2>&1 && ok "bishengir-compile 就位" || warn "PATH 里没有 bishengir-compile"

step "3/6 创建 venv"
if [ -x "$VENV/bin/python" ]; then
  ok "复用已存在的 venv: $VENV"
else
  [ "$DO_INSTALL" -eq 1 ] || die "venv 不存在且指定了 --no-install"
  python3 -m venv "$VENV" || die "venv 创建失败"
  ok "已创建 $VENV"
fi
. "$VENV/bin/activate"
python -V

if [ "$DO_INSTALL" -eq 1 ]; then
  step "4/6 安装依赖（首次约 5-30 分钟，取决于网络）"
  python -m pip install -q --upgrade pip
  echo "  --- torch (CPU 索引) ---"
  python -m pip install --no-cache-dir torch==2.13.0+cpu \
      --index-url https://download.pytorch.org/whl/cpu \
    || { warn "CPU 索引装不到，退回 pypi"; python -m pip install --no-cache-dir torch==2.13.0 || die "torch 安装失败"; }
  echo "  --- tilelang（慢的话可换国内镜像）---"
  python -m pip install --no-cache-dir tilelang==0.1.15 || die "tilelang 安装失败"
  echo "  --- torch-npu（用随包 wheel）---"
  python -m pip install --no-cache-dir --no-index --find-links "$HERE/wheels" torch-npu==2.13.0rc1 \
    || die "torch-npu 安装失败（确认 $HERE/wheels 下有 cp312 的 wheel）"
  echo "  --- pytest（真机精度用）---"
  python -m pip install --no-cache-dir pytest || warn "pytest 安装失败，真机精度前需手动补装"
else
  step "4/6 跳过安装（--no-install）"
fi

step "5/6 版本核对"
python - <<'PY'
import importlib.metadata as md, platform
print("  python :", platform.python_version())
for p in ("torch", "torch-npu", "tilelang"):
    try: print(f"  {p:10}", md.version(p))
    except Exception: print(f"  {p:10} <未安装>")
PY

step "6/6 无卡编译冒烟（不启动 kernel）"
export ASCEND_RT_VISIBLE_DEVICES=""
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$WORK/.cache/tilelang-950pr}"
mkdir -p "$TILELANG_CACHE_DIR"
echo "  ASCEND_RT_VISIBLE_DEVICES=[${ASCEND_RT_VISIBLE_DEVICES}]  (空=不占卡)"
if [ -f "$HERE/compile_smoke_npu.py" ]; then
  "$VENV/bin/python" "$HERE/compile_smoke_npu.py"
else
  warn "缺少 compile_smoke_npu.py，跳过冒烟"
fi

step "完成"
cat <<EOF
环境位于: $VENV
真机精度验证（会真正用卡）示例：
  export ASCEND_RT_VISIBLE_DEVICES=0
  export PYTHONPATH=$WORK/examples/TileOPs:\$PYTHONPATH
  $VENV/bin/python -m pytest $WORK/examples/TileOPs/tests/ops/test_paged_kv_cache_write.py -q
EOF
