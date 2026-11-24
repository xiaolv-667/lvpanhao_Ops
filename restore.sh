#!/usr/bin/env bash
# 一键恢复：clone 本仓库到新机器后，跑这一条即可重建可用环境。
#
#   bash restore.sh              # 完整恢复（装依赖 + 生成工具链锁 + 编译冒烟）
#   bash restore.sh --no-install # 只校验环境与工具链，不装包
#   bash restore.sh --submodules # 额外拉取 3rdparty 子模块（仅源码编译框架时需要）
#
# 依赖：CANN 已随驱动盘安装（/usr/local/Ascend），python3 >= 3.10，npu-smi 可用。
# 恢复目标（本仓库 env/ 下记录）：python 3.12 / torch 2.13.0+cpu /
#   torch-npu 2.13.0rc1 / tilelang 0.1.15 / CANN 9.2.1 / gcc 11.4.0

set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORK="${WORK:-$(dirname -- "$HERE")}"

step() { printf '\n===== %s =====\n' "$*"; }
ok()   { printf '  [OK] %s\n' "$*"; }
warn() { printf '  [WARN] %s\n' "$*"; }
die()  { printf '\n[ABORT] %s\n' "$*"; exit 1; }

INSTALL=1
SUBMODULES=0
for a in "$@"; do
  case "$a" in
    --no-install)  INSTALL=0 ;;
    --submodules)  SUBMODULES=1 ;;
    -h|--help)     sed -n '2,13p' "$0"; exit 0 ;;
    *)             die "未知参数: $a" ;;
  esac
done

step "0/5 前置检查"
command -v npu-smi >/dev/null 2>&1 || die "找不到 npu-smi：这不是装了 NPU 驱动的机器"
command -v python3 >/dev/null 2>&1 || die "找不到 python3"
ls -d /usr/local/Ascend/cann* >/dev/null 2>&1 || warn "未发现 /usr/local/Ascend/cann*，CANN 可能未安装"
ok "python3: $(python3 -V 2>&1)"
if [ -f "$HERE/env/versions.txt" ]; then
  echo "  ---- 本仓库记录的环境 ----"
  sed -n '1,20p' "$HERE/env/versions.txt" | sed 's/^/  /'
fi

step "1/5 依赖安装（setup.sh）"
if [ "$INSTALL" -eq 1 ]; then
  bash "$HERE/setup.sh" || die "setup.sh 失败，按上面报错修完再重跑"
else
  bash "$HERE/setup.sh" --no-install || die "setup.sh --no-install 失败"
fi

step "2/5 生成工具链锁（setup-framework.sh）"
if [ -f "$HERE/setup-framework.sh" ]; then
  bash "$HERE/setup-framework.sh" || warn "工具链锁生成失败：检查 env/versions.txt 与 /usr/local/Ascend 是否一致"
else
  warn "缺少 setup-framework.sh，跳过"
fi

if [ "$SUBMODULES" -eq 1 ]; then
  step "3/5 拉取 3rdparty 子模块（源码编译框架时才需要）"
  T="$HERE/tilelang-mlir-ascend-950/3rdparty"
  mkdir -p "$T"
  [ -n "$(ls -A "$T/tvm" 2>/dev/null)" ] || \
    git clone --depth 1 https://github.com/TileLang/tvm "$T/tvm" || warn "tvm 拉取失败"
  [ -n "$(ls -A "$T/AscendNPU-IR" 2>/dev/null)" ] || \
    git clone --depth 1 https://gitcode.com/Ascend/AscendNPU-IR.git "$T/AscendNPU-IR" || warn "AscendNPU-IR 拉取失败"
  [ -n "$(ls -A "$T/AscendNPU-IR-Dev" 2>/dev/null)" ] || \
    git clone --depth 1 -b master https://gitcode.com/Ascend/AscendNPU-IR-Dev.git "$T/AscendNPU-IR-Dev" || warn "AscendNPU-IR-Dev 拉取失败"
else
  step "3/5 跳过 3rdparty（需要时加 --submodules）"
fi

step "4/5 校验与恢复记录一致"
if [ -f "$HERE/env/env-manifest.json" ]; then
  NEW=$(bash "$HERE/env-fingerprint.sh" 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["details"]["fingerprint_sha256"])' 2>/dev/null)
  OLD=$(python3 -c 'import json;print(json.load(open("'"$HERE"'/env/env-manifest.json"))["details"]["fingerprint_sha256"])' 2>/dev/null)
  echo "  记录指纹: ${OLD:-<读取失败>}"
  echo "  当前指纹: ${NEW:-<采集失败>}"
  if [ -n "${NEW:-}" ] && [ "$NEW" = "${OLD:-}" ]; then
    ok "指纹一致：与记录环境等价"
  else
    warn "指纹不同：kernel 设计与代码可复用，但编译产物/性能数字必须在本机重做"
  fi
fi

step "5/5 完成"
cat <<EOF
仓库位置: $HERE
工作目录: $WORK

下一步（真机精度，会占用 NPU）:
  export ASCEND_RT_VISIBLE_DEVICES=0
  export PYTHONPATH=$HERE/tilelang-mlir-ascend-950/examples/TileOPs:\$PYTHONPATH
  $WORK/venv-ascend950/bin/python -m pytest \\
    $HERE/tilelang-mlir-ascend-950/examples/TileOPs/tests/ops/test_paged_kv_cache_write.py -q

启动迁移框架:
  cd $HERE/tilelang-mlir-ascend-950
  source .agents/platforms/toolchains/ascend/env-950pr.sh
  check_950pr && opencode

模型/数据集不在仓库里，见 $HERE/ASSETS.md
EOF
