#!/usr/bin/env bash
#
# configure-deepseek.sh — 在新机器上把 DeepSeek 模型接进 AtomCode
#
# 作用：幂等地写入/更新 $ATOMCODE_HOME/config.toml 的三处配置
#   ① 顶层  default_model = "<DS_MODEL_ID>"
#   ② section [models."<DS_MODEL_ID>"]
#   ③ section [provider_accounts.<DS_ACCOUNT>]
#
# 特性：
#   - 幂等：重复执行结果一致；已存在的同名 section 会被整段替换（其余配置不动）
#   - 安全：先备份原 config；写完 chmod 600；摘要里 API Key 打码
#   - 校验：写入后用 tomllib 反解确认 TOML 合法且三处配置生效（Python >= 3.11）
#
# 用法：
#   ./configure-deepseek.sh                              # 交互式输入 API Key
#   DEEPSEEK_API_KEY=sk-xxx ./configure-deepseek.sh      # 非交互（推荐放进 CI/引导脚本）
#   DS_BASE_URL=https://my-proxy.example.com/v1 ./configure-deepseek.sh
#   DS_USE_ENV_KEY=1 DEEPSEEK_API_KEY=sk-xxx ./configure-deepseek.sh
#       → 只写 api_key = "${DEEPSEEK_API_KEY}" 占位，不落盘明文（更安全，
#         但要求该环境变量在新机器上常驻）
#
# 可覆盖的环境变量：
#   DEEPSEEK_API_KEY     API Key（未设则看下面的内嵌 Key，再没有才交互式读取）
#   EMBEDDED_API_KEY     覆盖脚本内嵌的 Key（测试/临时用）
#   ATOMCODE_HOME        默认 $HOME/.atomcode
#   DS_USE_ENV_KEY       1 = 写 ${DEEPSEEK_API_KEY} 占位而非明文（默认 0，即明文）
#   DS_MODEL_ID          models.<id> 的键名          默认 deepseek-flash
#   DS_MODEL             provider 侧真实 model 名    默认 deepseek-flash
#   DS_ACCOUNT           provider_accounts 的键名    默认 deepseek
#   DS_BASE_URL          端点                        默认 https://api.deepseek.com
#   DS_CONTEXT_WINDOW    默认 1000000
#   DS_MAX_TOKENS        默认 384000
#   DS_EFFORT            默认 high
#   DS_VISION            true/false                  默认 true
#
set -euo pipefail

# ============================================================================
# API Key 来源（脚本内不再保存明文，因为本文件要进公开仓库）：
#   1) 环境变量 EMBEDDED_API_KEY（显式覆盖）
#   2) 同目录 .deepseek.key（单行 sk-...，已被 .gitignore 忽略，chmod 600）
#   3) 环境变量 DEEPSEEK_API_KEY
#   4) 交互式输入
# 新机器上想"零参数直接跑"：把 Key 写进 .deepseek.key 即可。
# ============================================================================
_KEYFILE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/.deepseek.key"
if [ -z "${EMBEDDED_API_KEY:-}" ] && [ -f "$_KEYFILE" ]; then
  EMBEDDED_API_KEY="$(tr -d '\r\n' < "$_KEYFILE")"
fi
EMBEDDED_API_KEY="${EMBEDDED_API_KEY:-}"

ATOMCODE_HOME="${ATOMCODE_HOME:-$HOME/.atomcode}"
CONFIG="$ATOMCODE_HOME/config.toml"

DS_MODEL_ID="${DS_MODEL_ID:-deepseek-flash}"
DS_MODEL="${DS_MODEL:-deepseek-flash}"
DS_ACCOUNT="${DS_ACCOUNT:-deepseek}"
DS_BASE_URL="${DS_BASE_URL:-https://api.deepseek.com}"
DS_CONTEXT_WINDOW="${DS_CONTEXT_WINDOW:-1000000}"
DS_MAX_TOKENS="${DS_MAX_TOKENS:-384000}"
DS_EFFORT="${DS_EFFORT:-high}"
DS_VISION="${DS_VISION:-true}"
DS_USE_ENV_KEY="${DS_USE_ENV_KEY:-0}"

# ---- 依赖检查 -------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "错误：本脚本需要 python3（用于安全地读写 TOML）。" >&2
  exit 1
fi

# ---- 取 API Key -----------------------------------------------------------
# 优先级：环境变量 DEEPSEEK_API_KEY > 脚本内嵌 EMBEDDED_API_KEY > 交互式输入
if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -n "$EMBEDDED_API_KEY" ]; then
  DEEPSEEK_API_KEY="$EMBEDDED_API_KEY"
  echo "使用 .deepseek.key / EMBEDDED_API_KEY 提供的 API Key。"
fi

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  if [ -t 0 ]; then
    printf '请输入 DeepSeek API Key (sk-...): ' >&2
    read -rs DEEPSEEK_API_KEY
    echo >&2
  else
    echo "错误：未设置 DEEPSEEK_API_KEY，且当前非交互环境无法提示输入。" >&2
    echo "      用法: DEEPSEEK_API_KEY=sk-xxx $0" >&2
    exit 1
  fi
fi
if [ -z "$DEEPSEEK_API_KEY" ]; then
  echo "错误：API Key 为空。" >&2
  exit 1
fi

mkdir -p "$ATOMCODE_HOME"

# ---- 备份 -----------------------------------------------------------------
if [ -f "$CONFIG" ]; then
  BACKUP="$CONFIG.bak.$(date +%Y%m%d%H%M%S)"
  cp -p "$CONFIG" "$BACKUP"
  echo "已备份原配置 -> $BACKUP"
else
  echo "未发现已有配置，将新建 $CONFIG"
fi

# ---- 幂等写入（交给 python3，避免 sed 破坏 TOML 结构）----------------------
CONFIG="$CONFIG" \
DS_MODEL_ID="$DS_MODEL_ID" \
DS_MODEL="$DS_MODEL" \
DS_ACCOUNT="$DS_ACCOUNT" \
DS_BASE_URL="$DS_BASE_URL" \
DS_CONTEXT_WINDOW="$DS_CONTEXT_WINDOW" \
DS_MAX_TOKENS="$DS_MAX_TOKENS" \
DS_EFFORT="$DS_EFFORT" \
DS_VISION="$DS_VISION" \
DS_USE_ENV_KEY="$DS_USE_ENV_KEY" \
DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
python3 - <<'PY'
import os
import re

path = os.environ["CONFIG"]
model_id = os.environ["DS_MODEL_ID"]
account = os.environ["DS_ACCOUNT"]


def read_text():
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def upsert_top_key(text, key, value):
    """在顶层键区 upsert `key = "value"`，不触碰任何 [section]。"""
    lines = text.splitlines()
    pat = re.compile(r"^\s*" + re.escape(key) + r"\s*=")
    out = []
    found = False
    for line in lines:
        if pat.match(line):
            out.append('%s = "%s"' % (key, value))
            found = True
        else:
            out.append(line)
    if not found:
        # 插到第一个 section 之前；没有 section 就追加到末尾
        idx = len(out)
        for i, l in enumerate(out):
            if l.lstrip().startswith("["):
                idx = i
                break
        if idx > 0 and out[idx - 1].strip() != "":
            out.insert(idx, "")
            idx += 1
        out.insert(idx, '%s = "%s"' % (key, value))
        out.insert(idx + 1, "")
    return "\n".join(out).rstrip("\n") + "\n"


def upsert_section(text, header, body):
    """整段替换或追加一个 TOML section。header 形如 '[models."x"]'。"""
    lines = text.splitlines()
    out = []
    i = 0
    found = False
    while i < len(lines):
        if lines[i].strip() == header:
            found = True
            i += 1
            # 跳过旧 section 的内容，直到下一个 section header。
            # 注意：紧接下一个 header 的空行属于"分隔符"而非 section 内容，
            # 必须保留，否则重复执行会不断吃掉空行（非幂等）。
            while i < len(lines):
                if lines[i].lstrip().startswith("["):
                    break
                if lines[i].strip() == "":
                    j = i
                    while j < len(lines) and lines[j].strip() == "":
                        j += 1
                    if j >= len(lines) or lines[j].lstrip().startswith("["):
                        break
                i += 1
            out.append(header)
            out.extend(body)
            continue
        out.append(lines[i])
        i += 1
    if not found:
        if out and out[-1].strip() != "":
            out.append("")
        out.append(header)
        out.extend(body)
    return "\n".join(out).rstrip("\n") + "\n"


text = read_text()
text = upsert_top_key(text, "default_model", model_id)

if os.environ["DS_USE_ENV_KEY"] == "1":
    api_key_line = 'api_key = "${DEEPSEEK_API_KEY}"'
else:
    api_key_line = 'api_key = "%s"' % os.environ["DEEPSEEK_API_KEY"]

text = upsert_section(
    text,
    "[provider_accounts.%s]" % account,
    [
        'provider = "openai"',
        'base_url = "%s"' % os.environ["DS_BASE_URL"],
        api_key_line,
    ],
)

text = upsert_section(
    text,
    '[models."%s"]' % model_id,
    [
        'account = "%s"' % account,
        'model = "%s"' % os.environ["DS_MODEL"],
        "supports_vision = %s" % os.environ["DS_VISION"].lower(),
        "context_window = %s" % os.environ["DS_CONTEXT_WINDOW"],
        "max_tokens = %s" % os.environ["DS_MAX_TOKENS"],
        'reasoning_effort_levels = ["low", "high"]',
        'reasoning_effort = "%s"' % os.environ["DS_EFFORT"],
    ],
)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

# ---- 反解校验 -------------------------------------------------------------
try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

if tomllib is not None:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    assert data.get("default_model") == model_id, "default_model 未生效"
    assert account in data.get("provider_accounts", {}), "provider_accounts 未写入"
    assert model_id in data.get("models", {}), "models 未写入"
    print("[OK] TOML 反解校验通过")
else:
    print("[INFO] python < 3.11，跳过 TOML 反解校验")
PY

# ---- 收紧权限（文件含密钥）-------------------------------------------------
chmod 600 "$CONFIG"
echo "已设置权限 600 -> $CONFIG"

# ---- 输出摘要（API Key 打码）----------------------------------------------
echo
echo "==================== 配置摘要 ===================="
python3 - "$CONFIG" "$DS_MODEL_ID" "$DS_ACCOUNT" <<'PY'
import re
import sys

path, model_id, account = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(path, encoding="utf-8").read().splitlines()


def mask(s):
    return re.sub(r'(api_key\s*=\s*")([^"]{0,6})[^"]*(")', r'\1\2...\3', s)


print("配置文件: %s" % path)
for l in lines:
    if re.match(r"^\s*default_model\s*=", l):
        print("顶层    : %s" % l.strip())

for header in ("[provider_accounts.%s]" % account, '[models."%s"]' % model_id):
    printed = False
    for i, l in enumerate(lines):
        if l.strip() == header:
            print()
            print(header)
            for x in lines[i + 1:i + 8]:
                if x.lstrip().startswith("["):
                    break
                print("   " + mask(x))
            printed = True
            break
    if not printed:
        print("\n(未找到 %s)" % header)
PY

echo
echo "完成。下一步："
echo "  1) 新开一个 AtomCode 会话（配置在构建时读取，需新会话或重启才生效）"
echo "  2) 会话内执行 /model，列表里应出现 ${DS_MODEL_ID}"
echo "  3) 会话内执行 describe_self(aspect=\"models\") 确认已挂载"
