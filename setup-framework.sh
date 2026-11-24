#!/usr/bin/env bash
# 在新机器上生成属于本站点的工具链锁，并验证平台可用
#
# 用法（在工作区根目录执行）：
#   bash setup-framework.sh
#
# 做三件事：
#   1. 采集本站点的 machine / python / CXX / 包版本 / 关键文件 sha256
#   2. 写入 tilelang-mlir-ascend-950/.agents/platforms/toolchains/ascend/toolchain.lock.json
#      （原锁备份为 toolchain.lock.<时间戳>.bak）
#   3. 跑 statectl platform --probe 验证

set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FW="$HERE/tilelang-mlir-ascend-950"
[ -d "$FW" ] || { echo "找不到框架仓: $FW"; exit 1; }

PY="${PY:-$HERE/venv-ascend950/bin/python}"
[ -x "$PY" ] || PY="${PY:-python3}"
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}"
LOCK="$FW/.agents/platforms/toolchains/ascend/toolchain.lock.json"
HARDWARE_ID="${HARDWARE_ID:-ascend-950pr-9579}"

echo "== 1/3 采集本站点环境 =="
echo "  python : $($PY -V 2>&1)"
echo "  CANN   : $CANN -> $(readlink -f "$CANN" 2>/dev/null)"
echo "  CXX    : ${CXX:-g++}"

# 让探测与编译使用同一套环境
set +u
[ -f "$CANN/set_env.sh" ] && . "$CANN/set_env.sh" >/dev/null 2>&1
set -u
export ASCEND_HOME_PATH="$CANN"
export ASCEND_RUNTIME_ROOT="${ASCEND_RUNTIME_ROOT:-$CANN}"
export ASCEND_NPU_ARCH="${ASCEND_NPU_ARCH:-dav-3510}"
export PATH="$CANN/bin:$CANN/tools/bishengir/bin:$PATH"

echo "== 2/3 生成站点锁 =="
[ -f "$LOCK" ] && cp -v "$LOCK" "$LOCK.$(date +%H%M%S).bak"

"$PY" - "$LOCK" "$HARDWARE_ID" <<'PY'
import hashlib, importlib.metadata as md, json, os, platform, subprocess, sys
from pathlib import Path

lock_path, hardware_id = Path(sys.argv[1]), sys.argv[2]
cann = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann")
rt = os.environ.get("ASCEND_RUNTIME_ROOT", cann)

def ver(p):
    try: return md.version(p)
    except Exception: return None

def sha(p):
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
        return h.hexdigest()
    except OSError:
        return None

def pkg_file(pkg, rel):
    try: return str(md.distribution(pkg).locate_file(rel))
    except Exception: return None

cxx = os.environ.get("CXX", "g++")
try: cxx_ver = subprocess.check_output([cxx, "-dumpfullversion"], text=True).strip()
except Exception: cxx_ver = "unknown"

comp_ver = ""
for f in (f"{cann}/compiler/version.info", f"{cann}/version.info"):
    if os.path.exists(f):
        for line in open(f):
            if line.startswith("Version="):
                comp_ver = line.split("=", 1)[1].strip(); break
        if comp_ver: break

files = [
    {"name": "Bisheng", "path": "bin/bisheng", "sha256": sha(f"{cann}/bin/bisheng")},
    {"name": "Cube data-move headers", "path": "asc/include/c_api/cube_datamove/cube_datamove.h",
     "sha256": sha(f"{cann}/asc/include/c_api/cube_datamove/cube_datamove.h")},
    {"name": "Cube compute headers", "path": "asc/include/c_api/cube_compute/cube_compute.h",
     "sha256": sha(f"{cann}/asc/include/c_api/cube_compute/cube_compute.h")},
    {"name": "ACL runtime", "path": "lib64/libascendcl.so", "root_env": "ASCEND_RUNTIME_ROOT",
     "sha256": sha(f"{rt}/lib64/libascendcl.so")},
    {"name": "OPP version", "path": "opp/version.info", "root_env": "ASCEND_RUNTIME_ROOT",
     "sha256": sha(f"{rt}/opp/version.info")},
]
for name, rel in (("TileLang native compiler", "tilelang/lib/libtilelang.so"),
                  ("TVM native compiler", "tilelang/lib/libtvm_compiler.so"),
                  ("Ascend pass pipeline", "tilelang/ascend/pipeline.py"),
                  ("Ascend common templates", "tilelang/src/tl_templates/ascend/common.h"),
                  ("Ascend SIMD templates", "tilelang/src/tl_templates/ascend/simd_inst.h")):
    p = pkg_file("tilelang", rel)
    files.append({"name": name, "path": rel, "package": "tilelang", "sha256": sha(p) if p else None})
p = pkg_file("torch-npu", "torch_npu/lib/libtorch_npu.so")
files.append({"name": "Torch NPU native library", "path": "torch_npu/lib/libtorch_npu.so",
              "package": "torch-npu", "sha256": sha(p) if p else None})

lock = {
    "schema": 1,
    "profile": f"{hardware_id}-{platform.machine()}-site-v1",
    "machine": platform.machine(),
    "python": platform.python_version(),
    "packages": {"torch": ver("torch"), "torch-npu": ver("torch-npu"), "tilelang": ver("tilelang")},
    "cxx_version": cxx_ver,
    "compiler_sdk": f"CANN{comp_ver or '?'} {cann}",
    "runtime_opp": f"CANN{comp_ver or '?'} {rt}/lib64",
    "hardware": {"model": "Ascend950PR_9579", "cube": 28, "vector": 56},
    "cann_artifact": {"url": None,
                      "note": "Site-generated lock: hashes measured on this host; no packaged installer digest.",
                      "installer_sha256": None},
    "files": files,
    "uncertified_features": ["ascend.batch_schedule cross-block barrier dependency"],
}
lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
missing = [f["name"] for f in files if not f["sha256"]]
print(f"  已写入 {lock_path}")
print(f"  profile={lock['profile']}  python={lock['python']}  cxx={cxx_ver}")
if missing: print("  [WARN] 以下文件未取到指纹:", ", ".join(missing))
PY

echo "== 3/3 验证平台 =="
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
cd "$FW" || exit 1
PYTHONPATH="$FW/.agents/tools:$PYTHONPATH" "$PY" .agents/tools/statectl.py platform \
  --hardware "$HARDWARE_ID" --toolchain ascend --probe 2>&1 | tail -5

echo
echo "若上面输出 ok=true，即可在该机器直接启动框架："
echo "  cd $FW && opencode        # 选 tilelang-op-conductor"
echo "若报 fingerprint differs，把上面输出贴回来定位。"
