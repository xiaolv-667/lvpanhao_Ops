#!/usr/bin/env bash
# 环境指纹：生成可 diff 的标准 JSON，用于判断两台机器环境是否相同
#
# 用法：  bash env-fingerprint.sh > env-A.json
#         （另一台）bash env-fingerprint.sh > env-B.json
#         diff <(python3 -m json.tool env-A.json) <(python3 -m json.tool env-B.json)
#
# 判定原则：版本号相同 ≠ 环境相同。必须以「文件指纹」为准。

set +eu
PY="${PY:-python3}"
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}"

j() { printf '"%s": %s' "$1" "$2"; }

echo "{"
printf '  "generated_at": "%s",\n' "$(date -Is)"
printf '  "hostname": "%s",\n' "$(hostname 2>/dev/null)"
printf '  "machine": "%s",\n' "$(uname -m)"
printf '  "os": "%s",\n' "$(uname -sr | tr -d '"')"

# ---- 硬件层 ----
NPU_NAME=$(npu-smi info -t board -i 0 2>/dev/null | grep -m1 'NPU Name' | awk -F: '{print $2}' | tr -d ' ')
NPU_CHIP=$(npu-smi info -t board -i 0 2>/dev/null | grep -m1 'Chip Name' | awk -F: '{print $2}' | tr -d ' ')
NPU_CNT=$(npu-smi info -l 2>/dev/null | grep -m1 'Total Count' | awk -F: '{print $2}' | tr -d ' ')
printf '  "npu": {"name": "%s", "chip": "%s", "count": "%s"},\n' "${NPU_NAME:-}" "${NPU_CHIP:-}" "${NPU_CNT:-}"

# ---- CANN 层 ----
CANN_REAL=$(readlink -f "$CANN" 2>/dev/null)
COMP_VER=$(grep -m1 '^Version=' "$CANN/compiler/version.info" 2>/dev/null | cut -d= -f2)
OPP_VER=$(grep -m1 '^Version=' "$CANN/opp/version.info" 2>/dev/null | cut -d= -f2)
printf '  "cann": {"root": "%s", "real": "%s", "compiler": "%s", "opp": "%s"},\n' \
  "$CANN" "${CANN_REAL:-}" "${COMP_VER:-}" "${OPP_VER:-}"

# ---- 宿主编译器 ----
printf '  "cxx": "%s",\n' "$(${CXX:-g++} -dumpfullversion 2>/dev/null || echo missing)"

# ---- Python 层 ----
printf '  "python": {"version": "%s", "executable": "%s"},\n' \
  "$($PY -c 'import platform;print(platform.python_version())' 2>/dev/null)" "$(command -v $PY)"

# ---- 包层 + 文件指纹层 ----
$PY - <<'PY'
import hashlib, importlib.metadata as md, json, os, sys

def ver(p):
    try: return md.version(p)
    except Exception: return None

def sha(path):
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
        return h.hexdigest()
    except OSError:
        return None

def pkg_file(pkg, rel):
    try: return str(md.distribution(pkg).locate_file(rel))
    except Exception: return None

out = {"packages": {}, "files": {}}
for p in ("torch", "torch-npu", "tilelang"):
    out["packages"][p] = ver(p)

cann = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann")
targets = {
    "bisheng": os.path.join(cann, "bin/bisheng"),
    "cube_datamove_h": os.path.join(cann, "asc/include/c_api/cube_datamove/cube_datamove.h"),
    "cube_compute_h": os.path.join(cann, "asc/include/c_api/cube_compute/cube_compute.h"),
    "libascendcl": os.path.join(os.environ.get("ASCEND_RUNTIME_ROOT", cann), "lib64/libascendcl.so"),
    "libtilelang": pkg_file("tilelang", "tilelang/lib/libtilelang.so"),
    "libtvm_compiler": pkg_file("tilelang", "tilelang/lib/libtvm_compiler.so"),
    "tilelang_ascend_pipeline": pkg_file("tilelang", "tilelang/ascend/pipeline.py"),
    "tl_common_h": pkg_file("tilelang", "tilelang/src/tl_templates/ascend/common.h"),
    "tl_simd_h": pkg_file("tilelang", "tilelang/src/tl_templates/ascend/simd_inst.h"),
    "libtorch_npu": pkg_file("torch-npu", "torch_npu/lib/libtorch_npu.so"),
}
for k, p in targets.items():
    out["files"][k] = sha(p) if p else None

digest_src = json.dumps({"packages": out["packages"], "files": out["files"]}, sort_keys=True)
out["fingerprint_sha256"] = hashlib.sha256(digest_src.encode()).hexdigest()
print('  "details": ' + json.dumps(out, ensure_ascii=False, indent=2).replace("\n", "\n  ") + ",")
PY

printf '  "note": "compare fingerprint_sha256 first; if equal, environments are identical for this toolchain"\n}'
