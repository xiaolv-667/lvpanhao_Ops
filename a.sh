#!/usr/bin/env bash
# hidevlab 950PR 环境探测 v2（只读，不安装、不占卡）
# 用法：bash probe2.sh

set +e
line() { printf '\n---------- %s ----------\n' "$*"; }

line "1. soc 名称（aclInit -> aclrtSetDevice -> aclrtGetSocName）"
python3 - <<'PY' 2>&1 | head -24
import ctypes, glob
libs = (glob.glob("/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so")
        + glob.glob("/usr/local/Ascend/cann*/lib64/libascendcl.so")
        + glob.glob("/usr/local/Ascend/cann*/runtime/lib64/libascendcl.so"))
if not libs:
    print("未找到 libascendcl.so"); raise SystemExit
acl = ctypes.CDLL(libs[0])
print("lib:", libs[0])
print("aclInit rc =", acl.aclInit(None))
cnt = ctypes.c_uint32(0)
print("aclrtGetDeviceCount rc =", acl.aclrtGetDeviceCount(ctypes.byref(cnt)), "count =", cnt.value)
if cnt.value > 0:
    rc = acl.aclrtSetDevice(0)
    print("aclrtSetDevice(0) rc =", rc)
    buf = ctypes.create_string_buffer(64)
    rc = acl.aclrtGetSocName(buf)
    print("aclrtGetSocName rc =", rc, "->", buf.value.decode() if rc == 0 else "FAILED")
PY
echo "--- 兜底 1: npu-smi board ---"
npu-smi info -t board -i 0 2>/dev/null | head -12
echo "--- 兜底 2: driver 版本 ---"
head -3 /usr/local/Ascend/driver/version.info 2>/dev/null

line "2. /workspace 与根分区真实可用空间"
df -h / /workspace 2>/dev/null | sort -u
echo "--- du /workspace ---"; du -sh /workspace 2>/dev/null

line "3. CANN 版本细节"
for d in /usr/local/Ascend/cann /usr/local/Ascend/cann-9.2.1 /usr/local/Ascend/ascend-toolkit/latest; do
  [ -e "$d" ] || continue
  echo "--- $d -> $(readlink -f $d 2>/dev/null)"
  for f in version.cfg version.info compiler/version.info opp/version.info; do
    [ -f "$d/$f" ] && head -2 "$d/$f" | tr '\n' ' ' && echo "   [$f]"
  done
done

line "4. 平台配置里有哪些 950 型号（推断 soc）"
find /usr/local/Ascend -maxdepth 4 -name 'Ascend950*.ini' 2>/dev/null | sed 's|.*/||' | sort -u | head -20

line "5. pip 能否装到指定版本"
python3 -m pip --version 2>&1 | head -1
for pkg in "torch==2.13.0+cpu" "torch-npu==2.13.0rc1" "tilelang==0.1.15"; do
  printf '%-24s ' "$pkg"
  timeout 60 python3 -m pip index versions "${pkg%%==*}" 2>/dev/null | head -1 || echo "(index 查询不支持)"
done

line "6. 编译与运行头文件"
ls /usr/local/Ascend/cann*/asc/include/c_api/cube_datamove/cube_datamove.h 2>/dev/null | head -2
ls /usr/local/Ascend/cann*/asc/include/c_api/cube_compute/cube_compute.h 2>/dev/null | head -2

line "完成"
echo "把输出贴回来。未安装、未修改、未占卡。"
