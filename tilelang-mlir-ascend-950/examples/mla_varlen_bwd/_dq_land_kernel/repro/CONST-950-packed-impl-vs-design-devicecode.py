# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Stage 3 self-check: the shipped implementation compiles to the frozen design snapshot.

The design (DESIGN.md 3.5.4) recorded, for the packed form, 41 generated lines
with `vsts_pack_b32` and sha256 prefixes d0fde4920bf2c7ac (fp16) / 914c64f7abb2f600
(bf16).  Those came from a design-stage probe whose prim_func was named `_land`;
the shipped factory names it `_dq_land_main`, so the sha256 necessarily differs.
This script therefore compares the *normalised* device code (function names
replaced by the placeholder KERNEL) line by line, for both dtypes.

A mismatch means the implementation drifted from the frozen d-sign evidence and
has to be explained before the stage can pass.

Run (compile only, never launches):
    cd examples/mla_varlen_bwd/_dq_land_kernel
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        repro/CONST-950-packed-impl-vs-design-devicecode.py
"""

import hashlib
import importlib.util
import pathlib
import re

_HERE = pathlib.Path(__file__).resolve().parent
_IMPL = _HERE.parent / "_dq_land_kernel.py"
_SCALE = 192 ** -0.5

_EXPECTED_SHA = {
    "float16": "d0fde4920bf2c7ac",
    "bfloat16": "914c64f7abb2f600",
}


def _load_impl():
    spec = importlib.util.spec_from_file_location("dq_land_impl", _IMPL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalise(src):
    """Drop the generated symbol names that legitimately differ per prim_func."""
    out = []
    for line in src.splitlines():
        # the outlined VF body and the kernel entry carry the prim_func name
        line = re.sub(r"[A-Za-z_][A-Za-z_0-9]*_simd_vf_", "KERNEL_simd_vf_", line)
        line = re.sub(r"\b[A-Za-z_][A-Za-z_0-9]*_kernel\b", "KERNEL", line)
        line = re.sub(r"\b[A-Za-z_][A-Za-z_0-9]*_main\b", "KERNEL", line)
        out.append(line)
    return out


def main():
    impl = _load_impl()
    ok = True
    for dtype in ("float16", "bfloat16"):
        kernel = impl._dq_land_kernel(8, 192, _SCALE, dtype)
        src = kernel.get_kernel_source()
        digest = hashlib.sha256(src.encode()).hexdigest()[:16]
        snapshot = (_HERE.parent / "history_version" / f"probe_packed_gen_{dtype}.txt").read_text(
            encoding="utf-8"
        )
        live = _normalise(src)
        frozen = _normalise(snapshot)
        same = live == frozen
        print(
            f"[{dtype}] impl sha256={digest} lines={len(live)} "
            f"snapshot_sha256_prefix={_EXPECTED_SHA[dtype]} "
            f"snapshot_lines={len(frozen)} normalised_identical={same}",
            flush=True,
        )
        if not same:
            ok = False
            for i, (a, b) in enumerate(zip(live, frozen)):
                if a != b:
                    print(f"    first diff line {i + 1}:\n      impl: {a}\n      snap: {b}", flush=True)
                    break
            if len(live) != len(frozen):
                print(f"    line count differs ({len(live)} vs {len(frozen)})", flush=True)
        gate = impl._SRC_EVIDENCE.get((8, 192, _SCALE, dtype, 2))
        print(f"    gate record: {gate}", flush=True)
        if not gate or "vsts_pack_b32" not in gate["store"]:
            ok = False
    print("[PASS] implementation matches the frozen design device code" if ok else "[FAIL] drifted")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
