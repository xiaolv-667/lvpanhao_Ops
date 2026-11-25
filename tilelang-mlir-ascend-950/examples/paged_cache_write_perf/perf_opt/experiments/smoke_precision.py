"""Final smoke precision check: the three smoke cases of the migrated benchmark.

``bench_paged_cache_write.py`` marks every dtype combination of the manifest's
first workload as smoke (5 tokens, 2 KV heads, dim 64, 3 pages of 7 rows).  The
benchmark itself only profiles the tune cases, so the smoke cases are checked
here with the delivered module's own byte-exact comparison.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PERF_OPT = os.path.dirname(HERE)
sys.path.insert(0, PERF_OPT)

spec = importlib.util.spec_from_file_location(
    "pcw_final", os.path.join(PERF_OPT, "paged_cache_write.py")
)
mod = importlib.util.module_from_spec(spec)
sys.modules["pcw_final"] = mod
spec.loader.exec_module(mod)

SMOKE = [
    ("paged-write-smoke-float16", "float16", "float16"),
    ("paged-write-smoke-bfloat16", "bfloat16", "bfloat16"),
    ("paged-write-smoke-bfloat16-float8_e4m3fn", "bfloat16", "float8_e4m3fn"),
]

ok = True
for name, in_dtype, cache_dtype in SMOKE:
    quantized = cache_dtype == "float8_e4m3fn"
    passed = mod.run_case(
        level="smoke",
        name=name,
        tokens=5,
        heads_kv=2,
        dim=64,
        num_pages=3,
        page_size=7,
        in_dtype=in_dtype,
        cache_dtype=cache_dtype,
        k_scale=1.25 if quantized else None,
        v_scale=0.75 if quantized else None,
        seed=1,
        repeat=3,
    )
    ok = ok and passed
print(f"[smoke] {'PRECISION_PASS' if ok else 'PRECISION_FAIL'} max_diff={mod._MAX_DIFF[0]:.6g}")
sys.exit(0 if ok else 1)
