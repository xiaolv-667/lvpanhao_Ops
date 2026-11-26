"""Single-invocation msopprof driver for one RoPE kernel variant (Stage 4 harness).

Measurement protocol (task platform `ascend-950pr-9579` / `ascend`):

  ``msopprof --output=<dir> --launch-count=1 --warm-up=0 --kernel-name=main``
  around **one** operator call, then read ``Task Duration(us)`` from
  ``OpBasicInfo*.csv``.  The operator issues exactly one kernel launch
  (``main_kernel``); the driver asserts that and sums whatever was captured, so
  T_op is "one collection, one call" -- never a single-kernel median and never
  an NPU event.

Warm-up launches run in a *separate* process so no unmeasured launch can land in
the capture window; that first process also populates the TileLang JIT cache so
the traced process does not compile inside the capture window.

Usage (cwd = the task directory ``examples/rope_perf``):

    python perf_opt/experiments/bench_one.py --module <variant.py> --workload <wid> --reps 3
    python perf_opt/experiments/bench_one.py --module <variant.py> --workload <wid> --dump-src
"""

import argparse
import csv
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

# NOTE: never put the repository root on sys.path here -- its ``tilelang/``
# source tree would shadow the installed TileLang 0.1.15 package (platform rule).

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

# workload_id -> geometry of the migrated benchmark's parametrized cases
# (examples/TileOPs/benchmarks/ops/bench_rope.py::test_rope_fwd_bench over
# tileops/manifest/rope.yaml).  ``positions_shape`` and ``query_shape`` are the
# manifest's; ``key_shape`` None means a query-only call.
WORKLOADS = {
    "llama3-8b-decode-b64-bfloat16": dict(
        positions_shape=(64,), query_shape=(64, 4096), key_shape=(64, 1024),
        head_size=128, rotary_dim=128, max_position_embeddings=8192, base=500000.0,
        is_neox_style=True, dtype="bfloat16", position_dtype="int64",
    ),
    "llama3-8b-prefill-4k-bfloat16": dict(
        positions_shape=(4096,), query_shape=(4096, 4096), key_shape=(4096, 1024),
        head_size=128, rotary_dim=128, max_position_embeddings=8192, base=500000.0,
        is_neox_style=True, dtype="bfloat16", position_dtype="int64",
    ),
    "llama3-8b-prefill-4k-float16": dict(
        positions_shape=(4096,), query_shape=(4096, 4096), key_shape=(4096, 1024),
        head_size=128, rotary_dim=128, max_position_embeddings=8192, base=500000.0,
        is_neox_style=True, dtype="float16", position_dtype="int64",
    ),
    "llama3-8b-b2-2k-float32": dict(
        positions_shape=(2, 2048), query_shape=(2, 2048, 4096), key_shape=(2, 2048, 1024),
        head_size=128, rotary_dim=128, max_position_embeddings=8192, base=500000.0,
        is_neox_style=True, dtype="float32", position_dtype="int64",
    ),
    "gptj-6b-prefill-float16": dict(
        positions_shape=(2048,), query_shape=(2048, 4096), key_shape=(2048, 4096),
        head_size=256, rotary_dim=64, max_position_embeddings=2048, base=10000.0,
        is_neox_style=False, dtype="float16", position_dtype="int64",
    ),
    "gptj-6b-decode-b32-float16": dict(
        positions_shape=(32,), query_shape=(32, 4096), key_shape=(32, 4096),
        head_size=256, rotary_dim=64, max_position_embeddings=2048, base=10000.0,
        is_neox_style=False, dtype="float16", position_dtype="int64",
    ),
    "phi-2-query-only-bfloat16": dict(
        positions_shape=(1000,), query_shape=(1000, 2560), key_shape=None,
        head_size=80, rotary_dim=32, max_position_embeddings=2048, base=10000.0,
        is_neox_style=True, dtype="bfloat16", position_dtype="int32",
    ),
}

_DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
_PDT = {"int32": torch.int32, "int64": torch.int64}


def geometry(wid):
    """(tokens, query_heads, key_heads, head_size, rotary_dim, max_position, ...)."""
    w = WORKLOADS[wid]
    tokens = 1
    for extent in w["positions_shape"]:
        tokens *= int(extent)
    head_size = w["head_size"]
    return dict(
        tokens=tokens,
        query_heads=w["query_shape"][-1] // head_size,
        key_heads=0 if w["key_shape"] is None else w["key_shape"][-1] // head_size,
        head_size=head_size,
        rotary_dim=w["rotary_dim"],
        max_position=w["max_position_embeddings"],
        base=w["base"],
        neox=w["is_neox_style"],
        dtype=w["dtype"],
        position_dtype=w["position_dtype"],
    )


def load_module(path):
    name = "rope_variant_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_inputs(wid, seed=0):
    g = geometry(wid)
    gen = torch.Generator().manual_seed(seed)
    tokens = g["tokens"]
    q_cols = g["query_heads"] * g["head_size"]
    k_cols = g["key_heads"] * g["head_size"]
    pos = torch.randint(0, g["max_position"], (tokens,), generator=gen,
                        dtype=torch.int64).to(_PDT[g["position_dtype"]])
    query = torch.randn((tokens, q_cols), generator=gen, dtype=torch.float32).to(_DT[g["dtype"]])
    key = None
    if g["key_heads"]:
        key = torch.randn((tokens, k_cols), generator=gen,
                          dtype=torch.float32).to(_DT[g["dtype"]])
    # cos/sin tables: the caller's already-rounded inputs, (P, R/2).
    angles = torch.rand((g["max_position"], g["rotary_dim"] // 2), generator=gen,
                        dtype=torch.float32) * 6.283185307179586
    cos = angles.cos().to(_DT[g["dtype"]])
    sin = angles.sin().to(_DT[g["dtype"]])
    return pos, query, key, cos, sin


def build(mod, wid, units_per_block=None, staged=False):
    g = geometry(wid)
    upb = mod._UNITS_PER_BLOCK if units_per_block is None else units_per_block
    return mod._build(
        g["query_heads"], g["key_heads"], g["head_size"], g["rotary_dim"],
        g["max_position"], g["dtype"], g["position_dtype"], None, g["neox"],
        upb, staged,
    )


CHILD = r'''
import importlib.util, sys, torch, torch_npu
torch.npu.set_device(0)
path, args_path, upb, staged = {path!r}, {args!r}, {upb!r}, {staged!r}
spec = importlib.util.spec_from_file_location("variant_mod", path)
mod = importlib.util.module_from_spec(spec)
sys.modules["variant_mod"] = mod
spec.loader.exec_module(mod)
pos, query, key, cos, sin, key_heads = torch.load(args_path)
pos, query, cos, sin = pos.to("npu"), query.to("npu"), cos.to("npu"), sin.to("npu")
key = None if key is None else key.to("npu")
g = {geom!r}
fn = mod._build(g["query_heads"], g["key_heads"], g["head_size"], g["rotary_dim"],
                g["max_position"], g["dtype"], g["position_dtype"], None, g["neox"],
                upb, staged)
torch.npu.synchronize()
mod._launch(fn, pos, query, key, cos, sin, None, key_heads)
torch.npu.synchronize()
'''


def parse_capture(out_dir):
    """Sum every captured ``Task Duration(us)`` row; return (total, rows)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "**", "OpBasicInfo*.csv"),
                                 recursive=True)):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                name = (row.get("Op Name") or row.get("Op Name(msprof)") or "").strip()
                raw = (row.get("Task Duration(us)") or "").strip()
                if raw:
                    rows.append((path, name, row))
    if not rows:
        raise RuntimeError("no OpBasicInfo rows captured")
    total = sum(float(r[2]["Task Duration(us)"]) for r in rows)
    return total, rows


def run_capture(module_path, wid, out_dir, metrics, upb, staged, warmup):
    g = geometry(wid)
    tmp = tempfile.mkdtemp(prefix="rope_bench_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        args_path = os.path.join(tmp, "args.pt")
        pos, query, key, cos, sin = make_inputs(wid)
        torch.save([pos, query, key, cos, sin, g["key_heads"]], args_path)
        del pos, query, key, cos, sin
        torch.npu.empty_cache()
        script = os.path.join(tmp, "run.py")
        with open(script, "w") as f:
            f.write(CHILD.format(path=module_path, args=args_path, upb=upb,
                                 staged=staged, geom=g))
        env = dict(os.environ)
        if warmup:
            r = subprocess.run([sys.executable, script], cwd=os.getcwd(), env=env,
                               capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                raise RuntimeError(f"warm-up process failed: {r.stdout[-2000:]}{r.stderr[-3000:]}")
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        os.chmod(out_dir, 0o700)
        cmd = [
            "msopprof", f"--output={out_dir}", "--launch-count=1", "--warm-up=0",
            f"--aic-metrics={metrics}", "--kernel-name=main", sys.executable, script,
        ]
        res = subprocess.run(cmd, cwd=os.getcwd(), env=env, capture_output=True,
                             text=True, timeout=1800)
        if res.returncode != 0:
            raise RuntimeError(f"msopprof rc={res.returncode}: "
                               f"{res.stdout[-2000:]}{res.stderr[-3000:]}")
        return parse_capture(out_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--units-per-block", type=int, default=None)
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--metrics", default="BasicInfo")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--no-l0", action="store_true", help="skip the --level L0 run")
    ap.add_argument("--dump-src", action="store_true", help="print generated source and exit")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    module = os.path.abspath(args.module)
    here = os.path.dirname(os.path.abspath(__file__))
    perf_opt = os.path.dirname(here)
    upb = None
    staged = bool(args.staged)

    if args.dump_src:
        mod = load_module(module)
        fn = build(mod, args.workload, args.units_per_block, staged)
        src = None
        for attr in ("get_kernel_source", "kernel_source"):
            if hasattr(fn, attr):
                src = getattr(fn, attr)() if callable(getattr(fn, attr)) else getattr(fn, attr)
                break
        if src is None:
            src = str(fn)
        out = os.path.join(perf_opt, "profiles", "src",
                           os.path.basename(module)[:-3] + "_" + args.workload + ".cce")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(src)
        print(f"source -> {out}  ({len(src)} chars, {src.count(chr(10))} lines)")
        return 0

    if not args.no_l0:
        r = subprocess.run([sys.executable, module, "--level", "L0"], cwd=os.path.dirname(module),
                           capture_output=True, text=True, timeout=3600)
        ok = "cases passed" in r.stdout and "0 case(s) failed" not in r.stdout
        log_dir = os.path.join(perf_opt, "logs", "l0")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, os.path.basename(module)[:-3] + ".log"), "w") as f:
            f.write(r.stdout + r.stderr)
        print(f"[L0] rc={r.returncode} ok={ok}")
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            return 1

    # warm up / compile in this process too, so a bad variant fails fast
    mod = load_module(module)
    build(mod, args.workload, args.units_per_block, staged)
    upb = mod._UNITS_PER_BLOCK if args.units_per_block is None else args.units_per_block

    out_root = os.path.abspath(args.out_root or os.path.join(perf_opt, "profiles",
                                                             "experiments"))
    tag = args.tag or (os.path.basename(module)[:-3] + "_" + args.workload)
    results, per_op = [], None
    for rep in range(args.reps):
        out_dir = os.path.join(out_root, f"{tag}_r{rep}")
        total, rows = run_capture(module, args.workload, out_dir, args.metrics, upb, staged,
                                  warmup=(rep == 0))
        names = sorted({r[1] for r in rows})
        results.append(total)
        per_op = {r[1]: float(r[2]["Task Duration(us)"]) for r in rows}
        print(json.dumps({"run": os.path.basename(out_dir), "total_us": round(total, 3),
                          "rows": len(rows), "ops": names, "per_op": per_op}), flush=True)
        if len(rows) != 1:
            print(f"[WARN] {len(rows)} captured rows (expected one launch per call)")
    med = sorted(results)[len(results) // 2]
    summary = {"module": os.path.basename(module), "workload": args.workload,
               "samples": [round(v, 3) for v in results], "median_us": round(med, 3),
               "per_op": per_op, "metrics": args.metrics}
    print(json.dumps(summary))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
