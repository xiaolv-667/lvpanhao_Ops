"""Single-invocation msopprof driver for one paged_cache_write kernel variant.

Measurement protocol (task platform `ascend-950pr-9579` / `ascend`):

  ``msopprof --output=<dir> --launch-count=1 --warm-up=0``
  around **one** operator call, then read ``Task Duration(us)`` from
  ``OpBasicInfo*.csv``.  One operator call issues exactly one kernel launch, so
  T_op = "one collection, one call" -- never an NPU event, never a median over
  several launches inside the capture window.

Warm-up launches run in a *separate* process so no unmeasured launch can land in
the capture window; that first process also populates the TileLang JIT cache so
the traced process does not compile inside the capture window.

The variant module must expose the frozen factory
``_paged_cache_write_kernel(heads_kv, dim, physical_tokens, in_dtype, cache_dtype)``
returning a callable ``(block_rows, block_cols) -> compiled callable``, exactly
like the Stage 3 module.  ``_launch`` reproduces the wrapper's ``forward``
(flat-row views, zero-token skip, scales appended only when quantized).

Usage (cwd = the task directory ``examples/paged_cache_write_perf``):

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
# (examples/TileOPs/benchmarks/ops/bench_paged_cache_write.py over
# tileops/manifest/attention.yaml::PagedKVCacheWriteFwdOp).
WORKLOADS = {
    "paged-write-smoke-float16": dict(
        tokens=5, heads_kv=2, dim=64, num_pages=3, page_size=7,
        dtype="float16", cache_dtype="float16", smoke=True,
    ),
    "paged-write-smoke-bfloat16": dict(
        tokens=5, heads_kv=2, dim=64, num_pages=3, page_size=7,
        dtype="bfloat16", cache_dtype="bfloat16", smoke=True,
    ),
    "paged-write-smoke-bfloat16-float8_e4m3fn": dict(
        tokens=5, heads_kv=2, dim=64, num_pages=3, page_size=7,
        dtype="bfloat16", cache_dtype="float8_e4m3fn", smoke=True,
    ),
    "llama-8b-decode-bfloat16-bfloat16": dict(
        tokens=64, heads_kv=8, dim=128, num_pages=4096, page_size=16,
        dtype="bfloat16", cache_dtype="bfloat16",
    ),
    "llama-8b-decode-float16-float16": dict(
        tokens=64, heads_kv=8, dim=128, num_pages=4096, page_size=16,
        dtype="float16", cache_dtype="float16",
    ),
    "llama-8b-prefill-bfloat16-bfloat16": dict(
        tokens=8192, heads_kv=8, dim=128, num_pages=4096, page_size=16,
        dtype="bfloat16", cache_dtype="bfloat16",
    ),
    "qwen3-235b-fp8-bfloat16-float8_e4m3fn": dict(
        tokens=4096, heads_kv=4, dim=128, num_pages=4096, page_size=16,
        dtype="bfloat16", cache_dtype="float8_e4m3fn",
    ),
    "gpt-oss-120b-bfloat16-bfloat16": dict(
        tokens=256, heads_kv=8, dim=64, num_pages=4096, page_size=16,
        dtype="bfloat16", cache_dtype="bfloat16",
    ),
    "odd-tokens-and-page-float16-float16": dict(
        tokens=7, heads_kv=2, dim=64, num_pages=5, page_size=7,
        dtype="float16", cache_dtype="float16",
    ),
    "qwen3-235b-fp8-fp16-float16-float8_e4m3fn": dict(
        tokens=1024, heads_kv=4, dim=128, num_pages=512, page_size=32,
        dtype="float16", cache_dtype="float8_e4m3fn",
    ),
}

_DT = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
}


def geometry(wid):
    w = WORKLOADS[wid]
    return dict(
        tokens=w["tokens"], heads_kv=w["heads_kv"], dim=w["dim"],
        num_pages=w["num_pages"], page_size=w["page_size"],
        dtype=w["dtype"], cache_dtype=w["cache_dtype"],
    )


def load_module(path):
    name = "pcw_variant_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    mod_dir = os.path.dirname(os.path.abspath(path))
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_inputs(wid, seed=0):
    """The benchmark workload's inputs, on the CPU (moved to the NPU by the child)."""
    g = geometry(wid)
    tokens, rw = g["tokens"], g["heads_kv"] * g["dim"]
    physical = g["num_pages"] * g["page_size"]
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn((tokens, rw), generator=gen, dtype=torch.float32).to(_DT[g["dtype"]])
    v = torch.randn((tokens, rw), generator=gen, dtype=torch.float32).to(_DT[g["dtype"]])
    pool = torch.randn((physical, rw), generator=gen, dtype=torch.float32)
    kp = pool.to(_DT[g["cache_dtype"]])
    vp = (-pool).to(_DT[g["cache_dtype"]])
    perm = torch.randperm(physical, generator=gen)[:tokens]
    slots = perm.to(torch.int64)
    quant = g["cache_dtype"] == "float8_e4m3fn"
    ks = torch.tensor([1.25], dtype=torch.float32) if quant else None
    vs = torch.tensor([0.75], dtype=torch.float32) if quant else None
    return k, v, kp, vp, slots, ks, vs


def block_config(mod, wid):
    """The (block_rows, block_cols) the variant's own default policy picks."""
    g = geometry(wid)
    if hasattr(mod, "pick_config"):
        return mod.pick_config(
            g["tokens"], g["heads_kv"], g["dim"],
            g["num_pages"] * g["page_size"], g["dtype"], g["cache_dtype"],
        )
    cfg = getattr(mod, "_DEFAULT_CONFIG", {"block_rows": 16, "block_cols": 128})
    return cfg["block_rows"], cfg["block_cols"]


def launch(fn, k, v, kp, vp, slots, ks, vs):
    """Call the program the way the frozen wrapper's ``forward`` does."""
    tokens = k.shape[0]
    if tokens == 0:
        return
    rw = kp.shape[1]
    args = [k.view(tokens, rw), v.view(tokens, rw), kp, vp, slots]
    if ks is not None:
        args += [ks, vs]
    fn(*args)


CHILD = r'''
import importlib.util, os, sys, torch, torch_npu
torch.npu.set_device(0)
path, args_path, cfg = {path!r}, {args!r}, {cfg!r}
sys.path.insert(0, os.path.dirname(path))
spec = importlib.util.spec_from_file_location("variant_mod", path)
mod = importlib.util.module_from_spec(spec)
sys.modules["variant_mod"] = mod
spec.loader.exec_module(mod)
g = {geom!r}
k, v, kp, vp, slots, ks, vs = torch.load(args_path)
k, v, kp, vp, slots = k.to("npu"), v.to("npu"), kp.to("npu"), vp.to("npu"), slots.to("npu")
ks = None if ks is None else ks.to("npu")
vs = None if vs is None else vs.to("npu")
fn = mod._paged_cache_write_kernel(g["heads_kv"], g["dim"],
                                   g["num_pages"] * g["page_size"],
                                   g["dtype"], g["cache_dtype"])(*cfg)
torch.npu.synchronize()
rw = kp.shape[1]
args = [k.view(k.shape[0], rw), v.view(v.shape[0], rw), kp, vp, slots]
if ks is not None:
    args += [ks, vs]
fn(*args)
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


def run_capture(module_path, wid, out_dir, metrics, cfg, warmup):
    g = geometry(wid)
    tmp = tempfile.mkdtemp(prefix="pcw_bench_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        args_path = os.path.join(tmp, "args.pt")
        k, v, kp, vp, slots, ks, vs = make_inputs(wid)
        torch.save([k, v, kp, vp, slots, ks, vs], args_path)
        del k, v, kp, vp, slots
        torch.npu.empty_cache()
        script = os.path.join(tmp, "run.py")
        with open(script, "w") as f:
            f.write(CHILD.format(path=module_path, args=args_path, cfg=cfg, geom=g))
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
            f"--aic-metrics={metrics}", sys.executable, script,
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
    ap.add_argument("--block-rows", type=int, default=None)
    ap.add_argument("--block-cols", type=int, default=None)
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

    if args.dump_src:
        mod = load_module(module)
        cfg = block_config(mod, args.workload)
        if args.block_rows is not None:
            cfg = (args.block_rows, args.block_cols or cfg[1])
        g = geometry(args.workload)
        fn = mod._paged_cache_write_kernel(
            g["heads_kv"], g["dim"], g["num_pages"] * g["page_size"],
            g["dtype"], g["cache_dtype"])(*cfg)
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
        ok = "[PRECISION_PASS]" in r.stdout
        log_dir = os.path.join(perf_opt, "logs", "l0")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, os.path.basename(module)[:-3] + ".log"), "w") as f:
            f.write(r.stdout + r.stderr)
        print(f"[L0] rc={r.returncode} ok={ok}")
        if r.returncode != 0 or not ok:
            print(r.stdout[-3000:], r.stderr[-3000:])
            return 1

    mod = load_module(module)
    cfg = block_config(mod, args.workload)
    if args.block_rows is not None:
        cfg = (args.block_rows, args.block_cols or cfg[1])
    # warm up / compile in this process too, so a bad variant fails fast
    g = geometry(args.workload)
    mod._paged_cache_write_kernel(
        g["heads_kv"], g["dim"], g["num_pages"] * g["page_size"],
        g["dtype"], g["cache_dtype"])(*cfg)

    out_root = os.path.abspath(args.out_root or os.path.join(perf_opt, "profiles",
                                                             "experiments"))
    tag = args.tag or (os.path.basename(module)[:-3] + "_" + args.workload)
    results, per_op = [], None
    for rep in range(args.reps):
        out_dir = os.path.join(out_root, f"{tag}_r{rep}")
        total, rows = run_capture(module, args.workload, out_dir, args.metrics, cfg,
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
               "config": list(cfg),
               "samples": [round(v, 3) for v in results], "median_us": round(med, 3),
               "per_op": per_op, "metrics": args.metrics}
    print(json.dumps(summary))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
