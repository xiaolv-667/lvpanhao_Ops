"""Launch-floor probe for the EngramDecode launch geometry (Stage 4 diagnostic).

Same measurement protocol as ``bench_one.py`` (msopprof, kernel-only
``Task Duration(us)``, one operator call per capture, median of `--reps`
independent capture processes).  The probed kernel is a minimal 1-D-grid TileLang
kernel: per block one row GM->UB load + one row UB->GM store, nothing else.  Its
duration is the *floor* a kernel of that (grid, row width) geometry cannot beat,
so it bounds what remains to be optimised in the real launch.

Usage (cwd = examples/TileOPs):

    python probe_floor.py --grids 16,4 --widths 512,256 --reps 3
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

CHILD = r'''
import sys, torch, torch_npu, tilelang, tilelang.ascend.language as T
import functools
torch.npu.set_device(0)


@functools.lru_cache(maxsize=None)
def _build(grid, width, dtype_str):
    @T.prim_func
    def engram_floor_probe(x: T.Tensor((grid, width), dtype_str),
                           y: T.Tensor((grid, width), dtype_str)):
        with T.Kernel(grid) as bx:
            buf = T.alloc_shared((width,), dtype_str)
            T.copy(x[bx, :], buf)
            T.copy(buf, y[bx, :])

    return tilelang.compile(engram_floor_probe, target="ascend", out_idx=[])


grid, width, dt = {geom!r}
dtype = getattr(torch, dt)
x = torch.randn(grid, width, dtype=dtype, device="npu")
y = torch.empty(grid, width, dtype=dtype, device="npu")
fn = _build(grid, width, dt)
torch.npu.synchronize()
fn(x, y)
torch.npu.synchronize()
print("ok", float(y.abs().mean()))
'''


def capture(geom, out_root, tag, warmup, reps):
    tmp = tempfile.mkdtemp(prefix="probe_floor_", dir=os.environ.get("TMPDIR", "/workspace"))
    try:
        cwd = os.getcwd()
        script = os.path.join(tmp, "run.py")
        with open(script, "w") as f:
            f.write(CHILD.format(geom=geom))
        env = dict(os.environ)
        env["PYTHONPATH"] = cwd + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if warmup:
            r = subprocess.run([sys.executable, script], cwd=cwd, env=env,
                               capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                raise RuntimeError(f"warmup failed: {r.stderr[-3000:]}")
        samples = []
        for rep in range(reps):
            out = os.path.join(out_root, f"{tag}_r{rep}")
            if os.path.isdir(out):
                shutil.rmtree(out)
            os.makedirs(out)
            os.chmod(out, 0o700)
            cmd = ["msopprof", f"--output={out}", "--launch-count=1", "--warm-up=0",
                   "--kernel-name=engram", "--aic-metrics=BasicInfo",
                   sys.executable, script]
            res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                                 timeout=3600)
            if res.returncode != 0:
                raise RuntimeError(f"msopprof failed: {res.stderr[-3000:]}")
            total, rows = 0.0, 0
            for path in sorted(glob.glob(os.path.join(out, "**", "OpBasicInfo*.csv"),
                                         recursive=True)):
                with open(path, newline="") as fh:
                    for row in csv.DictReader(fh):
                        name = (row.get("Op Name") or "").strip()
                        raw = (row.get("Task Duration(us)") or "").strip()
                        if name and raw:
                            total += float(raw)
                            rows += 1
            if not rows:
                raise RuntimeError("no OpBasicInfo rows captured")
            samples.append(total)
        return samples
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="16,8,4")
    ap.add_argument("--widths", default="512,256")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out_root = os.path.abspath(args.out_root or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "profiles", "floor"))
    os.makedirs(out_root, exist_ok=True)
    rows = []
    for grid in [int(g) for g in args.grids.split(",")]:
        for width in [int(w) for w in args.widths.split(",")]:
            tag = f"floor_g{grid}_w{width}_{args.dtype}"
            samples = capture((grid, width, args.dtype), out_root, tag, 1, args.reps)
            s = sorted(samples)
            med = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
            rows.append({"grid": grid, "width": width, "dtype": args.dtype,
                         "samples": [round(v, 3) for v in samples],
                         "median_us": round(med, 3)})
            print(json.dumps(rows[-1]), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
