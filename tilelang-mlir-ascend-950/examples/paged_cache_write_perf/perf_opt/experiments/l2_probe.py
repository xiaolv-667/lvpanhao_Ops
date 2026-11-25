"""Is the prefill case's achieved bandwidth above the nominal peak a defect?

The acceptance guard marks a case ``invalid`` when the roofline bytes divided by
the measured ``Task Duration`` exceed the platform's nominal HBM peak (1.6 TB/s)
by more than 1%.  The tuned prefill case lands at 1.62 TB/s, i.e. just over the
line, so this probe separates the two candidate explanations:

  (a) the timing is wrong (kernel exits before the writes land), or
  (b) the roofline counts 4 streams of HBM traffic while the 256 MB of page
      pools sit under a 128 MB L2, so part of the write traffic is absorbed and
      never reaches HBM inside the measured window.

Method: same kernel, same row count, same per-row work; only the pool footprint
changes (working set 64 MB -> 512 MB per pool).  If (b) holds, the measured time
grows with the footprint and the achieved "roofline bandwidth" falls back to (or
below) the nominal peak; if (a) holds, the time stays flat with the footprint.

    python l2_probe.py [--reps 3]
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import torch
import torch_npu  # noqa: F401

TOKENS, HEADS, DIM = 8192, 8, 128
ROW = HEADS * DIM
POOLS = {"manifest_64MB": 4096, "big_512MB": 32768}  # num_pages x 16 rows
PAGE_SIZE = 16
ROOFLINE_BYTES = 2 * TOKENS * ROW * (2 + 2) + TOKENS * 8 + 8

CHILD = r'''
import sys, torch, torch_npu
torch.npu.set_device(0)
import tileops  # noqa: F401  (package path comes from cwd)
from tileops.ops.attention.paged_cache_write import PagedKVCacheWriteFwdOp
k, v, kp, vp, slots = [t.npu() for t in torch.load({args_path!r})]
torch.npu.synchronize()
op = PagedKVCacheWriteFwdOp()
op(k, v, kp, vp, slots)
torch.npu.synchronize()
'''


def make_inputs(num_pages):
    gen = torch.Generator().manual_seed(0)
    row = (HEADS, DIM)
    k = torch.randn((TOKENS, *row), generator=gen, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn((TOKENS, *row), generator=gen, dtype=torch.float32).to(torch.bfloat16)
    pool = torch.randn((num_pages, PAGE_SIZE, *row), generator=gen, dtype=torch.float32)
    kp = pool.to(torch.bfloat16)
    vp = (-pool).to(torch.bfloat16)
    physical = num_pages * PAGE_SIZE
    slots = torch.randperm(physical, generator=gen)[:TOKENS].to(torch.int64)
    return [k, v, kp, vp, slots]


def capture(num_pages, reps):
    work = tempfile.mkdtemp(prefix="l2probe_", dir=os.environ.get("TMPDIR", "/tmp"))
    args_path = os.path.join(work, "args.pt")
    torch.save(make_inputs(num_pages), args_path)
    torch.npu.empty_cache()
    script = os.path.join(work, "run.py")
    with open(script, "w") as f:
        f.write(CHILD.format(args_path=args_path))
    env = dict(os.environ)
    vals = []
    for rep in range(reps):
        # warm-up in its own process so the capture holds one launch, no compile
        subprocess.run([sys.executable, script], check=True, capture_output=True,
                       text=True, timeout=1800, env=env)
        out_dir = os.path.join(work, f"cap{rep}")
        os.makedirs(out_dir)
        cmd = ["msopprof", f"--output={out_dir}", "--launch-count=1", "--warm-up=0",
               "--aic-metrics=BasicInfo", sys.executable, script]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
        if res.returncode != 0:
            raise RuntimeError(res.stdout[-1200:] + res.stderr[-1200:])
        total = 0.0
        for path in glob.glob(os.path.join(out_dir, "**", "OpBasicInfo*.csv"), recursive=True):
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    raw = (row.get("Task Duration(us)") or "").strip()
                    if raw:
                        total += float(raw)
        vals.append(total)
    shutil.rmtree(work, ignore_errors=True)
    vals.sort()
    return vals[len(vals) // 2], vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    print(f"roofline bytes (the operator's own model): {ROOFLINE_BYTES}")
    for name, num_pages in POOLS.items():
        med, vals = capture(num_pages, args.reps)
        tbs = ROOFLINE_BYTES / (med * 1e-6) / 1e12
        footprint = 2 * num_pages * PAGE_SIZE * ROW * 2 / 2**20
        print(f"[{name}] pool rows={num_pages * PAGE_SIZE} footprint={footprint:.0f} MB "
              f"median={med:.3f} us samples={[round(v, 3) for v in vals]} "
              f"roofline-BW={tbs:.3f} TB/s ({100 * tbs / 1.6:.1f}% of nominal 1.6)")
    print("[note] the same kernel and the same 8192 written rows in both rows; only the")
    print("       pool footprint (and therefore the L2-resident fraction) differs.")


if __name__ == "__main__":
    sys.exit(main())
