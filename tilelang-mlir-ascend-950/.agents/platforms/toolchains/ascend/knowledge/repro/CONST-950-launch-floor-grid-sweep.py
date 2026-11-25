"""CONST-950-launch-floor-grid-sweep + PL-950-serial-copy-chain-hoist -- repro.

repro for entries: CONST-950-launch-floor-grid-sweep, PL-950-serial-copy-chain-hoist.
First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-25;
origin_task engram_decode-engram_decode-20261124T171256Z (Stage 4).
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa

Kernel-only launch floor of the ascend-950pr-9579 `Task Duration(us)` metric for the
small 1-D grids a fused decode-step kernel actually uses (4 / 8 / 16 blocks), and the
derived floor of a **two-launch** operator.

Probed kernel: `T.Kernel(grid)`, per block one row GM->UB load + one row UB->GM
store of a 512-element fp16 row, nothing else.  Whatever it measures is the floor a
kernel of that geometry cannot beat, so a two-launch operator cannot go below
2 x that number.

Measured on 2026-11-24 (`perf_opt/experiments/probe_floor.py`, median of 3
independent msopprof captures each): grid 4 = 2.690 us, grid 8 = 2.961 us,
grid 16 = 2.894 us.  This script asserts the band and prints the two-launch floor.

Run (cwd = examples/TileOPs, after `source .agents/platforms/toolchains/ascend/env-950pr.sh`):

    python <this file>            # 1 capture per grid, asserts the band
"""

import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile

CHILD = r'''
import functools, sys, torch, torch_npu, tilelang, tilelang.ascend.language as T
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

# (grid, width, dtype, expected upper bound in us) -- observed 2.69 / 2.96 / 2.89
CASES = [(4, 512, "float16", 3.3), (8, 512, "float16", 3.4), (16, 512, "float16", 3.4)]


def capture(geom):
    tmp = tempfile.mkdtemp(prefix="floor_repro_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        script = os.path.join(tmp, "run.py")
        with open(script, "w") as f:
            f.write(CHILD.format(geom=geom))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.getcwd() + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        subprocess.run([sys.executable, script], check=True, capture_output=True, timeout=3600)
        out = os.path.join(tmp, "prof")
        os.makedirs(out)
        os.chmod(out, 0o700)
        subprocess.run(
            ["msopprof", f"--output={out}", "--launch-count=1", "--warm-up=0",
             "--kernel-name=engram", "--aic-metrics=BasicInfo", sys.executable, script],
            check=True, capture_output=True, timeout=3600, env=env,
        )
        total, rows = 0.0, 0
        for path in glob.glob(os.path.join(out, "**", "OpBasicInfo*.csv"), recursive=True):
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    raw = (row.get("Task Duration(us)") or "").strip()
                    if raw:
                        total += float(raw)
                        rows += 1
        assert rows == 1, f"expected exactly one captured launch, got {rows}"
        return total
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    floors = []
    for grid, width, dt, limit in CASES:
        us = capture((grid, width, dt))
        print(f"grid={grid:3d} width={width} {dt}: Task Duration = {us:.3f} us "
              f"(band <= {limit} us)")
        assert 1.5 < us < limit, f"grid={grid}: {us:.3f} us outside the measured band"
        floors.append(us)
    two_launch = min(floors) + min(floors)
    print(f"two-launch floor (2 x min single-launch floor) = {two_launch:.2f} us")
    assert 5.0 < two_launch < 6.4, two_launch
    print("REPRO_OK")


if __name__ == "__main__":
    main()
