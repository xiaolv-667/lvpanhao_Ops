"""TRAP-950-rowstore-mte3-drain -- repro.

First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / AscendPO950PR_9579 (dav-3510), 2026-11-24;
origin_task paged_cache_write-paged_cache_write-20261124T171256Z (Stage 4).
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa

Claim: a scattered row store writes one row per destination, and the Ascend
scheduler drains the store pipe before *every* UB->GM ``T.copy``
(``asc_sync_pipe(PIPE_MTE3)``, emitted by ``insert_sync.cc`` when the producer
and consumer pipe are the same).  A row-sized DMA store therefore costs a fixed
~0.25-0.3 us of core time, i.e. it scales with the number of rows, not with the
bytes.  For a 2 KB row that is ~7 GB/s per core; the same payload as one 2-D
copy of 32 rows runs at the DMA ceiling instead.

The countermeasure measured here: keep the contiguous *load* on the DMA path and
let the vector unit issue the scattered *store* (16 B/lane, ``slot >= 0`` as the
lane predicate).  It has no per-row drain and wins once a row is small.

Code shapes asserted (no device needed): each store form compiles and the DMA
store form contains ``asc_sync_pipe(PIPE_MTE3)`` while the lane store form does
not.

Run (device required; keep the repository root off PYTHONPATH so the installed
``tilelang`` is imported):

    python TRAP-950-rowstore-mte3-drain.py            # compile + code-shape asserts
    python TRAP-950-rowstore-mte3-drain.py --time     # also measure the three forms
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

TOKENS = 8192  # rows to scatter
RW = 1024  # elements per row (2 KB in bf16)
ROWS = 32  # rows per block (the tile the tuned kernel uses here)
PHYS = 65536
THREADS = 512


def build(mode):
    tokens = T.dynamic("tokens")

    @T.prim_func
    def main(
        X: T.Tensor((tokens, RW), "bfloat16"),
        Y: T.Tensor((tokens, RW), "bfloat16"),
        XPages: T.Tensor((PHYS, RW), "bfloat16"),
        YPages: T.Tensor((PHYS, RW), "bfloat16"),
        Slots: T.Tensor((tokens,), "int64"),
    ) -> None:
        with T.Kernel(T.ceildiv(tokens, ROWS)) as bx:
            row0 = bx * ROWS
            nr = T.min(ROWS, tokens - row0)
            one = T.min(nr, 1)  # provably 1; a literal 1 collapses the region
            x_ub = T.alloc_shared((ROWS, RW), "bfloat16")
            y_ub = T.alloc_shared((ROWS, RW), "bfloat16")
            T.copy(X[row0 : row0 + nr, 0:RW], x_ub[0:nr, 0:RW])
            T.copy(Y[row0 : row0 + nr, 0:RW], y_ub[0:nr, 0:RW])
            if mode == "rowstore":
                for i in T.unroll(ROWS):
                    if row0 + i < tokens:
                        slot = T.cast(Slots[row0 + i], "int32")
                        if slot >= 0:
                            T.copy(x_ub[i : i + 1, 0:RW], XPages[slot : slot + one, 0:RW])
                            T.copy(y_ub[i : i + 1, 0:RW], YPages[slot : slot + one, 0:RW])
            elif mode == "lanestore":
                slot_ub = T.alloc_shared((ROWS,), "int32")
                with T.SimtVF(threads=THREADS):
                    for i in T.Parallel(ROWS):
                        slot_ub[i] = T.cast(Slots[T.min(row0 + i, tokens - 1)], "int32")
                    for i, c in T.Parallel(ROWS, RW):
                        slot = slot_ub[i]
                        if (row0 + i < tokens) and (slot >= 0):
                            XPages[slot, c] = x_ub[i, c]
                            YPages[slot, c] = y_ub[i, c]
            elif mode == "blockstore":
                # one 2-D DMA per pool for the whole tile: contiguous
                # destination, the control case for the per-row drain
                T.copy(x_ub[0:nr, 0:RW], XPages[row0 : row0 + nr, 0:RW])
                T.copy(y_ub[0:nr, 0:RW], YPages[row0 : row0 + nr, 0:RW])
            else:
                raise SystemExit(f"unknown mode {mode}")

    return main


def compile_one(mode):
    fn = tilelang.compile(build(mode), target="ascend", out_idx=[])
    src = None
    for attr in ("get_kernel_source", "kernel_source"):
        if hasattr(fn, attr):
            src = getattr(fn, attr)() if callable(getattr(fn, attr)) else getattr(fn, attr)
            break
    return fn, (src or "")


CHILD = r'''
import importlib.util, sys, torch, torch_npu
torch.npu.set_device(0)
spec = importlib.util.spec_from_file_location("rowstore_repro", {repro_path!r})
mod = importlib.util.module_from_spec(spec)
sys.modules["rowstore_repro"] = mod
spec.loader.exec_module(mod)
fn, _ = mod.compile_one({mode!r})
torch.manual_seed(0)
# inputs are materialised on the host and only moved, so the capture window
# holds exactly one kernel launch (a randperm/randn on the device would be
# captured first and inflate the Task Duration)
args = torch.load({args_path!r})
x, y, xp, yp, slots = [t.npu() for t in args]
torch.npu.synchronize()
fn(x, y, xp, yp, slots)
torch.npu.synchronize()
'''


def measure(mode):
    out_dir = os.path.join(tempfile.mkdtemp(prefix="rowstore_", dir=os.environ.get("TMPDIR", "/tmp")),
                           "cap")
    repro_dir = os.path.dirname(os.path.abspath(__file__))
    assert os.path.isfile(os.path.join(repro_dir, "TRAP-950-rowstore-mte3-drain.py"))
    script = os.path.join(os.path.dirname(out_dir), "run.py")
    args_path = os.path.join(os.path.dirname(out_dir), "args.pt")
    gen = torch.Generator().manual_seed(0)
    torch.save(
        [
            torch.randn((TOKENS, RW), generator=gen, dtype=torch.float32).to(torch.bfloat16),
            torch.randn((TOKENS, RW), generator=gen, dtype=torch.float32).to(torch.bfloat16),
            torch.randn((PHYS, RW), generator=gen, dtype=torch.float32).to(torch.bfloat16),
            torch.randn((PHYS, RW), generator=gen, dtype=torch.float32).to(torch.bfloat16),
            torch.randperm(PHYS, generator=gen)[:TOKENS].to(torch.int64),
        ],
        args_path,
    )
    with open(script, "w") as f:
        f.write(CHILD.format(repro_path=os.path.join(repro_dir, "TRAP-950-rowstore-mte3-drain.py"),
                             args_path=args_path, mode=mode, tokens=TOKENS, rw=RW, phys=PHYS))
    # warm-up in its own process so the capture sees one launch and no compile
    subprocess.run([sys.executable, script], check=True, capture_output=True, text=True,
                   timeout=1800)
    os.makedirs(out_dir)
    cmd = ["msopprof", f"--output={out_dir}", "--launch-count=1", "--warm-up=0",
           "--aic-metrics=BasicInfo", sys.executable, script]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if res.returncode != 0:
        raise RuntimeError(res.stdout[-1500:] + res.stderr[-1500:])
    total, rows = 0.0, 0
    for path in glob.glob(os.path.join(out_dir, "**", "OpBasicInfo*.csv"), recursive=True):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                raw = (row.get("Task Duration(us)") or "").strip()
                if raw:
                    total += float(raw)
                    rows += 1
    shutil.rmtree(os.path.dirname(out_dir), ignore_errors=True)
    if not rows:
        raise RuntimeError("no OpBasicInfo rows captured")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    args = ap.parse_args()
    sources = {}
    for mode in ("rowstore", "lanestore", "blockstore"):
        _, src = compile_one(mode)
        sources[mode] = src
        print(f"[compiled] {mode}: {len(src.splitlines())} lines, "
              f"asc_sync_pipe(PIPE_MTE3) x{src.count('asc_sync_pipe(PIPE_MTE3)')}, "
              f"ub2gm x{src.count('asc_copy_ub2gm')}, gm2ub x{src.count('asc_copy_gm2ub')}")
    # the per-row DMA store drains the store pipe once per row; the lane store
    # and the single 2-D block store do not
    assert sources["rowstore"].count("asc_sync_pipe(PIPE_MTE3)") >= 1
    assert sources["lanestore"].count("asc_sync_pipe(PIPE_MTE3)") == 0
    assert sources["blockstore"].count("asc_sync_pipe(PIPE_MTE3)") <= 1
    print("[TRAP_CONFIRMED] per-row DMA store carries a PIPE_MTE3 drain; the lane "
          "store and the single block store do not")
    if not args.time:
        print("pass --time to measure the three forms")
        return
    got = {mode: measure(mode) for mode in ("blockstore", "rowstore", "lanestore")}
    payload = TOKENS * RW * 2 * 4  # two reads + two writes of 2 KB rows
    for mode, us in got.items():
        print(f"[time] {mode:10s} {us:9.2f} us")
    # a contiguous single-copy store is the cheapest; the per-row store pays the
    # drain, and the lane store beats it because the drain is not per row
    assert got["blockstore"] < got["rowstore"], got
    assert got["lanestore"] < got["rowstore"], got
    print(f"[TIME_CONFIRMED] row store {got['rowstore']:.1f} us vs lane store "
          f"{got['lanestore']:.1f} us vs one block DMA {got['blockstore']:.1f} us "
          f"({payload / 1e6 / got['rowstore'] * 1e3:.0f} GB/s over the row store)")


if __name__ == "__main__":
    main()
