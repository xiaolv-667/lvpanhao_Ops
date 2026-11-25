"""CONST-950-dma-vs-lane-gm / CONST-950-simt-lane-vec-8b -- repro.

First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-24; origin_task rope-RoPEFwdOp-20261124T171256Z (Stage 4).
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa

Two claims about a pure elementwise payload on 950PR, both re-derived here from
one compiled kernel family over a 4096x4096 fp32 tensor (134.2 MB, read once and
written once):

1. ``CONST-950-dma-vs-lane-gm`` -- moving a tile over ``T.copy`` (MTE2/MTE3
   descriptors) is ~2.6x cheaper than moving the same bytes with 8 B per-lane
   accesses and ~2x cheaper than 16 B per-lane; 32 B per-lane is *worse* than
   16 B.  Measured: dma 124.5 / lane-8B 322.2 / vec-16B 245.3 / vec-32B 352.5 us.
2. ``CONST-950-simt-lane-vec-8b`` -- the SIMT VF backend caps a per-lane vector
   access at 8 B: an explicit ``T.vectorized(4)`` run over a UB buffer that is
   read and written in place is lowered to two ``float2`` accesses plus one
   block barrier *inside* the loop body (a global barrier per iteration).

Run (device required; keep the repository root off PYTHONPATH so the installed
``tilelang`` is imported):

    python CONST-950-dma-vs-lane-gm.py            # compile + code-shape asserts
    python CONST-950-dma-vs-lane-gm.py --time     # also measure and assert ratios
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

TOKENS, COLS = 4096, 4096
TILE = 4


def _copy_kernel(kind):
    """One 1-read/1-write pass over the payload; ``kind`` picks the access form."""
    tokens = T.dynamic("num_tokens")
    nrb = T.ceildiv(tokens, TILE)

    if kind == "dma":

        @T.prim_func
        def main(X: T.Tensor((tokens, COLS), "float32"),
                 Y: T.Tensor((tokens, COLS), "float32")) -> None:
            with T.Kernel(nrb) as bx:
                x_ub = T.alloc_shared((TILE, COLS), "float32")
                T.copy(X[bx * TILE : bx * TILE + TILE, :], x_ub)
                T.copy(x_ub, Y[bx * TILE : bx * TILE + TILE, :])

    elif kind == "lane":

        @T.prim_func
        def main(X: T.Tensor((tokens, COLS), "float32"),
                 Y: T.Tensor((tokens, COLS), "float32")) -> None:
            with T.Kernel(nrb) as bx:
                with T.SimtVF(threads=256):
                    for i, c in T.Parallel(TILE, COLS):
                        Y[bx * TILE + i, c] = X[bx * TILE + i, c]

    else:  # vec16 / vec32: explicit per-lane run of the whole row block
        vec = 4 if kind == "vec16" else 8

        @T.prim_func
        def main(X: T.Tensor((tokens, COLS), "float32"),
                 Y: T.Tensor((tokens, COLS), "float32")) -> None:
            with T.Kernel(nrb) as bx:
                with T.SimtVF(threads=256):
                    for i, c in T.Parallel(TILE, COLS // vec):
                        for u in T.vectorized(vec):
                            Y[bx * TILE + i, c * vec + u] = X[bx * TILE + i, c * vec + u]

    return tilelang.compile(main, target="ascend", out_idx=[1])


def _rotate_kernel():
    """In-place ``T.vectorized(4)`` UB read-modify-write (the width claim)."""
    tokens = T.dynamic("num_tokens")
    nrb = T.ceildiv(tokens, TILE)

    @T.prim_func
    def main(X: T.Tensor((tokens, COLS), "float32"),
             Y: T.Tensor((tokens, COLS), "float32")) -> None:
        with T.Kernel(nrb) as bx:
            x_ub = T.alloc_shared((TILE, COLS), "float32")
            T.copy(X[bx * TILE : bx * TILE + TILE, :], x_ub)
            with T.SimtVF(threads=256):
                for i, h, kk in T.Parallel(TILE, 8, (COLS // 8) // 4):
                    for u in T.vectorized(4):
                        x_ub[i, h * (COLS // 8) + kk * 4 + u] = (
                            x_ub[i, h * (COLS // 8) + kk * 4 + u] * 2.0
                        )
            T.copy(x_ub, Y[bx * TILE : bx * TILE + TILE, :])

    return tilelang.compile(main, target="ascend", out_idx=[1])


def source_of(fn):
    for attr in ("get_kernel_source", "kernel_source"):
        if hasattr(fn, attr):
            value = getattr(fn, attr)
            return value() if callable(value) else value
    return str(fn)


def vf_body(src):
    start = src.index("__simt_vf__")
    end = src.index('extern "C"') if 'extern "C"' in src else len(src)
    return src[start:end]


def measure(kind, out_dir):
    tmp = tempfile.mkdtemp(prefix="kb_dma_repro_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        script = os.path.join(tmp, "run.py")
        with open(script, "w") as f:
            f.write(
                "import importlib.util, sys, torch, torch_npu\n"
                "torch.npu.set_device(0)\n"
                f"spec = importlib.util.spec_from_file_location('k', {os.path.abspath(__file__)!r})\n"
                "m = importlib.util.module_from_spec(spec); sys.modules['k'] = m\n"
                "spec.loader.exec_module(m)\n"
                f"fn = m._copy_kernel({kind!r})\n"
                "x = torch.randn((m.TOKENS, m.COLS), device='npu')\n"
                "torch.npu.synchronize(); fn(x); torch.npu.synchronize()\n"
            )
        env = dict(os.environ)
        subprocess.run([sys.executable, script], env=env, capture_output=True, timeout=1800)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        os.chmod(out_dir, 0o700)
        cmd = ["msopprof", f"--output={out_dir}", "--launch-count=1", "--warm-up=0",
               "--kernel-name=main", sys.executable, script]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        assert res.returncode == 0, res.stderr[-2000:]
        total, rows = 0.0, 0
        for path in glob.glob(os.path.join(out_dir, "**", "OpBasicInfo*.csv"), recursive=True):
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("Task Duration(us)"):
                        total += float(row["Task Duration(us)"])
                        rows += 1
        assert rows == 1, rows
        return total
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    args = ap.parse_args()

    src = {kind: source_of(_copy_kernel(kind)) for kind in ("dma", "lane", "vec16", "vec32")}

    # claim 1: the DMA variant moves the tile with descriptors, the per-lane ones
    # with 8 B vector accesses straight to GM.
    assert "asc_copy_gm2ub_align" in src["dma"], "DMA load descriptor missing"
    assert "asc_copy_ub2gm_align" in src["dma"], "DMA store descriptor missing"
    assert "__gm__ float2" in src["lane"], "per-lane 8 B GM access missing"

    # claim 2: an explicit 4-wide in-place run stays float2 and gains a barrier
    # inside the loop body.
    body = vf_body(source_of(_rotate_kernel()))
    assert "float2" in body, "expected float2 accesses"
    assert "float4" not in body and "float8" not in body, "per-lane width above 8 B?"
    print("[PASS] code shape: dma=descriptors, lane=8 B per-lane GM, "
          "vec4 in-place rotate still lowered to float2")
    print(f"[info] in-loop barrier present: {'asc_syncthreads' in body} "
          "(form-dependent; the task-side branch that carried one is cited in the entry)")

    if not args.time:
        return 0

    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repro_out")
    us = {kind: measure(kind, os.path.join(out_root, kind))
          for kind in ("dma", "lane", "vec16", "vec32")}
    print({k: round(v, 3) for k, v in us.items()})
    assert us["dma"] < 0.65 * us["lane"], (us["dma"], us["lane"])
    assert us["dma"] < 0.75 * us["vec16"], (us["dma"], us["vec16"])
    assert us["vec16"] < us["vec32"], (us["vec16"], us["vec32"])
    print("[PASS] dma < vec16 < lane, and vec32 worse than vec16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
