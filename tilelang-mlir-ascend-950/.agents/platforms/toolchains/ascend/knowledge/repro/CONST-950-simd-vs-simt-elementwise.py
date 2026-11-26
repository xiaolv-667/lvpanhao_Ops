"""CONST-950-simd-vs-simt-elementwise -- repro.

First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-25;
origin_task rope-RoPEFwdOp-20261125T120xxxZ (Stage 4 round 2).
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa

Claim: for a bandwidth/elementwise kernel whose UB tile is already streamed by ``T.copy``,
the elementwise pass is far cheaper in ``T.SimdVF`` (256 B vector registers, ``T.simd.vld`` /
``vmul`` / ``vsub`` / ``vsts``) than in ``T.SimtVF`` per-lane form, because one SIMT
per-lane machine step retires only ~8 B/lane at roughly 10 cycles per 64-lane step while a
vector-register step moves a full 256 B.  Measured on the RoPE rotate shape
(4 rows x 1024 fp32 columns per tile, 8 heads of 128 columns, half = 64):
SIMT 145.3 us -> SIMD 121.4 us for the same 169.9 MB payload (16.5% faster; the elementwise
pass itself drops from 9.33 to 5.22 us per block, ``aiv_vec_ratio`` 0.78 -> 0.53).

Run (device required; keep the repository root off PYTHONPATH so the installed tilelang is used):

    python CONST-950-simd-vs-simt-elementwise.py            # code-shape asserts
    python CONST-950-simd-vs-simt-elementwise.py --time     # also measure and assert simd < simt
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

TOKENS, COLS = 4096, 4096      # payload of the llama3-8b-b2-2k case, fp32
BT, HPC, DIM, HALF = 4, 8, 128, 64
NSUB, NSTAGE = 8, 3


def _kernel(kind):
    """One tile stream + one rotate pass; ``kind`` selects the pass's execution model."""
    tokens = T.dynamic("num_tokens")
    step = BT * NSUB
    nrb = T.ceildiv(tokens, step)

    @T.macro
    def side(X, Y, x_ub, cs_ub, sn_ub, hpc, cols, rb, cb):
        c0 = cb * cols
        base = rb * step
        for s in T.Pipelined(NSUB, num_stages=NSTAGE):
            row0 = base + s * BT
            nr = T.min(BT, tokens - row0)
            T.copy(X[row0 : row0 + nr, c0 : c0 + cols], x_ub[0:nr, 0:cols])
            if kind == "simd":
                with T.SimdVF():
                    full = T.simd.pset(32)
                    for i in T.serial(BT):
                        for ch in T.serial(HALF // 64):
                            cs = T.simd.vld(cs_ub[s * BT + i, ch * 64])
                            sn = T.simd.vld(sn_ub[s * BT + i, ch * 64])
                            for h in T.serial(hpc):
                                lo = h * DIM + ch * 64
                                hi = lo + HALF
                                x_lo = T.simd.vld(x_ub[i, lo])
                                x_hi = T.simd.vld(x_ub[i, hi])
                                r_lo = T.simd.vsub(T.simd.vmul(x_lo, cs), T.simd.vmul(x_hi, sn))
                                r_hi = T.simd.vadd(T.simd.vmul(x_hi, cs), T.simd.vmul(x_lo, sn))
                                T.simd.vsts(x_ub[i, lo], r_lo)
                                T.simd.vsts(x_ub[i, hi], r_hi)
            else:
                with T.SimtVF(threads=256):
                    for i, p in T.Parallel(BT, hpc * HALF):
                        h = p // HALF
                        k = p - h * HALF
                        lo = h * DIM + k
                        hi = lo + HALF
                        cs = cs_ub[s * BT + i, k]
                        sn = sn_ub[s * BT + i, k]
                        x_lo = x_ub[i, lo]
                        x_hi = x_ub[i, hi]
                        x_ub[i, lo] = x_lo * cs - x_hi * sn
                        x_ub[i, hi] = x_hi * cs + x_lo * sn
            T.copy(x_ub[0:nr, 0:cols], Y[row0 : row0 + nr, c0 : c0 + cols])

    @T.prim_func
    def main(
        cos_table: T.Tensor((TOKENS, HALF), "float32"),
        sin_table: T.Tensor((TOKENS, HALF), "float32"),
        query: T.Tensor((tokens, COLS), "float32"),
        query_out: T.Tensor((tokens, COLS), "float32"),
    ) -> None:
        with T.Kernel(nrb * (COLS // (HPC * DIM))) as bx:
            x_ub = T.alloc_shared((BT, HPC * DIM), "float32")
            cs_ub = T.alloc_shared((step, HALF), "float32")
            sn_ub = T.alloc_shared((step, HALF), "float32")
            rb = bx // (COLS // (HPC * DIM))
            cb = bx % (COLS // (HPC * DIM))
            with T.SimtVF(threads=256):
                for i, k in T.Parallel(step, HALF):
                    cs_ub[i, k] = cos_table[T.min(rb * step + i, TOKENS - 1), k]
                for i, k in T.Parallel(step, HALF):
                    sn_ub[i, k] = sin_table[T.min(rb * step + i, TOKENS - 1), k]
            side(query, query_out, x_ub, cs_ub, sn_ub, HPC, HPC * DIM, rb, cb)

    return tilelang.compile(main, target="ascend", out_idx=[3])


def source_of(fn):
    for attr in ("get_kernel_source", "kernel_source"):
        if hasattr(fn, attr):
            value = getattr(fn, attr)
            return value() if callable(value) else value
    return str(fn)


def measure(kind, out_dir):
    tmp = tempfile.mkdtemp(prefix="kb_simd_repro_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        script = os.path.join(tmp, "run.py")
        with open(script, "w") as f:
            f.write(
                "import importlib.util, sys, torch, torch_npu\n"
                "torch.npu.set_device(0)\n"
                f"spec = importlib.util.spec_from_file_location('k', {os.path.abspath(__file__)!r})\n"
                "m = importlib.util.module_from_spec(spec); sys.modules['k'] = m\n"
                "spec.loader.exec_module(m)\n"
                f"fn = m._kernel({kind!r})\n"
                "cos = torch.rand((m.TOKENS, m.HALF), device='npu') * 2 - 1\n"
                "sin = torch.rand((m.TOKENS, m.HALF), device='npu') * 2 - 1\n"
                "q = torch.randn((m.TOKENS, m.COLS), device='npu')\n"
                "torch.npu.synchronize(); fn(cos, sin, q); torch.npu.synchronize()\n"
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

    simd_src = source_of(_kernel("simd"))
    simt_src = source_of(_kernel("simt"))
    assert "__simd_vf__" in simd_src and "vlds_norm" in simd_src, "SIMD pass not vectorised"
    assert "vsts_norm" in simd_src
    assert "__simt_vf__" in simt_src, "SIMT pass missing"
    assert "float2" in simt_src, "SIMT per-lane width expectation changed"
    print("[PASS] code shape: simd=__simd_vf__ + vlds_norm/vsts_norm, simt=__simt_vf__ per-lane")

    if not args.time:
        return 0
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repro_out")
    us = {k: measure(k, os.path.join(out_root, k)) for k in ("simt", "simd")}
    print({k: round(v, 3) for k, v in us.items()})
    assert us["simd"] < us["simt"], us
    print("[PASS] simd rotate faster than simt rotate for the same payload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
