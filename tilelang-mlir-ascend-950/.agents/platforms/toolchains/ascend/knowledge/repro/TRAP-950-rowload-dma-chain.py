"""TRAP-950-rowload-dma-chain + CONST-950-kernel-launch-floor-48blk + PL-950-window-bulk-copy -- repro.

repro for entries: TRAP-950-rowload-dma-chain, CONST-950-kernel-launch-floor-48blk,
PL-950-window-bulk-copy.
First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-24;
origin_task engram_gate_conv-engram_gate_conv-20261124T171256Z (Stage 4).
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa

Claim: at a fixed launch geometry (grid = 48 blocks, d = 768, bf16) the cost of
moving one 41.5 KB row window from GM to UB scales with the number of DMA
descriptors, not with the bytes: 27 one-row `T.copy` descriptors (each followed
by an `asc_sync_wait(PIPE_MTE2, PIPE_V)` before its use) cost ~3.2 us more core
time than 3-6 bulk `(rc+3, d)` region copies of the same payload.  Replacing the
per-row chain with bulk window copies is therefore a ~20% kernel-level win, and
the residual 2.7 us is the launch floor of that grid (the cost of one more
launch, which is why a two-launch structure loses).

Measures, on one fixed launch geometry (grid = 48 blocks, d = 768, bf16), how the
`Task Duration(us)` of an Ascend 950PR kernel depends on **how many GM->UB
descriptors** move a fixed 41.5 KB window, and what the launch floor of the same
geometry is.

    kind=0  1 row load + 2 row stores                       -> floor
    kind=1  27 one-row loads (the per-row DMA chain)         -> row-chain cost
    kind=4  3 bulk (rc+3, d) loads                           -> bulk cost
    kind=5  6 bulk loads (owned rc rows + 3 causal halo rows) -> bulk cost, guarded form

Assertions (median of `--reps` independent capture processes each):

    * kind_1 >= 1.4 * kind_4    (the per-row descriptor chain is the cost driver)
    * kind_5 <= 1.25 * kind_4   (the causal-guarded split keeps the bulk form)
    * kind_0  in [1.5, 4.0] us  (launch floor of a 48-block 1-D grid)

Protocol: msopprof `--launch-count=1 --warm-up=0 --aic-metrics=Roofline` around
**one** operator call, then read `Task Duration(us)` from `OpBasicInfo*.csv`
(the task platform's kernel-only metric). The warm-up call runs in its own
process so no unmeasured launch can enter the capture window.

run (cwd must be examples/TileOPs, per the platform execution context):
    python TRAP-950-rowload-dma-chain.py [--reps 3]
"""

import argparse
import csv
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile

import torch
import torch_npu  # noqa: F401

M, S, D, RC, NCH = 3, 96, 768, 6, 16
GRID = M * NCH
DT = "bfloat16"
FP32 = "float32"

_SRC = r'''
import os, sys
sys.path.insert(0, {repo!r})
import importlib.util
spec = importlib.util.spec_from_file_location("probe", {path!r})
mod = importlib.util.module_from_spec(spec)
sys.modules["probe"] = mod
spec.loader.exec_module(mod)
import torch, torch_npu
torch.npu.set_device(0)
args = torch.load({args!r})
args = [a.to("npu") for a in args]
fn = mod.build({kind!r})
torch.npu.synchronize()
fn(*args)
torch.npu.synchronize()
'''


def build(kind):
    import tilelang
    import tilelang.ascend.language as T

    rows = 9

    @T.prim_func
    def main(
        H: T.Tensor((M, S, D), DT),
        K: T.Tensor((M, S, D), DT),
        V: T.Tensor((M, S, D), DT),
        Y: T.Tensor((M, S, D), DT),
        VHAT: T.Tensor((M, S, D), DT),
        RH: T.Tensor((M, S), FP32),
    ):
        with T.Kernel(GRID) as bx:
            b = bx // NCH
            c = bx % NCH
            t0 = c * RC
            h_ub = T.alloc_shared((D,), DT)
            k_ub = T.alloc_shared((D,), DT)
            v_ub = T.alloc_shared((D,), DT)
            y_ub = T.alloc_shared((D,), DT)
            h_win = T.alloc_shared((RC + 3, D), DT)
            k_win = T.alloc_shared((RC + 3, D), DT)
            v_win = T.alloc_shared((RC + 3, D), DT)
            st = T.alloc_shared((8,), FP32)
            T.annotate_buffer_versions(
                {h_ub: 1, k_ub: 1, v_ub: 1, y_ub: 1, h_win: 1, k_win: 1, v_win: 1, st: 1}
            )
            if kind == 0:
                T.copy(H[b, t0, :], h_ub[:D])
            if kind == 1:
                for i in T.serial(rows):
                    T.copy(K[b, t0, :], k_ub[:D])
                    T.copy(V[b, t0, :], v_ub[:D])
                    T.copy(H[b, t0, :], h_ub[:D])
            if kind == 4:
                T.copy(H[b, t0 : t0 + RC, :], h_win[3 : 3 + RC, :D])
                T.copy(K[b, t0 : t0 + RC, :], k_win[3 : 3 + RC, :D])
                T.copy(V[b, t0 : t0 + RC, :], v_win[3 : 3 + RC, :D])
            if kind == 5:
                T.copy(H[b, t0 : t0 + RC, :], h_win[3 : 3 + RC, :D])
                T.copy(K[b, t0 : t0 + RC, :], k_win[3 : 3 + RC, :D])
                T.copy(V[b, t0 : t0 + RC, :], v_win[3 : 3 + RC, :D])
                if t0 >= 3:
                    T.copy(H[b, t0 - 3 : t0, :], h_win[0:3, :D])
                    T.copy(K[b, t0 - 3 : t0, :], k_win[0:3, :D])
                    T.copy(V[b, t0 - 3 : t0, :], v_win[0:3, :D])
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                T.simd.vsts(st[0], T.simd.vdup(T.float32(1), FP32, full), one, extent=1)
            # two row stores keep the epilogue identical across kinds
            T.copy(h_ub[:D], Y[b, t0, :])
            T.copy(y_ub[:D], VHAT[b, t0, :])
            T.copy(st[0:1], RH[b, t0 : t0 + 1])

    return tilelang.compile(main, target="ascend", out_idx=[3, 4, 5])


PATHS = {}


def load_probe():
    """Import this file as a module in the child process."""
    return build


def capture(kind, out_root, tag, args_path, repo, warmup=True):
    tmp = tempfile.mkdtemp(prefix="repro_rowload_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        script = os.path.join(tmp, "run.py")
        with open(script, "w") as f:
            f.write(_SRC.format(repo=repo, path=os.path.abspath(__file__), args=args_path, kind=kind))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.getcwd() + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if warmup:
            subprocess.run([sys.executable, script], cwd=os.getcwd(), env=env,
                           capture_output=True, timeout=1800)
        out = os.path.join(out_root, tag)
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.makedirs(out)
        os.chmod(out, 0o700)
        cmd = ["msopprof", f"--output={out}", "--launch-count=1", "--warm-up=0",
               "--kernel-name=main", "--aic-metrics=Roofline", sys.executable, script]
        res = subprocess.run(cmd, cwd=os.getcwd(), env=env, capture_output=True, text=True,
                             timeout=1800)
        if res.returncode != 0:
            raise RuntimeError(f"msopprof failed: {res.stderr[-2000:]}")
        total = 0.0
        got = False
        for path in glob.glob(os.path.join(out, "**", "OpBasicInfo*.csv"), recursive=True):
            for row in csv.DictReader(open(path, newline="")):
                raw = (row.get("Task Duration(us)") or "").strip()
                if (row.get("Op Name") or "").strip() and raw:
                    total += float(raw)
                    got = True
        if not got:
            raise RuntimeError("no OpBasicInfo rows")
        return total
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    repo = os.path.abspath(os.getcwd())
    out_root = os.path.abspath(args.out_root or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_out_rowload"))
    os.makedirs(out_root, exist_ok=True)

    gen = torch.Generator().manual_seed(0)
    H = torch.randn(M, S, D, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    K = torch.randn(M, S, D, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    V = torch.randn(M, S, D, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    args_path = os.path.join(out_root, "args.pt")
    torch.save([H, K, V], args_path)

    # compile every kind in this process first (the JIT cache is shared with the
    # child processes, so no capture window pays compilation)
    for kind in (0, 1, 4, 5):
        build(kind)
    torch.npu.synchronize()

    res = {}
    for kind in (0, 1, 4, 5):
        samples = [capture(kind, out_root, f"kind{kind}_r{rep}", args_path, repo)
                   for rep in range(args.reps)]
        res[kind] = statistics.median(samples)
        print(json.dumps({"kind": kind, "samples_us": [round(s, 3) for s in samples],
                          "median_us": round(res[kind], 3)}), flush=True)

    print(json.dumps({"summary_us": {str(k): round(v, 3) for k, v in res.items()}}))
    assert res[1] >= 1.4 * res[4], f"row-chain {res[1]:.3f} not >= 1.4x bulk {res[4]:.3f}"
    assert res[5] <= 1.25 * res[4], f"guarded bulk {res[5]:.3f} > 1.25x plain bulk {res[4]:.3f}"
    assert 1.5 <= res[0] <= 4.0, f"launch floor {res[0]:.3f} outside [1.5, 4.0] us"
    print("REPRO_OK")


if __name__ == "__main__":
    main()
