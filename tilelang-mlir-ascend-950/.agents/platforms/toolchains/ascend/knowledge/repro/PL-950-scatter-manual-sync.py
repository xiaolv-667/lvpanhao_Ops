"""PL-950-scatter-manual-sync / TRAP-950-rowstore-mte3-drain (countermeasure) -- repro.

First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-25;
origin_task paged_cache_write-paged_cache_write-20261125 (Stage 4 round 2).
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa

Claim: for a scatter whose destination rows are provably distinct, replacing the
auto-scheduler's same-pipe conservatism (a PIPE_MTE3 drain before *every* UB->GM
row copy) with two explicit flag pairs + one per-block pipe barrier removes the
per-row store cost.  Same kernel, same shape (fp8 4096x512, 16-row tiles, 512
lanes): auto 37.56 us vs manual 19.99 us, both byte-exact against a host
reference.

Original probe doc follows.

Runs the exact fp8 shape of the tuned kernel (4096 tokens x 512, bf16 -> e4m3,
16-row tiles, 512 SimtVF lanes) in two builds:

  auto    -- auto-schedule on (today's kernel): the scheduler drains the store
             pipe before every UB->GM row copy (``asc_sync_pipe(PIPE_MTE3)``).
  manual  -- ``tl.enable_auto_schedule=False`` with the only two dependencies the
             kernel actually has: MTE2->V (loads before the conversion reads UB)
             and V->MTE3 (conversion before the stores read UB), plus one
             end-of-block PIPE_MTE3 barrier.  Distinct non-negative slots are
             the op's contract, so two row stores never collide and need no
             ordering between them.

Correctness is checked against a host reference for both builds; timing is the
framework's protocol (``--warm-up=5 --launch-count=10``, median).

    python probe_manual_sync.py [--variant auto|manual|both]
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

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

TOKENS, HEADS, DIM = 4096, 4, 128
RW = HEADS * DIM
ROWS, THREADS = 16, 512
PAGES, PAGE = 4096, 16
PHYS = PAGES * PAGE
FP8 = "float8_e4m3fn"
MAXV = 448.0
ITERS = 15  # 5 warm-up + 10 measured


def build(variant):
    tokens = T.dynamic("tokens")
    one_row = None

    @T.prim_func
    def main(
        K: T.Tensor((tokens, RW), "bfloat16"),
        V: T.Tensor((tokens, RW), "bfloat16"),
        KP: T.Tensor((PHYS, RW), FP8),
        VP: T.Tensor((PHYS, RW), FP8),
        Slots: T.Tensor((tokens,), "int64"),
        KS: T.Tensor((1,), "float32"),
        VS: T.Tensor((1,), "float32"),
    ) -> None:
        with T.Kernel(T.ceildiv(tokens, ROWS)) as bx:
            row0 = bx * ROWS
            nr = T.min(ROWS, tokens - row0)
            one = T.min(nr, 1)
            k_ub = T.alloc_shared((ROWS, RW), "bfloat16")
            v_ub = T.alloc_shared((ROWS, RW), "bfloat16")
            k8 = T.alloc_shared((ROWS, RW), FP8)
            v8 = T.alloc_shared((ROWS, RW), FP8)
            T.copy(K[row0 : row0 + nr, 0:RW], k_ub[0:nr, 0:RW])
            T.copy(V[row0 : row0 + nr, 0:RW], v_ub[0:nr, 0:RW])
            if variant == "manual":
                T.ascend_set_flag("MTE2_V", 0)
                T.ascend_wait_flag("MTE2_V", 0)
            with T.SimtVF(threads=THREADS):
                for i, c in T.Parallel(ROWS, RW):
                    k8[i, c] = T.cast(
                        T.clamp(T.cast(k_ub[i, c], "float32") / KS[0], -MAXV, MAXV), FP8
                    )
                for i, c in T.Parallel(ROWS, RW):
                    v8[i, c] = T.cast(
                        T.clamp(T.cast(v_ub[i, c], "float32") / VS[0], -MAXV, MAXV), FP8
                    )
            if variant == "manual":
                T.ascend_set_flag("V_MTE3", 0)
                T.ascend_wait_flag("V_MTE3", 0)
            for i in T.unroll(ROWS):
                if row0 + i < tokens:
                    slot = T.cast(Slots[row0 + i], "int32")
                    if slot >= 0:
                        T.copy(k8[i : i + 1, 0:RW], KP[slot : slot + one, 0:RW])
                        T.copy(v8[i : i + 1, 0:RW], VP[slot : slot + one, 0:RW])
            if variant == "manual":
                T.ascend_pipe_barrier("PIPE_MTE3")

    if variant == "manual":
        return tilelang.compile(main, target="ascend", out_idx=[],
                                pass_configs={"tl.enable_auto_schedule": False})
    return tilelang.compile(main, target="ascend", out_idx=[])


CHILD = r'''
import importlib.util, sys, torch, torch_npu
torch.npu.set_device(0)
spec = importlib.util.spec_from_file_location("probe_ms", {probe!r})
mod = importlib.util.module_from_spec(spec)
sys.modules["probe_ms"] = mod
spec.loader.exec_module(mod)
fn = mod.build({variant!r})
k, v, kp, vp, slots, ks, vs = [t.npu() for t in torch.load({args!r})]
torch.npu.synchronize()
for _ in range({iters}):
    fn(k, v, kp, vp, slots, ks, vs)
torch.npu.synchronize()
'''


def make_inputs():
    gen = torch.Generator().manual_seed(0)
    k = torch.randn((TOKENS, RW), generator=gen, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn((TOKENS, RW), generator=gen, dtype=torch.float32).to(torch.bfloat16)
    pool = torch.randn((PHYS, RW), generator=gen, dtype=torch.float32)
    kp = pool.to(torch.float8_e4m3fn)
    vp = (-pool).to(torch.float8_e4m3fn)
    slots = torch.randperm(PHYS, generator=gen)[:TOKENS].to(torch.int64)
    ks = torch.tensor([1.25], dtype=torch.float32)
    vs = torch.tensor([0.75], dtype=torch.float32)
    return [k, v, kp, vp, slots, ks, vs]


def reference(inp):
    k, v, kp, vp, slots, ks, vs = inp
    kp, vp = kp.clone(), vp.clone()
    rows = slots
    for pages, new, scale in ((kp, k, ks), (vp, v, vs)):
        limit = torch.finfo(pages.dtype).max
        stored = (new.float() / scale[0]).clamp(-limit, limit).to(pages.dtype)
        host = pages.cpu()
        host.view(-1, RW)[rows] = stored
        pages.copy_(host)
    return kp, vp


def check(variant, inp):
    fn = build(variant)
    dev = [t.npu() for t in inp]
    fn(*dev)
    torch.npu.synchronize()
    gold_k, gold_v = reference(inp)
    ok = True
    for got, want in ((dev[2], gold_k), (dev[3], gold_v)):
        g, w = got.cpu().contiguous().view(torch.uint8), want.contiguous().view(torch.uint8)
        g = torch.where(g == 0xFF, torch.tensor(0x7F, dtype=torch.uint8), g)
        w = torch.where(w == 0xFF, torch.tensor(0x7F, dtype=torch.uint8), w)
        if not torch.equal(g, w):
            ok = False
            print(f"[{variant}] mismatch: {int((g != w).sum())} bytes")
    return ok


def time(variant, inp, probe_path):
    work = tempfile.mkdtemp(prefix="msprobe_", dir=os.environ.get("TMPDIR", "/tmp"))
    args_path = os.path.join(work, "args.pt")
    torch.save(inp, args_path)
    torch.npu.empty_cache()
    script = os.path.join(work, "run.py")
    with open(script, "w") as f:
        f.write(CHILD.format(probe=probe_path, variant=variant, args=args_path, iters=ITERS))
    env = dict(os.environ)
    subprocess.run([sys.executable, script], check=True, capture_output=True, text=True,
                   timeout=1800, env=env)
    out_dir = os.path.join(work, "cap")
    os.makedirs(out_dir)
    cmd = ["msopprof", f"--output={out_dir}", "--launch-count=10", "--warm-up=5",
           "--aic-metrics=BasicInfo", sys.executable, script]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    if res.returncode != 0:
        raise RuntimeError(res.stdout[-1500:] + res.stderr[-1500:])
    vals = []
    for path in glob.glob(os.path.join(out_dir, "**", "OpBasicInfo*.csv"), recursive=True):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                raw = (row.get("Task Duration(us)") or "").strip()
                if raw:
                    vals.append(float(raw))
    shutil.rmtree(work, ignore_errors=True)
    if not vals:
        raise RuntimeError("no capture rows")
    vals.sort()
    return vals[len(vals) // 2], vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="both", choices=["auto", "manual", "both"])
    args = ap.parse_args()
    probe_path = os.path.abspath(__file__)
    inp = make_inputs()
    variants = ["auto", "manual"] if args.variant == "both" else [args.variant]
    payload = 2 * TOKENS * RW * 2 + 2 * TOKENS * RW * 1
    for variant in variants:
        ok = check(variant, inp)
        med, vals = time(variant, inp, probe_path)
        print(json.dumps({"variant": variant, "byte_exact": ok, "median_us": round(med, 3),
                          "samples": [round(v, 2) for v in vals],
                          "roofline_TBs": round(payload / (med * 1e-6) / 1e12, 3)}),
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
