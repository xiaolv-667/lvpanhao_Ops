"""Production regression after the Stage-4 adoption of the A5 kernel.

Two checks, both on the *production* geometry ``B=2 / S=8192 / H=128``:

1. ``dk``/``dv`` bit-identical between the adopted ``perf_opt/_bwd_kernel.py``
   and the frozen v4 baseline (``mla_dense_bwd_kernel/_bwd_kernel.py``), plus the
   expected ``dq_acc`` store-atomic drift (``baseline vs baseline`` is measured in
   the same process as the floor);
2. non-silent output: request 1's key rows 0..63 are populated (a packed 3-D
   view would let request 0's tail block zero them), no all-zero head rows, and
   every value finite.

Usage (cwd ``examples/TileOPs``, platform python):
    python ../../mla_dense_bwd_perf2/perf_opt/check_prod.py
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401  -- registers torch.npu

TILEOPS = Path(__file__).resolve().parents[2] / "TileOPs"
if str(TILEOPS) not in sys.path:
    sys.path.insert(0, str(TILEOPS))

PKG = TILEOPS / "tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd_kernel"
INPUTS = Path("/workspace/mla_dense_bwd_opt/inputs/ds-v3-b2-8k-bfloat16.pt")
KEYS = ("do", "q", "k", "v", "o", "lse")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(mod, tensors, extra=None, batch=2, seq_q=8192, seq_kv=8192, heads=128,
        dim_qk=192, dim_v=128, dtype="bfloat16", scale=192 ** -0.5):
    """One wrapper call; ``extra`` supplies the unchanged delta/land factories."""
    q, k, v, o, do, lse = (tensors[key] for key in ("q", "k", "v", "o", "do", "lse"))
    hlp = extra if extra is not None else mod
    prep = hlp._delta_kernel(batch, heads, dim_v, seq_q, dtype)
    land = hlp._dq_land_kernel(heads, dim_qk, scale, dtype)
    main = mod._bwd_kernel(batch, heads, dim_qk, dim_v, seq_q, seq_kv, True, scale, dtype)
    cu_q = torch.arange(0, (batch + 1) * seq_q, seq_q, dtype=torch.int32, device=q.device)
    cu_kv = torch.arange(0, (batch + 1) * seq_kv, seq_kv, dtype=torch.int32, device=q.device)
    # the delta launch takes the packed (T, H, D) view (wrapper ``forward`` does
    # the same ``view`` before calling ``_run``); ``lse`` stays head-major.
    o3 = o.view(-1, heads, dim_v)
    do3 = do.view(-1, heads, dim_v)
    delta = prep(o3, do3)
    dq_acc = torch.zeros(o3.shape[0], heads, dim_qk, dtype=torch.float32,
                         device=q.device)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    main(q.view(batch, seq_q, heads, dim_qk), k.view(batch, seq_kv, heads, dim_qk),
         v.view(batch, seq_kv, heads, dim_v), do.view(batch, seq_q, heads, dim_v),
         lse, delta, cu_q, cu_kv, dq_acc.view(batch, seq_q, heads, dim_qk),
         dk.view(batch, seq_kv, heads, dim_qk), dv.view(batch, seq_kv, heads, dim_v))
    dq = land(dq_acc)
    torch.npu.synchronize()
    return dq, dk, dv


def cmp(tag, got, ref):
    gap = (got.float() - ref.float()).abs()
    gap = torch.nan_to_num(gap, nan=0.0, posinf=0.0, neginf=0.0)
    same = bool(torch.equal(got, ref))
    print(f"  {tag:6s} bitwise={'yes' if same else 'no ':3s} max_abs_diff={float(gap.max()):.3e} "
          f"ref_absmax={float(ref.float().abs().max()):.4g}")
    return same, float(gap.max())


def main():
    payload = torch.load(INPUTS, weights_only=False)
    torch.npu.set_device(0)
    tensors = {key: payload[key].to("npu") for key in KEYS}
    v4 = load("prod_v4", PKG / "_bwd_kernel.py")
    new = load("prod_perf_opt", PKG / "perf_opt/_bwd_kernel.py")
    print(f"[check] B=2 S=8192 H=128 bf16; v4={PKG / '_bwd_kernel.py'}")
    print(f"[check] adopted={PKG / 'perf_opt/_bwd_kernel.py'}")

    helpers = load("prod_helpers",
                   PKG / "_delta_kernel.py")  # unchanged launch, any copy works
    land_mod = load("prod_land", PKG / "_dq_land_kernel.py")
    helpers._dq_land_kernel = land_mod._dq_land_kernel

    dq0, dk0, dv0 = run(v4, tensors, helpers)
    dk0c, dv0c = dk0.cpu(), dv0.cpu()
    dq0c = dq0.cpu()
    dq1, dk1, dv1 = run(new, tensors, helpers)
    print("  adopted vs v4:")
    ok_dk, _ = cmp("dk", dk1.cpu(), dk0c)
    ok_dv, _ = cmp("dv", dv1.cpu(), dv0c)
    dq_gap = float((dq1.cpu().float() - dq0c.float()).abs().max())
    print(f"  dq     bitwise=no (store-atomic order) max_abs_diff={dq_gap:.3e} "
          f"ref_absmax={float(dq0c.abs().max()):.4g}")
    # run-to-run floor of v4 itself for dq (the same atomic-order drift)
    dq2, _, _ = run(v4, tensors, helpers)
    print(f"  v4 vs v4 dq drift={float((dq2.cpu().float() - dq0c.float()).abs().max()):.3e}")

    # ---- non-silent / request isolation --------------------------------------
    print("  non-silent (adopted):")
    for b in range(2):
        head0 = dk1[b, 0:64].float()
        dv_head0 = dv1[b, 0:64].float()
        print(f"    request {b} key 0..63: dk zeros={int((head0 == 0).sum())}/{head0.numel()} "
              f"max|.|={float(head0.abs().max()):.3e} | dv zeros={int((dv_head0 == 0).sum())}/"
              f"{dv_head0.numel()} max|.|={float(dv_head0.abs().max()):.3e}")
    per_head = dk1.float().abs().amax(dim=(0, 1))  # (H, D_qk)
    dead = int((per_head.amax(dim=-1) == 0).sum())
    print(f"    heads with an all-zero dk row-set: {dead}/{per_head.shape[0]}")
    finite = bool(torch.isfinite(dk1).all() and torch.isfinite(dv1).all()
                  and torch.isfinite(dq1).all())
    print(f"    all finite: {finite}")
    r1_ok = int((dk1[1, 0:64].float() == 0).sum()) < dk1[1, 0:64].numel() // 100
    ok = ok_dk and ok_dv and dead == 0 and finite and r1_ok
    print(f"  [{'PROD_CHECK_PASS' if ok else 'PROD_CHECK_FAIL'}]")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
