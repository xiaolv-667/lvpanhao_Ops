"""Stage 3 probe: one case, verbose (fresh device context per process)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util, torch
spec = importlib.util.spec_from_file_location(
    "css", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "_chain_speculative_sampling_kernel.py"))
css = importlib.util.module_from_spec(spec); spec.loader.exec_module(css)

B, N, V = (int(x) for x in sys.argv[1:4])
mode = sys.argv[4] if len(sys.argv) > 4 else "random"
gseed = int(sys.argv[5]) if len(sys.argv) > 5 else 13
draft, ids, target, seed, offset = css._make_inputs(B, N, V, gseed, mode)
ref_t, ref_a = css.golden_chain_speculative_sampling(draft, ids, target, seed, offset)
print("ref tokens:", ref_t.tolist())
print("ref accepted:", ref_a.tolist())
args = (draft.to("npu"), ids.to("npu"), target.to("npu"), seed.to("npu"), offset.to("npu"))
gt, ga = css._run(*args)
torch.npu.synchronize()
print("got tokens:", gt.cpu().tolist())
print("got accepted:", ga.cpu().tolist())
gt, ga = gt.cpu(), ga.cpu()
try:
    css.verify_chain_outputs(gt, ga, draft, ids, target)
    print("structure: OK")
except AssertionError as exc:
    print("structure: FAIL", repr(exc))
    padded = torch.cat([draft, torch.zeros_like(draft[:, :1])], 1)
    at = ga.long()
    rows = torch.arange(B)
    w = (target[rows, at] - padded[rows, at]).clamp_min(0)
    print("residual positive counts:", (w > 0).sum(1).tolist())
    print("drawn:", gt[rows, at].tolist(), "weights at drawn:", w[rows, gt[rows, at].long()].tolist())
    pos = torch.arange(N+1)[None]
    print("token==-1 vs pos>n:", (gt == -1).tolist(), (pos > ga[:, None]).tolist())
