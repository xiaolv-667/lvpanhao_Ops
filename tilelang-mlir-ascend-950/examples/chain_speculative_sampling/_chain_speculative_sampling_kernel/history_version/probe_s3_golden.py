import sys, torch
sys.path.insert(0, "/workspace/tilelang-mlir-ascend-950/examples/TileOPs")
sys.path.insert(0, "/workspace/tilelang-mlir-ascend-950/examples/chain_speculative_sampling/_chain_speculative_sampling_kernel")
import importlib.util
spec = importlib.util.spec_from_file_location("css", "/workspace/tilelang-mlir-ascend-950/examples/chain_speculative_sampling/_chain_speculative_sampling_kernel/_chain_speculative_sampling_kernel.py")
css = importlib.util.module_from_spec(spec); spec.loader.exec_module(css)
from tileops.workloads import sampling as ref

torch.manual_seed(0)
for (b, n, v) in [(2,2,64), (4,3,64), (4,600,7), (3,3,4099), (5,1,17), (4,2,1)]:
    gen = torch.Generator().manual_seed(b*100+n)
    draft = torch.randn(b, n, v, generator=gen).softmax(-1)
    target = torch.randn(b, n+1, v, generator=gen).softmax(-1)
    ids = torch.multinomial(draft.reshape(-1, v), 1, generator=gen).view(b, n).to(torch.int32)
    seed = torch.tensor([ref.SEED], dtype=torch.int64); off = torch.tensor([ref.OFFSET], dtype=torch.int64)
    u_mine = css.golden_philox_uniform(seed, off, b, n+1)
    u_ref = ref.philox_uniform(seed, off, b, n+1)
    tok_m, acc_m = css.golden_chain_speculative_sampling(draft, ids, target, seed, off)
    tok_r, acc_r = ref.chain_speculative_sampling(draft, ids, target, seed, off)
    print(f"B={b} N={n} V={v}: u_bit={torch.equal(u_mine,u_ref)} tok={torch.equal(tok_m,tok_r)} acc={torch.equal(acc_m,acc_r)}")
    if not torch.equal(tok_m, tok_r):
        d = (tok_m != tok_r).nonzero()
        print("   mismatch idx", d[:5].tolist())
    if not torch.equal(acc_m, acc_r):
        print("   acc", acc_m, acc_r)
