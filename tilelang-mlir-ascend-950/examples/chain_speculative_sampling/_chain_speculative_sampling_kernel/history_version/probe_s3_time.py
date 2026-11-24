"""Stage 3 probe: time steady-state kernel calls (compile excluded)."""
import os, sys, time, importlib.util, torch
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("css", os.path.join(HERE, "_chain_speculative_sampling_kernel.py"))
css = importlib.util.module_from_spec(spec); spec.loader.exec_module(css)
b, n, v = (int(x) for x in sys.argv[1:4])
draft, ids, target, seed, offset = css._make_inputs(b, n, v, 27)
args = (draft.to("npu"), ids.to("npu"), target.to("npu"), seed.to("npu"), offset.to("npu"))
start = time.time()
gt, ga = css._run(*args)   # includes compile
torch.npu.synchronize()
print(f"first call (compile+run): {time.time()-start:.2f} s", flush=True)
for trial in range(3):
    t0 = time.time()
    css._run(*args)
    torch.npu.synchronize()
    print(f"steady call {trial}: {(time.time()-t0)*1e3:.2f} ms => per-row {(time.time()-t0)*1e3/b:.4f} ms", flush=True)
