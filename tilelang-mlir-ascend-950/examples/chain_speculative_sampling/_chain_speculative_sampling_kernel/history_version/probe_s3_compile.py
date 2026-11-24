"""Stage 3 probe: compile the kernel at a several shapes (no launch)."""
import os, sys, traceback
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "css", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "_chain_speculative_sampling_kernel.py"))
css = importlib.util.module_from_spec(spec); spec.loader.exec_module(css)

cases = [(2, 2, 64), (4, 5, 64), (8, 3, 4099), (2, 3, 151936), (4, 600, 64), (300, 3, 32000)]
for (b, n, v) in cases:
    print(f"===== B={b} N={n} V={v} =====", flush=True)
    try:
        fn = css._chain_speculative_sampling_kernel(b, n, v)
        print("COMPILE: OK", flush=True)
    except Exception:
        print("COMPILE: FAIL", flush=True)
        traceback.print_exc(limit=6)
        continue
    if (b, n, v) == (2, 2, 64):
        src = fn.get_kernel_source()
        with open("history_version/probe_s3_gen_b2n2v64.txt", "w") as fh:
            fh.write(src)
        print("source lines:", len(src.splitlines()), flush=True)
