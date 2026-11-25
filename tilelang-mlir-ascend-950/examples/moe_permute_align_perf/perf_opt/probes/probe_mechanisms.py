"""Mechanism probes for the Stage 4 restructure (design-time measurement).

Each probe is one launch over the max workload geometry (numel=32768 routes,
E=384, B=64, P=57023) with the production mapping (56 core blocks x 128 SIMT
lanes, 5 routes/lane) and isolates one mechanism so its cost can be measured
with the platform protocol (msprof kernel `Task Duration`):

  contig   per-route store at its own index (vectorisable control)
  rand     per-route store at a scrambled index inside a 32768-entry GM array
  ubscat   per-route store at a scrambled index inside a UB window + one DMA out
  histcmp  the current per-expert compare/accumulate histogram (control)
  histatom per-route `atomic_add` into a UB histogram
  gmcur    per-route `atomic_add` on a GM cursor with return_prev (current scatter)

Run: ``python probes/probe_mechanisms.py`` (it drives bench_kernel_msprof).
"""

import math
import os
import sys

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

NUMEL = 32768
E = 384
B = 64
P = NUMEL + (E + 1) * (B - 1)
CORES = 56
THREADS = 128
EPT = math.ceil(math.ceil(NUMEL / CORES) / THREADS)  # 5
E_PAD = 384
CHUNK = math.ceil(math.ceil(NUMEL / CORES) / 64) * 64   # 640
KERNELS = {}


def _kernels():
    if KERNELS:
        return KERNELS

    @tilelang.jit(out_idx=[], target="ascend")
    def contig():
        @T.prim_func
        def main(flat: T.Tensor((NUMEL,), "int32"), out: T.Tensor((NUMEL,), "int32")):
            with T.Kernel(CORES) as bid, T.SimtVF(threads=THREADS):
                for i in T.serial(EPT):
                    for j in T.Parallel(THREADS):
                        idx = (bid * EPT + i) * THREADS + j
                        out[T.min(idx, NUMEL - 1)] = idx
        return main

    @tilelang.jit(out_idx=[], target="ascend")
    def rand():
        @T.prim_func
        def main(flat: T.Tensor((NUMEL,), "int32"), out: T.Tensor((NUMEL,), "int32")):
            with T.Kernel(CORES) as bid, T.SimtVF(threads=THREADS):
                for i in T.serial(EPT):
                    for j in T.Parallel(THREADS):
                        idx = (bid * EPT + i) * THREADS + j
                        out[(T.min(idx, NUMEL - 1) * 7919) % NUMEL] = idx
        return main

    @tilelang.jit(out_idx=[], target="ascend")
    def ubscat():
        @T.prim_func
        def main(flat: T.Tensor((NUMEL,), "int32"), out: T.Tensor((NUMEL,), "int32")):
            with T.Kernel(CORES) as bid, T.SimtVF(threads=THREADS):
                s_out = T.alloc_shared((NUMEL,), "int32")
                for i in T.serial(EPT):
                    for j in T.Parallel(THREADS):
                        idx = (bid * EPT + i) * THREADS + j
                        s_out[(T.min(idx, NUMEL - 1) * 7919) % NUMEL] = idx
                T.copy(s_out, out)
        return main

    @tilelang.jit(out_idx=[], target="ascend")
    def histcmp():
        @T.prim_func
        def main(flat: T.Tensor((NUMEL,), "int32"), partial: T.Tensor((CORES, E_PAD), "int32")):
            with T.Kernel(CORES) as g:
                s_x = T.alloc_shared((CHUNK,), "int32")
                s_hist = T.alloc_shared((E_PAD,), "int32")
                with T.SimtVF(threads=E_PAD):
                    for e in T.Parallel(E_PAD):
                        s_hist[e] = 0
                base = g * CHUNK
                n = T.max(0, T.min(CHUNK, NUMEL - base))
                T.copy(flat[base:base + n], s_x[0:n])
                with T.SimtVF(threads=E_PAD):
                    for e in T.Parallel(E_PAD):
                        acc = T.alloc_local([1], "int32")
                        acc[0] = 0
                        for i in T.serial(n):
                            acc[0] = acc[0] + T.if_then_else(s_x[i] == e, 1, 0)
                        s_hist[e] = s_hist[e] + acc[0]
                T.copy(s_hist, partial[g, :])
        return main

    @tilelang.jit(out_idx=[], target="ascend")
    def histunroll():
        @T.prim_func
        def main(flat: T.Tensor((NUMEL,), "int32"), partial: T.Tensor((CORES, E_PAD), "int32")):
            with T.Kernel(CORES) as g:
                s_x = T.alloc_shared((CHUNK,), "int32")
                s_hist = T.alloc_shared((E_PAD,), "int32")
                with T.SimtVF(threads=E_PAD):
                    for e in T.Parallel(E_PAD):
                        s_hist[e] = 0
                base = g * CHUNK
                n = T.max(0, T.min(CHUNK, NUMEL - base))
                T.copy(flat[base:base + n], s_x[0:n])
                with T.SimtVF(threads=E_PAD):
                    for e in T.Parallel(E_PAD):
                        acc = T.alloc_local([8], "int32")
                        for j in range(8):
                            acc[j] = 0
                        for i in T.serial(CHUNK // 8):
                            for j in range(8):
                                acc[j] = acc[j] + T.if_then_else(s_x[i * 8 + j] == e, 1, 0)
                        s_hist[e] = s_hist[e] + acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7]
                T.copy(s_hist, partial[g, :])
        return main

    @tilelang.jit(out_idx=[], target="ascend")
    def ubatom():
        # Mirrors the Stage 3 scatter exactly (T.Kernel + T.SimtVF in one header,
        # route id from GM), with the bump cursor in shared memory instead of GM.
        @T.prim_func
        def main(flat: T.Tensor((NUMEL,), "int32"), out: T.Tensor((NUMEL,), "int32")):
            with T.Kernel(CORES) as bid, T.SimtVF(threads=THREADS):
                s_cur = T.alloc_shared((E_PAD,), "int32")
                eid = T.alloc_local([1], "int32")
                pos = T.alloc_local([1], "int32")
                for k in T.serial(E_PAD // THREADS):
                    for j in T.Parallel(THREADS):
                        s_cur[k * THREADS + j] = (k * THREADS + j) * 16
                for i in T.serial(EPT):
                    for j in T.Parallel(THREADS):
                        idx = (bid * EPT + i) * THREADS + j
                        eid[0] = flat[T.min(idx, NUMEL - 1)]
                        if idx < NUMEL:
                            pos[0] = T.atomic_add(s_cur[eid[0]], T.int32(1), return_prev=True)
                            out[T.min(idx, NUMEL - 1)] = pos[0]
        return main

    @tilelang.jit(out_idx=[], target="ascend")
    def histatom2():
        # per-route UB atomic histogram, Stage 3 scatter structure
        @T.prim_func
        def main(flat: T.Tensor((NUMEL,), "int32"), partial: T.Tensor((CORES, E_PAD), "int32")):
            with T.Kernel(CORES) as g, T.SimtVF(threads=THREADS):
                s_hist = T.alloc_shared((E_PAD,), "int32")
                eid = T.alloc_local([1], "int32")
                for k in T.serial(E_PAD // THREADS):
                    for j in T.Parallel(THREADS):
                        s_hist[k * THREADS + j] = 0
                for i in T.serial(EPT):
                    for j in T.Parallel(THREADS):
                        idx = (g * EPT + i) * THREADS + j
                        if idx < NUMEL:
                            eid[0] = flat[idx]
                            T.atomic_add(s_hist[eid[0]], T.int32(1))
                T.copy(s_hist, partial[g, :])
        return main

    @tilelang.jit(out_idx=[], target="ascend")
    def histatom():
        @T.prim_func
        def main(flat: T.Tensor((NUMEL,), "int32"), partial: T.Tensor((CORES, E_PAD), "int32")):
            with T.Kernel(CORES) as g:
                s_x = T.alloc_shared((CHUNK,), "int32")
                s_hist = T.alloc_shared((E_PAD,), "int32")
                with T.SimtVF(threads=E_PAD):
                    for e in T.Parallel(E_PAD):
                        s_hist[e] = 0
                base = g * CHUNK
                n = T.max(0, T.min(CHUNK, NUMEL - base))
                T.copy(flat[base:base + n], s_x[0:n])
                eid = T.alloc_local([1], "int32")
                for i in T.serial(EPT):
                    for j in T.Parallel(THREADS):
                        idx = i * THREADS + j
                        if idx < n:
                            eid[0] = s_x[idx]
                            T.atomic_add(s_hist[eid[0]], T.int32(1))
                T.copy(s_hist, partial[g, :])
        return main

    @tilelang.jit(out_idx=[], target="ascend")
    def gmcur():
        @T.prim_func
        def main(flat: T.Tensor((NUMEL,), "int32"), out: T.Tensor((NUMEL,), "int32"),
                 cur: T.Tensor((E + 1,), "int32")):
            with T.Kernel(CORES) as bid, T.SimtVF(threads=THREADS):
                for i in T.serial(EPT):
                    for j in T.Parallel(THREADS):
                        idx = (bid * EPT + i) * THREADS + j
                        eid = flat[T.min(idx, NUMEL - 1)]
                        if idx < NUMEL:
                            slot = T.atomic_add(cur[eid], T.int32(1), return_prev=True)
                            out[slot] = idx
        return main

    KERNELS.update(
        contig=contig, rand=rand, ubscat=ubscat, histcmp=histcmp, histatom=histatom,
        histunroll=histunroll, ubatom=ubatom, gmcur=gmcur, histatom2=histatom2,
    )
    return KERNELS


def _build(name):
    return _kernels()[name]()


def contig(flat, out):
    _build("contig")(flat, out)


def rand(flat, out):
    _build("rand")(flat, out)


def ubscat(flat, out):
    _build("ubscat")(flat, out)


def histcmp(flat, partial):
    _build("histcmp")(flat, partial)


def histatom(flat, partial):
    _build("histatom")(flat, partial)


def histunroll(flat, partial):
    _build("histunroll")(flat, partial)


def ubatom(flat, out):
    _build("ubatom")(flat, out)


def histatom2(flat, partial):
    _build("histatom2")(flat, partial)


def gmcur(flat, out, cur):
    _build("gmcur")(flat, out, cur)


CASES = [
    ("contig", contig, lambda: (_flat(), _nump()),
     "per-route store at its own index (control)"),
    ("rand", rand, lambda: (_flat(), _nump()),
     "per-route store at a scrambled GM index"),
    ("ubscat", ubscat, lambda: (_flat(), _nump()),
     "per-route store into a UB window + one DMA out"),
    ("histcmp", histcmp, lambda: (_flat(), _hist()),
     "compare/accumulate histogram (current)"),
    ("histatom", histatom, lambda: (_flat(), _hist()),
     "per-route UB atomic histogram"),
    ("gmcur", gmcur, lambda: (_flat(), _nump(), _cur()),
     "per-route GM atomic cursor (current scatter)"),
    ("histunroll", histunroll, lambda: (_flat(), _hist()),
     "compare/accumulate histogram, 8 independent chains"),
    ("ubatom", ubatom, lambda: (_flat(), _nump()),
     "per-route UB atomic cursor + contiguous store"),
    ("histatom2", histatom2, lambda: (_flat(), _hist()),
     "per-route UB atomic histogram"),
]


def _flat():
    g = torch.Generator().manual_seed(7)
    return torch.randint(0, E, (NUMEL,), generator=g, dtype=torch.int32)


def _nump():
    return torch.zeros(NUMEL, dtype=torch.int32)


def _hist():
    return torch.zeros((CORES, E_PAD), dtype=torch.int32)


def _cur():
    vals = torch.zeros(E + 1, dtype=torch.int32)
    vals[1:] = torch.arange(E, dtype=torch.int32) * 85
    return vals


def main():
    from tileops.benchmark.msprof import bench_kernel_msprof

    here = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = here + (
        ":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )
    os.environ.setdefault("TILEOPS_MSPROF_WARM_UP", "0")
    os.environ.setdefault("TILEOPS_MSPROF_LAUNCH_COUNT", "1")
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, fn, make_args, note in CASES:
        if only and name not in only.split(","):
            continue
        args = make_args()
        try:
            lat, out_dir = bench_kernel_msprof(fn, args=tuple(args))
            print(f"[probe] {name:9s} {lat * 1000:9.3f} us   {note}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] {name:9s} FAILED {type(exc).__name__}: {str(exc)[:400]}", flush=True)


if __name__ == "__main__":
    if __name__ != "probe_mechanisms":
        import importlib

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        importlib.import_module("probe_mechanisms").main()
    else:
        main()
