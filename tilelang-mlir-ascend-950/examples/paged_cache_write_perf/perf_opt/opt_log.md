# paged_cache_write — Stage 4 tuning log (PagedKVCacheWriteFwdOp)

- Task: `paged_cache_write-paged_cache_write-20261124T171256Z`, project `paged_cache_write`,
  operator `PagedKVCacheWriteFwdOp`, mode `full`.
- Platform: `ascend-950pr-9579` / `ascend` (Ascend950PR_9579, 28 Cube / 56 Vector, dav-3510),
  TileLang ascend 0.1.15 + CANN 9.2.1, single card.
- Baseline (read-only): `examples/TileOPs/tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py`
  driven by the frozen wrapper `.../paged_cache_write/paged_cache_write.py`.
- Delivered: `perf_opt/paged_cache_write.py` (self-contained: golden + kernel + layered tests;
  byte-identical alias `perf_opt/PagedKVCacheWriteFwdOp.py`).

## 1. Measurement protocol (task platform ruleset)

`msopprof --output=<dir> --launch-count=1 --warm-up=0 --aic-metrics=BasicInfo <call script>`
around **one** operator call, then the `Task Duration(us)` column of `OpBasicInfo*.csv`.
One operator call issues exactly one kernel launch, and the driver asserts a single captured
row, so T_op = "one collection, one call" — never an NPU event and never a median of several
launches inside the capture window. Warm-up/compile runs in a *separate* process so no
unmeasured launch and no compilation can land in the capture window; inputs are built on the
host and only moved, so the first device kernel is the operator under test.
`perf_opt/experiments/bench_one.py` implements the protocol, `perf_opt/experiments/run_experiments.py`
runs it over the workload set and appends `perf_records.jsonl`. Reps: 3 per workload for the
merged/final candidates, 2 for exploratory branches; the record carries every sample and the
median is the reported number.

## 2. Phase 1 — workload inventory and baseline

`bench_paged_cache_write.py::test_paged_cache_write_bench` expands
`tileops.manifest.load_workloads('PagedKVCacheWriteFwdOp')` into 10 parametrized cases; the
manifest's first workload carries `pytest.mark.smoke` (3 dtype combos, precision only), the
other seven are `pytest.mark.full` → tune. Nothing is skipped. Full inventory:
`perf_opt/workload_inventory.json`.

`kernel_id` for every record:
`tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py::_paged_cache_write_kernel::main`.

### Performance Test Data — baseline (Stage 3, `phase=baseline`, round 0)

Baseline tile = what the frozen wrapper picks today (`block_rows=16`, `block_cols=128`, and
`block_cols = min(1024, ceil(row_width/128)*128)` once `tokens >= 6144`, i.e. the whole-row
switch for the 8192-token prefill).

| kernel_id | workload_id | Task Duration (µs) | samples (µs) | traffic (B) | effective BW | MBU vs 1.6 TB/s | kernel |
|---|---|---:|---|---:|---:|---:|---|
| …::_paged_cache_write_kernel::main | llama-8b-decode-bfloat16-bfloat16 | 16.115 | 16.080/16.115/16.115 | 524,800 | 0.033 TB/s | 2.0% | `_16bit_kernel` |
| …::_paged_cache_write_kernel::main | llama-8b-decode-float16-float16 | 16.097 | 16.219/16.028/16.097 | 524,800 | 0.033 TB/s | 2.0% | `_16bit_kernel` |
| …::_paged_cache_write_kernel::main | llama-8b-prefill-bfloat16-bfloat16 | 811.486 | 812.227/811.486/810.990 | 67,174,400 | 0.083 TB/s | 5.2% | `_16bit_kernel` |
| …::_paged_cache_write_kernel::main | qwen3-235b-fp8-bfloat16-float8_e4m3fn | 270.596 | 271.716/270.140/270.596 | 12,615,688 | 0.047 TB/s | 2.9% | `_fp8_kernel` |
| …::_paged_cache_write_kernel::main | gpt-oss-120b-bfloat16-bfloat16 | 26.567 | 26.572/26.554/26.567 | 1,050,624 | 0.040 TB/s | 2.5% | `_16bit_kernel` |
| …::_paged_cache_write_kernel::main | odd-tokens-and-page-float16-float16 | 8.200 | 8.649/8.200/7.938 | 7,224 | 0.001 TB/s | 0.05% | `_16bit_kernel` |
| …::_paged_cache_write_kernel::main | qwen3-235b-fp8-fp16-float16-float8_e4m3fn | 79.721 | 79.698/79.721/80.437 | 3,153,928 | 0.040 TB/s | 2.5% | `_fp8_kernel` |

Raw captures: `perf_opt/profiles/experiments/baseline-stage3_<workload>_r*/`; L0 evidence
`perf_opt/logs/l0/_paged_cache_write_kernel.log` (10/10, byte-exact).

Baseline ordering (measured burden, which sets the tuning order): prefill (811 µs, 67 MB) →
fp8 (271 µs) → fp8-fp16 (80 µs) → gpt-oss (27 µs) → decode ×2 (16 µs) → odd (8 µs).

### Design estimate vs measured (D-2 deviation row)

`DESIGN.md` §1.6.0 estimated the *floor* per workload from `traffic/1.47 TB/s` plus a launch
term: prefill 46.0 µs, fp8 54.9 µs (launch-dominated), fp8-fp16 13.7 µs, gpt-oss 0.72 µs,
decode 0.4 µs, odd 0.01 µs.

| workload | design estimate (µs) | baseline measured (µs) | baseline / estimate | final measured (µs) | final / estimate |
|---|---:|---:|---:|---:|---:|
| llama-8b-prefill | 46.0 | 811.5 | 17.6x | 48.1 | 1.05x |
| qwen3-235b-fp8 | 54.9 | 270.6 | 4.9x | 37.7 | 0.69x |
| qwen3-235b-fp8-fp16 | 13.7 | 79.7 | 5.8x | 15.9 | 1.16x |
| gpt-oss-120b | 0.72 | 26.6 | 37x | 6.6 | 9.2x |
| llama-8b-decode | 0.4 | 16.1 | 40x | 4.5 | 11x |
| odd-tokens | 0.01 | 8.2 | 820x | 2.7 | 269x |

Reading: the design's *traffic* floor was right (final prefill 48.1 vs 46.0 estimated) and the
baseline was 17.6x above it — the gap was the elementwise memory path, not the model. The fp8
estimate was pessimistic: its launch term charged 6 vector ops/block that the DMA path does not
pay. The small shapes miss the estimate because the model carries no per-launch floor; measured
launch floor on this stack is ~2.4 µs (odd-tokens: 7 rows of work).

## 3. Phase 2 — Round 1 (six branches off the baseline)

**Phenomenon at the base profile.** The Stage 3 kernel is a pure scatter with per-lane GM
accesses on both directions, and every element re-reads `Slots[token]` (8 B) — the access
pattern, not arithmetic, is the cost. Effective bandwidth 83 GB/s on the largest case (5% of
nominal), i.e. 17.6x above the design's own traffic floor. Platform knowledge
(`CONST-950-dma-vs-lane-gm`, origin task `rope-RoPEFwdOp-20261124T171256Z`) says a descriptor
copy runs 2-3x faster than per-lane for the same payload, so the candidate set targets the
memory path only.

**Branches** (each one main change, all from `baseline-stage3`):

| id | candidate file | single change |
|---|---|---|
| r1_c1 | `paged_cache_write_round1_c1.py` | row-tile **DMA load** + **per-row DMA store** (both pools in one block, 128 lanes) |
| r1_c2 | `paged_cache_write_round1_c2.py` | row-tile DMA load + **per-lane vector store** |
| r1_c3 | `paged_cache_write_round1_c3.py` | **store path per cache width** (16-bit lane, fp8 DMA) |
| r1_c4 | `paged_cache_write_round1_c4.py` | r1_c3 **split by pool** (2x blocks, one stream each) |
| r1_c5 | `paged_cache_write_round1_c5.py` | r1_c3 with **256 SimtVF lanes** |
| r1_c6 | `paged_cache_write_round1_c6.py` | r1_c3 with **512 SimtVF lanes** |

**Candidate vs current best** (current best = `baseline-stage3` for this round; Task Duration
in µs, platform protocol, median of the listed reps):

| workload | current best | r1_c1 | r1_c2 | r1_c3 | r1_c4 | r1_c5 | r1_c6 | best vs current best |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| llama-8b-prefill (32×1024 tile) | 811.486 | 78.314 | 60.921 | 63.632 | 65.644 | 55.667 | **46.491** | -94.3% |
| qwen3-235b-fp8 (16×512 tile) | 270.596 | 41.644 | 157.587 ✗ | 42.546 | 48.700 | 38.147 | **37.269** | -86.2% |
| llama-8b-decode-bf16 (2×1024) | 16.115 | — | — | **4.401** | — | — | — | -72.7% |
| gpt-oss-120b (8×512) | 26.567 | — | — | **6.473** | — | — | — | -75.6% |
| odd-tokens (1×128) | 8.200 | — | — | **2.986** | — | — | — | -63.6% |
| qwen3-235b-fp8-fp16 (8×512) | 79.721 | — | — | **15.982** | — | — | — | -80.0% |

✗ = the branch is invalid on that workload: `r1_c2` stores fp8 per lane, and a one-byte element
per lane cannot be widened, so small fp8 tiles hit `TRAP-950-fp8-scalar-store` (scalar e4m3 GM
store writes 0). Its own `--level L0` fails `L0-5 saturates-fp8-overflow`
(`perf_opt/logs/l0/paged_cache_write_round1_c2.log`), and its 157 µs timing shows the lane path
is 3.7x slower for fp8 even where it is correct.

**L0 (byte-exact, both pools) per branch**: r1_c1 10/10, r1_c2 9/10 (**rejected**: fp8 scalar
store), r1_c3 10/10, r1_c4 10/10, r1_c5 10/10, r1_c6 10/10
(`perf_opt/logs/l0/paged_cache_write_round1_c*.log`).

**What the round established.**

1. *The DMA load is the big win*: 811.5 → 78.3 µs (r1_c1 vs baseline) comes from staging the
   contiguous source rows with one `T.copy` per pool per block instead of per-lane reads
   (and from dropping the per-element `Slots` re-read).
2. *The scattered store must not be a row-sized DMA for 16-bit*: the scheduler drains the
   store pipe before every UB→GM `T.copy` (`asc_sync_pipe(PIPE_MTE3)`, emitted by
   `insert_sync.cc` when producer and consumer pipe are the same), so a 2 KB row store costs a
   fixed ~0.25-0.3 µs of core time and scales with row count. Prefill: 16384 row stores =
   78.3 µs; the vector store path has no drain and lands 60.9 µs. Isolated repro and generated
   code: `TRAP-950-rowstore-mte3-drain`
   (`knowledge/repro/TRAP-950-rowstore-mte3-drain.py`: row 80.4 µs vs lane 48.2 µs vs one
   contiguous block DMA 44.3 µs on a 67 MB payload).
3. *fp8 is the mirror image*: a 512 B row store is not worth 128 lanes of scalar-ised byte
   stores, so the quantised build keeps the DMA store (r1_c3) — the lane path is both slower
   and wrong on small tiles.
4. *Lane count matters once a tile is wide*: 128 → 256 → 512 lanes takes prefill
   63.6 → 55.7 → 46.5 µs (per-lane store throughput), and fp8 42.5 → 38.1 → 37.3 µs.
5. *Splitting by pool is not a win* (r1_c4: prefill 65.6, fp8 48.7): the extra blocks and the
   doubled `Slots` reads cost more than the extra stream parallelism buys.

**Round 1 winner**: r1_c3's structure (DMA load + per-dtype store path) with the lane count
following the tile size (512 lanes from an 8192-element tile up, else 128, i.e. r1_c6 for the
wide cases and r1_c3 for the small ones). Current best after the merge below.

**Records note (append-only discipline).** Two groups of rows in `perf_records.jsonl` need
reading with care and are *not* deleted: (a) the first two `r1_c1_dma_store` rows
(prefill 64.197, fp8 157.440) were measured while that branch file was mis-wired to the lane
path — they were superseded by the corrected `r1_c1_dma_store` rows (78.314 / 41.644) which
match the module as it now stands; (b) the `r1_c2_lane_store` fp8 row carries `l0_pass=true`
from the runner's default at the time; the branch's own L0 run later failed L0-5, so that flag
is invalid and the branch is recorded as rejected (runner fixed to claim `l0_pass` only from a
run it made or a log it was pointed at).

## 4. Phase 2 — Round 2 (merge onto one kernel)

Merged candidate `r2_c1_merged` (`paged_cache_write_round2_c1.py`): r1_c3 structure + the two
measured policies —

- store path: `lane` iff the tile is 16-bit and `rows*cols >= 4096`, else `dma`;
- lane count: 512 iff `rows*cols >= 8192`, else 128;
- row tile: `rows = min(clamp(ceil(tokens/step),1,32), max(1, tokens//4))` with
  `step = 32` for 16-bit and `128` for fp8 (fp8 rows are half the bytes, so its step is 4x
  smaller; cap 16), and `rows >= ceil(256/row_width)` for fp8 so the conversion pass keeps two
  elements per lane.

**Candidate vs current best** (current best = round-1 winner r1_c6 for the wide shapes, r1_c3
for the small ones; Task Duration µs, median of 3):

| workload | current best | r1_c1 (128 L, DMA store) | r1_c2 (128 L, lane store) | r1_c3 (128 L, per-dtype) | r1_c6 (512 L, per-dtype) | r2_c1 merged | L0 |
|---|---:|---:|---:|---:|---:|---:|---|
| llama-8b-decode-bfloat16 | 4.401 | 4.465 (2×1024) | 4.974 (2×1024) | 4.401 | 6.558 (1×1024) | **4.335** | 10/10 |
| llama-8b-decode-float16 | 4.401 | — | — | — | — | **4.499** | 10/10 |
| llama-8b-prefill | 46.491 | 78.314 | 60.921 | 63.632 | 46.491 | **47.563** | 10/10 |
| qwen3-235b-fp8 | 37.269 | 41.644 | 157.587 ✗ | 42.546 | 37.269 | **37.476** | 10/10 |
| gpt-oss-120b | 6.473 | 7.369 (8×512) | — | 6.473 | 10.015 (2×512) | **6.564** | 10/10 |
| odd-tokens-and-page | 2.986 | 2.496 (1×128) | — | 2.986 | 3.914 (2×128) | **2.662** | 10/10 |
| qwen3-235b-fp8-fp16 | 15.982 | 16.095 (8×512) | — | 15.982 | 15.952 (4×512) | **15.709** | 10/10 |

Merge verification: the merged kernel was re-measured on **all seven** tune workloads (above)
and re-ran L0 10/10 (`perf_opt/logs/l0/paged_cache_write_round2_c1.log`) plus the three smoke
geometries byte-exact (`perf_opt/experiments/smoke_precision.py`). No workload regressed
against its round-1 best (largest deviation 1.5% on prefill, inside the 3% noise threshold; the
0.5-1.7% spreads of the merged run bracket it). Winner merged.

## 5. Phase 2 — Round 3 (final file + full-set verification)

`perf_opt/paged_cache_write.py` is the merged kernel written as one self-contained module
(golden + kernel + L0/L1/L2/boundary). Re-measured from scratch on all seven tune workloads
(`phase=final`), L0 10/10, L1 2/2, L2 3/3, boundary 10/10 — every case byte-exact
(`max_diff = 0`, including the fp8 raw-byte comparison with NaN-sign canonicalisation).

## 6. Autotune / search log (manual, no autotuner in this kernel)

The kernel has three discrete knobs (row-tile height, lane count, store path) and no continuous
space, so a manual coarse grid plus a winner neighbourhood check was used instead of an
autotuner. All numbers: `msopprof` single-launch Task Duration, 2 reps, median. The exploratory
sweep captures were pruned after the round to keep the evidence tree to the 39 recorded
experiments; the sweep numbers below are as read from those captures at the time, and every
`perf_records.jsonl` row still points at its own surviving capture.

**Row-tile height sweep (prefill 8192×1024, 16-bit)** — r1_c1 (DMA store, 128 lanes):
1→215.8, 8→86.5, 16→78.7, 32→78.3, 48→82.8 µs. r1_c3 (lane store, 128 lanes):
16→67.8, 24→65.0, **32→61.5**, 40→62.8, 48→68.8. r1_c5 (256 lanes): 16→59.4, **32→53.4**,
48→58.3. r1_c6 (512 lanes): **32→46.5-49.5**.

**Row-tile height sweep (qwen3-235b-fp8 4096×512)** — r1_c1: 1→122.3, 8→45.0, 16→41.6,
32→51.2, 64→71.7. r1_c3: 8→44.7, **16→41.4**, 24→47.7, 32→49.4, 64→71.0. r1_c6:
**16→37.3-37.9**, 32→39.8.

**Row-tile height sweep (small / mid shapes)** — decode (64×1024) r1_c1: 1→4.91,
**2→4.47**, 4→5.35; r1_c3: 1→6.57, **2→4.97**, 4→5.21. gpt-oss (256×512) r1_c1:
1→10.1, 2→8.9, 4→8.4, **8→7.37**; r1_c3: 2→10.1, **8→6.33**, 16→6.54. fp8-fp16 (1024×512)
r1_c1: 4→17.3, **8→16.1**, 16→18.3, 32→21.2; r1_c3: 4→17.2, **8→15.6**, 16→17.5, 32→20.1.
odd-tokens (7×128) r1_c1: **1→2.50**, 2→2.91; r1_c3: 1→3.36, 2→3.82.

**Lane-count sweep**: 128 → 256 → 512 lanes: prefill 63.6 → 55.7 → 46.5, fp8 42.5 → 38.1 →
37.3 µs. 512 lanes hurt tile shapes under 8192 elements (decode 4.4 → 6.6, gpt-oss 6.3 → 10.0),
which is why the merged policy keeps 128 there.

**Store-path sweep** (same tile, same lanes): 16-bit prefill 32×1024 → DMA 78.3 vs lane 60.9;
fp8 16×512 → DMA 41.6 vs lane 157.6 (and lane fails L0-5); small fp8 tiles have no legal lane
form at all. Hence the per-cache-width rule.

**Split-by-pool sweep**: prefill 32×1024 65.6 (split) vs 63.6 (joint, 128 lanes); fp8 32×512
48.7 vs 49.4 — no gain, branch closed.

Final config per workload (from the policy): decode 2×1024, prefill 32×1024, fp8 16×512,
gpt-oss 8×512, odd 1×128, fp8-fp16 8×512; lane count 512 for prefill/fp8, 128 elsewhere.

## Final Performance Test Data

Same final candidate for every row (`final_rowdma_tuned`, `perf_opt/paged_cache_write.py`,
sha256 recorded in `perf_records.jsonl`), platform protocol, median of 3 samples.

| kernel_id | workload_id | Task Duration baseline (µs) | Task Duration final (µs) | final candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py::_paged_cache_write_kernel::main | llama-8b-decode-bfloat16-bfloat16 | 16.115 | 4.471 | final_rowdma_tuned |
| tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py::_paged_cache_write_kernel::main | llama-8b-decode-float16-float16 | 16.097 | 4.201 | final_rowdma_tuned |
| tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py::_paged_cache_write_kernel::main | llama-8b-prefill-bfloat16-bfloat16 | 811.486 | 48.119 | final_rowdma_tuned |
| tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py::_paged_cache_write_kernel::main | qwen3-235b-fp8-bfloat16-float8_e4m3fn | 270.596 | 37.695 | final_rowdma_tuned |
| tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py::_paged_cache_write_kernel::main | gpt-oss-120b-bfloat16-bfloat16 | 26.567 | 6.622 | final_rowdma_tuned |
| tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py::_paged_cache_write_kernel::main | odd-tokens-and-page-float16-float16 | 8.200 | 2.694 | final_rowdma_tuned |
| tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py::_paged_cache_write_kernel::main | qwen3-235b-fp8-fp16-float16-float8_e4m3fn | 79.721 | 15.852 | final_rowdma_tuned |

Derived (same numbers, the columns a reviewer wants next to them):

## Derived metrics and goal check

| workload | baseline → final | improvement | final effective BW | final MBU vs 1.6 TB/s |
|---|---|---:|---:|---:|
| llama-8b-prefill | 811.486 → 48.119 µs | **-94.1% (16.9x)** | 1.40 TB/s | 87% |
| qwen3-235b-fp8 | 270.596 → 37.695 µs | **-86.1% (7.2x)** | 0.33 TB/s | 21% |
| qwen3-235b-fp8-fp16 | 79.721 → 15.852 µs | -80.1% (5.0x) | 0.20 TB/s | 12% |
| gpt-oss-120b | 26.567 → 6.622 µs | -75.1% (4.0x) | 0.16 TB/s | 10% |
| llama-8b-decode-bf16 | 16.115 → 4.471 µs | -72.3% (3.6x) | 0.12 TB/s | 7% |
| llama-8b-decode-fp16 | 16.097 → 4.201 µs | -73.9% (3.8x) | 0.12 TB/s | 8% |
| odd-tokens-and-page | 8.200 → 2.694 µs | -67.1% (3.0x) | launch floor | launch floor |

Goal check: the MFU-dominant fp8 case and the byte-largest prefill both far exceed the "at least
-40%" bar (-86% / -94%), the small decode shapes do not regress (3.6x faster), and the fp8
output contract is unchanged (clamp/RNE/NaN, neg-slot skip, distinct-slot last-writer-wins):
every precision level is byte-exact. Prefill lands at 87% of nominal HBM (94% of the measured
1.49 TB/s DMA ceiling), i.e. the design's own floor (46.0 µs) with 5% overhead. The fp8 case
sits at 21% MBU, below the 30-50% band: it is store-issue bound — 8192 scattered 512 B rows at
~0.25 µs of core time each is 36.6 µs of the 37.7 µs measured, and a one-byte element cannot be
widened per lane. Reaching the band would need a different store granularity (several scattered
rows per descriptor), which the hardware's row-scatter form does not offer; recorded here as
the remaining headroom rather than a design-layer ceiling, because the measured bandwidth floor
for this payload class (0.83 TB/s per-row DMA vs 1.51 TB/s contiguous) bounds the shape, not the
design's algorithm.

## 8. Smoke / skipped

| workload_id | kind | precision | note |
|---|---|---|---|
| paged-write-smoke-float16-float16 | smoke | PASS (byte-exact) | manifest's first workload, `pytest.mark.smoke`; no performance collection |
| paged-write-smoke-bfloat16-bfloat16 | smoke | PASS (byte-exact) | same |
| paged-write-smoke-bfloat16-float8_e4m3fn | smoke | PASS (byte-exact, NaN canonicalised) | same |

No workload is skipped. Smoke evidence: `perf_opt/experiments/smoke_precision.py`
(`[smoke] PRECISION_PASS max_diff=0`), plus L0/L1/L2/boundary logs under `perf_opt/logs/`.

## 9. Precise semantics kept (why the fast path is still exact)

- Load/stage moves the same bytes; the fp8 conversion is the same
  `cast_f32 → /scale → clamp(-448,448) → cast_e4m3` expression, now over a staged UB tile.
- A negative slot skips the row on both paths: DMA path gates the per-row `T.copy` with
  `if slot >= 0`; lane path uses `slot >= 0` as the store predicate.
- Tail blocks clamp `nr = min(rows, tokens - row0)` for the load and predicate every row store
  with `row0 + i < tokens`; the `Slots` read in the lane path is index-clamped so no out-of-range
  read happens on a tail block.
- The destination region keeps a symbolic single-row extent (`slot : slot + one`,
  `one = min(nr,1)`): a literal 1 collapses the region to 1-D and the compiler then rejects the
  dynamic row index (`CopyNode::MakeIndices: Index 1 out of bounds 1`). Per-row stores are
  unrolled for the same reason (a dynamic source row inside a `T.serial` loop hits the same
  collapse). Both are lowering constraints, not numerical ones.
- Tests use `atol=rtol=0` value comparison plus a raw-byte comparison (with the documented
  negative-NaN canonicalisation), and L0-10 asserts that no row outside the named slots changed.

## 10. Reproduction

```bash
cd examples/paged_cache_write_perf
source ../../.agents/platforms/toolchains/ascend/env-950pr.sh
export TMPDIR=/workspace TILELANG_CACHE_DIR=/tmp/opencode/tl-cache-opt-paged
PY=/workspace/venv-ascend950/bin/python

# baseline (frozen Stage 3 module, wrapper's tile policy)
$PY perf_opt/experiments/run_experiments.py --candidate-id baseline-stage3 --round 0 \
    --phase baseline --reps 3 --l0 --baseline-policy
# a branch
$PY perf_opt/experiments/run_experiments.py --module perf_opt/paged_cache_write_round1_c6.py \
    --candidate-id r1_c6_512lanes --parent-id r1_c3_perdtype --round 1 --phase candidate \
    --reps 2 --workloads llama-8b-prefill-bfloat16-bfloat16 \
    --configs "llama-8b-prefill-bfloat16-bfloat16=32:1024"
# the delivered file
$PY perf_opt/paged_cache_write.py --level all
$PY perf_opt/experiments/run_experiments.py --module perf_opt/paged_cache_write.py \
    --candidate-id final_rowdma_tuned --round 3 --phase final --reps 3
```
The final module is regenerated by `perf_opt/experiments/make_final.py` from the tuned kernel
body plus the byte-exact harness; re-running it changes the file hash, so re-record `final`
afterwards.

## 11. Skill Retrospective

### What went well
- Reading the platform pattern library first paid off twice: `CONST-950-dma-vs-lane-gm` pointed
  at the descriptor path (+ a measured 1.49 TB/s ceiling to aim at), and the Stage 3 case entry
  told me the fp8 scalar-store trap before I hit it.
- The single-change branch discipline isolated the two independent effects cleanly (load path
  vs store path), and the store path turned out to depend on the *element width*, which no
  branch alone would have shown.
- Generating code and reading it (`.cce`) was decisive: my first DMA attempt silently compiled
  every copy into a SIMT per-lane loop because the copies sat inside `T.SimtVF`; the timing
  (663 µs) made no sense until the source showed `_simt_vf_0` doing the loads.

### What went wrong / process issues
1. **The template's first version wrapped the whole block in `T.SimtVF`**, which silently turns
   every `T.copy` into a per-lane SIMT loop (no diagnostic, no warning). Cost: one round of
   measurements (600-663 µs) that had to be thrown away. The knowledge base has no entry for
   "copies inside `T.SimtVF` are not DMA" — added below as a proposal.
2. **The runner defaulted `l0_pass=true`** when no L0 was run, which put a false flag on one
   branch row. Fixed to require evidence (`--l0` or `--l0-log`), and the affected row is
   documented in §3 rather than deleted (append-only).
3. **A branch file was left mis-wired** after the template's factory signature changed
   (positional args shifted), and its first measurement (157 µs fp8) was recorded before the
   mistake was caught. The stale rows stay in the log with an explanation; a signature change
   should invalidate the branch files in the same edit.
4. **First repro probe measured the wrong kernel**: inputs were created with device-side
   `randn/randperm`, so the first captured launch was that generation, inflating 44 µs to
   105 µs. Fixed by materialising inputs on the host and only moving them (now recorded in the
   entry's boundary note).

### Value points (vp_type, evidence)

- **D** — *per-row UB→GM `T.copy` drains the MTE3 pipe per row; row-granular scatter stores
  scale with row count*: prefill 16384 row stores 78.3 µs (0.83 TB/s) vs lane 60.9 µs vs one
  contiguous 2-D DMA 44.3 µs on the same 67 MB payload. evidence: `perf_opt/opt_log.md` §3,
  `perf_records.jsonl` rows `r1_c1_dma_store` / `r1_c2_lane_store`, generated code in
  `knowledge/repro/TRAP-950-rowstore-mte3-drain.py` (`asc_sync_pipe(PIPE_MTE3)` count 1 vs 0);
  repro `knowledge/repro/TRAP-950-rowstore-mte3-drain.py --time`; toolchain_stamp
  `a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc`. Written back as
  `TRAP-950-rowstore-mte3-drain` in `knowledge/traps-950pr-store-copy.md` + `INDEX.md`.
- **P** — *staged-DMA pattern for scatter-with-quantise operators*: one block-level `T.copy`
  for the contiguous source tile (never per-lane), conversion as one vector pass over the staged
  tile, then the scattered store path chosen by element width (≥2 B: `T.Parallel` per-lane
  predicate store; 1 B: per-row DMA), lane count by tile size (512 lanes from an 8192-element
  tile up), row tile by cache width. evidence: prefill 811.5→48.1 µs, fp8 270.6→37.7 µs,
  decode 16.1→4.5 µs (final records); repro: the branch files
  `perf_opt/paged_cache_write_round1_c{1,2,3,6}.py` are the minimal before/after pair per axis;
  toolchain_stamp as above. Proposed pattern entry: `PL-950-scatter-staged-dma` (P, Tier 1 —
  needs a second independent task).
- **R** — *stage-4 runner must not claim precision it did not run*: `l0_pass` should default to
  false and require `--l0`/`--l0-log`. evidence: this task's `r1_c2_lane_store` fp8 row carries
  `l0_pass=true` while the branch's own L0 run failed L0-5; the runner in this repo now
  implements the fix (`perf_opt/experiments/run_experiments.py`). BP proposal:
  `BP_stage4_l0_evidence_required` (R, Tier 2 — process rule).
- **C** — *case index*: `CASE-paged-write-stage4` — elementwise scatter → staged DMA in one
  round, with the two traps that cost a round each (copies inside `T.SimtVF`, literal-1 region
  extent rejecting a dynamic row index). evidence: this file, `perf_opt/profiles/src/*.cce`
  (before/after generated code) and the records. propose adding to `knowledge/cases.md`.

### Proposals for the skill / knowledge base
- `BP_row_scatter_dma`: when a stage-4 profile shows a scatter op whose destinations are
  single rows, check the store *granularity* first: a row-sized DMA store has a fixed per-row
  cost on this backend, so measure row-store vs lane-store vs the row count before tuning tiles.
- `BP_prim_func_scope`: before tuning, dump the generated `.cce` once and confirm the copies
  are `asc_copy_*` descriptors rather than loops inside `_simt_vf_*`; a `T.copy` placed inside
  `T.SimtVF` is silently lowered per-lane. (Candidate trap entry; not yet written back because
  the mechanism was only observed through this task's kernels, one origin task.)

## 12. Remaining headroom (not pursued in this pass)

- fp8 store-issue bound (21% MBU): would need >1 scattered row per descriptor. The alternative
  measured and rejected here (lane store) is both slower and unsafe for small tiles. A
  structured attempt (e.g. packing two rows when slots are adjacent, or a uint32 view of the
  fp8 pool for 4-byte lane stores) is untested.
- Prefill is at 94% of the measured DMA ceiling; the remaining 6% is the per-row store cost
  (16-bit lane path) plus the launch floor.

## 13. Integration note for the wrapper switch

The tuned tile is a *per-call* policy (the row tile depends on the call's token count, which is
a dynamic extent), so the wrapper's `forward` must resolve `block_rows/block_cols` through the
module-level `pick_config(tokens, heads_kv, dim, physical_tokens, in_dtype, cache_dtype)` that
`perf_opt/paged_cache_write.py` exports, instead of the Stage 3 defaults
(`block_rows=16`, `block_cols=128`, plus the `tokens >= 6144` whole-row switch). The compiled
program stays cached per `(geometry, rows, cols)`, exactly as before. If the wrapper keeps
passing its own default config, the kernel is still correct but runs the 16x128 tile, i.e. it
loses the whole gain. Nothing else in the factory signature changed:
`_paged_cache_write_kernel(heads_kv, dim, physical_tokens, in_dtype, cache_dtype)` returns a
callable `(rows, cols) -> compiled callable`, and the call arguments are unchanged
(`k, v, k_pages, v_pages, slot_mapping[, k_scale, v_scale]` as flat row views).

## 14. Production adoption (integration package + wrapper switch)

### Adopted files

| file | change | sha256 |
|---|---|---|
| `examples/TileOPs/tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/perf_opt/_paged_cache_write_kernel.py` | **new** — the Stage 4 tuned module, copied verbatim from `perf_opt/paged_cache_write.py` (same factory signature, plus the `pick_config` policy) | `99a43c0d94adc6829f4b058069379fe1af66b7995c7a188efe099922c1687685` |
| `…/paged_cache_write_kernel/perf_opt/__init__.py` | **new** — re-exports `_paged_cache_write_kernel` and `pick_config` | — |
| `examples/TileOPs/tileops/kernels/attention/paged_cache_write/paged_cache_write.py` | **modified** — import switch flipped to perf_opt + tile routed through `pick_config` (below) | — |
| `…/paged_cache_write_kernel/history_version/paged_cache_write_pre_opt.py` | **backup** of the pre-optimisation wrapper | `93bac281eb6c07c76dae3eba825ef16712f900d6e7bdca267d48b94efc4fd0f5` |

Nothing else changed: the Stage 3 kernel dir (`paged_cache_write_kernel/_paged_cache_write_kernel.py`),
the op layer, the manifest, the tests and the benchmark are untouched.

### Wrapper diff (complete)

```diff
-# --- baseline (Stage 3) ------------------------------------------------------
-from .paged_cache_write_kernel import _paged_cache_write_kernel
 # --- perf_opt (Stage 4 tuned) ------------------------------------------------
-# Tried: full-row column tile. ... Reverted; the selection has to depend on the token count.
-# from .paged_cache_write_kernel.perf_opt._paged_cache_write_kernel import _paged_cache_write_kernel
+# Staged row-tile DMA load + a scattered store path chosen by cache width, with
+# the tile resolved per call by ``pick_config`` (see ``forward``). ...
+from .paged_cache_write_kernel.perf_opt._paged_cache_write_kernel import (
+    _paged_cache_write_kernel,
+    pick_config as _tuned_pick_config,
+)
+# --- baseline (Stage 3) ------------------------------------------------------
+# from .paged_cache_write_kernel import _paged_cache_write_kernel
+# _tuned_pick_config = None  # uncomment with the line above to toggle back

         # K9: the compiled callable takes (block_rows, block_cols) ...
-        block_cols = self.config["block_cols"]
-        if not self._user_config and k.shape[0] >= _FULL_ROW_MIN_TOKENS:
-            block_cols = min(_FULL_ROW_MAX, -(-self.row_width // 128) * 128)
-        cfg = (self.config["block_rows"], block_cols)
+        if self._user_config or _tuned_pick_config is None:
+            block_cols = self.config["block_cols"]
+            if not self._user_config and k.shape[0] >= _FULL_ROW_MIN_TOKENS:
+                block_cols = min(_FULL_ROW_MAX, -(-self.row_width // 128) * 128)
+            cfg = (self.config["block_rows"], block_cols)
+        else:
+            cfg = _tuned_pick_config(
+                k.shape[0], self.heads_kv, self.dim, self.physical_tokens,
+                self.dtype_str, self.dtype_to_str(self.cache_dtype),
+            )
+            # keep the kernel's published config equal to the tile that runs, so
+            # reports/benchmarks that read ``kernel.config`` record the truth
+            self.config = {"block_rows": cfg[0], "block_cols": cfg[1]}
```

Contract preserved: constructor signature, `default_config`, `_user_config` (a caller-supplied
tile is still honoured verbatim), the flat-row view handed to the kernel, the zero-token early
return, the `None` return, and the compiled-program cache key `(geometry, rows, cols)`. Toggling
the two comment blocks restores the Stage 3 behaviour exactly (the `or _tuned_pick_config is
None` branch keeps the old `_FULL_ROW_*` policy reachable).

The last two lines are the §13 requirement in code: the tile is resolved per call, and the tile
that actually ran is what the report publishes (before it, the report's `config` column showed
the frozen `16x128` default while the call ran something else).

### Production regression

1. `pytest tests/ops/test_paged_kv_cache_write.py -q` (cwd `examples/TileOPs`): **15 passed**
   in 4.85 s — 6 parametrized byte-exact cases (fp16/bf16/fp8 caches, decode + prefill),
   the saturation/NaN/skip cases and the contract cases.
2. Byte-exact pool comparison through the **op layer** (not the Stage 4 harness), i.e. exactly
   the path the benchmark and the tests take (`perf_opt/experiments/integration_check.py`):

| case | tile the wrapper routed | bytes moved | result |
|---|---|---:|---|
| qwen3-235b-fp8 (4096x512, bf16->e4m3) | `(16, 512)` | 12,582,912 | byte_exact=True (fp8 raw bytes, NaN canonicalised) |
| llama-8b-prefill (8192x1024, bf16->bf16) | `(32, 1024)` | 67,108,864 | byte_exact=True |
| llama-8b-decode (64x1024, bf16->bf16) | `(2, 1024)` | 524,288 | byte_exact=True |

   `[integration] BYTE_EXACT_PASS`.

### Single-operator report (msprof)

```
TILEOPS_TASK_DIR=…/examples/paged_cache_write TMPDIR=/workspace \
  /workspace/venv-ascend950/bin/python -m tileops.reporting.cli run \
  --op PagedKVCacheWriteFwdOp --prof-mode msprof
```

Accepted run **`20261124_211651_753746_PagedKVCacheWriteFwdOp`** — `status=passed`,
correctness 100% (15/15, avg max abs error 0.00e+00), **0 invalid**:

| case | baseline T_op (µs) | production T_op (µs) | improvement | achieved BW |
|---|---:|---:|---:|---:|
| qwen3-235b-fp8-bfloat16-float8_e4m3fn | 267.690 | **32.486** | -87.9% (8.2x) | 0.394 TB/s |
| llama-8b-prefill-bfloat16 | 884.653 | **41.834** | -95.3% (21.1x) | 1.606 TB/s |
| qwen3-235b-fp8-fp16-float16-float8_e4m3fn | 78.891 | **13.294** | -83.1% (5.9x) | 0.233 TB/s |
| gpt-oss-120b-bfloat16 | 25.768 | **4.999** | -80.6% (5.2x) | 0.224 TB/s |
| llama-8b-decode-bfloat16 | 15.218 | **3.071** | -79.8% (5.0x) | 0.163 TB/s |
| llama-8b-decode-float16 | 15.418 | **3.027** | -80.4% (5.1x) | 0.165 TB/s |
| odd-tokens-and-page-float16 | 6.463 | **2.194** | -66.1% (2.9x) | 0.004 TB/s |

No case regressed (baseline column = `examples/batch2/mfu_baseline_v1.json`, same framework
protocol). The report's `config` column now shows the tile that ran
(`block_rows/block_cols` = 2/1024, 2/1024, 32/1024, 16/512, 8/512, 1/128, 8/512).

### Why the production numbers are ~15% better than the Stage 4 numbers (protocol, not code)

The framework's msprof path is `warm_up=5, launch_count=10`, latency = **median of the 10
measured launches in one process** (`tileops/benchmark/msprof.py`), so K/V (33.6 MB) and the
scattered destination rows stay hot in the 128 MB L2 across launches. The Stage 4 harness
captured **one** launch per `msopprof` invocation with the warm-up in a *separate* process, so
it measured a cold cache. Same kernel, same protocol family, different steady-state:

| prefill 8192x1024 bf16 | Task Duration | roofline BW (67,174,408 B) |
|---|---:|---:|
| Stage 4 single cold launch (`l2_probe`, 3 samples) | 47.6-48.1 µs | 1.40-1.41 TB/s |
| framework, median of 10 warm launches | 41.4-41.8 µs | 1.61 TB/s |

`perf_opt/experiments/l2_probe.py` re-ran the same kernel with the pool footprint grown from
256 MB to 2 GB (L2 cannot cover it): 47.80 → 47.58 µs, i.e. **flat**. The gain therefore comes
from re-reading the same 33.6 MB of K/V out of L2 inside one process, not from a smaller HBM
working set, and it is a property of the measurement protocol (identical launches), not of the
kernel.

### One marginal acceptance flag (disclosed, not hidden)

A second, otherwise identical run **`20261124_211917_044010_PagedKVCacheWriteFwdOp`** returned
`status=partial` with **1 invalid**: prefill was flagged `invalid (capture)` because its roofline
bandwidth reached 1.6214 TB/s, past the guard `achieved > nominal peak x 1.01` (1.6 TB/s x 1.01 =
1.616). Its other cases were in the same range as the accepted run (fp8 32.040, decode 3.243 /
3.188, gpt-oss 4.689, odd 1.726, fp8-fp16 13.526) and correctness was again 15/15.

Reading: the operator's roofline counts four HBM streams, while under this protocol the input
stays L2-resident, so the "achieved bandwidth" is an upper bound on HBM traffic and sits right at
the nominal peak. Prefill crosses the guard by ~0.3% depending on which of the two runs you look
at, which is why one run passes and the other is partial. It is not a kernel defect (the same
fingerprint is byte-exact and the cold-capture number is 1.40 TB/s, comfortably below nominal).

Recommendation to the conductor (platform-metadata item, outside Stage 4's write scope): for
scatter/quantise operators whose input fits in L2, the ±1% "physically-impossible" guard against
the nominal 1.6 TB/s should either use the measured streaming peak (`1.424 TB/s` in
`mfu_baseline_v1.json`) with an explicit L2 allowance, or the operator should be exempted from
that particular check with the reason recorded — otherwise the fastest runs of this operator are
the ones most likely to be marked invalid.

### Records

`perf_records.jsonl` gained 7 rows (`phase=merged`, `candidate_id=production_integrated`,
`parent_id=final_rowdma_tuned`, round 4, artifact = the integrated perf_opt module with its
sha256, `msprof_raw_path` = the accepted report's artifact dir). The Stage 4 rows are unchanged
(append-only); the `final` rows stay the ones the Final Performance Test Data table is
reconciled against.
