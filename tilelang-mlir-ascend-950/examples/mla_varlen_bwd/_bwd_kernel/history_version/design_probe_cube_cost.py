#!/usr/bin/env python3
"""Design-time cost model for the packed MLA backward, C2 (Cube) tile 96/64.

Pure arithmetic -- no device, no torch.  Every number quoted in DESIGN.md
§1.6.0 (R3 table, per-iteration port decomposition, roofline rows) is produced
here so a reviewer can recompute it.

Iteration model (bottom-right causal, per request b):
    K_b   = ceil(len_kv_b / BLOCK_N)          key blocks
    M_b   = ceil(len_q_b  / BLOCK_M)          query blocks
    for key block kn (n0 = kn*BLOCK_N):
        loop_st = max(0, (n0 - shift_b) // BLOCK_M)   shift_b = len_kv_b - len_q_b
        loop_ed = max(M_b, loop_st + 1)               (non-empty: T.clear(L0C) unsupported)
        iters  += max(0, loop_ed - loop_st)

Run: python3 history_version/design_probe_cube_cost.py
"""

from __future__ import annotations

D_QK, D_V = 192, 128
LOG2E = 1.4426950408889634

# manifest workloads (src/tileops/manifest/spec/attention.yaml, MLAVarlenBwdOp)
WORKLOADS = [
    ("ds-v3-8x4k", 8, [4096] * 8, 128),
    ("ds-v3-mixed", 6, [512, 1024, 2048, 4096, 8192, 16384], 128),
    ("kimi-k2-4x8k", 4, [8192] * 4, 64),
]


def iters_of(lens, heads, block_n, block_m, causal=True):
    total = 0
    blocks = 0
    for ln in lens:
        shift = 0  # len_q == len_kv in every manifest workload
        n_q = -(-ln // block_m)
        for kn in range(-(-ln // block_n)):
            n0 = kn * block_n
            st = max(0, (n0 - shift) // block_m) if causal else 0
            ed = max(n_q, st + 1)
            total += max(0, ed - st)
            blocks += 1
    return total * heads, blocks * heads


def report(block_n, block_m, tag):
    print(f"\n=== {tag}: block_n={block_n}, block_m={block_m} ===")
    for name, batch, lens, heads in WORKLOADS:
        it, blocks = iters_of(lens, heads, block_n, block_m)
        scores = it * block_n * block_m
        flops = it * (block_n * block_m * 2 * (3 * D_QK + 2 * D_V) + block_m * 2 * D_V)
        rows = sum(lens) * heads
        tq = 358e12  # CONST-950-gemm-mte2-burst: bf16 cube 356-359 TF/s
        print(
            f"{name:15s} H={heads:3d} blocks={blocks:7d} iters={it:9d} "
            f"scores={scores/1e9:7.3f}e9 FLOPs={flops/1e12:6.2f}T "
            f"cube@358TF={flops/tq*1e3:7.2f} ms  fill/row={it/rows:6.2f}"
        )
        # GM traffic (bytes)
        fill = it * (block_m * D_QK * 2 * 2 + block_m * D_V * 2 * 2 + 512)
        resident = blocks * (block_n * D_QK * 2 + block_n * D_V * 2 + D_QK * block_n * 2)
        atomic = it * block_m * D_QK * 4
        write = blocks * (block_n * D_QK * 2 + block_n * D_V * 2)
        print(
            f"                GM: fill={fill/1e9:6.1f} GB resident={resident/1e9:5.1f} "
            f"dq_atomic={atomic/1e9:6.1f} dk/dv={write/1e9:5.2f} "
            f"total={(fill+resident+atomic+write)/1e9:6.1f} GB"
        )
        # per-iteration L1/UB port traffic
        port = (
            block_n * D_QK * 2        # a_k <- k_s
            + block_m * D_QK * 2      # b_lo <- q_s
            + block_n * D_V * 2       # a_k <- v_s
            + block_m * D_V * 2       # b_lo <- do_s
            + block_n * block_m * 2   # a_k <- pT_s
            + D_V * block_m * 2       # b_hi <- doT_s
            + block_n * block_m * 2   # a_k <- dsT_s (dK)
            + D_QK * block_m * 2      # b_hi <- qT_s
            + block_n * block_m * 2   # a_t <- dsT_s (dQ transposed copy)
            + D_QK * block_n * 2      # b_hi <- kT_s
        )
        ub2l1 = 2 * (block_n // 2) * block_m * 2 * 2      # p16_u + dsT_u, both AIVs
        gm2l1 = block_m * D_QK * 2 * 2 + block_m * D_V * 2 * 2 + 512
        resident_amort = (block_n * D_QK * 2 + block_n * D_V * 2 + D_QK * block_n * 2) / (
            it / blocks
        )
        total_port = port + ub2l1 + gm2l1 + resident_amort
        print(
            f"                port/iter: gemm_ops={port} ub2l1={ub2l1} gm2l1={gm2l1:.0f} "
            f"resident={resident_amort:.0f} total={total_port:.0f} B  "
            f"FLOP/B={flops/it/total_port:5.2f}  total={it*total_port/1e12:5.3f} TB"
        )


def main():
    report(96, 64, "C2 = adopted (dense-cube core)")
    report(64, 64, "C1 = v0/v3 baseline (L1 path, Nk=64)")
    report(112, 64, "alt (not adopted)")


if __name__ == "__main__":
    main()
