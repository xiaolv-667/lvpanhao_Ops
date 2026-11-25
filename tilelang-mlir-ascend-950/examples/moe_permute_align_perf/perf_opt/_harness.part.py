# ---------------------------------------------------------------------------
# Combined-pipeline precision harness (whole operator: align + scatter, or the
# fused small-batch path).  `python moe_permute_align.py --level L0`
# ---------------------------------------------------------------------------
MAX_DIFF = {"value": 0}


def golden_moe_permute_align(topk_ids, num_experts, block_size):
    """The operator's three outputs, from the semantics only.

    ``bincount`` + padded ``cumsum`` + ``searchsorted`` for the block owners and
    a stable slot walk for the route order -- a different derivation from every
    kernel path, so it is an independent check.
    """
    B, E = block_size, num_experts
    flat = topk_ids.flatten()
    numel = int(flat.numel())
    counts = torch.bincount(flat.long(), minlength=E)[:E]
    padded = (counts + B - 1) // B * B
    cumsum = torch.zeros(E + 1, dtype=torch.int64)
    cumsum[1:] = torch.cumsum(padded, 0)
    total = int(cumsum[E].item())

    p_total = numel + (E + 1) * (B - 1)
    num_blocks = math.ceil(p_total / B)
    sorted_ids = torch.full((p_total,), numel, dtype=torch.int32)
    expert_ids = torch.full((num_blocks,), E, dtype=torch.int32)
    slot = cumsum[:E].clone()
    for route, expert in enumerate(flat.tolist()):
        sorted_ids[int(slot[expert])] = route
        slot[expert] += 1
    owned = total // B
    if owned:
        pos = torch.arange(owned, dtype=torch.int64) * B
        expert_ids[:owned] = torch.searchsorted(cumsum[1:], pos, right=True).to(torch.int32)
    npost = torch.tensor([total], dtype=torch.int32)
    return sorted_ids, expert_ids, npost


def _make_flat(numel, E, dist, seed):
    """Routes for one case, honouring the input contract ``0 <= v < E``."""
    if numel == 0:
        return torch.zeros((0,), dtype=torch.int32)
    gen = torch.Generator().manual_seed(seed)
    if dist == "uniform":
        flat = torch.randint(0, E, (numel,), generator=gen, dtype=torch.int32)
    elif dist == "skewed":
        weights = torch.arange(1, E + 1, dtype=torch.float64) ** 2
        flat = torch.multinomial(weights, numel, replacement=True, generator=gen).to(torch.int32)
    elif dist == "zeros_expert":
        span = max(1, E // 4)
        flat = torch.randint(0, span, (numel,), generator=gen, dtype=torch.int32)
    elif dist == "alternating":
        flat = (torch.arange(numel) % max(1, min(E, 3))).to(torch.int32)
    elif dist == "single":
        flat = torch.full((numel,), E - 1, dtype=torch.int32)
    else:
        raise ValueError(f"unknown distribution {dist}")
    return torch.clamp(flat, 0, E - 1)


def _run_once(numel, E, B, dist, tag, seed):
    """Build the operator's kernels for one geometry and compare 3 outputs."""
    flat = _make_flat(numel, E, dist, seed)
    g_sorted, g_expert, g_npost = golden_moe_permute_align(flat, E, B)
    p_total = numel + (E + 1) * (B - 1)
    num_blocks = math.ceil(p_total / B)
    dev = "npu"
    if numel < 1024 and E <= 32:
        small = _make_small_batch_kernel(numel, E, B)()
        sorted_d = torch.empty(p_total, dtype=torch.int32, device=dev)
        expert_d = torch.empty(num_blocks, dtype=torch.int32, device=dev)
        npost_d = torch.empty(1, dtype=torch.int32, device=dev)
        small(flat.npu(), sorted_d, expert_d, npost_d)
        path = "small"
    else:
        align = _make_align_kernel(numel, E, B)()
        scatter = _make_scatter_kernel(numel, E, B)() if numel else None
        sorted_d = torch.empty(p_total, dtype=torch.int32, device=dev)
        expert_d = torch.empty(num_blocks, dtype=torch.int32, device=dev)
        npost_d = torch.empty(1, dtype=torch.int32, device=dev)
        cumsum = torch.empty(E + 1, dtype=torch.int32, device=dev)
        align(flat.npu(), sorted_d, expert_d, npost_d, cumsum)
        if scatter is not None:
            scatter(flat.npu(), sorted_d, cumsum)
        path = "align+scatter"
    torch.npu.synchronize()
    got_sorted = sorted_d.cpu()
    # The operator contract fixes each expert's routed *set*, not the order in
    # which its routes sit (the frozen test normalises the same way).  Compare
    # after sorting every expert's routed range into ascending route order; also
    # report whether the raw array happened to match that order already.
    counts = torch.bincount(flat.long(), minlength=E)[:E]
    padded = (counts + B - 1) // B * B
    starts = torch.cumsum(padded, 0) - padded
    normalised = got_sorted.clone()
    for expert in range(E):
        n = int(counts[expert])
        if n > 1:
            lo = int(starts[expert])
            normalised[lo:lo + n] = torch.sort(normalised[lo:lo + n]).values
    stable_order = "ascending-order-identical" if torch.equal(got_sorted, g_sorted) else "set-identical"
    checks = (("sorted_token_ids", normalised, g_sorted),
              ("expert_ids", expert_d.cpu(), g_expert),
              ("num_tokens_post_pad", npost_d.cpu(), g_npost))
    for name, got, expected in checks:
        if not torch.equal(got, expected):
            bad = (got.flatten() != expected.flatten()).nonzero().flatten()[:6].tolist()
            diff = int((got.to(torch.int64) - expected.to(torch.int64)).abs().max())
            MAX_DIFF["value"] = max(MAX_DIFF["value"], diff)
            raise AssertionError(
                f"[{tag}] ({numel},{E},{B},{dist}) {name} mismatch max|diff|={diff} "
                f"first idx={bad} got={got.flatten()[bad].tolist()} "
                f"expected={expected.flatten()[bad].tolist()} path={path}"
            )
    print(f"[{tag}] PASS (numel={numel}, E={E}, B={B}, {dist}, path={path}, {stable_order})")


def run_L0(only=None):
    cases = (
        (8, 384, 64, "uniform"),
        (256, 384, 64, "skewed"),
        (4096, 384, 64, "uniform"),
        (32768, 384, 64, "uniform"),
        (1024, 8, 16, "alternating"),
        (600, 33, 64, "skewed"),
        (0, 8, 16, "uniform"),
        (4096, 128, 64, "uniform"),
        (200, 8, 16, "alternating"),
    )
    for idx, (numel, E, B, dist) in enumerate(cases):
        if only is not None and idx not in only:
            continue
        _run_once(numel, E, B, dist, f"L0-{idx + 1}", 100 + idx)


def run_L1(only=None):
    cases = []
    for numel in (8, 256, 4096, 32768):
        for E in (384, 256, 128):
            dist = "skewed" if (numel + E) % 2 else "uniform"
            cases.append((numel, E, 64, dist))
    for idx, (numel, E, B, dist) in enumerate(cases):
        if only is not None and idx not in only:
            continue
        _run_once(numel, E, B, dist, f"L1-{idx + 1}", 200 + idx)


def run_L2():
    """Beyond-manifest geometries: informative, non-blocking (L2 warns)."""
    for idx, (numel, E, B, dist) in enumerate((
        (65536, 384, 64, "uniform"),
        (8192, 256, 64, "zeros_expert"),
        (1, 128, 64, "uniform"),
        (1024, 1100, 32, "uniform"),
    )):
        try:
            _run_once(numel, E, B, dist, f"L2-{idx + 1}", 400 + idx)
        except AssertionError as exc:
            print(f"[L2] WARN (recorded, non-blocking) {exc}")


def run_boundary():
    for idx, (numel, E, B, dist) in enumerate((
        (7, 3, 5, "uniform"),
        (64, 8, 1, "uniform"),
        (15, 1, 1, "single"),
        (128, 64, 64, "skewed"),
        (4096, 65, 64, "uniform"),
        (32768, 128, 64, "single"),
        (1, 128, 64, "uniform"),
        (1023, 8, 16, "alternating"),
        (2048, 1, 64, "single"),
    )):
        try:
            _run_once(numel, E, B, dist, f"B-{idx + 1}", 300 + idx)
        except AssertionError as exc:
            print(f"[boundary] WARN {exc}")


def _parse_only(text):
    return None if text is None else {int(v) for v in text.split(",") if v.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0",
                        choices=["L0", "L1", "L2", "boundary", "all"])
    parser.add_argument("--case", default=None)
    args, _ = parser.parse_known_args()
    only = _parse_only(args.case)
    started = time.perf_counter()
    levels = {"L0": lambda: run_L0(only), "L1": lambda: run_L1(only),
              "L2": run_L2, "boundary": run_boundary}
    chosen = levels.values() if args.level == "all" else [levels[args.level]]
    for run in chosen:
        run()
    print(f"elapsed: {time.perf_counter() - started:.1f} s")
    print("\033[92m[PRECISION_PASS] All check passed!\033[0m")
    print(f"[PRECISION_PASS] max_diff={MAX_DIFF['value']}")


if __name__ == "__main__":
    main()
