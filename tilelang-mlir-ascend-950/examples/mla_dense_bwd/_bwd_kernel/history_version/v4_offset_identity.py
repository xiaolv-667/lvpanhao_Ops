"""A5 offset-identity judge (REVIEW.md stage-3 item 1, replaces P11 alone).

Reads the compiled ``device_kernel.asc`` of the dense ``_bwd_main`` and checks,
from the *addresses* rather than from the source, that the rotation really
pairs the lagged gradient GEMMs with the other half's buffers:

  (1) the six per-iteration L1 buffers written by half body A (generic offset
      set A) are exactly the ones read by half body B's GB, and vice versa;
  (2) the ``dq`` store row offset is ``(kq-1)*BLOCK_M`` in both half bodies and
      ``(loop_ed-1)*BLOCK_M`` in the drain;
  (3) the drain's two branches read set A (odd) and set B (even) respectively;
  (4) no ``asc_sync_wait(PIPE_V, ...)`` inside the AIC body (auxiliary: the
      gradient GEMMs must not wait for the vector unit);
  (5) ``asc_mmad`` / ``__simt_vf__`` counts and the generated-source size.

Usage:
    python history_version/v4_offset_identity.py <device_kernel.asc>

Exit 0 = all five hold, 1 = at least one fails (the design's rollback trigger).
"""
import re
import sys

BLOCK_M = 64
H_STRIDE = 192            # H*D of this geometry (B=2,S=8192,H=8,D=192)
DQ_ROW_STRIDE = 8 * 192   # H*D
ROW_DQ = 1536             # H*D

COPY_RE = re.compile(
    r"asc_copy_l12l0([ab]|a_transpose)\(.*?buf_dyn_l1(?: \+ (\d+))?", re.S)


def aic_block(text):
    a = text.index("if ASC_IS_AIC {")
    b = text.index("if ASC_IS_AIV {")
    return text[a:b]


def aiv_block(text):
    return text[text.index("if ASC_IS_AIV {"):]


def fill_offsets(aic):
    """L1 offsets written by the two F halves, in program order.

    Signature of the four loads of one F: ``nd2nz ... 64, 192`` (q_s),
    ``nd2nz ... 64, 128`` (do_s), ``dn2nz ... 192, 64`` (qT_s),
    ``dn2nz ... 128, 64`` (doT_s).  Two occurrences of each -> [A, B].
    """
    pat = re.compile(r"asc_copy_gm2l1_(nd2nz|dn2nz)\(\(\(__cbuf__ bfloat16_t\*\)buf_dyn_l1"
                     r"(?: \+ (\d+))?\).*?, (\d+), static_cast<asc_load_l2_cache_mode>\(0\), "
                     r"(\d+), (\d+), 0, 0\)")
    got = {"q": [], "do": [], "qT": [], "doT": []}
    for kind, off, _row, r1, r2 in pat.findall(aic):
        off = int(off or 0)
        if kind == "nd2nz" and (r1, r2) == ("64", "192"):
            got["q"].append(off)
        elif kind == "nd2nz" and (r1, r2) == ("64", "128"):
            got["do"].append(off)
        elif kind == "dn2nz" and (r1, r2) == ("192", "64"):
            got["qT"].append(off)
        elif kind == "dn2nz" and (r1, r2) == ("128", "64"):
            got["doT"].append(off)
    assert all(len(v) == 2 for v in got.values()), got
    return got


def ep_offsets(aiv):
    """L1 offsets written by the two EP halves (UB -> L1), in program order."""
    pat = re.compile(r"asc_copy_ub2l1\(\(__cbuf__ void\*\)\(\(\(__cbuf__ bfloat16_t\*\)"
                     r"buf_dyn_l1 \+ \(\(sid \* 768\) \+ (\d+)\)\)\)")
    offs = [int(x) for x in pat.findall(aiv)]
    assert len(offs) == 4, offs
    return {"A": offs[0:2], "B": offs[2:4]}


def gb_groups(aic):
    """Split the AIC body at each ``asc_copy_l0c2gm`` and classify the operand
    copies of the GB body that precedes it by their (pipe, dims) signature.

    ``pT``  : l12l0a 96x64      (6, 4)      ``doT`` : l12l0b 128x64 (8, 4)
    ``dsT`` : l12l0a 96x64 (6, 4) **and** l12l0a_transpose 96x64 (6, 4)
    ``qT``  : l12l0b 192x64     (12, 4)     ``kT``  : l12l0b 192x96 (12, 6)

    GA's copies are (6, 12) / (6, 8) / (4, 12) / (4, 8), so they never collide
    with these signatures; the fill loads are GM->L1 and do not appear here.
    """
    stores = [m.start() for m in re.finditer(r"asc_copy_l0c2gm", aic)]
    assert len(stores) == 4, f"expected 4 dq stores (2 half bodies + 2 drain), got {len(stores)}"
    bounds = [0] + stores
    cop = re.compile(r"asc_copy_l12l0(a|b|a_transpose)\(\(\(__c[ab]__ bfloat16_t\*\)buf_dyn_l0[ab]"
                     r"[^,]*, \(\(__cbuf__ bfloat16_t\*\)buf_dyn_l1 \+ (\d+)\), 0, 0, "
                     r"(\d+), (\d+),")
    groups = []
    for i in range(1, len(bounds)):
        seg = aic[bounds[i - 1]:bounds[i]]
        found = {"dsT_t": set(), "doT": set(), "qT": set(), "kT": set()}
        plain_a = []
        for pipe, off, m, k in cop.findall(seg):
            off, m, k = int(off), int(m), int(k)
            if pipe == "b" and (m, k) == (12, 6):
                found["kT"].add(off)
            elif pipe == "b" and (m, k) == (12, 4):
                found["qT"].add(off)
            elif pipe == "b" and (m, k) == (8, 4):
                found["doT"].add(off)
            elif pipe == "a_transpose" and (m, k) == (6, 4):
                found["dsT_t"].add(off)
            elif pipe == "a" and (m, k) == (6, 4):
                plain_a.append(off)
        assert len(found["dsT_t"]) == 1, found
        dsT = found["dsT_t"].pop()
        pT = [o for o in plain_a if o != dsT]
        assert len(plain_a) == 2 and len(pT) == 1, (plain_a, dsT)
        srcs = sorted({pT[0], dsT, found["doT"].pop(), found["qT"].pop()})
        store = aic[stores[i - 1]:stores[i - 1] + 260]
        groups.append({"srcs": srcs, "store": store, "kT": found["kT"]})
    return groups


def main(path):
    text = open(path).read()
    aic, aiv = aic_block(text), aiv_block(text)
    f = fill_offsets(aic)
    ep = ep_offsets(aiv)
    setA = {f["q"][0], f["do"][0], f["qT"][0], f["doT"][0]} | set(ep["A"])
    setB = {f["q"][1], f["do"][1], f["qT"][1], f["doT"][1]} | set(ep["B"])
    # GB reads only the four lagged operands; q_s/do_s are consumed by GA of
    # their own half body (a same-body RAW, so they carry no version identity).
    gbA = {f["qT"][0], f["doT"][0]} | set(ep["A"])
    gbB = {f["qT"][1], f["doT"][1]} | set(ep["B"])
    ok = True

    print("per-iteration L1 buffers (12):")
    print(f"  version A: q={f['q'][0]} do={f['do'][0]} qT={f['qT'][0]} "
          f"doT={f['doT'][0]} pT={ep['A'][0]} dsT={ep['A'][1]}")
    print(f"  version B: q={f['q'][1]} do={f['do'][1]} qT={f['qT'][1]} "
          f"doT={f['doT'][1]} pT={ep['B'][0]} dsT={ep['B'][1]}")
    assert len(setA) == 6 and len(setB) == 6 and not (setA & setB), "buffers not disjoint"
    assert len(gbA) == 4 and len(gbB) == 4 and not (gbA & gbB), "GB operands not disjoint"

    g = gb_groups(aic)
    # program order: GB(A half) -> GB(B half) -> drain if -> drain else
    names = ["half A GB", "half B GB", "drain if", "drain else"]
    for name, grp in zip(names, g):
        print(f"  {name:11s} reads {grp['srcs']}")
    ok &= g[0]["srcs"] == sorted(gbB)   # half A's GB reads version B
    ok &= g[1]["srcs"] == sorted(gbA)   # half B's GB reads version A
    ok &= g[2]["srcs"] == sorted(gbA)   # odd  drain reads A
    ok &= g[3]["srcs"] == sorted(gbB)   # even drain reads B
    print(f'[1] cross-version pairing           : {"PASS" if ok else "FAIL"}')

    # (2) dq row offsets: half A carries "- 98304" (= BLOCK_M * H*D), half B not,
    #     drain uses (loop_ed-1)*BLOCK_M*H*D.
    minus = "- 98304"
    a_ok = minus in g[0]["store"]
    b_ok = minus not in g[1]["store"]
    last_off = (8192 // BLOCK_M - 1) * BLOCK_M * ROW_DQ
    d_ok = all(str(last_off) in grp["store"] for grp in g[2:])
    print(f'[2] dq offset (kq-1)*64            : halfA {a_ok} halfB {b_ok} '
          f'drain {last_off} {d_ok} -> {"PASS" if (a_ok and b_ok and d_ok) else "FAIL"}')
    ok &= a_ok and b_ok and d_ok

    # (3) drain parity: the "if" branch must be the lowered `n_it % 2 == 1`.
    m = re.search(r"if \(\(\(\(rmod_\d+ <= 0\) \? rmod_\d+ : \(rmod_\d+ - 2\)\) \* -1\) == 1\)",
                  aic)
    parity_ok = bool(m)
    first_drain = aic.find("rmod_")
    print(f'[3] drain if-branch is the odd test  : {"PASS" if parity_ok else "FAIL"} '
          f'(first rmod use at {first_drain})')
    ok &= parity_ok

    # (4) auxiliary: no PIPE_V predecessor wait inside the AIC body.
    vwait = "asc_sync_wait(PIPE_V" in aic
    print(f'[4] no PIPE_V wait in the AIC body   : {"PASS" if not vwait else "FAIL"}')
    ok &= not vwait

    # (5) code shape.
    mmad = text.count("asc_mmad")
    vf = text.count("__simt_vf__")
    size = len(text)
    shape_ok = mmad == 16 and vf == 5 and 50000 <= size <= 53000
    print(f"[5] asc_mmad={mmad} __simt_vf__={vf} size={size} B  : "
          f"{'PASS' if shape_ok else 'FAIL'}")
    ok &= shape_ok
    print(f"[OFFSET_IDENTITY] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
