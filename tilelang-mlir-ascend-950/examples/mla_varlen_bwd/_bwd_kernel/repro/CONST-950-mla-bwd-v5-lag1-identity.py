#!/usr/bin/env python3
"""CONST-950-mla-bwd-v5-lag1-identity: how to *prove* a hand-rotated (lag-1)
consumer reads the other buffer version -- from the generated code, mechanically.

Why this script exists
----------------------
The obvious structural criterion for "the gradient GEMMs were rotated one
iteration behind the epilogue" is *"no ``PIPE_V``-predecessor ``asc_sync_wait``
before the GB copies"* (DESIGN.md v5 V13 / 1.6.4 F-3).  Measured here: the
**frozen v4 body**, which has no rotation at all, also has **zero** such waits
-- the AIV->AIC ordering is not emitted as an explicit ``wait(PIPE_V, PIPE_M)``
in either version.  The criterion therefore passes for both bodies and proves
nothing (REVIEW 5-1).

The discriminating criterion is **buffer-offset identity**: ``GB(k-1)``'s four
per-iteration ``L1 -> L0`` source offsets must be the *other* half body's
statically named buffers and must be disjoint from what the current half body
wrote.  This script extracts the offsets from
``kernel.get_kernel_source()`` and asserts:

  1. half A writes 6 per-iteration tiles (4 fills + ``pT``/``dsT``) and 4 ``GB``
     sources, all of them **outside** the offsets half A wrote -- i.e. the
     gradients read the previous half body's version;
  2. half B's ``GB`` reads 4 offsets **inside** half A's written set (the other
     half again, one iteration later);
  3. the drain's odd branch reads the version the last A half wrote and the even
     branch the B version (the parity mapping of DESIGN.md 3.3 note 9);
  4. exactly one in-loop ``dq`` store carries the lagged row offset
     (``- BM*stride``), the other does not;
  5. the same checks **reject** the unrotated v4 body while the ``PIPE_V``
     criterion accepts both -- the criterion has no discriminating power.

Note on reading the generated code: the kernel body is emitted as an
``ASC_IS_AIC`` section followed by an ``ASC_IS_AIV`` section, so *textual* order
is only program order **within** a unit.  The fills (MTE2, AIC) and the epilogue
writes (MTE3, AIV) are therefore paired by their own per-half order (first
occurrence = half A, second = half B), not by line number across sections.

Run (compile only, no device; cwd = the operator dir):
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    TMPDIR=/workspace ASCEND_RT_VISIBLE_DEVICES="" \
        /workspace/venv-ascend950/bin/python \
        repro/CONST-950-mla-bwd-v5-lag1-identity.py
"""
import importlib.util
import re
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
V5 = TASK / "_bwd_kernel.py"
V4 = TASK / "history_version/_bwd_kernel_impl_s3_v5_attempt1.py"

L1 = r"\(\(__cbuf__ bfloat16_t\*\)buf_dyn_l1"
RE_GM2L1 = re.compile(L1 + r"(?: \+ (\d+))?\)")
RE_UB2L1 = re.compile(r"asc_copy_ub2l1\(\(__cbuf__ void\*\)\(\(\(__cbuf__ bfloat16_t\*\)"
                      r"buf_dyn_l1 \+ \(\(sid \* \d+\) \+ (\d+)\)\)\)")
RE_L12L0 = re.compile(r"asc_copy_l12l0[ab](?:_transpose)?\(+__c[ab]__ bfloat16_t\*\)[^,]+"
                      r", " + L1 + r"(?: \+ (\d+))?")
RE_DQ = re.compile(r"asc_copy_l0c2gm\(+&\(dq_acc\[")
RE_PARITY = re.compile(r"& 1\) == 1")
RE_VWAIT = re.compile(r"asc_sync_wait\(PIPE_V, PIPE_M")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_source(mod):
    """Generated source of the packed body at the smallest useful geometry."""
    fn = mod._bwd_kernel(2, 2, 192, 128, 0, 0, True, 192 ** -0.5, "bfloat16", True, 2)
    return fn.get_kernel_source()


def events(src):
    """Ordered ``(in_loop, kind, value, line)`` stream.

    ``kind``: ``fill`` (GM->L1 dst), ``epi`` (UB->L1 dst), ``src`` (L1->L0 src),
    ``dq`` (L0C->GM store; value -1 when the row range is lagged).
    """
    lines = src.splitlines()
    # no parity/drain branch at all in the unrotated body: everything is "loop"
    split = next((i for i, line in enumerate(lines) if RE_PARITY.search(line)),
                 len(lines))
    out = []
    for index, line in enumerate(lines):
        if "asc_copy_gm2l1" in line:
            out.append((index < split, "fill", int(RE_GM2L1.search(line).group(1) or 0), index))
        if "asc_copy_ub2l1" in line:
            out.append((index < split, "epi", int(RE_UB2L1.search(line).group(1)), index))
        if "asc_copy_l12l0" in line:
            out.append((index < split, "src", int(RE_L12L0.search(line).group(1) or 0), index))
        if RE_DQ.search(line):
            lagged = bool(re.search(r"- \(int64_t\)\d+", line))
            out.append((index < split, "dq", -1 if lagged else -2, index))
    return out


def groups(src):
    """Per-half written tiles, the two in-loop GB source groups, the drain groups."""
    ev = events(src)
    lines = src.splitlines()
    loop = [e for e in ev if e[0]]
    drain = [e for e in ev if not e[0]]
    dqs = [e for e in loop if e[1] == "dq"]
    assert len(dqs) == 2, f"loop has {len(dqs)} dq stores, expected 2"
    first_dq, second_dq = dqs[0][3], dqs[1][3]
    # the query loop header = nearest `for (int32_t i = ...` above the first dq
    loop_start = max(i for i, line in enumerate(lines)
                     if line.strip().startswith("for (int32_t i =") and i < first_dq)
    # NB: the epilogue writes live in the AIV section, i.e. *after* the drain
    # textually (the body is emitted AIC-section then AIV-section), so they are
    # collected from the whole stream while the fills are AIC-only.
    fills = [e[2] for e in ev if e[1] == "fill" and e[0] and e[3] > loop_start]
    epis = [e[2] for e in ev if e[1] == "epi"]
    srcs = [e for e in loop if e[1] == "src" and e[3] > loop_start]
    assert len(fills) == 8, f"{len(fills)} per-iteration fills, expected 8 (4 x 2 halves)"
    assert len(epis) == 4, f"{len(epis)} epilogue tile writes, expected 4 (2 x 2 halves)"
    written_a = set(fills[:4]) | set(epis[:2])
    written_b = set(fills[4:]) | set(epis[2:])
    gb_a = [e[2] for e in srcs if e[3] < first_dq][-6:]
    gb_b = [e[2] for e in srcs if first_dq < e[3] < second_dq][-6:]
    d_src = [e[2] for e in drain if e[1] == "src"]
    assert len(d_src) == 12, f"drain has {len(d_src)} L1->L0 sources, expected 12"
    d_odd, d_even = d_src[:6], d_src[6:]
    resident = set(gb_a) & set(gb_b) & set(d_odd) & set(d_even)
    assert len(resident) == 1, f"expected exactly one resident (kT) tile, got {resident}"
    strip = lambda g: set(g) - resident  # noqa: E731
    return {
        "written_a": written_a, "written_b": written_b,
        "gb_a": strip(gb_a), "gb_b": strip(gb_b),
        "d_odd": strip(d_odd), "d_even": strip(d_even),
        "lagged_dq": sum(1 for e in dqs if e[2] == -1),
    }


def check(src):
    g = groups(src)
    assert len(g["gb_a"]) == 4 and len(g["gb_b"]) == 4, g
    # 1. the lagged GB reads nothing the current half wrote (rotation exists)
    assert not (g["gb_a"] & g["written_a"]), (
        "GB(k-1) reads a buffer the *current* half body wrote: the rotation is not "
        f"expressed -- overlap {sorted(g['gb_a'] & g['written_a'])}")
    # 2. ... and exactly the other half's version
    assert g["gb_a"] <= g["written_b"], (sorted(g["gb_a"]), sorted(g["written_b"]))
    assert g["gb_b"] <= g["written_a"], (sorted(g["gb_b"]), sorted(g["written_a"]))
    # 3. drain parity mapping
    assert g["d_odd"] == g["gb_b"], f"drain(odd) {sorted(g['d_odd'])} != A {sorted(g['gb_b'])}"
    assert g["d_even"] == g["gb_a"], "drain(even) != version B"
    # 4. lagged dq row range: the A-half store is `- BM*stride`
    assert g["lagged_dq"] == 1, f"{g['lagged_dq']} lagged dq stores in the loop"
    return g


def pipv_waits(src):
    """Secondary (non-discriminating) criterion: PIPE_V -> PIPE_M waits in AIC.

    Returns ``(total, in_aic_section)``.  The AIC section is what the rotation
    would have to shorten; the AIV section's own PIPE_V waits are the epilogue
    waiting for its own vector work and exist in both bodies.
    """
    lines = src.splitlines()
    aic = next(i for i, line in enumerate(lines) if "ASC_IS_AIC" in line)
    aiv = next((i for i, line in enumerate(lines) if "ASC_IS_AIV" in line), len(lines))
    return (len(RE_VWAIT.findall(src)),
            sum(1 for line in lines[aic:aiv] if RE_VWAIT.search(line)))


def main():
    g = check(build_source(load(V5, "v5_ident")))
    print("[v5] half A wrote  ", sorted(g["written_a"]))
    print("[v5] half B wrote  ", sorted(g["written_b"]))
    print("[v5] GB(A) reads   ", sorted(g["gb_a"]), "-> version B, disjoint from A")
    print("[v5] GB(B) reads   ", sorted(g["gb_b"]), "-> version A")
    print("[v5] drain odd/even", sorted(g["d_odd"]), "/", sorted(g["d_even"]))
    print("[v5] lagged dq stores in the loop:", g["lagged_dq"])
    src5 = build_source(load(V5, "v5_waitcount"))
    src4 = build_source(load(V4, "v4_waitcount"))
    t5, a5 = pipv_waits(src5)
    t4, a4 = pipv_waits(src4)
    print(f"[PIPE_V->PIPE_M waits] v5 total={t5} aic={a5} | v4 total={t4} aic={a4} "
          "(aic zero in both => that criterion cannot separate the two bodies)")
    try:
        check(src4)
    except AssertionError as exc:
        print(f"[v4] rejected by the offset-identity check, as expected: {str(exc)[:150]}")
    else:
        raise AssertionError("the offset-identity check accepted the unrotated v4 body")
    print("\n[IDENTITY_SUMMARY] ALL_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
