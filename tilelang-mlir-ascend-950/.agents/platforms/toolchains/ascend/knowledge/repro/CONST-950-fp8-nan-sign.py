"""CONST-950-fp8-nan-sign: e4m3 NaN sign is canonicalised on device, torch keeps it.
First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-23; origin_task batch2-20261123.

Measured 2026-11-23 on ascend-950pr-9579 (dav-3510), TileLang ascend 0.1.15,
CANN 9.2.1, torch 2.13.0 / torch_npu 2.13.0.rc1.

Claim under test
----------------
For an fp8_e4m3fn store of ``clamp(x / scale, -448, 448)``:

- ``torch`` keeps the sign bit of an input NaN: ``+nan -> 0x7F``,
  ``-nan -> 0xFF`` for both the value and the byte pattern.
- The device conversion writes ``0x7F`` for either sign: the NaN still
  propagates (it is *not* turned into +-448), but its sign is dropped.
  ``T.min/T.max`` instead of ``T.clamp`` would lose the NaN entirely, which is
  why the operator clamps explicitly.

Every non-NaN byte (finite values, +-inf saturation, ties, +-0) is bit-exact
on both sides; only the NaN sign bit differs, and the operator contract
compares floats with ``equal_nan=True``.

Run
---
    cd <task_dir> && source .agents/platforms/toolchains/ascend/env-950pr.sh
    /workspace/venv-ascend950/bin/python repro/CONST-950-fp8-nan-sign.py
"""

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T

FP8 = "float8_e4m3fn"
LIM = 448.0

VALUES = [
    float("nan"),
    -float("nan"),
    float("inf"),
    -float("inf"),
    -1000.0,
    1000.0,
    448.0,
    -448.0,
    1.2,
    1.0625,
    1.1875,
    0.0,
    -0.0,
    0.5,
    7.0,
    -7.0,
]


@T.prim_func
def main(
    K: T.Tensor((1, 16), "float16"),
    KP: T.Tensor((2, 16), FP8),
    Slots: T.Tensor((1,), "int64"),
) -> None:
    with T.Kernel(1) as bx, T.SimtVF(threads=128):
        for i, c in T.Parallel(16, 16):
            token = bx * 16 + i
            if token < 1:
                slot = T.cast(Slots[token], "int32")
                if slot >= 0:
                    KP[slot, c] = T.cast(
                        T.clamp(T.cast(K[token, c], "float32"), -LIM, LIM), FP8
                    )


def main_():
    k = torch.tensor([VALUES], dtype=torch.float16, device="npu")
    kp = torch.zeros((2, 16), dtype=torch.float8_e4m3fn, device="npu")
    slots = torch.tensor([1], dtype=torch.int64, device="npu")
    fn = tilelang.compile(main, target="ascend", out_idx=[])
    fn(k, kp, slots)
    torch.npu.synchronize()

    got = kp[1].cpu().view(torch.uint8).tolist()
    gold = torch.clamp(k.cpu().float(), -LIM, LIM).to(torch.float8_e4m3fn).reshape(-1)
    gold_b = gold.view(torch.uint8).tolist()
    print("device bytes:", [hex(x) for x in got])
    print("torch  bytes:", [hex(x) for x in gold_b])

    # Non-NaN bytes are identical on both sides (saturation, ties, +-0).
    for idx, (g, e) in enumerate(zip(got, gold_b, strict=True)):
        if e in (0x7F, 0xFF):
            continue
        assert g == e, f"byte {idx} differs on a non-NaN input: {hex(g)} != {hex(e)}"

    # torch keeps the NaN sign; the device canonicalises it to 0x7F.
    assert gold_b[0] == 0x7F and gold_b[1] == 0xFF, (
        "torch is expected to keep the NaN sign"
    )
    assert got[0] == 0x7F and got[1] == 0x7F, (
        "device is expected to write 0x7F for both NaN signs"
    )

    # Value-space equality with equal_nan: this is what the operator contract
    # asserts, and it holds.
    torch.testing.assert_close(kp[1].cpu(), gold, atol=0, rtol=0, equal_nan=True)

    # A min/max clamp would destroy the NaN entirely; keep that measured too.
    assert got[2] == 0x7E and got[3] == 0xFE, "+-inf must saturate to +-448"
    print(
        "CONST-950-fp8-nan-sign: assertions hold "
        "(device NaN sign canonicalised to 0x7F, non-NaN bytes bit-exact, equal_nan passes)"
    )


if __name__ == "__main__":
    main_()
