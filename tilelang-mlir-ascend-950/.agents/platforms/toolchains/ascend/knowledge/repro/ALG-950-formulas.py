"""ALG-950-formulas: CPU formula checks, not Ascend kernel certification.
First verified: 2026-11-19; PyTorch 2.13.0+cpu, 2026-11-19.
"""

import json
import torch
import torch.nn.functional as F


def main():
    torch.manual_seed(0)
    for n in (8, 17):
        x = torch.randn(2, n, dtype=torch.float64)
        eps = 1e-5
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).square().mean(-1, keepdim=True)
        exponent = (x - x.max(-1, keepdim=True).values).exp()
        pairs = {
            "softmax": (
                exponent / exponent.sum(-1, keepdim=True),
                torch.softmax(x, -1),
            ),
            "layernorm": (
                (x - mean) / torch.sqrt(variance + eps),
                F.layer_norm(x, (n,), eps=eps),
            ),
            "rmsnorm": (
                x / torch.sqrt(x.square().mean(-1, keepdim=True) + eps),
                F.rms_norm(x, (n,), eps=eps),
            ),
        }
        q, k, v = [torch.randn(1, 1, 4, n, dtype=torch.float64) for _ in range(3)]
        pairs["attention"] = (
            torch.softmax((q @ k.transpose(-1, -2)) / n**0.5, -1) @ v,
            F.scaled_dot_product_attention(q, k, v, dropout_p=0),
        )
        for name, (actual, expected) in pairs.items():
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            print(
                json.dumps(
                    {"formula": name, "width": n, "device": "cpu", "passed": True}
                )
            )


if __name__ == "__main__":
    main()
