"""
First prototype of the shift-and-add dot product (before phi_prune.register.dot).

Synthetic Laplace conv weights (256 x 128 x 3 x 3), φ-pruned and encoded per
output channel with 10 digits, against half-normal activations quantized to
0-255; 8 kept channels, one patch. Prints additions per dot product, settling
rounds per addition and accumulator width for a φ and a binary accumulator.
The notebook .docs/register_settling.ipynb grew out of this.
"""

import time

import numpy as np
import torch

from phi_prune.encoding import FibonacciEncoder
from phi_prune.masks import phi_mask
from phi_prune.register import BINARY, KINDS, PHI, from_int, normalize_rounds, place_values, pour, value


def main():
    g = torch.Generator().manual_seed(0)
    u = torch.rand((256, 128, 3, 3), generator=g) - 0.5
    w = -torch.sign(u) * torch.log1p(-2 * u.abs())
    mask = phi_mask(w)
    enc = FibonacciEncoder(10)
    e, scale, _, off = enc.encode_tensor(w, mask, return_offset=True, axis=0)
    rng = np.random.default_rng(0)
    x = np.clip(np.round(np.abs(rng.normal(0, 64, 1152))), 0, 255).astype(int)
    rows = [c for c in range(256) if mask[c, 0, 0, 0]][:8]
    t = time.time()
    for kind in KINDS:
        adds = rounds = worst = width = 0
        for c in rows:
            lv = enc.levels(e[c], scale[c], off[c]).astype(int)
            acc = [0]
            for level, xi in zip(lv, x):
                if xi == 0 or level == 0:
                    continue
                ds = from_int(int(level), kind)
                pv = place_values(kind, len(ds))
                for k, d in enumerate(ds):
                    if d:
                        acc, r = normalize_rounds(pour(acc, from_int(int(xi) * pv[k], kind)), kind)
                        adds += 1
                        rounds += r
                        worst = max(worst, r)
            width = max(width, len(acc))
            assert value(acc, kind) == int(np.dot(lv, x))
        print(f"{kind:<6} additions per dot product {adds / len(rows):.0f}, rounds per addition "
              f"{rounds / adds:.2f} (worst {worst}), accumulator width {width}, {time.time() - t:.1f} s")


if __name__ == "__main__":
    main()
