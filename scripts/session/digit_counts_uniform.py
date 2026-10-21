"""
1 digits per value over uniformly spread levels (.docs/register.md, section 5c).

Mean nonzero digits over the values 0 .. L-1 in Zeckendorf, canonical
signed digit (CSD/NAF) and binary form: where CSD overtakes Zeckendorf.
"""

import sys
from pathlib import Path

import numpy as np

from phi_prune.register import PHI, from_int

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformer_check import csd_nonzeros  # noqa: E402


def main():
    print(f"{'levels':>7}{'bits':>7}{'Zeckendorf':>12}{'CSD':>8}{'binary':>8}")
    for n in (16, 21, 55, 144, 377, 987, 4181, 17711):
        z = np.mean([sum(from_int(v, PHI)) for v in range(n)])
        c = np.mean([csd_nonzeros(v) for v in range(n)])
        b = np.mean([bin(v).count("1") for v in range(n)])
        print(f"{n:>7}{np.log2(n):>7.2f}{z:>12.3f}{c:>8.3f}{b:>8.3f}")


if __name__ == "__main__":
    main()
