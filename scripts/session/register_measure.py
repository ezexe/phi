"""
φ-register against binary register measurements (.docs/register.md, sections 3 and 6).

1. Single-bit flips over every canonical register of a width: share detected,
   share uniquely correctable; rules fired per random addition (random order).
2. Rules fired per sum, for every pair of values below 21, 55, 144.
3. Parallel settling rounds (normalize_rounds) for every pair below 21 .. 610.
4. GPU-style lockstep: expected rounds when 32 lanes wait for the slowest.
"""

import random
import statistics as st

from phi_prune.register import (
    BINARY, KINDS, PHI, add, correction_candidates, flip_detected, from_int, is_canonical,
    normalize_rounds, place_values, pour, value,
)


def flips_and_steps():
    print("1. flips and rules fired per random addition (random firing order)")
    rng = random.Random(0)
    for n in (4, 6, 8, 10, 13):
        for kind in KINDS:
            top = (1 << n) if kind == BINARY else place_values(PHI, n + 1)[-1]
            words = [from_int(v, kind, n) for v in range(top)]
            det = cor = tot = 0
            for w in words:
                for i in range(n):
                    f = list(w)
                    f[i] ^= 1
                    tot += 1
                    if flip_detected(f, kind):
                        det += 1
                        c = correction_candidates(f)
                        assert w in c
                        cor += len(c) == 1
            steps = []
            for _ in range(2000):
                a, b = rng.choice(words), rng.choice(words)
                s, k = add(a, b, kind, rng)
                assert value(s, kind) == value(a, kind) + value(b, kind) and is_canonical(s, kind)
                steps.append(k)
            print(f"  {n:>2} digits {kind:<6} {len(words):>5} values  detected {det / tot:.3f}  "
                  f"correctable {cor / tot:.3f}  rules fired mean {st.mean(steps):.2f} max {max(steps)}")


def equal_values():
    print("\n2. rules fired per sum, same values in both registers (lowest position first)")
    for n in (21, 55, 144):
        for kind in KINDS:
            steps = [add(from_int(a, kind), from_int(b, kind), kind)[1] for a in range(n) for b in range(n)]
            print(f"  values < {n:<4} {kind:<6} mean {st.mean(steps):.2f} max {max(steps)}")


def all_rounds(n, kind):
    return [normalize_rounds(pour(from_int(a, kind), from_int(b, kind)), kind)[1] for a in range(n) for b in range(n)]


def rounds_and_lockstep():
    print("\n3. parallel settling rounds per sum")
    for n in (21, 55, 144, 610):
        for kind in KINDS:
            r = all_rounds(n, kind)
            print(f"  values < {n:<4} {kind:<6} width {len(from_int(n - 1, kind)):>2}  mean {st.mean(r):.2f} worst {max(r)}")
    print("\n4. lockstep: expected rounds when lanes wait for the slowest (values < 144)")
    rng = random.Random(0)
    for kind in KINDS:
        r = all_rounds(144, kind)
        for lanes in (1, 32):
            m = [max(rng.choice(r) for _ in range(lanes)) for _ in range(3000)]
            print(f"  {kind:<6} {lanes:>2} lanes  {st.mean(m):.2f}")


if __name__ == "__main__":
    flips_and_steps()
    equal_values()
    rounds_and_lockstep()
