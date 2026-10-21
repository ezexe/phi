"""
φ-register and binary register normalization, side by side.

A register holds digits d[0], d[1], ... (LSB first) that may exceed 1 for a
while, e.g. right after two registers are added digit by digit.
Normalization rewrites it to canonical form with local rules that keep its
value; each rule touches at most three nearby digits, so rules can fire in
any order, with no global sweep deciding which goes first.

φ-register, place values 1, 2, 3, 5, 8, ... (FibonacciEncoder's digits):
    merge   d[k], d[k+1] >= 1  ->  d[k] - 1, d[k+1] - 1, d[k+2] + 1
            F(k) + F(k+1) = F(k+2)
    split   d[k] >= 2          ->  d[k] - 2, d[k+1] + 1, d[k-2] + 1
            2 F(k) = F(k+1) + F(k-2); at k = 1, 2 + 2 = 3 + 1 (d[2], d[0]);
            at k = 0, 1 + 1 = 2 (d[1] only)
    canonical: digits in {0, 1} with no two adjacent 1s (Zeckendorf form)

binary register, place values 1, 2, 4, 8, ...:
    carry   d[k] >= 2          ->  d[k] - 2, d[k+1] + 1
    canonical: digits in {0, 1}

Both rule sets reach the same canonical form whatever the order, because
each rule keeps the value and each value has one canonical form; neither
needs a clock for that. What differs: a φ carry also moves down (the split
rule's d[k-2]), and a single flipped bit in a canonical φ-register can
break the no-adjacent-1s rule and be seen, while every binary digit
pattern is a valid number. Normalizing a corrupted φ-register does not
undo the flip: the rules keep the corrupted value and make it canonical.
"""

import random
from typing import List, Optional, Tuple

from phi_prune.integrity import adjacency_check

PHI = "phi"
BINARY = "binary"
KINDS = (PHI, BINARY)


def place_values(kind: str, n: int) -> List[int]:
    """The first n place values of a register: 1, 2, 3, 5, ... or 1, 2, 4, 8, ..."""
    if kind == BINARY:
        return [1 << k for k in range(n)]
    if kind == PHI:
        fibs = [1, 2]
        while len(fibs) < n:
            fibs.append(fibs[-1] + fibs[-2])
        return fibs[:n]
    raise ValueError(f"Unknown register kind {kind!r}: expected one of {KINDS}")


def value(digits: List[int], kind: str) -> int:
    """Integer value of a digit list (LSB first), digits allowed above 1."""
    return sum(d * p for d, p in zip(digits, place_values(kind, len(digits))))


def from_int(v: int, kind: str, n_digits: Optional[int] = None) -> List[int]:
    """Canonical digits of v (greedy from the top), padded to n_digits (no leading zeros when None)."""
    if v < 0:
        raise ValueError("registers hold non-negative integers")
    n = 1
    while place_values(kind, n)[-1] <= v:
        n += 1
    places = place_values(kind, n)
    digits = [0] * n
    for k in range(n - 1, -1, -1):
        if places[k] <= v:
            digits[k] = 1
            v -= places[k]
    return (digits + [0] * (n_digits or 1))[:max(n_digits or 1, _top(digits) + 1)]


def is_canonical(digits: List[int], kind: str) -> bool:
    """Digits all 0 or 1, and for a φ-register no two adjacent 1s."""
    if any(d not in (0, 1) for d in digits):
        return False
    return kind == BINARY or adjacency_check(list(digits))


def rewrites(digits: List[int], kind: str) -> List[Tuple[str, int]]:
    """Every rule that can fire on digits, as (rule, k) pairs."""
    out = []
    for k, d in enumerate(digits):
        if d >= 2:
            out.append(("carry" if kind == BINARY else "split", k))
        if kind == PHI and d >= 1 and k + 1 < len(digits) and digits[k + 1] >= 1:
            out.append(("merge", k))
    return out


def apply(digits: List[int], rule: str, k: int) -> List[int]:
    """Fire one rule at position k; returns a new list, grown when a carry runs off the top."""
    d = list(digits) + [0, 0]
    if rule == "carry":
        d[k] -= 2
        d[k + 1] += 1
    elif rule == "merge":
        d[k] -= 1
        d[k + 1] -= 1
        d[k + 2] += 1
    elif rule == "split":
        d[k] -= 2
        d[k + 1] += 1
        if k >= 2:
            d[k - 2] += 1
        elif k == 1:
            d[0] += 1
    else:
        raise ValueError(f"Unknown rule {rule!r}")
    return d[:max(len(digits), _top(d) + 1)]


def normalize(
    digits: List[int],
    kind: str,
    rng: Optional[random.Random] = None,
    max_steps: int = 100_000,
) -> Tuple[List[int], int]:
    """
    Fire rules until the register is canonical.

    With rng, each step fires a rule picked at random among those that
    can fire, standing in for unclocked hardware where any local rewrite
    may happen first; without it, always the lowest position first.

    Returns:
        (canonical digits, number of rules fired)
    """
    d = list(digits)
    for steps in range(max_steps + 1):
        options = rewrites(d, kind)
        if not options:
            return d, steps
        rule, k = rng.choice(options) if rng is not None else options[0]
        d = apply(d, rule, k)
    raise RuntimeError(f"no canonical form after {max_steps} rewrites")


def normalize_rounds(digits: List[int], kind: str, max_rounds: int = 10_000) -> Tuple[List[int], int]:
    """
    Normalize in rounds, each firing at once every rule it can without two
    rules touching the same digit (picked from the lowest position up).

    Rounds stand in for settling time when every cell rewrites itself in
    parallel. A self-timed register stops after the rounds a sum needs;
    a clocked one waits the worst case on every sum.

    Returns:
        (canonical digits, number of rounds)
    """
    d = list(digits)
    for rounds in range(max_rounds + 1):
        options = rewrites(d, kind)
        if not options:
            return d, rounds
        used = set()
        for rule, k in options:
            cells = {"carry": {k, k + 1}, "merge": {k, k + 1, k + 2}, "split": {k - 2, k, k + 1}}[rule]
            if not cells & used:
                used |= cells
                d = apply(d, rule, k)
    raise RuntimeError(f"no canonical form after {max_rounds} rounds")


def pour(a: List[int], b: List[int]) -> List[int]:
    """Digit-by-digit sum of two registers, digits allowed above 1."""
    n = max(len(a), len(b))
    return [x + y for x, y in zip(a + [0] * (n - len(a)), b + [0] * (n - len(b)))]


def add(a: List[int], b: List[int], kind: str, rng: Optional[random.Random] = None) -> Tuple[List[int], int]:
    """pour, then normalize; returns (digits, rules fired)."""
    return normalize(pour(a, b), kind, rng)


def flip_detected(codeword: List[int], kind: str) -> bool:
    """Whether a register reads as corrupt: only a φ-register can, via adjacent 1s."""
    return not is_canonical(codeword, kind)


def correction_candidates(codeword: List[int]) -> List[List[int]]:
    """
    Canonical φ-registers one 1 -> 0 change away from a corrupt codeword.

    Only 0 -> 1 flips are ever detected, so a single detected flip is
    undone by clearing one of the 1s in an adjacent pair; the candidates
    are the clears that leave no adjacent 1s. One candidate means the flip
    can be corrected; two (a lone 11 pair) means it can only be detected.
    """
    out = []
    for i in range(len(codeword)):
        if codeword[i] == 1 and (
            (i > 0 and codeword[i - 1] == 1) or (i + 1 < len(codeword) and codeword[i + 1] == 1)
        ):
            fixed = list(codeword)
            fixed[i] = 0
            if adjacency_check(fixed):
                out.append(fixed)
    return out


def _top(digits: List[int]) -> int:
    """Index of the highest nonzero digit, -1 for an all-zero register."""
    for k in range(len(digits) - 1, -1, -1):
        if digits[k]:
            return k
    return -1
