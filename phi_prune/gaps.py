"""
Fibonacci coding of the gaps between sorted positions.

A sparse delta keeps a few positions of a tensor. This module writes them as
the gaps between consecutive kept positions, each gap in its Fibonacci code.
On the delta of the small GPT that scripts/transformer_check.py --part delta
fine-tunes, that takes fewer bits per kept position than Elias gamma, Elias
delta or a fixed-length code, at 1% and at 10% of the entries kept, in each
of the run's three seeds.

The Fibonacci code of n >= 1 is the Zeckendorf representation of n, least
significant digit first, followed by an appended 1. A Zeckendorf
representation never holds two adjacent 1s and its top digit is always 1, so
the appended 1 makes the first '11' of each codeword, the comma that ends it.
The code is therefore a prefix code, and a stream of codewords needs no
lengths or separators:

    1 -> 11    2 -> 011    3 -> 0011    4 -> 1011    5 -> 00011

Bitstreams are strings of '0' and '1', as FibonacciEncoder.encode_to_stream
writes them.
"""

import math
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from phi_prune.register import PHI, from_int


@lru_cache(maxsize=4096)
def fib_code(n: int) -> str:
    """Fibonacci code of n >= 1: its Zeckendorf representation, least significant digit first, then the appended 1."""
    if n < 1:
        raise ValueError(f"Fibonacci codes start at 1, got {n}")
    return "".join(map(str, from_int(n, PHI))) + "1"


def fib_code_length(n: int) -> int:
    """Bits in fib_code(n). Like fib_code, it refuses n below 1."""
    if n < 1:
        raise ValueError(f"Fibonacci codes start at 1, got {n}")
    return len(from_int(n, PHI)) + 1


def encode_gaps(gaps: Iterable[int]) -> str:
    """Concatenated Fibonacci codes of gaps, each >= 1."""
    return "".join(fib_code(int(g)) for g in gaps)


def read_codes(bits: str, start: int = 0, count: Optional[int] = None) -> Tuple[List[int], int]:
    """
    Read Fibonacci codes from bits[start:], count of them or to the end.

    Returns the values and the index just past the last code read. Running out
    of bits inside a codeword, or before count codes, raises ValueError.
    """
    places = [1, 2]
    values, total, k, prev = [], 0, 0, "0"
    i = start
    while i < len(bits) and (count is None or len(values) < count):
        bit = bits[i]
        i += 1
        if bit == "1" and prev == "1":  # the comma
            values.append(total)
            total, k, prev = 0, 0, "0"
            continue
        if bit == "1":
            while len(places) <= k:
                places.append(places[-1] + places[-2])
            total += places[k]
        elif bit != "0":
            raise ValueError(f"bitstream holds {bit!r} at {i - 1}, expected '0' or '1'")
        k += 1
        prev = bit
    if k:
        raise ValueError(f"bitstream ends inside a codeword, {k} digits read without a comma")
    if count is not None and len(values) < count:
        raise ValueError(f"bitstream holds {len(values)} codes from {start}, expected {count}")
    return values, i


def decode_gaps(bits: str) -> List[int]:
    """Gaps back from encode_gaps; a trailing partial codeword raises ValueError."""
    return read_codes(bits)[0]


def positions_to_gaps(positions) -> np.ndarray:
    """Gaps of strictly increasing positions >= 0; the first gap counts from -1, so every gap is >= 1."""
    pos = np.asarray(positions, dtype=np.int64)
    gaps = np.diff(pos, prepend=-1)
    if gaps.size and gaps.min() < 1:
        raise ValueError("positions must be strictly increasing and non-negative")
    return gaps


def gaps_to_positions(gaps: Iterable[int]) -> np.ndarray:
    """Inverse of positions_to_gaps."""
    return np.cumsum(np.asarray(list(gaps), dtype=np.int64)) - 1


def encode_positions(positions) -> str:
    """Strictly increasing positions as a Fibonacci-coded bitstream of their gaps."""
    return encode_gaps(positions_to_gaps(positions))


def decode_positions(bits: str) -> np.ndarray:
    """Positions back from encode_positions."""
    return gaps_to_positions(decode_gaps(bits))


def top_positions(tensor: torch.Tensor, keep: float) -> np.ndarray:
    """Sorted flat positions of the k largest entries of |tensor|, with k = keep times its size, rounded, at least 1."""
    flat = tensor.detach().abs().flatten()
    k = max(1, round(keep * flat.numel()))
    return np.sort(torch.topk(flat, k).indices.cpu().numpy())


def pooled_gaps(tensors: Iterable[torch.Tensor], keep: float):
    """Gaps between each tensor's top_positions, pooled, and the largest tensor's size."""
    gaps, n_max = [], 1
    for t in tensors:
        gaps.append(positions_to_gaps(top_positions(t, keep)))
        n_max = max(n_max, t.numel())
    return np.concatenate(gaps), n_max


def gap_bits(gaps, n_positions: int) -> Dict[str, float]:
    """Mean bits per kept position for each code of the gaps. A gap below 1 has no code and raises ValueError."""
    gaps = np.asarray(gaps)
    fib_len = {g: fib_code_length(int(g)) for g in np.unique(gaps)}
    lg = np.floor(np.log2(gaps))
    _, counts = np.unique(gaps, return_counts=True)
    p = counts / counts.sum()
    return {
        "fibonacci": float(np.mean([fib_len[g] for g in gaps])),
        "elias-gamma": float(np.mean(2 * lg + 1)),
        "elias-delta": float(np.mean(lg + 2 * np.floor(np.log2(lg + 1)) + 1)),
        "fixed-width": float(math.ceil(math.log2(n_positions))),
        # zeroth-order empirical entropy: no code that writes each gap on its own averages fewer bits
        "entropy": float(-(p * np.log2(p)).sum()),
    }
