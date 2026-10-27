"""
Sparse weight deltas between two checkpoints, in Fibonacci codes.

For continual training in rounds: each round's update (new - base) keeps its
largest keep fraction per tensor; the kept positions go out as Fibonacci gap
codes (phi_prune.gaps) and each kept value as a sign digit and the Fibonacci
code of its magnitude level + 1, on a symmetric grid of
FibonacciEncoder(n_digits).max_value levels per side, scaled per tensor.

    header, payload = encode_delta(base.state_dict(), tuned.state_dict(), keep=0.01)
    save_delta("round-07.phid", header, payload)
    ...
    state = base.state_dict()
    apply_delta(state, *load_delta("round-07.phid"))
    base.load_state_dict(state)

The file is one JSON header line and the payload as '0'/'1' text, the layout
export_bitstream uses. Every code ends in '11', so the payload frames itself;
a count, a position or a shape that the header does not match raises
ValueError before any weight is written.
"""

import json
from typing import Dict, Tuple

import numpy as np
import torch

from phi_prune.encoding import FibonacciEncoder
from phi_prune.gaps import encode_positions, fib_code, gaps_to_positions, read_codes, top_positions


def encode_delta(
    base: Dict[str, torch.Tensor],
    new: Dict[str, torch.Tensor],
    keep: float = 0.01,
    n_digits: int = 6,
) -> Tuple[Dict, str]:
    """
    Header and payload for the largest keep fraction of each floating tensor of new - base.

    Tensors missing from base, of another shape, or not floating point are skipped
    and named in the header's "skipped". n_digits sets the magnitude grid:
    FibonacciEncoder(n_digits).max_value levels per side (6 digits: 20, near 5 bits
    with the sign), the 4-6 bit range where Zeckendorf digits beat binary and CSD.
    """
    levels = FibonacciEncoder(n_digits).max_value
    header = {"keep": keep, "n_digits": n_digits, "levels": levels, "tensors": [], "skipped": []}
    parts = []
    for name, w in new.items():
        b = base.get(name)
        if b is None or b.shape != w.shape or not w.is_floating_point():
            header["skipped"].append(name)
            continue
        d = (w.detach().float() - b.detach().float()).flatten().cpu()
        pos = top_positions(d, keep)
        vals = d[torch.from_numpy(pos)].numpy()
        scale = float(np.abs(vals).max()) / levels or 1.0
        mags = np.rint(np.abs(vals) / scale).astype(np.int64)
        signs = vals < 0
        values = "".join(("1" if s else "0") + fib_code(int(m) + 1) for s, m in zip(signs, mags))
        positions = encode_positions(pos)
        header["tensors"].append({"name": name, "shape": list(w.shape), "n_kept": len(pos), "scale": scale})
        parts.append(positions + values)
    return header, "".join(parts)


def decode_delta(header: Dict, payload: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """name -> (flat positions, values) from encode_delta's output."""
    out, i = {}, 0
    for t in header["tensors"]:
        n, numel = t["n_kept"], int(np.prod(t["shape"]))
        gaps, i = read_codes(payload, i, n)
        pos = gaps_to_positions(gaps)
        if n and pos[-1] >= numel:
            raise ValueError(f"{t['name']}: position {pos[-1]} past its {numel} entries")
        vals = np.empty(n)
        for k in range(n):
            if i >= len(payload):
                raise ValueError(f"{t['name']}: payload ends at value {k} of {n}")
            sign = -1.0 if payload[i] == "1" else 1.0
            (code,), i = read_codes(payload, i + 1, 1)
            vals[k] = sign * (code - 1) * t["scale"]
        out[t["name"]] = (pos, vals)
    if i != len(payload):
        raise ValueError(f"payload holds {len(payload) - i} bits past the last tensor")
    return out


def apply_delta(state: Dict[str, torch.Tensor], header: Dict, payload: str) -> Dict[str, torch.Tensor]:
    """Add a decoded delta into state's tensors in place, after checking every tensor; returns state."""
    decoded = decode_delta(header, payload)
    for t in header["tensors"]:
        if t["name"] not in state or list(state[t["name"]].shape) != t["shape"]:
            raise ValueError(f"{t['name']}: not in state, or not of shape {t['shape']}")
    with torch.no_grad():
        for name, (pos, vals) in decoded.items():
            flat = state[name].view(-1)
            idx = torch.from_numpy(pos).to(flat.device)
            flat[idx] += torch.from_numpy(vals).to(flat.device, flat.dtype)
    return state


def delta_stats(header: Dict, payload: str) -> Dict[str, float]:
    """Kept entries, total entries covered, and payload bits per kept entry."""
    kept = sum(t["n_kept"] for t in header["tensors"])
    total = sum(int(np.prod(t["shape"])) for t in header["tensors"])
    return {"kept": kept, "entries": total, "bits_per_kept": len(payload) / kept if kept else 0.0}


def save_delta(path: str, header: Dict, payload: str) -> str:
    with open(path, "w", newline="\n") as f:
        f.write(json.dumps(header) + "\n")
        f.write(payload)
    return path


def load_delta(path: str) -> Tuple[Dict, str]:
    with open(path) as f:
        header = json.loads(f.readline())
        return header, f.read().strip()
