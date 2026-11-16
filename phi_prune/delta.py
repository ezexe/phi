"""
Sparse deltas between two checkpoints, in Fibonacci codes.

A model trained in rounds can ship each round as its delta, new minus base.
encode_delta keeps, in each floating tensor of the delta, the top-k entries
by absolute value, with k the fraction keep of the tensor's entries. Each
kept value is quantized symmetrically: its absolute value over the tensor's
scale is rounded to a quantization level from 0 to L, where L is
FibonacciEncoder(n_digits).max_value and the scale is the step that puts the
largest kept value at L, or 1.0 when every kept value is 0. The value is
written as a sign digit, 1 for a negative value, followed by the Fibonacci
code of its level plus 1. The kept positions are written as Fibonacci-coded
gaps, by phi_prune.gaps.

    header, payload = encode_delta(base.state_dict(), tuned.state_dict(), keep=0.01)
    save_delta("round-07.phid", header, payload)
    ...
    state = base.state_dict()
    apply_delta(state, *load_delta("round-07.phid"))
    base.load_state_dict(state)

A .phid file is one JSON header line followed by the payload. The header
holds keep, n_digits and levels, which is L; each tensor the delta carries,
with its name, shape, n_kept and scale; the names of the tensors it skipped;
and with changed_only, that flag and the names of the tensors that did not
change. A writer may add keys of its own, as phi_rank adds round and parent
to every round it ships, and decode_delta passes over them. The payload
holds the tensors in the header's order, each as its n_kept gaps followed by
its n_kept values. It is written as '0' and '1' text,
the layout export_bitstream uses, or with packed=True as bits packed eight to
a byte, an eighth of the size, with their count in the header's
"packed_bits". The Fibonacci code is a prefix code, so with the n_kept counts
a reader splits the payload without lengths or separators. A count, a
position or a shape that the header does not match raises ValueError before
any weight is written.

changed_only=True takes keep as a fraction of the entries that changed rather
than of all of them. A sparse delta, such as that of an embedding table whose
untouched rows never move, then keeps its largest changes and never its
zeros, and a tensor with no change is named in the header's "unchanged" and
costs nothing.
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
    changed_only: bool = False,
) -> Tuple[Dict, str]:
    """
    Header and payload for the top-k entries of each floating tensor of new - base, k a fraction keep of its entries.

    Tensors missing from base, of another shape, or not floating point are skipped
    and named in the header's "skipped", and so is a tensor with no entries, which
    changed_only names in "unchanged" instead. keep outside 0 to 1, and a change
    that is not finite, raise ValueError with the name of what is wrong.

    n_digits sets L, the largest quantization level, to
    FibonacciEncoder(n_digits).max_value. 6 digits give 20, so each kept value
    decodes as the scale times one of the 41 integers from -20 to 20.
    """
    if not 0 <= keep <= 1:
        raise ValueError(f"keep is a fraction from 0 to 1, got {keep}")
    levels = FibonacciEncoder(n_digits).max_value
    header = {"keep": keep, "n_digits": n_digits, "levels": levels, "tensors": [], "skipped": []}
    if changed_only:
        header["changed_only"], header["unchanged"] = True, []
    parts = []
    for name, w in new.items():
        b = base.get(name)
        if b is None or b.shape != w.shape or not w.is_floating_point():
            header["skipped"].append(name)
            continue
        d = (w.detach().float() - b.detach().float()).flatten().cpu()
        bad = int((~torch.isfinite(d)).sum())
        if bad:
            raise ValueError(f"{name}: the change is not finite at {bad} of {d.numel()} entries")
        if changed_only:
            nnz = int(torch.count_nonzero(d))
            if not nnz:
                header["unchanged"].append(name)
                continue
            pos = top_positions(d, keep * nnz / d.numel())
        elif not d.numel():
            header["skipped"].append(name)
            continue
        else:
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
    """Add a decoded delta into state's tensors in place, after checking each tensor's name and shape; returns state."""
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


def save_delta(path: str, header: Dict, payload: str, packed: bool = False) -> str:
    """The header line, then the payload: '0'/'1' text, or with packed=True its bits eight to a byte."""
    if not packed:
        with open(path, "w", newline="\n") as f:
            f.write(json.dumps(header) + "\n")
            f.write(payload)
        return path
    bits = np.frombuffer(payload.encode("ascii"), dtype=np.uint8) - ord("0")
    with open(path, "wb") as f:
        f.write((json.dumps({**header, "packed_bits": int(bits.size)}) + "\n").encode("utf-8"))
        f.write(np.packbits(bits).tobytes())
    return path


def load_delta(path: str) -> Tuple[Dict, str]:
    """(header, payload) from save_delta's file, text or packed."""
    with open(path, "rb") as f:
        header = json.loads(f.readline().decode("utf-8"))
        rest = f.read()
    n = header.pop("packed_bits", None)
    if n is None:
        return header, rest.decode("ascii").strip()
    bits = np.unpackbits(np.frombuffer(rest, dtype=np.uint8))
    if bits.size < n or bits.size - n >= 8:
        raise ValueError(f"packed payload holds {bits.size} bits for the header's {n}")
    return header, (bits[:n] + ord("0")).tobytes().decode("ascii")
