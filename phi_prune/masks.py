"""
Adjacency-constrained mask generation.

The adjacency constraint: no two consecutive positions may both be active.
This is the same (1,∞)-RLL constraint that governs Fibonacci coding,
CSD arithmetic, and the golden mean shift. Capacity: log₂(φ) ≈ 0.694
bits per position, guaranteeing ≤50% density.

Mask selection uses dynamic programming to find the maximum-weight
independent set under the adjacency constraint — the optimal subset
of weights to keep, given their magnitudes.

two_four_mask builds the 2-of-4 baseline the constraint is compared against.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def adjacency_dp(scores: np.ndarray) -> np.ndarray:
    """
    Find the maximum-weight independent set with no two adjacent elements.

    Dynamic programming over the 1D score array. At each position i:
      keep[i] = skip[i-1] + scores[i]   (take this one, must have skipped previous)
      skip[i] = max(keep[i-1], skip[i-1]) (skip this one, best of either previous)

    Args:
        scores: 1D array of non-negative importance scores (e.g. weight magnitudes)

    Returns:
        Binary mask array, same length as scores. 1 = keep, 0 = prune.
    """
    n = len(scores)
    if n == 0:
        return np.array([], dtype=np.int32)
    if n == 1:
        return np.array([1], dtype=np.int32)

    # Forward pass: compute optimal values
    keep = np.zeros(n, dtype=np.float64)
    skip = np.zeros(n, dtype=np.float64)

    keep[0] = scores[0]
    skip[0] = 0.0

    for i in range(1, n):
        keep[i] = skip[i - 1] + scores[i]
        skip[i] = max(keep[i - 1], skip[i - 1])

    # Backward pass: reconstruct the mask
    mask = np.zeros(n, dtype=np.int32)
    i = n - 1
    while i >= 0:
        if i == 0:
            if keep[0] >= skip[0]:
                mask[0] = 1
            break
        if keep[i] >= skip[i]:
            mask[i] = 1
            i -= 2  # must skip the previous
        else:
            i -= 1

    return mask


def adjacency_dp_density(scores: np.ndarray, density: float, iters: int = 60) -> np.ndarray:
    """
    adjacency_dp with a density target below the constraint's cap.

    Subtracts a threshold from every score and runs adjacency_dp, which then
    keeps only positions whose score beats it; a bisection on the threshold
    finds the mask whose kept count is closest to round(density * n). The
    count can miss by one or two when scores tie, and never exceeds what the
    constraint allows ((n + 1) // 2).

    Args:
        scores: 1D array of non-negative importance scores
        density: Target fraction of positions to keep, in [0, 1]

    Returns:
        Binary mask array, same length as scores. 1 = keep, 0 = prune.
    """
    scores = np.asarray(scores, dtype=np.float64)
    target = round(density * len(scores))
    best = adjacency_dp(scores)
    if best.sum() <= target:
        return best
    lo, hi = 0.0, float(scores.max()) + 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        mask = adjacency_dp(scores - mid)
        if abs(int(mask.sum()) - target) < abs(int(best.sum()) - target):
            best = mask
        if mask.sum() > target:
            lo = mid
        elif mask.sum() < target:
            hi = mid
        else:
            break
    return best


def phi_mask(
    weight: torch.Tensor,
    axis: int = 0,
    score_fn: str = "magnitude",
    density: Optional[float] = None,
) -> torch.Tensor:
    """
    Generate an adjacency-constrained pruning mask for a weight tensor.

    The mask enforces: no two adjacent positions along `axis` are both active.
    Active positions are chosen to maximize total importance (weight magnitude
    by default).

    For Conv2d weights (shape: [out_c, in_c, kH, kW]), axis=0 constrains
    along output channels. For Linear weights (shape: [out, in]), axis=0
    constrains along output features.

    The mask is broadcast across all other dimensions — if output channel i
    is pruned, all its weights are zeroed.

    Args:
        weight: Parameter tensor (2D or 4D)
        axis: Dimension to apply the adjacency constraint along
        score_fn: Importance scoring method. Currently: "magnitude"
        density: Keep about this fraction of positions (adjacency_dp_density)
                 instead of the heaviest set the constraint allows, which
                 lands just under 0.5

    Returns:
        Binary mask tensor, same shape as weight. Ready to multiply.
    """
    # Compute per-position importance scores along the pruning axis
    if score_fn == "magnitude":
        # Sum of absolute values across all other dimensions
        dims_to_reduce = [d for d in range(weight.dim()) if d != axis]
        scores = weight.abs().sum(dim=dims_to_reduce).cpu().numpy()
    else:
        raise ValueError(f"Unknown score function: {score_fn}")

    # Run DP to find optimal mask
    mask_1d = adjacency_dp(scores) if density is None else adjacency_dp_density(scores, density)

    # Broadcast mask to full tensor shape
    shape = [1] * weight.dim()
    shape[axis] = weight.shape[axis]
    mask = torch.tensor(mask_1d, dtype=weight.dtype, device=weight.device)
    mask = mask.reshape(shape).expand_as(weight)

    return mask


def two_four_mask(
    weight: torch.Tensor,
    axis: int = 0,
    score_fn: str = "magnitude",
) -> torch.Tensor:
    """
    Generate a 2-of-4 pruning mask: in every group of 4 consecutive positions
    along `axis`, keep the 2 with the highest importance.

    The baseline that .docs/experiment/Zeckendorf.py compares the φ
    pattern against. Like phi_mask, it scores whole slices along `axis`
    and broadcasts, so for axis=0 it keeps or drops whole output channels —
    unlike NVIDIA's hardware 2:4 sparsity, which keeps 2 of every 4 weights
    inside each row. A trailing group shorter than 4 is kept whole.

    Returns:
        Binary mask tensor, same shape as weight. Ready to multiply.
    """
    if score_fn == "magnitude":
        dims_to_reduce = [d for d in range(weight.dim()) if d != axis]
        scores = weight.abs().sum(dim=dims_to_reduce) if dims_to_reduce else weight.abs()
    else:
        raise ValueError(f"Unknown score function: {score_fn}")

    n = scores.shape[0]
    mask_1d = torch.ones(n, dtype=weight.dtype, device=weight.device)
    full = n - n % 4
    if full:
        groups = scores[:full].reshape(-1, 4)
        top2 = groups.topk(2, dim=1).indices
        mask_1d[:full] = torch.zeros_like(groups, dtype=weight.dtype).scatter_(1, top2, 1.0).reshape(-1)

    shape = [1] * weight.dim()
    shape[axis] = n
    return mask_1d.reshape(shape).expand_as(weight)


def verify_mask(mask_1d) -> bool:
    """
    Verify that a 1D binary pattern satisfies the adjacency constraint:
    no two consecutive 1s.

    Args:
        mask_1d: Iterable of 0s and 1s

    Returns:
        True if valid (no adjacent 1s), False otherwise
    """
    prev = 0
    for val in mask_1d:
        if val == 1 and prev == 1:
            return False
        prev = val
    return True


def verify_two_four(mask_1d) -> bool:
    """
    Verify that a 1D binary pattern keeps exactly 2 of every full group of 4
    consecutive positions, and all of a trailing group shorter than 4.

    Args:
        mask_1d: Iterable of 0s and 1s

    Returns:
        True if valid, False otherwise
    """
    bits = [int(v) for v in mask_1d]
    full = len(bits) - len(bits) % 4
    return all(sum(bits[i:i + 4]) == 2 for i in range(0, full, 4)) and all(bits[full:])


def mask_stats(mask: torch.Tensor, axis: int = 0, pattern: str = "phi") -> dict:
    """
    Compute statistics for a pruning mask.

    Args:
        mask: Mask tensor from phi_mask or two_four_mask
        axis: Dimension the pattern runs along
        pattern: Which rule "valid" checks — a key of PATTERNS

    Returns:
        dict with density, sparsity, active_count, total_count,
        capacity_utilization (actual density / theoretical max of 0.5)
    """
    # Extract 1D pattern along axis
    idx = [0] * mask.dim()
    idx[axis] = slice(None)
    pattern_1d = mask[tuple(idx)]
    while pattern_1d.dim() > 1:
        pattern_1d = pattern_1d[0]

    active = pattern_1d.sum().item()
    total = pattern_1d.numel()
    density = active / total if total > 0 else 0

    return {
        "density": density,
        "sparsity": 1.0 - density,
        "active_count": int(active),
        "total_count": total,
        "capacity_utilization": density / 0.5 if density > 0 else 0,
        "valid": get_pattern(pattern)[1](pattern_1d.cpu().numpy()),
    }


# Mask patterns by name: (mask generator, 1D verifier)
PATTERNS = {
    "phi": (phi_mask, verify_mask),
    "2:4": (two_four_mask, verify_two_four),
}


def get_pattern(name: str):
    """The (mask generator, 1D verifier) pair for a pattern name; ValueError for an unknown one."""
    if name not in PATTERNS:
        raise ValueError(f"Unknown pattern {name!r}: expected one of {sorted(PATTERNS)}")
    return PATTERNS[name]
