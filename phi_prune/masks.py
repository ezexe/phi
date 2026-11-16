"""
Pruning masks: the φ mask, under the (1,∞)-RLL constraint, and the baseline that keeps 2 of every 4 positions.

The (1,∞)-RLL constraint: no two adjacent positions are both kept. Zeckendorf representations
and canonical signed digit forms obey the same rule, and so does the golden mean shift
over infinite sequences. Of n positions it keeps at most (n + 1) // 2.

phi_mask keeps the maximum-weight independent set on the path graph of the positions, found by
dynamic programming: of every subset with no two adjacent, the one with the largest total score.
With a density below what that set keeps, it keeps instead the set adjacency_dp_density finds nearest that fraction.
two_four_mask builds the baseline the φ mask is compared against.
"""

import torch
import numpy as np
from typing import Optional


def adjacency_dp(scores: np.ndarray) -> np.ndarray:
    """
    The maximum-weight independent set on a path graph: the positions to keep, no two adjacent,
    with the largest total score.

    Dynamic programming over the 1D score array, where at each position i
      keep[i] = skip[i-1] + scores[i]     keeps i, so i-1 was skipped
      skip[i] = max(keep[i-1], skip[i-1]) skips i, after the better of the two
    and a backward pass reads the mask off.

    Args:
        scores: 1D array of scores, such as the L1 norm of each channel

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
    adjacency_dp keeping about a given fraction of positions, by Lagrangian relaxation.

    Subtracts a penalty λ from every score and runs adjacency_dp, which then keeps only
    positions whose score is above λ, and bisects λ until the kept count is closest to
    round(density * n). Ties can leave the count one or two off, and it never exceeds
    (n + 1) // 2, the most the constraint allows.

    Args:
        scores: 1D array of non-negative scores
        density: Fraction of positions to keep, in [0, 1]

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
    The φ mask of a weight tensor: no two adjacent positions along `axis` are both kept.

    Each position is scored by the L1 norm of its slice, and the kept set is the heaviest one the
    constraint allows, from adjacency_dp, or the one near the asked fraction, from
    adjacency_dp_density.

    Along axis 0, a Conv2d weight, shaped [out_c, in_c, kH, kW], and a Linear weight, shaped
    [out, in], are scored by output channel. The mask is broadcast across every other dimension,
    so a pruned output channel loses all of its weights.

    Args:
        weight: Parameter tensor, such as a 2D Linear or a 4D Conv2d weight, read without autograd;
                a 1D tensor's positions are scored by their absolute values
        axis: Dimension the constraint runs along
        score_fn: How a position is scored; "magnitude", the one value, is the L1 norm of its slice
        density: Keep about this fraction of positions, through adjacency_dp_density, instead of
                 the heaviest set the constraint allows

    Returns:
        Binary mask tensor, same shape as weight. Ready to multiply.
    """
    # Score each position along axis
    if score_fn == "magnitude":
        # The L1 norm of the slice at each position; the slices of a 1D tensor are its entries
        dims_to_reduce = [d for d in range(weight.dim()) if d != axis]
        w = weight.detach().abs()
        scores = w.sum(dim=dims_to_reduce) if dims_to_reduce else w
        scores = scores.cpu().numpy()
    else:
        raise ValueError(f"Unknown score function: {score_fn}")

    # The heaviest set with no two adjacent, or the one near the asked fraction
    mask_1d = adjacency_dp(scores) if density is None else adjacency_dp_density(scores, density)

    # Broadcast the mask to the weight's shape
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
    The mask that keeps 2 of every 4 consecutive positions along `axis`: in each group of 4, the 2
    with the largest L1 norm. A trailing group shorter than 4 is kept whole.

    It is the baseline the φ mask is compared against. Like phi_mask, it scores whole slices along
    `axis` and broadcasts, so along axis 0 it keeps or drops whole output channels, unlike NVIDIA's
    2:4 sparsity, which keeps 2 of every 4 weights inside each row.

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
    Whether a 1D binary pattern satisfies the (1,∞)-RLL constraint: no two adjacent 1s.

    Args:
        mask_1d: Iterable of 0s and 1s

    Returns:
        True if no two adjacent values are 1, False otherwise
    """
    prev = 0
    for val in mask_1d:
        if val == 1 and prev == 1:
            return False
        prev = val
    return True


def verify_two_four(mask_1d) -> bool:
    """
    Whether a 1D binary pattern keeps exactly 2 of every full group of 4 consecutive positions,
    and all of a trailing group shorter than 4.

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
    Statistics of a pruning mask, read along `axis` at index 0 of every other dimension.

    Args:
        mask: Mask tensor from phi_mask or two_four_mask
        axis: Dimension the pattern runs along
        pattern: The rule "valid" checks, a key of PATTERNS

    Returns:
        dict with density, sparsity, active_count, total_count, capacity_utilization, the kept count
        over the most the pattern keeps along the axis, and valid, whether the pattern holds. The
        most is (n + 1) // 2 of n positions for "phi", and for "2:4" 2 of every full group of 4 and
        all of a trailing group shorter than 4
    """
    # The mask along axis, at index 0 of every other dimension
    idx = [0] * mask.dim()
    idx[axis] = slice(None)
    pattern_1d = mask[tuple(idx)]
    while pattern_1d.dim() > 1:
        pattern_1d = pattern_1d[0]

    active = pattern_1d.sum().item()
    total = pattern_1d.numel()
    density = active / total if total > 0 else 0
    # The most positions of total the pattern keeps
    most = 2 * (total // 4) + total % 4 if pattern == "2:4" else (total + 1) // 2

    return {
        "density": density,
        "sparsity": 1.0 - density,
        "active_count": int(active),
        "total_count": total,
        "capacity_utilization": active / most if most > 0 else 0,
        "valid": get_pattern(pattern)[1](pattern_1d.cpu().numpy()),
    }


# Mask patterns by name, each with its mask generator and its 1D verifier
PATTERNS = {
    "phi": (phi_mask, verify_mask),
    "2:4": (two_four_mask, verify_two_four),
}


def get_pattern(name: str):
    """The mask generator and 1D verifier of a pattern name, as a pair; ValueError for an unknown name."""
    if name not in PATTERNS:
        raise ValueError(f"Unknown pattern {name!r}: expected one of {sorted(PATTERNS)}")
    return PATTERNS[name]
