"""
Two checks of quantized weights: that each kept weight sits on its quantization level, and the
minimal-form check of Zeckendorf digits.

check_tensor_integrity and cassini_check read the weights encode_tensor wrote: a kept weight passes
when it is finite, its quantization level round((w - offset) * scale) lies in [0, max_value], and it
equals level / scale + offset in the tensor's dtype, the value encode_tensor and load_bitstream
write. Noise, NaN and a flipped bit of a stored float almost never land on a level, so they fail.

adjacency_check is the minimal-form check of one list of Zeckendorf digits. A Zeckendorf
representation never holds two adjacent 1s, so a single-bit error that makes a pair of them in
stored digits shows up with no parity bits or CRC, by a scan for 11. Only a 0 flipped to 1 beside a
set digit does that, so the share of single-digit flips the scan catches depends on the quantization
levels stored: on synthetic weights, simulated rates run from 0.4116 to 0.5181 of flips at 8 and 10
digits, and over every level of 8 and 10 digits the expectation is 0.4091 and 0.4167, as
experiments/phi-prune/results/measure_detection.txt in ezexe/p4-phi holds, from the
scripts/measure_detection.py beside it. simulate_corruption draws such flips.
"""

import numpy as np
import torch
from typing import Dict, Tuple


def adjacency_check(codeword: list) -> bool:
    """
    The minimal-form check of one list of Zeckendorf digits.

    Args:
        codeword: List of 0s and 1s

    Returns:
        True if no two adjacent digits are 1, False otherwise
    """
    for i in range(len(codeword) - 1):
        if codeword[i] == 1 and codeword[i + 1] == 1:
            return False
    return True


def check_tensor_integrity(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    encoder,
    scale: float,
    offset: float,
) -> Dict:
    """
    Check that every kept weight of a quantized tensor sits exactly on its quantization level.

    A kept weight w passes when it is finite, its level round((w - offset) * scale) lies in
    [0, max_value], and w equals level / scale + offset cast to the tensor's dtype, the value
    encode_tensor writes.

    Args:
        tensor: Weight tensor that encode_tensor quantized
        mask: Binary pruning mask; None checks every position
        encoder: The FibonacciEncoder that quantized it
        scale: The scale encode_tensor returned, a float or, with axis set, a per-slice array
        offset: The quantization offset encode_tensor returned, in the same form as scale

    Returns:
        dict with passed, failed, total and pass_rate, which is 1.0 when the mask keeps nothing
    """
    w = tensor.detach().cpu()
    keep = np.ones(w.shape, dtype=bool) if mask is None else mask.detach().cpu().bool().numpy()
    w64 = w.double().numpy()
    scale = np.asarray(scale, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        level = np.broadcast_to(np.round((w64 - offset) * scale), w.shape)
        on_level = np.isfinite(w64) & (level >= 0) & (level <= encoder.max_value)
        expected = np.broadcast_to(np.where(on_level, level, 0.0) / scale + offset, w.shape)
    exact = (torch.tensor(expected, dtype=w.dtype) == w).numpy()
    total = int(keep.sum())
    passed = int((on_level & exact & keep).sum())
    return {
        "passed": passed,
        "failed": total - passed,
        "total": total,
        "pass_rate": passed / total if total > 0 else 1.0,
    }


def cassini_check(model, masks: Dict, encoder, scales: Dict) -> Dict:
    """
    check_tensor_integrity on every masked layer of a model.

    Args:
        model: PyTorch model whose masked weights encode_tensor quantized
        masks: Parameter name to mask
        encoder: The FibonacciEncoder that quantized them
        scales: Parameter name to the scale and quantization offset encode_tensor returned; a masked
                layer missing from it raises ValueError

    Returns:
        dict of each layer's report, and "_aggregate" over all of them
    """
    results = {}
    total_passed = 0
    total_failed = 0

    for name, param in model.named_parameters():
        if name not in masks:
            continue
        if name not in scales:
            raise ValueError(f"layer {name} is masked but has no scale and offset in scales")
        scale, offset = scales[name]
        report = check_tensor_integrity(param.data, masks[name], encoder, scale, offset)
        results[name] = report
        total_passed += report["passed"]
        total_failed += report["failed"]

    total = total_passed + total_failed
    results["_aggregate"] = {
        "passed": total_passed,
        "failed": total_failed,
        "total": total,
        "pass_rate": total_passed / total if total > 0 else 1.0,
    }

    return results


def simulate_corruption(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    encoder,
    scale: float,
    offset: float,
    n_flips: int = 1000,
) -> Tuple[int, int]:
    """
    Flip one random digit of the Zeckendorf digits of a random kept weight's quantization level,
    n_flips times, and count the flips the minimal-form check catches.

    Returns:
        detected and n_flips, as a pair
    """
    import random

    levels = encoder.levels(tensor, scale, offset, mask)
    if len(levels) == 0:
        return 0, 0

    detected = 0

    for _ in range(n_flips):
        # The Zeckendorf digits of a random kept weight's quantization level
        cw = encoder.to_codeword(int(levels[random.randrange(len(levels))]))

        # Flip a random digit
        bit_pos = random.randrange(len(cw))
        cw[bit_pos] = 1 - cw[bit_pos]

        # Caught when the flip made two adjacent 1s
        if not adjacency_check(cw):
            detected += 1

    return detected, n_flips
