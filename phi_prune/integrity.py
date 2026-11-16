"""
The minimal-form check: no two adjacent 1s in a list of Zeckendorf digits.

A Zeckendorf representation never holds two adjacent 1s, so a single-bit error that makes a pair
of them in stored digits shows up with no parity bits or CRC, by a scan for 11. Only a 0 flipped
to 1 beside a set digit does that, so the share of single-digit flips the scan catches depends on
the quantization levels stored: on synthetic weights, simulated rates run from 0.4116 to 0.5181 of
flips at 8 and 10 digits, and over every level of 8 and 10 digits the expectation is 0.4091 and
0.4167, as results/prune/measure_detection.txt holds from scripts/measure_detection.py.

check_tensor_integrity and cassini_check run the scan on digits they derive again from each kept
weight's quantization level, and to_codeword never writes two adjacent 1s, so they count every
weight as passing whatever the tensor holds.
"""

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
    The minimal-form check on the Zeckendorf digits of every kept weight's quantization level.

    Each weight's level comes from encoder.levels and its digits from to_codeword, which writes no
    two adjacent 1s, so every weight passes whatever the tensor holds.

    Args:
        tensor: Weight tensor that encode_tensor quantized
        mask: Binary pruning mask
        encoder: The FibonacciEncoder that quantized it
        scale: The scale encode_tensor returned, a float or, with axis set, a per-slice array
        offset: The quantization offset encode_tensor returned, in the same form as scale

    Returns:
        dict with passed, failed, total and pass_rate, which is 0 when the mask keeps nothing
    """
    passed = 0
    failed = 0

    for grid_val in encoder.levels(tensor, scale, offset, mask):
        cw = encoder.to_codeword(int(grid_val))

        if adjacency_check(cw):
            passed += 1
        else:
            failed += 1

    return {
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "pass_rate": passed / (passed + failed) if (passed + failed) > 0 else 0,
    }


def cassini_check(model, masks: Dict, encoder, scales: Dict) -> Dict:
    """
    check_tensor_integrity on every masked layer of a model that scales has an entry for; a masked
    layer missing from scales is left out.

    Args:
        model: PyTorch model whose masked weights encode_tensor quantized
        masks: Parameter name to mask
        encoder: The FibonacciEncoder that quantized them
        scales: Parameter name to the scale and quantization offset encode_tensor returned

    Returns:
        dict of each checked layer's report, and "_aggregate" over all of them
    """
    results = {}
    total_passed = 0
    total_failed = 0

    for name, param in model.named_parameters():
        if name in masks and name in scales:
            scale, offset = scales[name]
            report = check_tensor_integrity(
                param.data, masks[name], encoder, scale, offset
            )
            results[name] = report
            total_passed += report["passed"]
            total_failed += report["failed"]

    total = total_passed + total_failed
    results["_aggregate"] = {
        "passed": total_passed,
        "failed": total_failed,
        "total": total,
        "pass_rate": total_passed / total if total > 0 else 0,
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
