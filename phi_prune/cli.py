"""
The phi-prune command, for a checkpoint that save_checkpoint wrote: check reads every mask against its
pattern and checks that each pruned weight in the state dict is zero, and info prints how many weights
each mask keeps. The command does not prune: a model's class cannot be rebuilt from its saved weights,
so pruning and fine-tuning go through the Python API, phi_prune.prune and phi_prune.finetune.
check and info read the file with torch.load(weights_only=True), which runs no code from it, and exit
1 for a file that load refuses or that has no masks entry, check also when a mask or a pruned weight
fails; a command line that argparse cannot read exits 2.
Checkpoints that save_checkpoint wrote before it stored scales as tensors hold numpy arrays, which that load allows.

Usage:
    phi-prune check pruned.pt
    phi-prune info pruned.pt

The command exports nothing; phi_prune.export.export_onnx writes a model to ONNX.
"""

import argparse
import pickle
import sys

import numpy as np
import torch

# A saved checkpoint can name the φ pattern "zeckendorf"
PATTERN_ALIASES = {"zeckendorf": "phi"}


def _numpy_arrays():
    """
    The globals a numpy array rebuilds itself by, which a checkpoint holds whose per-channel scales are numpy arrays, as
    save_checkpoint wrote them before it stored float64 tensors, and the float dtype classes whose state the unpickler
    restores.
    """
    multiarray = np._core.multiarray if hasattr(np, "_core") else np.core.multiarray
    return [multiarray._reconstruct, np.ndarray, np.dtype] + [type(np.dtype(t)) for t in (np.float64, np.float32)]


def _load(path):
    """
    The file at path, read with torch.load(weights_only=True), which runs no code from it, with numpy arrays allowed;
    exit 1 when it refuses.
    """
    try:
        if hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals(_numpy_arrays()):
                return torch.load(path, map_location="cpu", weights_only=True)
        # torch 2.4 has no safe_globals, so the allowance holds for the rest of the process
        torch.serialization.add_safe_globals(_numpy_arrays())
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as e:
        # torch gives its reason after "WeightsUnpickler error:", on the same line or the next
        reason = str(e).partition("WeightsUnpickler error:")[2].strip().split("\n")[0]
        print(f"ERROR: torch.load(weights_only=True) refuses {path}.")
        if reason:
            print(reason)
        print("phi-prune reads what save_checkpoint writes; a file you trust loads in Python with weights_only=False.")
        sys.exit(1)


def cmd_check(args):
    """
    Check every mask of a checkpoint against its pattern, along the axis it records, and that each
    pruned weight in its state dict is zero; 0 when all hold, 1 otherwise.
    """
    print(f"Loading {args.model}...")
    data = _load(args.model)

    if "masks" not in data:
        print("ERROR: Checkpoint does not contain masks.")
        print("Save one with phi_prune.export.save_checkpoint(model, masks, path), which records the pattern too.")
        sys.exit(1)

    from phi_prune.masks import mask_stats

    masks = data["masks"]
    state = data.get("state_dict", {})
    # A checkpoint with no pattern key is checked as holding φ masks, and one with no axis key
    # as pruned along axis 0
    kind = data.get("pattern", "phi")
    kind = PATTERN_ALIASES.get(kind, kind)
    axis = data.get("axis", 0)
    all_valid = True
    all_zero = True

    for name, mask in masks.items():
        valid = mask_stats(mask, axis=axis, pattern=kind)["valid"]
        # Every pruned weight is within 1e-10 of zero, as phi_prune.check reads it; a mask whose weight
        # the state dict lacks, or holds in another shape, fails
        weight = state.get(name)
        zeros = (weight is not None and weight.shape == mask.shape
                 and bool((weight[~mask.bool()].abs() < 1e-10).all()))
        density = mask.float().mean().item()
        status = "✓" if valid and zeros else "✗"
        print(f"  {status} {name}: density={density:.3f} valid={valid} zeros={zeros}")
        if not valid:
            all_valid = False
        if not zeros:
            all_zero = False

    rule = "the (1,∞)-RLL constraint" if kind == "phi" else f"the {kind} pattern"
    print()
    if all_valid and all_zero:
        print(f"✓ All masks satisfy {rule}, and every pruned weight is zero.")
    if not all_valid:
        print(f"✗ CORRUPTION DETECTED: some masks break {rule}.")
    if not all_zero:
        print("✗ CORRUPTION DETECTED: some pruned weights are not zero.")
    return 0 if all_valid and all_zero else 1


def cmd_info(args):
    """
    Print how many weights each mask of a checkpoint keeps, and the total over all of them, or that no
    layer is masked; 0, or 1 for a file that torch.load(weights_only=True) refuses or that has no masks
    entry.
    """
    data = _load(args.model)

    if isinstance(data, dict) and "masks" in data:
        masks = data["masks"]
        total_active = 0
        total_params = 0
        for name, mask in masks.items():
            active = mask.sum().item()
            total = mask.numel()
            total_active += active
            total_params += total
            print(f"  {name}: {active:.0f}/{total} ({100*active/total:.1f}%)")
        if total_params:
            print(f"\n  Overall: {total_active:.0f}/{total_params}, "
                  f"{100*total_active/total_params:.1f}% density")
        else:
            print("  No layer is masked.")
        return 0
    print("Not a phi-prune checkpoint: it holds no masks.")
    return 1


def main():
    # φ, ∞, ✓ and ✗ print as ? on a console whose encoding lacks them, in place of UnicodeEncodeError
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(
        prog="phi-prune",
        description="φ-pruning: channel pruning under the (1,∞)-RLL constraint",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # check
    p = sub.add_parser("check", help="Check every mask against its pattern, and that every pruned weight is zero")
    p.add_argument("model", help="A checkpoint that save_checkpoint wrote")

    # info
    p = sub.add_parser("info", help="Print how many weights each mask keeps")
    p.add_argument("model", help="A checkpoint that save_checkpoint wrote")

    args = parser.parse_args()

    if args.command == "check":
        sys.exit(cmd_check(args))
    elif args.command == "info":
        sys.exit(cmd_info(args))


if __name__ == "__main__":
    main()
