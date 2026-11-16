"""
The phi-prune command, for a checkpoint that save_checkpoint wrote: check reads every mask against its
pattern and checks that each pruned weight in the state dict is zero, and info prints how many weights
each mask keeps. prune loads a saved state dict, prints how to prune with the Python API and exits 1,
since a model's class cannot be rebuilt from its saved weights.
check and info exit 1 for a file without a masks entry, check also when a mask or a pruned weight
fails, and a command line that argparse cannot read exits 2.

Usage:
    phi-prune check pruned.pt
    phi-prune info pruned.pt

The command exports nothing; phi_prune.export.export_onnx writes a model to ONNX.
"""

import argparse
import sys

import torch

# A saved checkpoint can name the φ pattern "zeckendorf"
PATTERN_ALIASES = {"zeckendorf": "phi"}


def cmd_prune(args):
    """Load a saved state dict, print how to prune with the Python API, and exit 1."""
    print(f"Loading {args.model}...")
    state_dict = torch.load(args.model, map_location="cpu", weights_only=True)

    # A model's class cannot be read from its saved weights, so the command has no model to prune
    print("ERROR: phi-prune prune cannot rebuild a model from its saved weights, so it cannot prune one.")
    print("Prune and fine-tune with the Python API instead:")
    print()
    print("    from phi_prune import prune, finetune")
    print("    model, masks = prune(model)")
    print("    finetune(model, train_loader, epochs=40, masks=masks)")
    sys.exit(1)


def cmd_check(args):
    """
    Check every mask of a checkpoint against its pattern, along the axis it records, and that each
    pruned weight in its state dict is zero; 0 when all hold, 1 otherwise.
    """
    print(f"Loading {args.model}...")
    data = torch.load(args.model, map_location="cpu", weights_only=False)

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
    layer is masked; 0, or 1 for a file without a masks entry.
    """
    data = torch.load(args.model, map_location="cpu", weights_only=False)

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

    # prune
    p = sub.add_parser("prune", help="Print how to prune with the Python API, and exit 1")
    p.add_argument("model", help="A saved state dict, a .pt file")
    p.add_argument("--output", "-o", default="pruned.pt")
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--finetune-epochs", type=int, default=40)

    # check
    p = sub.add_parser("check", help="Check every mask against its pattern, and that every pruned weight is zero")
    p.add_argument("model", help="A checkpoint that save_checkpoint wrote")

    # info
    p = sub.add_parser("info", help="Print how many weights each mask keeps")
    p.add_argument("model", help="A checkpoint that save_checkpoint wrote")

    args = parser.parse_args()

    if args.command == "prune":
        cmd_prune(args)
    elif args.command == "check":
        sys.exit(cmd_check(args))
    elif args.command == "info":
        sys.exit(cmd_info(args))


if __name__ == "__main__":
    main()
