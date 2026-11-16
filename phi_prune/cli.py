"""
The phi-prune command, for a checkpoint that save_checkpoint wrote: check reads every mask against its
pattern, and info prints how many weights each mask keeps. prune loads a saved state dict, prints how to
prune with the Python API and exits 1, since a model's class cannot be rebuilt from its saved weights.

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
    """Check every mask of a checkpoint against its pattern, along the axis it records; 0 when all hold, 1 otherwise."""
    print(f"Loading {args.model}...")
    data = torch.load(args.model, map_location="cpu", weights_only=False)

    if "masks" not in data:
        print("ERROR: Checkpoint does not contain masks.")
        print("Save one with phi_prune.export.save_checkpoint(model, masks, path), which records the pattern too.")
        sys.exit(1)

    from phi_prune.masks import mask_stats

    masks = data["masks"]
    # A checkpoint with no pattern key is checked as holding φ masks, and one with no axis key
    # as pruned along axis 0
    kind = data.get("pattern", "phi")
    kind = PATTERN_ALIASES.get(kind, kind)
    axis = data.get("axis", 0)
    all_valid = True

    for name, mask in masks.items():
        valid = mask_stats(mask, axis=axis, pattern=kind)["valid"]
        density = mask.float().mean().item()
        status = "✓" if valid else "✗"
        print(f"  {status} {name}: density={density:.3f} valid={valid}")
        if not valid:
            all_valid = False

    rule = "the (1,∞)-RLL constraint" if kind == "phi" else f"the {kind} pattern"
    if all_valid:
        print(f"\n✓ All masks satisfy {rule}.")
    else:
        print(f"\n✗ CORRUPTION DETECTED: some masks break {rule}.")
    return 0 if all_valid else 1


def cmd_info(args):
    """Print how many weights each mask of a checkpoint keeps, and the total over all of them."""
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
        print(f"\n  Overall: {total_active:.0f}/{total_params}, "
              f"{100*total_active/total_params:.1f}% density")
    else:
        print("Not a phi-prune checkpoint: it holds no masks.")


def main():
    parser = argparse.ArgumentParser(
        prog="phi-prune",
        description="φ-pruning: channel pruning under the (1,∞)-RLL constraint",
    )
    sub = parser.add_subparsers(dest="command")

    # prune
    p = sub.add_parser("prune", help="Print how to prune with the Python API, and exit 1")
    p.add_argument("model", help="A saved state dict, a .pt file")
    p.add_argument("--output", "-o", default="pruned.pt")
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--finetune-epochs", type=int, default=40)

    # check
    p = sub.add_parser("check", help="Check every mask against its pattern")
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
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
