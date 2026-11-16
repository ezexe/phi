"""
Command-line interface for phi-prune.

Usage:
    phi-prune prune model.pt --output pruned.pt --finetune-epochs 40
    phi-prune check pruned.pt
    phi-prune info pruned.pt

There is no export command yet; use phi_prune.export.export_onnx.
"""

import argparse
import sys

import torch

# Pattern names that checkpoints saved by older versions carry
PATTERN_ALIASES = {"zeckendorf": "phi"}


def cmd_prune(args):
    """Prune a saved model checkpoint."""
    print(f"Loading {args.model}...")
    state_dict = torch.load(args.model, map_location="cpu", weights_only=True)

    # User must provide a model class — we can't infer architecture from weights
    print("ERROR: Standalone pruning requires --arch flag (not yet implemented).")
    print("Use the Python API instead:")
    print()
    print("    from phi_prune import prune, finetune")
    print("    model, masks = prune(model, density=0.5)")
    print("    finetune(model, train_loader, epochs=40, masks=masks)")
    sys.exit(1)


def cmd_check(args):
    """Verify integrity of a pruned checkpoint."""
    print(f"Loading {args.model}...")
    data = torch.load(args.model, map_location="cpu", weights_only=False)

    if "masks" not in data:
        print("ERROR: Checkpoint does not contain masks.")
        print("Save with: torch.save({'state_dict': model.state_dict(), 'masks': masks}, path)")
        sys.exit(1)

    from phi_prune.masks import get_pattern

    masks = data["masks"]
    # checkpoints saved before 2:4 existed have no pattern, and before 0.3.0 this one was "zeckendorf"
    kind = data.get("pattern", "phi")
    kind = PATTERN_ALIASES.get(kind, kind)
    verify = get_pattern(kind)[1]
    all_valid = True

    for name, mask in masks.items():
        # Extract 1D pattern
        pattern = mask.flatten()
        if mask.dim() > 1:
            idx = [0] * mask.dim()
            idx[0] = slice(None)
            pattern = mask[tuple(idx)]
            while pattern.dim() > 1:
                pattern = pattern[0]

        valid = verify(pattern.numpy())
        density = mask.float().mean().item()
        status = "✓" if valid else "✗"
        print(f"  {status} {name}: density={density:.3f} valid={valid}")
        if not valid:
            all_valid = False

    rule = "the adjacency constraint" if kind == "phi" else f"the {kind} pattern"
    if all_valid:
        print(f"\n✓ All masks satisfy {rule}.")
    else:
        print(f"\n✗ CORRUPTION DETECTED: some masks break {rule}.")
    return 0 if all_valid else 1


def cmd_info(args):
    """Print statistics about a pruned checkpoint."""
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
        print(f"\n  Overall: {total_active:.0f}/{total_params} "
              f"({100*total_active/total_params:.1f}% density)")
    else:
        print("Not a phi-prune checkpoint (no masks found).")


def main():
    parser = argparse.ArgumentParser(
        prog="phi-prune",
        description="φ-pruning: adjacency-constrained structured sparsity",
    )
    sub = parser.add_subparsers(dest="command")

    # prune
    p = sub.add_parser("prune", help="Prune a model")
    p.add_argument("model", help="Path to model checkpoint (.pt)")
    p.add_argument("--output", "-o", default="pruned.pt")
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--finetune-epochs", type=int, default=40)

    # check
    p = sub.add_parser("check", help="Verify mask integrity")
    p.add_argument("model", help="Path to pruned checkpoint")

    # info
    p = sub.add_parser("info", help="Print pruning statistics")
    p.add_argument("model", help="Path to pruned checkpoint")

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
