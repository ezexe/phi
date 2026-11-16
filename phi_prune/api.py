"""
High-level API for φ-pruning.

    from phi_prune import prune, finetune, check

    model, masks = prune(model, density=0.5)
    finetune(model, train_loader, epochs=40, masks=masks)
    report = check(model, masks)
"""

import copy
from typing import Dict, Tuple, Optional, Iterable

import torch
import torch.nn as nn
import torch.optim as optim

from phi_prune.masks import get_pattern, mask_stats


# ───────────────────────────────────────────────────────────
# PRUNING
# ───────────────────────────────────────────────────────────

def prune(
    model: nn.Module,
    density: float = 0.5,
    axis: int = 0,
    layer_types: Tuple = (nn.Conv2d, nn.Linear),
    min_dim: int = 8,
    inplace: bool = False,
    score_fn: str = "magnitude",
    prune_head: bool = False,
    exclude: Iterable[str] = (),
    pattern: str = "phi",
) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    """
    Apply adjacency-constrained pruning to a model.

    Generates an adjacency-constrained mask for each eligible layer
    and zeros the pruned weights. pattern="2:4" applies the 2-of-4
    baseline instead.

    Args:
        model: Trained PyTorch model
        density: Target density (informational — actual density is
                 determined by the DP algorithm, always ≤0.5)
        axis: Dimension to apply constraint along (0 = output features)
        layer_types: Which layer types to prune
        min_dim: Skip layers smaller than this along the prune axis
        inplace: If False, works on a deep copy
        score_fn: Weight importance metric ("magnitude")
        prune_head: Also prune the model's last weight layer (its last
                    Conv2d, Linear or layer_types module in registration
                    order) — the classifier head in standard architectures —
                    when layer_types selects it. Off by default: along axis 0
                    its positions are the classes, and the constraint keeps
                    at most half of them, leaving the rest with a bias-only
                    logit
        exclude: Module names to leave dense (e.g. {"fc"}), for a head
                 that is not registered last or any other layer to keep
        pattern: "phi" (no two adjacent positions along `axis`) or
                 "2:4" (2 of every 4 consecutive positions along `axis`, the
                 baseline in .docs/experiment/Zeckendorf.py; for axis=0 it
                 keeps whole channels, not NVIDIA's 2-of-4 weights per row)

    Returns:
        (pruned_model, masks) where masks maps param names to mask tensors
    """
    make_mask = get_pattern(pattern)[0]
    if not inplace:
        model = copy.deepcopy(model)

    targets = [(name, m) for name, m in model.named_modules() if isinstance(m, layer_types)]
    keep_dense = {exclude} if isinstance(exclude, str) else set(exclude)
    if not prune_head:
        # The head is the model's last weight layer whatever layer_types selects: with
        # layer_types=(nn.Conv2d,) a Linear head is already out, and the last conv is no head.
        extra = layer_types if isinstance(layer_types, tuple) else (layer_types,)
        weight_layers = [
            n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear) + extra)
        ]
        if weight_layers:
            keep_dense.add(weight_layers[-1])

    masks = {}
    stats = {"total_params": 0, "active_params": 0, "pruned_layers": 0, "pattern": pattern}

    for name, module in targets:
        for pname, param in module.named_parameters(prefix=name):
            if "weight" not in pname:
                continue
            if name in keep_dense or param.shape[axis] < min_dim:
                stats["total_params"] += param.numel()
                stats["active_params"] += param.numel()
                continue

            mask = make_mask(param.data, axis=axis, score_fn=score_fn)
            masks[pname] = mask

            # Apply mask
            param.data.mul_(mask)

            stats["total_params"] += param.numel()
            stats["active_params"] += mask.sum().item()
            stats["pruned_layers"] += 1

    # Attach stats
    stats["density"] = stats["active_params"] / stats["total_params"] if stats["total_params"] > 0 else 0
    stats["sparsity"] = 1.0 - stats["density"]
    model._phi_prune_stats = stats

    return model, masks


# ───────────────────────────────────────────────────────────
# FINE-TUNING
# ───────────────────────────────────────────────────────────

def finetune(
    model: nn.Module,
    train_loader,
    epochs: int = 40,
    masks: Dict[str, torch.Tensor] = None,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    val_loader=None,
    verbose: bool = True,
    criterion: nn.Module = None,
    amp: bool = False,
) -> Dict:
    """
    Fine-tune a pruned model with mask enforcement.

    Zeros gradients on pruned positions after each backward pass,
    ensuring the sparsity pattern is maintained throughout training.

    Args:
        model: Pruned model
        train_loader: Training data
        epochs: Fine-tuning epochs
        masks: Dict of pruning masks (from prune())
        lr: Learning rate
        device: Compute device
        val_loader: Optional validation loader for tracking accuracy
        verbose: Print progress
        criterion: Loss function (default: CrossEntropyLoss)
        amp: Run forward passes under fp16 autocast with loss scaling
             (CUDA devices only; elsewhere training stays fp32)

    Returns:
        dict with training history and best accuracy
    """
    if device is None:
        device = next(model.parameters()).device
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    use_amp = _amp_enabled(amp, device)

    model.to(device)
    model.train()

    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = _grad_scaler(use_amp)
    masked = [
        (param, masks[name].to(param.device))
        for name, param in model.named_parameters()
        if masks and name in masks
    ]

    best_acc = 0.0
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        # Summed on the device and read once per epoch, instead of an .item() sync every step
        total_loss = torch.zeros((), device=device)
        correct = torch.zeros((), dtype=torch.long, device=device)
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()

            # Enforce masks: zero gradients on pruned positions
            for param, mask in masked:
                if param.grad is not None:
                    param.grad.mul_(mask)

            scaler.step(optimizer)
            scaler.update()

            # Re-apply masks to weights (belt and suspenders)
            with torch.no_grad():
                for param, mask in masked:
                    param.mul_(mask)

            total_loss += loss.detach() * inputs.size(0)
            correct += outputs.argmax(1).eq(targets).sum()
            total += inputs.size(0)

        scheduler.step()
        train_acc = 100.0 * correct.item() / total
        avg_loss = total_loss.item() / total

        # Validation
        val_acc = None
        if val_loader is not None:
            val_acc = evaluate(model, val_loader, device, amp=amp)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })

        if verbose and (epoch + 1) % max(1, epochs // 4) == 0:
            val_str = f" val={val_acc:.1f}%" if val_acc is not None else ""
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f} "
                  f"train={train_acc:.1f}%{val_str}")

    # Restore best checkpoint if we tracked validation
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_val_acc": best_acc if val_loader else None,
        "final_train_acc": history[-1]["train_acc"],
    }


def evaluate(
    model: nn.Module,
    loader,
    device: Optional[torch.device] = None,
    amp: bool = False,
    fp32_fallback: bool = True,
) -> float:
    """
    Top-1 accuracy (%) of a model on a data loader.

    Args:
        model: Model to evaluate (put in eval mode)
        loader: Yields (inputs, targets) batches
        device: Compute device (default: the model's)
        amp: Run forward passes under fp16 autocast (CUDA devices only)
        fp32_fallback: Under amp, re-run in fp32 any batch whose fp16
                       logits are not all finite. Activations past fp16's
                       range (65504) turn logits into inf/NaN, and argmax
                       over a NaN row picks class 0, so without this an
                       encoded model that overflows fp16 scores chance
                       level on a balanced test set
    """
    if device is None:
        device = next(model.parameters()).device
    use_amp = _amp_enabled(amp, device)
    model.eval()
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(inputs)
            if use_amp and fp32_fallback and not torch.isfinite(logits).all():
                logits = model(inputs)
            correct += logits.argmax(1).eq(targets).sum()
            total += inputs.size(0)
    return 100.0 * correct.item() / total


def _amp_enabled(amp: bool, device) -> bool:
    """fp16 autocast runs on CUDA devices only; elsewhere amp=True falls back to fp32."""
    return amp and torch.device(device).type == "cuda"


def _grad_scaler(enabled: bool):
    """torch.amp.GradScaler("cuda") from PyTorch 2.3 on, torch.cuda.amp.GradScaler before it."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


# ───────────────────────────────────────────────────────────
# INTEGRITY CHECK
# ───────────────────────────────────────────────────────────

def check(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    pattern: Optional[str] = None,
) -> Dict:
    """
    Verify pruned model integrity.

    Checks:
    1. All masks satisfy their pattern (the adjacency constraint by default)
    2. All pruned positions are actually zero
    3. Density statistics

    Args:
        model: Pruned model
        masks: Dict of pruning masks
        pattern: The pattern to validate ("phi" or "2:4"); None reads
                 it from the stats prune() attached, else "phi"

    Returns:
        dict with per-layer and aggregate results
    """
    if pattern is None:
        pattern = getattr(model, "_phi_prune_stats", {}).get("pattern", "phi")
    results = {}
    all_valid = True
    all_zeros_enforced = True

    for name, param in model.named_parameters():
        if name not in masks:
            continue

        mask = masks[name].to(param.device)
        ms = mask_stats(mask, pattern=pattern)

        # Check pruned positions are zero
        pruned_vals = param.data[~mask.bool()]
        zeros_ok = (pruned_vals.abs() < 1e-10).all().item()

        results[name] = {
            **ms,
            "zeros_enforced": zeros_ok,
        }

        if not ms["valid"]:
            all_valid = False
        if not zeros_ok:
            all_zeros_enforced = False

    results["_summary"] = {
        "all_masks_valid": all_valid,
        "all_zeros_enforced": all_zeros_enforced,
        "n_layers": len(results) - 1,
        "pattern": pattern,
    }

    return results
