"""
Pruning, fine-tuning, evaluation and the mask pattern check of a model.
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
    Prune a model, one mask per eligible weight, and zero the pruned weights.

    Each weight of a layer_types module, apart from those prune_head, exclude and min_dim leave
    dense, gets a φ mask, which keeps the positions along `axis` with the largest total L1 norm
    such that no two are adjacent, the (1,∞)-RLL constraint.
    pattern="2:4" keeps instead the 2 heaviest of every 4 consecutive positions.

    Args:
        model: Trained PyTorch model
        density: Not read. A φ mask keeps the heaviest set the constraint allows, at most
                 (n + 1) // 2 of n positions, and pattern="2:4" keeps 2 of every full group of 4
        axis: Dimension the pattern runs along; 0 is the output channels
        layer_types: Module types whose weights are pruned
        min_dim: A weight with fewer positions than this along `axis` stays dense
        inplace: Prune `model` itself; when False, a deep copy is pruned
        score_fn: How a position is scored; "magnitude", the one value, is the L1 norm of its slice
        prune_head: Also prune the model's last weight layer, the last Conv2d, Linear or
                    layer_types module in registration order, which is the classifier head in
                    standard architectures. It stays dense by default: along axis 0 its n positions
                    are the classes, the constraint keeps at most (n + 1) // 2 of them, and each
                    class left out gets a logit that is its bias alone
        exclude: Module names to leave dense, such as {"fc"} for a head that is not
                 registered last, or any other layer to keep whole
        pattern: "phi", no two adjacent positions kept along `axis`, or "2:4", 2 of every 4
                 consecutive positions with a trailing group shorter than 4 kept whole. Along
                 axis 0 both keep or drop whole channels, unlike NVIDIA's 2:4, which keeps 2 of
                 every 4 weights inside each row

    Returns:
        pruned_model and masks, as a pair; masks maps each pruned parameter's name to its mask.
        pruned_model._phi_prune_stats holds total_params, active_params, pruned_layers, pattern,
        density and sparsity, counted over every weight of the layer_types modules, the dense ones
        included
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

            # Zero the pruned weights
            param.data.mul_(mask)

            stats["total_params"] += param.numel()
            stats["active_params"] += mask.sum().item()
            stats["pruned_layers"] += 1

    # Over every weight of the layer_types modules, the ones left dense included
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
    Train a model with SGD and a cosine learning-rate schedule, holding the masks fixed.

    After each backward pass every masked weight's gradient is multiplied by its mask, and after
    each step the weight is, so pruned weights stay zero. With no masks every weight trains,
    which is how a dense model is fine-tuned before pruning.

    Args:
        model: Model to train
        train_loader: Yields batches of inputs and targets
        epochs: Epochs to train; the cosine schedule spans them all
        masks: Parameter name to mask, as prune returns them; None trains every weight
        lr: Learning rate at the start of the schedule
        momentum: SGD momentum
        weight_decay: SGD weight decay
        device: Compute device; the model's by default
        val_loader: Batches scored after every epoch; the weights of the best-scoring epoch
                    are restored at the end
        verbose: Print the loss and accuracies every max(1, epochs // 4) epochs
        criterion: Loss function; cross-entropy by default
        amp: On a CUDA device, run forward passes under fp16 autocast with loss scaling;
             on any other device training stays in fp32

    Returns:
        dict with history, one entry per epoch, best_val_acc, which is None without val_loader,
        and final_train_acc
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

            # Hold the masks fixed: zero the gradients of pruned weights
            for param, mask in masked:
                if param.grad is not None:
                    param.grad.mul_(mask)

            scaler.step(optimizer)
            scaler.update()

            # Multiply the weights by their masks too, so pruned weights end every step at zero
            with torch.no_grad():
                for param, mask in masked:
                    param.mul_(mask)

            total_loss += loss.detach() * inputs.size(0)
            correct += outputs.argmax(1).eq(targets).sum()
            total += inputs.size(0)

        scheduler.step()
        train_acc = 100.0 * correct.item() / total
        avg_loss = total_loss.item() / total

        # Score val_loader, keeping the weights of the best epoch
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

    # Restore the weights of the best epoch when val_loader was scored
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
    Top-1 accuracy of a model on a data loader, in percent.

    Args:
        model: Model to evaluate; it is put in eval mode
        loader: Yields batches of inputs and targets
        device: Compute device; the model's by default
        amp: On a CUDA device, run forward passes under fp16 autocast
        fp32_fallback: Under amp, run again in fp32 any batch whose fp16 logits are not all
                       finite. An activation past fp16's largest value, 65504, makes a logit
                       inf or NaN, and argmax over a row of NaN picks class 0
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
    """phi runs fp16 autocast on CUDA devices only; on any other device amp=True runs in fp32."""
    return amp and torch.device(device).type == "cuda"


def _grad_scaler(enabled: bool):
    """torch.amp.GradScaler("cuda") from PyTorch 2.3 on, torch.cuda.amp.GradScaler before it."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


# ───────────────────────────────────────────────────────────
# MASK PATTERN CHECK
# ───────────────────────────────────────────────────────────

def check(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    pattern: Optional[str] = None,
) -> Dict:
    """
    Check every mask against its pattern, and that every pruned weight is zero.

    For each masked parameter it reports mask_stats, whose "valid" reads the pattern along
    axis 0, and zeros_enforced, whether every pruned weight is within 1e-10 of zero.

    Args:
        model: Pruned model
        masks: Parameter name to mask
        pattern: The pattern to check, "phi" or "2:4"; None reads it from the stats prune
                 attached, else "phi"

    Returns:
        dict of each masked parameter's report, and "_summary" with all_masks_valid,
        all_zeros_enforced, n_layers and pattern
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

        # Every pruned weight is zero
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
