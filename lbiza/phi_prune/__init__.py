"""
phi-prune: channel pruning under the (1,∞)-RLL constraint, which never keeps two adjacent output channels of a layer.

    from phi_prune import prune, finetune, check
    from phi_prune.export import export_onnx

    # Prune: each eligible layer keeps the output channels with the largest total L1 norm, no two adjacent
    pruned_model, masks = prune(model)

    # Fine-tune with the masks held fixed, so pruned weights stay zero
    finetune(pruned_model, train_loader, epochs=40, masks=masks)

    # Check every mask against its pattern, and that every pruned weight is zero
    report = check(pruned_model, masks)

    # Export to ONNX through torch.onnx.export, with the pruned weights as zeros in the graph
    export_onnx(pruned_model, "model_sparse.onnx")
"""

__version__ = "0.3.1"

from phi_prune.api import prune, finetune, check, evaluate
from phi_prune.encoding import FibonacciEncoder
from phi_prune.integrity import cassini_check
from phi_prune.masks import phi_mask, verify_mask

__all__ = [
    "prune",
    "finetune",
    "check",
    "evaluate",
    "FibonacciEncoder",
    "cassini_check",
    "phi_mask",
    "verify_mask",
]
