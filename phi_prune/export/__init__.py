"""
Save a pruned model, export it to ONNX, and write its quantized weights as a Fibonacci-coded bitstream
and read them back.

save_checkpoint writes a PyTorch checkpoint holding the state dict, the masks, the pattern, the format
and version, mask_stats of each mask and, when given, the encoder's settings, the scales and metadata.
export_onnx writes the model to ONNX through torch.onnx.export, with the pruned weights as zeros in the
graph. export_bitstream writes every kept weight's level, and load_bitstream reads them back into a model.
"""

import torch
import numpy as np
import json
from typing import Dict, Optional

from phi_prune.encoding import FibonacciEncoder
from phi_prune.masks import mask_stats


def save_checkpoint(
    model: torch.nn.Module,
    masks: Dict[str, torch.Tensor],
    path: str,
    encoder: Optional[FibonacciEncoder] = None,
    scales: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
):
    """
    Save a pruned model as a phi-prune checkpoint.

    The file holds the state dict, the masks, the pattern, the format "phi-prune" and the version,
    mask_stats of each mask read along axis 0 and, when given, the encoder's settings, the scales and
    metadata. Load it with
    data = torch.load(path, weights_only=False); model.load_state_dict(data['state_dict']).
    weights_only=False is needed when scales hold the per-channel arrays of encode_tensor(axis=...),
    which are numpy arrays, and is safe only for a file you trust.
    """
    pattern = getattr(model, "_phi_prune_stats", {}).get("pattern", "phi")
    checkpoint = {
        "state_dict": model.state_dict(),
        "masks": masks,
        "pattern": pattern,
        "format": "phi-prune",
        "version": "0.3.0",
    }

    if encoder is not None:
        checkpoint["encoding"] = {
            "n_digits": encoder.n_digits,
            "n_levels": encoder.n_levels,
            "max_value": encoder.max_value,
        }

    if scales is not None:
        # Each layer's scale and offset, as encode_tensor returned them
        checkpoint["scales"] = {
            name: {"scale": s, "offset": o} for name, (s, o) in scales.items()
        }

    if metadata is not None:
        checkpoint["metadata"] = metadata

    # mask_stats of each mask, read along axis 0
    stats = {}
    for name, mask in masks.items():
        stats[name] = mask_stats(mask, pattern=pattern)
    checkpoint["prune_stats"] = stats

    torch.save(checkpoint, path)
    return path


def export_onnx(
    model: torch.nn.Module,
    path: str,
    input_shape: tuple = (1, 3, 32, 32),
    opset: int = 17,
):
    """
    Export a pruned model to ONNX through torch.onnx.export.

    The pruned weights stay in the graph as zeros, so the exported graph is as dense as the model.
    The model is traced on a random input of input_shape at the given opset, with its first axis
    named batch and dynamic through dynamic_axes. ImportError is raised when onnx is not installed.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError("pip install onnx onnxruntime")

    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(*input_shape, device=device)

    torch.onnx.export(
        model, dummy, path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    return path


def export_bitstream(
    model: torch.nn.Module,
    masks: Dict[str, torch.Tensor],
    encoder: FibonacciEncoder,
    scales: Dict,
    path: str,
):
    """
    Write the levels of a model's kept weights as a Fibonacci-coded bitstream.

    The file holds:
    1. A JSON line: each layer's name, shape, kept count, scale and offset, and the encoder's
       n_digits, codeword_digits and level_bias
    2. The payload: the Fibonacci code of every kept weight's level, in parameter order, written
       as the characters 0 and 1, one byte per bit. Each codeword is level + level_bias in up to
       codeword_digits digits, as encode_to_stream writes it, so level 0 still ends in 11

    A reader splits the payload at each 11 with no shapes, takes level_bias off each value to get
    the level, and FibonacciEncoder(n_digits).decode_from_stream(payload) does both. Level l of a
    layer is the weight l / scale + offset, and the layers' kept counts split the levels in
    payload order.

    scales maps each parameter name to the scale and offset encode_tensor returns with
    return_offset=True: floats, or with axis set, per-slice arrays, which the header stores
    flattened with their shape as scale_shape.

        encoded, scale, _, offset = encoder.encode_tensor(
            param.data, mask=mask, return_offset=True)
        param.data.copy_(encoded)
        scales[name] = (scale, offset)

    load_bitstream reads the file back into a model. The returned dict counts the stream in bits,
    and the file holds one byte for each.
    """
    # encode_to_stream's layout: level + 1 in n_digits + 1 digits, so level 0 still ends in 11
    header = {
        "layers": [],
        "encoder": {
            "n_digits": encoder.n_digits,
            "codeword_digits": encoder.n_digits + 1,
            "level_bias": 1,
        },
    }
    all_levels = []

    for name, param in model.named_parameters():
        if name not in masks or name not in scales:
            continue

        scale, offset = scales[name]
        layer_levels = encoder.levels(param.data, scale, offset, masks[name]).tolist()

        layer = {"name": name, "shape": list(param.shape), "n_active": len(layer_levels)}
        if np.ndim(scale) == 0:
            layer.update(scale=float(scale), offset=float(offset))
        else:  # per-slice pairs from encode_tensor(axis=...), flattened, with their shape
            layer.update(
                scale=np.asarray(scale, dtype=np.float64).ravel().tolist(),
                offset=np.asarray(offset, dtype=np.float64).ravel().tolist(),
                scale_shape=list(np.shape(scale)),
            )
        header["layers"].append(layer)
        all_levels.extend(layer_levels)

    bitstream = encoder.encode_to_stream(all_levels)

    # Write header + payload, with a "\n" newline on every platform
    with open(path, "w", newline="\n") as f:
        f.write(json.dumps(header) + "\n")
        f.write(bitstream)

    bits_per_weight = len(bitstream) / len(all_levels) if all_levels else 0
    return {
        "path": path,
        "n_weights": len(all_levels),
        "bitstream_length": len(bitstream),
        "bits_per_weight": bits_per_weight,
        "header_bytes": len(json.dumps(header)),
    }


def load_bitstream(
    model: torch.nn.Module,
    masks: Dict[str, torch.Tensor],
    path: str,
) -> Dict:
    """
    Read a bitstream that export_bitstream wrote back into a model.

    Decodes the payload's levels and writes each layer's weights, level / scale + offset, into its
    kept positions in place; the other positions keep their values. The header holds each layer's
    shape, kept count, scale and offset but not its mask, so pass the masks the export used, which
    save_checkpoint stores.

    ValueError is raised before any weight is written for a header whose codeword_digits and
    level_bias are not n_digits + 1 and 1, for a payload whose count of levels differs from the
    header's, and for a layer whose shape, kept count or scale shape does not fit. A layer missing from the model or from masks
    raises KeyError.

    Returns:
        The file's JSON header
    """
    with open(path) as f:
        header = json.loads(f.readline())
        payload = f.read().strip()

    layout = header["encoder"]
    n_digits = layout["n_digits"]
    if (layout.get("codeword_digits"), layout.get("level_bias")) != (n_digits + 1, 1):
        raise ValueError(
            f"{path}: header lacks codeword_digits {n_digits + 1} / level_bias 1; files exported "
            "before those fields cannot be parsed past level 0, so export again"
        )
    levels = FibonacciEncoder(n_digits).decode_from_stream(payload)
    total = sum(layer["n_active"] for layer in header["layers"])
    if total != len(levels):
        raise ValueError(f"{path}: the payload holds {len(levels)} levels, the header counts {total}")

    # Check every layer before writing any, so a refused file leaves the model as it was
    params = dict(model.named_parameters())
    targets = []
    for layer in header["layers"]:
        param = params[layer["name"]]
        keep = masks[layer["name"]].bool().to(param.device)
        if list(param.shape) != layer["shape"] or int(keep.sum()) != layer["n_active"]:
            raise ValueError(f"{path}: layer {layer['name']} does not match the model's shape or mask")
        scale = np.asarray(layer["scale"], dtype=np.float64)
        offset = np.asarray(layer["offset"], dtype=np.float64)
        if "scale_shape" in layer:  # per-slice pairs: one value per weight, in mask order
            try:
                keep_np = keep.cpu().numpy()
                scale = np.broadcast_to(scale.reshape(layer["scale_shape"]), param.shape)[keep_np]
                offset = np.broadcast_to(offset.reshape(layer["scale_shape"]), param.shape)[keep_np]
            except ValueError:
                raise ValueError(f"{path}: layer {layer['name']}'s scale_shape does not fit its shape") from None
        targets.append((param, keep, layer, scale, offset))

    start = 0
    for param, keep, layer, scale, offset in targets:
        n_active = layer["n_active"]
        # encode_tensor's scale-back arithmetic, so encoded weights come back bit for bit
        layer_levels = np.asarray(levels[start:start + n_active], dtype=np.float64)
        decoded = layer_levels / scale + offset
        param.data[keep] = torch.tensor(decoded, dtype=param.dtype, device=param.device)
        start += n_active
    return header
