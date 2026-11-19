"""export_bitstream and load_bitstream write and read with phi_prune.gaps, the one Fibonacci coder: a fixed model's
exported file has pinned bytes and loads back bit for bit, a text payload from before the packing still loads,
FibonacciEncoder holds no stream methods, and a payload with a level above the grid or bits past its last codeword is
refused."""
import hashlib
import json

import numpy as np
import pytest
import torch

from phi_prune.encoding import FibonacciEncoder
from phi_prune.export import export_bitstream, load_bitstream
from phi_prune.gaps import encode_gaps

EXPORT_SHA256 = "d974f9dce2cd589bb5f1f4487c0d9f3dfb653f3ecfb65d770393e405f6bb4dbc"  # the packed payload, ezexe/phi#77


def _mlp():
    return torch.nn.Sequential(torch.nn.Linear(16, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))


def _exported(tmp_path):
    torch.manual_seed(0)
    model = _mlp()
    enc = FibonacciEncoder(6)
    masks, scales = {}, {}
    for name, param in model.named_parameters():
        if name.endswith("weight"):
            masks[name] = (torch.rand_like(param) > 0.3).float()
            encoded, scale, _, offset = enc.encode_tensor(param.data, mask=masks[name], return_offset=True)
            param.data.copy_(encoded)
            scales[name] = (scale, offset)
    path = str(tmp_path / "model.phi")
    export_bitstream(model, masks, enc, scales, path)
    return model, masks, enc, path


def _restored(model, masks):
    restored = _mlp()
    restored.load_state_dict(model.state_dict())
    for name, param in restored.named_parameters():
        if name in masks:
            param.data[masks[name].bool()] = 0.0
    return restored


def test_encoder_holds_no_stream_methods():
    for name in ("to_bitstream", "encode_to_stream", "decode_from_stream", "parse_bitstream"):
        assert not hasattr(FibonacciEncoder, name), name


def test_exported_file_keeps_its_bytes(tmp_path):
    model, masks, _, path = _exported(tmp_path)
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == EXPORT_SHA256
    restored = _restored(model, masks)
    load_bitstream(restored, masks, path)
    for (name, before), after in zip(model.named_parameters(), restored.parameters()):
        assert torch.equal(before, after), name


def _rewrite(path, change):
    """Unpack the payload to '0'/'1' text, change it, and pack it back with payload_bits set to its new length."""
    head, rest = open(path, "rb").read().split(b"\n", 1)
    header = json.loads(head)
    bits = np.unpackbits(np.frombuffer(rest, dtype=np.uint8))[:header["payload_bits"]]
    payload = change(header, (bits + ord("0")).tobytes().decode())
    packed = np.packbits(np.frombuffer(payload.encode(), dtype=np.uint8) - ord("0")).tobytes()
    with open(path, "wb") as f:
        f.write(json.dumps({**header, "payload_bits": len(payload)}).encode() + b"\n" + packed)


def test_text_payload_from_before_the_packing_loads(tmp_path):
    """A file exported before the payload was packed, '0'/'1' text with no payload_bits, loads to the same weights."""
    model, masks, _, path = _exported(tmp_path)
    head, rest = open(path, "rb").read().split(b"\n", 1)
    header = json.loads(head)
    text = (np.unpackbits(np.frombuffer(rest, dtype=np.uint8))[:header.pop("payload_bits")] + ord("0")).tobytes()
    with open(path, "wb") as f:
        f.write(json.dumps(header).encode() + b"\n" + text)
    restored = _restored(model, masks)
    load_bitstream(restored, masks, path)
    for (name, before), after in zip(model.named_parameters(), restored.parameters()):
        assert torch.equal(before, after), name


def test_level_above_the_grid_refused(tmp_path):
    model, masks, enc, path = _exported(tmp_path)
    # The first codeword becomes the code of max_value + 2, a level one above the grid
    _rewrite(path, lambda h, p: encode_gaps([enc.max_value + 2]) + p[p.index("11") + 2:])
    with pytest.raises(ValueError):
        load_bitstream(_restored(model, masks), masks, path)


def test_bits_past_the_last_codeword_refused(tmp_path):
    model, masks, _, path = _exported(tmp_path)
    _rewrite(path, lambda h, p: p + "0101")
    with pytest.raises(ValueError):
        load_bitstream(_restored(model, masks), masks, path)
