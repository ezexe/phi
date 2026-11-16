"""export_bitstream and load_bitstream write and read with phi_prune.gaps, the one Fibonacci coder: a fixed model's
exported file keeps its bytes, FibonacciEncoder holds no stream methods, and a payload with a level above the grid or
bits past its last codeword is refused."""
import hashlib
import json

import pytest
import torch

from phi_prune.encoding import FibonacciEncoder
from phi_prune.export import export_bitstream, load_bitstream
from phi_prune.gaps import encode_gaps

EXPORT_SHA256 = "2b22346e10bbfe5f65694bd227eaf903d5fccd77691bd4c55c21f4adb81edc9d"


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
    head, payload = open(path).read().split("\n", 1)
    with open(path, "w", newline="\n") as f:
        f.write(head + "\n" + change(json.loads(head), payload))


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
