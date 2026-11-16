"""Single-bit flips of a saved .phid file, header or payload, text or packed, are refused by load_delta with ValueError."""
import io
import json
import random

import pytest
import torch

from phi_prune.delta import decode_delta, encode_delta, load_delta, save_delta


def _delta():
    torch.manual_seed(0)
    return encode_delta({"w": torch.zeros(8, 8)}, {"w": torch.randn(8, 8)}, keep=0.25)


@pytest.mark.parametrize("packed", [False, True])
def test_every_single_bit_flip_refused(tmp_path, monkeypatch, packed):
    """Every bit of the check line and the header, and 256 payload bits spread by a fixed seed, read from memory."""
    header, payload = _delta()
    path = save_delta(str(tmp_path / "d.phid"), header, payload, packed=packed)
    data = open(path, "rb").read()
    head_bits = (data.index(b"\n", data.index(b"\n") + 1) + 1) * 8
    payload_bits = range(head_bits, len(data) * 8)
    bits = list(range(head_bits)) + sorted(random.Random(45).sample(payload_bits, min(256, len(payload_bits))))
    current = [data]
    monkeypatch.setattr("phi_prune.delta.open", lambda p, mode="r": io.BytesIO(current[0]), raising=False)
    accepted = []
    for i in bits:
        flipped = bytearray(data)
        flipped[i // 8] ^= 1 << (i % 8)
        current[0] = bytes(flipped)
        try:
            decode_delta(*load_delta(path))
        except ValueError:
            continue
        except Exception:  # noqa: BLE001 - a flip refused by any other error counts as not refused cleanly
            pass
        accepted.append(i)
    assert not accepted, f"{len(accepted)} of {len(bits)} single-bit flips were not refused with ValueError"


def test_header_field_name_flip_is_value_error(tmp_path):
    header, payload = _delta()
    path = save_delta(str(tmp_path / "d.phid"), header, payload, packed=True)
    data = open(path, "rb").read()
    at = data.index(b'"n_kept"') + 1
    flipped = bytearray(data)
    flipped[at] ^= 1
    open(path, "wb").write(bytes(flipped))
    with pytest.raises(ValueError):
        decode_delta(*load_delta(path))


def test_file_written_before_the_check_still_loads(tmp_path):
    header, payload = _delta()
    path = str(tmp_path / "d.phid")
    open(path, "wb").write((json.dumps(header) + "\n" + payload).encode("ascii"))
    assert load_delta(path) == (header, payload)
