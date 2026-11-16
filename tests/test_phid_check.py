"""Every single-bit flip of a saved .phid file, header or payload, text or packed, is refused by load_delta with ValueError."""
import json

import pytest
import torch

from phi_prune.delta import decode_delta, encode_delta, load_delta, save_delta


def _delta():
    torch.manual_seed(0)
    return encode_delta({"w": torch.zeros(8, 8)}, {"w": torch.randn(8, 8)}, keep=0.25)


@pytest.mark.parametrize("packed", [False, True])
def test_every_single_bit_flip_refused(tmp_path, packed):
    header, payload = _delta()
    path = save_delta(str(tmp_path / "d.phid"), header, payload, packed=packed)
    data = open(path, "rb").read()
    accepted = []
    for i in range(len(data) * 8):
        flipped = bytearray(data)
        flipped[i // 8] ^= 1 << (i % 8)
        open(path, "wb").write(bytes(flipped))
        try:
            decode_delta(*load_delta(path))
        except ValueError:
            continue
        except Exception:  # noqa: BLE001 - a flip refused by any other error counts as not refused cleanly
            pass
        accepted.append(i)
    assert not accepted, f"{len(accepted)} of {len(data) * 8} single-bit flips were not refused with ValueError"


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
