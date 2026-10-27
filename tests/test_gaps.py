"""Tests for phi_prune.gaps and phi_prune.delta: Fibonacci gap codes and sparse checkpoint deltas."""

import os

import numpy as np
import pytest
import torch

from phi_prune.delta import apply_delta, decode_delta, delta_stats, encode_delta, load_delta, save_delta
from phi_prune.gaps import (
    decode_gaps, decode_positions, encode_gaps, encode_positions, fib_code, fib_code_length, gap_bits,
    pooled_gaps, positions_to_gaps, read_codes, top_positions,
)


class TestFibCode:
    def test_known_codes(self):
        assert [fib_code(n) for n in range(1, 9)] == ["11", "011", "0011", "1011", "00011", "10011", "01011", "000011"]

    def test_one_closing_pair(self):
        for n in range(1, 2000):
            code = fib_code(n)
            assert code.endswith("11") and code.find("11") == len(code) - 2
            assert fib_code_length(n) == len(code)

    def test_zero_refused(self):
        with pytest.raises(ValueError):
            fib_code(0)


class TestGaps:
    def test_round_trip(self):
        gaps = list(range(1, 300)) + [1, 1, 1, 10_000, 1]
        assert decode_gaps(encode_gaps(gaps)) == gaps

    def test_positions_round_trip(self):
        rng = np.random.default_rng(0)
        pos = np.sort(rng.choice(100_000, 500, replace=False))
        assert np.array_equal(decode_positions(encode_positions(pos)), pos)
        assert positions_to_gaps([0, 1, 5])[0] == 1

    def test_unsorted_positions_refused(self):
        with pytest.raises(ValueError):
            positions_to_gaps([3, 3])

    def test_truncated_stream_refused(self):
        with pytest.raises(ValueError):
            decode_gaps(encode_gaps([5, 7])[:-1])

    def test_read_codes_count_and_end(self):
        bits = encode_gaps([4, 2, 9])
        values, end = read_codes(bits, 0, 2)
        assert values == [4, 2] and end == len(fib_code(4)) + len(fib_code(2))
        with pytest.raises(ValueError):
            read_codes(bits, 0, 4)

    def test_gap_bits_matches_stream(self):
        torch.manual_seed(0)
        t = torch.randn(64, 64)
        gaps, n = pooled_gaps([t], 0.1)
        bits = gap_bits(gaps, n)
        assert bits["fibonacci"] == pytest.approx(len(encode_gaps(gaps)) / len(gaps))
        assert bits["fixed-width"] == 12

    def test_top_positions(self):
        t = torch.tensor([0.0, -5.0, 1.0, 4.0, 0.1])
        assert top_positions(t, 0.4).tolist() == [1, 3]


class TestDelta:
    def states(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        base = {"a.weight": torch.randn(32, 16, generator=g), "a.bias": torch.randn(32, generator=g),
                "steps": torch.tensor(3)}
        new = {k: (v + 0.01 * torch.randn(v.shape, generator=g) if v.is_floating_point() else v + 1)
               for k, v in base.items()}
        return base, new

    def test_round_trip_within_a_level(self):
        base, new = self.states()
        header, payload = encode_delta(base, new, keep=0.1, n_digits=6)
        assert header["skipped"] == ["steps"]
        for t in header["tensors"]:
            pos, vals = decode_delta(header, payload)[t["name"]]
            true = (new[t["name"]] - base[t["name"]]).flatten()[torch.from_numpy(pos)].numpy()
            assert np.abs(vals - true).max() <= t["scale"] / 2 + 1e-9
            assert np.array_equal(pos, top_positions(new[t["name"]] - base[t["name"]], 0.1))

    def test_apply_moves_toward_new(self, tmp_path):
        base, new = self.states()
        header, payload = encode_delta(base, new, keep=1.0, n_digits=8)
        header, payload = load_delta(save_delta(str(tmp_path / "d.phid"), header, payload))
        state = {k: v.clone() for k, v in base.items()}
        apply_delta(state, header, payload)
        for k in ("a.weight", "a.bias"):
            before = (new[k] - base[k]).abs().max()
            assert (new[k] - state[k]).abs().max() < before / 10
        assert delta_stats(header, payload)["kept"] == 32 * 16 + 32

    def test_corrupt_payload_refused(self):
        base, new = self.states()
        header, payload = encode_delta(base, new, keep=0.1)
        with pytest.raises(ValueError):
            decode_delta(header, payload[:-3])
        with pytest.raises(ValueError):
            decode_delta(header, payload + "011")

    def test_shape_mismatch_refused_before_writing(self):
        base, new = self.states()
        header, payload = encode_delta(base, new, keep=0.1)
        state = {"a.weight": base["a.weight"].clone(), "a.bias": torch.zeros(31)}
        with pytest.raises(ValueError):
            apply_delta(state, header, payload)
        assert torch.equal(state["a.weight"], base["a.weight"])


class TestPluginDelta:
    """A round delta written by the vlds plugin's scripts/pick_model.py (standard library only) reads back here."""

    def test_reads_the_plugins_round(self):
        import json
        from pathlib import Path

        here = Path(__file__).parent / "fixtures"
        header, payload = load_delta(str(here / "pick-model.r01.phid"))
        expect = json.loads((here / "pick-model.r01.json").read_text())
        old = {int(k): v for k, v in expect["old"].items()}
        new = {int(k): v for k, v in expect["new"].items()}
        pos, vals = decode_delta(header, payload)["w"]
        changed = sorted(k for k in set(old) | set(new) if new.get(k, 0) != old.get(k, 0))
        assert pos.tolist() == changed
        scale = header["tensors"][0]["scale"]
        for k, v in zip(pos, vals):
            assert abs(old.get(int(k), 0) + v - new.get(int(k), 0)) <= scale / 2 + 1e-12


class TestLiveRoundFormat:
    """The options a live model's rounds ship with: keep over the entries that moved, and packed bits."""

    def test_changed_only_skips_the_still_entries(self):
        base = {"emb": torch.zeros(100, 4), "head": torch.zeros(3)}
        new = {"emb": base["emb"].clone(), "head": base["head"].clone()}
        new["emb"][7] = torch.tensor([1.0, -2.0, 3.0, -4.0])          # one row of a sparse table moved
        header, payload = encode_delta(base, new, keep=0.5, changed_only=True)
        assert header["unchanged"] == ["head"]
        (t,) = header["tensors"]
        assert t["n_kept"] == 2                                        # half of the 4 that moved, not of 400
        pos, _vals = decode_delta(header, payload)["emb"]
        assert set(pos) == {30, 31}                                    # -4.0 and 3.0, the largest moves

    def test_packed_round_trip(self, tmp_path):
        torch.manual_seed(0)
        base = {"w": torch.randn(64, 64)}
        new = {"w": base["w"] + 0.1 * torch.randn(64, 64)}
        header, payload = encode_delta(base, new, keep=0.1)
        text = save_delta(str(tmp_path / "t.phid"), header, payload)
        packed = save_delta(str(tmp_path / "p.phid"), header, payload, packed=True)
        assert os.path.getsize(packed) * 4 < os.path.getsize(text)
        h2, p2 = load_delta(packed)
        assert h2 == header and p2 == payload
        assert load_delta(text) == (header, payload)

    def test_packed_truncation_refused(self, tmp_path):
        header, payload = encode_delta({"w": torch.zeros(8)}, {"w": torch.arange(8.0)}, keep=1.0)
        path = save_delta(str(tmp_path / "p.phid"), header, payload, packed=True)
        data = open(path, "rb").read()
        open(path, "wb").write(data[:-1])
        with pytest.raises(ValueError):
            load_delta(path)
