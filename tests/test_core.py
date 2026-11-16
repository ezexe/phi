"""Tests of the pruning core: the masks, quantization and the Fibonacci code, the minimal-form check, the API and the command."""

import copy
import json
import numpy as np
import torch
import torch.nn as nn
import pytest

from phi_prune.masks import (
    adjacency_dp, adjacency_dp_density, phi_mask, verify_mask, two_four_mask, verify_two_four,
)
from phi_prune.encoding import FibonacciEncoder
from phi_prune.integrity import adjacency_check, simulate_corruption
from phi_prune.api import prune, finetune, check, evaluate
from phi_prune.export import export_bitstream, load_bitstream


# ── The maximum-weight independent set ──

class TestAdjacencyDP:
    def test_no_adjacent_ones(self):
        """The mask never holds two adjacent 1s."""
        for _ in range(100):
            scores = np.random.rand(np.random.randint(1, 200))
            mask = adjacency_dp(scores)
            assert verify_mask(mask), f"Adjacent 1s found in mask for n={len(scores)}"

    def test_keeps_at_most_half_rounded_up(self):
        """Of n positions at most (n + 1) // 2 are kept."""
        for n in [1, 2, 3, 10, 50, 100, 199]:
            scores = np.ones(n)
            mask = adjacency_dp(scores)
            assert mask.sum() <= (n + 1) // 2

    def test_optimal_uniform(self):
        """On equal scores 5 of 10 positions are kept."""
        scores = np.ones(10)
        mask = adjacency_dp(scores)
        assert mask.sum() == 5  # ⌊10/2⌋ = 5

    def test_prefers_high_scores(self):
        """The heavier positions are kept."""
        scores = np.array([0.1, 10.0, 0.1, 10.0, 0.1])
        mask = adjacency_dp(scores)
        assert mask[1] == 1 and mask[3] == 1

    def test_empty(self):
        mask = adjacency_dp(np.array([]))
        assert len(mask) == 0

    def test_single(self):
        mask = adjacency_dp(np.array([5.0]))
        assert mask[0] == 1


    def test_density_target(self):
        """adjacency_dp_density keeps about the asked fraction, no two adjacent and never more than the constraint allows."""
        rng = np.random.default_rng(0)
        for density in (0.1, 0.25, 0.28, 0.38):
            scores = rng.random(512)
            mask = adjacency_dp_density(scores, density)
            assert verify_mask(mask)
            assert abs(int(mask.sum()) - round(density * 512)) <= 2
        assert adjacency_dp_density(np.ones(10), 0.9).sum() == 5  # capped by the constraint


# ── φ masks of weight tensors ──

class TestPhiMask:
    def test_conv2d_shape(self):
        w = torch.randn(16, 8, 3, 3)
        mask = phi_mask(w, axis=0)
        assert mask.shape == w.shape

    def test_linear_shape(self):
        w = torch.randn(64, 32)
        mask = phi_mask(w, axis=0)
        assert mask.shape == w.shape

    def test_broadcast_consistency(self):
        """Every weight of an output channel shares the channel's mask value."""
        w = torch.randn(16, 8, 3, 3)
        mask = phi_mask(w, axis=0)
        for i in range(16):
            vals = mask[i].unique()
            assert len(vals) == 1  # all 0 or all 1


# ── The baseline that keeps 2 of every 4 channels ──

class TestTwoFourMask:
    def test_keeps_top_two_of_every_four(self):
        w = torch.zeros(8, 2, 3, 3)
        for c, s in enumerate([1, 5, 3, 2, 9, 1, 1, 8]):
            w[c] = s
        mask = two_four_mask(w, axis=0)
        assert mask.shape == w.shape
        kept = [int(mask[c].max()) for c in range(8)]
        assert kept == [0, 1, 1, 0, 1, 0, 0, 1]
        assert verify_two_four(kept)

    def test_trailing_partial_group_kept_whole(self):
        mask = two_four_mask(torch.randn(6, 4), axis=0)
        kept = [int(mask[c].max()) for c in range(6)]
        assert sum(kept[:4]) == 2 and kept[4:] == [1, 1]

    def test_verify_two_four(self):
        assert verify_two_four([1, 1, 0, 0, 0, 1, 0, 1])
        assert not verify_two_four([1, 1, 1, 0])
        assert not verify_two_four([1, 1, 0, 0, 0, 1])  # a trailing partial group must be kept whole


# ── Quantization and the Fibonacci code ──

class TestFibonacciEncoder:
    def test_grid_size(self):
        enc = FibonacciEncoder(n_digits=8)
        assert enc.n_levels == 55  # F(10) = 55 for 8 digits

    def test_no_adjacent_ones_in_grid(self):
        """No quantization level's Zeckendorf digits hold two adjacent 1s."""
        enc = FibonacciEncoder(n_digits=8)
        for val in enc.grid:
            cw = enc.to_codeword(int(val))
            assert adjacency_check(cw), f"Adjacent 1s in the digits of quantization level {int(val)}"

    def test_encode_snaps_to_nearest_level(self):
        """A value between two quantization levels goes to the nearer one, below as well as above."""
        enc = FibonacciEncoder(n_digits=10)  # quantization levels: every integer from 0 to 143
        # 0 and 143 pin the min-max scaling to the identity, so 4.3 and 4.7 reach the levels unscaled
        t = torch.tensor([0.0, 4.3, 4.7, 143.0])
        encoded, scale, rmse = enc.encode_tensor(t)
        assert scale == 1.0
        assert encoded.tolist() == [0.0, 4.0, 5.0, 143.0]
        assert rmse == pytest.approx(0.3 / np.sqrt(2), abs=1e-6)

    def test_encode_return_offset(self):
        """return_offset adds the quantization offset, the smallest kept value, which level 0 decodes back to unchanged."""
        enc = FibonacciEncoder(n_digits=8)
        t = torch.randn(8, 4)
        mask = torch.zeros_like(t)
        mask[::2] = 1
        encoded, scale, rmse, offset = enc.encode_tensor(t, mask=mask, return_offset=True)
        assert offset == t[mask.bool()].min().item() == encoded[mask.bool()].min().item()
        plain = enc.encode_tensor(t, mask=mask)
        assert len(plain) == 3
        assert torch.equal(plain[0], encoded) and plain[1:] == (scale, rmse)
        # The degenerate cases: a constant tensor, and a mask that keeps nothing
        assert enc.encode_tensor(torch.full((4,), 0.25), return_offset=True)[3] == 0.25
        assert enc.encode_tensor(t, mask=torch.zeros_like(t), return_offset=True)[3] == 0.0

    def test_encode_per_channel(self):
        """axis quantizes each slice as encode_tensor would alone, with a scale and quantization offset that broadcast."""
        enc = FibonacciEncoder(n_digits=8)
        torch.manual_seed(0)
        t = torch.randn(6, 5, 3)
        t[0, 0, 0] = 40.0  # one outlier would stretch a per-tensor range
        mask = torch.ones_like(t)
        mask[3] = 0  # a fully pruned channel
        encoded, scale, rmse, offset = enc.encode_tensor(t, mask=mask, return_offset=True, axis=0)
        assert scale.shape == offset.shape == (6, 1, 1)
        for c in range(6):
            alone, s, _, o = enc.encode_tensor(t[c], mask=mask[c], return_offset=True)
            assert torch.equal(encoded[c], alone) and scale[c].item() == s and offset[c].item() == o
        assert torch.equal(encoded[3], t[3]) and (scale[3].item(), offset[3].item()) == (1.0, 0.0)
        # Every kept channel's minimum lands on quantization level 0, and levels() inverts the scale-back
        levels = enc.levels(encoded, scale, offset, mask).reshape(5, -1)
        assert (levels.min(axis=1) == 0).all() and levels.max() <= enc.max_value
        # Per channel, the outlier leaves the other channels' levels fine
        assert rmse < enc.encode_tensor(t, mask=mask)[2] / 2
        # Any axis, negative ones included
        assert enc.encode_tensor(t, axis=-1, return_offset=True)[1].shape == (1, 1, 3)

    def test_stream_roundtrip(self):
        """Quantization levels written as a Fibonacci code read back unchanged."""
        enc = FibonacciEncoder(n_digits=8)
        values = [0, 1, 5, 12, 33]
        stream = enc.encode_to_stream(values)
        recovered = enc.decode_from_stream(stream)
        assert recovered == values

    def test_stream_all_grid_values(self):
        """Every quantization level reads back from the Fibonacci code."""
        enc = FibonacciEncoder(n_digits=8)
        values = [int(v) for v in enc.grid]
        stream = enc.encode_to_stream(values)
        recovered = enc.decode_from_stream(stream)
        assert recovered == values


# ── The minimal-form check ──

class TestIntegrity:
    def test_corrupted_detected(self):
        """A digit flipped to 1 beside a 1 is caught."""
        cw = [1, 0, 1, 0, 1, 0, 0, 0]  # no two adjacent 1s
        assert adjacency_check(cw)
        cw_corrupt = [1, 1, 1, 0, 1, 0, 0, 0]  # digit 1 flipped
        assert not adjacency_check(cw_corrupt)

    def test_simulate_corruption_matches_exact_rate(self):
        """The simulated share of caught flips agrees with counting every single-digit flip."""
        import random

        enc = FibonacciEncoder(n_digits=8)
        tensor = torch.tensor(enc.grid, dtype=torch.float64)  # one weight per quantization level
        mask = torch.ones_like(tensor)

        rates = []
        for val in enc.grid:
            cw = enc.to_codeword(int(val))
            caught = 0
            for i in range(len(cw)):
                flipped = list(cw)
                flipped[i] = 1 - flipped[i]
                caught += not adjacency_check(flipped)
            rates.append(caught / len(cw))
        exact = float(np.mean(rates))

        random.seed(0)
        n = 20_000
        detected, total = simulate_corruption(tensor, mask, enc, 1.0, 0.0, n_flips=n)
        assert total == n
        sigma = np.sqrt(exact * (1 - exact) / n)
        assert abs(detected / n - exact) < 4 * sigma
        assert 0.40 < exact < 0.42  # 0.4091 over every 8-digit quantization level, as results/prune/measure_detection.txt holds

    def test_per_channel_passes_integrity_check(self):
        """cassini_check reads per-channel scale and quantization offset arrays, and passes every quantized weight."""
        from phi_prune.integrity import cassini_check

        pruned, masks, enc, scales = encoded_mlp(axis=0)
        report = cassini_check(pruned, masks, enc, scales)["_aggregate"]
        assert report["total"] == sum(int(m.sum()) for m in masks.values())
        assert report["failed"] == 0


# ── prune, finetune, evaluate and check ──

class TestAPI:
    def _make_model(self):
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

    def test_prune_returns_masks(self):
        model = self._make_model()
        pruned, masks = prune(model)
        assert len(masks) > 0
        for name, mask in masks.items():
            assert mask.shape == dict(pruned.named_parameters())[name].shape

    def test_pruned_weights_are_zero(self):
        model = self._make_model()
        pruned, masks = prune(model)
        for name, param in pruned.named_parameters():
            if name in masks:
                pruned_vals = param.data[~masks[name].bool()]
                assert (pruned_vals.abs() < 1e-10).all()

    def test_check_passes(self):
        model = self._make_model()
        pruned, masks = prune(model)
        report = check(pruned, masks)
        assert report["_summary"]["all_masks_valid"]
        assert report["_summary"]["all_zeros_enforced"]

    def test_density_under_half(self):
        model = self._make_model()
        pruned, masks = prune(model, prune_head=True)  # every eligible layer pruned, head included
        assert pruned._phi_prune_stats["density"] <= 0.51  # every width here is even, so no mask keeps more than half

    def test_head_left_dense_by_default(self):
        """The last eligible layer, the 10-class head here, keeps every row."""
        model = self._make_model()
        pruned, masks = prune(model)
        assert "6.weight" not in masks
        assert torch.equal(pruned[6].weight, model[6].weight)

    def test_prune_head_opt_in(self):
        model = self._make_model()
        pruned, masks = prune(model, prune_head=True)
        assert "6.weight" in masks

    def test_conv_only_layer_types_prunes_every_conv(self):
        """layer_types=(nn.Conv2d,) already leaves the Linear head out; every conv still gets pruned."""
        model = self._make_model()
        pruned, masks = prune(model, layer_types=(nn.Conv2d,))
        assert set(masks) == {"0.weight", "2.weight"}

    def test_exclude_keeps_named_layers_dense(self):
        model = self._make_model()
        pruned, masks = prune(model, exclude={"0"})
        assert "0.weight" not in masks
        assert "2.weight" in masks

    def test_two_four_pattern(self):
        """pattern="2:4" keeps 2 of every 4 output channels, and check() validates that pattern."""
        model = self._make_model()
        pruned, masks = prune(model, pattern="2:4", layer_types=(nn.Conv2d,))  # density counts convs only
        assert set(masks) == {"0.weight", "2.weight"}
        for mask in masks.values():
            assert verify_two_four(mask.flatten(1).amax(1).tolist())  # one value per output channel
        assert pruned._phi_prune_stats["pattern"] == "2:4"
        assert pruned._phi_prune_stats["density"] == pytest.approx(0.5)  # 16 and 32 channels: whole groups
        report = check(pruned, masks)
        assert report["_summary"]["pattern"] == "2:4"
        assert report["_summary"]["all_masks_valid"]
        assert report["_summary"]["all_zeros_enforced"]

    def test_unknown_pattern_rejected(self):
        with pytest.raises(ValueError, match="Unknown pattern"):
            prune(self._make_model(), pattern="3:4")

    def test_finetune_keeps_pruned_weights_zero(self):
        """amp=True falls back to fp32 off CUDA, so both settings run on any machine."""
        model = self._make_model()
        pruned, masks = prune(model)
        batches = [(torch.randn(4, 3, 8, 8), torch.randint(0, 10, (4,))) for _ in range(2)]
        for amp in (False, True):
            result = finetune(pruned, batches, epochs=1, masks=masks, val_loader=batches,
                              verbose=False, amp=amp)
            assert set(result) == {"history", "best_val_acc", "final_train_acc"}
            assert check(pruned, masks)["_summary"]["all_zeros_enforced"]

    @pytest.mark.filterwarnings("ignore:CUDA is not available")  # amp forced on without CUDA
    def test_evaluate_reruns_non_finite_amp_batches_in_fp32(self, monkeypatch):
        """A batch whose fp16 logits overflow is run again in fp32, not scored as class 0."""
        import phi_prune.api as api

        class OverflowsOnce(nn.Module):
            """Right on every call except the first, which returns NaN like an fp16 overflow."""
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.zeros(1))
                self.calls = 0

            def forward(self, x):
                self.calls += 1
                logits = torch.nn.functional.one_hot(x.long(), 10).float()
                return torch.full_like(logits, float("nan")) if self.calls == 1 else logits

        targets = torch.tensor([3, 7, 7, 9])
        batches = [(targets, targets)]  # the model echoes its input as the predicted class
        monkeypatch.setattr(api, "_amp_enabled", lambda amp, device: amp)

        assert evaluate(OverflowsOnce(), batches, amp=True) == 100.0
        assert evaluate(OverflowsOnce(), batches, amp=True, fp32_fallback=False) == 0.0
        model = OverflowsOnce()
        evaluate(model, batches, amp=False)
        assert model.calls == 1  # no re-run without amp


# ── The Fibonacci-coded bitstream ──

def encoded_mlp(axis=None):
    """A pruned MLP with its masked layers quantized in place, and the scale and quantization offset of each."""
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 10))
    pruned, masks = prune(model)
    enc = FibonacciEncoder(n_digits=8)
    scales = {}
    for name, param in pruned.named_parameters():
        if name in masks:
            encoded, scale, _, offset = enc.encode_tensor(param.data, mask=masks[name],
                                                          return_offset=True, axis=axis)
            param.data.copy_(encoded)
            scales[name] = (scale, offset)
    return pruned, masks, enc, scales


class TestExport:
    def test_bitstream_roundtrip(self, tmp_path):
        """The payload parses back to the quantization levels, level 0, each layer's minimum, included."""
        pruned, masks, enc, scales = encoded_mlp()
        expected = []
        for name, param in pruned.named_parameters():
            if name in masks:
                scale, offset = scales[name]
                kept = param.data[masks[name].bool()].double()
                levels = ((kept - offset) * scale).round().long().tolist()
                assert min(levels) == 0
                expected.extend(levels)

        path = tmp_path / "model.phi"
        stats = export_bitstream(pruned, masks, enc, scales, str(path))
        assert b"\r" not in path.read_bytes()  # the header line ends in "\n" on every platform
        header_line, payload = path.read_text().split("\n", 1)
        header = json.loads(header_line)
        assert sum(layer["n_active"] for layer in header["layers"]) == stats["n_weights"] == len(expected)

        # codeword_digits and level_bias give each codeword's width and what it adds to its quantization level
        width = header["encoder"].get("codeword_digits", header["encoder"]["n_digits"])
        bias = header["encoder"].get("level_bias", 0)
        fibs = FibonacciEncoder(n_digits=width).fibs
        recovered = [sum(f * d for f, d in zip(fibs, cw)) - bias
                     for cw in FibonacciEncoder.parse_bitstream(payload, width)]
        assert recovered == expected

    @pytest.mark.parametrize("axis", [None, 0])
    def test_load_bitstream_restores_encoded_weights(self, tmp_path, axis):
        """load_bitstream writes the de-quantized weights back into the kept positions, bit for bit."""
        pruned, masks, enc, scales = encoded_mlp(axis)
        path = str(tmp_path / "model.phi")
        export_bitstream(pruned, masks, enc, scales, path)

        restored = copy.deepcopy(pruned)
        for name, param in restored.named_parameters():
            if name in masks:
                param.data[masks[name].bool()] = 0.0
        header = load_bitstream(restored, masks, path)
        assert [layer["name"] for layer in header["layers"]] == list(masks)
        for (name, before), after in zip(pruned.named_parameters(), restored.parameters()):
            assert torch.equal(before, after), name

    def test_load_bitstream_refusals(self, tmp_path):
        """load_bitstream refuses a header without codeword_digits and level_bias, a short payload or a mask that does not fit, and writes nothing."""
        pruned, masks, enc, scales = encoded_mlp()
        path = tmp_path / "model.phi"
        export_bitstream(pruned, masks, enc, scales, str(path))
        header_line, payload = path.read_text().split("\n", 1)
        old = json.loads(header_line)
        del old["encoder"]["codeword_digits"], old["encoder"]["level_bias"]
        last = list(masks)[-1]
        misfit = {**masks, last: torch.ones_like(masks[last])}
        refusals = [
            (json.dumps(old) + "\n" + payload, masks, "level_bias"),
            (header_line + "\n" + payload[:-2], masks, "payload holds"),  # the last codeword cut off
            (header_line + "\n" + payload, misfit, "does not match"),
        ]

        target = copy.deepcopy(pruned)
        for name, param in target.named_parameters():
            if name in masks:
                param.data[masks[name].bool()] = 0.0
        before = copy.deepcopy(target.state_dict())
        for text, file_masks, message in refusals:
            path.write_text(text)
            with pytest.raises(ValueError, match=message):
                load_bitstream(target, file_masks, str(path))
        assert all(torch.equal(before[k], v) for k, v in target.state_dict().items())


# ── The phi-prune command ──

class TestCLI:
    @pytest.mark.parametrize("saved_pattern", ["phi", "zeckendorf", None])
    def test_cli_check_reads_phi_zeckendorf_and_no_pattern(self, tmp_path, saved_pattern):
        """phi-prune check reads the pattern "phi", its other name "zeckendorf", and a checkpoint with no pattern key."""
        import argparse
        from phi_prune.cli import cmd_check
        from phi_prune.export import save_checkpoint

        pruned, masks, _, _ = encoded_mlp()
        path = str(tmp_path / "model.pt")
        save_checkpoint(pruned, masks, path)
        data = torch.load(path, weights_only=False)
        assert data["pattern"] == "phi" and data["format"] == "phi-prune"
        if saved_pattern is None:
            del data["pattern"]
        else:
            data["pattern"] = saved_pattern
        torch.save(data, path)
        assert cmd_check(argparse.Namespace(model=path)) == 0
