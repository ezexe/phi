"""Tests of the pruning core: the masks, quantization and the Fibonacci code, the minimal-form check, the API and the command."""

import argparse
import contextlib
import copy
import io
import json
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import pytest

from phi_prune.masks import (
    adjacency_dp, adjacency_dp_density, mask_stats, phi_mask, verify_mask, two_four_mask, verify_two_four,
)
from phi_prune.encoding import FibonacciEncoder
from phi_prune.gaps import decode_gaps, encode_gaps
from phi_prune.integrity import adjacency_check, simulate_corruption
from phi_prune.api import prune, finetune, check, evaluate
from phi_prune.export import export_bitstream, load_bitstream


@pytest.fixture(autouse=True)
def _seeded():
    """Seed random, numpy and torch at 0 before each test."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def test_each_test_starts_seeded():
    """Every test of this file starts from seed 0 in random, numpy and torch, so a failure repeats."""
    assert torch.initial_seed() == 0
    assert np.array_equal(np.random.get_state()[1], np.random.RandomState(0).get_state()[1])
    assert random.getstate() == random.Random(0).getstate()


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

    def test_phi_mask_on_a_1d_tensor(self):
        """A 1-D tensor is scored by the absolute value of each entry, as two_four_mask scores it."""
        w = torch.tensor([1.0, 5.0, 1.0, 5.0, 1.0, 5.0])
        assert phi_mask(w).tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    def test_phi_mask_on_a_parameter(self):
        """A layer's own weight, which requires grad, gets the mask of its data, and the mask needs no grad."""
        layer = nn.Linear(16, 32)
        mask = phi_mask(layer.weight)
        assert torch.equal(mask, phi_mask(layer.weight.data)) and not mask.requires_grad


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

    def test_capacity_utilization_is_kept_over_the_most_the_pattern_allows(self):
        """A full φ mask and a full 2:4 mask use all of their capacity, at an even and an odd width."""
        for n in (8, 9):
            assert mask_stats(phi_mask(torch.ones(n, 3)))["capacity_utilization"] == pytest.approx(1.0), n
        for n in (6, 7):
            stats = mask_stats(two_four_mask(torch.randn(n, 3)), pattern="2:4")
            assert stats["capacity_utilization"] == pytest.approx(1.0), n


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

    @pytest.mark.parametrize("n_digits", range(6, 15))
    def test_levels_by_rounding_match_the_grid_search(self, n_digits):
        """Each kept weight goes to the level the nearest-of-two grid search picked, ties going up, at every width."""
        enc = FibonacciEncoder(n_digits)
        t = torch.cat([torch.randn(20000), torch.rand(2000) * 1e-3, torch.tensor([0.0, 1.0, -1.0, 0.5])])
        mask = torch.rand_like(t) > 0.2
        for m in (None, mask):
            active = t if m is None else t[m]
            vmin, vmax = active.min().item(), active.max().item()
            scale = enc.max_value / (vmax - vmin)
            scaled = (active.numpy() - vmin) * scale
            idx = np.clip(np.searchsorted(enc.grid, scaled, side="left"), 0, len(enc.grid) - 1)
            upper, lower = enc.grid[idx], enc.grid[np.maximum(idx - 1, 0)]
            searched = np.where(np.abs(scaled - lower) < np.abs(scaled - upper), lower, upper)
            want = torch.tensor(searched / scale + vmin, dtype=t.dtype)
            got = enc.encode_tensor(t, mask=m)[0]
            assert torch.equal(got if m is None else got[m], want)

    @pytest.mark.parametrize("weights, dtype, want", [
        ([0.0, 0.49999997, 143.0], torch.float32, [0.0, 0.0, 143.0]),  # 0.49999997 + 0.5 rounds to 1.0 in float32
        ([0.0, 0.5 - 2 ** -54, 143.0], torch.float64, [0.0, 0.0, 143.0]),  # the float64 twin
        ([0.0, 0.5, 143.0], torch.float32, [0.0, 1.0, 143.0]),  # an exact tie goes up
        ([0.0, 1.0, float("inf")], torch.float32, [float("nan"), float("nan"), float("inf")]),  # as the grid search gave
    ])
    def test_levels_round_exactly_near_a_half(self, weights, dtype, want):
        """A scaled value just under a half goes down in every dtype, a half goes up, and an inf weight decodes as before."""
        got = FibonacciEncoder(10).encode_tensor(torch.tensor(weights, dtype=dtype))[0]
        assert torch.equal(got.isnan(), torch.tensor(want).isnan()), got
        assert torch.equal(got.nan_to_num(), torch.tensor(want, dtype=dtype).nan_to_num()), got

    def test_stream_all_grid_values(self):
        """Every quantization level, written plus 1 as export_bitstream writes it, reads back from the Fibonacci code."""
        enc = FibonacciEncoder(n_digits=8)
        values = [int(v) for v in enc.grid]
        assert [code - 1 for code in decode_gaps(encode_gaps(v + 1 for v in values))] == values


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
        # 0.4091 over every 8-digit quantization level, as experiments/phi-prune/results/measure_detection.txt in
        # ezexe/p4-phi holds
        assert 0.40 < exact < 0.42

    @pytest.mark.parametrize("axis", [None, 0])
    def test_check_fails_on_noise_nan_and_a_flipped_float_bit(self, axis):
        """cassini_check passes every quantized weight, per tensor and per channel, and fails noise, NaN and a flipped bit."""
        from phi_prune.integrity import cassini_check

        torch.manual_seed(0)
        pruned, masks, enc, scales = encoded_mlp(axis)
        report = cassini_check(pruned, masks, enc, scales)["_aggregate"]
        assert report["total"] == sum(int(m.sum()) for m in masks.values()) and report["failed"] == 0
        name = next(iter(masks))
        param = dict(pruned.named_parameters())[name]
        keep = masks[name].bool()
        n = int(keep.sum())
        clean = param.data.clone()
        for corrupt, at_least in (("noise", n), ("nan", n), ("bit", 1)):
            param.data.copy_(clean)
            if corrupt == "noise":
                param.data[keep] += torch.randn(n) * 0.05
            elif corrupt == "nan":
                param.data[keep] = float("nan")
            else:  # the lowest exponent bit of one kept float32 weight, which doubles or halves it
                i = int(keep.view(-1).nonzero()[0])
                param.data.view(-1).view(torch.int32)[i] ^= 1 << 23
            assert cassini_check(pruned, masks, enc, scales)[name]["failed"] >= at_least, corrupt

    def test_check_passes_an_empty_mask_and_refuses_a_layer_without_scales(self):
        """A mask that keeps nothing passes with pass_rate 1.0, and a masked layer missing from scales raises."""
        from phi_prune.integrity import cassini_check, check_tensor_integrity

        report = check_tensor_integrity(torch.randn(4), torch.zeros(4), FibonacciEncoder(n_digits=8), 1.0, 0.0)
        assert report["total"] == 0 and report["pass_rate"] == 1.0
        pruned, masks, enc, scales = encoded_mlp()
        del scales[next(iter(scales))]
        with pytest.raises(ValueError, match="scales"):
            cassini_check(pruned, masks, enc, scales)

    @pytest.mark.parametrize("axis", [None, 0])
    def test_check_passes_weights_encoded_from_a_range_under_1e_10(self, axis):
        """encode_tensor puts kept weights that span less than 1e-10 on level 0, per tensor and per channel, and cassini_check passes them."""
        from phi_prune.integrity import cassini_check

        tiny = [0.0, 5e-11, 2e-11, 3e-11]
        rows = [tiny] if axis is None else [tiny, [0.1, -0.2, 0.3, 0.05]]
        layer = nn.Linear(4, len(rows), bias=False)
        enc = FibonacciEncoder(n_digits=8)
        encoded, scale, _, offset = enc.encode_tensor(torch.tensor(rows), return_offset=True, axis=axis)
        layer.weight.data.copy_(encoded)
        masks = {"weight": torch.ones_like(layer.weight.data)}
        report = cassini_check(layer, masks, enc, {"weight": (scale, offset)})["_aggregate"]
        assert report["total"] == 4 * len(rows) and report["failed"] == 0


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

    def test_prune_density_below_half_reaches_the_target(self):
        """A density below 0.5 keeps about that fraction of each layer, 0.5 keeps the default's masks, and 2:4 refuses it."""
        torch.manual_seed(0)
        model = self._make_model()
        pruned, masks = prune(model, density=0.25, prune_head=True)
        for mask in masks.values():
            kept = int(mask.flatten(1).amax(1).sum())  # one value per output channel
            assert abs(kept - round(0.25 * mask.shape[0])) <= 2
        full = prune(model, prune_head=True)[1]
        half = prune(model, density=0.5, prune_head=True)[1]
        assert all(torch.equal(full[k], half[k]) for k in full)
        with pytest.raises(ValueError, match="2:4"):
            prune(model, pattern="2:4", density=0.25)

    def test_grad_scaler_is_torch_amp_alone(self):
        """pyproject floors torch at 2.4.1, where torch.amp.GradScaler always exists, so no older branch is kept."""
        import inspect
        from phi_prune import api
        assert isinstance(api._grad_scaler(False), torch.amp.GradScaler)
        assert "torch.cuda.amp" not in inspect.getsource(api._grad_scaler)

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

def mlp():
    """The small MLP the export and command tests prune."""
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 10))


def encoded_mlp(axis=None):
    """A pruned MLP with its masked layers quantized in place, and the scale and quantization offset of each."""
    pruned, masks = prune(mlp())
    enc = FibonacciEncoder(n_digits=8)
    scales = {}
    for name, param in pruned.named_parameters():
        if name in masks:
            encoded, scale, _, offset = enc.encode_tensor(param.data, mask=masks[name],
                                                          return_offset=True, axis=axis)
            param.data.copy_(encoded)
            scales[name] = (scale, offset)
    return pruned, masks, enc, scales


def _read_phi(path):
    """The header line and the payload as '0'/'1' text of a file export_bitstream wrote."""
    head, rest = path.read_bytes().split(b"\n", 1)
    n = json.loads(head)["payload_bits"]
    return head.decode(), (np.unpackbits(np.frombuffer(rest, dtype=np.uint8))[:n] + ord("0")).tobytes().decode()


def _write_phi(path, header_line, payload):
    """Write a header line and a '0'/'1' payload as export_bitstream lays them out, packed, with payload_bits set."""
    header = {**json.loads(header_line), "payload_bits": len(payload)}
    bits = np.frombuffer(payload.encode(), dtype=np.uint8) - ord("0")
    path.write_bytes(json.dumps(header).encode() + b"\n" + np.packbits(bits).tobytes())


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
        assert b"\r" not in path.read_bytes().split(b"\n", 1)[0]  # the header line ends in "\n" on every platform
        header_line, payload = _read_phi(path)
        header = json.loads(header_line)
        assert sum(layer["n_active"] for layer in header["layers"]) == stats["n_weights"] == len(expected)

        # codeword_digits and level_bias give each codeword's width and what it adds to its quantization level
        width = header["encoder"].get("codeword_digits", header["encoder"]["n_digits"])
        bias = header["encoder"].get("level_bias", 0)
        assert width == enc.n_digits + 1
        recovered = [code - bias for code in decode_gaps(payload)]
        assert recovered == expected

    def test_payload_packs_eight_bits_a_byte(self, tmp_path):
        """The payload holds its bits eight to a byte, with their count in the header, and loads back bit for bit."""
        pruned, masks, enc, scales = encoded_mlp()
        path = tmp_path / "model.phi"
        stats = export_bitstream(pruned, masks, enc, scales, str(path))
        head, payload = path.read_bytes().split(b"\n", 1)
        assert json.loads(head)["payload_bits"] == stats["bitstream_length"]
        assert len(payload) == (stats["bitstream_length"] + 7) // 8
        restored = copy.deepcopy(pruned)
        for name, param in restored.named_parameters():
            if name in masks:
                param.data[masks[name].bool()] = 0.0
        load_bitstream(restored, masks, str(path))
        for (name, before), after in zip(pruned.named_parameters(), restored.parameters()):
            assert torch.equal(before, after), name

    def test_payload_refuses_a_bit_count_or_padding_that_does_not_fit(self, tmp_path):
        """A payload_bits the bytes cannot hold, or a set bit in the padding of the last byte, raises ValueError."""
        pruned, masks, enc, scales = encoded_mlp()
        path = tmp_path / "model.phi"
        n = export_bitstream(pruned, masks, enc, scales, str(path))["bitstream_length"]
        head, payload = path.read_bytes().split(b"\n", 1)
        header = json.loads(head)
        # n + 8 asks for a byte more than the file holds; n - 2 leaves the last codeword's comma, 11, as padding
        for bits in (n + 8, n - 2):
            path.write_bytes(json.dumps({**header, "payload_bits": bits}).encode() + b"\n" + payload)
            with pytest.raises(ValueError):
                load_bitstream(copy.deepcopy(pruned), masks, str(path))

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
        header_line, payload = _read_phi(path)
        old = json.loads(header_line)
        del old["encoder"]["codeword_digits"], old["encoder"]["level_bias"]
        last = list(masks)[-1]
        misfit = {**masks, last: torch.ones_like(masks[last])}
        refusals = [
            (json.dumps(old) + "\n" + payload, masks, "level_bias"),
            (header_line + "\n" + payload[:-2], masks, "inside a codeword"),  # the last codeword cut off
            (header_line + "\n" + payload, misfit, "does not match"),
        ]

        target = copy.deepcopy(pruned)
        for name, param in target.named_parameters():
            if name in masks:
                param.data[masks[name].bool()] = 0.0
        before = copy.deepcopy(target.state_dict())
        for text, file_masks, message in refusals:
            _write_phi(path, *text.split("\n", 1))
            with pytest.raises(ValueError, match=message):
                load_bitstream(target, file_masks, str(path))
        assert all(torch.equal(before[k], v) for k, v in target.state_dict().items())

    def test_load_bitstream_refuses_a_model_or_masks_without_a_layer_of_the_file(self, tmp_path):
        """load_bitstream raises ValueError naming a layer the model or masks lack or whose mask has another shape, and writes nothing."""
        pruned, masks, enc, scales = encoded_mlp()
        path = str(tmp_path / "model.phi")
        export_bitstream(pruned, masks, enc, scales, path)
        with pytest.raises(ValueError, match="2.weight"):
            load_bitstream(nn.Sequential(nn.Linear(16, 32)), masks, path)
        with pytest.raises(ValueError, match="0.weight"):
            load_bitstream(pruned, {}, path)

        target = copy.deepcopy(pruned)
        for name, param in target.named_parameters():
            if name in masks:
                param.data[masks[name].bool()] = 0.0
        before = copy.deepcopy(target.state_dict())
        last = list(masks)[-1]
        with pytest.raises(ValueError, match=last):
            load_bitstream(target, {**masks, last: masks[last].reshape(1, -1)}, path)
        assert all(torch.equal(before[k], v) for k, v in target.state_dict().items())


# ── The phi-prune command ──

def run_cli(fn, path):
    """A command function's exit code and printed text, for the file at path."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            code = fn(argparse.Namespace(model=path))
    except SystemExit as e:
        code = e.code
    return code or 0, out.getvalue()


def run_main(monkeypatch, *args, encoding="utf-8"):
    """phi-prune run in this process with args, on streams in encoding as a console's are; its exit code and printed text."""
    from phi_prune.cli import main

    out = io.TextIOWrapper(io.BytesIO(), encoding=encoding)
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", out)
    monkeypatch.setattr(sys, "argv", ["phi-prune", *args])
    try:
        main()
        code = 0
    except SystemExit as e:
        code = e.code or 0
    out.flush()
    return code, out.buffer.getvalue().decode(encoding)


class TestCLI:
    @pytest.mark.parametrize("saved_pattern", ["phi", "zeckendorf", None])
    def test_cli_check_reads_phi_zeckendorf_and_no_pattern(self, tmp_path, saved_pattern):
        """phi-prune check reads the pattern "phi", its other name "zeckendorf", and a checkpoint with no pattern key."""
        from phi_prune.cli import cmd_check
        from phi_prune.export import save_checkpoint

        pruned, masks, _, _ = encoded_mlp()
        path = str(tmp_path / "model.pt")
        save_checkpoint(pruned, masks, path)
        data = torch.load(path, weights_only=True)
        assert data["pattern"] == "phi" and data["format"] == "phi-prune"
        if saved_pattern is None:
            del data["pattern"]
        else:
            data["pattern"] = saved_pattern
        torch.save(data, path)
        assert cmd_check(argparse.Namespace(model=path)) == 0

    def test_check_reads_the_axis_prune_used(self, tmp_path):
        """check and phi-prune check read the masks of a model pruned along axis 1 along axis 1."""
        from phi_prune.cli import cmd_check
        from phi_prune.export import save_checkpoint

        pruned, masks = prune(mlp(), axis=1)
        assert check(pruned, masks)["_summary"]["all_masks_valid"]
        path = str(tmp_path / "axis1.pt")
        save_checkpoint(pruned, masks, path)
        assert torch.load(path, weights_only=True)["axis"] == 1
        assert run_cli(cmd_check, path)[0] == 0

    def test_cli_check_fails_when_pruned_weights_are_not_zero(self, tmp_path):
        """phi-prune check exits 1 on a checkpoint whose pruned weights were overwritten, as check reports them."""
        from phi_prune.cli import cmd_check
        from phi_prune.export import save_checkpoint

        pruned, masks = prune(mlp())
        for name, param in pruned.named_parameters():
            if name in masks:
                param.data.fill_(1.0)
        assert not check(pruned, masks)["_summary"]["all_zeros_enforced"]
        path = str(tmp_path / "overwritten.pt")
        save_checkpoint(pruned, masks, path)
        assert run_cli(cmd_check, path)[0] == 1

    def test_cli_runs_on_a_console_that_is_not_utf8(self, tmp_path, monkeypatch):
        """phi-prune --help, check and info exit 0 on a cp1252 console, which has no φ, ∞, ✓ or ✗, and print a ? for each."""
        from phi_prune.export import save_checkpoint

        pruned, masks = prune(mlp())
        path = str(tmp_path / "model.pt")
        save_checkpoint(pruned, masks, path)
        printed = {}
        for args in (["--help"], ["check", path], ["info", path]):
            code, printed[args[0]] = run_main(monkeypatch, *args, encoding="cp1252")
            assert code == 0, (args, printed[args[0]][-300:])
        assert "?" in printed["--help"] and "?" in printed["check"]

    def test_cli_exit_codes(self, tmp_path, monkeypatch):
        """info on a file without masks exits 1, and phi-prune with no command exits 2, argparse's code for a usage error."""
        from phi_prune.cli import cmd_info

        path = str(tmp_path / "list.pt")
        torch.save([1, 2, 3], path)
        assert run_cli(cmd_info, path)[0] == 1
        assert run_main(monkeypatch, "info", path)[0] == 1
        assert run_main(monkeypatch)[0] == 2

    def test_cli_offers_no_prune_that_always_exits_1(self, tmp_path, monkeypatch):
        """phi-prune --help offers check and info only, and phi-prune prune is a usage error that exits 2."""
        path = str(tmp_path / "state.pt")
        torch.save(mlp().state_dict(), path)
        code, help_text = run_main(monkeypatch, "--help")
        assert code == 0 and "{check,info}" in help_text
        assert run_main(monkeypatch, "prune", path)[0] == 2

    def test_cli_info_on_a_checkpoint_with_no_masked_layer(self, tmp_path):
        """info on the checkpoint of a model that prune left dense prints that no layer is masked and exits 0."""
        from phi_prune.cli import cmd_info
        from phi_prune.export import save_checkpoint

        pruned, masks = prune(nn.Sequential(nn.Linear(16, 4)))
        assert masks == {}
        path = str(tmp_path / "dense.pt")
        save_checkpoint(pruned, masks, path)
        assert run_cli(cmd_info, path)[0] == 0

    def test_cli_reads_a_checkpoint_whose_scales_are_numpy_arrays(self, tmp_path):
        """check and info read a checkpoint whose per-channel scales are numpy arrays, as save_checkpoint stored them before."""
        from phi_prune.cli import cmd_check, cmd_info
        from phi_prune.export import save_checkpoint

        pruned, masks, enc, scales = encoded_mlp(axis=0)
        path = str(tmp_path / "model.pt")
        save_checkpoint(pruned, masks, path, encoder=enc, scales=scales)
        data = torch.load(path, weights_only=False)  # the test's own file, read as the base read every file
        data["scales"] = {name: {"scale": s, "offset": o} for name, (s, o) in scales.items()}
        assert all(isinstance(pair["scale"], np.ndarray) for pair in data["scales"].values())
        torch.save(data, path)
        assert run_cli(cmd_check, path)[0] == 0 and run_cli(cmd_info, path)[0] == 0

    def test_per_channel_checkpoint_loads_under_weights_only(self, tmp_path):
        """A checkpoint with per-channel scales loads under weights_only=True with the same values, and check and info read it."""
        from phi_prune.cli import cmd_check, cmd_info
        from phi_prune.export import save_checkpoint

        pruned, masks, enc, scales = encoded_mlp(axis=0)
        path = str(tmp_path / "model.pt")
        save_checkpoint(pruned, masks, path, encoder=enc, scales=scales)
        data = torch.load(path, weights_only=True)
        for name, (scale, offset) in scales.items():
            assert np.array_equal(np.asarray(data["scales"][name]["scale"]), scale)
            assert np.array_equal(np.asarray(data["scales"][name]["offset"]), offset)
        assert run_cli(cmd_check, path)[0] == 0 and run_cli(cmd_info, path)[0] == 0
        # metadata that weights_only refuses ends the command with exit 1 and a message, not a traceback
        save_checkpoint(pruned, masks, path, metadata={"t": np.float32(1.0)})
        code, out = run_cli(cmd_check, path)
        assert code == 1 and "weights_only" in out

    def test_checkpoint_records_the_package_version(self, tmp_path, monkeypatch):
        """save_checkpoint records phi_prune.__version__, so a checkpoint names the release that wrote it."""
        import phi_prune
        from phi_prune.export import save_checkpoint

        monkeypatch.setattr(phi_prune, "__version__", "9.9.9")
        pruned, masks = prune(mlp())
        path = str(tmp_path / "model.pt")
        save_checkpoint(pruned, masks, path)
        assert torch.load(path, weights_only=True)["version"] == "9.9.9"

    def test_check_fails_a_mask_bit_flipped_outside_the_slice_it_reads(self, tmp_path):
        """check and phi-prune check fail a mask with a bit flipped from 0 to 1 outside the slice mask_stats reads."""
        from phi_prune.cli import cmd_check
        from phi_prune.export import save_checkpoint

        torch.manual_seed(0)
        pruned, masks = prune(mlp())
        name = next(iter(masks))
        mask = masks[name].clone()
        i = int((mask[:, 0] == 0).nonzero()[0])  # a pruned output channel
        mask[i, 3] = 1.0
        flipped = {**masks, name: mask}
        assert not check(pruned, flipped)["_summary"]["all_masks_valid"]
        path = str(tmp_path / "flipped.pt")
        save_checkpoint(pruned, flipped, path)
        assert run_cli(cmd_check, path)[0] == 1

    def test_cli_check_on_a_file_that_holds_a_bare_tensor(self, tmp_path):
        """phi-prune check on a file that holds a bare tensor prints that it holds no masks and exits 1, raising nothing."""
        from phi_prune.cli import cmd_check

        path = str(tmp_path / "bare.pt")
        torch.save(torch.zeros(3), path)
        code, out = run_cli(cmd_check, path)
        assert code == 1 and "masks" in out
