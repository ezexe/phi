"""Tests for phi_prune.register: the φ-register's local rewrites against a binary register's carries."""

import random

import pytest

from phi_prune.encoding import FibonacciEncoder
from phi_prune.register import (
    BINARY, KINDS, PHI, add, apply, correction_candidates, flip_detected, from_int,
    is_canonical, normalize, place_values, rewrites, value,
)


def canonical_words(kind, n):
    """Every canonical n-digit register of a kind."""
    top = (1 << n) if kind == BINARY else place_values(PHI, n + 1)[-1]
    return [from_int(v, kind, n) for v in range(top)]


def single_flips(word):
    """(bit position, flipped word) for every single-bit flip of word."""
    for i in range(len(word)):
        flipped = list(word)
        flipped[i] ^= 1
        yield i, flipped


class TestRewrites:
    @pytest.mark.parametrize("kind", KINDS)
    def test_rules_keep_value(self, kind):
        rng = random.Random(0)
        for _ in range(500):
            d = [rng.randint(0, 3) for _ in range(rng.randint(1, 12))]
            for rule, k in rewrites(d, kind):
                assert value(apply(d, rule, k), kind) == value(d, kind)

    @pytest.mark.parametrize("kind", KINDS)
    def test_any_order_reaches_the_same_canonical_form(self, kind):
        """No global sweep order is needed: random firing orders all agree."""
        rng = random.Random(1)
        for _ in range(200):
            d = [rng.randint(0, 3) for _ in range(rng.randint(1, 10))]
            expected = from_int(value(d, kind), kind)
            for seed in range(10):
                out, _ = normalize(d, kind, random.Random(seed))
                assert is_canonical(out, kind)
                assert out[:len(expected)] == expected and not any(out[len(expected):])

    def test_phi_canonical_form_matches_encoder(self):
        enc = FibonacciEncoder(n_digits=10)
        for v in enc.grid:
            assert from_int(int(v), PHI, 10) == enc.to_codeword(int(v))

    @pytest.mark.parametrize("kind", KINDS)
    def test_add_exhaustive(self, kind):
        words = canonical_words(kind, 6)
        for a in words:
            for b in words:
                s, _ = add(a, b, kind)
                assert is_canonical(s, kind)
                assert value(s, kind) == value(a, kind) + value(b, kind)

    def test_phi_carry_runs_down_binary_does_not(self):
        """2·3 = 5 + 1: a φ split reaches two places below; a binary carry only goes up."""
        assert apply([0, 0, 2], "split", 2) == [1, 0, 0, 1]
        assert apply([0, 0, 2], "carry", 2) == [0, 0, 0, 1]

    def test_phi_needs_more_rewrites_than_binary_for_the_same_values(self):
        """Over every sum of two values below 55, φ fires more rules, mean and worst case."""
        counts = {}
        for kind in KINDS:
            steps = [add(from_int(a, kind), from_int(b, kind), kind)[1] for a in range(55) for b in range(55)]
            counts[kind] = (sum(steps) / len(steps), max(steps))
        assert counts[PHI][0] > counts[BINARY][0]
        assert counts[PHI][1] > counts[BINARY][1]


class TestCorruption:
    def test_binary_never_detects_a_flip(self):
        for word in canonical_words(BINARY, 8):
            for _, flipped in single_flips(word):
                assert not flip_detected(flipped, BINARY)

    def test_phi_detects_only_zero_to_one_flips(self):
        caught = total = 0
        for word in canonical_words(PHI, 8):
            for i, flipped in single_flips(word):
                total += 1
                if flip_detected(flipped, PHI):
                    caught += 1
                    assert word[i] == 0
        assert 0.40 < caught / total < 0.42  # ~40.9%, as in test_core's exhaustive count

    def test_normalizing_keeps_the_corrupted_value(self):
        """The rewrites make a corrupt register canonical again but do not undo the flip."""
        places = place_values(PHI, 8)
        for word in canonical_words(PHI, 8):
            for i, flipped in single_flips(word):
                if flip_detected(flipped, PHI):
                    out, _ = normalize(flipped, PHI)
                    assert is_canonical(out, PHI)
                    assert value(out, PHI) == value(word, PHI) + places[i]

    def test_correction_candidates(self):
        assert correction_candidates([1, 1, 1]) == [[1, 0, 1]]  # flip between two 1s: correctable
        assert correction_candidates([0, 1, 1, 0]) == [[0, 0, 1, 0], [0, 1, 0, 0]]  # lone 11: detect only

        corrected = total = 0
        for word in canonical_words(PHI, 8):
            for _, flipped in single_flips(word):
                total += 1
                if flip_detected(flipped, PHI):
                    candidates = correction_candidates(flipped)
                    assert word in candidates
                    corrected += len(candidates) == 1
        assert 0.08 < corrected / total < 0.09  # ~8.6% of all flips can be corrected, not just caught
