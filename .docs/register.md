# φ-Register vs Binary Register: a Walkthrough

This walkthrough covers the φ-register work on branch `claude/kind-hopper-juyjmk`, in the order it was built:

1. The growth-rate equations that set up the comparison
2. `phi_prune/register.py`, function by function
3. `tests/test_register.py`, test by test, with what each one establishes
4. The changes to `.docs/phi/phi.md` and `.docs/phi/matrix.md`
5. The notebook `.docs/register_settling.ipynb`, cell by cell, with the saved run's output
5b–5d. The transformer checks: low and high density and level counts, and fine-tuning updates
6. What holds up, what doesn't, and what is still open

Run everything with:

```bash
pip install -e ".[dev]"
pytest -q tests/test_register.py          # 16 tests, a few seconds
jupyter nbconvert --to notebook --execute .docs/register_settling.ipynb   # about a minute on CPU
```

### Scripts, outputs, and where each is explained

Every number in this walkthrough comes from one of these scripts. Their saved output is in [`results/`](results/).

| Section | What | Script | Saved output |
|---|---|---|---|
| 0 | Choice each mask pattern leaves | [`scripts/session/mask_choice.py`](../scripts/session/mask_choice.py) | [`mask_choice.txt`](results/mask_choice.txt) |
| 1 | Growth rates and digits per level count | [`scripts/session/growth_rates.py`](../scripts/session/growth_rates.py) | [`growth_rates.txt`](results/growth_rates.txt) |
| 2 | The register module | [`phi_prune/register.py`](../phi_prune/register.py) | — |
| 3 | Tests | [`tests/test_register.py`](../tests/test_register.py) | `pytest -q` |
| 3, 6 | Flips, rules fired, settling rounds, 32-lane lockstep | [`scripts/session/register_measure.py`](../scripts/session/register_measure.py) | [`register_measure.txt`](results/register_measure.txt) |
| 5 | First dot-product prototype | [`scripts/session/dot_prototype.py`](../scripts/session/dot_prototype.py) | [`dot_prototype.txt`](results/dot_prototype.txt) |
| 5 | Notebook generator | [`scripts/session/make_register_notebook.py`](../scripts/session/make_register_notebook.py) | [`register_settling.ipynb`](register_settling.ipynb), outputs saved in the notebook |
| 5b | Transformer check, 50% and 25% masks, 16–32 levels | [`scripts/transformer_check.py`](../scripts/transformer_check.py) | [`transformer_check_low.log`](results/transformer_check_low.log), [`.json`](results/transformer_check_low.json) |
| 5c | Same, `--part high`: 75% masks, 55–987 levels | same script | [`transformer_check_high.log`](results/transformer_check_high.log), [`.json`](results/transformer_check_high.json) |
| 5c | 1 digits over uniformly spread levels | [`scripts/session/digit_counts_uniform.py`](../scripts/session/digit_counts_uniform.py) | [`digit_counts_uniform.txt`](results/digit_counts_uniform.txt) |
| 5d | Same, `--part delta`: fine-tuning updates | same script | [`transformer_check_delta.log`](results/transformer_check_delta.log), [`.json`](results/transformer_check_delta.json) |
| — | Smoke runs (too short to mean anything; they only showed the pipeline works) | same script | [`results/smoke/`](results/smoke/) |

Rerun the scripts with, for example, `python scripts/session/register_measure.py` or `python scripts/transformer_check.py --part high`.

---

## 0. Starting point: why φ masks lose to 2:4 at 50%

This section started as analysis only. The density target it proposes is now `adjacency_dp_density` in `phi_prune/masks.py`, and section 5b tests it.
The φ pruner keeps the heaviest set of channels with no two neighbours, which lands at 48.8–49.8% density.
At that density almost no masks satisfy the rule, so the mask can't follow where the weights matter.
The table below counts how much choice each mask has on a 64-channel layer, in bits of choice per channel:

| Mask | Density | Choice (bits/channel) |
|---|---|---|
| φ at its cap | 50% | 0.08 |
| 2:4 | 50% | 0.65 |
| φ | 37.5% | 0.58 |
| φ | 28% | 0.66 |
| 1:4 | 25% | 0.50 |

Below its cap the φ mask has as much choice as 2:4 does at 50%, and more than 1:4, its fixed-pattern rival at that density.
The next step proposed for this is a density target for `adjacency_dp` (subtract a threshold λ from every score and search λ) plus a 1:4 baseline.

## 1. Growth rates

**Binary:** n digits, every pattern allowed, gives B(n) = 2ⁿ values: 2, 4, 8, 16 … 1024 … 8192.

**Fibinary:** n digits with no two adjacent 1s.
A valid string ends in 0 (any valid n−1 string before it) or in 01 (any valid n−2 string before it), so

```
Z(n) = Z(n−1) + Z(n−2),   Z(0) = 1, Z(1) = 2   →   Z(n) = F(n+2) = 1, 2, 3, 5, 8, 13, 21 …
Z(n) = (φⁿ⁺² − ψⁿ⁺²) / √5 ≈ 0.724 · φⁿ,   φ = 1.618, ψ = −0.618
```

Each extra digit multiplies the count by φ in fibinary and by 2 in binary; φ is the largest eigenvalue of the Q matrix `M = [[1,1],[1,0]]`.
The ratio Z(n)/B(n) shrinks as (φ/2)ⁿ ≈ 0.809ⁿ.
Each fibinary digit carries log₂φ = 0.694 bits, so holding N values takes about 1.44× as many digits as binary:

| Levels | Binary digits | Fibinary digits | Extra |
|---|---|---|---|
| 8 | 3 | 4 | +1 |
| 55 | 6 | 8 | +2 |
| 144 | 8 | 10 | +2 |
| 1024 | 10 | 15 | +5 |
| 8192 | 13 | 19 | +6 |

The cost is a fixed 44% of the digit count, so it is small in absolute terms at the low end, where phi-prune works (10 digits for 144 levels).

---

## 2. `phi_prune/register.py`

The module models both registers as **local rewrite rules**.
A register is a list of digits, least significant first.
A digit may go above 1 for a while, for example right after two registers are added digit by digit.

### Constants and place values

```python
PHI = "phi"
BINARY = "binary"
KINDS = (PHI, BINARY)


def place_values(kind: str, n: int) -> List[int]:
    """The first n place values of a register: 1, 2, 3, 5, ... or 1, 2, 4, 8, ..."""
    if kind == BINARY:
        return [1 << k for k in range(n)]
    if kind == PHI:
        fibs = [1, 2]
        while len(fibs) < n:
            fibs.append(fibs[-1] + fibs[-2])
        return fibs[:n]
    raise ValueError(f"Unknown register kind {kind!r}: expected one of {KINDS}")
```

The φ place values 1, 2, 3, 5, 8 … are the same digits `FibonacciEncoder` uses, so a φ-register and an encoded weight share one numbering.

### Reading and writing values

```python
def value(digits: List[int], kind: str) -> int:
    """Integer value of a digit list (LSB first), digits allowed above 1."""
    return sum(d * p for d, p in zip(digits, place_values(kind, len(digits))))


def from_int(v: int, kind: str, n_digits: Optional[int] = None) -> List[int]:
    """Canonical digits of v (greedy from the top), padded to n_digits (no leading zeros when None)."""
    ...


def is_canonical(digits: List[int], kind: str) -> bool:
    """Digits all 0 or 1, and for a φ-register no two adjacent 1s."""
    if any(d not in (0, 1) for d in digits):
        return False
    return kind == BINARY or adjacency_check(list(digits))
```

- **`value`** reads any register, including one that is still settling.
- **`from_int`** writes a number greedily from the top, which gives the Zeckendorf form for φ and plain binary for binary.
- **`is_canonical`** is the "settled" test. For φ it reuses the package's own `integrity.adjacency_check`, so the check for being finished is literally the integrity check.

### The rewrite rules

```python
def rewrites(digits: List[int], kind: str) -> List[Tuple[str, int]]:
    """Every rule that can fire on digits, as (rule, k) pairs."""
    ...


def apply(digits: List[int], rule: str, k: int) -> List[int]:
    """Fire one rule at position k; returns a new list, grown when a carry runs off the top."""
    d = list(digits) + [0, 0]
    if rule == "carry":                 # binary:  2·2^k = 2^(k+1)
        d[k] -= 2
        d[k + 1] += 1
    elif rule == "merge":               # φ:  F(k) + F(k+1) = F(k+2)
        d[k] -= 1
        d[k + 1] -= 1
        d[k + 2] += 1
    elif rule == "split":               # φ:  2·F(k) = F(k+1) + F(k−2)
        d[k] -= 2
        d[k + 1] += 1
        if k >= 2:
            d[k - 2] += 1
        elif k == 1:
            d[0] += 1                   # 2 + 2 = 3 + 1
    ...
```

| Register | Rule | Fires when | Effect | Identity |
|---|---|---|---|---|
| φ | merge | `d[k]` and `d[k+1]` both ≥ 1 | −1, −1, +1 at k+2 | F(k) + F(k+1) = F(k+2) |
| φ | split | `d[k]` ≥ 2 | −2, +1 at k+1, +1 at k−2 | 2F(k) = F(k+1) + F(k−2) |
| binary | carry | `d[k]` ≥ 2 | −2, +1 at k+1 | 2·2ᵏ = 2ᵏ⁺¹ |

Every rule touches at most three nearby digits and keeps the value.
The φ split is the only rule that sends a carry **down**.

### Normalizing: one rule at a time

```python
def normalize(digits, kind, rng=None, max_steps=100_000):
    """
    Fire rules until the register is canonical.

    With rng, each step fires a rule picked at random among those that
    can fire, standing in for unclocked hardware where any local rewrite
    may happen first; without it, always the lowest position first.

    Returns:
        (canonical digits, number of rules fired)
    """
```

Passing an `rng` fires rules in a random order.
The tests use this to show that no global sweep order is needed.

### Normalizing: in parallel rounds

```python
def normalize_rounds(digits, kind, max_rounds=10_000):
    """
    Normalize in rounds, each firing at once every rule it can without two
    rules touching the same digit (picked from the lowest position up).

    Rounds stand in for settling time when every cell rewrites itself in
    parallel. A self-timed register stops after the rounds a sum needs;
    a clocked one waits the worst case on every sum.
    """
    d = list(digits)
    for rounds in range(max_rounds + 1):
        options = rewrites(d, kind)
        if not options:
            return d, rounds
        used = set()
        for rule, k in options:
            cells = {"carry": {k, k + 1}, "merge": {k, k + 1, k + 2}, "split": {k - 2, k, k + 1}}[rule]
            if not cells & used:
                used |= cells
                d = apply(d, rule, k)
```

This is the timing model behind every settling number below.
One round is every cell rewriting at once.
- A **self-timed** register pays the rounds each sum actually takes.
- A **clocked** one pays its worst case on every sum.

### Adding

```python
def pour(a, b):
    """Digit-by-digit sum of two registers, digits allowed above 1."""


def add(a, b, kind, rng=None):
    """pour, then normalize; returns (digits, rules fired)."""
    return normalize(pour(a, b), kind, rng)
```

Addition is *pour, then settle*: add the digits position by position, then let the rules clean up.

### Dot products: the multiply-accumulate step

```python
def dot(levels, xs, kind):
    """
    Shift-and-add dot product of non-negative integer weight levels and
    activations, accumulated in one register of the given kind.
    ...
    Returns:
        (accumulator digits, rounds each addition took to settle)
    """
    acc, rounds = [0], []
    for level, x in zip(levels, xs):
        level, x = int(level), int(x)
        if level == 0 or x == 0:
            continue
        digits = from_int(level, kind)
        for k, p in enumerate(place_values(kind, len(digits))):
            if digits[k]:
                acc, r = normalize_rounds(pour(acc, from_int(x * p, kind)), kind)
                rounds.append(r)
    return acc, rounds
```

This is phi-prune's own workload: a Fibonacci-encoded weight times an activation.
- **Partial products:** each 1 digit of the weight level adds one partial product, `x · F(k)` or `x · 2ᵏ`, to the accumulator.
- **Settling:** after each addition the accumulator settles, and the rounds it took are recorded.
- **Why non-negative integers are enough:** encoded weights are `level / scale + offset` with `level ≥ 0`, and activations after ReLU are ≥ 0.
- **Partial products are taken as given:** in binary, `x · 2ᵏ` is just wiring. In φ, `x · F(k)` needs a ladder of additions; the notebook counts that cost separately.

### Corruption

```python
def flip_detected(codeword, kind):
    """Whether a register reads as corrupt: only a φ-register can, via adjacent 1s."""
    return not is_canonical(codeword, kind)


def correction_candidates(codeword):
    """
    Canonical φ-registers one 1 -> 0 change away from a corrupt codeword.
    ...
    One candidate means the flip can be corrected; two (a lone 11 pair)
    means it can only be detected.
    """
```

- **Detection:** only a 0→1 flip next to a 1 can be seen.
- **Correction:** a flip that lands between two 1s (`101 → 111`) has exactly one valid repair. A lone `11` could have come from either cell, so it can only be detected.

---

## 3. `tests/test_register.py`

Helpers: `canonical_words(kind, n)` lists every settled n-digit register; `single_flips(word)` yields every one-bit corruption of it.

### `TestRewrites`

| Test | What it checks | Result |
|---|---|---|
| `test_rules_keep_value` | Every rule, on 500 random registers with digits 0–3, keeps the value | Both registers |
| `test_any_order_reaches_the_same_canonical_form` | 200 random registers, each settled in 10 random orders, always give the same canonical form | Both: **no sweep order is needed in either system** |
| `test_phi_canonical_form_matches_encoder` | `from_int` agrees with `FibonacciEncoder.to_codeword` on all 144 levels | φ |
| `test_add_exhaustive` | Every pair of 6-digit registers adds correctly and settles | Both |
| `test_dot` | `dot` gives the exact dot product with one addition per 1 digit of each weight | Both |
| `test_phi_carry_runs_down_binary_does_not` | `[0,0,2]` splits to `[1,0,0,1]` (6 = 5 + 1); binary carries to `[0,0,0,1]` | — |
| `test_phi_needs_more_rewrites_than_binary_for_the_same_values` | Over every sum of two values below 55, φ fires more rules, on average and in the worst case | Binary does less work per sum |
| `test_parallel_rounds_self_timed_phi_against_clocked_binary` | Over every sum of two values below 144: φ settles in 2.7 rounds on average (8 at worst), binary in 2.1 (8 at worst); binary's worst case ÷ φ's average > 2.9 | The "about 3×" figure |

```python
def test_parallel_rounds_self_timed_phi_against_clocked_binary(self):
    stats = {}
    for kind in KINDS:
        rounds = []
        for a in range(144):
            for b in range(144):
                out, r = normalize_rounds(pour(from_int(a, kind), from_int(b, kind)), kind)
                assert value(out, kind) == a + b and is_canonical(out, kind)
                rounds.append(r)
        stats[kind] = (sum(rounds) / len(rounds), max(rounds))
    assert 2.6 < stats[PHI][0] < 2.8 and stats[PHI][1] == 8
    assert 2.0 < stats[BINARY][0] < 2.2 and stats[BINARY][1] == 8
    assert stats[BINARY][1] / stats[PHI][0] > 2.9
```

Settling rounds over every pair of values below the level count:

| Levels | φ rounds (average / worst) | Binary rounds (average / worst) |
|---|---|---|
| 21 | 1.8 / 6 | 1.2 / 5 |
| 55 | 2.3 / 7 | 1.7 / 6 |
| 144 | 2.7 / 8 | 2.1 / 8 |
| 610 | 3.2 / 11 | 2.5 / 10 |

The test fixes the 144-level row; the other rows come from the same measurement run separately.

### `TestCorruption`

| Test | What it checks | Result (8 digits) |
|---|---|---|
| `test_binary_never_detects_a_flip` | Every single flip of every 8-bit register is still a valid number | 0% detected |
| `test_phi_detects_only_zero_to_one_flips` | Exhaustive single flips; only 0→1 flips are ever caught | 40.9% detected |
| `test_normalizing_keeps_the_corrupted_value` | Settling a corrupt register gives a valid register worth original + F(i) | **Normalizing hides the error rather than fixing it** |
| `test_correction_candidates` | The original is always a candidate; `[1,1,1]` → one fix, `[0,1,1,0]` → two | 8.6% of flips correctable |

```python
def test_normalizing_keeps_the_corrupted_value(self):
    places = place_values(PHI, 8)
    for word in canonical_words(PHI, 8):
        for i, flipped in single_flips(word):
            if flip_detected(flipped, PHI):
                out, _ = normalize(flipped, PHI)
                assert is_canonical(out, PHI)
                assert value(out, PHI) == value(word, PHI) + places[i]
```

The whole suite, including the existing `test_core.py`, gives 60 passed and 1 skipped (fp16 needs CUDA).

---

## 4. Theory docs updated

**`.docs/phi/phi.md`**
- **The Clock is Convergence:** adds the completion signal. A settling register still holds a `11` or a 2, and a settled one holds neither, so "done" is the same local scan as the error check. It also adds the measured rounds, with both limits: binary can also be self-timed, and carry-lookahead adders settle in about log₂(width) gate delays.
- **Error Detection is Free:** replaces "self-correcting by applying the carry rule" with the tested behaviour. The carry rule keeps the corrupted value; 40.9% of flips are detected and 8.6% are correctable.
- **Arithmetic from the Single Identity:** corrects "carries propagate strictly upward"; the doubling rule sends one carry down.
- **Dual cost table:** adds three rows: settling, done signal and integrity check.
- **Switching Strategy:**
  - Conversion to the matrix pair is O(width), not "the cost of one normalization".
  - A new subsection, *Self-timing in the switching strategy*, places self-timing on the 1D side only. Value work runs self-timed in 1D; composition work runs clocked in the matrix pair, whose entries are ordinary integers multiplied on binary hardware.

**`.docs/phi/matrix.md`:** adds the same three rows to its dual cost table.

---

## 5. Notebook `.docs/register_settling.ipynb`

The question: **where does the self-timed φ-register save time in phi-prune's own workload?**
It runs on CPU in about a minute.

### Cell 0: install

```python
import importlib.util, subprocess, sys
if importlib.util.find_spec("phi_prune.register") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "git+https://github.com/ezexe/phi-prune.git@claude/kind-hopper-juyjmk"])
```

Installs the package from this branch if `phi_prune.register` isn't importable; point it at `main` once the branch is merged.

### Cell 1: one pruned, encoded conv layer

```python
try:
    import torchvision
    model = torchvision.models.resnet18(weights="DEFAULT").eval()
    ...
if model is not None:
    pruned, masks = prune(model)
    weight = dict(pruned.named_parameters())[LAYER + ".weight"].data
    mask = masks[LAYER + ".weight"]
else:
    u = torch.rand(256, 128, 3, 3) - 0.5
    weight = -torch.sign(u) * torch.log1p(-2 * u.abs())   # Laplace
    mask = phi_mask(weight)
```

Uses ResNet-18's `layer3.0.conv1` (256 output channels, fan-in 1152) with ImageNet weights, φ-pruned.
Without network access it falls back to Laplace-distributed synthetic weights of the same shape and prints which one it used.

> Saved run: `synthetic Laplace weights (URLError: no pretrained weights)` · `layer3.0.conv1: 256 output channels, 128 kept, fan-in 1152`

### Cell 2: activations

```python
patches = F.unfold(seen[0], conv.kernel_size, padding=conv.padding, stride=conv.stride)
...
acts = [np.round((p / p.max() * 255).clamp(0, 255).numpy()).astype(int) for p in pick]
```

Hooks the layer's real input (CIFAR-10 test images at 224 px, or random images if CIFAR-10 can't be downloaded) and cuts it into 3 × 3 patch vectors.
Four patches are quantized to 0–255.
The synthetic fallback uses half-normal activations.

> Saved run: `half-normal synthetic activations` · `nonzero activations per patch: [1144, 1145, 1148, 1150] of 1152`

### Cell 3: the dot products in both registers

```python
def run(n_digits):
    enc = FibonacciEncoder(n_digits)
    encoded, scale, _, offset = enc.encode_tensor(weight, mask, return_offset=True, axis=0)
    ...
        levels = enc.levels(encoded[c], scale[c], offset[c])
        for xs in acts:
            expected = int(np.dot(levels, xs))
            for kind in KINDS:
                acc, rounds = dot(levels, xs, kind)
                assert sum(d * p for d, p in zip(acc, place_values(kind, len(acc)))) == expected
```

Encodes per output channel and takes 8 kept channels × 4 patches, 32 dot products in all.
Each runs through both accumulators and is checked against the exact sum.

> Saved run:
> ```
>    phi:    3495 additions per dot product, 3.44 rounds per addition on average, 15 at worst
> binary:    4132 additions per dot product, 2.45 rounds per addition on average, 17 at worst
> ```

### Cell 4: cost per dot product, four ways to time it

```python
def ladder_rounds(x, n_digits):
    """Rounds to build x·F(0) .. x·F(n_digits-1) in φ-registers by the ladder."""
    ...
rows = {
    "φ, self-timed": sum(res[PHI]["rounds"]) / n + ladder_per_dot,
    "binary, self-timed": sum(res[BINARY]["rounds"]) / n,
    "binary, clocked ripple": res[BINARY]["adds"] / n * width,
    "binary, clocked lookahead": res[BINARY]["adds"] / n * math.ceil(math.log2(width)),
}
```

- **φ, self-timed:** the rounds each addition actually took, plus the ladder that builds `x · F(k)`, shared across the 128 kept output channels.
- **Binary, self-timed:** an asynchronous adder that signals when its carries are done.
- **Binary, clocked ripple:** each addition waits the full 26-bit accumulator width.
- **Binary, clocked lookahead:** each addition waits ⌈log₂ 26⌉ = 5, the closest of the four to a CPU's adder.

> Saved run:
>
> | Timing | Rounds per dot product | Relative to self-timed φ |
> |---|---|---|
> | φ, self-timed | 12,267 | 1.00× |
> | binary, self-timed | 10,114 | 0.82× |
> | binary, clocked ripple | 107,438 | 8.76× |
> | binary, clocked lookahead | 20,661 | 1.68× |

### Cell 5: the low end, fewer digits

Reruns the same layer at 6, 8 and 10 Fibonacci digits.

> Saved run:
>
> | Digits | Levels | φ adds | Binary adds | φ self-timed | Binary self-timed | Binary lookahead | Binary ripple |
> |---|---|---|---|---|---|---|---|
> | 6 | 21 | 2,224 | 2,639 | 7,304 | 6,444 | 13,194 | 60,693 |
> | 8 | 55 | 2,860 | 3,428 | 9,739 | 8,402 | 17,142 | 82,282 |
> | 10 | 144 | 3,495 | 4,132 | 12,267 | 10,114 | 20,661 | 107,438 |

The ordering is the same at every width.

### Cell 6: integrity of the accumulator

Flips every bit of every settled accumulator from cell 3 in turn.

> Saved run: `phi: 45.3% of 1034 single-bit flips detected` · `binary: 0.0% of 733 single-bit flips detected`

### Cell 7: conclusions

These are written from the saved run, which used the synthetic fallback.

---

## 5b. Transformer check: `scripts/transformer_check.py`

This is a pre-registered test of the ideas on a small transformer.
- **Model and data:** a character-level GPT (4 layers, width 128, MLP width 512) learns Python's standard library source on CPU.
- **Runs:** 3 seeds, each with 1500 dense steps, then 300 mask-aware fine-tune steps per pruning method.
- **Masks:** pruning removes whole MLP hidden neurons.
- **Metric:** bits per character (bpc) on held-out files; lower is better.
- **Raw results:** [`results/transformer_check_low.json`](results/transformer_check_low.json), log [`results/transformer_check_low.log`](results/transformer_check_low.log).

It also uses a new package feature: `adjacency_dp_density(scores, density)` and `phi_mask(..., density=)` hit a target density below the 50% cap, by subtracting a bisected threshold from the scores before the DP.

### Pruning (mean of 3 seeds, dense 1.9985 bpc)

| Method | Density | One-shot | Fine-tuned |
|---|---|---|---|
| φ | 0.494 | 2.6533 | 2.0689 |
| φ, neurons reordered | 0.500 | 2.7236 | 2.0527 |
| 2:4 | 0.500 | 2.6254 | 2.0530 |
| heaviest 50% | 0.500 | 2.7236 | 2.0527 |
| random 50% | 0.500 | 2.6885 | 2.0770 |
| φ at 25% | 0.250 | 3.5714 | 2.1801 |
| 1:4 | 0.250 | 3.4960 | 2.1795 |
| heaviest 25% | 0.250 | 3.6058 | 2.1796 |

Seed-to-seed spread is about 0.024 bpc, mostly from how good each seed's dense model is.
Comparing methods within each seed (paired) is much sharper:

| Within-seed difference (fine-tuned bpc) | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| φ − 2:4 | +0.0166 | +0.0131 | +0.0179 |
| φ − heaviest 50% | +0.0190 | +0.0147 | +0.0148 |
| φ reordered − heaviest 50% | 0.0000 | 0.0000 | 0.0000 |
| φ at 25% − 1:4 | +0.0009 | +0.0005 | +0.0003 |
| φ at 25% − heaviest 25% | −0.0016 | +0.0006 | +0.0024 |

### Encoding: 1 digits per weight, all Linear layers quantized per output channel

| Levels (bits) | bpc | Binary | Zeckendorf | CSD |
|---|---|---|---|---|
| 16 (4.00) | 2.0279 | 2.000 | 1.695 | 1.832 |
| 21 (4.39) | 2.0132 | 2.220 | 1.862 | 2.000 |
| 32 (5.00) | 2.0055 | 2.500 | 2.098 | 2.193 |

These counts are in offset form, `(w − min)·scale`, the form the package uses.
In sign-magnitude form at 21 levels, Zeckendorf needs 1.139 against 1.258 for CSD and 1.296 for binary.
The bpc depends only on the number of levels: the 6-digit Fibonacci grid *is* 21 uniform levels, because every integer 0..20 has a Zeckendorf form.

### Verdicts (the criteria are in the script's docstring, written before any run)

| Criterion | As registered | Paired reading |
|---|---|---|
| K1: φ within σ of 2:4 at 50% | survives (+0.016 against σ 0.025) | **φ is worse in all 3 seeds** (+0.013 to +0.018); the unpaired σ, as registered, was too lenient |
| K2: φ at 25% beats 1:4 by more than σ | **killed** (−0.0006) | the three 25% masks are indistinguishable after fine-tuning; the pattern doesn't matter at 25% |
| K3: φ reordered within σ of the heaviest 50% | survives | identical masks, 0.0000 in every seed: reordering turns the φ rule into "keep the heaviest half" |
| K4: Zeckendorf has fewer 1 digits than CSD at 21 levels | **survives** | 1.862 against 2.000 (−7%), and −16% against binary; the same ordering at 16 and 32 levels |

What this says:
- **The φ rule's cost at 50% comes from neuron order.** The no-neighbours rule in arbitrary neuron order costs about 0.015 bpc. Reordering removes the cost entirely, but then the rule is simply "keep the heaviest half". The rule itself doesn't add anything to a transformer's MLP.
- **Going low doesn't help the mask.** At 25% every pattern recovers to the same loss after fine-tuning, so φ's extra choice there isn't an advantage in practice.
- **The digit saving is the result that survives.** Zeckendorf codes need fewer 1 digits per weight than both binary and CSD on real trained weights, so a shift-and-add multiply needs fewer additions. Two costs stand against it: storing the digits takes more bits than storing the level index (so store the index), and each φ addition settles in more rounds (section 5).
- **The scale is small:** a 0.8M-parameter model, 1500 steps, CPU. Read these as directions, not sizes.

---

## 6. Where this leaves the claims

**Holds up**
- **Fewer additions:** Zeckendorf levels have fewer 1 digits than binary, giving 15% fewer partial products per dot product. This is the low-end advantage, and it doesn't depend on timing.
- **Self-timed against clocked:** a self-timed φ accumulator beats a clocked binary one, by 1.7× against lookahead and 8.8× against ripple (in rounds).
- **One scan, three jobs:** the same no-`11` scan signals "done", checks integrity and frames self-delimiting streams. Binary needs extra carry-completion wires to know it's done, and can't detect a flip at all.
- **Free integrity:** 41% of single flips detected in 8-digit registers, 45% in the accumulators; 8.6% correctable.

**Doesn't hold up**
- **"Self-correcting":** normalizing keeps the corrupted value and hides the error. Check before settling.
- **"No sweep order" is not φ-only:** binary carries are order-independent too.
- **Against self-timed binary, φ loses:** 0.82×. Binary additions settle in fewer rounds, which outweighs φ's fewer additions.
- **Per sum, φ does more work:** it fires more rules, and its carries spread both up and down.

**Still open**
- Run the notebook with torchvision's weights and CIFAR-10 activations, for example on Colab, to replace the synthetic numbers.
- Rounds are a model of the rewrite rules, not gate delays. A gate-level or FPGA model of both accumulators would turn this into a hardware result.
- Signed-digit binary (CSD, about 1/3 of digits nonzero) is the stronger binary competitor for counting additions and hasn't been compared.
- The density-targeted φ mask from section 0, against a 1:4 baseline.
