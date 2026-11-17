# phi

φ-pruning keeps whole output channels of a layer and never keeps two adjacent ones, the (1,∞)-RLL constraint.
phi holds that pruner, the Fibonacci codes it writes quantized weights and model deltas in, and a Zeckendorf register modeled beside a binary register.

- **[Quick start](https://ezexe.github.io/phi/docs/quickstart.html):** install, prune a model, check it, encode it and run the tests.
- **[Documentation](https://ezexe.github.io/phi/docs/index.html):** every page, published from `docs/`.

## Pruning

- `phi_prune.prune` gives each eligible Conv2d and Linear weight a φ mask, the heaviest set of output channels with no two adjacent, by their L1 norms.
- The model's last weight layer, the classifier head in standard architectures, stays dense unless `prune_head=True`.
- `pattern="2:4"` keeps instead the 2 heaviest of every 4 consecutive channels, the mask the φ mask is measured against.
- `finetune` trains with the masks held fixed, so pruned weights stay zero.
- `check`, and the `phi-prune check` command on a saved checkpoint, verify every mask against its pattern and that every pruned weight is zero.
- Pages: [Pruning a model](https://ezexe.github.io/phi/docs/pho/pruning.html), [The masks](https://ezexe.github.io/phi/docs/pho/masks.html) and [Measurements](https://ezexe.github.io/phi/docs/pho/benchmarks.html).

## Number codes and the `.phid` delta

- `FibonacciEncoder` quantizes kept weights onto the integers 0 to F(n+2) − 1, per tensor or per output channel, and writes each level as its Zeckendorf representation.
- `phi_prune.export.export_bitstream` writes a model's levels as Fibonacci codewords after one JSON header line, and `load_bitstream` reads them back.
- `cassini_check` runs the minimal-form check on each kept weight's digits, which catches a digit flipped from 0 to 1 beside a 1.
- `phi_prune.gaps` writes the positions of kept entries as Fibonacci-coded gaps, and `gap_bits` prices them against Elias gamma, Elias delta, a fixed-length code and the entropy.
- `phi_prune.delta` writes a model's delta between two checkpoints as a `.phid` file: the top-k changes of each tensor, their positions as gaps and each value as a sign digit and a Fibonacci-coded level.
- Pages: [Quantization and the code](https://ezexe.github.io/phi/docs/pho/encoding.html) and [Fibonacci codes](https://ezexe.github.io/phi/docs/phi/codes.html).

## The register model

- `phi_prune.register` holds a Zeckendorf register and a binary register, each a list of digits that local rewrite rules bring to its normal form.
- The rules can fire in any order and every order ends at the same normal form.
- The module runs phi-prune's own multiply–accumulate step in each register, so the two are measured on the same sums.
- Pages: [The register model](https://ezexe.github.io/phi/docs/register.html), with the Zeckendorf register, the binary register, computing F(n), the Fibonacci Q-matrix and related work beside it.

## The repository

- [CONTRIBUTING.md](CONTRIBUTING.md), [SUPPORT.md](SUPPORT.md), [SECURITY.md](SECURITY.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- [CITATION.cff](CITATION.cff), which GitHub shows as "Cite this repository".
- [LICENSE](LICENSE): Apache-2.0.
