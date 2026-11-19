# phi

φ-pruning keeps whole output channels of a layer and never keeps two adjacent ones, the (1,∞)-RLL constraint.
phi holds that pruner, the Fibonacci codes it writes quantized weights and model deltas in, and a Zeckendorf register modeled beside a binary register.
Beside them, `phi_store`, `phi_rank` and `phi_model` read the store the p4 plugins write in a project, and train and check its ranking models.

- **[Quick start](https://ezexe.github.io/phi/quickstart.html):** install, prune a model, check it, encode it and run the tests.
- **[Documentation](https://ezexe.github.io/phi/):** every page, published from `docs/`.

## Pruning

- `phi_prune.prune` gives each eligible Conv2d and Linear weight a φ mask, the heaviest set of output channels with no two adjacent, by their L1 norms.
- The model's last weight layer, the classifier head in standard architectures, stays dense unless `prune_head=True`.
- `pattern="2:4"` keeps instead the 2 heaviest of every 4 consecutive channels, the mask the φ mask is measured against.
- `finetune` trains with the masks held fixed, so pruned weights stay zero.
- `check`, and the `phi-prune check` command on a saved checkpoint, verify every mask against its pattern and that every pruned weight is zero.
- Pages: [Pruning a model](https://ezexe.github.io/phi/pho/pruning.html), [The masks](https://ezexe.github.io/phi/pho/masks.html) and [Measurements](https://ezexe.github.io/phi/pho/benchmarks.html).

## Number codes and the `.phid` delta

- `FibonacciEncoder` quantizes kept weights onto the integers 0 to F(n+2) − 1, per tensor or per output channel, and writes each level as its Zeckendorf representation.
- `phi_prune.export.export_bitstream` writes a model's levels as Fibonacci codewords after one JSON header line, and `load_bitstream` reads them back.
- `cassini_check` runs the minimal-form check on each kept weight's digits, which catches a digit flipped from 0 to 1 beside a 1.
- `phi_prune.gaps` writes the positions of kept entries as Fibonacci-coded gaps, and `gap_bits` prices them against Elias gamma, Elias delta, a fixed-length code and the entropy.
- `phi_prune.delta` writes a model's delta between two checkpoints as a `.phid` file: the top-k changes of each tensor, their positions as gaps and each value as a sign digit and a Fibonacci-coded level.
- Pages: [Quantization and the code](https://ezexe.github.io/phi/pho/encoding.html) and [Fibonacci codes](https://ezexe.github.io/phi/phi/codes.html).

## The register model

- `phi_prune.register` holds a Zeckendorf register and a binary register, each a list of digits that local rewrite rules bring to its normal form.
- The rules can fire in any order and every order ends at the same normal form.
- The module runs phi-prune's own multiply–accumulate step in each register, so the two are measured on the same sums.
- Pages: [The register model](https://ezexe.github.io/phi/register.html), with the Zeckendorf register, the binary register, computing F(n), the Fibonacci Q-matrix and related work beside it.

## The models of a store

- A store is a repository's `.claude/p4` directory, written by the p4 plugins of [ezexe/p4](https://github.com/ezexe/p4): p4-phi keeps its entries under `phi/store` and its training files under `phi/models`, and p4-temportal keeps memory notes under `temportal/memory`.
- `phi_store` reads a store as training records, each a text with its session, its time and its weight, and `split` holds out every record timed at or after a cutoff.
- `phi_rank` trains an asymmetric dual encoder that scores a store's documents against a session's query, by feature hashing of words, word bigrams and character trigrams, with one head for queries and one for documents.
- It pretrains contrastively with in-batch negatives on the text files of the store that its `.gitignore` does not exclude, then fine-tunes on the store's labelled query groups by Localized Contrastive Estimation.
- The dual encoder learns in rounds: git LFS keeps a full checkpoint, and a later round is warm-started from the model the checkpoints rebuild and ships an incremental checkpoint, a `.phid` file that holds the largest 10% of the entries the round changed.
- A round ships a full checkpoint again when the cut costs more than 0.02 of the training query groups' R-precision, when eight incremental checkpoints follow the full one, or when the pretrained base has changed.
- `phi_rank.tree` builds an ensemble of shard models, K small rankers at the first K store addresses, which answers by dynamic classifier selection or by the mean of every member's z-scored scores.
- `phi_model` is model monitoring and on-demand retraining for a store: `python -m phi_model` takes `check`, `init`, `train`, `test`, `update` and `mode`, over the p4-phi plugin's logistic regression ranker and the dual encoder.
- `mode` writes the `recall` section of the store's `p4.json`: `shadow` trains and logs retrieval's lines without adding them to a session's context, `active` adds the rule and fact lines, and `off` trains nothing and adds nothing.
- The dual encoder is an experiment that retrieval does not serve: `check`, `init`, `train` and `update` leave it untrained and find nothing of it stale unless `VLDS_PT_RANKER=1` is set, and `test` evaluates it either way.
- The [p4-model skill](.claude/skills/p4-model/SKILL.md) gives the routine, and [ezexe/p4-phi-experiments](https://github.com/ezexe/p4-phi-experiments) holds the measurements.

## The repository

- [CONTRIBUTING.md](CONTRIBUTING.md), [SUPPORT.md](SUPPORT.md), [SECURITY.md](SECURITY.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- [CITATION.cff](CITATION.cff), which GitHub shows as "Cite this repository".
- [LICENSE](LICENSE): Apache-2.0.
