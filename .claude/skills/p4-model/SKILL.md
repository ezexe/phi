---
name: p4-model
description: Build, check, test, train and update the p4 models of a repository's .claude/p4 vlds store (the pool ranker, the dispatch model, the .pt ranker), answer the retrain notices other repositories' sessions leave in models/retrain-needed.tsv, and set the store to realtime shadow learning or active. Use when asked to init/update/check/test/train "the p4 model", to point it at a repository's .claude/p4, or when a session reports "p4 model: N retrain notice(s) open".
---

# p4-model — the model checkpoint for a repository's .claude/p4 store

This repository (ezexe/phi) is where a p4 store's models are built and kept current. Sessions in the store's own repository only *use* the models: on any unexpected model failure the vlds plugin falls back (the JSON ranker, the children, the operator), appends one row to the store's `models/retrain-needed.tsv`, and carries on. Those sessions never debug, retrain or realign on a model error. That work is this skill's, done here, through `phi_model`.

## What a store's models are

| Model | File | Built by | Used by |
|---|---|---|---|
| pool ranker (JSON) | `models/pick-model.json` | the plugin's `scripts/pick_model.py train` | the model road's picks, when no `.pt` server answers |
| dispatch model (JSON) | `models/dispatch-model.json` | the plugin's `scripts/dispatch_model.py train` | the prompt hook (`dispatch-model: shadow \| on`) |
| `.pt` ranker | the chain: `models/ranker-anchor.pt` and `models/ranker.rNNN.phid` (git LFS), `models/ranker.json`, `models/ranker-oof.json`; `models/ranker.pt` is each machine's ignored build of it | `phi_rank` (a small base, from the plugin's shared p4 base, pretrained on every text file git keeps in the store; each later round fine-tuned on the labelled pools and shipped as a φ-delta) | `phi_serve` on localhost; its score is a feature of the JSON ranker, kept only when it raises the held-out r-precision |

Every turn close in the store's repository already retrains what is behind, in the background (the `.pt` ranker only where `models/.ranker-python` or `VLDS_RANKER_PYTHON` names an interpreter with this package), and stops retrying a model while a notice about it is open. This skill is the checkpoint for everything that path cannot do on its own: the first build, a failed training, a notice, a change of mode, an evaluation.

## Before anything

1. **The target.** The user names a repository (a local path) or a store repository (ezexe/AudioPlugin-p4, ezexe/p4-p4). `--project` takes the repository (its `.claude/p4` is used) or the store's root directly. In a cloud session, attach and clone the repository first.
2. **The plugin.** `phi_model` finds the vlds plugin (newest version) under `~/.claude/plugins`, `$VLDS_PLUGIN_ROOT`, or a checkout of ezexe/p4 (formerly ezexe/prompty-prompter-pioneer-puppeteer) beside the project. If it is not found, clone the marketplace and pass `--plugin <clone>/lbiza/plugins/vlds`.
3. **This package.** `pip install -e .` here (torch included). The `.pt` ranker trains and serves with this interpreter. `init` (and `update`, when it is missing) writes it to the store's `models/.ranker-python`: a local, untracked file the plugin's hooks read to start the server and train at turn close. Without it, or `VLDS_RANKER_PYTHON`, the store's sessions use the JSON ranker alone and `check` says so. A tracked file never names an interpreter: a shared store must not be able to point another machine at an executable.

## The commands

All take `--project <repo or store>`; `--plugin` and `--now` are optional.

```bash
python -m phi_model check  --project <…>            # what stands, what is behind, open notices; exit 1 = owed
python -m phi_model update --project <…>            # the checkpoint: check, retrain what is owed, answer notices, check again
python -m phi_model init   --project <…> --mode shadow   # first time: seed git/.gitignore/LFS, train all three, set the mode
python -m phi_model train  --project <…> --only json,dispatch,ranker [--minutes 20]
python -m phi_model test   --project <…>            # held-out evals of all three, and a live server check
python -m phi_model mode   --project <…> shadow|active|off
```

- **`shadow`** is realtime shadow learning: `pool-road: model-shadow` (the children pick and fold every pool, the model's picks are logged beside theirs in `models/shadow.tsv`) and `dispatch-model: shadow` (the dispatch model proposes and logs, the operator decides). Every turn close retrains from the new labels. This is the default for `init`.
- **`active`** lets the models decide: `pool-road: model` (the operator reviews the model's picks; one session in `pool-audit` runs the children), `dispatch-model: on`. A `pool-audit:` the owner set is kept; `pool-audit: 5` is written only where none is.
- **`off`** returns the store to the children and the operator, the models left on disk.

## The checkpoint routine

When asked to update, or when a session reports open notices:

1. `check`. Read `owed`, the notices and any `log_errors`.
2. `update`. It trains only what is owed, answers the notices of each model it rebuilt (appending them to `models/retrain-resolved.tsv`; `retrain-needed.tsv` is append-only and never rewritten), and checks again. A model with nothing to learn yet is no failure and answers no notice. A training the turn close is already running is reported as busy; run `update` again after it.
3. If a training failed, read its output: printed by `update`/`train` for the trainings it runs, in `models/train.log` or `models/ranker.log` for the ones a turn close started. Fix the cause **in this repository** (`phi_rank`, `phi_serve`, `phi_model`, `phi_prune`) or report it as a plugin bug for ezexe/p4. Do not edit the store's sessions to make a model train.
4. `test` when a model changed materially (a new base, a mode change, a fix). Report the held-out numbers against the previous ones and against term overlap. `test` writes nothing the store serves: the `.pt` evaluation never saves the pretrained base (a missing one is built in memory, up to `--minutes`), and its server check runs under its own port file beside any live server.
5. In the store's repository, commit `models/` (the anchor and the ranker's `.phid` deltas go through git LFS by the store's `.gitattributes`; check `git lfs ls-files` shows them; `models/ranker.pt` is ignored, and `check` asks to untrack it where an older store tracked it) and the recall config if the mode changed, and push. In a store repository attached to a session, that is a branch and a pull request, merged when the user has asked for merges.

## The balanced model: what the measurements settle

This section turns phi's measurements into one configuration. Every figure is from `.docs/p4/nested-store.md` (AudioPlugin-p4, 22 labelled pools, held-out r-precision; k = 4 over 2 seeds, k = 8 over 1). Re-derive a figure before relying on it: rerun `scripts/session/nested_suite.py` on the store it is about, since the stores keep growing.

**What picks.** The JSON ranker (`pick-model.json`) picks. It scored 0.380, and no `.pt` arm comes close on its own (0.15–0.26). The `.pt` models feed it a feature, and only when that feature raises its leave-one-out r-precision. The plugin already enforces that.

**Effort tiers.** The answer's cost is chosen by how much the task asks:

| Effort | Who answers | Measured | Built |
|---|---|---|---|
| low | the JSON ranker alone | 0.380; no `.pt` server needed | yes |
| medium | the JSON ranker + one node's score, the node chosen by the φ pruning mask (the Q-matrix route) | picks the best node in 50% of pools at k = 4 and 32% at k = 8 (chance: 25% and 12.5%); as a feature 0.392 at k = 4 (mean), 0.357 at k = 8; one node answers in about 9 ms | yes: `effort: medium` |
| high | the JSON ranker + the Zeckendorf tree's k = 4 ensemble | 0.235 held-out against one large model's 0.220, on both seeds; about 33 ms | yes: `effort: high` |
| background | the grown tree (every node full size, k× the parameters), retrained off the hot path; its root feeds the JSON ranker | 0.258 held-out at k = 8; as a feature 0.387 and 0.397, the only `.pt` that helped the JSON ranker at both sizes | no |

**Updates.** A new pool goes to one node, never to the whole tree. Either the newest node takes it (time-routed) or the node the mask routes to does (matrix-routed).
- Forgetting: time-routed lost 0.000 on the oldest pools, matrix-routed φ 0.267 and 0.173, against 0.356 and 0.204 for one large model updated on everything.
- Message size: one node's φ-delta is 23–73 KB, against 133 KB for one large model, and the address adds 2n bits.
- Merging back: the tree is merged back (a node with two children starts from their mean) at the slower rebase cadence: 8 rounds, or a new base.

**Pretraining.** The shared p4 base helps the single ranker (0.227 against 0.206 from scratch), but it does not fit a node: the base is 2^15 rows and a k = 4 node 2^13. So a leaf starts from scratch, and a node with children starts from their mean. The nodes pretrain in parallel, each on its own shard: 90 s at k = 4 and 45 s at k = 8, against 361 s for one large model on the same budget.

**Size: 4 nodes.** k = 4 is the only size at which the Zeckendorf ensemble beat one large model on both seeds. At k = 8 it fell behind (0.207 against 0.222). The owner reads 4 nodes as the 4 positions of the Q matrix [[n1, n2], [n3, n4]]. Whether the win comes from that shape or from the shard size is not settled. Testing k = 2, 3, 5 and 6 on the same store would tell.

**The corpus.** The store's run logs stay in pretraining, on the owner's ruling. They lowered the standalone `.pt` ranker from 0.227 to 0.200. Weighting the store's decisions over its logs is parked until the store has grown.

**Not yet true, so not yet claimed.**
- A routed node alone ranks below every ensemble (0.190 against 0.235 at k = 4). Medium effort trades quality for speed and small updates, so it is never the default for a high-effort task.
- The φ adjacency rule means nothing about the data in this ranker: hashing scatters features over the table's rows at random. The routing works because the masks keep each node's heaviest rows.

**Where the tiers live.**
- `python -m phi_rank.tree train --p4 <store> --pairs <pairs.jsonl> --k 4 --minutes 6` builds the tree: `models/tree/<address>.pt` and `models/tree.json`.
- `phi_serve` answers `/score` by the request's `effort`: `medium` is the routed node (`source: tree:medium:<address>`), `high` is the ensemble (`source: tree:high:4`). Any other effort, or no tree on disk, is the single ranker (`source: ranker`). The server reloads the tree when `tree.json` changes.
- The tree's training also writes `models/tree-oof.json`: every labelled pool scored at medium and at high by fold trees that never fine-tuned on it, and each effort's held-out r-precision.
- The JSON ranker fits its `pt` weight on the out-of-fold scores of the source the store's `pool-effort:` names: `tree-oof.json` for medium or high, `ranker-oof.json` otherwise (vlds 0.0.75). It keeps the weight only when it raises the leave-one-out r-precision, and asks `/score` at the effort it was fitted on. A change of `pool-effort:` takes effect at the JSON ranker's next `train`.
- The background tier (the grown tree) is not built.

## Rules

- A notice is cleared only by a training that succeeded for its component. Never delete `retrain-needed.tsv` by hand.
- The mode is the owner's ruling: change it only when the user asks (`init` sets `shadow` because asking for the setup is asking for it).
- The models train on what git keeps in the store. Do not widen that by editing the store's `.gitignore` without the user's word.
- Report every number with what it was measured against; no claim that a model "works" without a held-out figure.
