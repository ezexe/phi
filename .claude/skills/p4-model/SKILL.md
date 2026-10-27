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
2. **The plugin.** `phi_model` finds the vlds plugin (newest version) under `~/.claude/plugins`, `$VLDS_PLUGIN_ROOT`, or a checkout of ezexe/p4 (formerly ezexe/prompty-prompter-pioneer-puppeteer) beside the project. If it is not found, clone the marketplace and pass `--plugin <clone>/Ibiza/plugins/vlds`.
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

## Rules

- A notice is cleared only by a training that succeeded for its component. Never delete `retrain-needed.tsv` by hand.
- The mode is the owner's ruling: change it only when the user asks (`init` sets `shadow` because asking for the setup is asking for it).
- The models train on what git keeps in the store. Do not widen that by editing the store's `.gitignore` without the user's word.
- Report every number with what it was measured against; no claim that a model "works" without a held-out figure.
