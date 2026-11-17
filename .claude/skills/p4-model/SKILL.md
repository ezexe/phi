---
name: p4-model
description: Check, test, train and update the models of a repository's .claude/p4 store, the p4-phi plugin's logistic regression ranker and phi's asymmetric dual encoder, answer the retrain notices sessions leave in phi/models/retrain-needed.tsv, and set the store's mode to shadow, active or off. Use when asked to init, update, check, test or train "the p4 model", to point it at a repository's .claude/p4, or when a session reports open retrain notices.
---

# p4-model: model monitoring and on-demand retraining for a repository's .claude/p4 store

This repository, ezexe/phi, is where a store's models are built and kept current.
Sessions in the store's own repository only use the models.
When the logistic regression ranker's file is unreadable, a session falls back to BM25, and when a hook raises, the session goes on; either way it appends one row to the store's `phi/models/retrain-needed.tsv`.
Those sessions never debug or retrain on a model error.
That work is this skill's, done here, through `phi_model`.

## What a store's models are

| Model | Files | Built by | Used by |
|---|---|---|---|
| logistic regression ranker | `phi/models/ranker.json` | the plugin's `scripts/p4.py train` | retrieval, which ranks a prompt's candidates by it, or by BM25 alone when it is missing |
| asymmetric dual encoder | the full checkpoint `phi/models/ranker-anchor.pt` and the incremental checkpoints `phi/models/ranker.rNNN.phid`, both in git LFS, and the sidecar `phi/models/ranker-pt.json`; `phi/models/ranker.pt` is each machine's ignored build of them | `phi_rank`: a small base pretrained on every text file git keeps in the store, then each round fine-tuned on the labelled query groups | nothing: it is an experiment |

The plugin's Stop hook retrains the logistic regression ranker in the store's repository when a round is due, unless `recall.learn` in `p4.json` is false.
This skill covers what that hook does not: the first setup, a failed training, an open notice, a change of mode and an evaluation.

## Before anything

1. **The target.** ezexe names a repository by its local path, or a repository that is itself a store. `--project` takes the repository, whose `.claude/p4` is used, or the store's root directly. In a cloud session, attach and clone the repository first.
2. **The plugin.** `phi_model` finds the p4-phi plugin: `--plugin`, then `$P4_RECALL_PLUGIN_ROOT`, then the newest of the copies under `~/.claude/plugins` and of a checkout of ezexe/p4 beside the project. If none is found, clone ezexe/p4 and pass `--plugin <clone>/lbiza/plugins/p4-phi`.
3. **This package.** Install it with `pip install -e .` here, torch included, or use the `.venv` that a p4 checkout's `scripts/make_venv.py` builds. Run every command below with that interpreter, `.venv/bin/python` or on Windows `.venv\Scripts\python.exe`, never a bare system `python`. A `python -m phi_model` started without numpy runs itself again in the first `.venv` it finds, this checkout's, the one beside it or a sibling `p4/`'s, and says so on stderr. On a machine with no torch build, `check` still checks the ranker, the labels and the notices, and says that it did not check the dual encoder.

## The commands

All take `--project <repo or store>`; `--plugin` and `--now` are optional.

```bash
python -m phi_model check  --project <…>            # what stands, what is stale, open notices; exit 1 when anything is stale
python -m phi_model update --project <…>            # check, retrain what is stale, answer notices, check again
python -m phi_model init   --project <…> --mode shadow   # first time: seed git, .gitattributes and .gitignore, train the logistic regression ranker, set the mode; VLDS_PT_RANKER=1 trains the dual encoder too
python -m phi_model train  --project <…> --only json,ranker [--minutes 20]
python -m phi_model test   --project <…>            # held-out evaluations of both models
python -m phi_model mode   --project <…> shadow|active|off
```

A mode writes `recall.inject` and `recall.learn` in `p4.json`:
- **`shadow`** sets `inject` off and `learn` true: retrieval computes and logs its lines and the usage labels say `shadow`, but none reaches the model, and the Stop hook trains. This is `init`'s default.
- **`active`** sets `inject` to `rules-facts` and `learn` true: the rule and fact lines reach the model, and the Stop hook trains.
- **`off`** sets `inject` off and `learn` false: nothing reaches the model, nothing trains, and the models stay on disk.

## The routine

When asked to update, or when a session reports open notices:

1. `check`. Read `owed`, the notices and anything marked `CORRUPT`.
2. `update`. It trains only what is stale, answers the notices of each model it rebuilt by appending them to `phi/models/retrain-resolved.tsv`, and checks again. `retrain-needed.tsv` is append-only and never rewritten. A model with nothing to learn yet is no failure and answers no notice.
3. If a training failed, read its output, which `update` and `train` print. Fix the cause in this repository, in `phi_rank`, `phi_model` or `phi_prune`, or report it as a plugin bug for ezexe/p4. Do not edit the store's sessions to make a model train.
4. `test` when a model changed materially, as a new base, a change of mode or a fix does. Report the held-out numbers against the previous ones and against BM25. `test` writes nothing the store serves: the dual encoder's evaluation never saves the pretrained base, and builds a missing one in memory, up to `--minutes`.
5. In the store's repository, commit `phi/models/`, and `p4.json` if the mode changed, and push. The checkpoints go through git LFS by the store's `.gitattributes`, which `git lfs ls-files` shows, and `ranker.pt` is ignored; `check` asks to untrack `ranker.pt` where a store tracks it. In a repository that is a store and is attached to a session, that is a branch and a pull request, merged when ezexe has asked for merges.

## The dual encoder

The dual encoder is an experiment, and retrieval has never served it.
`check`, `update`, `init` and `train` leave it untrained, and find nothing of it stale, unless `VLDS_PT_RANKER=1` is set.
Its measurements are in ezexe/p4-phi: `experiments/phi_depth/README.md` for the ensemble of `phi_rank.tree`, and `experiments/phi-p4-models` for the runs that compared it with the logistic regression ranker.
Before switching it on for a store, show that it raises the logistic regression ranker's held-out R-precision on that store.

## Rules

- A notice is cleared only by a training that succeeded for its model. Never delete `retrain-needed.tsv` by hand.
- The mode is ezexe's decision: change it only when ezexe asks, and `init` sets `shadow` because asking for the setup is asking for it.
- The models train on what git keeps in the store. Do not widen that by editing the store's `.gitignore` without ezexe's word.
- Report every number with what it was measured against, and claim no model works without a held-out figure.
