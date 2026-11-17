# Contributing to phi

phi is the Python package `phi-prune`: φ-pruning, the quantization and Fibonacci coding of pruned weights, and the ranking models the p4 plugins train in a project's store.
This page says how to set it up, run its tests and commit to it.

## Set up

phi needs Python 3.11 or later.
From a clone of the repository:

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -e ".[dev]"
```

On Windows the second step is `.venv\Scripts\activate`.
torch's CPU wheels are all the tests need of torch; without the third step, the fourth installs torch from PyPI instead.
`pip install -e ".[dev,export]"` adds onnx, onnxruntime and onnxscript, which `phi_prune.export.export_onnx` and `tests/test_export_onnx.py` need.
`pip install markdown nbconvert` adds what `tests/test_site.py` needs.
The tests of `tests/test_p4model_recall.py` need the p4-phi plugin of ezexe/p4, found through `P4_RECALL_PLUGIN_ROOT` or a checkout of ezexe/p4 beside this one.
A test whose packages or plugin are missing is skipped, and the run prints the reason.

## Run the tests

```bash
python -m pytest
```

That one command runs the whole suite under `tests/`, with the options `[tool.pytest.ini_options]` of `pyproject.toml` gives it: the duration of every setup, call and teardown, and every outcome but a pass, each skip with its reason.
`test_the_whole_run_fits_its_budget` in `tests/test_suite.py` runs after every other test and fails a run that took longer than `budget_seconds`, 60 seconds, the budget set in the same table.
On a CI runner, where `CI` is set to anything but `0` or `false`, nothing fails on time, and pytest's last line gives the time.
While working on one module, run its file alone, such as `python -m pytest tests/test_core.py`.

The tests workflow, `.github/workflows/tests.yml`, runs `python -m pytest` on every push to `main` and every pull request, twice: on the floors `pyproject.toml` admits, Python 3.11 with torch 2.4.1 and numpy 1.24.0, and on Python 3.15 with the newest torch and numpy.
Its export job installs the export extra on Python 3.14, the newest Python onnxruntime has wheels for, and fails when `tests/test_export_onnx.py` skips.

## Commit

- One change per commit, with the tests that prove it in the same commit.
- The subject names the module, file or area the commit touches, a colon, then what is true once it lands, such as `encoding: encode_tensor rounds to the nearest level instead of searching the grid`.
- The body says why, and what was run to prove it and what that run showed.
- A commit that closes an issue says `Closes #N` in its body, and one that only works towards it says `Refs #N`.

## Where things are

- `phi_prune/`: pruning, the masks, quantization, the Fibonacci code, the checks of quantized weights, export, sparse deltas, the Zeckendorf and binary register model of `register.py` that the code builds on, and the `phi-prune` command; `p4.py`, `p4rank.py`, `p4serve.py` and `p4model.py` forward the older module names to the packages below.
- `phi_store/`, `phi_rank/`, `phi_serve/` and `phi_model/`: the reader of a `.claude/p4` store and the ranking models the p4 plugins train and serve.
- `tests/`: the suite.
- `docs/`: the documentation, published to https://ezexe.github.io/phi/ on every push to `main`.
- The measurement scripts, notebooks and result files live in [ezexe/p4-phi-experiments](https://github.com/ezexe/p4-phi-experiments), and a page that shows a measured figure names the result file it comes from there.
