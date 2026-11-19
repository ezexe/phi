# phi-prune

φ-pruning under the (1,∞)-RLL constraint, Fibonacci coding of quantized weights, and the ranking models the p4 plugins train in a project's store.

This folder is the Python project of [ezexe/phi](https://github.com/ezexe/phi): `pyproject.toml`, the packages `phi_prune`, `phi_store`, `phi_rank` and `phi_model`, and their tests.

- **[Overview](https://github.com/ezexe/phi/blob/main/README.md):** what each package does.
- **[Quick start](https://ezexe.github.io/phi/quickstart.html):** install, prune a model, check it, encode it and run the tests.
- **[Documentation](https://ezexe.github.io/phi/):** every page of the site.

## Install

```bash
pip install "git+https://github.com/ezexe/phi#subdirectory=lbiza"
```

To work on the code, clone the repository and install this folder in editable mode with the `dev` extra, then run the tests from this folder:

```bash
pip install -e ".[dev]"
python -m pytest
```

## Releases

The [release workflow](https://github.com/ezexe/phi/blob/main/.github/workflows/release.yml) tags and releases a push to `main` whose `version` in `pyproject.toml` has no tag yet.
Each [release](https://github.com/ezexe/phi/releases) holds the sdist and the wheel built from this folder.
