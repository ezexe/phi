"""pyproject.toml agrees with the files beside it: the tests workflow installs the floors it declares, and the license
it declares is the one LICENSE holds, in the form its build requires."""
import pathlib
import re
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]


def release(version):
    """A version as three numbers, so that 3.11 and 3.11.0 compare equal."""
    numbers = [int(n) for n in version.split(".")]
    return tuple(numbers + [0] * (3 - len(numbers)))


def floor(name):
    """The lower bound pyproject.toml gives a dependency, such as 2.4.1 for torch>=2.4.1."""
    bounds = [re.fullmatch(rf"{name}>=([\d.]+)", spec) for spec in PROJECT["dependencies"]]
    return next(release(bound.group(1)) for bound in bounds if bound)


def test_the_tests_workflow_installs_the_floors_pyproject_declares():
    """One job of the tests workflow runs the oldest Python, torch and numpy that pyproject.toml admits."""
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    job = re.search(r'python-version: "([\d.]+)"\s+torch: "torch==([\d.]+)"\s+numpy: "numpy==([\d.]+)"', workflow)
    assert job, "a job of the tests workflow pins torch and numpy"
    python, torch, numpy = (release(version) for version in job.groups())
    assert python == release(re.fullmatch(r">=([\d.]+)", PROJECT["requires-python"]).group(1))
    assert torch == floor("torch")
    assert numpy == floor("numpy")


def test_the_license_pyproject_declares_is_the_one_LICENSE_holds():
    """An SPDX expression and its file, the form setuptools reads from version 77 on, which the build requires."""
    assert PROJECT["license"] == "Apache-2.0"
    assert PROJECT["license-files"] == ["LICENSE"]
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").split()[:4] == ["Apache", "License", "Version", "2.0,"]
    assert PYPROJECT["build-system"]["requires"] == ["setuptools>=77"]
