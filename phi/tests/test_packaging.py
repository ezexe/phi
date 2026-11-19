"""pyproject.toml agrees with the files around it: the tests workflow installs the floors it declares, the license it
declares is the one LICENSE holds, in the form its build requires, and the version it declares is the one CITATION.cff
cites and the release workflow tags."""
import pathlib
import re
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT.parent
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
    workflow = (REPO / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
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
    assert (ROOT / "LICENSE").read_bytes() == (REPO / "LICENSE").read_bytes(), "phi/LICENSE is a copy of LICENSE"
    assert PYPROJECT["build-system"]["requires"] == ["setuptools>=77"]


def test_the_build_reads_only_files_of_its_own_folder():
    """setuptools refuses a readme or a license file outside the folder of pyproject.toml, so phi/ holds both."""
    include = PYPROJECT["tool"]["setuptools"]["packages"]["find"]["include"]
    assert PROJECT["readme"] == "README.md"
    assert (ROOT / "README.md").is_file()
    assert include == ["phi_prune*", "phi_store*", "phi_rank*", "phi_model*"]
    assert all((ROOT / name.rstrip("*") / "__init__.py").is_file() for name in include)


def test_the_version_pyproject_declares_is_the_one_CITATION_cites():
    """A release raises both, and the release workflow tags what pyproject.toml declares."""
    cited = re.search(r'^version: "([^"]+)"$', (REPO / "CITATION.cff").read_text(encoding="utf-8"), re.M)
    assert cited and cited.group(1) == PROJECT["version"]


def test_the_release_workflow_tags_the_version_pyproject_declares():
    """It reads the version from phi/pyproject.toml, builds from phi/, runs the tests workflow first, and uploads to
    PyPI only where the repository variable PYPI_PUBLISH turns that on."""
    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "tomllib.load(open('phi/pyproject.toml', 'rb'))['project']['version']" in workflow
    assert "python -m build phi --outdir dist" in workflow
    assert "uses: ./.github/workflows/tests.yml" in workflow
    assert "workflow_call:" in (REPO / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert re.search(r"pypi:\n    needs: \[version, build, github-release\]\n    if: vars\.PYPI_PUBLISH == 'true'", workflow)
    name = PROJECT["name"].replace("-", "_")
    assert "dist/" + name + "-${{ needs.version.outputs.version }}.tar.gz" in workflow
