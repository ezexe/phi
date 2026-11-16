"""The suite itself: python -m pytest is the one command that runs it, here and in the tests workflow."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_tests_workflow_runs_the_one_command(pytestconfig):
    """python -m pytest takes its options from pyproject.toml, so the workflow prints what a run here prints: the
    duration of every setup, call and teardown, and every outcome but a pass, each skip with its reason."""
    assert pytestconfig.getini("testpaths") == ["tests"]
    assert pytestconfig.getini("addopts") == ["--durations=0", "--durations-min=0", "-ra"]
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert re.findall(r"run: (.*\bpytest\b.*)", workflow) == ["python -m pytest"]
