"""The suite itself: python -m pytest is the one command that runs it from phi/, here and in the tests workflow, and the whole
run fits a time budget on this machine."""
import os
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_the_tests_workflow_runs_the_one_command(pytestconfig):
    """python -m pytest takes its options from pyproject.toml, so the workflow prints what a run here prints: the
    duration of every setup, call and teardown, and every outcome but a pass, each skip with its reason."""
    assert pytestconfig.getini("testpaths") == ["tests"]
    assert pytestconfig.getini("addopts") == ["--durations=0", "--durations-min=0", "-ra"]
    workflow = (REPO / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert re.findall(r"run: (.*\bpytest\b.*)", workflow) == ["python -m pytest"]
    assert re.search(r"^defaults:\n  run:\n    working-directory: phi$", workflow, re.M), "the jobs run in phi/"


def in_ci(environ):
    """GitHub sets CI to true on every runner."""
    return environ.get("CI", "").lower() not in ("", "0", "false")


def over_budget(seconds, budget, environ):
    """The message a run of `seconds` fails with when it exceeds `budget` on this machine, or None.

    A CI runner is another machine, so there nothing fails on time, and pytest's last line prints the time."""
    if in_ci(environ) or seconds <= budget:
        return None
    return f"the run took {seconds:.1f} s, over its budget of {budget:g} s on this machine"


def test_the_whole_run_fits_its_budget(request, run_seconds, budget_seconds):
    """tests/conftest.py puts this test after every other one, so run_seconds covers collection and every test."""
    assert request.session.items[-1] is request.node, "the budget check runs after every test it measures"
    message = over_budget(run_seconds, budget_seconds, os.environ)
    assert message is None, message


def test_a_run_within_its_budget_passes():
    assert over_budget(59.9, 60.0, {}) is None


def test_a_run_over_its_budget_fails():
    assert over_budget(60.1, 60.0, {}) == "the run took 60.1 s, over its budget of 60 s on this machine"


def test_in_ci_a_run_over_its_budget_passes():
    assert over_budget(60.1, 60.0, {"CI": "true"}) is None
