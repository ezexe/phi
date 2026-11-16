"""The time budget of the whole run, which tests/test_suite.py checks.

pytest_sessionstart notes when the run began, and a test that asks for run_seconds goes after every other test, so it
sees the time of the whole run. The budget is set in one place, budget_seconds in [tool.pytest.ini_options] of
pyproject.toml.
"""
import time

import pytest

RUN_START = pytest.StashKey[float]()


class RunSecondsLast:
    """Puts a test that measures the run after every test it measures, once every other plugin has ordered the run.

    pytest's cache plugin moves the tests that failed last time to the front under --ff, and new tests under --nf, in
    tryfirst wrappers of pytest_collection_modifyitems that it registers in pytest_configure. Hook wrappers registered
    later are called first, so this wrapper, registered in this file's pytest_configure, wraps the cache plugin's and
    orders the items after it has."""

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, items):
        yield
        items.sort(key=lambda item: "run_seconds" in getattr(item, "fixturenames", ()))


def pytest_addoption(parser):
    parser.addini("budget_seconds", "seconds the whole run may take on this machine; a CI runner only prints its time")


def pytest_configure(config):
    config.pluginmanager.register(RunSecondsLast(), "run-seconds-last")


def pytest_sessionstart(session):
    session.stash[RUN_START] = time.perf_counter()


@pytest.fixture
def run_seconds(request):
    """The seconds since the run began: collection and every test before this one."""
    return time.perf_counter() - request.session.stash[RUN_START]


@pytest.fixture
def budget_seconds(pytestconfig):
    """budget_seconds from [tool.pytest.ini_options] of pyproject.toml."""
    return float(pytestconfig.getini("budget_seconds"))
