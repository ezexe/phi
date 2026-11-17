"""Tests for phi_model: where a project's store is, and which plugin it takes."""

import pytest

import phi_model


def test_project_resolves_to_its_store(tmp_path):
    for d in ("store", "models"):
        (tmp_path / ".claude" / "p4" / "phi" / d).mkdir(parents=True)
    assert phi_model.p4_root(str(tmp_path)) == str(tmp_path / ".claude" / "p4")
    with pytest.raises(SystemExit):
        phi_model.p4_root(str(tmp_path / ".claude"))


def test_mode_needs_no_plugin(tmp_path, monkeypatch):
    """`mode` writes p4.json alone, so it runs on a machine with a store and no plugin to be found."""
    home = tmp_path / "home"
    home.mkdir()
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(home))
    for var in ("P4_RECALL_PLUGIN_ROOT", "CLAUDE_CONFIG_DIR", "CLAUDE_CODE_PLUGIN_CACHE_DIR"):
        monkeypatch.delenv(var, raising=False)
    p4 = tmp_path / "a" / "b" / "c" / "d" / "e" / "proj" / ".claude" / "p4"
    for d in ("store", "models"):
        (p4 / "phi" / d).mkdir(parents=True)
    assert phi_model.main(["mode", "--project", str(p4), "off", "--now", "2026-11-17 10:00"]) == 0
    assert phi_model.read_recall(str(p4))["learn"] == "false"


@pytest.mark.parametrize("version, want", [("0.11.1", (0, 11, 1)), ("0.10.1-beta", (0, 10, 1)), ("2", (2,)),
                                            (3, (0,)), (["0", "10"], (0,)), (None, (0,)), ("beta", (0,))])
def test_plugin_version_reads_any_manifest(tmp_path, version, want):
    """A version that is no string, or that carries a suffix, still compares, and never raises."""
    import json
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "plugin.json").write_text(json.dumps({"version": version}), encoding="utf-8")
    assert phi_model.plugin_version(str(tmp_path)) == want


def environ_writes(path):
    """What running the module at path, as its import does, changes in os.environ, each variable it sets, changes or
    removes in any form, restored afterwards; and "os.putenv" for a call of putenv anywhere in it, which os.environ
    never sees. The module runs twice and the second run is read, so a variable a library sets on its first import
    is not taken for the module's own."""
    import ast
    import os
    import runpy
    changed = []
    for _ in range(2):
        before = dict(os.environ)
        try:
            runpy.run_path(path)
        finally:
            changed = sorted(k for k in set(before) | set(os.environ) if before.get(k) != os.environ.get(k))
            for k in changed:
                if k in before:
                    os.environ[k] = before[k]
                else:
                    del os.environ[k]
    tree = ast.parse(open(path, encoding="utf-8").read())
    if any(isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[-1] == "putenv" for n in ast.walk(tree)):
        changed.append("os.putenv")
    return changed


def test_the_recall_tests_leave_the_environment_alone():
    """Importing a test module, as collection does, sets no variable every later test and process would see: running
    each of the p4 package tests changes nothing in os.environ."""
    import glob
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for path in sorted(glob.glob(os.path.join(here, "test_p4*.py"))):
        assert environ_writes(path) == [], os.path.basename(path)


@pytest.mark.parametrize("write", [
    'os.environ["P4_ENV_PROBE"] = "1"',
    'if True:\n    os.environ["P4_ENV_PROBE"] = "1"',
    'try:\n    os.environ["P4_ENV_PROBE"] = "1"\nexcept OSError:\n    pass',
    'os.environ |= {"P4_ENV_PROBE": "1"}',
    'from os import environ\nenviron["P4_ENV_PROBE"] = "1"',
    'os.environ.__setitem__("P4_ENV_PROBE", "1")',
    'os.environ.setdefault("P4_ENV_PROBE", "1")',
    'def helper():\n    os.environ["P4_ENV_PROBE"] = "1"\n\n\nhelper()',
    'os.putenv("P4_ENV_PROBE", "1")',
])
def test_the_environment_check_sees_every_form_of_write(tmp_path, write):
    """environ_writes finds a write at a module's top in each form a test module could take, and leaves os.environ
    as it found it."""
    import os
    path = tmp_path / "test_p4_probe.py"
    path.write_text("import os\n" + write + "\n", encoding="utf-8")
    try:
        assert environ_writes(str(path)) != []
        assert "P4_ENV_PROBE" not in os.environ
    finally:
        os.unsetenv("P4_ENV_PROBE")


def test_explicit_plugin_wins(tmp_path):
    fake = tmp_path / "p4-phi"
    (fake / "scripts").mkdir(parents=True)
    for n in ("p4.py", "train.py"):
        (fake / "scripts" / n).write_text("", encoding="utf-8")
    assert phi_model.find_plugin(str(fake), str(tmp_path)) == str(fake)
    with pytest.raises(SystemExit):
        phi_model.find_plugin(str(tmp_path / "nothing"), str(tmp_path))
