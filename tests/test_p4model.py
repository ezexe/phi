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


def test_explicit_plugin_wins(tmp_path):
    fake = tmp_path / "p4-phi"
    (fake / "scripts").mkdir(parents=True)
    for n in ("p4.py", "train.py"):
        (fake / "scripts" / n).write_text("", encoding="utf-8")
    assert phi_model.find_plugin(str(fake), str(tmp_path)) == str(fake)
    with pytest.raises(SystemExit):
        phi_model.find_plugin(str(tmp_path / "nothing"), str(tmp_path))
