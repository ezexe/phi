"""Tests for phi_model: where a project's store is, and which plugin it takes."""

import pytest

import phi_model


def test_project_resolves_to_its_store(tmp_path):
    for d in ("store", "models"):
        (tmp_path / ".claude" / "p4" / "phi" / d).mkdir(parents=True)
    assert phi_model.p4_root(str(tmp_path)) == str(tmp_path / ".claude" / "p4")
    with pytest.raises(SystemExit):
        phi_model.p4_root(str(tmp_path / ".claude"))


def test_explicit_plugin_wins(tmp_path):
    fake = tmp_path / "p4-phi"
    (fake / "scripts").mkdir(parents=True)
    for n in ("p4.py", "train.py"):
        (fake / "scripts" / n).write_text("", encoding="utf-8")
    assert phi_model.find_plugin(str(fake), str(tmp_path)) == str(fake)
    with pytest.raises(SystemExit):
        phi_model.find_plugin(str(tmp_path / "nothing"), str(tmp_path))
