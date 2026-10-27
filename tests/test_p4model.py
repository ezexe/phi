"""Tests for phi_model: the recall config's modes, and the checkpoint routine on a store end to end.

The end-to-end test needs the vlds plugin (ezexe/prompty-prompter-pioneer-puppeteer): set VLDS_PLUGIN_ROOT to its
Ibiza/plugins/vlds, or it is skipped."""

import os
import sys

import pytest

import phi_model as p4model

RECALL = """# idx

register: 0

## recall

inject: local-storage.md, index.md
pool-road: children
operator-via: mister:roboto

updated: 2026-10-12 17:07 by a light pass

## hot

| file | live |
"""


def test_modes_rewrite_only_the_model_keys(tmp_path):
    (tmp_path / "sessions").mkdir()
    (tmp_path / "vindex").mkdir()
    (tmp_path / "vindex" / "phi-index.md").write_text(RECALL, encoding="utf-8")
    assert p4model.p4_root(str(tmp_path)) == str(tmp_path)
    assert p4model.mode_of(p4model.read_recall(str(tmp_path))) == "custom"
    p4model.write_mode(str(tmp_path), "shadow", "2026-10-27 10:00")
    cfg = p4model.read_recall(str(tmp_path))
    assert cfg["pool-road"] == "model-shadow" and cfg["dispatch-model"] == "shadow"
    assert cfg["inject"] == "local-storage.md, index.md" and cfg["operator-via"] == "mister:roboto"
    assert p4model.mode_of(cfg) == "shadow" and cfg["updated"].startswith("2026-10-27 10:00 by the p4-model skill")
    text = (tmp_path / "vindex" / "phi-index.md").read_text(encoding="utf-8")
    assert text.startswith("# idx\n\nregister: 0\n") and "## hot\n\n| file | live |" in text
    p4model.write_mode(str(tmp_path), "active", "2026-10-27 10:05")
    cfg = p4model.read_recall(str(tmp_path))
    assert (cfg["pool-road"], cfg["pool-audit"], cfg["dispatch-model"]) == ("model", "5", "on")
    p4model.write_mode(str(tmp_path), "off", "2026-10-27 10:10")
    cfg = p4model.read_recall(str(tmp_path))
    assert cfg["pool-audit"] == "5" and p4model.mode_of(cfg) == "off"   # an audit once set stays the owner's
    assert text.count("## recall") == 1


def test_mode_keeps_the_owners_audit_and_finds_the_real_heading(tmp_path):
    (tmp_path / "sessions").mkdir()
    (tmp_path / "vindex").mkdir()
    (tmp_path / "vindex" / "phi-index.md").write_text(
        "# idx\n\nnotes: see ### recall below, and ## recall-history\n\n## recall-history\n\nkeep: me\n\n"
        "## recall\n\npool-road: children\npool-audit: 9\n", encoding="utf-8")
    for mode in ("active", "shadow", "off", "active"):
        p4model.write_mode(str(tmp_path), mode, "")
        cfg = p4model.read_recall(str(tmp_path))
        assert cfg["pool-audit"] == "9", (mode, cfg)
    text = (tmp_path / "vindex" / "phi-index.md").read_text(encoding="utf-8")
    assert "## recall-history\n\nkeep: me" in text and text.count("pool-road:") == 1


def test_project_resolves_to_its_store(tmp_path):
    (tmp_path / ".claude" / "p4" / "sessions").mkdir(parents=True)
    assert p4model.p4_root(str(tmp_path)) == str(tmp_path / ".claude" / "p4")
    with pytest.raises(SystemExit):
        p4model.p4_root(str(tmp_path / ".claude"))


PLUGIN = os.environ.get("VLDS_PLUGIN_ROOT", "")


@pytest.mark.skipif(not os.path.exists(os.path.join(PLUGIN, "scripts", "pick_model.py")),
                    reason="VLDS_PLUGIN_ROOT does not name the vlds plugin")
def test_checkpoint_routine(tmp_path, capsys):
    sys.path.insert(0, os.path.join(PLUGIN, "scripts"))
    import test_pick_model as tp  # the plugin's seeded store: four labelled pools
    root, p4, stores = tp.seed()
    try:
        args = ["--project", root, "--plugin", PLUGIN, "--now", "2026-10-27 10:00"]
        assert p4model.main(["init"] + args + ["--minutes", "0.02"]) == 0
        rep = p4model.check(p4, PLUGIN)
        assert rep["mode"] == "shadow" and rep["owed"] == [] and rep["lfs"] and rep["gitignore"], rep
        assert rep["pick_model"]["round"] == 0 and rep["ranker"]["round"] == 0 and rep["ranker"]["checkpoint"]
        # a session meets a failure: one notice; the checkpoint owes the model it names, retrains, answers it
        pm, _dm = p4model.plugin_modules(PLUGIN)
        assert pm.notify_retrain(stores["aaaa1111"], "ranker server", ValueError("bad scores"), "s1")
        rep = p4model.check(p4, PLUGIN)
        assert rep["owed"] == ["train ranker"] and len(rep["notices"]) == 1
        assert p4model.main(["update"] + args + ["--minutes", "0.02"]) == 0
        rep = p4model.check(p4, PLUGIN)
        assert rep["owed"] == [] and rep["notices"] == [] and rep["ranker"]["round"] == 1
        resolved = open(os.path.join(p4, "models", "retrain-resolved.tsv"), encoding="utf-8").read()
        assert "ranker server" in resolved and "retrained 2026-10-27 10:00" in resolved
        assert p4model.main(["check"] + args) == 0
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.skipif(not os.path.exists(os.path.join(PLUGIN, "scripts", "pick_model.py")),
                    reason="VLDS_PLUGIN_ROOT does not name the vlds plugin")
def test_fresh_store_is_no_failure(tmp_path):
    """A store with no labelled pool and no judged repeat: init sets the mode, trains nothing, reports nothing to
    learn, leaves no retrain notice, and check owes nothing — the turn close bootstraps the models later."""
    store = tmp_path / ".claude" / "p4" / "sessions" / "f1e5a7a0" / "vlds"
    store.mkdir(parents=True)
    (store / "dispatch.md").write_text("# dispatch\n\n---\n", encoding="utf-8")
    args = ["--project", str(tmp_path), "--plugin", PLUGIN, "--now", "2026-10-27 10:00"]
    assert p4model.main(["init"] + args + ["--minutes", "0"]) == 0
    p4 = str(tmp_path / ".claude" / "p4")
    rep = p4model.check(p4, PLUGIN)
    assert rep["mode"] == "shadow" and rep["owed"] == [] and rep["notices"] == [], rep
    assert not os.path.exists(os.path.join(p4, "models", "retrain-needed.tsv"))
    assert p4model.main(["check"] + args) == 0



def test_explicit_plugin_wins(tmp_path):
    fake = tmp_path / "vlds"
    (fake / "scripts").mkdir(parents=True)
    (fake / "scripts" / "pick_model.py").write_text("", encoding="utf-8")
    assert p4model.find_plugin(str(fake), str(tmp_path)) == str(fake)
    with pytest.raises(SystemExit):
        p4model.find_plugin(str(tmp_path / "nothing"), str(tmp_path))


@pytest.mark.skipif(not os.path.exists(os.path.join(PLUGIN, "scripts", "pick_model.py")),
                    reason="VLDS_PLUGIN_ROOT does not name the vlds plugin")
def test_corrupt_model_and_append_only_resolution(tmp_path):
    sys.path.insert(0, os.path.join(PLUGIN, "scripts"))
    import test_pick_model as tp
    root, p4, stores = tp.seed()
    try:
        args = ["--project", root, "--plugin", PLUGIN, "--now", "2026-10-27 10:00"]
        assert p4model.main(["train", "--only", "json"] + args + ["--minutes", "0"]) == 0
        with open(os.path.join(p4, "models", "pick-model.json"), "w") as f:
            f.write("<<<<<<< HEAD\n")
        rep = p4model.check(p4, PLUGIN)
        assert "pick-model.json" in rep["corrupt"] and "train json" in rep["owed"]
        pm, _dm = p4model.plugin_modules(PLUGIN)
        assert pm.notify_retrain(stores["aaaa1111"], "pick_model picks", ValueError("x"), "s1")
        before = open(os.path.join(p4, "models", pm.NOTICES), encoding="utf-8").read()
        assert p4model.main(["train", "--only", "json"] + args + ["--minutes", "0"]) == 0
        assert open(os.path.join(p4, "models", pm.NOTICES), encoding="utf-8").read() == before
        rep = p4model.check(p4, PLUGIN)
        assert rep["notices"] == [] and rep["corrupt"] == {}
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
