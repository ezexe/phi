"""Tests for phi_prune.p4model: the recall config's modes, and the checkpoint routine on a store end to end.

The end-to-end test needs the vlds plugin (ezexe/prompty-prompter-pioneer-puppeteer): set VLDS_PLUGIN_ROOT to its
Ibiza/plugins/vlds, or it is skipped."""

import os
import sys

import pytest

from phi_prune import p4model

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
    assert "pool-audit" not in cfg and p4model.mode_of(cfg) == "off"
    assert text.count("## recall") == 1


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
