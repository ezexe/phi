"""phi_model on a store the p4-recall plugin writes: modes into p4.json, check / init / update through the plugin's
own `p4.py` and phi_rank, retrain notices answered. Needs the p4-recall plugin: P4_RECALL_PLUGIN_ROOT, or the p4
checkout beside this one (../p4/lbiza/plugins/p4-recall); skipped otherwise."""

import hashlib
import os
import subprocess
import sys

import pytest

import phi_model as p4model
from phi_model import recall

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.environ.get("P4_RECALL_PLUGIN_ROOT") or os.path.normpath(
    os.path.join(HERE, "..", "..", "p4", "lbiza", "plugins", "p4-recall"))
needs_plugin = pytest.mark.skipif(not recall.is_plugin(PLUGIN), reason="the p4-recall plugin is not beside this checkout")

TOPICS = ["ring buffer drain", "midi transport playhead", "toolbar latency readout", "crash dump symbols",
          "plan document decisions", "render thread priority", "cmake floor version", "branch per machine"]


def entry(pairs):
    return "- " + pairs[0][0] + ": " + pairs[0][1] + "\n" + "".join(f"  {k}: {v}\n" for k, v in pairs[1:]) + "\n"


def write_store(p4, n_tasks=8):
    """A p4-recall store: one rule per topic, one session's rows, usage labels pointing each task at its rule."""
    os.makedirs(os.path.join(p4, "store", "messages"))
    os.makedirs(os.path.join(p4, "models"))
    rules = "The owner's rulings.\nWritten when the owner rules.\nFields: ruling, status, time.\n\n"
    ids = []
    for i, t in enumerate(TOPICS):
        value = f"the {t} rule: keep the {t} as agreed, {t} detail"
        time = f"2026-10-{20 + i:02d} 10:00"
        rules += entry([("ruling", value), ("status", "LIVE"), ("time", time)])
        ids.append(f"rules.md:{time}:{hashlib.sha1(value.encode()).hexdigest()[:8]}")
    with open(os.path.join(p4, "store", "rules.md"), "w", encoding="utf-8") as f:
        f.write(rules)
    rows = "Every owner message.\nOpened by the prompt hook.\nFields: fingerprint, time, arrival, state, addressed.\n\n"
    tasks = "id\tsession\ttime\ttext\tkind\n"
    labels = "task\tentry\tlabel\tsource\tweight\ttime\tteacher\tedited\tnote\n"
    for i in range(n_tasks):
        t = TOPICS[i % len(TOPICS)]
        text = f"please check the {t} in the engine and tell me what was decided about the {t} last week"
        time = f"2026-11-05 {11 + i // 6:02d}:{(i * 7) % 60:02d}"
        rid = f"messages/aaaa1111.md:{time}:{hashlib.sha1(f'\"{text}\"'.encode()).hexdigest()[:8]}"
        rows += entry([("fingerprint", f'"{text}"'), ("time", time), ("arrival", "turn"), ("state", "FRESH"),
                       ("addressed", f"Done: the {t} follows the rule.")])
        tasks += f"{rid}\taaaa1111\t{time}\t{text}\tnew\n"
        labels += f"{rid}\t{ids[i % len(TOPICS)]}\t1\tusage\t0.3\t{time}\t\t\tused; shown\n"
        labels += f"{rid}\t{ids[(i + 1) % len(TOPICS)]}\t0\tusage\t0.3\t{time}\t\t\tunused; shown\n"
    with open(os.path.join(p4, "store", "messages", "aaaa1111.md"), "w", encoding="utf-8") as f:
        f.write(rows)
    with open(os.path.join(p4, "models", "tasks.tsv"), "w", encoding="utf-8") as f:
        f.write(tasks)
    with open(os.path.join(p4, "models", "labels.tsv"), "w", encoding="utf-8") as f:
        f.write(labels)
    with open(os.path.join(p4, ".gitignore"), "w", encoding="utf-8") as f:
        f.write("models/index.json\nmodels/repo.json\nmodels/*.tmp\nmodels/served/\nmodels/ranker.pt\n")
    subprocess.run(["git", "init", "-q", p4], check=True)


def test_p4_root_and_plugin_kind(tmp_path):
    p4 = str(tmp_path / ".claude" / "p4")
    os.makedirs(os.path.join(p4, "store"))
    os.makedirs(os.path.join(p4, "models"))
    assert p4model.p4_root(str(tmp_path)) == p4
    assert recall.is_store(p4) and not recall.is_store(str(tmp_path))
    fake = tmp_path / "p4-recall"
    (fake / "scripts").mkdir(parents=True)
    for n in ("p4.py", "train.py"):
        (fake / "scripts" / n).write_text("", encoding="utf-8")
    assert p4model.plugin_kind(str(fake)) == "p4-recall"
    assert p4model.find_plugin(str(fake), str(tmp_path)) == str(fake)


def test_find_plugin_reads_claude_codes_marketplace_registry(tmp_path, monkeypatch):
    """A directory marketplace is read in place, so the plugin sits wherever known_marketplaces.json says."""
    market = tmp_path / "market" / "lbiza"
    plugin = market / "plugins" / "p4-recall"
    (plugin / "scripts").mkdir(parents=True)
    for n in ("p4.py", "train.py"):
        (plugin / "scripts" / n).write_text("", encoding="utf-8")
    (plugin / ".claude-plugin").mkdir()
    (plugin / ".claude-plugin" / "plugin.json").write_text('{"name": "p4-recall", "version": "0.2.0"}', encoding="utf-8")
    home = tmp_path / "home"
    (home / ".claude" / "plugins").mkdir(parents=True)
    import json
    (home / ".claude" / "plugins" / "known_marketplaces.json").write_text(json.dumps(
        {"p4-marketplace": {"source": {"source": "directory", "path": str(market)}, "installLocation": str(market)}}), encoding="utf-8")
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(home))
    monkeypatch.delenv("P4_RECALL_PLUGIN_ROOT", raising=False)
    monkeypatch.delenv("VLDS_PLUGIN_ROOT", raising=False)
    project = tmp_path / "elsewhere" / "proj"
    (project / ".claude" / "p4" / "store").mkdir(parents=True)
    assert p4model.find_plugin("", str(project)) == str(plugin)


def test_modes_live_in_p4_json_and_keep_every_other_key(tmp_path):
    p4 = str(tmp_path)
    os.makedirs(os.path.join(p4, "store"))
    with open(os.path.join(p4, "p4.json"), "w", encoding="utf-8") as f:
        f.write('{"version": 1, "memory": {"retries": 5}, "recall": {"inject": "rules-facts", "pinned_chars": 1500}}')
    assert p4model.recall_path(p4).endswith("p4.json")
    assert p4model.mode_of(p4model.read_recall(p4)) == "active"
    p4model.write_mode(p4, "shadow", "2026-11-09 18:00")
    cfg = p4model.read_recall(p4)
    assert (cfg["inject"], cfg["learn"]) == ("off", "true") and p4model.mode_of(cfg) == "shadow"
    assert cfg["updated"].startswith("2026-11-09 18:00 by the p4-model skill")
    import json
    data = json.loads(open(os.path.join(p4, "p4.json"), encoding="utf-8").read())
    assert data["memory"] == {"retries": 5} and data["recall"]["pinned_chars"] == 1500   # untouched
    p4model.write_mode(p4, "off", "")
    assert p4model.mode_of(p4model.read_recall(p4)) == "off"
    p4model.write_mode(p4, "active", "")
    assert p4model.mode_of(p4model.read_recall(p4)) == "active"
    # a store the plugin has not configured yet gets a p4.json holding just the recall keys
    bare = str(tmp_path / "bare")
    os.makedirs(os.path.join(bare, "store"))
    p4model.write_mode(bare, "shadow", "")
    assert p4model.mode_of(p4model.read_recall(bare)) == "shadow"


@needs_plugin
def test_checkpoint_routine_on_a_p4_recall_store(tmp_path, capsys):
    root = str(tmp_path / "proj")
    p4 = os.path.join(root, ".claude", "p4")
    write_store(p4)
    args = ["--project", root, "--plugin", PLUGIN, "--now", "2026-11-09 18:00"]
    rep = p4model.check(p4, PLUGIN)
    assert rep["kind"] == "p4-recall" and rep["labelled_tasks"] == 8 and rep["json_ranker"] is None
    assert "train json" in rep["owed"] and "train ranker" in rep["owed"]
    assert p4model.main(["init"] + args + ["--minutes", "0.02", "--mode", "shadow"]) == 0
    out = capsys.readouterr().out
    assert "train dispatch: nothing to train" in out
    rep = p4model.check(p4, PLUGIN)
    assert rep["mode"] == "shadow", rep["recall"]
    assert rep["eval_last"] and rep["eval_last"]["round"] == "0", rep["eval_last"]
    assert rep["pt_ranker"] and rep["pt_ranker"]["round"] == 0 and rep["pt_ranker"]["checkpoint"] == "present", rep["pt_ranker"]
    assert os.path.exists(os.path.join(p4, "models", "ranker-pt.json"))        # never the plugin's ranker.json
    assert rep["lfs"] and "models/*.pt filter=lfs" in open(os.path.join(p4, ".gitattributes"), encoding="utf-8").read()
    assert rep["owed"] == [], rep["owed"]
    # a session meets a failure: one notice; the checkpoint owes the model it names, retrains, answers it
    sys.path.insert(0, os.path.join(PLUGIN, "scripts"))
    import notices
    assert notices.notify(p4, "pt ranker server", RuntimeError("bad scores"), "aaaa1111", now="2026-11-09 18:05")
    rep = p4model.check(p4, PLUGIN)
    assert rep["owed"] == ["train ranker"] and len(rep["notices"]) == 1
    assert p4model.main(["update"] + args + ["--minutes", "0.02"]) == 0
    rep = p4model.check(p4, PLUGIN)
    assert rep["notices"] == [] and rep["owed"] == [] and rep["pt_ranker"]["round"] == 1, rep
    resolved = open(os.path.join(p4, "models", notices.RESOLVED), encoding="utf-8").read()
    assert "pt ranker server" in resolved and "retrained 2026-11-09 18:00" in resolved
    # a corrupt JSON ranker is a finding and owes a round; the notice file is never rewritten
    before = open(os.path.join(p4, "models", notices.NOTICES), encoding="utf-8").read()
    with open(os.path.join(p4, "models", "ranker.json"), "w", encoding="utf-8") as f:
        f.write("<<<<<<< HEAD\n")
    rep = p4model.check(p4, PLUGIN)
    assert "ranker.json" in rep["corrupt"] and "train json" in rep["owed"]
    assert p4model.main(["train", "--only", "json"] + args) == 0
    assert open(os.path.join(p4, "models", notices.NOTICES), encoding="utf-8").read() == before
    assert "ranker.json" not in p4model.check(p4, PLUGIN)["corrupt"]
    assert p4model.main(["check"] + args) == 0


@needs_plugin
def test_fresh_p4_recall_store_is_no_failure(tmp_path, capsys):
    root = str(tmp_path / "proj")
    p4 = os.path.join(root, ".claude", "p4")
    os.makedirs(os.path.join(p4, "store"))
    os.makedirs(os.path.join(p4, "models"))
    args = ["--project", root, "--plugin", PLUGIN, "--now", "2026-11-09 18:00"]
    assert p4model.main(["init"] + args + ["--minutes", "0"]) == 0
    rep = p4model.check(p4, PLUGIN)
    assert rep["mode"] == "shadow" and rep["owed"] == [] and rep["notices"] == [], rep
    assert not os.path.exists(os.path.join(p4, "models", "retrain-needed.tsv"))
