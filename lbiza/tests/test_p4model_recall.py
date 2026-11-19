"""phi_model on a store the p4-phi plugin writes: modes into p4.json, check, init and update through the plugin's
own `p4.py` and phi_rank, retrain notices answered. Needs the p4-phi plugin: P4_RECALL_PLUGIN_ROOT, or the p4
checkout beside this one, at p4/lbiza/plugins/p4-phi beside this checkout's folder; skipped otherwise."""

import hashlib
import os
import subprocess
import sys

import pytest

import phi_model
from phi_model import recall

from phi_store import layout  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.environ.get("P4_RECALL_PLUGIN_ROOT") or next(
    (p for p in (os.path.normpath(os.path.join(HERE, "..", "..", "..", "p4", "lbiza", "plugins", "p4-phi")),)
     if os.path.exists(os.path.join(p, "scripts", "p4.py"))), "")
needs_plugin = pytest.mark.skipif(not recall.is_plugin(PLUGIN), reason="the p4-phi plugin is not beside this checkout")

TOPICS = ["ring buffer drain", "midi transport playhead", "toolbar latency readout", "crash dump symbols",
          "plan document decisions", "render thread priority", "cmake floor version", "branch per machine"]


@pytest.fixture(autouse=True)
def dual_encoder_on(monkeypatch):
    """These tests build the dual encoder, which is on only as an experiment; the variable is set for each test
    alone, so no other test file or later process sees it."""
    monkeypatch.setenv("VLDS_PT_RANKER", "1")


@pytest.fixture(autouse=True)
def in_process(monkeypatch):
    """`python -m phi_rank` runs in this process at two epochs on a base of 4096 buckets by 16, and git lfs answers ok
    without running: each command line is still the one recall.train builds, read by phi_rank's own parser, at no
    cost of an interpreter start, a torch import and thirty epochs. The plugin's p4.py and git run as they do in a
    store. phi_rank, and with it torch, is imported only by a test that runs it, so the tests that need no torch run
    on a machine without it."""
    import contextlib
    import io

    def fake(cmd):
        if cmd[1:3] == ["-m", "phi_rank"]:
            import phi_rank
            monkeypatch.setattr(phi_rank, "CONTINUE_EPOCHS", 2)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                code = phi_rank.main(cmd[3:] + ["--epochs", "2", "--buckets", "4096", "--dim", "16"])
            return subprocess.CompletedProcess(cmd, code, out.getvalue(), "")
        if "lfs" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return None
    real_r, real_m = recall.run, phi_model.run
    monkeypatch.setattr(recall, "run", lambda cmd, cwd=None: fake(cmd) or real_r(cmd, cwd))
    monkeypatch.setattr(phi_model, "run", lambda cmd, cwd=None, env=None: fake(cmd) or real_m(cmd, cwd, env))


def entry(pairs):
    return "- " + pairs[0][0] + ": " + pairs[0][1] + "\n" + "".join(f"  {k}: {v}\n" for k, v in pairs[1:]) + "\n"


def write_store(p4, n_tasks=8, flat=False):
    """A store in the layout p4-phi writes, or with flat a flat one as its older versions wrote it, which check
    moves into the layout: one rule per topic, one session's rows, usage labels pointing each query at its rule."""
    os.makedirs(os.path.join(p4, *([] if flat else ["phi"]), "store", "messages"))
    os.makedirs(layout.sub(p4, "models"))
    rules = "The owner's rulings.\nWritten when the owner rules.\nFields: ruling, status, time.\n\n"
    ids = []
    for i, t in enumerate(TOPICS):
        value = f"the {t} rule: keep the {t} as agreed, {t} detail"
        time = f"2026-10-{20 + i:02d} 10:00"
        rules += entry([("ruling", value), ("status", "LIVE"), ("time", time)])
        ids.append(f"rules.md:{time}:{hashlib.sha1(value.encode()).hexdigest()[:8]}")
    with open(layout.sub(p4, "store", "rules.md"), "w", encoding="utf-8") as f:
        f.write(rules)
    rows = "Every owner message.\nOpened by the prompt hook.\nFields: fingerprint, time, arrival, state, addressed.\n\n"
    tasks = "id\tsession\ttime\ttext\tkind\n"
    labels = "task\tentry\tlabel\tsource\tweight\ttime\tteacher\tedited\tnote\n"
    for i in range(n_tasks):
        t = TOPICS[i % len(TOPICS)]
        text = f"please check the {t} in the engine and tell me what was decided about the {t} last week"
        time = f"2026-11-05 {11 + i // 6:02d}:{(i * 7) % 60:02d}"
        quoted = f'"{text}"'
        rid = f"messages/aaaa1111.md:{time}:{hashlib.sha1(quoted.encode()).hexdigest()[:8]}"
        rows += entry([("fingerprint", quoted),("time", time), ("arrival", "turn"), ("state", "FRESH"),
                       ("addressed", f"Done: the {t} follows the rule.")])
        tasks += f"{rid}\taaaa1111\t{time}\t{text}\tnew\n"
        labels += f"{rid}\t{ids[i % len(TOPICS)]}\t1\tusage\t0.3\t{time}\t\t\tused; shown\n"
        labels += f"{rid}\t{ids[(i + 1) % len(TOPICS)]}\t0\tusage\t0.3\t{time}\t\t\tunused; shown\n"
    with open(layout.sub(p4, "store", "messages", "aaaa1111.md"), "w", encoding="utf-8") as f:
        f.write(rows)
    with open(layout.sub(p4, "models", "tasks.tsv"), "w", encoding="utf-8") as f:
        f.write(tasks)
    with open(layout.sub(p4, "models", "labels.tsv"), "w", encoding="utf-8") as f:
        f.write(labels)
    with open(os.path.join(p4, ".gitignore"), "w", encoding="utf-8") as f:
        m = layout.rel(p4, "models")
        f.write(f"{m}/index.json\n{m}/repo.json\n{m}/*.tmp\n{m}/served/\n{m}/ranker.pt\n")
    subprocess.run(["git", "init", "-q", p4], check=True)


def test_p4_root_and_explicit_plugin(tmp_path):
    p4 = str(tmp_path / ".claude" / "p4")
    os.makedirs(os.path.join(p4, "store"))
    os.makedirs(layout.sub(p4, "models"))
    assert phi_model.p4_root(str(tmp_path)) == p4
    assert recall.is_store(p4) and not recall.is_store(str(tmp_path))
    fake = tmp_path / "p4-recall"
    (fake / "scripts").mkdir(parents=True)
    for n in ("p4.py", "train.py"):
        (fake / "scripts" / n).write_text("", encoding="utf-8")
    assert phi_model.find_plugin(str(fake), str(tmp_path)) == str(fake)


def test_find_plugin_reads_claude_codes_marketplace_registry(tmp_path, monkeypatch):
    """A directory marketplace is read in place, so the plugin sits wherever known_marketplaces.json says."""
    market = tmp_path / "market" / "lbiza"
    plugin = market / "plugins" / "p4-phi"
    (plugin / "scripts").mkdir(parents=True)
    for n in ("p4.py", "train.py"):
        (plugin / "scripts" / n).write_text("", encoding="utf-8")
    (plugin / ".claude-plugin").mkdir()
    (plugin / ".claude-plugin" / "plugin.json").write_text('{"name": "p4-phi", "version": "0.2.0"}', encoding="utf-8")
    home = tmp_path / "home"
    (home / ".claude" / "plugins").mkdir(parents=True)
    import json
    (home / ".claude" / "plugins" / "known_marketplaces.json").write_text(json.dumps(
        {"p4-marketplace": {"source": {"source": "directory", "path": str(market)}, "installLocation": str(market)}}), encoding="utf-8")
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(home))
    monkeypatch.delenv("P4_RECALL_PLUGIN_ROOT", raising=False)
    project = tmp_path / "a" / "b" / "c" / "d" / "e" / "proj"
    (project / ".claude" / "p4" / "store").mkdir(parents=True)
    assert phi_model.find_plugin("", str(project)) == str(plugin)


def test_mode_needs_no_torch(tmp_path, monkeypatch):
    """`mode` writes p4.json on a machine with no torch build, where phi_rank does not import."""
    monkeypatch.setitem(sys.modules, "phi_rank", None)
    os.makedirs(os.path.join(str(tmp_path), "store"))
    phi_model.write_mode(str(tmp_path), "off", "2026-11-17 10:00")
    assert phi_model.mode_of(phi_model.read_recall(str(tmp_path))) == "off"


def test_the_tests_that_need_no_torch_run_without_it():
    """On a machine where torch does not import, this module's tests of `mode` and of find_plugin still pass: no
    fixture they take imports phi_rank."""
    code = ("import sys; sys.modules['torch'] = None; import pytest; "
            f"sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', '-k', 'mode_needs_no_torch or find_plugin', {__file__!r}]))")
    r = subprocess.run([sys.executable, "-B", "-c", code], cwd=os.path.dirname(HERE), capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0 and " passed" in r.stdout and "error" not in r.stdout, r.stdout[-2000:]


def fake_plugin(path, version):
    (path / "scripts").mkdir(parents=True)
    for n in ("p4.py", "train.py"):
        (path / "scripts" / n).write_text("", encoding="utf-8")
    (path / ".claude-plugin").mkdir()
    (path / ".claude-plugin" / "plugin.json").write_text(f'{{"name": "p4-phi", "version": "{version}"}}', encoding="utf-8")
    return path


def test_find_plugin_takes_the_copy_sessions_load(tmp_path, monkeypatch):
    """A plugin of a marketplace read in place loads from its folder there, whatever version its cache copy holds,
    and the registry is read under $CLAUDE_CODE_PLUGIN_CACHE_DIR when it is set."""
    import json
    market = tmp_path / "m" / "k" / "lbiza"
    folder = fake_plugin(market / "plugins" / "p4-phi", "0.10.1")
    home = tmp_path / "home"
    root = home / ".claude" / "plugins"
    cached = fake_plugin(root / "cache" / "p4-marketplace" / "p4-phi" / "0.11.0", "0.11.0")
    (root / "known_marketplaces.json").write_text(json.dumps({"p4-marketplace": {
        "source": {"source": "directory", "path": str(market)}, "installLocation": str(market)}}), encoding="utf-8")
    (root / "installed_plugins.json").write_text(json.dumps({"version": 2, "plugins": {"p4-phi@p4-marketplace": [
        {"scope": "user", "installPath": str(cached), "version": "0.11.0"}]}}), encoding="utf-8")
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(home))
    for var in ("P4_RECALL_PLUGIN_ROOT", "CLAUDE_CONFIG_DIR", "CLAUDE_CODE_PLUGIN_CACHE_DIR"):
        monkeypatch.delenv(var, raising=False)
    project = tmp_path / "a" / "b" / "c" / "d" / "e" / "proj"
    (project / ".claude" / "p4" / "store").mkdir(parents=True)
    assert phi_model.find_plugin("", str(project)) == str(folder)
    moved = tmp_path / "plugins-root"
    root.rename(moved)
    monkeypatch.setenv("CLAUDE_CODE_PLUGIN_CACHE_DIR", str(moved))
    assert phi_model.find_plugin("", str(project)) == str(folder)


def test_find_plugin_reads_a_file_marketplace_under_claude_config_dir(tmp_path, monkeypatch):
    """A marketplace added from its marketplace.json, a `file` source, is read in place from the folder that holds
    .claude-plugin/, with the plugin where the file's entry puts it; the registry is read under $CLAUDE_CONFIG_DIR's
    plugins folder when no $CLAUDE_CODE_PLUGIN_CACHE_DIR is set."""
    import json
    market = tmp_path / "m" / "lbiza"
    folder = fake_plugin(market / "kit" / "p4-phi", "0.11.5")
    manifest = market / ".claude-plugin" / "marketplace.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"name": "p4-marketplace", "plugins": [{"name": "p4-phi", "source": "./kit/p4-phi"}]}),
                        encoding="utf-8")
    root = tmp_path / "config" / "plugins"
    cached = fake_plugin(root / "cache" / "p4-marketplace" / "p4-phi" / "0.11.4", "0.11.4")
    (root / "known_marketplaces.json").write_text(json.dumps({"p4-marketplace": {
        "source": {"source": "file", "path": str(manifest)}, "installLocation": str(manifest)}}), encoding="utf-8")
    (root / "installed_plugins.json").write_text(json.dumps({"version": 2, "plugins": {"p4-phi@p4-marketplace": [
        {"scope": "user", "installPath": str(cached), "version": "0.11.4"}]}}), encoding="utf-8")
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(tmp_path / "home"))
    for var in ("P4_RECALL_PLUGIN_ROOT", "CLAUDE_CODE_PLUGIN_CACHE_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    project = tmp_path / "a" / "b" / "c" / "d" / "e" / "proj"
    (project / ".claude" / "p4" / "store").mkdir(parents=True)
    assert phi_model.find_plugin("", str(project)) == str(folder)


def test_modes_live_in_p4_json_and_keep_every_other_key(tmp_path):
    p4 = str(tmp_path)
    os.makedirs(os.path.join(p4, "store"))
    with open(os.path.join(p4, "p4.json"), "w", encoding="utf-8") as f:
        f.write('{"version": 1, "memory": {"retries": 5}, "recall": {"inject": "rules-facts", "pinned_chars": 1500}}')
    assert phi_model.recall_path(p4).endswith("p4.json")
    assert phi_model.mode_of(phi_model.read_recall(p4)) == "active"
    phi_model.write_mode(p4, "shadow", "2026-11-09 18:00")
    cfg = phi_model.read_recall(p4)
    assert (cfg["inject"], cfg["learn"]) == ("off", "true") and phi_model.mode_of(cfg) == "shadow"
    assert cfg["updated"].startswith("2026-11-09 18:00 by the p4-model skill")
    import json
    data = json.loads(open(os.path.join(p4, "p4.json"), encoding="utf-8").read())
    assert data["memory"] == {"retries": 5} and data["recall"]["pinned_chars"] == 1500   # untouched
    phi_model.write_mode(p4, "off", "")
    assert phi_model.mode_of(phi_model.read_recall(p4)) == "off"
    phi_model.write_mode(p4, "active", "")
    assert phi_model.mode_of(phi_model.read_recall(p4)) == "active"
    # a store the plugin has not configured yet gets a p4.json holding the recall keys alone
    bare = str(tmp_path / "bare")
    os.makedirs(os.path.join(bare, "store"))
    phi_model.write_mode(bare, "shadow", "")
    assert phi_model.mode_of(phi_model.read_recall(bare)) == "shadow"


@needs_plugin
def test_check_init_and_update_on_a_store(tmp_path, capsys):
    root = str(tmp_path / "proj")
    p4 = os.path.join(root, ".claude", "p4")
    write_store(p4)
    args = ["--project", root, "--plugin", PLUGIN, "--now", "2026-11-09 18:00"]
    rep = phi_model.check(p4, PLUGIN)
    assert rep["kind"] == "p4-recall" and rep["labelled_tasks"] == 8 and rep["json_ranker"] is None
    assert "train json" in rep["owed"] and "train ranker" in rep["owed"]
    assert phi_model.main(["init"] + args + ["--minutes", "0", "--mode", "shadow"]) == 0
    rep = phi_model.check(p4, PLUGIN)
    assert rep["mode"] == "shadow", rep["recall"]
    assert rep["eval_last"] and rep["eval_last"]["round"] == "0", rep["eval_last"]
    assert rep["pt_ranker"] and rep["pt_ranker"]["round"] == 0 and rep["pt_ranker"]["checkpoint"] == "present", rep["pt_ranker"]
    assert os.path.exists(layout.sub(p4, "models", "ranker-pt.json"))        # never the plugin's ranker.json
    assert rep["lfs"] and "models/*.pt filter=lfs" in open(os.path.join(p4, ".gitattributes"), encoding="utf-8").read()
    assert rep["owed"] == [], rep["owed"]
    # a session meets a failure: one notice; check finds the model it names stale, and update retrains it and
    # answers the notice
    sys.path.insert(0, os.path.join(PLUGIN, "scripts"))
    import notices
    assert notices.notify(p4, "pt ranker server", RuntimeError("bad scores"), "aaaa1111", now="2026-11-09 18:05")
    rep = phi_model.check(p4, PLUGIN)
    assert rep["owed"] == ["train ranker"] and len(rep["notices"]) == 1
    assert phi_model.main(["update"] + args + ["--minutes", "0"]) == 0
    rep = phi_model.check(p4, PLUGIN)
    assert rep["notices"] == [] and rep["owed"] == [] and rep["pt_ranker"]["round"] == 1, rep
    resolved = open(layout.sub(p4, "models", notices.RESOLVED), encoding="utf-8").read()
    assert "pt ranker server" in resolved and "retrained 2026-11-09 18:00" in resolved
    # a corrupt logistic regression ranker is a finding and makes a round due; the notice file is never rewritten
    before = open(layout.sub(p4, "models", notices.NOTICES), encoding="utf-8").read()
    with open(layout.sub(p4, "models", "ranker.json"), "w", encoding="utf-8") as f:
        f.write("<<<<<<< HEAD\n")
    rep = phi_model.check(p4, PLUGIN)
    assert "ranker.json" in rep["corrupt"] and "train json" in rep["owed"]
    assert phi_model.main(["train", "--only", "json"] + args) == 0
    assert open(layout.sub(p4, "models", notices.NOTICES), encoding="utf-8").read() == before
    assert "ranker.json" not in phi_model.check(p4, PLUGIN)["corrupt"]
    assert phi_model.main(["check"] + args) == 0


@needs_plugin
def test_fresh_store_is_no_failure(tmp_path, capsys):
    root = str(tmp_path / "proj")
    p4 = os.path.join(root, ".claude", "p4")
    os.makedirs(os.path.join(p4, "store"))
    os.makedirs(layout.sub(p4, "models"))
    args = ["--project", root, "--plugin", PLUGIN, "--now", "2026-11-09 18:00"]
    assert phi_model.main(["init"] + args + ["--minutes", "0"]) == 0
    rep = phi_model.check(p4, PLUGIN)
    assert rep["mode"] == "shadow" and rep["owed"] == [] and rep["notices"] == [], rep
    assert not os.path.exists(layout.sub(p4, "models", "retrain-needed.tsv"))


@needs_plugin
def test_check_without_torch_checks_what_retrieval_serves(tmp_path, monkeypatch, capsys):
    """On a machine with no torch build, check still reads the logistic regression ranker, the labels and the
    notices; the dual encoder is reported as not checked and is never stale, since retrieval serves without it."""
    p4 = os.path.join(str(tmp_path / "proj"), ".claude", "p4")
    write_store(p4)
    with_torch = phi_model.check(p4, PLUGIN)
    monkeypatch.setitem(sys.modules, "phi_rank", None)        # `from phi_rank import …` now raises ModuleNotFoundError
    rep = phi_model.check(p4, PLUGIN)
    assert rep["no_torch"] == "phi_rank"
    assert "train ranker" not in rep["owed"]
    assert rep["labels"] == with_torch["labels"] and rep["labelled_tasks"] == with_torch["labelled_tasks"]
    assert [o for o in with_torch["owed"] if o not in ("train ranker", "seed .gitignore/.gitattributes")] == rep["owed"]
    phi_model.print_check(rep)
    assert "pt ranker: not checked or trained here" in capsys.readouterr().out


@needs_plugin
def test_labels_since_the_round_counts_raw_labels_as_the_round_does(tmp_path):
    """eval.tsv records count_labels, labels-raw.tsv and labels.tsv together; check takes the count since the round
    the same way, so a store with raw labels never reads a negative count. The store is flat, so check moves it
    first."""
    p4 = os.path.join(str(tmp_path / "proj"), ".claude", "p4")
    write_store(p4, flat=True)
    m = recall.modules(PLUGIN, p4)
    raw = [{"task": "t", "entry": f"e{i}", "label": "1", "source": "store", "weight": "0.5", "time": "2026-11-01 10:00"}
           for i in range(7)]
    m.labels.write_tsv(m.labels.mpath(p4, m.labels.RAW), m.labels.RAW_HEAD, raw)
    total = m.labels.count_labels(p4)
    assert total > len(m.labels.read_tsv(m.labels.mpath(p4, m.labels.TRAIN)))
    with open(layout.sub(p4, "models", "eval.tsv"), "w", encoding="utf-8") as f:
        f.write("round\ttime\tlabels\n1\t2026-11-09 10:00\t%d\n" % total)
    assert phi_model.check(p4, PLUGIN)["labels_since_round"] == 0


def phi_store_repo(p4):
    """A store in the layout p4-phi writes, a git repository with p4.json, and nothing trained."""
    for d in ("store", "models"):
        os.makedirs(os.path.join(p4, "phi", d))
    with open(os.path.join(p4, "p4.json"), "w", encoding="utf-8") as f:
        f.write('{"version": 1}')
    subprocess.run(["git", "init", "-q", p4], check=True)


def git(p4, *args):
    return subprocess.run(["git", "-C", p4, "-c", "user.name=t", "-c", "user.email=t@t", *args], capture_output=True,
                          text=True)


@needs_plugin
def test_check_asks_to_untrack_the_local_build_in_the_stores_layout(tmp_path):
    """A store that tracks phi/models/ranker.pt is asked to untrack it, at the path git tracks."""
    p4 = str(tmp_path / "proj" / ".claude" / "p4")
    phi_store_repo(p4)
    with open(os.path.join(p4, "phi", "models", "ranker.pt"), "wb") as f:
        f.write(b"x")
    git(p4, "add", "-f", "phi/models/ranker.pt")
    git(p4, "commit", "-qm", "x")
    assert "untrack phi/models/ranker.pt with git rm --cached: the checkpoints git keeps rebuild it" in \
        phi_model.check(p4, PLUGIN)["owed"]


@needs_plugin
def test_a_store_as_the_plugin_seeds_it_is_current(tmp_path, monkeypatch):
    """A store p4-phi seeded, with no dual encoder asked for, owes no seeding of the LFS lines only the dual
    encoder's checkpoints need, so check exits 0."""
    monkeypatch.delenv("VLDS_PT_RANKER")
    root = tmp_path / "proj"
    p4 = str(root / ".claude" / "p4")
    phi_store_repo(p4)
    with open(os.path.join(p4, ".gitignore"), "w", encoding="utf-8") as f:
        f.write("phi/models/index.json\nphi/models/ranker.pt\n")
    assert phi_model.check(p4, PLUGIN)["owed"] == []
    assert phi_model.main(["check", "--project", str(root), "--plugin", PLUGIN]) == 0
    monkeypatch.setenv("VLDS_PT_RANKER", "1")
    assert "seed .gitignore/.gitattributes" in phi_model.check(p4, PLUGIN)["owed"]


@needs_plugin
def test_update_fails_while_anything_is_stale(tmp_path, monkeypatch):
    """A notice of the logistic regression ranker on a store with too few labelled queries to train it stays open
    after update, so the store stays stale, and update then exits 1, as the check after it does."""
    root = str(tmp_path / "proj")
    p4 = os.path.join(root, ".claude", "p4")
    write_store(p4, n_tasks=2)
    monkeypatch.delenv("VLDS_PT_RANKER")
    args = ["--project", root, "--plugin", PLUGIN, "--now", "2026-11-09 18:00", "--minutes", "0"]
    assert phi_model.check(p4, PLUGIN)["owed"] == []
    sys.path.insert(0, os.path.join(PLUGIN, "scripts"))
    import notices
    assert notices.notify(p4, "json ranker", ValueError("unreadable"), "aaaa1111", now="2026-11-09 18:05")
    assert phi_model.check(p4, PLUGIN)["owed"] == ["train json"]
    assert phi_model.main(["update"] + args) == 1
    assert phi_model.main(["check", "--project", root, "--plugin", PLUGIN]) == 1


@needs_plugin
def test_a_release_notice_owes_nothing_and_stays_open(tmp_path, monkeypatch):
    """A hook's notice is of the family release, which a later version of the plugin answers: check owes no
    training for it, and update, which retrains the logistic regression ranker for a notice of its own beside it,
    answers that one, exits 0 and leaves the release notice open."""
    root = str(tmp_path / "proj")
    p4 = os.path.join(root, ".claude", "p4")
    write_store(p4)
    monkeypatch.delenv("VLDS_PT_RANKER")
    args = ["--project", root, "--plugin", PLUGIN, "--now", "2026-11-09 18:00", "--minutes", "0"]
    assert phi_model.main(["init"] + args) == 0
    sys.path.insert(0, os.path.join(PLUGIN, "scripts"))
    import notices
    assert notices.notice_family("hook stop") == "release"
    assert notices.notify(p4, "hook stop", KeyError("session_id"), "aaaa1111", now="2026-11-09 18:05")
    assert phi_model.check(p4, PLUGIN)["owed"] == []
    assert notices.notify(p4, "json ranker", ValueError("unreadable"), "aaaa1111", now="2026-11-09 18:06")
    assert phi_model.check(p4, PLUGIN)["owed"] == ["train json"]
    assert phi_model.main(["update"] + args) == 0
    assert [n["component"] for n in notices.open_notices(p4)] == ["hook stop"]


@needs_plugin
def test_update_with_nothing_to_train_fails_while_anything_is_stale(tmp_path, monkeypatch, capsys):
    """When check owes no training but owes what only a person does, an untrack of the local build, update trains
    nothing, names what is still stale and exits 1."""
    monkeypatch.delenv("VLDS_PT_RANKER", raising=False)
    p4 = str(tmp_path / "proj" / ".claude" / "p4")
    phi_store_repo(p4)
    with open(os.path.join(p4, ".gitignore"), "w", encoding="utf-8") as f:
        f.write("phi/models/index.json\nphi/models/ranker.pt\n")
    with open(os.path.join(p4, "phi", "models", "ranker.pt"), "wb") as f:
        f.write(b"x")
    git(p4, "add", "-f", ".gitignore", "phi/models/ranker.pt")
    git(p4, "commit", "-qm", "x")
    assert not any(o.startswith("train ") for o in phi_model.check(p4, PLUGIN)["owed"])
    assert phi_model.update(p4, PLUGIN, "2026-11-09 18:00", 0) == 1
    out = capsys.readouterr().out
    assert "update — nothing to train" in out and "update — still stale: untrack phi/models/ranker.pt" in out


@needs_plugin
def test_init_and_train_leave_the_dual_encoder_alone_unless_asked(tmp_path, monkeypatch, capsys):
    """Without VLDS_PT_RANKER=1 init and train succeed, train answers the dual encoder with off, and no checkpoint
    of the dual encoder is written; with two labelled queries, under the three it needs, the plugin's trainer builds
    no logistic regression ranker either."""
    root = str(tmp_path / "proj")
    p4 = os.path.join(root, ".claude", "p4")
    write_store(p4, n_tasks=2)
    monkeypatch.delenv("VLDS_PT_RANKER")
    args = ["--project", root, "--plugin", PLUGIN, "--now", "2026-11-09 18:00", "--minutes", "0"]
    assert phi_model.main(["init"] + args) == 0
    assert phi_model.main(["train", "--only", "ranker"] + args) == 0
    assert "train ranker: off" in capsys.readouterr().out
    assert not any(os.path.exists(layout.sub(p4, "models", n)) for n in ("ranker-anchor.pt", "ranker.pt", "ranker-pt.json"))


@needs_plugin
def test_labels_since_the_round_count_from_this_machines_round(tmp_path):
    """eval.tsv holds every machine's rounds once synced: check counts the labels since this machine's last round,
    a row with no machine counting as any machine's, as the plugin's own train.owed does."""
    p4 = os.path.join(str(tmp_path / "proj"), ".claude", "p4")
    write_store(p4)
    m = recall.modules(PLUGIN, p4)
    sys.path.insert(0, os.path.join(PLUGIN, "scripts"))
    import train
    me = train.machine(p4)
    total = m.labels.count_labels(p4)
    with open(layout.sub(p4, "models", "eval.tsv"), "w", encoding="utf-8") as f:
        f.write("round\ttime\tlabels\tmachine\n")
        f.write(f"1\t2026-11-09 10:00\t{total - 9}\t\n")
        f.write(f"2\t2026-11-09 11:00\t{total - 2}\t{me}\n")
        f.write(f"3\t2026-11-09 12:00\t{total}\tanother-{me}\n")
    rep = phi_model.check(p4, PLUGIN)
    assert rep["labels_since_round"] == 2 and rep["eval_last"]["round"] == "2", rep["eval_last"]
    with open(layout.sub(p4, "models", "eval.tsv"), "a", encoding="utf-8") as f:
        f.write(f"4\t2026-11-09 13:00\t{total - 5}\t\n")
    assert phi_model.check(p4, PLUGIN)["labels_since_round"] == 5


@needs_plugin
def test_the_dual_encoder_is_off_unless_asked(tmp_path, monkeypatch, capsys):
    """Retrieval has never served the dual encoder: without VLDS_PT_RANKER=1 check reports what stands and finds no
    training of it due, on a round or on a notice."""
    p4 = os.path.join(str(tmp_path / "proj"), ".claude", "p4")
    write_store(p4)
    on = phi_model.check(p4, PLUGIN)
    monkeypatch.delenv("VLDS_PT_RANKER")
    rep = phi_model.check(p4, PLUGIN)
    assert rep["pt_off"] and "train ranker" not in rep["owed"]
    assert [o for o in on["owed"] if o not in ("train ranker", "seed .gitignore/.gitattributes")] == rep["owed"]
    phi_model.print_check(rep)
    assert "pt ranker: off" in capsys.readouterr().out
