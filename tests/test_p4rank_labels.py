"""Tests for phi_rank.read_label_pools: pools built from the p4-recall plugin's models/labels.tsv, tasks.tsv and
index.json (its on-disk shape: doc_fields and rows), and `train` / `eval` with no --pairs."""

import json

import phi_rank as p4rank

TOPICS = ["ring buffer drain", "midi transport playhead", "toolbar latency readout", "crash dump symbols",
          "plan document decisions", "render thread priority"]
FIELDS = ["id", "path", "line", "file", "kind", "time", "state", "show", "plain", "dl", "toks", "head", "match",
          "supersedes", "fkind"]


def doc(did, path, file, kind, time, show, toks=None):
    toks = toks if toks is not None else " ".join(dict.fromkeys(show.lower().split()))
    return dict(zip(FIELDS, [did, path, 1, file, kind, time, "LIVE", show, False, len(toks.split()), toks,
                             toks, None, None, None]))


def labelled_store(root, rules_per_topic=4):
    """Each topic has rules_per_topic rules dated October, one rule dated after every task, and a repository
    file; task k names topic k, labels rules 0 and 1 of its topic 1 and rule 2 of the next topic 0."""
    (root / "models").mkdir(parents=True)
    docs = []
    for i, t in enumerate(TOPICS):
        for j in range(rules_per_topic):
            docs.append(doc(f"rules.md:2026-10-{10 + j:02d} 09:00:r{i}{j}", "store/rules.md", "rules.md", "rule",
                            f"2026-10-{10 + j:02d} 09:00", f"the {t} rule number {j}"))
        docs.append(doc(f"rules.md:2026-12-01 09:00:late{i}", "store/rules.md", "rules.md", "rule",
                        "2026-12-01 09:00", f"the {t} rule from later"))
        docs.append(doc(f"repo:src/{t.replace(' ', '_')}.cc", f"src/{t.replace(' ', '_')}.cc", "repo", "file", "",
                        f"{t} implementation"))
    index = {"version": 6, "n": len(docs), "doc_fields": FIELDS, "docs": [[d.get(k) for k in FIELDS] for d in docs]}
    (root / "models" / "index.json").write_text(json.dumps(index), encoding="utf-8")
    tasks = ["id\tsession\ttime\ttext\tkind"]
    labels = ["task\tentry\tlabel\tsource\tweight\ttime\tteacher\tedited\tnote"]
    for k, t in enumerate(TOPICS):
        tid = f"messages/s{k}.md:2026-11-0{k + 1} 12:00:t{k}"
        tasks.append(f"{tid}\ts{k}\t2026-11-0{k + 1} 12:00\ta plan about the {t}\tnew")
        labels.append(f"{tid}\trules.md:2026-10-10 09:00:r{k}0\t1\tusage\t0.3\t2026-11-0{k + 1} 12:01\t\t\tused")
        labels.append(f"{tid}\trules.md:2026-10-11 09:00:r{k}1\t1\trepo\t1.0\t2026-11-0{k + 1} 12:01\t\t\tedited")
        nxt = (k + 1) % len(TOPICS)
        labels.append(f"{tid}\trules.md:2026-10-12 09:00:r{nxt}2\t0\tusage\t0.3\t2026-11-0{k + 1} 12:01\t\t\tunused")
    tasks.append("messages/s9.md:2026-11-09 12:00:t9\ts9\t2026-11-09 12:00\t<agent-message>a hand-back</agent-message>\tnew")
    labels.append("messages/s9.md:2026-11-09 12:00:t9\trules.md:2026-10-10 09:00:r00\t1\tusage\t0.3\t2026-11-09 12:01\t\t\tused")
    tasks.append("messages/s8.md:2026-11-08 12:00:t8\ts8\t2026-11-08 12:00\tnothing labelled positive\tnew")
    labels.append("messages/s8.md:2026-11-08 12:00:t8\trules.md:2026-10-10 09:00:r00\t0\tusage\t0.3\t2026-11-08 12:01\t\t\tunused")
    (root / "models" / "tasks.tsv").write_text("\n".join(tasks) + "\n", encoding="utf-8")
    (root / "models" / "labels.tsv").write_text("\n".join(labels) + "\n", encoding="utf-8")
    (root / "store").mkdir()
    (root / "store" / "rules.md").write_text("\n".join(f"- ruling: {d['show']}\n  time: {d['time']}\n" for d in docs
                                                       if d["file"] == "rules.md"), encoding="utf-8")


def test_pools_from_labels_and_index(tmp_path):
    labelled_store(tmp_path)
    pools = p4rank.read_label_pools(str(tmp_path), negatives=4)
    assert [p["id"].split(":")[-1] for p in pools] == [f"t{k}" for k in range(len(TOPICS))], \
        "one pool per task with a positive, in time order; never a harness message's task"
    p = pools[0]
    assert p["task"] == f"a plan about the {TOPICS[0]}" and p["time"] == "2026-11-01 12:00" and p["session"] == "s0"
    by_id = {r["id"]: r for r in p["rows"]}
    assert by_id["rules.md:2026-10-10 09:00:r00"]["label"] == 1 and by_id["rules.md:2026-10-11 09:00:r01"]["label"] == 1
    assert by_id["rules.md:2026-10-12 09:00:r12"]["label"] == 0, "a label-0 row is a negative"
    assert sum(r["label"] for r in p["rows"]) == 2
    assert len(p["rows"]) == 3 + 4, "the labelled rows plus `negatives` unlabelled BM25 candidates"
    assert not any("later" in r["text"] for r in p["rows"]), "an entry dated after the task was not there to pick"
    negs = [r for r in p["rows"] if r["id"] not in by_id or r["label"] == 0]
    assert any("ring buffer drain" in r["text"] for r in negs if r["id"].startswith("rules.md:2026-10-1")), \
        "BM25 negatives are the task's own near misses"
    assert all(r["file"] for r in p["rows"]) and by_id["rules.md:2026-10-10 09:00:r00"]["file"] == "store/rules.md"
    repo = [r for p in pools for r in p["rows"] if r["id"].startswith("repo:")]
    assert repo and all(r["text"].startswith("src/") for r in repo), "a repository entry reads as path: show"


def test_train_and_eval_without_pairs(tmp_path):
    labelled_store(tmp_path)
    args = ["--p4", str(tmp_path), "--minutes", "0.05", "--epochs", "30", "--now", "2026-11-09", "--negatives", "6"]
    assert p4rank.main(["train"] + args) == 0
    side = json.loads((tmp_path / "models" / "ranker.json").read_text())
    assert side["round"] == 0 and len(side["pools"]) == len(TOPICS) and side["pools"][0].startswith("messages/s0.md")
    model, _meta = p4rank.load(str(tmp_path / "models" / "ranker.pt"))
    pools = p4rank.read_label_pools(str(tmp_path), negatives=6)
    assert p4rank.mean_rprecision(model, pools) >= 0.8, "it learned the pools it was trained on"
    assert p4rank.main(["eval", "--folds", "3"] + args) == 0


def test_no_labels_is_a_clean_exit(tmp_path, capsys):
    (tmp_path / "models").mkdir()
    assert p4rank.read_label_pools(str(tmp_path)) == []
    assert p4rank.main(["train", "--p4", str(tmp_path), "--minutes", "0"]) == 1
    assert "no labelled pool" in capsys.readouterr().out
