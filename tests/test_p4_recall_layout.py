"""Tests for phi_store and phi_model on the layout the p4-recall plugin writes: no sessions/, the entries in
store/*.md, one message file per session under store/messages/, memory notes, flights with a plan and a log, and
the plugin's labels and tasks under models/."""

import pytest

import phi_model as p4model
from phi_store import main, read_p4, sha8, split

from test_p4 import store, write


def recall_store(tmp_path):
    p4 = tmp_path / "p4"
    write(p4 / "store" / "rules.md",
          "The owner's rulings: what holds past the turn, in the owner's own words.\n"
          "Written when the owner rules something; a later retraction frees a ruling, never an edit here.\n"
          "Fields: ruling, owner-words, scope, form, status, time, pinned, plain.\n\n"
          "- ruling: pin CMake 4.4 everywhere, never kept backward compatible\n"
          '  owner-words: "require the latest cmake"\n'
          "  scope: the AudioPlugin project\n"
          "  status: LIVE\n"
          "  time: 2026-10-29 11:05\n"
          "  from: sessions/macos-setup--150e7eb7/vlds/local-storage.md:8\n\n"
          "- ruling: pull every repo before a long build\n"
          "  status: LIVE\n"
          "  time: 2026-10-28 23:00\n")
    write(p4 / "store" / "corrections.md",
          "Readings that steered work and were wrong.\nWritten when the owner corrects a reading.\n"
          "Fields: key, time, match, meant, delta.\n\n"
          "- key: cmake-floor\n  time: 2026-10-29 11:05\n  match: keep BuildCEF.cmake working under CMake 3.x\n"
          "  meant: require the latest CMake everywhere\n"
          "  delta: cmake_minimum_required(VERSION 4.4) in all six files\n")
    write(p4 / "store" / "facts.md",
          "Claims checked against a source.\nWritten when a claim is verified.\nFields: claim, source, verified, time.\n")
    write(p4 / "store" / "messages" / "ab12cd34.md",
          "Every owner message of one session.\nOpened by the prompt hook and completed by the close hook.\n"
          "Fields: fingerprint, time, arrival, state, addressed, match, freed-by, resembles.\n\n"
          '- fingerprint: "should we pin cmake 4.4 or keep 3.x working?"\n'
          "  time: 2026-11-02 05:16\n  arrival: turn\n  state: FRESH\n"
          "  addressed: Pinned CMake 4.4 in every CMake file.\n"
          "  recommended: You should pin CMake 4.4 everywhere rather than keep 3.x working.\n\n"
          '- fingerprint: "<task-notification>done</task-notification>"\n'
          "  time: 2026-11-02 05:20\n  arrival: notification\n  state: FRESH\n"
          "  addressed: (a notification; complete on arrival)\n\n"
          '- fingerprint: "ship it"\n'
          "  time: 2026-11-02 05:30\n  arrival: turn\n")
    write(p4 / "memory" / "MEMORY.md", "- [cef](cef.md)\n")
    write(p4 / "memory" / "cef.md", "---\nname: cef\ndescription: how a CEF build is watched\n---\n\n"
          "# Watching a CEF build\n\nThe ninja log names objects by directory.\n")
    flight = p4 / "flights" / "20261105-1126-land-pointer-plan-wave5"
    write(flight / "plan.md", "flight: wave5\ngoal: the fences of step 25\n")
    write(flight / "log", "2026-11-05 11:26\ttakeoff\n")
    rule_id = "rules.md:2026-10-29 11:05:" + sha8("pin CMake 4.4 everywhere, never kept backward compatible")
    task = "messages/ab12cd34.md:2026-11-02 05:16:" + sha8('"should we pin cmake 4.4 or keep 3.x working?"')
    mem_id = "memory/cef.md::" + sha8("Watching a CEF build")
    agent = "messages/ab12cd34.md:2026-11-02 05:40:deadbeef"
    empty = "messages/ab12cd34.md:2026-11-02 05:50:feedface"
    write(p4 / "models" / "tasks.tsv", "id\tsession\ttime\ttext\tkind\n"
          f"{task}\tab12cd34\t2026-11-02 05:16\tshould we pin cmake 4.4 or keep 3.x working?\tnew\n"
          f"{agent}\tab12cd34\t2026-11-02 05:40\t<agent-message>a hand-back</agent-message>\tnew\n"
          f"{empty}\tab12cd34\t2026-11-02 05:50\tnothing positive here\tnew\n")
    write(p4 / "models" / "labels.tsv", "task\tentry\tlabel\tsource\tweight\ttime\tteacher\tedited\tnote\n"
          f"{task}\t{rule_id}\t1\tusage\t0.3\t2026-11-02 05:17\t\t\tused\n"
          f"{task}\trepo:src/CMakeLists.txt\t1\trepo\t1.0\t2026-11-02 05:17\t\t\tedited\n"
          f"{task}\trepo:README.md\t0\trepo\t0.3\t2026-11-02 05:17\t\t\tdropped\n"
          f"{task}\t{mem_id}\t1\tusage\t0.3\t2026-11-02 05:17\t\t\tused\n"
          f"{agent}\t{rule_id}\t1\tusage\t0.3\t2026-11-02 05:41\t\t\tused\n"
          f"{empty}\t{rule_id}\t0\tusage\t0.3\t2026-11-02 05:51\t\t\tunused\n")
    return p4, task, rule_id


def test_pretrain_records(tmp_path):
    p4, _task, _rule = recall_store(tmp_path)
    pre = [r for r in read_p4(str(p4)) if r.kind == "pretrain"]
    rule = next(r for r in pre if r.source.startswith("store/rules.md") and "pin CMake" in r.text)
    assert rule.time == "2026-10-29 11:05" and "require the latest cmake" in rule.text
    assert "from:" not in rule.text and "status:" not in rule.text, "pointer fields are not training text"
    assert any(r.source.startswith("store/corrections.md") and "cmake-floor" in r.text for r in pre)
    msgs = [r for r in pre if r.source.startswith("store/messages/ab12cd34.md")]
    assert len(msgs) == 1, "the owner's completed row; never a notification or an open row"
    assert msgs[0].session == "ab12cd34" and msgs[0].time == "2026-11-02 05:16"
    assert "keep 3.x working?" in msgs[0].text and "rather than keep 3.x" in msgs[0].text
    assert any(r.source == "memory/cef.md" for r in pre) and not any(r.source == "memory/MEMORY.md" for r in pre)
    plan = next(r for r in pre if r.source == "flights/20261105-1126-land-pointer-plan-wave5/plan.md")
    assert plan.time == "2026-11-05 11:26", "a flight with no takeoff.json is timed by its name"


def test_label_pairs(tmp_path):
    p4, task, _rule = recall_store(tmp_path)
    pairs = [r for r in read_p4(str(p4)) if r.kind == "pair"]
    assert len(pairs) == 1, "a task with no positive, or whose text is a harness message, is no pair"
    p = pairs[0]
    assert p.source == f"models/labels.tsv:{task}" and p.session == "ab12cd34" and p.time == "2026-11-02 05:16"
    assert p.text == "should we pin cmake 4.4 or keep 3.x working?"
    assert "pin CMake 4.4 everywhere" in p.target, "a store entry's text"
    assert "src/CMakeLists.txt" in p.target, "a repository entry reads as its path"
    assert "ninja log names objects" in p.target, "a memory section's text"
    assert "README.md" not in p.target, "a label-0 entry is not the target"
    assert p.weight == 1.0
    train, held = split(list(read_p4(str(p4))), "2026-11-01 00:00")
    assert p in held and any(r.source.startswith("store/rules.md") for r in train)


def test_module_summary_on_both_layouts(tmp_path, capsys):
    assert main([str(recall_store(tmp_path)[0])]) == 0
    out = capsys.readouterr().out
    assert "pair" in out and "store/" in out
    assert main([str(store(tmp_path))]) == 0
    assert "dispatch rows" in capsys.readouterr().out


def test_store_with_store_and_models_is_a_root(tmp_path):
    (tmp_path / "store").mkdir()
    (tmp_path / "models").mkdir()
    assert p4model.p4_root(str(tmp_path)) == str(tmp_path)
    proj = tmp_path / "proj"
    (proj / ".claude" / "p4" / "store").mkdir(parents=True)
    (proj / ".claude" / "p4" / "models").mkdir()
    assert p4model.p4_root(str(proj)) == str(proj / ".claude" / "p4")
    with pytest.raises(SystemExit):
        p4model.p4_root(str(tmp_path / "models"))
