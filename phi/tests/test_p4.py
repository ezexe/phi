"""Tests for phi_store: a .claude/p4 store read as training records."""

import json
import pathlib
import re
import subprocess
import sys

from phi_store import read_p4, split


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def store(tmp_path):
    p4 = tmp_path / "p4"
    write(p4 / "memory" / "rule.md", "commit only the paths you staged")
    for name, verdict, time in (("f-landed", "landed", "2026-10-21 08:00"), ("f-held", "held", "2026-10-26 08:00")):
        fl = p4 / "flights" / name
        write(fl / "takeoff.json", json.dumps({"time": time}))
        write(fl / "close.json", json.dumps({"verdict": verdict, "note": f"{name} note"}))
        write(fl / "status", f"{verdict}\n\nclearance: do the {name} thing\n")
        write(fl / "plan.md", f"plan for {name}")
        write(fl / "diff.patch", "+ a line")
        write(fl / "report.md", "x" * 70_000)
    write(p4 / "scratchpad" / "big.trace.json", "{}")
    return p4


def test_records(tmp_path):
    recs = list(read_p4(str(store(tmp_path))))
    sources = {r.source for r in recs}
    assert "memory/rule.md" in sources
    assert not any("scratchpad" in s for s in sources)
    report = next(r for r in recs if r.source == "flights/f-landed/report.md")
    assert len(report.text) == 64_000
    pairs = {r.session: r for r in recs if r.kind == "pair"}
    assert pairs["f-landed"].weight == 1.0 and pairs["f-held"].weight < 1.0
    assert pairs["f-landed"].text.startswith("do the f-landed thing") and "+ a line" in pairs["f-landed"].target


def test_planned_work_is_read_wherever_it_is_kept(tmp_path):
    """No reader names the folder that holds work folders: a folder is planned work by the files it holds, in the
    store under any name, and in the repository's plans directory when that is given."""
    p4 = store(tmp_path)
    (p4 / "flights").rename(p4 / "work-in-progress")                 # another name inside the store
    plans = tmp_path / "repo" / ".claude" / "plans"
    fl = plans / "flights" / "20261111-0946-wxi-write2"
    write(fl / "plan.md", "plan for wxi")
    write(fl / "status", "landed\n\nclearance: write the wxi thing\n")
    write(fl / "close.json", json.dumps({"verdict": "landed", "note": "wxi note"}))
    write(plans / "ring-drain.md", "# Ring drain\n\nwritten 2026-11-10 09:00\n")
    write(plans / ".archive" / "old.md", "an archive worktree is not entered")
    recs = list(read_p4(str(p4), str(plans)))
    sources = {r.source for r in recs}
    assert "work-in-progress/f-landed/plan.md" in sources and not any(s.startswith("flights/") for s in sources)
    assert "plans/flights/20261111-0946-wxi-write2/plan.md" in sources
    assert "plans/ring-drain.md" in sources and not any("old.md" in s for s in sources)
    assert next(r for r in recs if r.source == "plans/ring-drain.md").time == "2026-11-10 09:00"
    pairs = {r.session: r for r in recs if r.kind == "pair"}
    assert pairs["20261111-0946-wxi-write2"].time == "" and pairs["20261111-0946-wxi-write2"].weight == 1.0
    assert set(pairs) == {"f-landed", "f-held", "20261111-0946-wxi-write2"}
    assert len([r for r in read_p4(str(p4)) if r.source.startswith("plans/")]) == 0      # only when it is given


def test_work_folders_skip_the_layouts_build_output(tmp_path):
    """In the layout p4-phi and p4-temportal write, temportal's scratchpad, tmp and src folders hold build output
    and session files, phi's archive holds what a migration kept verbatim, never planned work, and the store's own
    parts have readers of their own."""
    p4 = tmp_path / "p4"
    for rel in ("phi/store/rules.md", "phi/models/x/plan.md", "temportal/memory/x/plan.md",
                "temportal/scratchpad/job/plan.md", "temportal/tmp/s/part/report.md", "temportal/src/probe/plan.md",
                "phi/archive/s1/plan.md", "work/w1/plan.md"):
        write(p4 / rel, "build output\n")
    assert [r.source for r in read_p4(str(p4))] == ["work/w1/plan.md"]


def test_a_memory_note_has_one_name(tmp_path):
    """A memory note's records name it memory/<name> in any layout, as store_entries does, and group as memory/."""
    from phi_store import source_group, store_entries
    p4 = tmp_path / "p4"
    write(p4 / "phi" / "store" / "rules.md", "- ruling: x\n  time: 2026-11-01 10:00\n")
    write(p4 / "temportal" / "memory" / "cef.md", "---\ndescription: the cef build\n---\nbody\n")
    note = [r.source for r in read_p4(str(p4)) if r.source.endswith("cef.md")]
    assert note == ["memory/cef.md"] and source_group(note[0]) == "memory/"
    assert {s for s, _t, _x in store_entries(p4).values()} >= {"memory/cef.md"}


def test_split(tmp_path):
    p4 = store(tmp_path)
    train, held = split(list(read_p4(str(p4))), "2026-10-25 00:00")
    assert {r.session for r in held} == {"f-held"}
    assert all(not r.time or r.time < "2026-10-25 00:00" for r in train)



def test_the_helpers_phi_rank_and_phi_model_share_are_kept_once_in_phi_store():
    """phi_store needs no torch, so phi_rank and phi_model.recall import the helpers from it and keep no copy."""
    root = pathlib.Path(__file__).resolve().parents[1]
    found = {}
    for package in ("phi_store", "phi_rank", "phi_model"):
        for path in sorted((root / package).glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for name in re.findall(r"^\s*def (_tsv|sidecar_name|read_sidecar|load_json)\b", text, re.M):
                found.setdefault(name, []).append(f"{package}/{path.name}")
    assert found == {"_tsv": ["phi_store/__init__.py"], "sidecar_name": ["phi_store/__init__.py"],
                     "read_sidecar": ["phi_store/__init__.py"], "load_json": ["phi_model/recall.py"]}


def test_the_sidecar_is_read_where_torch_cannot_be_imported(tmp_path):
    """phi_model check reads the dual encoder's sidecar through phi_store on a machine with no torch build."""
    p4 = tmp_path / "p4"
    (p4 / "store").mkdir(parents=True)
    write(p4 / "models" / "ranker-pt.json", json.dumps({"round": 3}))
    write(p4 / "models" / "ranker.json", json.dumps({"kind": "p4 recall ranker"}))
    code = ("import sys; sys.modules['torch'] = None; import phi_store, phi_model.recall; "
            f"print(phi_store.sidecar_name({str(p4)!r}), phi_store.read_sidecar({str(p4)!r})['round'], "
            "'torch' in sys.modules and sys.modules['torch'] is not None)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=pathlib.Path(__file__).resolve().parents[1])
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["ranker-pt.json", "3", "False"]
