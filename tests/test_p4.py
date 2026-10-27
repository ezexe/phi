"""Tests for phi_store: a .claude/p4 store read as training records."""

import json

from phi_store import dispatch_rows, mind_reader_scores, pick_pairs, read_p4, session_id, split


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def store(tmp_path):
    p4 = tmp_path / "p4"
    write(p4 / "memory" / "rule.md", "commit only the paths you staged")
    sess = p4 / "sessions" / "plan-the-ring--abcd1234"
    write(sess / "vlds" / "ledger.md", "# ledger\n\n---\n\n- correction: the ring drains first\n  time: 2026-10-20 10:00\n")
    write(sess / "vlds" / "idb" / "runs" / "data-store-000001.md", "- claim: archived\n  time: 2026-10-01 09:00\n")
    write(sess / "src" / "frags.md", "frag notes")
    write(sess / "src" / "probe-20261020" / "probe.py", "print('probe')")
    write(sess / "src" / "probe-20261020" / "out.wav", "RIFF")
    write(sess / "vlds" / "dispatch.md", "# dispatch\n\n---\n"
          '\n- fingerprint: "plan the ring drain"\n  time: 2026-10-20 09:00\n  arrival: turn\n  state: FRESH\n'
          '  addressed: planned\n'
          '\n- fingerprint: "plan the ring drain"\n  time: 2026-10-20 09:01\n  arrival: turn\n  state: ECHO\n'
          '  match: the 09:00 row, unchanged\n'
          '\n- fingerprint: "a template"\n  time: x\n  state: FRESH | ECHO | SUPERSEDED\n')
    write(p4 / "mrindex" / "history.tsv", "# time\tsession\tscore\n2026-10-20 10:00\tplan-the-ring--abcd1234\t0.5\n"
          "2026-10-20 10:05\tplan-the-ring--abcd1234\t0.25\n")
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
    assert any(s.endswith("idb/runs/data-store-000001.md") for s in sources)
    assert any(s.endswith("probe.py") for s in sources) and not any(s.endswith(".wav") for s in sources)
    assert not any("scratchpad" in s for s in sources)
    ledger = next(r for r in recs if r.source.endswith("ledger.md"))
    assert ledger.session == "abcd1234" and ledger.time == "2026-10-20 10:00" and ledger.weight == 0.375
    report = next(r for r in recs if r.source == "flights/f-landed/report.md")
    assert len(report.text) == 64_000
    pairs = {r.session: r for r in recs if r.kind == "pair"}
    assert pairs["f-landed"].weight == 1.0 and pairs["f-held"].weight < 1.0
    assert pairs["f-landed"].text.startswith("do the f-landed thing") and "+ a line" in pairs["f-landed"].target


def test_split_and_scores(tmp_path):
    p4 = store(tmp_path)
    assert mind_reader_scores(p4) == {"abcd1234": 0.375}
    assert session_id("plan-the-ring--abcd1234") == "abcd1234"
    train, held = split(list(read_p4(str(p4))), "2026-10-25 00:00")
    assert {r.session for r in held} == {"f-held"}
    assert all(not r.time or r.time < "2026-10-25 00:00" for r in train)


def test_dispatch_and_pick_pairs(tmp_path):
    rows = list(dispatch_rows(str(store(tmp_path))))
    assert [r["state"] for r in rows] == ["FRESH", "ECHO"] and rows[1]["match"].startswith("the 09:00 row")
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({"task": "t", "file": "ledger.md", "text": "x", "label": 1}) + "\n\n", encoding="utf-8")
    assert pick_pairs(str(path)) == [{"task": "t", "file": "ledger.md", "text": "x", "label": 1}]
