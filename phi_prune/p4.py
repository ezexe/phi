"""
A .claude/p4 store as training records for small models trained in rounds.

It reads the layout the vlds plugin writes (sessions/*/vlds, sessions/*/src,
flights/, memory/, mrindex/), as AudioPlugin-p4 holds it. A record is one text
with its session, time and weight:

    pretrain   memory/*.md, sessions/*/vlds/**/*.md (archived runs too),
               sessions/*/src/frags.md and the frags' scripts,
               flights/*/plan.md and report.md (capped)
    pair       one per flight: the clearance and plan as the prompt, the close
               note and diff as the target; landed flights weigh 1, others less
               (the gates fail them for a reason)

A session's records take its mean mind-reader score (mrindex/history.tsv) as a
weight multiplier when it has one. scratchpad/ is skipped: it is build output,
traces and binaries. split() holds out everything after a cutoff time, so the
evaluation is on sessions the model never saw.

    from phi_prune.p4 import read_p4, split
    recs = list(read_p4("/path/to/repo/.claude/p4"))
    train, held = split(recs, "2026-10-25 00:00")

Run as a module for a summary: python -m phi_prune.p4 <p4 root>
"""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterator, List, Optional, Tuple

REPORT_CAP = 64_000  # report.md runs to 470 KB; the head carries the findings
NOT_LANDED_WEIGHT = 0.3
SRC_SUFFIXES = {".py", ".ts", ".js", ".ps1", ".sh", ".cmake"}  # the frags' scripts; wav, png and json run output stay out


@dataclass
class Record:
    kind: str  # "pretrain" or "pair"
    source: str  # path within the p4 tree
    session: str  # session id (the 8-hex tail), flight name, or "" for shared memory
    time: str  # "YYYY-MM-DD HH:MM", or "" when the file carries none
    text: str  # pretrain text, or the prompt of a pair
    target: str = ""
    weight: float = 1.0


def session_id(name: str) -> str:
    """'p4-plugins-and-marketplace-migration--8257a0d2' -> '8257a0d2'."""
    return name.rsplit("--", 1)[-1]


def mind_reader_scores(p4: Path) -> Dict[str, float]:
    """Mean turn score per session id, from mrindex/history.tsv."""
    path = p4 / "mrindex" / "history.tsv"
    if not path.exists():
        return {}
    by_session: Dict[str, List[float]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader((line for line in f if not line.startswith("#")), delimiter="\t"):
            if len(row) >= 3 and row[2]:
                by_session.setdefault(session_id(row[1]), []).append(float(row[2]))
    return {s: mean(v) for s, v in by_session.items()}


def _read(path: Path, cap: Optional[int] = None) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:cap] if cap else text


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _time_of(text: str) -> str:
    m = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", text)
    return m.group(0) if m else ""


def flight_pair(flight: Path) -> Optional[Record]:
    """The flight's ask and plan as the prompt, its close note and diff as the target."""
    takeoff, close = _json(flight / "takeoff.json"), _json(flight / "close.json")
    status = _read(flight / "status") if (flight / "status").exists() else ""
    clearance = re.search(r"^clearance:\s*(.*)$", status, re.M)
    plan = _read(flight / "plan.md") if (flight / "plan.md").exists() else ""
    prompt = "\n\n".join(x for x in (clearance.group(1) if clearance else "", plan) if x)
    diff = _read(flight / "diff.patch") if (flight / "diff.patch").exists() else ""
    target = "\n\n".join(x for x in (close.get("note", ""), diff) if x)
    if not prompt or not target:
        return None
    verdict = close.get("verdict") or status.split("\n", 1)[0].strip()
    return Record("pair", str(flight.name), flight.name, takeoff.get("time", ""), prompt, target,
                  1.0 if verdict == "landed" else NOT_LANDED_WEIGHT)


def read_p4(root: str) -> Iterator[Record]:
    p4 = Path(root)
    scores = mind_reader_scores(p4)

    for path in sorted((p4 / "memory").glob("*.md")):
        yield Record("pretrain", str(path.relative_to(p4)), "", "", _read(path))

    for session in sorted(p for p in (p4 / "sessions").iterdir() if p.is_dir()):
        sid = session_id(session.name)
        w = scores.get(sid, 1.0)
        paths = sorted(session.glob("vlds/**/*.md")) + sorted(
            p for p in session.glob("src/**/*") if p.suffix in SRC_SUFFIXES or p.name == "frags.md")
        for path in paths:
            text = _read(path)
            yield Record("pretrain", str(path.relative_to(p4)), sid, _time_of(text), text, weight=w)

    for flight in sorted(p for p in (p4 / "flights").iterdir() if p.is_dir()):
        time = _json(flight / "takeoff.json").get("time", "")
        for name, cap in (("plan.md", None), ("report.md", REPORT_CAP)):
            if (flight / name).exists():
                yield Record("pretrain", f"flights/{flight.name}/{name}", flight.name, time, _read(flight / name, cap))
        pair = flight_pair(flight)
        if pair:
            yield pair


def pick_pairs(jsonl: str) -> List[dict]:
    """The pool pick pairs the vlds plugin writes (`scripts/pick_model.py pairs --jsonl`): one row per candidate
    barrier row of each labelled pool, with its task, file, kind, state, distilled text and 0/1 label. The plugin
    builds them because it owns the barrier; a ranker trained here reads the same pairs."""
    with open(jsonl, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dispatch_rows(root: str) -> Iterator[dict]:
    """Every session's dispatch rows: the message fingerprint, its time, the barrier's state (FRESH, ECHO or
    SUPERSEDED), the match the operator gave and what was addressed. Labels for an open-moment classifier; on
    AudioPlugin-p4 as of 2026-10-27 they are 3,449 FRESH against 35 ECHO and 10 SUPERSEDED, too few repeats to train on."""
    p4 = Path(root)
    for path in sorted(p4.glob("sessions/*/vlds/dispatch.md")):
        text = _read(path)
        body = text.split("\n---\n", 1)[1] if "\n---\n" in text else ""
        for block in re.split(r"\n(?=- fingerprint:)", body):
            if not block.startswith("- fingerprint:"):
                continue
            fields = dict(re.findall(r"^\s*-?\s*([a-z-]+):\s*(.*)$", block, re.M))
            m = re.match(r"(FRESH|ECHO|SUPERSEDED)\b(?!\s*\|)", fields.get("state", "").strip())
            if not m:   # an open row, or the header's shape line copied as a row
                continue
            state = m.group(1)
            yield {"session": session_id(path.parent.parent.name), "time": fields.get("time", ""),
                   "fingerprint": fields.get("fingerprint", "").strip('"'), "state": state,
                   "match": fields.get("match", ""), "addressed": fields.get("addressed", "")}


def split(records: List[Record], cutoff: str) -> Tuple[List[Record], List[Record]]:
    """Records timed before cutoff train; timed at or after it are held out; untimed ones train."""
    train = [r for r in records if not r.time or r.time < cutoff]
    held = [r for r in records if r.time and r.time >= cutoff]
    return train, held


if __name__ == "__main__":
    import sys
    from collections import Counter

    recs = list(read_p4(sys.argv[1]))
    kinds = Counter(r.kind for r in recs)
    chars = sum(len(r.text) + len(r.target) for r in recs)
    print(f"{len(recs)} records {dict(kinds)}, {chars / 1e6:.1f} M characters")
    print("dispatch rows:", dict(Counter(d["state"] for d in dispatch_rows(sys.argv[1]))))
