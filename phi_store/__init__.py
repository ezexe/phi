"""
A .claude/p4 store as training records for small models trained in rounds.

It reads the store the p4-phi plugin writes, its entries under phi/store and
its training files under phi/models, and p4-temportal's memory notes under
temportal/memory, wherever phi_store.layout finds them. A record is one text
with its session, time and weight:

    pretrain   every memory note but MEMORY.md, the index of links
               every entry of the store's rules.md, corrections.md,
               retractions.md and facts.md, its text fields and none of its
               pointers, and every completed row of ezexe's own messages in
               messages/<s8>.md, its fingerprint, addressed and recommended
               fields
               the plan.md and the capped report.md of every folder that holds
               one, wherever it is kept: inside the store, outside the parts
               of the layout and the top folders STORE_PARTS names, and under
               the repository's plans
               directory when read_p4 is given it, whose single-file plans are
               read too; a folder with no takeoff.json is timed by its name,
               as 20261105-1126-... is
    pair       one per such folder that has them: the clearance and plan as the
               prompt, the close note and diff as the target; landed work weighs
               1 and other work less, since the gates failed it for a reason
               one per query of tasks.tsv with a positive in labels.tsv: the
               query's text as the prompt and the texts of its label-1
               documents as the target, a document being a store entry's
               fields, a memory section or a repository file's path; the
               heaviest positive's weight is the record's

split() holds out everything after a cutoff time, so the evaluation is on
sessions the model never saw.

    from phi_store import read_p4, split
    recs = list(read_p4("/path/to/repo/.claude/p4"))
    train, held = split(recs, "2026-10-25 00:00")

Run as a module for a summary: python -m phi_store <store root>
"""

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from . import layout  # the store's layout: phi/ and temportal/

REPORT_CAP = 64_000  # the head of a long report.md carries its findings
NOT_LANDED_WEIGHT = 0.3
STORE_FILES = ("rules.md", "corrections.md", "retractions.md", "facts.md")   # the store's four files of entries
# an entry's pointer fields, never training text: the plugin's store.NOT_TEXT, and state and freed-by beside them
NOT_TEXT = frozenset(("time", "status", "from", "arrival", "pinned", "added", "id", "session", "points-at",
                      "supersedes", "resembles", "state", "freed-by"))
NOT_OWNER_ARRIVAL = re.compile(r"(?i)\s*(task notification|notification|agent message|hook injection)")
STAMPED_NAME_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})\b")
WORK_FILES = (("plan.md", None), ("report.md", REPORT_CAP))   # what makes a folder a piece of planned work
# the top folders of a flat store that other readers cover, or that hold build output, traces and binaries
STORE_PARTS = ("store", "memory", "models", "scratchpad", "sessions", "mrindex")
# the same parts in the store's layout, under phi/ and temportal/
LAYOUT_PARTS = ("store", "models", "archive", "memory", "scratchpad", "tmp", "src")


@dataclass
class Record:
    kind: str  # "pretrain" or "pair"
    source: str  # path within the store
    session: str  # the session ID's 8-hex tail, a work folder's name, or "" for shared memory
    time: str  # "YYYY-MM-DD HH:MM", or "" when the file carries none
    text: str  # pretrain text, or the prompt of a pair
    target: str = ""
    weight: float = 1.0


def replace_retrying(tmp: str, path: str, tries: int = 50) -> None:
    """os.replace, retried: Windows refuses it while a reader, such as a hook, holds the target open."""
    for attempt in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == tries - 1:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
            time.sleep(min(0.002 * (attempt + 1), 0.05))


def write_text_atomic(text: str, path: str) -> None:
    """Written whole or not at all: a temp file beside it, then the retried replace."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    replace_retrying(tmp, path)


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


def folder_pair(folder: Path) -> Optional[Record]:
    """A work folder's ask and plan as the prompt, its close note and diff as the target: any folder that holds
    them, wherever it is kept."""
    takeoff, close = _json(folder / "takeoff.json"), _json(folder / "close.json")
    status = _read(folder / "status") if (folder / "status").exists() else ""
    clearance = re.search(r"^clearance:\s*(.*)$", status, re.M)
    plan = _read(folder / "plan.md") if (folder / "plan.md").exists() else ""
    prompt = "\n\n".join(x for x in (clearance.group(1) if clearance else "", plan) if x)
    diff = _read(folder / "diff.patch") if (folder / "diff.patch").exists() else ""
    target = "\n\n".join(x for x in (close.get("note", ""), diff) if x)
    if not prompt or not target:
        return None
    verdict = close.get("verdict") or status.split("\n", 1)[0].strip()
    return Record("pair", str(folder.name), folder.name, takeoff.get("time", ""), prompt, target,
                  1.0 if verdict == "landed" else NOT_LANDED_WEIGHT)


def folder_time(folder: Path) -> str:
    """takeoff.json's time, or the time in the folder's name: 20261105-1126-land-... gives 2026-11-05 11:26."""
    t = _json(folder / "takeoff.json").get("time", "")
    if t:
        return t
    m = STAMPED_NAME_RE.match(folder.name)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}" if m else ""


def work_folders(root: Path, top_skip=()) -> List[Path]:
    """Every folder under root that holds a plan.md or a report.md, in path order: a piece of planned work and
    what came of it. Nothing here knows where such folders are kept, in a store, a repository's plans directory or
    anywhere else: the files make the folder, not its place. Dot directories are not entered, nor the folders whose
    paths relative to root, with forward slashes, top_skip names: the parts of a store another reader already
    covers, and its build output."""
    out = []
    skip = set(top_skip)
    for d, dirs, files in os.walk(root):
        at_top = Path(d) == root
        here = "" if at_top else Path(d).relative_to(root).as_posix() + "/"
        dirs[:] = sorted(x for x in dirs if not x.startswith(".") and here + x not in skip)
        if not at_top and any(n in files for n, _ in WORK_FILES):
            out.append(Path(d))
    return out


def work_records(root: Path, prefix: str = "", top_skip=()) -> Iterator[Record]:
    """The records of every work folder under root, its files as pretraining text and its pair; `prefix` leads
    each source."""
    for folder in work_folders(root, top_skip):
        time, rel = folder_time(folder), prefix + folder.relative_to(root).as_posix()
        for name, cap in WORK_FILES:
            if (folder / name).exists():
                yield Record("pretrain", f"{rel}/{name}", folder.name, time, _read(folder / name, cap))
        pair = folder_pair(folder)
        if pair:
            yield pair


# --- the store's own files: its entries, its message rows, labels.tsv and tasks.tsv ---------------------------

def sha8(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:8]


def parse_entries(text: str) -> List[Tuple[int, List[Tuple[str, str]]]]:
    """[(line, [(key, value)])] of a store file: a column-0 `- key: value` line opens an entry, two-space `key:`
    lines are its fields, a four-space line continues the field before it; the header prose is skipped. It is
    the plugin's store.parse without its ids, which entry_id computes."""
    out: List[Tuple[int, List[Tuple[str, str]]]] = []
    cur: Optional[List[Tuple[str, str]]] = None
    for n, line in enumerate(text.split("\n"), 1):
        m = re.match(r"^- ([a-z][a-z0-9_-]*):(?: (.*))?$", line)
        if m:
            cur = [(m.group(1), m.group(2) or "")]
            out.append((n, cur))
            continue
        if cur is None:
            continue
        m = re.match(r"^  ([a-z][a-z0-9_-]*):(?: (.*))?$", line)
        if m:
            cur.append((m.group(1), m.group(2) or ""))
        elif line.startswith("    ") and cur:
            k, v = cur[-1]
            cur[-1] = (k, v + "\n" + line[4:])
        elif line.strip():
            cur = None
    return out


def _fields(pairs: List[Tuple[str, str]]) -> Dict[str, str]:
    """{key: first value}."""
    return dict(reversed(pairs))


def entry_id(file: str, pairs: List[Tuple[str, str]]) -> str:
    """The plugin's id of an entry: <file>:<time>:<sha8 of the head value>. The plugin gives a repeat ~2, ~3 and
    so on, and here the first wins."""
    return f"{file}:{_fields(pairs).get('time', '').strip()}:{sha8(pairs[0][1] if pairs else '')}"


def entry_text(pairs: List[Tuple[str, str]]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in pairs if k not in NOT_TEXT and v.strip())


def owner_row(pairs: List[Tuple[str, str]]) -> bool:
    """A completed row of ezexe's own words: addressed, and neither a notification, an agent's message nor a
    system reminder, whose fingerprint opens with `<`."""
    f = _fields(pairs)
    fp = f.get("fingerprint", "").strip().strip('"')
    return bool(f.get("addressed", "").strip()) and not NOT_OWNER_ARRIVAL.match(f.get("arrival", "")) \
        and not fp.lstrip().startswith("<")


def memory_sections(text: str) -> List[Tuple[str, str]]:
    """[(heading, body)] of a memory note, the plugin's way: one per heading, the frontmatter's description, or else
    the first line, heading what comes before the first heading."""
    lines = text.split("\n")
    desc, start = "", 0
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                start = i + 1
                break
            m = re.match(r"^(description|name):\s*(.*)$", lines[i])
            if m and (m.group(1) == "description" or not desc):
                desc = m.group(2).strip()
    pre: List[str] = []
    sections: List[Tuple[str, List[str]]] = []
    cur = pre
    for line in lines[start:]:
        m = re.match(r"^#{1,6}\s+(.*)$", line)
        if m:
            cur = []
            sections.append((m.group(1).strip(), cur))
        else:
            cur.append(line)
    if any(x.strip() for x in pre) or not sections:
        sections.insert(0, (desc or (lines[0].strip() if lines else ""), pre))
    return [(t, " ".join(x.strip() for x in body if x.strip())) for t, body in sections]


def store_entries(p4: Path) -> Dict[str, Tuple[str, str, str]]:
    """{entry id: (source path, time, text)} for every entry of the store's files, every row of its messages/*.md
    and every section of a memory note, under the plugin's ids."""
    out: Dict[str, Tuple[str, str, str]] = {}
    store = layout.sub(p4, "store")
    files = [(name, store / name) for name in STORE_FILES]
    if (store / "messages").is_dir():
        files += [(f"messages/{p.name}", p) for p in sorted((store / "messages").glob("*.md"))]
    for rel, path in files:
        if not path.exists():
            continue
        for line, pairs in parse_entries(_read(path)):
            out.setdefault(entry_id(rel, pairs), (f"store/{rel}:{line}", _fields(pairs).get("time", "").strip(),
                                                  entry_text(pairs)))
    for path in sorted(layout.sub(p4, "memory").glob("*.md")):
        if path.name == "MEMORY.md":
            continue
        for title, body in memory_sections(_read(path)):
            out.setdefault(f"memory/{path.name}::{sha8(title)}", (f"memory/{path.name}", "", f"{title}\n{body}"))
    return out


def _tsv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    lines = _read(path).split("\n")
    head = lines[0].split("\t") if lines and lines[0] else []
    return [dict(zip(head, l.split("\t"))) for l in lines[1:] if l.strip()]


def label_pairs(p4: Path, entries: Optional[Dict[str, Tuple[str, str, str]]] = None) -> Iterator[Record]:
    """One pair per query of tasks.tsv with a label-1 row in labels.tsv: the query's text as the prompt, its
    positives' texts as the target, a repo: document read as its path, and the heaviest positive's weight as the
    record's. A query whose text is a system reminder or an agent's message, opening with `<`, is skipped."""
    entries = store_entries(p4) if entries is None else entries
    tasks = {r["id"]: r for r in _tsv(layout.sub(p4, "models") / "tasks.tsv") if r.get("id")}
    positives: Dict[str, Dict[str, float]] = {}
    for r in _tsv(layout.sub(p4, "models") / "labels.tsv"):
        if str(r.get("label")) == "1" and r.get("task") in tasks:
            try:
                w = float(r.get("weight") or 1.0)
            except ValueError:
                w = 1.0
            got = positives.setdefault(r["task"], {})
            got[r["entry"]] = max(got.get(r["entry"], 0.0), w)
    for tid, task in tasks.items():
        pos = positives.get(tid)
        text = (task.get("text") or "").strip()
        if not pos or not text or text.startswith("<"):
            continue
        texts = []
        for eid in pos:
            if eid in entries:
                texts.append(entries[eid][2])
            elif eid.startswith("repo:"):
                texts.append(eid[5:].replace("#", " "))
            else:
                texts.append(eid)
        yield Record("pair", f"models/labels.tsv:{tid}", task.get("session", ""), task.get("time", "")[:16], text,
                     "\n\n".join(t for t in texts if t), max(pos.values()))


def read_recall_layout(p4: Path) -> Iterator[Record]:
    """The store's own records: one of pretraining text per entry and per completed row of ezexe's messages, then
    the label pairs."""
    entries = store_entries(p4)
    store = layout.sub(p4, "store")
    for name in STORE_FILES:
        if (store / name).exists():
            for line, pairs in parse_entries(_read(store / name)):
                text = entry_text(pairs)
                if text:
                    yield Record("pretrain", f"store/{name}:{line}", "", _fields(pairs).get("time", "").strip(), text)
    if (store / "messages").is_dir():
        for path in sorted((store / "messages").glob("*.md")):
            for line, pairs in parse_entries(_read(path)):
                if owner_row(pairs):
                    yield Record("pretrain", f"store/messages/{path.name}:{line}", path.stem,
                                 _fields(pairs).get("time", "").strip(), entry_text(pairs))
    yield from label_pairs(p4, entries)


def read_p4(root: str, plans: Optional[str] = None) -> Iterator[Record]:
    """Every record of the store at `root`, and of the repository's plans directory `plans` when it is given."""
    p4 = Path(root)
    new = layout.sub(p4, "store").is_dir()

    for path in sorted(layout.sub(p4, "memory").glob("*.md")):
        if new and path.name == "MEMORY.md":
            continue                        # the index of links, not a note
        yield Record("pretrain", path.relative_to(p4).as_posix(), "", "", _read(path))

    if new:
        yield from read_recall_layout(p4)

    # planned work, wherever it is kept: inside the store, and in the repository's plans directory when one is
    # given
    yield from work_records(p4, top_skip=STORE_PARTS + tuple(layout.rel(p4, k) for k in LAYOUT_PARTS))
    if plans and Path(plans).is_dir():
        base = Path(plans)
        in_folders = set()
        for r in work_records(base, "plans/"):
            in_folders.add(r.source)
            yield r
        for path in sorted(base.rglob("*.md")):         # a plan written as one file, not as a folder
            rel = "plans/" + path.relative_to(base).as_posix()
            if rel not in in_folders and not any(part.startswith(".") for part in path.relative_to(base).parts):
                text = _read(path, REPORT_CAP)
                yield Record("pretrain", rel, path.stem, _time_of(text), text)


def split(records: List[Record], cutoff: str) -> Tuple[List[Record], List[Record]]:
    """Records timed before cutoff train; timed at or after it are held out; untimed ones train."""
    train = [r for r in records if not r.time or r.time < cutoff]
    held = [r for r in records if r.time and r.time >= cutoff]
    return train, held


def source_group(source: str) -> str:
    """store/rules.md:5 gives store/rules.md; store/messages/ab.md:9 gives store/messages/; work/x/plan.md gives
    work/; models/labels.tsv:<query> gives models/labels.tsv; temportal/memory/x.md gives temportal/."""
    parts = source.split(":", 1)[0].split("/")
    if parts[0] == "store":
        return "store/messages/" if parts[1] == "messages" else "/".join(parts[:2])
    if parts[0] == "models":
        return "/".join(parts[:2])
    return parts[0] + "/"


def main(argv: Optional[List[str]] = None) -> int:
    """python -m phi_store <store root>: records per kind and per source, and characters."""
    import sys
    from collections import Counter

    argv = list(sys.argv[1:] if argv is None else argv)
    plans = None
    if "--plans" in argv:
        i = argv.index("--plans")
        plans = argv[i + 1] if i + 1 < len(argv) else None
        del argv[i:i + 2]
    if len(argv) != 1:
        print("usage: python -m phi_store <store root> [--plans <the repository's plans directory>]", file=sys.stderr)
        return 2
    root = argv[0]
    recs = list(read_p4(root, plans))
    kinds = Counter(r.kind for r in recs)
    chars = sum(len(r.text) + len(r.target) for r in recs)
    print(f"{len(recs)} records {dict(kinds)}, {chars / 1e6:.1f} M characters")
    print("by source:", dict(sorted(Counter(source_group(r.source) for r in recs).items())))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
