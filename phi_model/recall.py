"""
The p4-model checkpoint on a store the p4-recall plugin writes (the fresh marketplace, ezexe/p4 lbiza/plugins/p4-recall).

Such a store has `store/` and `models/` at its top and its settings in `p4.json`; its models are the plugin's own
JSON ranker (`models/ranker.json`, trained by `p4.py train`, evaluated by `p4.py eval`) and this package's .pt ranker
(`phi_rank`, its sidecar `models/ranker-pt.json` there, trained from `models/labels.tsv` with no pairs file). There
is no dispatch model: the close hook judges ECHO and SUPERSEDED itself. The plugin leaves its retrain notices in
`models/retrain-needed.tsv` (its scripts/notices.py), which `update` answers.

Modes, written to p4.json's `recall` section (the plugin reads `inject` on every message and `learn` at every Stop):
    shadow — inject off, learn true: recall's lines are computed and logged (usage labels say `shadow`) but never
             shown; every Stop still trains
    active — inject rules-facts, learn true: the rule and fact lines are shown (finding 28); every Stop trains
    off    — inject off, learn false: nothing shown, nothing trained; the models stay on disk, untouched
"""

import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Dict, List

MODES = {"shadow": {"inject": "off", "learn": True},
         "active": {"inject": "rules-facts", "learn": True},
         "off": {"inject": "off", "learn": False}}
MIN_JSON_TASKS = 3            # p4.py train: "fewer than 3 labelled tasks"
MIN_PT_POOLS = 4              # phi_rank folds its threshold on at least four pools


def is_plugin(path: str) -> bool:
    return bool(path) and all(os.path.exists(os.path.join(path, "scripts", n)) for n in ("p4.py", "train.py"))


def is_store(p4: str) -> bool:
    return os.path.isdir(os.path.join(p4, "store")) or os.path.exists(os.path.join(p4, "p4.json"))


def modules(plugin: str):
    """The plugin's labels, notices, p4conf and ranker modules, imported from its scripts/ (standard library only)."""
    d = os.path.join(plugin, "scripts")
    if d not in sys.path:
        sys.path.insert(0, d)
    import labels  # noqa: E402
    import notices  # noqa: E402
    import p4conf  # noqa: E402
    import ranker  # noqa: E402
    return SimpleNamespace(labels=labels, notices=notices, conf=p4conf, ranker=ranker)


def run(cmd: List[str], cwd=None) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")   # the trainers print arrows; Windows consoles are cp1252
    return subprocess.run(cmd, cwd=cwd or tempfile.gettempdir(), env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


# ─── the recall config: p4.json ────────────────────────────────────────────────────────────────────────

def config_path(p4: str) -> str:
    return os.path.join(p4, "p4.json")


def read_recall(p4: str) -> Dict[str, str]:
    """The recall section's mode keys as strings: inject, learn, updated (absent keys left out)."""
    try:
        with open(config_path(p4), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    sec = data.get("recall") if isinstance(data, dict) else None
    if not isinstance(sec, dict):
        return {}
    out = {}
    for k in ("inject", "learn", "updated"):
        if k in sec:
            v = sec[k]
            out[k] = ("true" if v else "false") if isinstance(v, bool) else ("off" if v is False else str(v))
    return out


def write_mode(p4: str, mode: str, now: str) -> str:
    """Set recall.inject and recall.learn in p4.json to the mode, every other key kept; a file the plugin has not
    written yet is created with just the recall section (the plugin merges its defaults over it on first use)."""
    path = config_path(p4)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {"version": 1}
    sec = data.setdefault("recall", {})
    if not isinstance(sec, dict):
        sec = data["recall"] = {}
    sec.update(MODES[mode])
    if now:
        sec["updated"] = f"{now} by the p4-model skill (phi_model mode {mode})"
    from phi_rank import write_text_atomic
    write_text_atomic(json.dumps(data, indent=1, ensure_ascii=False) + "\n", path)
    return path


def mode_of(cfg: Dict[str, str]) -> str:
    inject = str(cfg.get("inject", "rules-facts")).lower()
    learn = str(cfg.get("learn", "true")).lower() not in ("false", "0", "no", "off")
    if inject in ("off", "false", "0", "none"):
        return "shadow" if learn else "off"
    return "active"


# ─── check ─────────────────────────────────────────────────────────────────────────────────────────────

def checkpoint_state(path: str) -> str:
    try:
        with open(path, "rb") as f:
            head = f.read(64)
    except OSError:
        return "missing"
    return "lfs-pointer" if head.startswith(b"version https://git-lfs") else "present"


def load_json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"


def labelled_tasks(m, p4: str) -> int:
    """Tasks with at least one label 1 whose task row exists."""
    tasks = m.labels.read_tasks(p4)
    pos = {r["task"] for r in m.labels.read_tsv(m.labels.mpath(p4, m.labels.TRAIN)) if r.get("label") == "1" and r.get("task") in tasks}
    return len(pos)


def check(p4: str, plugin: str, version: str, server_round) -> Dict:
    """Everything that stands and everything owed, for a p4-recall store; `owed` lists what update would do."""
    no_torch = None
    try:
        from phi_rank import ANCHOR, CURRENT, read_sidecar, sidecar_name
    except ModuleNotFoundError as e:
        # A machine with no torch build (an Intel Mac on Python 3.13 or later has none) still checks everything recall
        # serves: the JSON ranker, the labels, the notices. Only the .pt ranker, which recall does not serve, is out.
        no_torch = e.name
        ANCHOR, CURRENT = "ranker-anchor.pt", "ranker.pt"

        def sidecar_name(p4: str) -> str:
            return "ranker-pt.json" if os.path.isdir(os.path.join(p4, "store")) else "ranker.json"

        def read_sidecar(p4: str) -> Dict:
            side, _err = load_json(os.path.join(p4, "models", sidecar_name(p4)))
            return side if isinstance(side, dict) and not str(side.get("kind", "")).startswith("p4 recall") else {}
    # The .pt ranker is an experiment, off unless VLDS_PT_RANKER=1: recall never served it, and measured on one store
    # (2026-11-11, 67 tasks, held out, every candidate scored by a model fine-tuned without the task) its score as a
    # feature moved the JSON ranker's r-precision from 0.085 to 0.110, 0.078 and 0.100 at 30, 100 and 300 epochs,
    # and ranked alone at 0.013 to 0.028 against term overlap's 0.079. What stands is reported; nothing is owed.
    pt_off = os.environ.get("VLDS_PT_RANKER") != "1"
    skip_pt = bool(no_torch) or pt_off
    m = modules(plugin)
    mdir = os.path.join(p4, "models")
    rep: Dict = {"kind": "p4-recall", "p4": p4, "plugin": plugin, "plugin_version": version, "owed": [], "corrupt": {},
                 "no_torch": no_torch, "pt_off": pt_off, "python": sys.executable}
    cfg = read_recall(p4)
    rep["recall"] = cfg
    rep["mode"] = mode_of(cfg)
    rep["git"] = run(["git", "-C", p4, "rev-parse", "--show-toplevel"]).returncode == 0
    attrs = os.path.join(p4, ".gitattributes")
    rep["lfs"] = os.path.exists(attrs) and "models/*.pt filter=lfs" in open(attrs, encoding="utf-8").read()
    gi = os.path.join(p4, ".gitignore")
    rep["gitignore"] = os.path.exists(gi)
    local_build = rep["gitignore"] and "models/ranker.pt" in open(gi, encoding="utf-8").read().split("\n")
    if not rep["gitignore"] or not rep["lfs"] or not local_build:
        rep["owed"].append("seed .gitignore/.gitattributes")

    labels = m.labels.read_tsv(m.labels.mpath(p4, m.labels.TRAIN))
    rep["labels"] = {}
    for r in labels:
        rep["labels"][r.get("source", "")] = rep["labels"].get(r.get("source", ""), 0) + 1
    rep["tasks"] = len(m.labels.read_tasks(p4))
    n_lab = labelled_tasks(m, p4)
    rep["labelled_tasks"] = n_lab
    owed_labels = int(m.conf.load(p4, write=False)["recall"]["train"].get("owed_labels", 10))

    rpath = os.path.join(mdir, "ranker.json")
    jr = m.ranker.load(rpath)
    if jr is None and os.path.exists(rpath):
        rep["corrupt"]["ranker.json"] = "unreadable, or not a p4 recall ranker"
    rep["json_ranker"] = None if not jr else {"round": jr.get("round"), "trained": jr.get("trained"),
                                               "threshold": jr.get("threshold"), "features": jr.get("features")}
    evals = m.labels.read_tsv(m.labels.mpath(p4, "eval.tsv"))
    last = evals[-1] if evals else {}
    rep["eval_last"] = {k: last.get(k, "") for k in ("round", "time", "labels", "r_precision", "precision", "promoted", "why")} if last else None
    # eval.tsv's labels column is the plugin's count_labels (labels-raw.tsv and labels.tsv together), so the count
    # since the last round is taken the same way; labels.tsv alone read 46 below the round on a store with raw labels
    total = m.labels.count_labels(p4) if hasattr(m.labels, "count_labels") else len(labels)
    try:
        since = total - int(last.get("labels", 0) or 0)
    except ValueError:
        since = total
    rep["labels_since_round"] = since
    if "ranker.json" in rep["corrupt"] or (n_lab >= MIN_JSON_TASKS and (not evals or since >= owed_labels)):
        rep["owed"].append("train json")

    n_pools = n_lab                         # the pools phi_rank would build now: a task with a positive the index holds
    if os.path.exists(os.path.join(mdir, "index.json")):    # no index yet (before the first JSON round): every labelled task
        try:
            from phi_rank import read_label_pools
            n_pools = len(read_label_pools(p4))
        except Exception:  # noqa: BLE001 — a store mid-write: the labelled count stands
            pass
    rep["pt_pools_now"] = n_pools
    side_name = sidecar_name(p4)
    side, err = load_json(os.path.join(mdir, side_name))
    if err:
        rep["corrupt"][side_name] = err
    side = read_sidecar(p4) if side and not err else {}
    state = checkpoint_state(os.path.join(mdir, CURRENT))
    if state == "missing":
        state = checkpoint_state(os.path.join(mdir, ANCHOR))
    rep["pt_ranker"] = None if not side else {
        "round": side.get("round"), "trained": side.get("trained"), "threshold": side.get("threshold"),
        "threshold_from": side.get("threshold_from"), "heldout_rprecision": side.get("heldout_rprecision"),
        "anchor_round": side.get("anchor_round"), "chain": len(side.get("chain", [])), "pools": len(side.get("pools", [])),
        "unseen_pools": max(0, n_pools - len(side.get("pools", []))), "checkpoint": state, "serving_round": server_round(p4)}
    if not skip_pt and (err or (n_pools >= MIN_PT_POOLS and (not side or rep["pt_ranker"]["unseen_pools"]
                                                             or state == "missing"))):
        rep["owed"].append("train ranker")
    if state == "lfs-pointer" and not pt_off:
        rep["owed"].append("git lfs pull (the ranker's checkpoint is a pointer)")
    if run(["git", "-C", p4, "ls-files", "--error-unmatch", "models/ranker.pt"]).returncode == 0:
        rep["owed"].append("untrack models/ranker.pt (git rm --cached; the chain replaces it)")

    notices = m.notices.open_notices(p4)
    rep["notices"] = notices
    for n in notices:
        fam = m.notices.notice_family(n["component"])
        if not (skip_pt and fam == "ranker"):
            rep["owed"].append("train " + ("ranker" if fam == "ranker" else "json"))
    rep["owed"] = list(dict.fromkeys(rep["owed"]))
    return rep


def print_check(rep: Dict) -> None:
    print(f"p4 store {rep['p4']} — p4-recall {rep['plugin_version']} at {rep['plugin']}")
    print(f"  mode: {rep['mode']} ({', '.join(f'{k}: {v}' for k, v in rep['recall'].items()) or 'p4.json has no recall keys: the plugin defaults'})")
    print(f"  git: {'yes' if rep['git'] else 'NO'}; lfs for checkpoints: {'yes' if rep['lfs'] else 'no'}; .gitignore: {'yes' if rep['gitignore'] else 'no'}")
    print(f"  labels: {rep['labels'] or 'none'}; tasks: {rep['tasks']}, with a positive: {rep['labelled_tasks']}; "
          f"labels since the last round: {rep['labels_since_round']}")
    print(f"  json ranker: {json.dumps(rep['json_ranker']) if rep['json_ranker'] else 'none (the no-ranker road)'}; "
          f"last round: {json.dumps(rep['eval_last']) if rep['eval_last'] else 'none'}")
    if rep.get("no_torch"):
        print(f"  pt ranker: not checked or trained here: {rep['no_torch']} is not installed in {rep['python']} "
              "(recall serves without it; `python3 <p4 checkout>/scripts/make_venv.py` installs it where a build exists)")
    elif rep.get("pt_off"):
        print(f"  pt ranker: off, an experiment recall does not serve (VLDS_PT_RANKER=1 trains it); on disk: "
              f"{json.dumps(rep['pt_ranker']) if rep['pt_ranker'] else 'none'}")
    else:
        print(f"  pt ranker: {json.dumps(rep['pt_ranker']) if rep['pt_ranker'] else 'none'}")
    for f, e in rep["corrupt"].items():
        print(f"  CORRUPT models/{f}: {e}")
    for n in rep["notices"]:
        print(f"  NOTICE {n['time']} {n['component']}: {n['kind']} — {n['error'][:120]}")
    print("  owed: " + (", ".join(rep["owed"]) if rep["owed"] else "nothing — every model is current"))


# ─── train / test ──────────────────────────────────────────────────────────────────────────────────────

def train(p4: str, plugin: str, only: List[str], now: str, minutes: float) -> Dict[str, Dict]:
    """The plugin's own `p4.py train` for the JSON ranker and `python -m phi_rank train` (pools from labels.tsv)
    for the .pt ranker, so a model is built here exactly as the Stop hook builds it. `dispatch` is a no-op: a
    p4-recall store has no dispatch model."""
    scripts = os.path.join(plugin, "scripts")
    steps = {"json": [sys.executable, "-I", os.path.join(scripts, "p4.py"), "train", "--p4", p4] + (["--now", now] if now else []),
             "ranker": [sys.executable, "-m", "phi_rank", "train", "--p4", p4, "--minutes", str(minutes)] + (["--now", now] if now else [])}
    out: Dict[str, Dict] = {}
    for name in only:
        if name == "dispatch":
            out[name] = {"ok": True, "code": 0, "output": ["p4-recall has no dispatch model: the close hook judges ECHO and SUPERSEDED"],
                         "empty": True, "busy": False}
            print("train dispatch: nothing to train — " + out[name]["output"][0])
            continue
        if name not in steps:
            out[name] = {"ok": False, "code": 2, "output": [f"unknown model {name}"], "empty": False, "busy": False}
            print(f"train {name}: FAILED — unknown model")
            continue
        r = run(steps[name])
        text = (r.stdout + r.stderr).strip()
        empty = ("fewer than 3 labelled tasks" in text or "no labelled pool" in text or "nothing to learn" in text)
        ok = r.returncode == 0 or empty
        out[name] = {"ok": ok, "code": r.returncode, "output": text.splitlines()[-4:], "empty": empty, "busy": False}
        print(f"train {name}: {'nothing to learn yet' if empty else 'ok' if ok else 'FAILED'} — " + (text.splitlines()[-1] if text else ""))
    return out


def resolve_notices(p4: str, plugin: str, trained: Dict[str, Dict], now: str) -> int:
    """Answer the open notices of every model a training just rebuilt (a hook's or a round's failure counts as the
    JSON ranker's), appending to models/retrain-resolved.tsv; retrain-needed.tsv is never rewritten."""
    m = modules(plugin)
    rebuilt = {k for k, v in trained.items() if v["ok"] and not v.get("empty")}
    fam_of = lambda c: "ranker" if m.notices.notice_family(c) == "ranker" else "json"  # noqa: E731
    answered = [n for n in m.notices.open_notices(p4) if fam_of(n["component"]) in rebuilt]
    if not answered:
        return 0
    res = os.path.join(p4, "models", m.notices.RESOLVED)
    new = not os.path.exists(res)
    with open(res, "a", encoding="utf-8", newline="\n") as f:
        if new:
            f.write("# p4 model retrain notices answered — time\tepoch\tsession\tcomponent\tkind\terror\tresolved\n")
        for n in answered:
            f.write(f"{n['time']}\t{n['epoch']:.6f}\t{n['session']}\t{n['component']}\t{n['kind']}\t{n['error']}\tretrained {now}\n")
    return len(answered)


def test(p4: str, plugin: str, now: str, minutes: float, folds: int, smoke_server) -> Dict[str, List[str]]:
    """Held-out evaluations of both rankers (`p4.py eval`, `phi_rank eval` on the labels' pools) and a live server
    check when a checkpoint exists. Nothing the store serves is written."""
    from phi_rank import ANCHOR, CURRENT
    scripts = os.path.join(plugin, "scripts")
    out: Dict[str, List[str]] = {}
    r = run([sys.executable, "-I", os.path.join(scripts, "p4.py"), "eval", "--p4", p4] + (["--now", now] if now else []))
    out["json ranker"] = (r.stdout + r.stderr).strip().splitlines()[-3:]
    r = run([sys.executable, "-m", "phi_rank", "eval", "--p4", p4, "--minutes", str(minutes), "--folds", str(folds)]
            + (["--now", now] if now else []))
    out[".pt ranker"] = (r.stdout + r.stderr).strip().splitlines()[-2:]
    if "present" in (checkpoint_state(os.path.join(p4, "models", n)) for n in (CURRENT, ANCHOR)):
        out["server"] = [smoke_server(p4)]
    for k, v in out.items():
        print(f"{k}:")
        for l in v:
            print(f"  {l}")
    return out
