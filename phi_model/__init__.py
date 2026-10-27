"""
The p4 model checkpoint: one place where a repository's .claude/p4 store gets its models built, checked, tested,
trained and set learning — so the agents working in that repository never have to.

A store's models are the vlds plugin's two JSON models (the pool ranker, scripts/pick_model.py; the dispatch model,
scripts/dispatch_model.py) and this package's .pt ranker (phi_rank, served by phi_serve). Sessions in
the store's repository use them and, on any unexpected model failure, fall back (the JSON ranker, the children, the
operator) and append one row to models/retrain-needed.tsv — nothing more. This module is where those rows are read
and answered:

    python -m phi_model check  --project <repo or its .claude/p4>    what stands, what is behind, what failed
    python -m phi_model init   --project <…> [--mode shadow]          set up, train everything, set learning
    python -m phi_model train  --project <…> [--only json,dispatch,ranker]
    python -m phi_model test   --project <…>                          held-out evals and a live server check
    python -m phi_model update --project <…>                          check, retrain what is behind or failed,
                                                                              clear the notices it answered, check again
    python -m phi_model mode   --project <…> shadow|active|off       the recall config's model settings

Modes, written to the store's recall config (vindex/phi-index.md, ## recall):
    shadow — realtime shadow learning: the children pick and fold every pool while the model's picks are logged
             beside theirs (pool-road: model-shadow), the dispatch model proposes without deciding (dispatch-model:
             shadow), and every turn close retrains on the new labels, in the background, from then on.
    active — the models decide: pool-road: model (the operator reviews the picks; one session in pool-audit runs the
             children), dispatch-model: on. Retrained at every turn close the same way.
    off    — pool-road: children, dispatch-model: off; the models stay on disk, untouched.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from typing import Dict, List, Optional

MODES = {
    "shadow": {"pool-road": "model-shadow", "dispatch-model": "shadow"},
    "active": {"pool-road": "model", "pool-audit": "5", "dispatch-model": "on"},   # pool-audit: 5 only where unset
    "off": {"pool-road": "children", "dispatch-model": "off"},
}
MODE_KEYS = ("pool-road", "pool-audit", "dispatch-model")


# ─── where things are ──────────────────────────────────────────────────────────────────────────────────

LOCAL_BUILD = "models/ranker.pt"     # each machine's build of the chain: ignored, never tracked


def p4_root(project: str) -> str:
    """The store's root: <project>/.claude/p4, or the path itself when it is a store repository (sessions/ and
    vindex/ at its top, as ezexe/AudioPlugin-p4 and ezexe/p4-p4 are)."""
    project = os.path.abspath(project)
    for cand in (os.path.join(project, ".claude", "p4"), project):
        if os.path.isdir(os.path.join(cand, "sessions")):
            return cand
    raise SystemExit(f"p4model — no .claude/p4 store at or under {project} (looked for sessions/)")


def session_store(p4: str) -> str:
    """Any session's store: the plugin's scripts read every session's store from any one of them."""
    stores = sorted(glob.glob(os.path.join(p4, "sessions", "*", "vlds")))
    if not stores:
        raise SystemExit(f"p4model — {p4} has no sessions/*/vlds store yet: run one session with the vlds plugin")
    return stores[0]


def plugin_version(root: str) -> tuple:
    try:
        with open(os.path.join(root, ".claude-plugin", "plugin.json"), encoding="utf-8") as f:
            return tuple(int(x) for x in json.load(f)["version"].split("."))
    except (OSError, ValueError, KeyError):
        return (0,)


def is_plugin(path: str) -> bool:
    return bool(path) and os.path.exists(os.path.join(path, "scripts", "pick_model.py"))


def find_plugin(given: str = "", project: str = "") -> str:
    """The vlds plugin's root. --plugin, then $VLDS_PLUGIN_ROOT, win outright when they name the plugin; otherwise
    the newest of the installed copies (~/.claude/plugins) and of a marketplace checkout beside the project — never a
    wider sweep of the home directory, whose code this would import."""
    for explicit in (given, os.environ.get("VLDS_PLUGIN_ROOT", "")):
        if explicit:
            if not is_plugin(explicit):
                raise SystemExit(f"p4model — {explicit} is not the vlds plugin (no scripts/pick_model.py)")
            return os.path.abspath(explicit)
    home = os.path.expanduser("~")
    cands = []
    for pattern in (os.path.join(home, ".claude", "plugins", "**", "vlds", ".claude-plugin", "plugin.json"),
                    os.path.join(os.path.dirname(os.path.abspath(project or ".")), "*", "Ibiza", "plugins", "vlds",
                                 ".claude-plugin", "plugin.json")):
        cands += [os.path.dirname(os.path.dirname(p)) for p in glob.glob(pattern, recursive=True)]
    found = [c for c in dict.fromkeys(cands) if is_plugin(c)]
    if not found:
        raise SystemExit("p4model — the vlds plugin was not found: pass --plugin <…/Ibiza/plugins/vlds> or set "
                         "VLDS_PLUGIN_ROOT")
    return max(found, key=plugin_version)


def plugin_modules(plugin: str):
    """The plugin's pick_model and dispatch_model modules, imported from its scripts/ (standard library only)."""
    for d in (os.path.join(plugin, "scripts"), os.path.join(plugin, "hooks")):
        if d not in sys.path:
            sys.path.insert(0, d)
    import dispatch_model  # noqa: E402
    import pick_model  # noqa: E402
    return pick_model, dispatch_model


def run(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")


# ─── the recall config ─────────────────────────────────────────────────────────────────────────────────

def recall_path(p4: str) -> str:
    return os.path.join(p4, "vindex", "phi-index.md")


def read_recall(p4: str) -> Dict[str, str]:
    """The `## recall` section's key: value lines."""
    try:
        with open(recall_path(p4), encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return {}
    m = re.search(r"^## recall\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not m:
        return {}
    return {k: v.strip() for k, v in re.findall(r"^([a-z][a-z-]*):\s*(.*)$", m.group(1), re.M)}


def write_mode(p4: str, mode: str, now: str) -> str:
    """Set the recall config's model keys to a mode, every other line of the file kept as it is."""
    path = recall_path(p4)
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        text = "# vlds recall config\n"
    if not re.search(r"^## recall\s*$", text, re.M):
        text = text.rstrip("\n") + "\n\n## recall\n\n"
    h = re.search(r"^## recall\s*$", text, re.M)          # cut at the heading's own position, never a lookalike
    head, rest = text[:h.start()], text[h.end():]
    m = re.search(r"^## ", rest, re.M)
    section, tail = (rest[:m.start()], rest[m.start():]) if m else (rest, "")
    audit = [l for l in section.split("\n") if l.startswith("pool-audit:")]   # the owner's: never removed
    lines = [l for l in section.split("\n") if not re.match(r"^(pool-road|dispatch-model):", l)]
    body = [l for l in lines if l.strip()]
    keep_upd = [l for l in body if l.startswith("updated:")]
    body = [l for l in body if not l.startswith("updated:")]
    body += [f"{k}: {v}" for k, v in MODES[mode].items() if not (k == "pool-audit" and audit)]
    body += [f"updated: {now} by the p4-model skill (phi_model mode {mode})"] if now else keep_upd
    new = head + "## recall\n\n" + "\n".join(body) + "\n" + ("\n" + tail if tail else "")
    from phi_rank import write_text_atomic          # the replace retried while Windows refuses it
    write_text_atomic(new, path)
    return path


def mode_of(cfg: Dict[str, str]) -> str:
    for name, keys in MODES.items():
        if all(cfg.get(k) == v for k, v in keys.items() if k != "pool-audit"):
            return name
    return "custom" if any(k in cfg for k in MODE_KEYS) else "default"


# ─── check ─────────────────────────────────────────────────────────────────────────────────────────────

def server_round(p4: str) -> Optional[int]:
    """The round this store's ranker server is serving, or None — a server naming another store is not this one's."""
    try:
        with open(os.path.join(p4, "models", "ranker.server.json"), encoding="utf-8") as f:
            port = int(json.load(f)["port"])
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            out = json.loads(r.read())
        same = os.path.normcase(os.path.realpath(out.get("p4", ""))) == os.path.normcase(os.path.realpath(p4))
        return out.get("round") if out.get("ok") and same else None
    except Exception:  # noqa: BLE001 — down is an answer
        return None


def load_json(path: str):
    """(data, None) or (None, error): a corrupt or conflicted model file is a finding, never a crash."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"


def check(p4: str, plugin: str) -> Dict:
    """Everything that stands and everything owed, as one dict; `owed` lists the actions update would take."""
    pm, dm = plugin_modules(plugin)
    store = session_store(p4)
    mdir = os.path.join(p4, "models")
    rep: Dict = {"p4": p4, "plugin": plugin, "plugin_version": ".".join(map(str, plugin_version(plugin))),
                 "owed": [], "corrupt": {}}
    cfg = read_recall(p4)
    rep["recall"] = {k: cfg.get(k) for k in MODE_KEYS}
    rep["mode"] = mode_of(cfg)
    rep["git"] = run(["git", "-C", p4, "rev-parse", "--show-toplevel"]).returncode == 0
    attrs = os.path.join(p4, ".gitattributes")
    rep["lfs"] = os.path.exists(attrs) and "models/*.pt filter=lfs" in open(attrs, encoding="utf-8").read()
    gi = os.path.join(p4, ".gitignore")
    rep["gitignore"] = os.path.exists(gi)
    local_build = rep["gitignore"] and LOCAL_BUILD in open(gi, encoding="utf-8").read().split("\n")
    if not rep["gitignore"] or not rep["lfs"] or not local_build:
        rep["owed"].append("seed .gitignore/.gitattributes")

    ids = {p["id"] for p in pm.label_pools(store)}
    rep["labelled_pools"] = len(ids)
    m, err = load_json(os.path.join(mdir, "pick-model.json"))
    if err:
        rep["corrupt"]["pick-model.json"] = err
        rep["pick_model"] = None
        rep["owed"].append("train json")
    elif m:
        rep["pick_model"] = {"round": m.get("round"), "trained": m.get("trained"), "threshold": m.get("threshold"),
                             "unseen_pools": len(ids - set(m.get("seen", [])))}
        if rep["pick_model"]["unseen_pools"]:
            rep["owed"].append("train json")
    else:
        rep["pick_model"] = None
        if ids:
            rep["owed"].append("train json")

    rows = dm.turn_rows_count(store)
    repeats = sum(r[3] for r in dm.labelled(store))
    rep["judged_repeats"] = repeats
    d, err = load_json(os.path.join(mdir, "dispatch-model.json"))
    if err:
        rep["corrupt"]["dispatch-model.json"] = err
        rep["dispatch_model"] = None
        rep["owed"].append("train dispatch")
    elif d:
        rep["dispatch_model"] = {"trained": d.get("trained"), "threshold": d.get("threshold"),
                                 "rows_since": rows - d.get("turn_rows", 0), "repeats": d.get("repeats")}
        if rep["dispatch_model"]["rows_since"] >= 20:
            rep["owed"].append("train dispatch")
    else:
        rep["dispatch_model"] = None
        if repeats:
            rep["owed"].append("train dispatch")

    r, err = load_json(os.path.join(mdir, "ranker.json"))
    state = pm.checkpoint_state(os.path.join(mdir, "ranker.pt"))
    if state == "missing":
        # a clone holds the chain (anchor + φ-deltas), never the local build: the server rebuilds ranker.pt from it
        state = pm.checkpoint_state(os.path.join(mdir, "ranker-anchor.pt"))
    seen = pm.ranker_seen(store)
    if err:
        rep["corrupt"]["ranker.json"] = err
    rep["ranker"] = None if not r else {
        "round": r.get("round"), "trained": r.get("trained"), "threshold": r.get("threshold"),
        "threshold_from": r.get("threshold_from"), "heldout_rprecision": r.get("heldout_rprecision"),
        "anchor_round": r.get("anchor_round"), "chain": len(r.get("chain", [])),
        "unseen_pools": len(ids - (seen or set())), "checkpoint": state, "serving_round": server_round(p4)}
    if err or (r and (rep["ranker"]["unseen_pools"] or state == "missing")) or (not r and ids):
        rep["owed"].append("train ranker")
    if state == "lfs-pointer":
        rep["owed"].append("git lfs pull (the ranker's checkpoint is a pointer)")
    if r and "anchor_round" in r and run(["git", "-C", p4, "ls-files", "--error-unmatch", "models/ranker.pt"]
                                         ).returncode == 0:
        # the chain is what git keeps; ranker.pt is each machine's own build of it
        rep["owed"].append("untrack models/ranker.pt (git rm --cached; the chain replaces it)")
    py = pm.ranker_python(store)
    rep["ranker_python"] = py
    if not py:
        rep["owed"].append("ranker python for this machine (init writes models/.ranker-python)")

    notices = pm.open_notices(store)
    rep["notices"] = notices
    for n in notices:
        rep["owed"].append("train " + pm.notice_family(n["component"]))

    shadow = os.path.join(mdir, "shadow.tsv")
    if os.path.exists(shadow):
        rows_ = [l.rstrip("\n").split("\t") for l in open(shadow, encoding="utf-8") if not l.startswith("#")]
        by: Dict[str, List[float]] = {}
        for row in [x for x in rows_ if len(x) >= 7][-10:]:
            key = (row[7] if len(row) > 7 else "child") + (f" / {row[8]}" if len(row) > 8 else "")
            by.setdefault(key, []).append(float(row[5]))
        rep["shadow_recall"] = {k: round(sum(v) / len(v), 3) for k, v in by.items()}
    for log in ("train.log", "ranker.log"):
        p = os.path.join(mdir, log)
        if os.path.exists(p):
            tail = open(p, encoding="utf-8", errors="replace").read().splitlines()[-40:]
            errs = [l for l in tail if "Traceback" in l or "Error" in l]
            if errs:
                rep.setdefault("log_errors", {})[log] = errs[-3:]
    rep["owed"] = list(dict.fromkeys(rep["owed"]))
    return rep


def print_check(rep: Dict) -> None:
    print(f"p4 store {rep['p4']} — vlds {rep['plugin_version']} at {rep['plugin']}")
    print(f"  mode: {rep['mode']} ({', '.join(f'{k}: {v}' for k, v in rep['recall'].items() if v) or 'nothing named'})")
    print(f"  git: {'yes' if rep['git'] else 'NO'}; lfs for checkpoints: {'yes' if rep['lfs'] else 'no'}; "
          f".gitignore: {'yes' if rep['gitignore'] else 'no'}; ranker python: {rep['ranker_python'] or 'none'}")
    print(f"  labelled pools: {rep['labelled_pools']}; judged repeats: {rep['judged_repeats']}")
    for name in ("pick_model", "dispatch_model", "ranker"):
        print(f"  {name}: {json.dumps(rep[name]) if rep[name] else 'none'}")
    for f, e in rep["corrupt"].items():
        print(f"  CORRUPT models/{f}: {e}")
    if rep.get("shadow_recall"):
        print(f"  shadow recall, last 10 pools (judge / source): {rep['shadow_recall']}")
    for n in rep["notices"]:
        print(f"  NOTICE {n['time']} {n['component']}: {n['kind']} — {n['error'][:120]}")
    for log, errs in rep.get("log_errors", {}).items():
        print(f"  {log}: " + " | ".join(e[:100] for e in errs))
    print("  owed: " + (", ".join(rep["owed"]) if rep["owed"] else "nothing — every model is current"))


# ─── train / test ──────────────────────────────────────────────────────────────────────────────────────

def train(p4: str, plugin: str, only: List[str], now: str, minutes: float) -> Dict[str, Dict]:
    """The plugin's own trainers, so a model is built here exactly as the turn close builds it. Output is printed,
    not logged: the models/*.log files hold only the trainings the turn close started."""
    store = session_store(p4)
    scripts = os.path.join(plugin, "scripts")
    env = dict(os.environ, VLDS_RANKER_PYTHON=sys.executable)
    out: Dict[str, Dict] = {}
    steps = {"json": [sys.executable, os.path.join(scripts, "pick_model.py"), "--store", store, "train", "--now", now],
             "dispatch": [sys.executable, os.path.join(scripts, "dispatch_model.py"), "--store", store, "train",
                          "--now", now],
             "ranker": [sys.executable, os.path.join(scripts, "pick_model.py"), "--store", store, "ranker-train",
                        "--now", now, "--minutes", str(minutes)]}
    for name in only:
        r = run(steps[name], env=env, cwd=tempfile.gettempdir())
        text = (r.stdout + r.stderr).strip()
        # "nothing to learn yet" is no failure: a fresh store has no labelled pool and no judged repeat, and the
        # turn close trains each model by itself once its bootstrap count lands
        empty = ("no labelled pool" in text or "nothing to learn" in text
                 or (name == "dispatch" and "no ECHO or SUPERSEDED" in text))
        busy = "another training holds" in text
        ok = r.returncode == 0 or empty
        out[name] = {"ok": ok and not busy, "code": r.returncode, "output": text.splitlines()[-4:], "empty": empty,
                     "busy": busy}
        state = "busy (a turn-close training is running)" if busy else "nothing to learn yet" if empty \
            else "ok" if ok else "FAILED"
        print(f"train {name}: {state} — " + (text.splitlines()[-1] if text else ""))
    return out


def resolve_notices(p4: str, plugin: str, trained: Dict[str, Dict], now: str) -> int:
    """Answer the open notices of every model a training just rebuilt, by appending them to
    models/retrain-resolved.tsv. retrain-needed.tsv is never rewritten, so a notice a hook appends meanwhile is
    never lost; a model that found nothing to learn answers nothing."""
    pm, _dm = plugin_modules(plugin)
    store = session_store(p4)
    rebuilt = {k for k, v in trained.items() if v["ok"] and not v.get("empty")}
    answered = [n for n in pm.open_notices(store) if pm.notice_family(n["component"]) in rebuilt]
    if not answered:
        return 0
    res = os.path.join(p4, "models", pm.RESOLVED)
    new = not os.path.exists(res)
    with open(res, "a", encoding="utf-8", newline="\n") as f:
        if new:
            f.write("# p4 model retrain notices answered — time\tepoch\tsession\tcomponent\tkind\terror\tresolved\n")
        for n in answered:
            f.write(f"{n['time']}\t{n['epoch']:.6f}\t{n['session']}\t{n['component']}\t{n['kind']}\t"
                    f"{n['error']}\tretrained {now}\n")
    return len(answered)


def test(p4: str, plugin: str, now: str, minutes: float, folds: int) -> Dict[str, List[str]]:
    """Held-out evaluations of all three models and a live server check. Nothing the store serves is touched: the
    .pt evaluation never writes the pretrained base, and the server check uses its own port file."""
    store = session_store(p4)
    scripts = os.path.join(plugin, "scripts")
    out: Dict[str, List[str]] = {}
    r = run([sys.executable, os.path.join(scripts, "pick_model.py"), "--store", store, "eval"], cwd=tempfile.gettempdir())
    out["json ranker"] = (r.stdout + r.stderr).strip().splitlines()[-3:]
    r = run([sys.executable, os.path.join(scripts, "dispatch_model.py"), "--store", store, "eval"],
            cwd=tempfile.gettempdir())
    out["dispatch model"] = (r.stdout + r.stderr).strip().splitlines()[-3:]
    pairs = os.path.join(p4, "models", f".pairs.test.{os.getpid()}.jsonl")
    try:
        run([sys.executable, os.path.join(scripts, "pick_model.py"), "--store", store, "pairs", "--jsonl", pairs],
            cwd=tempfile.gettempdir())
        import phi_rank as p4rank
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            p4rank.main(["eval", "--p4", p4, "--pairs", pairs, "--minutes", str(minutes), "--folds", str(folds),
                         "--now", now])
        out[".pt ranker"] = buf.getvalue().strip().splitlines()[-2:]
    finally:
        try:
            os.remove(pairs)
        except OSError:
            pass
    pm, _dm = plugin_modules(plugin)
    if "present" in (pm.checkpoint_state(os.path.join(p4, "models", n)) for n in ("ranker.pt", "ranker-anchor.pt")):
        out["server"] = [smoke_server(p4)]
    for k, v in out.items():
        print(f"{k}:")
        for l in v:
            print(f"  {l}")
    return out


def smoke_server(p4: str) -> str:
    """A server on the store's checkpoint in a thread, under its own port file (never the live server's), asked
    /health; it stops itself when idle."""
    import threading
    from phi_serve import serve
    info = "ranker.server.test.json"
    ready = threading.Event()
    threading.Thread(target=lambda: serve(p4, 0, 0.01, ready, info_name=info), daemon=True).start()
    if not ready.wait(60):
        return "server: did not start in 60 s"
    try:
        with open(os.path.join(p4, "models", info), encoding="utf-8") as f:
            port = int(json.load(f)["port"])
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
            out = json.loads(r.read())
        return f"server: answered /health, round {out.get('round')}, ok {out.get('ok')}"
    except Exception as e:  # noqa: BLE001
        return f"server: started but /health failed ({type(e).__name__}: {e})"
    finally:
        try:
            os.remove(os.path.join(p4, "models", info))
        except OSError:
            pass


# ─── init / update / mode ──────────────────────────────────────────────────────────────────────────────

def seed_repo(p4: str, plugin: str) -> List[str]:
    """git, the plugin's .gitignore and .gitattributes, git-lfs's hooks, and this machine's interpreter for the .pt
    ranker (models/.ranker-python, untracked). A .gitattributes of the owner's own gets the LFS lines appended when
    it lacks them; nothing else of the owner's is changed."""
    done = []
    if run(["git", "-C", p4, "rev-parse", "--show-toplevel"]).returncode != 0:
        run(["git", "init", "-q", p4])
        done.append("git init")
    for name, src in ((".gitignore", "p4-gitignore"), (".gitattributes", "p4-gitattributes")):
        dst, tpl = os.path.join(p4, name), os.path.join(plugin, "hooks", src)
        if not os.path.exists(tpl):
            continue
        with open(tpl, encoding="utf-8") as f:
            template = f.read()
        if not os.path.exists(dst):
            with open(dst, "w", encoding="utf-8", newline="\n") as g:
                g.write(template)
            done.append(f"seeded {name}")
        elif name == ".gitattributes":
            have = open(dst, encoding="utf-8").read()
            missing = [l for l in template.splitlines() if "filter=lfs" in l and l not in have]
            if missing:
                with open(dst, "a", encoding="utf-8", newline="\n") as g:
                    g.write(("" if have.endswith("\n") else "\n")
                            + "# the models' checkpoints in git LFS (added by the p4-model skill)\n"
                            + "\n".join(missing) + "\n")
                done.append("added the LFS lines to .gitattributes")
    gi = os.path.join(p4, ".gitignore")
    have = open(gi, encoding="utf-8").read() if os.path.exists(gi) else ""
    if LOCAL_BUILD not in have.split("\n"):
        with open(gi, "a", encoding="utf-8", newline="\n") as g:
            g.write(("" if not have or have.endswith("\n") else "\n")
                    + "# the .pt ranker's local build: git keeps its chain, models/ranker-anchor.pt and the "
                      "ranker.rNNN.phid deltas (added by the p4-model skill)\n" + LOCAL_BUILD + "\n")
        done.append(f"added {LOCAL_BUILD} to .gitignore")
    if run(["git", "lfs", "version"]).returncode == 0:
        if run(["git", "-C", p4, "lfs", "install", "--local"]).returncode == 0:
            done.append("git lfs install --local")
    else:
        done.append("git-lfs NOT installed: install it before committing models/*.pt")
    py_file = os.path.join(p4, "models", ".ranker-python")
    os.makedirs(os.path.dirname(py_file), exist_ok=True)
    from phi_rank import write_text_atomic
    write_text_atomic(sys.executable + "\n", py_file)
    done.append(f"models/.ranker-python → {sys.executable}")
    return done


def update(p4: str, plugin: str, now: str, minutes: float) -> int:
    rep = check(p4, plugin)
    print_check(rep)
    if any(o.startswith("seed") or o.startswith("ranker python") for o in rep["owed"]):
        print("seed: " + (", ".join(seed_repo(p4, plugin)) or "nothing"))
    owed = list(dict.fromkeys(o.split()[1] for o in rep["owed"] if o.startswith("train ")))
    if not owed:
        print("update — nothing to train")
        return 0
    trained = train(p4, plugin, owed, now, minutes)
    n = resolve_notices(p4, plugin, trained, now)
    print(f"update — rebuilt {', '.join(k for k, v in trained.items() if v['ok'] and not v['empty']) or 'nothing'}; "
          f"{n} notice(s) answered")
    print_check(check(p4, plugin))
    return 0 if all(v["ok"] for v in trained.values()) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phi_model", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "init", "train", "test", "update", "mode"):
        p = sub.add_parser(name)
        p.add_argument("--project", required=True, help="the repository, or its .claude/p4 store")
        p.add_argument("--plugin", default="", help="the vlds plugin root (found when omitted)")
        p.add_argument("--now", default=time.strftime("%Y-%m-%d %H:%M"))
        if name in ("init", "train", "test", "update"):
            p.add_argument("--minutes", type=float, default=5, help="the .pt ranker's pretraining budget")
        if name == "check":
            p.add_argument("--json", action="store_true")
        if name == "init":
            p.add_argument("--mode", default="shadow", choices=sorted(MODES))
        if name == "train":
            p.add_argument("--only", default="json,dispatch,ranker")
        if name == "test":
            p.add_argument("--folds", type=int, default=5)
        if name == "mode":
            p.add_argument("set", choices=sorted(MODES))
    args = ap.parse_args(argv)
    p4 = p4_root(args.project)
    plugin = find_plugin(args.plugin, args.project)
    if args.cmd == "check":
        rep = check(p4, plugin)
        print(json.dumps(rep, indent=1, default=str)) if args.json else print_check(rep)
        return 1 if rep["owed"] else 0
    if args.cmd == "mode":
        print(f"mode {args.set} → {write_mode(p4, args.set, args.now)}")
        return 0
    if args.cmd == "train":
        trained = train(p4, plugin, [x for x in args.only.split(",") if x], args.now, args.minutes)
        print(f"{resolve_notices(p4, plugin, trained, args.now)} notice(s) answered")
        return 0 if all(v["ok"] for v in trained.values()) else 1
    if args.cmd == "test":
        test(p4, plugin, args.now, args.minutes, args.folds)
        return 0
    if args.cmd == "init":
        print("seed: " + (", ".join(seed_repo(p4, plugin)) or "nothing to seed"))
        trained = train(p4, plugin, ["json", "dispatch", "ranker"], args.now, args.minutes)
        resolve_notices(p4, plugin, trained, args.now)
        print(f"mode {args.mode} → {write_mode(p4, args.mode, args.now)}")
        print_check(check(p4, plugin))
        return 0 if all(v["ok"] for v in trained.values()) else 1
    return update(p4, plugin, args.now, args.minutes)


if __name__ == "__main__":
    sys.exit(main())
