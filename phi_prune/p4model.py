"""
The p4 model checkpoint: one place where a repository's .claude/p4 store gets its models built, checked, tested,
trained and set learning — so the agents working in that repository never have to.

A store's models are the vlds plugin's two JSON models (the pool ranker, scripts/pick_model.py; the dispatch model,
scripts/dispatch_model.py) and this package's .pt ranker (phi_prune.p4rank, served by phi_prune.p4serve). Sessions in
the store's repository use them and, on any unexpected model failure, fall back (the JSON ranker, the children, the
operator) and append one row to models/retrain-needed.tsv — nothing more. This module is where those rows are read
and answered:

    python -m phi_prune.p4model check  --project <repo or its .claude/p4>    what stands, what is behind, what failed
    python -m phi_prune.p4model init   --project <…> [--mode shadow]          set up, train everything, set learning
    python -m phi_prune.p4model train  --project <…> [--only json,dispatch,ranker]
    python -m phi_prune.p4model test   --project <…>                          held-out evals and a live server check
    python -m phi_prune.p4model update --project <…>                          check, retrain what is behind or failed,
                                                                              clear the notices it answered, check again
    python -m phi_prune.p4model mode   --project <…> shadow|active|off       the recall config's model settings

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
import time
import urllib.request
from typing import Dict, List, Optional

MODES = {
    "shadow": {"pool-road": "model-shadow", "dispatch-model": "shadow"},
    "active": {"pool-road": "model", "pool-audit": "5", "dispatch-model": "on"},
    "off": {"pool-road": "children", "dispatch-model": "off"},
}
MODE_KEYS = ("pool-road", "pool-audit", "dispatch-model")


# ─── where things are ──────────────────────────────────────────────────────────────────────────────────

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


def find_plugin(given: str = "", project: str = "") -> str:
    """The vlds plugin's root, newest version first among: --plugin, $VLDS_PLUGIN_ROOT, the installed plugin caches
    under ~/.claude/plugins, and a checkout of the marketplace beside the project or in the home directory."""
    cands = [given, os.environ.get("VLDS_PLUGIN_ROOT", "")]
    home = os.path.expanduser("~")
    for pattern in (os.path.join(home, ".claude", "plugins", "**", "vlds", ".claude-plugin", "plugin.json"),
                    os.path.join(home, "*", "Ibiza", "plugins", "vlds", ".claude-plugin", "plugin.json"),
                    os.path.join(os.path.dirname(os.path.abspath(project or ".")), "*", "Ibiza", "plugins", "vlds",
                                 ".claude-plugin", "plugin.json")):
        cands += [os.path.dirname(os.path.dirname(p)) for p in glob.glob(pattern, recursive=True)]
    found = [c for c in dict.fromkeys(cands) if c and os.path.exists(os.path.join(c, "scripts", "pick_model.py"))]
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
    head, _, rest = text.partition(re.search(r"^## recall\s*$", text, re.M).group(0))
    m = re.search(r"^## ", rest, re.M)
    section, tail = (rest[:m.start()], rest[m.start():]) if m else (rest, "")
    lines = [l for l in section.split("\n") if not re.match(rf"^({'|'.join(MODE_KEYS)}):", l)]
    body = [l for l in lines if l.strip()]
    keep_upd = [l for l in body if l.startswith("updated:")]
    body = [l for l in body if not l.startswith("updated:")]
    body += [f"{k}: {v}" for k, v in MODES[mode].items()]
    body += [f"updated: {now} by the p4-model skill (phi_prune.p4model mode {mode})"] if now else keep_upd
    new = head + "## recall\n\n" + "\n".join(body) + "\n" + ("\n" + tail if tail else "")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(new)
    os.replace(tmp, path)
    return path


def mode_of(cfg: Dict[str, str]) -> str:
    for name, keys in MODES.items():
        if all(cfg.get(k) == v for k, v in keys.items() if k != "pool-audit"):
            return name
    return "custom" if any(k in cfg for k in MODE_KEYS) else "default"


# ─── check ─────────────────────────────────────────────────────────────────────────────────────────────

def server_round(p4: str) -> Optional[int]:
    try:
        with open(os.path.join(p4, "models", "ranker.server.json"), encoding="utf-8") as f:
            port = int(json.load(f)["port"])
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
            out = json.loads(r.read())
        return out.get("round") if out.get("ok") else None
    except Exception:  # noqa: BLE001 — down is an answer
        return None


def check(p4: str, plugin: str) -> Dict:
    """Everything that stands and everything owed, as one dict; `owed` lists the actions update would take."""
    pm, dm = plugin_modules(plugin)
    store = session_store(p4)
    mdir = os.path.join(p4, "models")
    rep: Dict = {"p4": p4, "plugin": plugin, "plugin_version": ".".join(map(str, plugin_version(plugin))),
                 "owed": []}
    cfg = read_recall(p4)
    rep["recall"] = {k: cfg.get(k) for k in MODE_KEYS}
    rep["mode"] = mode_of(cfg)
    git = run(["git", "-C", p4, "rev-parse", "--show-toplevel"])
    rep["git"] = git.returncode == 0
    attrs = os.path.join(p4, ".gitattributes")
    rep["lfs"] = os.path.exists(attrs) and "filter=lfs" in open(attrs, encoding="utf-8").read()
    rep["gitignore"] = os.path.exists(os.path.join(p4, ".gitignore"))
    if not rep["gitignore"] or not rep["lfs"]:
        rep["owed"].append("seed .gitignore/.gitattributes (init)")

    pools = pm.label_pools(store)
    ids = {p["id"] for p in pools}
    rep["labelled_pools"] = len(ids)
    pmodel = os.path.join(mdir, "pick-model.json")
    if os.path.exists(pmodel):
        m = pm.load_model(pmodel)
        rep["pick_model"] = {"round": m.get("round"), "trained": m.get("trained"), "threshold": m.get("threshold"),
                             "unseen_pools": len(ids - set(m.get("seen", [])))}
        if rep["pick_model"]["unseen_pools"]:
            rep["owed"].append("train json")
    else:
        rep["pick_model"] = None
        if ids:
            rep["owed"].append("train json")

    dmodel = os.path.join(mdir, "dispatch-model.json")
    rows = dm.turn_rows_count(store)
    repeats = sum(r[3] for r in dm.labelled(store))
    if os.path.exists(dmodel):
        with open(dmodel, encoding="utf-8") as f:
            d = json.load(f)
        rep["dispatch_model"] = {"trained": d.get("trained"), "threshold": d.get("threshold"),
                                 "rows_since": rows - d.get("turn_rows", 0), "repeats": d.get("repeats")}
        if rep["dispatch_model"]["rows_since"] >= 20:
            rep["owed"].append("train dispatch")
    else:
        rep["dispatch_model"] = None
        if repeats:
            rep["owed"].append("train dispatch")
    rep["judged_repeats"] = repeats

    side = os.path.join(mdir, "ranker.json")
    if os.path.exists(side):
        with open(side, encoding="utf-8") as f:
            r = json.load(f)
        rep["ranker"] = {"round": r.get("round"), "trained": r.get("trained"), "threshold": r.get("threshold"),
                         "unseen_pools": len(ids - set(r.get("pools", []))),
                         "checkpoint": os.path.exists(os.path.join(mdir, "ranker.pt")),
                         "serving_round": server_round(p4)}
        if rep["ranker"]["unseen_pools"] or not rep["ranker"]["checkpoint"]:
            rep["owed"].append("train ranker")
    else:
        rep["ranker"] = None
        if ids:
            rep["owed"].append("train ranker")

    notices = pm.open_notices(store)
    rep["notices"] = notices
    if notices:
        comps = {n["component"] for n in notices}
        rep["owed"] += sorted({"train ranker" if "ranker" in c else "train dispatch" if "dispatch" in c
                               else "train json" for c in comps} - set(rep["owed"]))

    shadow = os.path.join(mdir, "shadow.tsv")
    if os.path.exists(shadow):
        rows_ = [l.rstrip("\n").split("\t") for l in open(shadow, encoding="utf-8") if not l.startswith("#")]
        last = [r for r in rows_ if len(r) >= 7][-10:]
        by: Dict[str, List[float]] = {}
        for r in last:
            by.setdefault(r[7] if len(r) > 7 else "child", []).append(float(r[5]))
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
          f".gitignore: {'yes' if rep['gitignore'] else 'no'}")
    print(f"  labelled pools: {rep['labelled_pools']}; judged repeats: {rep['judged_repeats']}")
    for name in ("pick_model", "dispatch_model", "ranker"):
        print(f"  {name}: {json.dumps(rep[name]) if rep[name] else 'none'}")
    if rep.get("shadow_recall"):
        print(f"  shadow recall, last 10 pools: {rep['shadow_recall']}")
    for n in rep["notices"]:
        print(f"  NOTICE {n['time']} {n['component']}: {n['kind']} — {n['error'][:120]}")
    for log, errs in rep.get("log_errors", {}).items():
        print(f"  {log}: " + " | ".join(e[:100] for e in errs))
    print("  owed: " + (", ".join(rep["owed"]) if rep["owed"] else "nothing — every model is current"))


# ─── train / test ──────────────────────────────────────────────────────────────────────────────────────

def train(p4: str, plugin: str, only: List[str], now: str, minutes: float) -> Dict[str, Dict]:
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
        r = run(steps[name], env=env)
        text = (r.stdout + r.stderr).strip()
        # the dispatch model's "nothing to learn" is no failure: a store with no judged repeat has no model yet
        ok = r.returncode == 0 or (name == "dispatch" and "no ECHO or SUPERSEDED" in text)
        out[name] = {"ok": ok, "code": r.returncode, "output": text.splitlines()[-4:]}
        print(f"train {name}: {'ok' if ok else 'FAILED'} — " + (text.splitlines()[-1] if text else ""))
    return out


def resolve_notices(p4: str, plugin: str, trained: Dict[str, Dict], now: str) -> int:
    """Move the notices a successful training answered to models/retrain-resolved.tsv; the rest stay open."""
    pm, _dm = plugin_modules(plugin)
    path = os.path.join(p4, "models", pm.NOTICES)
    if not os.path.exists(path):
        return 0
    ok = {k for k, v in trained.items() if v["ok"]}

    def answered(component: str) -> bool:
        kind = "ranker" if "ranker" in component else "dispatch" if "dispatch" in component else "json"
        return kind in ok
    keep, done = [], []
    with open(path, encoding="utf-8") as f:
        for l in f:
            parts = l.rstrip("\n").split("\t")
            if l.startswith("#") or len(parts) != 6:
                keep.append(l)
            elif answered(parts[3]):
                done.append(l.rstrip("\n") + f"\tretrained {now}\n")
            else:
                keep.append(l)
    if done:
        res = os.path.join(p4, "models", "retrain-resolved.tsv")
        new = not os.path.exists(res)
        with open(res, "a", encoding="utf-8", newline="\n") as f:
            if new:
                f.write("# p4 model retrain notices answered — time\tepoch\tsession\tcomponent\tkind\terror\tresolved\n")
            f.writelines(done)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(keep)
        os.replace(tmp, path)
    return len(done)


def test(p4: str, plugin: str, now: str, minutes: float, folds: int) -> Dict[str, List[str]]:
    store = session_store(p4)
    scripts = os.path.join(plugin, "scripts")
    out: Dict[str, List[str]] = {}
    r = run([sys.executable, os.path.join(scripts, "pick_model.py"), "--store", store, "eval"])
    out["json ranker"] = (r.stdout + r.stderr).strip().splitlines()[-3:]
    r = run([sys.executable, os.path.join(scripts, "dispatch_model.py"), "--store", store, "eval"])
    out["dispatch model"] = (r.stdout + r.stderr).strip().splitlines()[-3:]
    pairs = os.path.join(p4, "models", ".pairs.jsonl")
    run([sys.executable, os.path.join(scripts, "pick_model.py"), "--store", store, "pairs", "--jsonl", pairs])
    from phi_prune import p4rank
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        p4rank.main(["eval", "--p4", p4, "--pairs", pairs, "--minutes", str(minutes), "--folds", str(folds),
                     "--now", now])
    out[".pt ranker"] = buf.getvalue().strip().splitlines()[-2:]
    if os.path.exists(os.path.join(p4, "models", "ranker.pt")):
        out["server"] = [smoke_server(p4)]
    for k, v in out.items():
        print(f"{k}:")
        for l in v:
            print(f"  {l}")
    return out


def smoke_server(p4: str) -> str:
    """Start a server on the store's checkpoint in a thread, ask /health and /score, report the round served."""
    import threading
    from phi_prune.p4serve import serve
    ready = threading.Event()
    threading.Thread(target=lambda: serve(p4, 0, 0.05, ready), daemon=True).start()
    if not ready.wait(60):
        return "server: did not start in 60 s"
    rnd = server_round(p4)
    return f"server: answered /health, round {rnd}" if rnd is not None else "server: started but /health failed"


# ─── init / update / mode ──────────────────────────────────────────────────────────────────────────────

def seed_repo(p4: str, plugin: str) -> List[str]:
    """git, the plugin's .gitignore and .gitattributes when absent, and git-lfs's hooks in the store's repository."""
    done = []
    if run(["git", "-C", p4, "rev-parse", "--show-toplevel"]).returncode != 0:
        run(["git", "init", "-q", p4])
        done.append("git init")
    for name, src in ((".gitignore", "p4-gitignore"), (".gitattributes", "p4-gitattributes")):
        dst, tpl = os.path.join(p4, name), os.path.join(plugin, "hooks", src)
        if not os.path.exists(dst) and os.path.exists(tpl):
            with open(tpl, encoding="utf-8") as f, open(dst, "w", encoding="utf-8", newline="\n") as g:
                g.write(f.read())
            done.append(f"seeded {name}")
    if run(["git", "lfs", "version"]).returncode == 0:
        if run(["git", "-C", p4, "lfs", "install", "--local"]).returncode == 0:
            done.append("git lfs install --local")
    else:
        done.append("git-lfs NOT installed: install it before committing models/*.pt")
    return done


def update(p4: str, plugin: str, now: str, minutes: float) -> int:
    rep = check(p4, plugin)
    print_check(rep)
    owed = [o.split()[-1] for o in rep["owed"] if o.startswith("train ")]
    if any(o.startswith("seed") for o in rep["owed"]):
        print("seed: " + (", ".join(seed_repo(p4, plugin)) or "nothing"))
    if not owed:
        print("update — nothing owed")
        return 0
    trained = train(p4, plugin, owed, now, minutes)
    n = resolve_notices(p4, plugin, trained, now)
    print(f"update — trained {', '.join(k for k, v in trained.items() if v['ok']) or 'nothing'}; "
          f"{n} notice(s) answered")
    after = check(p4, plugin)
    print_check(after)
    return 0 if all(v["ok"] for v in trained.values()) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phi_prune.p4model", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "init", "train", "test", "update", "mode"):
        p = sub.add_parser(name)
        p.add_argument("--project", required=True, help="the repository, or its .claude/p4 store")
        p.add_argument("--plugin", default="", help="the vlds plugin root (found when omitted)")
        p.add_argument("--now", default=time.strftime("%Y-%m-%d %H:%M"))
        if name in ("init", "train", "test", "update"):
            p.add_argument("--minutes", type=float, default=20, help="the .pt ranker's pretraining budget")
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
