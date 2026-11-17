"""
The p4 model checkpoint: one place where a repository's .claude/p4 store gets its models built, checked, tested,
trained and set learning — so the agents working in that repository never have to.

Two plugins write such stores. The retired vlds plugin's store (sessions/*/vlds, vindex/phi-index.md) has its two
JSON models (the pool ranker, scripts/pick_model.py; the dispatch model, scripts/dispatch_model.py) and this
package's .pt ranker (phi_rank, served by phi_serve). The fresh marketplace's p4-recall store (store/, models/,
p4.json) has the plugin's JSON ranker (`p4.py train`) and the .pt ranker trained from its models/labels.tsv; its
modes, checks and trainings are in phi_model.recall. Either way the sessions use the models and, on any unexpected
model failure, fall back (the JSON ranker, the children, the operator) and append one row to
models/retrain-needed.tsv — nothing more. This module is where those rows are read and answered:

    python -m phi_model check  --project <repo or its .claude/p4>    what stands, what is behind, what failed
    python -m phi_model init   --project <…> [--mode shadow]          set up, train everything, set learning
    python -m phi_model train  --project <…> [--only json,dispatch,ranker]
    python -m phi_model test   --project <…>                          held-out evals and a live server check
    python -m phi_model update --project <…>                          check, retrain what is behind or failed,
                                                                              clear the notices it answered, check again
    python -m phi_model mode   --project <…> shadow|active|off       the recall config's model settings

Modes, written to the store's recall config (vindex/phi-index.md ## recall for a vlds store; p4.json's recall
section for a p4-recall store, see phi_model.recall):
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
from typing import Dict, List, Optional
from phi_store import layout  # noqa: E402  (the store's layout: recall/, temportal/)

# ─── where things are ──────────────────────────────────────────────────────────────────────────────────

LOCAL_BUILD = "models/ranker.pt"     # each machine's build of the chain: ignored, never tracked


def in_layout(p4: str, text: str) -> str:
    """Store-relative paths in `text` (`models/...` at a line's start) as the store's layout names them: under
    recall/ once the plugins moved it, at the top in a store an older plugin wrote flat."""
    m = layout.rel(p4, "models")
    return text if m == "models" else re.sub(r"(?m)^models/", m + "/", text)


LFS_LINES = ("# the models' checkpoints in git LFS\nmodels/*.pt filter=lfs diff=lfs merge=lfs -text\n"
             "models/*.phid filter=lfs diff=lfs merge=lfs -text\n")


from . import recall  # noqa: E402  (the p4-recall branch)


def p4_root(project: str) -> str:
    """The store's root: <project>/.claude/p4, or the path itself when it is a store repository: store/ and models/
    (the p4-recall layout, as ezexe/ap-p4 is)."""
    project = os.path.abspath(project)
    for cand in (os.path.join(project, ".claude", "p4"), project):
        if os.path.isdir(layout.sub(cand, "store")) and os.path.isdir(layout.sub(cand, "models")):
            return cand
    raise SystemExit(f"p4model — no .claude/p4 store at or under {project} (looked for store/ and models/)")


def plugin_version(root: str) -> tuple:
    try:
        with open(os.path.join(root, ".claude-plugin", "plugin.json"), encoding="utf-8") as f:
            return tuple(int(x) for x in json.load(f)["version"].split("."))
    except (OSError, ValueError, KeyError):
        return (0,)


def is_plugin(path: str) -> bool:
    return recall.is_plugin(path)


def find_plugin(given: str = "", project: str = "") -> str:
    """The plugin's root: p4-recall for a store with store/ at its top, vlds for one with sessions/. --plugin, then
    $P4_RECALL_PLUGIN_ROOT, win outright when they name a plugin; otherwise the newest of the
    installed copies (~/.claude/plugins) and of a marketplace checkout beside the project
    — never a wider sweep of the home directory, whose code this would import."""
    for explicit in (given, os.environ.get("P4_RECALL_PLUGIN_ROOT", "")):
        if explicit:
            if not is_plugin(explicit):
                raise SystemExit(f"p4model — {explicit} is not recall's plugin, p4-phi or p4-recall (scripts/p4.py and train.py)")
            return os.path.abspath(explicit)
    home = os.path.expanduser("~")
    cands = []
    # a marketplace checkout beside the project, or beside one of its four nearest ancestors (a store checked out
    # under a project's experiments/ still finds the checkout at the project's top)
    ancestors, d = [], os.path.dirname(os.path.abspath(project or "."))
    for _ in range(5):
        ancestors.append(d)
        up = os.path.dirname(d)
        if up == d:
            break
        d = up
    for name in ("p4-phi",):
        for pattern in (os.path.join(home, ".claude", "plugins", "**", name, ".claude-plugin", "plugin.json"),
                        *(os.path.join(a, "*", "lbiza", "plugins", name, ".claude-plugin", "plugin.json")
                          for a in ancestors)):
            cands += [os.path.dirname(os.path.dirname(p)) for p in glob.glob(pattern, recursive=True)]
    # the marketplaces Claude Code knows (a directory marketplace is read in place, never copied under cache/)
    for reg, key in (("known_marketplaces.json", "installLocation"), ("installed_plugins.json", "installPath")):
        try:
            with open(os.path.join(home, ".claude", "plugins", reg), encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        entries = data.get("plugins", {}).values() if reg == "installed_plugins.json" else data.values()
        for entry in entries:
            for rec in (entry if isinstance(entry, list) else [entry]):
                loc = rec.get(key, "") if isinstance(rec, dict) else ""
                if not loc:
                    continue
                cands += [loc] if is_plugin(loc) else [os.path.join(loc, "plugins", "p4-phi")]
    found = [c for c in dict.fromkeys(cands) if is_plugin(c)]
    if not found:
        raise SystemExit("p4model — no p4-recall plugin was found: pass --plugin <…/lbiza/plugins/p4-phi> or set "
                         "P4_RECALL_PLUGIN_ROOT")
    return max(found, key=plugin_version)


def run(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")


# ─── the recall config ─────────────────────────────────────────────────────────────────────────────────

def recall_path(p4: str) -> str:
    """p4.json (phi_model.recall)."""
    return recall.config_path(p4)


def read_recall(p4: str) -> Dict[str, str]:
    """p4.json's recall mode keys (phi_model.recall)."""
    return recall.read_recall(p4)


def write_mode(p4: str, mode: str, now: str) -> str:
    """Set the recall config's model keys to a mode, every other key of the file kept as it is."""
    return recall.write_mode(p4, mode, now)


def mode_of(cfg: Dict[str, str]) -> str:
    return recall.mode_of(cfg)


# ─── check ─────────────────────────────────────────────────────────────────────────────────────────────

def check(p4: str, plugin: str) -> Dict:
    """Everything that stands and everything owed, as one dict; `owed` lists the actions update would take."""
    return recall.check(p4, plugin, ".".join(map(str, plugin_version(plugin))))


def print_check(rep: Dict) -> None:
    return recall.print_check(rep)


# ─── train / test ──────────────────────────────────────────────────────────────────────────────────────

def train(p4: str, plugin: str, only: List[str], now: str, minutes: float) -> Dict[str, Dict]:
    """The plugin's own trainers, so a model is built here exactly as the turn close builds it. Output is printed,
    not logged: the models/*.log files hold only the trainings the turn close started."""
    return recall.train(p4, plugin, only, now, minutes)


def resolve_notices(p4: str, plugin: str, trained: Dict[str, Dict], now: str) -> int:
    """Answer the open notices of every model a training just rebuilt, by appending them to
    models/retrain-resolved.tsv. retrain-needed.tsv is never rewritten, so a notice a hook appends meanwhile is
    never lost; a model that found nothing to learn answers nothing."""
    return recall.resolve_notices(p4, plugin, trained, now)


def test(p4: str, plugin: str, now: str, minutes: float, folds: int) -> Dict[str, List[str]]:
    """Held-out evaluations of all three models. Nothing the store serves is touched: the
    .pt evaluation never writes the pretrained base."""
    return recall.test(p4, plugin, now, minutes, folds)


# ─── init / update / mode ──────────────────────────────────────────────────────────────────────────────

def seed_repo(p4: str, plugin: str) -> List[str]:
    """git, the plugin's .gitignore and .gitattributes, and git-lfs's hooks. A .gitattributes of the owner's own gets the LFS lines appended when
    it lacks them; nothing else of the owner's is changed."""
    done = []
    if run(["git", "-C", p4, "rev-parse", "--show-toplevel"]).returncode != 0:
        run(["git", "init", "-q", p4])
        done.append("git init")
    for name, src in ((".gitignore", "p4-gitignore"), (".gitattributes", "p4-gitattributes")):
        dst, tpl = os.path.join(p4, name), os.path.join(plugin, "hooks", src)
        if os.path.exists(tpl):
            with open(tpl, encoding="utf-8") as f:
                template = f.read()
        elif name == ".gitattributes":
            template = in_layout(p4, LFS_LINES)   # p4-recall seeds the .gitignore itself and ships no .gitattributes
        else:
            continue
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
    if in_layout(p4, LOCAL_BUILD) not in have.split("\n"):
        with open(gi, "a", encoding="utf-8", newline="\n") as g:
            g.write(("" if not have or have.endswith("\n") else "\n")
                    + "# the .pt ranker's local build: git keeps its chain, models/ranker-anchor.pt and the "
                      "ranker.rNNN.phid deltas (added by the p4-model skill)\n" + in_layout(p4, LOCAL_BUILD) + "\n")
        done.append(f"added {in_layout(p4, LOCAL_BUILD)} to .gitignore")
    if run(["git", "lfs", "version"]).returncode == 0:
        if run(["git", "-C", p4, "lfs", "install", "--local"]).returncode == 0:
            done.append("git lfs install --local")
    else:
        done.append("git-lfs NOT installed: install it before committing models/*.pt")
    return done


def update(p4: str, plugin: str, now: str, minutes: float) -> int:
    rep = check(p4, plugin)
    print_check(rep)
    if any(o.startswith("seed") for o in rep["owed"]):
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
        p.add_argument("--plugin", default="", help="the p4-recall plugin root (found when omitted)")
        p.add_argument("--now", default=time.strftime("%Y-%m-%d %H:%M"))
        if name in ("init", "train", "test", "update"):
            p.add_argument("--minutes", type=float, default=5, help="the .pt ranker's pretraining budget")
        if name == "check":
            p.add_argument("--json", action="store_true")
        if name == "init":
            p.add_argument("--mode", default="shadow", choices=sorted(recall.MODES))
        if name == "train":
            p.add_argument("--only", default="json,dispatch,ranker")
        if name == "test":
            p.add_argument("--folds", type=int, default=5)
        if name == "mode":
            p.add_argument("set", choices=sorted(recall.MODES))
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
