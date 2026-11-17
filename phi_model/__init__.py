"""
Model monitoring and on-demand retraining for a repository's .claude/p4 store: one place where its models are built,
checked, tested, trained and set learning, so the agents working in that repository never have to.

The store is the one the p4-phi plugin writes, with its entries under phi/store, its training files under phi/models
and its settings in p4.json. Its models are the plugin's logistic regression ranker, trained by `p4.py train`, and
this package's asymmetric dual encoder, phi_rank, trained from phi/models/labels.tsv. A session whose ranker file is
unreadable falls back to BM25, and a hook that raises goes on; each appends one row to phi/models/retrain-needed.tsv
and does nothing more. This module is where those rows are read and answered, and phi_model.recall holds the modes,
the check and the trainings:

    python -m phi_model check  --project <repo or its .claude/p4>    what stands, what is stale, what failed
    python -m phi_model init   --project <…> [--mode shadow]          set up, train the logistic regression ranker,
                                                                              and the dual encoder with VLDS_PT_RANKER=1,
                                                                              set the mode
    python -m phi_model train  --project <…> [--only json,ranker]
    python -m phi_model test   --project <…>                          held-out evaluations of both models
    python -m phi_model update --project <…>                          check, retrain what is stale or failed,
                                                                              answer the notices it retrained for, check again
    python -m phi_model mode   --project <…> shadow|active|off       the recall section of p4.json

The modes are phi_model.recall's MODES: shadow and active train at every Stop hook, only active adds retrieval's lines
to a session's context, and off trains nothing.
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
from phi_store import layout  # the store's layout: phi/ and temportal/

# ─── where things are ──────────────────────────────────────────────────────────────────────────────────

LOCAL_BUILD = "models/ranker.pt"     # each machine's build of the checkpoints git keeps: ignored, never tracked


def in_layout(p4: str, text: str) -> str:
    """Store-relative paths in `text`, each `models/...` at a line's start, as the store's layout names them: under
    the plugin's folder in the layout p4-phi writes, at the top in a store an older plugin wrote flat."""
    m = layout.rel(p4, "models")
    return text if m == "models" else re.sub(r"(?m)^models/", m + "/", text)


LFS_LINES = ("# the models' checkpoints in git LFS\nmodels/*.pt filter=lfs diff=lfs merge=lfs -text\n"
             "models/*.phid filter=lfs diff=lfs merge=lfs -text\n")


from . import recall  # noqa: E402  the modes, the check and the trainings


def p4_root(project: str) -> str:
    """The store's root: <project>/.claude/p4, or the path itself when it is a repository that is itself a store,
    holding store/ and models/ in its layout."""
    project = os.path.abspath(project)
    for cand in (os.path.join(project, ".claude", "p4"), project):
        if os.path.isdir(layout.sub(cand, "store")) and os.path.isdir(layout.sub(cand, "models")):
            return cand
    raise SystemExit(f"phi_model — no .claude/p4 store at or under {project}: looked for store/ and models/")


def plugin_version(root: str) -> tuple:
    try:
        with open(os.path.join(root, ".claude-plugin", "plugin.json"), encoding="utf-8") as f:
            return tuple(int(x) for x in json.load(f)["version"].split("."))
    except (OSError, ValueError, KeyError):
        return (0,)


def is_plugin(path: str) -> bool:
    return recall.is_plugin(path)


def find_plugin(given: str = "", project: str = "") -> str:
    """The p4-phi plugin's root. --plugin, then $P4_RECALL_PLUGIN_ROOT, win outright when they name the plugin.
    Otherwise the newest of the installed copies under ~/.claude/plugins and of a marketplace checkout beside the
    project; never a wider sweep of the home directory, whose code this would import."""
    for explicit in (given, os.environ.get("P4_RECALL_PLUGIN_ROOT", "")):
        if explicit:
            if not is_plugin(explicit):
                raise SystemExit(f"phi_model — {explicit} is not the p4-phi plugin: it lacks scripts/p4.py or scripts/train.py")
            return os.path.abspath(explicit)
    home = os.path.expanduser("~")
    cands = []
    # a marketplace checkout beside the project, or beside one of its four nearest ancestors, so that a store checked
    # out under a project's experiments/ still finds the checkout at the project's top
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
    # the marketplaces and plugins Claude Code's registry files name
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
        raise SystemExit("phi_model — no p4-phi plugin was found: pass --plugin <…/lbiza/plugins/p4-phi> or set "
                         "P4_RECALL_PLUGIN_ROOT")
    return max(found, key=plugin_version)


def run(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")


# ─── the recall section of p4.json ─────────────────────────────────────────────────────────────────────

def recall_path(p4: str) -> str:
    """p4.json, as phi_model.recall names it."""
    return recall.config_path(p4)


def read_recall(p4: str) -> Dict[str, str]:
    """The recall section's mode keys, as phi_model.recall reads them."""
    return recall.read_recall(p4)


def write_mode(p4: str, mode: str, now: str) -> str:
    """Set the recall section's mode keys to a mode, every other key of p4.json kept as it is."""
    return recall.write_mode(p4, mode, now)


def mode_of(cfg: Dict[str, str]) -> str:
    return recall.mode_of(cfg)


# ─── check ─────────────────────────────────────────────────────────────────────────────────────────────

def check(p4: str, plugin: str) -> Dict:
    """Everything that stands and everything stale, as one dict; `owed` lists the actions update would take."""
    return recall.check(p4, plugin, ".".join(map(str, plugin_version(plugin))))


def print_check(rep: Dict) -> None:
    return recall.print_check(rep)


# ─── train / test ──────────────────────────────────────────────────────────────────────────────────────

def train(p4: str, plugin: str, only: List[str], now: str, minutes: float) -> Dict[str, Dict]:
    """The plugin's own `p4.py train` for the logistic regression ranker, the trainer its Stop hook runs, and
    `python -m phi_rank train` for the dual encoder, which no hook trains. Their output is printed here."""
    return recall.train(p4, plugin, only, now, minutes)


def resolve_notices(p4: str, plugin: str, trained: Dict[str, Dict], now: str) -> int:
    """Answer the open notices of every model a training just rebuilt, by appending them to
    retrain-resolved.tsv. retrain-needed.tsv is never rewritten, so a notice a hook appends meanwhile is never lost,
    and a model that found nothing to learn answers nothing."""
    return recall.resolve_notices(p4, plugin, trained, now)


def test(p4: str, plugin: str, now: str, minutes: float, folds: int) -> Dict[str, List[str]]:
    """Held-out evaluations of both models. Nothing the store serves is written: the dual encoder's evaluation never
    writes the pretrained base."""
    return recall.test(p4, plugin, now, minutes, folds)


# ─── init / update / mode ──────────────────────────────────────────────────────────────────────────────

def seed_repo(p4: str) -> List[str]:
    """git, the LFS lines of .gitattributes, the local build's line of .gitignore, and git-lfs's hooks. A
    .gitattributes already in the store gets the LFS lines appended when it lacks them, and nothing else in it
    changes."""
    done = []
    if run(["git", "-C", p4, "rev-parse", "--show-toplevel"]).returncode != 0:
        run(["git", "init", "-q", p4])
        done.append("git init")
    name = ".gitattributes"
    dst = os.path.join(p4, name)
    template = in_layout(p4, LFS_LINES)   # p4-phi seeds the .gitignore itself and ships no .gitattributes
    if not os.path.exists(dst):
        with open(dst, "w", encoding="utf-8", newline="\n") as g:
            g.write(template)
        done.append(f"seeded {name}")
    else:
        have = open(dst, encoding="utf-8").read()
        missing = [l for l in template.splitlines() if "filter=lfs" in l and l not in have]
        if missing:
            with open(dst, "a", encoding="utf-8", newline="\n") as g:
                g.write(("" if have.endswith("\n") else "\n")
                        + "# the models' checkpoints in git LFS, added by the p4-model skill\n"
                        + "\n".join(missing) + "\n")
            done.append("added the LFS lines to .gitattributes")
    gi = os.path.join(p4, ".gitignore")
    have = open(gi, encoding="utf-8").read() if os.path.exists(gi) else ""
    if in_layout(p4, LOCAL_BUILD) not in have.split("\n"):
        with open(gi, "a", encoding="utf-8", newline="\n") as g:
            g.write(("" if not have or have.endswith("\n") else "\n")
                    + "# the dual encoder's local build: git keeps its full checkpoint ranker-anchor.pt and its "
                      "incremental checkpoints ranker.rNNN.phid, added by the p4-model skill\n" + in_layout(p4, LOCAL_BUILD) + "\n")
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
        print("seed: " + (", ".join(seed_repo(p4)) or "nothing"))
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
        p.add_argument("--plugin", default="", help="the p4-phi plugin's root, found when omitted")
        p.add_argument("--now", default=time.strftime("%Y-%m-%d %H:%M"))
        if name in ("init", "train", "test", "update"):
            p.add_argument("--minutes", type=float, default=5, help="the dual encoder's pretraining budget")
        if name == "check":
            p.add_argument("--json", action="store_true")
        if name == "init":
            p.add_argument("--mode", default="shadow", choices=sorted(recall.MODES))
        if name == "train":
            p.add_argument("--only", default="json,ranker")
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
        print("seed: " + (", ".join(seed_repo(p4)) or "nothing to seed"))
        trained = train(p4, plugin, ["json", "ranker"], args.now, args.minutes)
        resolve_notices(p4, plugin, trained, args.now)
        print(f"mode {args.mode} → {write_mode(p4, args.mode, args.now)}")
        print_check(check(p4, plugin))
        return 0 if all(v["ok"] for v in trained.values()) else 1
    return update(p4, plugin, args.now, args.minutes)


if __name__ == "__main__":
    sys.exit(main())
