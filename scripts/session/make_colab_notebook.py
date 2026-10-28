"""
Generates .docs/p4/q_matrix_colab.ipynb (without outputs): the p4 Q-matrix claims, proved where they are exact and
measured where they are not, on Google Colab, with every result, cache, model and log kept on Google Drive so a
disconnect or a runtime reset loses nothing. Re-running the notebook resumes from what Drive holds.

    python scripts/session/make_colab_notebook.py
"""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# The p4 Q matrix, tested on Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ezexe/phi/blob/main/.docs/p4/q_matrix_colab.ipynb)

This notebook checks what [`.docs/p4/nested-store.md`](https://github.com/ezexe/phi/blob/main/.docs/p4/nested-store.md) claims about the p4 store and its models. Every claim is either proved or measured, and every result lands on your Google Drive.

1. **Exact proofs**, in seconds:
   - The nesting rule (no p4 directly in a p4) is the matrix M = [[1,1],[1,0]].
   - The number of store addresses of each length is a Fibonacci number.
   - Each address is a distinct Zeckendorf integer.
   - Cassini's identity holds.
   - `phi_mask` never keeps two adjacent rows.
   - The address stream needs no length field.
2. **The labelled pairs** of the store, exported by the vlds plugin.
3. **Nested against one model** (`scripts/session/nested_suite.py`):
   - Runs at k = 2, 3, 4, 5, 6 and 8 nodes, on two seeds.
   - Answers the open question: is the k = 4 win a property of the Q matrix's 4 cells, or of shard size?
   - Measures whether the matrix route finds the best node above chance at every k.
   - Measures whether φ routing beats 2:4 routing.
4. **The effort tiers** on the store (`phi_rank.tree`):
   - The tree's held-out score at medium and at high.
   - Whether the JSON ranker keeps the tree's `.pt` weight.
5. **A summary** with the verdicts and a plot, written to Drive.
6. **Optional:** commit the results into ezexe/phi on a branch.

**Disconnects.** Colab can drop the runtime, the local disk and every variable at any time, so:
- Everything is written under `DRIVE_DIR/RUN` on your Drive as it is produced: results, model caches, logs and the pinned commit of every repository.
- Each step records itself as done in `state.json` on Drive.
- After a disconnect, reconnect and choose **Runtime → Run all**:
  - Finished steps are skipped.
  - The suite resumes from its last saved model, fold set or arm.
  - The repositories are re-cloned at the same pinned commits, so the numbers stay comparable across sessions.

**Runtime: choose the T4 GPU** (Runtime → Change runtime type → T4 GPU).
- `phi_rank` trains on a CUDA GPU when it finds one. Section 2 runs a short self-check on it first, and falls back to the CPU, saying so, if the check fails.
- The TPU runtime gives no speedup here: the models are small embedding tables that need `torch_xla` to use a TPU at all. On a TPU runtime the notebook trains on its CPU.
- Each run pretrains for a fixed number of minutes, so faster hardware trains more steps and every arm improves. Compare arms within this notebook's run, not against the figures in the doc, which came from an 8-core CPU.
- Keep one runtime type for the whole plan. The notebook records the device each run used, and warns if a resumed session runs on a different one.
- The full plan (6 node counts × 2 seeds) may span several Colab sessions; the resume makes that safe.

**Secrets.** The store repository is private. Add a GitHub token as a Colab secret named `GITHUB_TOKEN` (🔑 in the left bar), with read access to the store repository, plus write access to ezexe/phi if you set `PUSH_RESULTS`. The token is read from Colab's secrets and never printed or saved.
"""))

cells.append(md("## 0. Settings\nEdit, then **Runtime → Run all**. A new `RUN` name starts a fresh run; the same name resumes."))
cells.append(code('''DRIVE_DIR = "/content/drive/MyDrive/phi-colab"   # everything this notebook keeps lives under DRIVE_DIR/RUN
RUN = "q-matrix-1"

STORE_REPO = "ezexe/AudioPlugin-p4"   # the p4 store measured (the doc's figures come from this store)
STORE_BRANCH = "master"
STORE_PATH = ""                       # or a store already on disk/Drive; then STORE_REPO is not cloned
PHI_REF = "main"                      # ezexe/phi: the models and the suite
P4_REF = "main"                       # ezexe/p4: the vlds plugin (pairs export, the JSON ranker)

K_LIST = [2, 3, 4, 5, 6, 8]           # node counts for the nested-vs-one suite
SEEDS = [0, 1]
MINUTES = 6.0                         # pretraining budget per seed (the big model's; each node gets MINUTES / k)
EPOCHS = 10
FOLDS = 4
WARM = 8

TREE_K = 4                            # the effort tiers' tree
TREE_MINUTES = 6.0

DEVICE = "auto"                       # auto: the GPU when the runtime has one; "cpu" forces the CPU

PUSH_RESULTS = False                  # True: commit the results into ezexe/phi on PUSH_BRANCH (a PR is yours to open)
PUSH_BRANCH = "colab/q-matrix-results"
'''))

cells.append(md("## 1. Drive, state and logging\nMounts Drive and creates the run's folders. `state.json` records each finished step; every command's output is appended to a log on Drive as it runs."))
cells.append(code('''import datetime, json, os, pathlib, shutil, subprocess, sys, time

try:
    from google.colab import drive
    drive.mount("/content/drive")
    IN_COLAB = True
except ImportError:                    # run elsewhere: DRIVE_DIR is an ordinary folder
    IN_COLAB = False

ROOT = pathlib.Path(DRIVE_DIR) / RUN
for d in ("results", "cache", "logs", "models", "pairs"):
    (ROOT / d).mkdir(parents=True, exist_ok=True)
STATE = ROOT / "state.json"
WORK = pathlib.Path("/content/work" if IN_COLAB else ROOT.parent / (RUN + "-work"))
WORK.mkdir(parents=True, exist_ok=True)


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def save_json(path, obj):
    """Write through a temporary file and rename, so a disconnect mid-write never leaves half a file on Drive."""
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def load_state():
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"created": now(), "done": {}}


def mark(step, **info):
    st = load_state()
    st.setdefault("done", {})[step] = {"at": now(), **info}
    save_json(STATE, st)


def done(step):
    return step in load_state().get("done", {})


def run(cmd, log, cwd=None, env=None, check=True):
    """Run a command, echo its output here and append it to logs/<log>.log on Drive line by line."""
    path = ROOT / "logs" / f"{log}.log"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\\n# {now()} $ {' '.join(map(str, cmd))}\\n")
        f.flush()
        p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env={**os.environ, **(env or {})},
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
        code = p.wait()
        f.write(f"# exit {code}\\n")
    if check and code != 0:
        raise RuntimeError(f"{cmd[0]} exited {code}; see {path}")
    return code


print("run folder:", ROOT)
print("done so far:", ", ".join(load_state().get("done", {})) or "nothing")
'''))

cells.append(md("## 2. Repositories, pinned\nClones ezexe/phi, ezexe/p4 and the store into the runtime's disk. The first session records each commit in `state.json`; later sessions check out those same commits, so a resumed run measures the same code on the same data. Delete `pins` from `state.json` to move to newer commits."))
cells.append(code('''def token():
    if not IN_COLAB:
        return os.environ.get("GITHUB_TOKEN", "")
    from google.colab import userdata
    try:
        return userdata.get("GITHUB_TOKEN") or ""
    except Exception:                  # no such secret, or access not granted to this notebook
        return ""


def url(repo):
    t = token()
    return f"https://x-access-token:{t}@github.com/{repo}.git" if t else f"https://github.com/{repo}.git"


def checkout(repo, ref, dest, pin):
    """Clone with full history (the shards are cut by when git first added each file) but no LFS objects, then
    check out the pinned commit, or `ref` on the first session."""
    env = {"GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    if not (dest / ".git").exists():
        r = subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", url(repo), str(dest)],
                           env={**os.environ, **env}, capture_output=True, text=True)
        if r.returncode != 0:          # the error may hold the URL, so the token is stripped before printing
            raise RuntimeError(f"cloning {repo} failed: " + r.stderr.replace(token() or "\\0", "***"))
    target = pin or f"origin/{ref}"
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", "--force", target], env={**os.environ, **env}, check=True)
    return subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


st = load_state()
pins = st.get("pins", {})
PHI = WORK / "phi"
P4 = WORK / "p4"
pins["phi"] = checkout("ezexe/phi", PHI_REF, PHI, pins.get("phi"))
pins["p4"] = checkout("ezexe/p4", P4_REF, P4, pins.get("p4"))
if STORE_PATH:
    STORE = pathlib.Path(STORE_PATH)
else:
    STORE = WORK / "store"
    pins["store"] = checkout(STORE_REPO, STORE_BRANCH, STORE, pins.get("store"))
st = load_state()
st["pins"] = pins
save_json(STATE, st)

PLUGIN = P4 / "lbiza" / "plugins" / "vlds"
PICK = PLUGIN / "scripts" / "pick_model.py"
SUITE = PHI / "scripts" / "session" / "nested_suite.py"
run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(PHI)], "setup")
sys.path.insert(0, str(PHI))

# the training device, checked once per runtime: a short pretrain and fine-tune on it, the CPU if that fails
import torch
import phi_rank as pr
want = "cpu" if DEVICE == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
if want == "cuda":
    try:
        os.environ["PHI_DEVICE"] = "cuda"
        m = pr.Ranker(1 << 13, 16)
        texts = [f"the ring buffer drain rule {i}" for i in range(64)]
        assert pr.pretrain(m, list(zip(texts, texts[1:])), 0.05, log=lambda *_: None) > 0
        pr.finetune(m, [{"task": "the ring buffer", "rows": [{"text": t, "label": int(i < 3)}
                                                               for i, t in enumerate(texts)]}], epochs=1)
        assert next(m.parameters()).device.type == "cpu"
    except Exception as e:                 # a broken CUDA path must not stop the run: the CPU still measures
        print(f"GPU self-check failed ({type(e).__name__}: {e}); training on the CPU")
        want = "cpu"
os.environ["PHI_DEVICE"] = want
gpu = torch.cuda.get_device_name(0) if want == "cuda" else ""
print(f"training on: {want} {gpu}".strip(), f"| CPU cores: {os.cpu_count()}")
st = load_state()
devices = st.setdefault("devices", [])
if devices and devices[-1]["device"] != (gpu or want):
    print(f"note: earlier sessions of this run trained on {devices[-1]['device']}. Runs already on Drive keep "
          "their models; new runs will differ in how many steps fit the time budget.")
devices.append({"at": now(), "device": gpu or want, "cpus": os.cpu_count()})
save_json(STATE, st)

SESSION = next(p for p in sorted((STORE / "sessions").iterdir()) if (p / "vlds").is_dir()) / "vlds"
print("pins:", json.dumps(pins))
print("store:", STORE, "| a session store for the plugin:", SESSION.relative_to(STORE))
'''))

cells.append(md("""## 3. Exact proofs
No model here: each claim is checked for every case up to a bound, and the cell stops with an error if one fails.
- The automaton of the nesting rule has two states: after a `.claude` (0) and after a `p4` (1). Counting the allowed moves between them gives M.
- M<sup>n</sup> = [[F(n+1), F(n)], [F(n), F(n−1)]], and det M<sup>n</sup> = (−1)<sup>n</sup> (Cassini).
- Store addresses start and end with a p4 and never put a p4 directly in a p4. There are F(n−2) of them of length n ≥ 2.
- With Fibonacci weights, every address up to length 20 is a distinct integer.
- `phi_rank.tree.parent` is defined for every address, so the tree has a single root.
- `phi_mask` never keeps two adjacent rows.
- The stream test (`stream_logic.py`) shows that the forbidden `11` delimits a stream of addresses with no length field."""))
cells.append(code('''import itertools
import torch
import phi_rank.tree as tree
from phi_prune.masks import phi_mask

if not done("proofs"):
    proofs = {}
    # the automaton: state 0 = after a .claude, 1 = after a p4; a p4 may never follow a p4
    allowed = {(a, b): not (a == 1 and b == 1) for a in (0, 1) for b in (0, 1)}
    M = [[int(allowed[(a, b)]) for b in (0, 1)] for a in (0, 1)]
    assert M == [[1, 1], [1, 0]], M
    proofs["automaton_is_M"] = M

    F = [0, 1]
    while len(F) < 64:
        F.append(F[-1] + F[-2])
    P = [[1, 0], [0, 1]]
    for n in range(1, 40):
        P = [[P[0][0] + P[0][1], P[0][0]], [P[1][0] + P[1][1], P[1][0]]]        # P · M
        assert P == [[F[n + 1], F[n]], [F[n], F[n - 1]]], n
        assert P[0][0] * P[1][1] - P[0][1] * P[1][0] == (-1) ** n, n
    proofs["M_power_and_cassini_to_n"] = 39

    def legal(s):
        return s[0] == "1" and s[-1] == "1" and "11" not in s
    counts = {}
    values = set()
    for n in range(1, 21):
        addrs = [a for a in ("".join(t) for t in itertools.product("01", repeat=n)) if legal(a)]
        counts[n] = len(addrs)
        if n >= 2:
            assert len(addrs) == F[n - 2], (n, len(addrs))
        for a in addrs:
            w = [1, 2]
            while len(w) < len(a):
                w.append(w[-1] + w[-2])
            v = sum(int(c) * x for c, x in zip(reversed(a), w))
            assert v not in values, a
            values.add(v)
            assert tree.parent(a) is None or legal(tree.parent(a)), a
    assert tree.addresses(5) == ["1", "101", "1001", "10001", "10101"]
    proofs["addresses_per_length"] = counts
    proofs["distinct_zeckendorf_values"] = len(values)

    g = torch.Generator().manual_seed(0)
    for trial in range(200):
        w = torch.randn(64 + trial, 8, generator=g)
        kept = phi_mask(w, axis=0, density=0.5).flatten(1).any(dim=1)
        assert not bool((kept[1:] & kept[:-1]).any()), trial
    proofs["phi_mask_no_adjacent_rows_trials"] = 200

    out = subprocess.run([sys.executable, str(PHI / "scripts" / "session" / "stream_logic.py"),
                          "--items", "2000", "--maxlen", "12"], capture_output=True, text=True, check=True).stdout
    proofs["stream_logic"] = json.loads(out)
    save_json(ROOT / "results" / "proofs.json", proofs)
    mark("proofs")
proofs = json.loads((ROOT / "results" / "proofs.json").read_text())
print("all exact claims hold; addresses per length:", proofs["addresses_per_length"])
print("stream codes:", {k: v["flips_detected_by_structure"] for k, v in proofs["stream_logic"]["codes"].items()})
'''))

cells.append(md("## 4. The store's labelled pairs\nThe vlds plugin exports every labelled pool of the store (`pick_model.py pairs`). The file is kept on Drive."))
cells.append(code('''PAIRS = ROOT / "pairs" / "pairs.jsonl"
if not (done("pairs") and PAIRS.exists()):
    run([sys.executable, str(PICK), "--store", str(SESSION), "pairs", "--jsonl", str(PAIRS)], "pairs")
    n = sum(1 for _ in open(PAIRS, encoding="utf-8"))
    mark("pairs", rows=n)
pools = len({json.loads(l)["pool_id"] for l in open(PAIRS, encoding="utf-8")})
print(f"{pools} labelled pools in {PAIRS}")
assert pools >= FOLDS, "too few labelled pools to fold"
'''))

cells.append(md("""## 5. Nested against one model, at every k
One `nested_suite.py` run per (k, seed), resumable:
- Its `--cache` folder on Drive keeps every pretrained model, fold set and arm as it finishes.
- A run cut short by a disconnect continues from there on the next **Run all**.
- A finished run's JSON lands in `results/suite-k<k>-s<seed>.json` and is never redone.

This is the long cell. Leave the tab open; Colab's free tier may still disconnect, and the resume covers that."""))
cells.append(code('''def suite_done(k, seed):
    p = ROOT / "results" / f"suite-k{k}-s{seed}.json"
    try:
        return len(json.loads(p.read_text())["runs"]) == 1
    except (OSError, ValueError, KeyError):
        return False


plan = [(k, s) for k in K_LIST for s in SEEDS]
for i, (k, seed) in enumerate(plan, 1):
    if suite_done(k, seed):
        print(f"[{i}/{len(plan)}] k={k} seed={seed}: done")
        continue
    print(f"[{i}/{len(plan)}] k={k} seed={seed}: running (resumes from Drive's cache)")
    t0 = time.time()
    run([sys.executable, str(SUITE), "--p4", str(STORE), "--pairs", str(PAIRS), "--plugin", str(PLUGIN),
         "--k", k, "--minutes", MINUTES, "--epochs", EPOCHS, "--folds", FOLDS, "--warm", WARM, "--seeds", seed,
         "--cache", ROOT / "cache", "--out", ROOT / "results" / f"suite-k{k}-s{seed}.json"], f"suite-k{k}-s{seed}")
    mark(f"suite-k{k}-s{seed}", seconds=round(time.time() - t0))
print("suite: every planned run is on Drive")
'''))

cells.append(md("""## 6. The effort tiers on the store
- `phi_rank.tree train` builds the k-node tree and writes `tree-oof.json`, the out-of-fold scores at medium and at high, with each effort's held-out r-precision. The tree's files are copied to Drive, and a new runtime restores them from there.
- The vlds plugin's `pick_model.py train` then fits the JSON ranker's `.pt` weight on each source in turn (the single ranker, the tree at medium, the tree at high) and records whether it keeps the weight.
- The `pool-effort:` line this cell writes goes only into the runtime's throwaway clone of the store. Nothing is pushed."""))
cells.append(code('''MODELS = STORE / "models"
TREE_FILES = ["tree.json", "tree-oof.json"]
saved = ROOT / "models"
if done("tree") and (saved / "tree-oof.json").exists():
    if not (MODELS / "tree-oof.json").exists():          # a fresh runtime: restore the tree from Drive
        shutil.copytree(saved / "tree", MODELS / "tree", dirs_exist_ok=True)
        for f in TREE_FILES:
            shutil.copy2(saved / f, MODELS / f)
else:
    run([sys.executable, "-m", "phi_rank.tree", "train", "--p4", STORE, "--pairs", PAIRS, "--k", TREE_K,
         "--minutes", TREE_MINUTES, "--epochs", EPOCHS, "--now", now()], "tree", cwd=PHI)
    shutil.copytree(MODELS / "tree", saved / "tree", dirs_exist_ok=True)
    for f in TREE_FILES:
        shutil.copy2(MODELS / f, saved / f)
    mark("tree")
tree_oof = json.loads((saved / "tree-oof.json").read_text())
print("tree held out:", tree_oof["heldout"])

if not done("pt-weights"):
    index = STORE / "vindex" / "phi-index.md"
    index.parent.mkdir(parents=True, exist_ok=True)
    original = index.read_text(encoding="utf-8") if index.exists() else None
    weights = {}
    try:
        for effort in ("", "medium", "high"):
            body = original or "# index\\n"
            lines = [l for l in body.split("\\n") if not l.startswith("pool-effort:")]
            body = "\\n".join(lines)
            if effort:
                body = body.replace("## recall\\n", f"## recall\\n\\npool-effort: {effort}\\n", 1) if "## recall" in body \\
                    else body + f"\\n## recall\\n\\npool-effort: {effort}\\n"
            index.write_text(body, encoding="utf-8")
            run([sys.executable, str(PICK), "--store", str(SESSION), "train", "--now", now()], "pt-weights")
            m = json.loads((MODELS / "pick-model.json").read_text())
            weights[effort or "ranker"] = {"pt": m.get("pt"), "pt_effort": m.get("pt_effort"), **m.get("heldout", {})}
    finally:                                               # the clone's index as it was
        if original is None:
            index.unlink(missing_ok=True)
        else:
            index.write_text(original, encoding="utf-8")
    save_json(ROOT / "results" / "pt-weights.json", weights)
    mark("pt-weights")
weights = json.loads((ROOT / "results" / "pt-weights.json").read_text())
for source, w in weights.items():
    if "with_pt" in w:
        print(f"{source:>7}: with the .pt {w['with_pt']}  without {w['without_pt']}  kept {bool(w.get('pt'))}")
    else:
        print(f"{source:>7}: no out-of-fold scores from this source, so nothing to weigh")
'''))

cells.append(md("""## 7. Summary and verdicts
Reads every result on Drive, including those from earlier sessions, and writes `summary.json`, `summary.md` and `summary.png`. A verdict is stated only for what the runs measured. A claim with no finished run shows as `not measured`."""))
cells.append(code('''import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = {}
for p in sorted((ROOT / "results").glob("suite-k*-s*.json")):
    d = json.loads(p.read_text())
    for r in d["runs"]:
        runs.setdefault(d["k"], []).append(r)

rows = []
for k in sorted(runs):
    q = [r["quality"] for r in runs[k]]
    mean = lambda key: round(statistics.mean(x[key] for x in q), 4)
    rows.append({"k": k, "seeds": len(q), "big": mean("big"), "zeck_ens": mean("zeck_ens"),
                 "zeck_wins_every_seed": all(x["zeck_ens"] > x["big"] for x in q),
                 "qrouted": mean("qrouted"), "q24routed": mean("q24routed"),
                 "route_hit": mean("qrouted_route_hit"), "route_hit_24": mean("q24routed_route_hit"),
                 "chance": round(1 / k, 4),
                 "deployed_zeck_ens": round(statistics.mean(r["deployed"]["zeck_ens"] for r in runs[k]), 4),
                 "deployed_json": round(statistics.mean(r["deployed"]["json_without"] for r in runs[k]), 4)})


def verdict(ok, measured):
    return "not measured" if not measured else ("holds" if ok else "does not hold")


wins = [r["k"] for r in rows if r["zeck_wins_every_seed"]]
full = [r for r in rows if r["seeds"] >= len(SEEDS)]
claims = {
    "k = 4 is the only node count where the Zeckendorf ensemble beats one large model on every seed":
        verdict(wins == [4], 4 in {r["k"] for r in full} and len(full) >= 2),
    "the matrix route finds the best node above chance at every k":
        verdict(all(r["route_hit"] > r["chance"] for r in rows), bool(rows)),
    "φ routing ranks better than 2:4 routing at every k":
        verdict(all(r["qrouted"] > r["q24routed"] for r in rows), bool(rows)),
    "the tree at high ranks better than at medium (held out)":
        verdict(tree_oof["heldout"]["high"] > tree_oof["heldout"]["medium"], True),
    "the JSON ranker keeps the tree's .pt weight at high":
        verdict(bool(weights.get("high", {}).get("pt")), "high" in weights),
}
summary = {"run": RUN, "at": now(), "pins": load_state().get("pins"), "devices": load_state().get("devices"),
           "plan": {"k": K_LIST, "seeds": SEEDS,
           "minutes": MINUTES, "epochs": EPOCHS}, "by_k": rows, "tree_heldout": tree_oof["heldout"],
           "pt_weights": weights, "claims": claims, "k_where_zeck_beats_big_every_seed": wins}
save_json(ROOT / "summary.json", summary)

trained_on = sorted({d["device"] for d in summary["devices"] or []})
lines = [f"# p4 Q matrix on Colab: {RUN}", "", f"Pinned commits: `{json.dumps(summary['pins'])}`",
         f"Trained on: {', '.join(trained_on)}", "",
         "| k | seeds | one large | Zeckendorf ensemble | beats it every seed | φ-routed | 2:4-routed | route hit (φ / 2:4) | chance |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['k']} | {r['seeds']} | {r['big']:.3f} | {r['zeck_ens']:.3f} | {'yes' if r['zeck_wins_every_seed'] else 'no'} "
                 f"| {r['qrouted']:.3f} | {r['q24routed']:.3f} | {r['route_hit']:.2f} / {r['route_hit_24']:.2f} | {r['chance']:.2f} |")
lines += ["", f"Tree (k = {TREE_K}) held out: medium {tree_oof['heldout']['medium']:.3f}, high {tree_oof['heldout']['high']:.3f}.", "",
          "| .pt source | with | without | kept |", "|---|---|---|---|"]
for s, w in weights.items():
    lines.append(f"| {s} | {w.get('with_pt', 'no scores')} | {w.get('without_pt', 'no scores')} | {'yes' if w.get('pt') else 'no'} |")
lines += ["", "## Verdicts", ""] + [f"- {c}: **{v}**" for c, v in claims.items()]
(ROOT / "summary.md").write_text("\\n".join(lines) + "\\n", encoding="utf-8")

if rows:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ks = [r["k"] for r in rows]
    for k in ks:
        ax[0].scatter([k] * len(runs[k]), [x["quality"]["zeck_ens"] - x["quality"]["big"] for x in runs[k]], color="#a8661a")
    ax[0].axhline(0, color="#888", lw=1)
    ax[0].set(xlabel="nodes k", ylabel="Zeckendorf ensemble − one large", title="held-out r-precision gap, per seed")
    ax[1].plot(ks, [r["route_hit"] for r in rows], "o-", color="#a8661a", label="φ route")
    ax[1].plot(ks, [r["route_hit_24"] for r in rows], "s--", color="#2f679f", label="2:4 route")
    ax[1].plot(ks, [r["chance"] for r in rows], ":", color="#888", label="chance 1/k")
    ax[1].set(xlabel="nodes k", ylabel="routed node is the best node", title="the matrix as the route")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "summary.png", dpi=140)

from IPython.display import Markdown, Image, display
display(Markdown((ROOT / "summary.md").read_text()))
if rows:
    display(Image(str(ROOT / "summary.png")))
'''))

cells.append(md("""## 8. Optional: commit the results into ezexe/phi
With `PUSH_RESULTS = True` and a `GITHUB_TOKEN` that can write to ezexe/phi, this cell:
- copies the results into `.docs/p4/nested-store-results/colab-<RUN>/`
- commits them on `PUSH_BRANCH`
- pushes the branch

Opening a pull request and updating `nested-store.md` stay yours. This cell runs once per run; delete `push` from `state.json` to push again."""))
cells.append(code('''if PUSH_RESULTS and not done("push"):
    if not token():
        raise RuntimeError("PUSH_RESULTS needs a GITHUB_TOKEN secret that can write to ezexe/phi")
    dest = WORK / "phi-push"
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(["git", "clone", "-q", "--depth", "1", url("ezexe/phi"), str(dest)], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", "-B", PUSH_BRANCH], check=True)
    out = dest / ".docs" / "p4" / "nested-store-results" / f"colab-{RUN}"
    out.mkdir(parents=True, exist_ok=True)
    for p in list((ROOT / "results").glob("*.json")) + [ROOT / "summary.json", ROOT / "summary.md", ROOT / "summary.png"]:
        if p.exists():
            shutil.copy2(p, out / p.name)
    git = ["git", "-C", str(dest), "-c", "user.name=phi colab", "-c", "user.email=phi-colab@users.noreply.github.com"]
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", f"nested-store results from Colab run {RUN}"], check=True)
    r = subprocess.run(["git", "-C", str(dest), "push", "-q", "-u", "origin", PUSH_BRANCH], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("push failed: " + r.stderr.replace(token(), "***"))
    mark("push", branch=PUSH_BRANCH)
    print(f"pushed to {PUSH_BRANCH}: open a pull request on https://github.com/ezexe/phi/compare/{PUSH_BRANCH}")
else:
    print("push:", "done earlier" if done("push") else "off (DEVICE = "auto"                       # auto: the GPU when the runtime has one; "cpu" forces the CPU

PUSH_RESULTS = False)")
'''))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}
out = Path(__file__).resolve().parents[2] / ".docs" / "p4" / "q_matrix_colab.ipynb"
nbf.write(nb, out)
print("wrote", out)
