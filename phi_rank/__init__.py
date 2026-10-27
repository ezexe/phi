"""
A .pt ranker for a .claude/p4 store: which store entries bear on a session's task.

Two stages, one model:

1. pretrain — self-supervised over every text file git keeps in the p4
   repository (the store, the flights, the frags, memory, scratchpad notes,
   readings and logs): the store's .gitignore decides what is trainable, and
   anything binary that slips past it is skipped; a file is read up to
   MAX_FILE_BYTES. Neighbouring
   passages of one file are pulled together and every other passage in the
   batch pushed apart (InfoNCE), so the encoder learns the repository's own
   vocabulary: its classes, files, measurements and rulings.
2. finetune — on the labelled pools the vlds plugin exports (`pick_model.py
   pairs --jsonl`): each pool's task against the entries its steering picked,
   the pool's other candidates as the negatives.

The encoder is a bag of hashed word, word-pair and character-trigram features
(no vocabulary to download) through an EmbeddingBag and a small MLP, with one
head for tasks and one for entries; a score is the cosine of the two. It trains
on a CPU in minutes. phi_serve serves it.

It learns continually, in rounds, and what git keeps of it is phi_prune's:

- the anchor, models/ranker-anchor.pt (fp16, git LFS): a whole fine-tuned
  model, the start of a chain;
- each later round continues from the model the chain rebuilds, on every pool
  (the new ones and a replay of the old), and ships only its change as
  models/ranker.rNNN.phid (git LFS): the largest KEEP of the entries that
  moved, positions in Fibonacci gap codes and values on the Zeckendorf grid
  (phi_prune.delta), packed eight bits to a byte;
- models/ranker.json names the anchor and the chain;
- models/ranker.pt is this machine's build of anchor + chain, in fp32 so the
  next round starts exactly where a reader of the store would: never tracked,
  rebuilt by load_current() when a pull brings rounds it has not seen.

A round is shipped whole, as a new anchor, when the cut costs more than
REBASE_GAP of the training pools' r-precision, when the chain reaches
MAX_CHAIN rounds, or when the pretrained base has changed. The pretrained base
learns continually too: when the repository's text has grown it continues
pretraining from the base it has. The pick threshold is chosen on held-out
pools (each fold's model fine-tuned from the base without the pools it is
scored on) and chosen again when the labelled pools have grown by a quarter.

    python -m phi_rank pretrain --p4 <p4 root> --minutes 5 [--buckets 32768 --dim 64]
    python -m phi_rank train --p4 <p4 root> --pairs pairs.jsonl --now "2026-10-27 09:00"
    python -m phi_rank eval --p4 <p4 root> --pairs pairs.jsonl
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from phi_prune.delta import apply_delta, delta_stats, encode_delta, load_delta, save_delta

NB = 1 << 15                  # hashed feature buckets
DIM = 64                      # 2^15 x 64 at fp16: a 4 MB checkpoint that pretrains in a few minutes on a CPU; a model
                              # keeps the size it was built at (its meta), and --buckets / --dim build another
MAX_FEATURES = 768            # features kept per text
PASSAGE_CHARS = 600
MAX_FILE_BYTES = 1 << 20      # a trace dump's first megabyte is plenty
PER_FILE = 6                  # the pool's cap per store file, as the children's
SKIP_EXT = {".wav", ".pak", ".vst3", ".dat", ".png", ".jpg", ".jpeg", ".gif", ".bin", ".dll", ".exe", ".pdb",
            ".lib", ".obj", ".o", ".so", ".dylib", ".zip", ".gz", ".7z", ".pt", ".phid", ".wasm", ".ico", ".pdf"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", "models"}
KEEP = 0.10                   # a round ships the largest 10% of the entries that moved (the notebook's dense-then-cut)
N_DIGITS = 6                  # Zeckendorf digits per value: 20 levels a side
CONTINUE_EPOCHS = 10          # a round's epochs from the shipped model; an anchor's are the full --epochs
REBASE_GAP = 0.02             # training r-precision the cut may cost before the round ships whole
MAX_CHAIN = 8                 # rounds after an anchor before the next anchor
REFOLD_GROWTH = 1.25          # labelled pools grown by this much since the threshold was chosen: choose it again
# the labels themselves stay out of pretraining: a pool's task and its steering picks, and the pick sheets a session
# writes beside them, would hand the encoder the answers its held-out evaluation asks for
LABEL_FILE_RE = re.compile(r"(^|/)(recall-pool\.md|pool-picks-[^/]*|pool-review-[^/]*|retrain-[^/]*\.tsv)$")
MIN_PRETRAIN_STEPS = 100      # a base with fewer steps (a --minutes 0 or tiny budget) is pretrained again when asked
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)*")


# ─── features ──────────────────────────────────────────────────────────────────────────────────────────

def words(text: str) -> List[str]:
    out = []
    for t in TOKEN_RE.findall(text):
        low = t.lower()
        out.append(low)
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", t)
        if len(parts) > 1:
            out += [p.lower() for p in parts]
    return out


def featurize(text: str, nb: int = NB) -> List[int]:
    """Hashed features of a text: words, neighbouring word pairs, and each word's character trigrams."""
    ws = words(text)
    feats = [f"w:{w}" for w in ws] + [f"b:{a}_{b}" for a, b in zip(ws, ws[1:])]
    for w in set(ws):
        padded = f"<{w}>"
        feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
    ids = [zlib.crc32(f.encode("utf-8")) % nb for f in feats[:MAX_FEATURES * 2]]   # the model's own buckets
    return ids[:MAX_FEATURES] or [0]


def batch_features(texts: Sequence[str], nb: int = NB) -> Tuple[torch.Tensor, torch.Tensor]:
    flat, offsets = [], []
    for t in texts:
        offsets.append(len(flat))
        flat += featurize(t, nb)
    return torch.tensor(flat, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


class Ranker(nn.Module):
    def __init__(self, nb: int = NB, dim: int = DIM):
        super().__init__()
        self.bag = nn.EmbeddingBag(nb, dim, mode="mean", sparse=True)   # only the rows a batch touches update
        self.body = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.task_head = nn.Linear(dim, dim)
        self.entry_head = nn.Linear(dim, dim)

    def encode(self, texts: Sequence[str], side: str) -> torch.Tensor:
        flat, offsets = batch_features(texts, self.bag.num_embeddings)
        h = self.bag(flat, offsets)
        h = h + self.body(h)
        h = (self.task_head if side == "task" else self.entry_head)(h)
        return F.normalize(h, dim=-1)


def ranker_like(state: Dict) -> "Ranker":
    """An untrained ranker of the size a state dict was saved at."""
    nb, dim = state["bag.weight"].shape
    return Ranker(nb, dim)


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    logits = a @ b.T / temperature
    target = torch.arange(len(a))
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


# ─── the repository's text ─────────────────────────────────────────────────────────────────────────────

def candidate_files(root: str) -> List[str]:
    """What git keeps in the p4 repository — tracked files and new ones its .gitignore does not exclude — so the
    store's .gitignore is the one rule for what is trainable; a directory that is no git repository is walked."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", root, "ls-files", "-co", "--exclude-standard", "-z"], capture_output=True,
                           timeout=120)
        if r.returncode == 0:
            rels = [x for x in r.stdout.decode("utf-8", "replace").split("\0") if x]
            return [os.path.join(root, x) for x in rels
                    if not set(x.split("/")[:-1]) & SKIP_DIRS and not LABEL_FILE_RE.search(x)]
    except (OSError, subprocess.SubprocessError):
        pass
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        out += [os.path.join(d, f) for f in files
                if not LABEL_FILE_RE.search(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/"))]
    return out


def text_files(root: str, exclude: Sequence[str] = ()) -> List[str]:
    """candidate_files(), less anything binary that slipped through: a skipped extension, or a NUL in its first
    4 KB; and less every path under one of `exclude` (prefixes relative to root)."""
    out = []
    for p in candidate_files(root):
        if exclude and os.path.relpath(p, root).replace(os.sep, "/").startswith(tuple(exclude)):
            continue
        if os.path.splitext(p)[1].lower() in SKIP_EXT or not os.path.isfile(p):
            continue
        try:
            with open(p, "rb") as fh:
                head = fh.read(4096)
        except OSError:
            continue
        if b"\0" not in head:
            out.append(p)
    return sorted(out)


def passages(path: str) -> List[str]:
    """A file's text in PASSAGE_CHARS pieces cut at line ends, its first MAX_FILE_BYTES."""
    try:
        with open(path, "rb") as fh:
            text = fh.read(MAX_FILE_BYTES).decode("utf-8", "replace")
    except OSError:
        return []
    out, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) > PASSAGE_CHARS and cur.strip():
            out.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        out.append(cur)
    return out


def corpus_pairs(root: str, exclude: Sequence[str] = ()) -> Tuple[List[Tuple[str, str]], int]:
    """(neighbouring passage pairs across the repository's text, bytes read)."""
    pairs, total = [], 0
    for p in text_files(root, exclude):
        ps = passages(p)
        total += sum(len(x) for x in ps)
        rel = os.path.relpath(p, root)
        pairs += [(f"{rel}\n{a}", b) for a, b in zip(ps, ps[1:])]
        if len(ps) == 1:
            pairs.append((rel, ps[0]))
    return pairs, total


# ─── the labelled pools ────────────────────────────────────────────────────────────────────────────────

def read_pools(pairs_jsonl: str) -> List[Dict]:
    """[{id, task, rows: [{file, text, label}]}] from the plugin's pairs export, one per labelled pool."""
    pools: Dict[str, Dict] = {}
    with open(pairs_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = r.get("pool_id") or f"{r.get('pool')}|{r.get('pooled')}"
            p = pools.setdefault(key, {"id": key, "task": r["task"], "rows": []})
            p["rows"].append({"file": r["file"], "text": r["text"], "label": int(r["label"])})
    return [p for p in pools.values() if any(x["label"] for x in p["rows"])]


def select(files: Sequence[str], scores: Sequence[float], threshold: float, per_file: int = PER_FILE) -> List[int]:
    by_file: Dict[str, List[int]] = {}
    for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
        if scores[i] < threshold:
            break
        lst = by_file.setdefault(files[i], [])
        if len(lst) < per_file:
            lst.append(i)
    return [i for lst in by_file.values() for i in lst]


def pool_metrics(labels: Sequence[int], scores: Sequence[float], picked: Sequence[int]) -> Dict[str, float]:
    n = sum(labels)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    hit = sum(labels[i] for i in picked)
    prec = hit / len(picked) if picked else 0.0
    rec = hit / n if n else 0.0
    return {"r-precision": sum(labels[i] for i in order[:n]) / n if n else 0.0,
            "recall@20": sum(labels[i] for i in order[:20]) / n if n else 0.0,
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


@torch.no_grad()
def score_pool(model: Ranker, task: str, texts: Sequence[str]) -> List[float]:
    model.eval()
    q = model.encode([task], "task")
    out = []
    for i in range(0, len(texts), 512):
        out += (model.encode(texts[i:i + 512], "entry") @ q.T).squeeze(1).tolist()
    return out


def best_threshold(model: Ranker, pools: Sequence[Dict]) -> float:
    scored = [(p, score_pool(model, p["task"], [r["text"] for r in p["rows"]])) for p in pools]
    best, best_f1 = 0.5, -1.0
    for k in range(-20, 96, 2):
        t = k / 100
        f1 = sum(pool_metrics([r["label"] for r in p["rows"]], sc,
                              select([r["file"] for r in p["rows"]], sc, t))["f1"] for p, sc in scored)
        if f1 > best_f1:
            best, best_f1 = t, f1
    return best


# ─── training ──────────────────────────────────────────────────────────────────────────────────────────

class Optim:
    """SparseAdam for the embedding table (only touched rows move, so a step costs what its batch touches) and Adam
    for the dense layers."""

    def __init__(self, model: "Ranker", lr: float):
        sparse = list(model.bag.parameters())
        dense = [p for n, p in model.named_parameters() if not n.startswith("bag.")]
        self.opts = [torch.optim.SparseAdam(sparse, lr=lr), torch.optim.Adam(dense, lr=lr)]

    def zero_grad(self):
        for o in self.opts:
            o.zero_grad()

    def step(self):
        for o in self.opts:
            o.step()


def pretrain(model: Ranker, pairs: List[Tuple[str, str]], minutes: float, batch: int = 256, lr: float = 2e-3,
             seed: int = 0, log=print) -> int:
    """InfoNCE over neighbouring passages until the time budget runs out; returns the steps taken."""
    if len(pairs) < 2:
        return 0
    rng = random.Random(seed)
    opt = Optim(model, lr)
    end, steps = time.time() + minutes * 60, 0
    model.train()
    while time.time() < end:
        b = rng.sample(pairs, min(batch, len(pairs)))
        loss = info_nce(model.encode([x for x, _ in b], "entry"), model.encode([y for _, y in b], "entry"))
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
        if steps % 200 == 0:
            log(f"pretrain step {steps}: loss {loss.item():.3f}")
    return steps


def finetune(model: Ranker, pools: Sequence[Dict], epochs: int = 30, negatives: int = 31, lr: float = 1e-3,
             seed: int = 0) -> None:
    """Each picked entry against the task, its pool's unpicked candidates as the negatives (listwise softmax)."""
    rng = random.Random(seed)
    opt = Optim(model, lr)
    model.train()
    for _ in range(epochs):
        for p in rng.sample(list(pools), len(pools)):
            pos = [r["text"] for r in p["rows"] if r["label"]]
            neg = [r["text"] for r in p["rows"] if not r["label"]]
            if not pos or not neg:
                continue
            q = model.encode([p["task"]], "task")
            loss = 0.0
            for t in pos:
                cands = [t] + rng.sample(neg, min(negatives, len(neg)))
                logits = (model.encode(cands, "entry") @ q.T).squeeze(1) / 0.05
                loss = loss + F.cross_entropy(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long))
            opt.zero_grad()
            (loss / len(pos)).backward()
            opt.step()


# ─── checkpoints ───────────────────────────────────────────────────────────────────────────────────────

def models_dir(p4: str) -> str:
    return os.path.join(p4, "models")


def replace_retrying(tmp: str, path: str, tries: int = 50) -> None:
    """os.replace, retried: Windows refuses it while a reader (the server, a hook) holds the target open."""
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


def save_atomic(obj, path: str) -> None:
    """Written whole or not at all: a temp file beside it, then the retried replace."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(obj, tmp)
    replace_retrying(tmp, path)


def write_text_atomic(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    replace_retrying(tmp, path)


def load(path: str) -> Tuple[Ranker, Dict]:
    ck = torch.load(path, map_location="cpu", weights_only=True)   # tensors and plain data only: no pickled code
    model = Ranker(ck["meta"]["nb"], ck["meta"]["dim"])
    model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    model.eval()
    return model, ck["meta"]


def save(model: Ranker, meta: Dict, path: str, half: bool = True) -> None:
    """The weights at fp16, half the bytes in git LFS (load() trains and scores in fp32); half=False keeps fp32, for
    the local build a next round continues from exactly."""
    save_atomic({"state": {k: v.half() if half else v.float() for k, v in model.state_dict().items()}, "meta": meta},
                path)


# ─── the chain: an anchor and its rounds' φ-deltas ──────────────────────────────────────────────────────

ANCHOR = "ranker-anchor.pt"     # git LFS: the chain's whole start
CURRENT = "ranker.pt"           # this machine's fp32 build of anchor + chain; never tracked
SIDECAR = "ranker.json"         # what the hooks read without torch: round, threshold, anchor, chain, pools seen
OOF = "ranker-oof.json"         # out-of-fold scores of every labelled pool: the JSON ranker's .pt feature


def read_sidecar(p4: str) -> Dict:
    try:
        with open(os.path.join(models_dir(p4), SIDECAR), encoding="utf-8") as f:
            side = json.load(f)
        return side if isinstance(side, dict) else {}
    except (OSError, ValueError):
        return {}


def chain_key(side: Dict) -> List:
    """What a build of the chain is a build of: the anchor's round and the deltas in order."""
    return [side.get("anchor_round")] + list(side.get("chain", []))


def base_id(meta: Dict) -> str:
    """The pretrained base a model was fine-tuned from, by its steps and the text it read."""
    return f"{meta.get('steps', meta.get('base_steps', 0))}:{meta.get('corpus_bytes', 0)}"


def served_meta(meta: Dict, side: Dict) -> Dict:
    keys = ("round", "threshold", "threshold_from", "per_file", "trained", "anchor_round", "chain")
    return {**meta, **{k: side[k] for k in keys if k in side}, "chain_key": chain_key(side)}


def rebuild(p4: str, side: Dict) -> Tuple[Ranker, Dict]:
    """The anchor plus every delta of the chain, in order, in fp32: what any reader of the store rebuilds."""
    mdir = models_dir(p4)
    model, meta = load(os.path.join(mdir, ANCHOR))
    if meta.get("round") != side.get("anchor_round"):
        raise ValueError(f"{ANCHOR} is round {meta.get('round')}, {SIDECAR} names round {side.get('anchor_round')}")
    state = model.state_dict()
    for name in side.get("chain", []):
        apply_delta(state, *load_delta(os.path.join(mdir, name)))
    model.load_state_dict(state)
    model.eval()
    return model, served_meta(meta, side)


def load_current(p4: str, write: bool = True) -> Tuple[Ranker, Dict]:
    """The model the store's chain names: models/ranker.pt when it is a build of that chain, else rebuilt from the
    anchor and the deltas (and, with write, saved as the new ranker.pt). A store from before the chain has no anchor:
    its ranker.pt is the model."""
    mdir = models_dir(p4)
    side = read_sidecar(p4)
    cur = os.path.join(mdir, CURRENT)
    if "anchor_round" not in side or not os.path.exists(os.path.join(mdir, ANCHOR)):
        return load(cur)
    if os.path.exists(cur):
        try:
            model, meta = load(cur)
            if meta.get("chain_key") == chain_key(side):
                return model, served_meta(meta, side)
        except Exception:  # noqa: BLE001 — a stale or torn build is rebuilt, never served
            pass
    model, meta = rebuild(p4, side)
    if write:
        save(model, meta, cur, half=False)
    return model, meta


def adopt_legacy(p4: str, side: Dict) -> bool:
    """A store from before the chain: its whole fine-tuned ranker.pt becomes the anchor. True when adopted."""
    mdir = models_dir(p4)
    cur, anc = os.path.join(mdir, CURRENT), os.path.join(mdir, ANCHOR)
    if os.path.exists(anc) or not os.path.exists(cur) or "anchor_round" in side:
        return False
    try:
        model, meta = load(cur)
    except Exception:  # noqa: BLE001 — a broken old checkpoint is no anchor; the round starts a fresh one
        return False
    if meta.get("stage") != "finetuned":
        return False
    save(model, meta, anc)
    side.update({"anchor_round": meta.get("round", 0), "chain": [], "anchor_base": base_id(meta)})
    return True


def mean_rprecision(model: Ranker, pools: Sequence[Dict]) -> float:
    got = [pool_metrics([r["label"] for r in p["rows"]], sc, [])["r-precision"]
           for p in pools for sc in [score_pool(model, p["task"], [r["text"] for r in p["rows"]])]]
    return sum(got) / len(got) if got else 0.0


# ─── commands ──────────────────────────────────────────────────────────────────────────────────────────

def cmd_pretrain(args) -> int:
    torch.manual_seed(0)
    pairs, total = corpus_pairs(args.p4)
    print(f"pretrain — {len(pairs):,} passage pairs from {total / 1e6:.1f} MB of text in {args.p4}")
    model = Ranker(args.buckets, args.dim)
    steps = pretrain(model, pairs, args.minutes)
    path = os.path.join(models_dir(args.p4), "ranker-base.pt")
    save(model, {"nb": args.buckets, "dim": args.dim, "stage": "pretrained", "corpus_bytes": total, "steps": steps,
                 "trained": args.now}, path)
    print(f"pretrain — {steps} steps → {path}")
    return 0


def is_checkpoint(path: str) -> bool:
    """A checkpoint on disk, not missing and not a git LFS pointer left by a clone without git-lfs."""
    try:
        with open(path, "rb") as f:
            return not f.read(64).startswith(b"version https://git-lfs")
    except OSError:
        return False


def base_model(p4: str, minutes: float, now: str, repretrain: float = 1.25, write: bool = True, nb: int = NB,
               dim: int = DIM, shared: str = "") -> Tuple[Ranker, Dict]:
    """The pretrained base: models/ranker-base.pt. Built when absent, or when it has fewer than MIN_PRETRAIN_STEPS
    (built with no or a tiny budget) and a budget is given now; when the repository's text has grown by `repretrain`
    times since, it continues pretraining from the base it has, so the base keeps learning what the store adds.
    write=False (an evaluation) builds a missing base in memory and saves nothing."""
    path = os.path.join(models_dir(p4), "ranker-base.pt")
    model, prior = None, 0
    if os.path.exists(path):
        model, meta = load(path)
        thin = minutes > 0 and meta.get("steps", 0) < MIN_PRETRAIN_STEPS
        if not write or (not thin and (minutes <= 0 or corpus_pairs(p4)[1] <= meta.get("corpus_bytes", 0) * repretrain)):
            return model, meta
        if thin:
            model = None
        else:
            prior = meta.get("steps", 0)
    torch.manual_seed(0)
    pairs, total = corpus_pairs(p4)
    from_shared = None
    if model is None and shared and is_checkpoint(shared):
        # a store with no base of its own starts from the shared p4 base: the contracts' vocabulary every store's
        # entries are written in, then continues on the store's own text
        model, smeta = load(shared)
        from_shared = {"steps": smeta.get("steps", 0), "corpus_bytes": smeta.get("corpus_bytes", 0),
                       "trained": smeta.get("trained", "")}
        prior = smeta.get("steps", 0)
    model = model or Ranker(nb, dim)       # a base keeps the size it was built at; nb/dim size a new one
    steps = pretrain(model, pairs, minutes) if minutes > 0 else 0
    nb, dim = model.bag.weight.shape
    meta = {"nb": nb, "dim": dim, "stage": "pretrained", "corpus_bytes": total, "steps": prior + steps, "trained": now}
    if from_shared:
        meta["from_shared"] = from_shared
    elif prior:
        meta["continued_from_steps"] = prior
    if write:
        save(model, meta, path)
    print(f"pretrain — {steps} steps over {total / 1e6:.1f} MB of text"
          + (f", from the shared p4 base ({prior} steps)" if from_shared else f", continuing {prior}" if prior else "")
          + (f" → {path}" if write else " (not saved)"))
    return model, meta


def text_key(text: str) -> str:
    """A candidate's key in the out-of-fold scores: the plugin hashes the same distilled text the same way."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def heldout_threshold(base_state: Dict, pools: Sequence[Dict], epochs: int, folds: int = 4
                      ) -> Tuple[float, str, Optional[float], Dict[str, Dict[str, float]]]:
    """The pick threshold from scores on pools each fold's model never trained on; those pools' mean r-precision;
    and the scores themselves, {pool id: {text key: score}} — out of fold, so the JSON ranker can learn how far to
    trust this model's score without learning from a score that saw its own label. With too few pools to fold, the
    training pools' own threshold, said so, no held-out figure and no scores."""
    if len(pools) < 4:
        model = ranker_like(base_state)
        model.load_state_dict(base_state)
        finetune(model, pools, epochs=epochs)
        return best_threshold(model, pools), "train", None, {}
    k = min(folds, len(pools))
    scored = []
    for i in range(k):
        held = pools[i::k]
        model = ranker_like(base_state)
        model.load_state_dict(base_state)
        torch.manual_seed(i)
        finetune(model, [p for j, p in enumerate(pools) if j % k != i], epochs=epochs)
        scored += [(p, score_pool(model, p["task"], [r["text"] for r in p["rows"]])) for p in held]
    best, best_f1 = 0.5, -1.0
    for step in range(-20, 96, 2):
        t = step / 100
        f1 = sum(pool_metrics([r["label"] for r in p["rows"]], sc, select([r["file"] for r in p["rows"]], sc, t))["f1"]
                 for p, sc in scored)
        if f1 > best_f1:
            best, best_f1 = t, f1
    rprec = sum(pool_metrics([r["label"] for r in p["rows"]], sc, [])["r-precision"] for p, sc in scored) / len(scored)
    oof = {p["id"]: {text_key(r["text"]): round(x, 5) for r, x in zip(p["rows"], sc)} for p, sc in scored}
    return best, f"held-out, {k} folds", rprec, oof


def cmd_train(args) -> int:
    """One round: the first (or a rebase) fine-tunes the base and ships a whole anchor; every other continues from
    the chain's model on every pool and ships its change as a φ-delta."""
    pools = read_pools(args.pairs)
    if not pools:
        print("train — no labelled pool in the pairs; nothing written")
        return 1
    mdir = models_dir(args.p4)
    base, bmeta = base_model(args.p4, args.minutes, args.now, nb=args.buckets, dim=args.dim, shared=args.shared_base)
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    side = read_sidecar(args.p4)
    adopted = adopt_legacy(args.p4, side)
    had = isinstance(side.get("round"), int) and (os.path.exists(os.path.join(mdir, ANCHOR))
                                                   or os.path.exists(os.path.join(mdir, CURRENT)))
    rnd = side["round"] + 1 if had else 0

    have_t = isinstance(side.get("threshold"), (int, float))
    t_pools = side.get("threshold_pools", len(side.get("pools", [])))
    if not have_t or "heldout_rprecision" not in side or len(pools) >= max(1, t_pools) * REFOLD_GROWTH:
        t, how, rprec, oof = heldout_threshold(base_state, pools, args.epochs)
        t_pools = len(pools)
        if oof:
            # tracked beside the models (text, small): the JSON ranker's training reads it on any machine, torch or not
            write_text_atomic(json.dumps({"round": rnd, "from": how, "pools": oof}, separators=(",", ":")) + "\n",
                              os.path.join(mdir, OOF))
    else:
        t, how, rprec = side["threshold"], side.get("threshold_from", ""), side["heldout_rprecision"]

    parent, why = None, ""
    if "anchor_round" not in side or not os.path.exists(os.path.join(mdir, ANCHOR)):
        why = "the first anchor"
    elif side.get("anchor_base") != base_id(bmeta):
        why = "the pretrained base changed"
    elif len(side.get("chain", [])) >= MAX_CHAIN:
        why = f"{MAX_CHAIN} rounds since the anchor"
    else:
        try:
            parent, _ = load_current(args.p4, write=False)
        except Exception as e:  # noqa: BLE001 — a chain that does not rebuild is replaced, not trusted
            why = f"the chain did not rebuild ({type(e).__name__}: {e})"

    shipped, note = None, ""
    if parent is not None:
        parent_state = {k: v.clone() for k, v in parent.state_dict().items()}
        torch.manual_seed(rnd)
        finetune(parent, pools, epochs=CONTINUE_EPOCHS, seed=rnd)
        header, payload = encode_delta(parent_state, parent.state_dict(), keep=KEEP, n_digits=N_DIGITS,
                                       changed_only=True)
        shipped = ranker_like(parent_state)
        shipped.load_state_dict(apply_delta({k: v.clone() for k, v in parent_state.items()}, header, payload))
        shipped.eval()
        dense_rp, ship_rp = mean_rprecision(parent, pools), mean_rprecision(shipped, pools)
        if dense_rp - ship_rp > REBASE_GAP:
            why = f"the cut cost {dense_rp - ship_rp:.3f} r-precision ({dense_rp:.3f} → {ship_rp:.3f})"
            model, shipped = parent, None
        else:
            name = f"ranker.r{rnd:03d}.phid"
            tmp = os.path.join(mdir, f"{name}.{os.getpid()}.tmp")
            save_delta(tmp, {**header, "round": rnd, "parent": chain_key(side)}, payload, packed=True)
            replace_retrying(tmp, os.path.join(mdir, name))
            side["chain"] = list(side.get("chain", [])) + [name]
            st = delta_stats(header, payload)
            note = (f"delta {name}: {st['kept']:,} of {st['entries']:,} entries, {st['bits_per_kept']:.1f} bits each, "
                    f"{os.path.getsize(os.path.join(mdir, name)) / 1e6:.2f} MB; training r-precision {dense_rp:.3f} "
                    f"dense, {ship_rp:.3f} shipped")
            model = shipped
    if shipped is None:
        if parent is None:
            model = ranker_like(base_state)
            model.load_state_dict(base_state)
            torch.manual_seed(0)
            finetune(model, pools, epochs=args.epochs)
        nb, dim = model.bag.weight.shape
        meta = {"nb": nb, "dim": dim, "stage": "finetuned", "round": rnd, "per_file": PER_FILE, "trained": args.now,
                "corpus_bytes": bmeta.get("corpus_bytes", 0), "base_steps": bmeta.get("steps", 0)}
        save(model, meta, os.path.join(mdir, ANCHOR))
        model, _ = load(os.path.join(mdir, ANCHOR))      # the anchor as a reader rebuilds it: fp16, back in fp32
        side.update({"anchor_round": rnd, "chain": [], "anchor_base": base_id(bmeta)})
        note = f"anchor {ANCHOR} (whole): {why}"

    side.update({"round": rnd, "threshold": t, "threshold_from": how, "threshold_pools": t_pools,
                 "heldout_rprecision": None if rprec is None else round(rprec, 4),
                 "trained": args.now, "pools": [p["id"] for p in pools], "per_file": PER_FILE})
    # `seen` is kept and widened by the plugin too (pools it met with nothing to learn count as met)
    side["seen"] = sorted(set(side.get("seen", [])) | set(side["pools"]))
    nb, dim = model.bag.weight.shape
    meta = served_meta({"nb": nb, "dim": dim, "stage": "finetuned"}, side)
    # the sidecar first: a reader that sees it before the build rebuilds the same model from the chain
    write_text_atomic(json.dumps(side, indent=1) + "\n", os.path.join(mdir, SIDECAR))
    save(model, meta, os.path.join(mdir, CURRENT), half=False)
    print(f"train — round {rnd}: {len(pools)} pools, threshold {t:.2f} ({how})"
          + (f", held-out r-precision {rprec:.3f}" if rprec is not None else "")
          + ("; adopted the old ranker.pt as the anchor" if adopted else "") + f"; {note}")
    return 0


def cmd_eval(args) -> int:
    pools = read_pools(args.pairs)
    if len(pools) < 3:
        print(f"eval — {len(pools)} labelled pools; at least 3 are needed")
        return 1
    base, _meta = base_model(args.p4, args.minutes, args.now or "", write=False, nb=args.buckets, dim=args.dim,
                             shared=args.shared_base)
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    folds = [pools[i::args.folds] for i in range(args.folds)]
    got = []
    for k, held in enumerate(folds):
        if not held:
            continue
        model = ranker_like(base_state)
        model.load_state_dict(base_state)
        torch.manual_seed(k)
        finetune(model, [p for j, f in enumerate(folds) if j != k for p in f], epochs=args.epochs)
        t = best_threshold(model, [p for j, f in enumerate(folds) if j != k for p in f])
        for p in held:
            sc = score_pool(model, p["task"], [r["text"] for r in p["rows"]])
            got.append(pool_metrics([r["label"] for r in p["rows"]], sc,
                                    select([r["file"] for r in p["rows"]], sc, t)))
    mean = lambda k: sum(g[k] for g in got) / len(got)  # noqa: E731
    print(f"eval — {args.folds}-fold over {len(pools)} pools: r-precision {mean('r-precision'):.3f}, recall@20 "
          f"{mean('recall@20'):.3f}, picks precision {mean('precision'):.3f} recall {mean('recall'):.3f}")
    return 0


DEFAULT_SHARED_EXCLUDE = ("Ibiza/plugins/legacy/",)   # superseded contracts: their retired terms never teach


def cmd_shared(args) -> int:
    """The shared p4 base: pretrained on the p4 marketplace's own prose and code (its contracts, the operator's and
    roboto's briefs, the flight checklists), the vocabulary every store's entries are written in. Public text only:
    no store's private text ever goes in, so it can ship with the plugin and seed every store."""
    torch.manual_seed(0)
    pairs, total, sources = [], 0, []
    for root in args.corpus:
        p, t = corpus_pairs(root, args.exclude)
        pairs += p
        total += t
        sources.append({"root": os.path.basename(os.path.abspath(root)), "pairs": len(p), "bytes": t})
    print(f"shared — {len(pairs):,} passage pairs from {total / 1e6:.2f} MB of text "
          f"({', '.join(x['root'] for x in sources)}; excluding {', '.join(args.exclude) or 'nothing'})")
    model = Ranker(args.buckets, args.dim)
    steps = pretrain(model, pairs, args.minutes)
    save(model, {"nb": args.buckets, "dim": args.dim, "stage": "shared", "corpus_bytes": total, "steps": steps,
                 "sources": sources, "exclude": list(args.exclude), "trained": args.now}, args.out)
    print(f"shared — {steps} steps → {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phi_rank")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pretrain")
    p.add_argument("--p4", required=True)
    p.add_argument("--minutes", type=float, default=5)
    p.add_argument("--now", default="")
    p.add_argument("--buckets", type=int, default=NB, help="hashed feature buckets of a new base")
    p.add_argument("--dim", type=int, default=DIM, help="embedding width of a new base")
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--p4", required=True)
        p.add_argument("--pairs", required=True, help="pick_model.py pairs --jsonl output")
        p.add_argument("--minutes", type=float, default=5, help="pretraining budget when the base is (re)built")
        p.add_argument("--buckets", type=int, default=NB, help="hashed feature buckets of a new base")
        p.add_argument("--dim", type=int, default=DIM, help="embedding width of a new base")
        p.add_argument("--shared-base", default=os.environ.get("PHI_SHARED_BASE", ""),
                       help="the shared p4 base a store with no base starts from (the plugin passes its own)")
        p.add_argument("--epochs", type=int, default=30)
        p.add_argument("--now", default="")
        if name == "eval":
            p.add_argument("--folds", type=int, default=5)
    p = sub.add_parser("shared", help="pretrain the shared p4 base on the p4 marketplace's own text")
    p.add_argument("--corpus", action="append", required=True, help="a repository to read (repeatable)")
    p.add_argument("--exclude", action="append", default=None, help="a path prefix to leave out (repeatable)")
    p.add_argument("--out", required=True)
    p.add_argument("--minutes", type=float, default=5)
    p.add_argument("--now", default="")
    p.add_argument("--buckets", type=int, default=NB)
    p.add_argument("--dim", type=int, default=DIM)
    args = ap.parse_args(argv)
    if args.cmd == "shared" and args.exclude is None:
        args.exclude = list(DEFAULT_SHARED_EXCLUDE)
    return {"pretrain": cmd_pretrain, "train": cmd_train, "eval": cmd_eval, "shared": cmd_shared}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
