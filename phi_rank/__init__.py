"""
An asymmetric dual encoder for a .claude/p4 store: which documents bear on a session's query.

Two stages, one model:

1. pretrain: contrastive pretraining with in-batch negatives over every text file git keeps in the store's
   repository. The store's .gitignore decides what is trainable, anything binary that slips past it is skipped, and
   a file is read up to MAX_FILE_BYTES. Neighbouring passages of one file form a positive pair and every other
   passage in the batch is a negative, by InfoNCE, so the encoder learns the repository's own vocabulary: its
   classes, files, measurements and decisions.
2. fine-tune: on labelled query groups, which read_label_pools builds from the store's labels.tsv and tasks.tsv
   joined over index.json. A group is a query with its label-1 documents as the positives, and its label-0
   documents and NEGATIVES BM25 negatives as the negatives, each candidate one the session could have been shown,
   dated before the query. Each positive is scored against negatives sampled from its own group under a softmax
   cross-entropy, which is Localized Contrastive Estimation.

The encoder hashes word, word-bigram and character-trigram features, so it has no vocabulary to download, and runs
them through an EmbeddingBag and a small MLP, with one head for queries and one for documents; a score is the cosine
of the two. It trains on a CPU in minutes, and on a CUDA GPU when one is present unless PHI_DEVICE=cpu: pretrain and
finetune move the model to device() for the run and back to the CPU after, so a model at rest, its checkpoints and
its deltas are always on the CPU. Each run keeps its texts' features in memory, up to FEATURE_CACHE_MB, so the GPU
does not wait on hashing.

It learns in rounds, each warm-started from the model before it, and what git keeps of it is phi_prune's:

- the full checkpoint, models/ranker-anchor.pt, at fp16 in git LFS: a whole fine-tuned model, which the later rounds
  build on;
- each later round continues from the model the checkpoints rebuild, on every query group, the new ones and a
  replay of the old, and ships only its change as an incremental checkpoint, models/ranker.rNNN.phid in git LFS:
  the largest KEEP of the entries that moved, their positions in Fibonacci gap codes and their values quantized
  symmetrically and written in Zeckendorf digits by phi_prune.delta, packed eight bits to a byte;
- the sidecar names the full checkpoint and the incremental ones: models/ranker.json, or models/ranker-pt.json in a
  store p4-phi writes, whose own logistic regression ranker is ranker.json;
- models/ranker.pt is this machine's build of those checkpoints, in fp32 so the next round starts exactly where a
  reader of the store would; it is never tracked, and load_current() rebuilds it when a pull brings rounds it has
  not seen.

A round is shipped whole, as a new full checkpoint, when the cut costs more than REBASE_GAP of the training query
groups' R-precision, when MAX_CHAIN incremental checkpoints follow the full one, or when the pretrained base has
changed. The pretrained base learns by continual pre-training: when the repository's text has grown, it continues
pretraining from the base it has. The decision threshold is chosen on held-out query groups, each fold's model
fine-tuned from the base without the groups it is scored on, and chosen again when the labelled query groups have
grown by a quarter.

    python -m phi_rank train --p4 <store root> --now "2026-10-27 09:00"
    python -m phi_rank eval --p4 <store root>
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from phi_store import layout  # noqa: E402  the store's layout: phi/ and temportal/
import torch.nn as nn
import torch.nn.functional as F

from phi_prune.delta import apply_delta, delta_stats, encode_delta, load_delta, save_delta
from phi_store import replace_retrying, write_text_atomic  # noqa: F401  phi_rank's names for them too

NB = 1 << 15                  # hashed feature buckets
DIM = 64                      # 2^15 x 64 at fp16: a 4 MB checkpoint that pretrains in a few minutes on a CPU; a model
                              # keeps the size its meta records, and --buckets and --dim build another
MAX_FEATURES = 768            # features kept per text
FEATURE_CACHE_MB = int(os.environ.get("PHI_FEATURE_CACHE_MB", "512"))   # per training run; full, it stops adding
PASSAGE_CHARS = 600
MAX_FILE_BYTES = 1 << 20      # a trace dump's first megabyte is plenty
PER_FILE = 6                  # the most documents one file gives a query's selection
SKIP_EXT = {".wav", ".pak", ".vst3", ".dat", ".png", ".jpg", ".jpeg", ".gif", ".bin", ".dll", ".exe", ".pdb",
            ".lib", ".obj", ".o", ".so", ".dylib", ".zip", ".gz", ".7z", ".pt", ".phid", ".wasm", ".ico", ".pdf"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", "models"}
KEEP = 0.10                   # a round ships the largest 10% of the entries that moved: top-k sparsification
N_DIGITS = 6                  # Zeckendorf digits per value: 20 levels a side
CONTINUE_EPOCHS = 10          # a round's epochs from the shipped model; a full checkpoint's are the full --epochs
REBASE_GAP = 0.02             # training R-precision the cut may cost before the round ships whole
MAX_CHAIN = 8                 # incremental checkpoints after a full checkpoint before the next full one
REFOLD_GROWTH = 1.25          # labelled query groups grown by this much since the threshold was chosen: choose it again
MIN_PRETRAIN_STEPS = 100      # a base with fewer steps, from --minutes 0 or a tiny budget, is pretrained again when asked
NEGATIVES = 10                # BM25 negatives added to a query group
POOL_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)*")
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
    """Hashed features of a text: words, word bigrams, and each word's character trigrams."""
    ws = words(text)
    feats = [f"w:{w}" for w in ws] + [f"b:{a}_{b}" for a, b in zip(ws, ws[1:])]
    for w in set(ws):
        padded = f"<{w}>"
        feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
    ids = [zlib.crc32(f.encode("utf-8")) % nb for f in feats[:MAX_FEATURES * 2]]   # the model's own buckets
    return ids[:MAX_FEATURES] or [0]


class FeatureCache:
    """One training run's features by text, as compact arrays, up to FEATURE_CACHE_MB: pretraining samples the same
    passages over and over, and hashing them again each time is what a GPU step would otherwise wait on."""

    def __init__(self, nb: int, mb: int = None):
        self.nb, self.room = nb, (FEATURE_CACHE_MB if mb is None else mb) << 20
        self.dtype = np.uint16 if nb <= 1 << 16 else np.int32
        self.got: Dict[str, np.ndarray] = {}

    def __call__(self, text: str) -> np.ndarray:
        a = self.got.get(text)
        if a is None:
            a = np.asarray(featurize(text, self.nb), dtype=self.dtype)
            if self.room >= a.nbytes:
                self.got[text] = a
                self.room -= a.nbytes
        return a


def batch_features(texts: Sequence[str], nb: int = NB, cache: Optional[FeatureCache] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    if cache is None or cache.nb != nb:
        flat, offsets = [], []
        for t in texts:
            offsets.append(len(flat))
            flat += featurize(t, nb)
        return torch.tensor(flat, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)
    parts = [cache(t) for t in texts]
    offsets = np.zeros(len(parts), dtype=np.int64)
    np.cumsum([len(a) for a in parts[:-1]], out=offsets[1:])
    return torch.from_numpy(np.concatenate(parts).astype(np.int64)), torch.from_numpy(offsets)


def device() -> torch.device:
    """Where training runs: PHI_DEVICE when set, as cpu, cuda or cuda:1, else a CUDA GPU when there is one."""
    want = os.environ.get("PHI_DEVICE", "").strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


class Ranker(nn.Module):
    def __init__(self, nb: int = NB, dim: int = DIM):
        super().__init__()
        self.bag = nn.EmbeddingBag(nb, dim, mode="mean", sparse=True)   # only the rows a batch touches update
        self.body = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.task_head = nn.Linear(dim, dim)
        self.entry_head = nn.Linear(dim, dim)

    def encode(self, texts: Sequence[str], side: str, cache: Optional[FeatureCache] = None) -> torch.Tensor:
        flat, offsets = batch_features(texts, self.bag.num_embeddings, cache)
        dev = self.bag.weight.device
        h = self.bag(flat.to(dev, non_blocking=True), offsets.to(dev, non_blocking=True))
        h = h + self.body(h)
        h = (self.task_head if side == "task" else self.entry_head)(h)
        return F.normalize(h, dim=-1)


def ranker_like(state: Dict) -> "Ranker":
    """An untrained ranker of the size a state dict was saved at."""
    nb, dim = state["bag.weight"].shape
    return Ranker(nb, dim)


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    logits = a @ b.T / temperature
    target = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


# ─── the repository's text ─────────────────────────────────────────────────────────────────────────────

def candidate_files(root: str) -> List[str]:
    """What git keeps in the store's repository, its tracked files and the untracked ones its .gitignore does not
    exclude, so the store's .gitignore is the one rule for what is trainable. A directory git lists nothing in, as
    one that is no git repository or one its repository ignores, is listed by git as its own work tree, so its own
    .gitignore files still decide; only where git cannot run is it walked whole."""
    import subprocess
    import tempfile

    def listed(cmd):
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        rels = [x for x in r.stdout.decode("utf-8", "replace").split("\0") if x] if r.returncode == 0 else []
        return [os.path.join(root, *x.split("/")) for x in rels if not set(x.split("/")[:-1]) & SKIP_DIRS]

    try:
        files = listed(["git", "-C", root, "ls-files", "-co", "--exclude-standard", "-z"])
        if files:
            return files
        # an empty throwaway repository whose work tree is the store: every file is untracked, and the store's own
        # .gitignore files exclude, as they would in a repository of the store's own
        with tempfile.TemporaryDirectory() as g:
            if subprocess.run(["git", "init", "-q", "--bare", g], capture_output=True, timeout=60).returncode == 0:
                return listed(["git", f"--git-dir={g}", f"--work-tree={root}", "ls-files", "-o", "--exclude-standard",
                               "-z"])
    except (OSError, subprocess.SubprocessError):
        pass
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        out += [os.path.join(d, f) for f in files]
    return out


def text_files(root: str) -> List[str]:
    """candidate_files(), less anything binary that slipped through: a skipped extension, or a NUL in its first
    4 KB."""
    out = []
    for p in candidate_files(root):
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


def corpus_pairs(root: str) -> Tuple[List[Tuple[str, str]], int]:
    """The positive pairs of neighbouring passages across the repository's text, and the bytes read."""
    pairs, total = [], 0
    for p in text_files(root):
        ps = passages(p)
        total += sum(len(x) for x in ps)
        rel = os.path.relpath(p, root)
        pairs += [(f"{rel}\n{a}", b) for a, b in zip(ps, ps[1:])]
        if len(ps) == 1:
            pairs.append((rel, ps[0]))
    return pairs, total


# ─── the labelled query groups ─────────────────────────────────────────────────────────────────────────

def _tsv(path: str) -> List[Dict]:
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
    except OSError:
        return []
    head = lines[0].split("\t") if lines and lines[0] else []
    return [dict(zip(head, l.split("\t"))) for l in lines[1:] if l.strip()]


def index_docs(p4: str) -> List[Dict]:
    """The p4-phi plugin's index.json documents as dicts: on disk they are rows under `doc_fields`."""
    try:
        with open(layout.sub(p4, "models", "index.json"), encoding="utf-8") as f:
            idx = json.load(f)
    except (OSError, ValueError):
        return []
    fields = idx.get("doc_fields")
    docs = idx.get("docs") or []
    return [dict(zip(fields, d)) if fields and isinstance(d, list) else d for d in docs]


def doc_text(d: Dict) -> str:
    """What a candidate reads as: the document's shown text, a repository file or commit prefixed by its path."""
    show, path = d.get("show") or "", d.get("path") or ""
    if path and d.get("kind") in ("file", "commit") and not show.startswith(path):
        return f"{path}: {show}"
    return show or path


def _bm25_order(task: str, docs: Sequence[Dict], k1: float = 1.2, b: float = 0.75) -> List[int]:
    """Indices of docs by BM25 of the query's words against each doc's indexed tokens, `toks`, best first."""
    toks = [(d.get("toks") or "").lower().split() for d in docs]
    n = len(toks)
    df: Dict[str, int] = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    avgdl = sum(len(t) for t in toks) / max(1, n)
    terms = list(dict.fromkeys(w.lower() for w in POOL_WORD_RE.findall(task) if len(w) >= 3))
    scores = []
    for t in toks:
        tf: Dict[str, int] = {}
        for w in t:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in terms:
            f = tf.get(w)
            if f:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / (avgdl or 1.0)))
        scores.append(s)
    return sorted(range(n), key=lambda i: (-scores[i], i))


def read_label_pools(p4: str, negatives: int = NEGATIVES) -> List[Dict]:
    """[{id, task, time, session, rows: [{id, file, text, label}]}] from the p4-phi plugin's labels.tsv, tasks.tsv
    and index.json: one query group per query with a label-1 document the index holds. Its rows are the query's
    labelled documents, label 1 positive and label 0 negative, and `negatives` BM25 negatives drawn from what the
    session could have been shown: store entries dated before the query, undated ones and repository files. A query
    whose text is a system reminder or an agent's message, opening with `<`, is skipped. Sorted by query time."""
    docs = index_docs(p4)
    by_id = {d["id"]: d for d in docs}
    tasks = {r["id"]: r for r in _tsv(layout.sub(p4, "models", "tasks.tsv")) if r.get("id")}
    by_task: Dict[str, Dict[str, int]] = {}
    for r in _tsv(layout.sub(p4, "models", "labels.tsv")):
        if r.get("entry") in by_id and r.get("task") in tasks:
            got = by_task.setdefault(r["task"], {})
            got[r["entry"]] = max(got.get(r["entry"], 0), int(str(r.get("label")) == "1"))
    pools = []
    for tid, labs in by_task.items():
        if not any(labs.values()):
            continue
        task = tasks[tid]
        text = (task.get("text") or "").strip()
        if not text or text.startswith("<"):
            continue
        when = task.get("time") or "9999"
        rows = [{"id": e, "file": by_id[e].get("path") or e, "text": doc_text(by_id[e]), "label": v}
                for e, v in labs.items()]
        pool = [d for d in docs if d["id"] not in labs and d["id"] != tid
                and (d.get("kind") == "file" or not d.get("time") or d["time"] < when)]
        for i in _bm25_order(text, pool)[:max(0, negatives)]:
            d = pool[i]
            rows.append({"id": d["id"], "file": d.get("path") or d["id"], "text": doc_text(d), "label": 0})
        pools.append({"id": tid, "task": text[:2000], "time": task.get("time", ""), "session": task.get("session", ""),
                      "rows": rows})
    pools.sort(key=lambda p: (p["time"], p["id"]))
    return pools


def pools_for(args) -> List[Dict]:
    """The query groups a command trains or evaluates on: the store's own labels."""
    return read_label_pools(args.p4, getattr(args, "negatives", NEGATIVES))


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
    """SparseAdam for the embedding table, where only touched rows move so that a step costs what its batch
    touches, and Adam for the dense layers."""

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
             seed: int = 0, log=print, select=None, select_every: float = 60.0,
             info: Optional[Dict] = None) -> int:
    """InfoNCE over neighbouring passages until the time budget runs out; returns the steps taken.

    Early stopping: with `select`, a function from a model to a score where higher is better, the model is scored
    every `select_every` seconds and once at the end, and the best-scoring weights are the ones it keeps. `info`, when given, receives
    the curve [(steps, score)], the best score and the steps it was reached at."""
    if len(pairs) < 2:
        return 0
    rng = random.Random(seed)
    cache = FeatureCache(model.bag.num_embeddings)
    curve, best, best_state = [], None, None

    def check(steps):
        nonlocal best, best_state
        model.eval()
        with torch.no_grad():
            score = float(select(model))
        model.train()
        curve.append((steps, round(score, 5)))
        if best is None or score > best[0]:
            best = (score, steps)
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    model.to(device())
    try:
        opt = Optim(model, lr)
        end, steps = time.time() + minutes * 60, 0
        next_check = time.time() + select_every
        model.train()
        if select:
            check(0)
        while time.time() < end:
            b = rng.sample(pairs, min(batch, len(pairs)))
            loss = info_nce(model.encode([x for x, _ in b], "entry", cache),
                            model.encode([y for _, y in b], "entry", cache))
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
            if steps % 200 == 0:
                log(f"pretrain step {steps}: loss {loss.item():.3f}")
            if select and time.time() >= next_check:
                check(steps)
                next_check = time.time() + select_every
        if select:
            check(steps)
            model.load_state_dict(best_state)
    finally:
        model.to("cpu")
    if info is not None and select:
        info.update({"curve": curve, "best_score": round(best[0], 5), "best_steps": best[1], "steps": steps})
    return steps


def finetune(model: Ranker, pools: Sequence[Dict], epochs: int = 30, negatives: int = 31, lr: float = 1e-3,
             seed: int = 0) -> None:
    """Each positive document against its query and negatives sampled from its query group, under a softmax
    cross-entropy: Localized Contrastive Estimation."""
    rng = random.Random(seed)
    cache = FeatureCache(model.bag.num_embeddings)
    dev = device()
    model.to(dev)
    try:
        opt = Optim(model, lr)
        model.train()
        zero = torch.zeros(1, dtype=torch.long, device=dev)
        for _ in range(epochs):
            for p in rng.sample(list(pools), len(pools)):
                pos = [r["text"] for r in p["rows"] if r["label"]]
                neg = [r["text"] for r in p["rows"] if not r["label"]]
                if not pos or not neg:
                    continue
                q = model.encode([p["task"]], "task", cache)
                loss = 0.0
                for t in pos:
                    cands = [t] + rng.sample(neg, min(negatives, len(neg)))
                    logits = (model.encode(cands, "entry", cache) @ q.T).squeeze(1) / 0.05
                    loss = loss + F.cross_entropy(logits.unsqueeze(0), zero)
                opt.zero_grad()
                (loss / len(pos)).backward()
                opt.step()
    finally:
        model.to("cpu")


# ─── checkpoints ───────────────────────────────────────────────────────────────────────────────────────

def models_dir(p4: str) -> str:
    return layout.sub(p4, "models")


def save_atomic(obj, path: str) -> None:
    """Written whole or not at all: a temp file beside it, then the retried replace."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(obj, tmp)
    replace_retrying(tmp, path)


def load(path: str) -> Tuple[Ranker, Dict]:
    ck = torch.load(path, map_location="cpu", weights_only=True)   # tensors and plain data only: no pickled code
    model = Ranker(ck["meta"]["nb"], ck["meta"]["dim"])
    model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    model.eval()
    return model, ck["meta"]


def save(model: Ranker, meta: Dict, path: str, half: bool = True) -> None:
    """The weights at fp16, half the bytes in git LFS, while load() trains and scores in fp32; half=False keeps
    fp32, for the local build a next round continues from exactly."""
    save_atomic({"state": {k: v.half() if half else v.float() for k, v in model.state_dict().items()}, "meta": meta},
                path)


# ─── the full checkpoint and its rounds' incremental checkpoints ───────────────────────────────────────

ANCHOR = "ranker-anchor.pt"     # git LFS: the full checkpoint the rounds build on
CURRENT = "ranker.pt"           # this machine's fp32 build of the checkpoints; never tracked
SIDECAR = "ranker.json"         # the sidecar: round, threshold, full and incremental checkpoints, query groups
SIDECAR_RECALL = "ranker-pt.json"   # the sidecar's name in a store p4-phi writes, whose own ranker is ranker.json


def sidecar_name(p4: str) -> str:
    """ranker.json, or ranker-pt.json in a store p4-phi writes, one with store/ in its layout: the plugin's
    ranker.py keeps its logistic regression ranker as ranker.json, and a train here must never overwrite it."""
    return SIDECAR_RECALL if os.path.isdir(layout.sub(p4, "store")) else SIDECAR


def sidecar_path(p4: str) -> str:
    return os.path.join(models_dir(p4), sidecar_name(p4))


def read_sidecar(p4: str) -> Dict:
    try:
        with open(sidecar_path(p4), encoding="utf-8") as f:
            side = json.load(f)
        if not isinstance(side, dict) or str(side.get("kind", "")).startswith("p4 recall"):
            return {}                       # the plugin's own logistic regression ranker is not this sidecar
        return side
    except (OSError, ValueError):
        return {}


def chain_key(side: Dict) -> List:
    """What a build is a build of: the full checkpoint's round and the incremental checkpoints in order."""
    return [side.get("anchor_round")] + list(side.get("chain", []))


def base_id(meta: Dict) -> str:
    """The pretrained base a model was fine-tuned from, by its steps and the text it read."""
    return f"{meta.get('steps', meta.get('base_steps', 0))}:{meta.get('corpus_bytes', 0)}"


def served_meta(meta: Dict, side: Dict) -> Dict:
    keys = ("round", "threshold", "threshold_from", "per_file", "trained", "anchor_round", "chain")
    return {**meta, **{k: side[k] for k in keys if k in side}, "chain_key": chain_key(side)}


def rebuild(p4: str, side: Dict) -> Tuple[Ranker, Dict]:
    """The full checkpoint plus every incremental checkpoint, in order, in fp32: what any reader of the store
    rebuilds."""
    mdir = models_dir(p4)
    model, meta = load(os.path.join(mdir, ANCHOR))
    if meta.get("round") != side.get("anchor_round"):
        raise ValueError(f"{ANCHOR} is round {meta.get('round')}, {sidecar_name(p4)} names round "
                         f"{side.get('anchor_round')}")
    state = model.state_dict()
    for name in side.get("chain", []):
        apply_delta(state, *load_delta(os.path.join(mdir, name)))
    model.load_state_dict(state)
    model.eval()
    return model, served_meta(meta, side)


def load_current(p4: str, write: bool = True) -> Tuple[Ranker, Dict]:
    """The model the sidecar names: models/ranker.pt when it is a build of those checkpoints, else rebuilt from the
    full checkpoint and the incremental ones and, with write, saved as ranker.pt. A store from before the incremental
    checkpoints has no full checkpoint, and its ranker.pt is the model."""
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
    """A store from before the incremental checkpoints: its whole fine-tuned ranker.pt becomes the full
    checkpoint. True when adopted."""
    mdir = models_dir(p4)
    cur, anc = os.path.join(mdir, CURRENT), os.path.join(mdir, ANCHOR)
    if os.path.exists(anc) or not os.path.exists(cur) or "anchor_round" in side:
        return False
    try:
        model, meta = load(cur)
    except Exception:  # noqa: BLE001 — a broken old checkpoint is no full checkpoint; the round starts a fresh one
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

def base_model(p4: str, minutes: float, now: str, repretrain: float = 1.25, write: bool = True, nb: int = NB,
               dim: int = DIM) -> Tuple[Ranker, Dict]:
    """The pretrained base, models/ranker-base.pt. It is built when absent, or when it has fewer than
    MIN_PRETRAIN_STEPS, having been built with no budget or a tiny one, and this call gives a budget. When the
    repository's text has grown `repretrain` times since, it continues pretraining from the base it has, so the base
    keeps learning what the store adds. write=False, for an evaluation, builds a missing base in memory and saves
    nothing."""
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
    model = model or Ranker(nb, dim)       # a base keeps the size it was built at; nb/dim size a new one
    steps = pretrain(model, pairs, minutes) if minutes > 0 else 0
    nb, dim = model.bag.weight.shape
    meta = {"nb": nb, "dim": dim, "stage": "pretrained", "corpus_bytes": total, "steps": prior + steps, "trained": now}
    if prior:
        meta["continued_from_steps"] = prior
    if write:
        save(model, meta, path)
    print(f"pretrain — {steps} steps over {total / 1e6:.1f} MB of text"
          + (f", continuing {prior}" if prior else "")
          + (f" → {path}" if write else " (not saved)"))
    return model, meta


def heldout_threshold(base_state: Dict, pools: Sequence[Dict], epochs: int, folds: int = 4
                      ) -> Tuple[float, str, Optional[float]]:
    """The decision threshold from scores on query groups each fold's model never trained on, and those groups'
    mean R-precision. With too few groups to fold, the training groups' own threshold, said so, and no held-out
    figure."""
    if len(pools) < 4:
        model = ranker_like(base_state)
        model.load_state_dict(base_state)
        finetune(model, pools, epochs=epochs)
        return best_threshold(model, pools), "train", None
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
    return best, f"held-out, {k} folds", rprec


def cmd_train(args) -> int:
    """One round: the first, or one that ships whole, fine-tunes the base and ships a full checkpoint; every other
    continues from the rebuilt model on every query group and ships its change as an incremental checkpoint."""
    pools = pools_for(args)
    if not pools:
        print("train — no labelled pool in models/labels.tsv + tasks.tsv + index.json; nothing written")
        return 1
    mdir = models_dir(args.p4)
    base, bmeta = base_model(args.p4, args.minutes, args.now, nb=args.buckets, dim=args.dim)
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    side = read_sidecar(args.p4)
    adopted = adopt_legacy(args.p4, side)
    had = isinstance(side.get("round"), int) and (os.path.exists(os.path.join(mdir, ANCHOR))
                                                   or os.path.exists(os.path.join(mdir, CURRENT)))
    rnd = side["round"] + 1 if had else 0

    have_t = isinstance(side.get("threshold"), (int, float))
    t_pools = side.get("threshold_pools", len(side.get("pools", [])))
    if not have_t or "heldout_rprecision" not in side or len(pools) >= max(1, t_pools) * REFOLD_GROWTH:
        t, how, rprec = heldout_threshold(base_state, pools, args.epochs)
        t_pools = len(pools)
    else:
        t, how, rprec = side["threshold"], side.get("threshold_from", ""), side["heldout_rprecision"]

    parent, why = None, ""
    if "anchor_round" not in side or not os.path.exists(os.path.join(mdir, ANCHOR)):
        why = "the first full checkpoint"
    elif side.get("anchor_base") != base_id(bmeta):
        why = "the pretrained base changed"
    elif len(side.get("chain", [])) >= MAX_CHAIN:
        why = f"{MAX_CHAIN} rounds since the full checkpoint"
    else:
        try:
            parent, _ = load_current(args.p4, write=False)
        except Exception as e:  # noqa: BLE001 — checkpoints that do not rebuild are replaced, not trusted
            why = f"the checkpoints did not rebuild: {type(e).__name__}: {e}"

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
            why = f"the cut cost {dense_rp - ship_rp:.3f} R-precision, {dense_rp:.3f} → {ship_rp:.3f}"
            model, shipped = parent, None
        else:
            name = f"ranker.r{rnd:03d}.phid"
            tmp = os.path.join(mdir, f"{name}.{os.getpid()}.tmp")
            save_delta(tmp, {**header, "round": rnd, "parent": chain_key(side)}, payload, packed=True)
            replace_retrying(tmp, os.path.join(mdir, name))
            side["chain"] = list(side.get("chain", [])) + [name]
            st = delta_stats(header, payload)
            note = (f"incremental checkpoint {name}: {st['kept']:,} of {st['entries']:,} entries, {st['bits_per_kept']:.1f} bits each, "
                    f"{os.path.getsize(os.path.join(mdir, name)) / 1e6:.2f} MB; training R-precision {dense_rp:.3f} "
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
        model, _ = load(os.path.join(mdir, ANCHOR))      # the full checkpoint as a reader rebuilds it: fp16, back in fp32
        side.update({"anchor_round": rnd, "chain": [], "anchor_base": base_id(bmeta)})
        note = f"full checkpoint {ANCHOR}: {why}"

    side.update({"round": rnd, "threshold": t, "threshold_from": how, "threshold_pools": t_pools,
                 "heldout_rprecision": None if rprec is None else round(rprec, 4),
                 "trained": args.now, "pools": [p["id"] for p in pools], "per_file": PER_FILE})
    side.pop("seen", None)
    nb, dim = model.bag.weight.shape
    meta = served_meta({"nb": nb, "dim": dim, "stage": "finetuned"}, side)
    # the sidecar first: a reader that sees it before the build rebuilds the same model from the checkpoints
    write_text_atomic(json.dumps(side, indent=1) + "\n", sidecar_path(args.p4))
    save(model, meta, os.path.join(mdir, CURRENT), half=False)
    print(f"train — round {rnd}: {len(pools)} query groups, threshold {t:.2f}, {how}"
          + (f", held-out R-precision {rprec:.3f}" if rprec is not None else "")
          + ("; adopted the old ranker.pt as the full checkpoint" if adopted else "") + f"; {note}")
    return 0


def cmd_eval(args) -> int:
    pools = pools_for(args)
    if len(pools) < 3:
        print(f"eval — {len(pools)} labelled query groups; at least 3 are needed")
        return 1
    base, _meta = base_model(args.p4, args.minutes, args.now or "", write=False, nb=args.buckets, dim=args.dim)
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
    print(f"eval — {args.folds}-fold over {len(pools)} query groups: R-precision {mean('r-precision'):.3f}, recall@20 "
          f"{mean('recall@20'):.3f}, above the threshold precision {mean('precision'):.3f} recall {mean('recall'):.3f}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phi_rank")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--p4", required=True)
        p.add_argument("--negatives", type=int, default=NEGATIVES,
                       help="BM25 negatives added to each query group")
        p.add_argument("--minutes", type=float, default=5, help="pretraining budget when the base is built or rebuilt")
        p.add_argument("--buckets", type=int, default=NB, help="hashed feature buckets of a new base")
        p.add_argument("--dim", type=int, default=DIM, help="embedding width of a new base")
        p.add_argument("--epochs", type=int, default=30)
        p.add_argument("--now", default="")
        if name == "eval":
            p.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)
    return {"train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
