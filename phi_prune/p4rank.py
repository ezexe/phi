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
on a CPU in minutes. The checkpoint, models/ranker.pt, holds the weights, the
config, the pick threshold and what it was trained on; it is written whole or
not at all; git LFS keeps every round's checkpoint (each round fine-tunes the
pretrained base afresh, so a delta between two rounds would not be a usable
record). phi_prune.p4serve serves it. The pick threshold it serves is chosen on
held-out pools: each fold's model is fine-tuned without the pools it is scored
on, and the final model is fine-tuned on all of them.

    python -m phi_prune.p4rank pretrain --p4 <p4 root> --minutes 20
    python -m phi_prune.p4rank train --p4 <p4 root> --pairs pairs.jsonl --now "2026-10-27 09:00"
    python -m phi_prune.p4rank eval --p4 <p4 root> --pairs pairs.jsonl
"""

import argparse
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

NB = 1 << 17                  # hashed feature buckets
DIM = 128                     # 2^17 x 128 at fp16: a 34 MB checkpoint
MAX_FEATURES = 768            # features kept per text
PASSAGE_CHARS = 600
MAX_FILE_BYTES = 1 << 20      # a trace dump's first megabyte is plenty
PER_FILE = 6                  # the pool's cap per store file, as the children's
SKIP_EXT = {".wav", ".pak", ".vst3", ".dat", ".png", ".jpg", ".jpeg", ".gif", ".bin", ".dll", ".exe", ".pdb",
            ".lib", ".obj", ".o", ".so", ".dylib", ".zip", ".gz", ".7z", ".pt", ".phid", ".wasm", ".ico", ".pdf"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", "models"}
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


def featurize(text: str) -> List[int]:
    """Hashed features of a text: words, neighbouring word pairs, and each word's character trigrams."""
    ws = words(text)
    feats = [f"w:{w}" for w in ws] + [f"b:{a}_{b}" for a, b in zip(ws, ws[1:])]
    for w in set(ws):
        padded = f"<{w}>"
        feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
    ids = [zlib.crc32(f.encode("utf-8")) % NB for f in feats[:MAX_FEATURES * 2]]
    return ids[:MAX_FEATURES] or [0]


def batch_features(texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    flat, offsets = [], []
    for t in texts:
        offsets.append(len(flat))
        flat += featurize(t)
    return torch.tensor(flat, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


class Ranker(nn.Module):
    def __init__(self, nb: int = NB, dim: int = DIM):
        super().__init__()
        self.bag = nn.EmbeddingBag(nb, dim, mode="mean", sparse=True)   # only the rows a batch touches update
        self.body = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.task_head = nn.Linear(dim, dim)
        self.entry_head = nn.Linear(dim, dim)

    def encode(self, texts: Sequence[str], side: str) -> torch.Tensor:
        flat, offsets = batch_features(texts)
        h = self.bag(flat, offsets)
        h = h + self.body(h)
        h = (self.task_head if side == "task" else self.entry_head)(h)
        return F.normalize(h, dim=-1)


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
    """(neighbouring passage pairs across the repository's text, bytes read)."""
    pairs, total = [], 0
    for p in text_files(root):
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


def save(model: Ranker, meta: Dict, path: str) -> None:
    """The weights at fp16, half the bytes in git LFS; load() trains and scores in fp32."""
    save_atomic({"state": {k: v.half() for k, v in model.state_dict().items()}, "meta": meta}, path)


# ─── commands ──────────────────────────────────────────────────────────────────────────────────────────

def cmd_pretrain(args) -> int:
    torch.manual_seed(0)
    pairs, total = corpus_pairs(args.p4)
    print(f"pretrain — {len(pairs):,} passage pairs from {total / 1e6:.1f} MB of text in {args.p4}")
    model = Ranker()
    steps = pretrain(model, pairs, args.minutes)
    path = os.path.join(models_dir(args.p4), "ranker-base.pt")
    save(model, {"nb": NB, "dim": DIM, "stage": "pretrained", "corpus_bytes": total, "steps": steps,
                 "trained": args.now}, path)
    print(f"pretrain — {steps} steps → {path}")
    return 0


def base_model(p4: str, minutes: float, now: str, repretrain: float = 1.25, write: bool = True) -> Tuple[Ranker, Dict]:
    """The pretrained base: models/ranker-base.pt, pretrained again when absent, when it has fewer than
    MIN_PRETRAIN_STEPS (built with no or a tiny budget) and a budget is given now, or when the repository's text has
    grown by `repretrain` times since. write=False (an evaluation) builds a missing base in memory and saves nothing."""
    path = os.path.join(models_dir(p4), "ranker-base.pt")
    if os.path.exists(path):
        model, meta = load(path)
        thin = minutes > 0 and meta.get("steps", 0) < MIN_PRETRAIN_STEPS
        if not write or (not thin and (minutes <= 0 or corpus_pairs(p4)[1] <= meta.get("corpus_bytes", 0) * repretrain)):
            return model, meta
    torch.manual_seed(0)
    pairs, total = corpus_pairs(p4)
    model = Ranker()
    steps = pretrain(model, pairs, minutes) if minutes > 0 else 0
    meta = {"nb": NB, "dim": DIM, "stage": "pretrained", "corpus_bytes": total, "steps": steps, "trained": now}
    if write:
        save(model, meta, path)
    print(f"pretrain — {steps} steps over {total / 1e6:.1f} MB of text" + (f" → {path}" if write else " (not saved)"))
    return model, meta


def heldout_threshold(base_state: Dict, pools: Sequence[Dict], epochs: int, folds: int = 4) -> Tuple[float, str]:
    """The pick threshold from scores on pools each fold's model never trained on; with too few pools to fold,
    the training pools' own threshold, said so."""
    if len(pools) < 4:
        model = Ranker()
        model.load_state_dict(base_state)
        finetune(model, pools, epochs=epochs)
        return best_threshold(model, pools), "train"
    k = min(folds, len(pools))
    scored = []
    for i in range(k):
        held = pools[i::k]
        model = Ranker()
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
    return best, f"held-out, {k} folds"


def cmd_train(args) -> int:
    pools = read_pools(args.pairs)
    if not pools:
        print("train — no labelled pool in the pairs; nothing written")
        return 1
    model, base = base_model(args.p4, args.minutes, args.now)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    t, how = heldout_threshold(base_state, pools, args.epochs)
    torch.manual_seed(0)
    finetune(model, pools, epochs=args.epochs)
    path = os.path.join(models_dir(args.p4), "ranker.pt")
    side_path = os.path.join(models_dir(args.p4), "ranker.json")
    try:
        with open(side_path, encoding="utf-8") as f:
            side = json.load(f)
    except (OSError, ValueError):
        side = {}
    rnd = side["round"] + 1 if isinstance(side.get("round"), int) and os.path.exists(path) else 0
    meta = {"nb": NB, "dim": DIM, "stage": "finetuned", "round": rnd, "threshold": t, "threshold_from": how,
            "per_file": PER_FILE, "trained": args.now, "pools": [p["id"] for p in pools],
            "corpus_bytes": base.get("corpus_bytes", 0), "base_steps": base.get("steps", 0)}
    save(model, meta, path)
    # the sidecar the plugin's hooks read without torch: what this checkpoint has trained on; `seen` is kept and
    # widened by the plugin (pools it met with nothing to learn count as met)
    side.update({k: meta[k] for k in ("round", "threshold", "threshold_from", "trained", "pools", "per_file")})
    side["seen"] = sorted(set(side.get("seen", [])) | set(meta["pools"]))
    write_text_atomic(json.dumps(side, indent=1) + "\n", side_path)
    print(f"train — round {rnd}: {len(pools)} pools, threshold {t:.2f} ({how}) → {path}")
    return 0


def cmd_eval(args) -> int:
    pools = read_pools(args.pairs)
    if len(pools) < 3:
        print(f"eval — {len(pools)} labelled pools; at least 3 are needed")
        return 1
    base, _meta = base_model(args.p4, args.minutes, args.now or "", write=False)
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    folds = [pools[i::args.folds] for i in range(args.folds)]
    got = []
    for k, held in enumerate(folds):
        if not held:
            continue
        model = Ranker()
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phi_prune.p4rank")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pretrain")
    p.add_argument("--p4", required=True)
    p.add_argument("--minutes", type=float, default=20)
    p.add_argument("--now", default="")
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--p4", required=True)
        p.add_argument("--pairs", required=True, help="pick_model.py pairs --jsonl output")
        p.add_argument("--minutes", type=float, default=20, help="pretraining budget when the base is (re)built")
        p.add_argument("--epochs", type=int, default=30)
        p.add_argument("--now", default="")
        if name == "eval":
            p.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)
    return {"pretrain": cmd_pretrain, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
