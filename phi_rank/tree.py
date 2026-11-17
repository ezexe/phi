"""
The node tree behind the p4-model skill's effort tiers: K small rankers at the first K Zeckendorf store addresses,
each fine-tuned on the store's labelled pools. Measured in docs/p4/nested-store.md (stage 3, nested_suite.py) and
in the p4-phi experiments' phi_depth (github.com/ezexe/p4-phi-experiments, 2026-11-09).

How it is built:
- the store's tracked text is cut into K shards of equal bytes by when git first added each file; the oldest shard
  goes to the deepest address
- a leaf node starts from scratch (a node is 2^(B - log2 K) rows: the shared base, 2^B rows, does not fit it); a node
  with children starts from the mean of their weights, so the root has learned from the whole tree
- each node is fine-tuned on every labelled pool. Two steps are off by default since phi_depth measured them on a
  97-pool store (3 seeds, K = 4, held-out R-precision at high): pretraining each node on its shard for MINUTES / K
  (0.621 with a minute of it against 0.633 without; one ranker 0.519 against 0.567), and keeping only the rows
  phi_mask(density 0.5) keeps (0.586 against 0.633 unmasked; every half mask, φ, 2:4, heaviest, random, cost the
  same, so the cost is halving the rows, not the pattern). --minutes and --mask turn them back on.
- what high answers is the tree's best measured tier: 0.633 against 0.567 for one ranker of the same total size.
  medium routes at chance on that store (the routed node was the best one for 47% of pools, 50% by chance).

The limit: those numbers are on phi_rank's pools, which hold every labelled entry of a task and ten BM25 negatives.
On p4-recall's own candidates (the top 200 of BM25, where the labelled entries are mostly absent) the same tree
ranked 0.012 R-precision against BM25's 0.096 on that store's newest quarter of tasks: it is not a reranker for
recall's lines as it stands.

How it answers (phi_serve, by the request's effort):
  medium  the one node whose rows carry the most weight on the task's features (a masked tree's zeroed rows count
          nothing, so there the pruning matrix is the route)
  high    every node, each node's scores z-scored and averaged; Tree.score's default
Any other effort, or no tree on disk, is the store's single ranker (phi_rank.load_current), as before.

Files, in the store's models/: tree/<address>.pt (fp16, git LFS), tree.json (the addresses, the shards' time spans,
when it was trained) and tree-oof.json: every labelled pool scored at medium and at high by fold trees that never
fine-tuned on it, with each effort's held-out r-precision. The JSON ranker fits its .pt weight on those scores when
the store's pool-effort names the tree, as it fits it on ranker-oof.json for the single ranker.
"""

import itertools
import json
import math
import os
import subprocess
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

import phi_rank as pr
from phi_prune.masks import phi_mask

TREE_DIR = "tree"
TREE_JSON = "tree.json"
EFFORTS = ("medium", "high")


# ─── the shards ────────────────────────────────────────────────────────────────────────────────────────

def first_added(root: str) -> Dict[str, int]:
    """{path: unix time git first added it} from one git log."""
    out = subprocess.run(["git", "-C", root, "log", "--format=@%at", "--name-only", "--diff-filter=A"],
                         capture_output=True, text=True).stdout
    when, t = {}, 0
    for line in out.split("\n"):
        if line.startswith("@"):
            t = int(line[1:])
        elif line.strip():
            when[line.strip()] = t            # later lines are older commits: the oldest wins
    return when


def shard_files(root: str, k: int) -> Tuple[List[List[str]], List[Tuple[int, int]]]:
    """The store's text files cut into k shards of equal bytes, oldest first, and each shard's first-added span."""
    files = pr.text_files(root)
    when = first_added(root)
    rel = lambda p: os.path.relpath(p, root).replace(os.sep, "/")  # noqa: E731
    files.sort(key=lambda p: (when.get(rel(p), 1 << 62), rel(p)))
    sizes = [min(os.path.getsize(p), pr.MAX_FILE_BYTES) for p in files]
    total, out, cur, acc = sum(sizes), [[] for _ in range(k)], 0, 0
    for p, s in zip(files, sizes):
        if acc >= total * (cur + 1) / k and cur < k - 1:
            cur += 1
        out[cur].append(p)
        acc += s
    spans = []
    for fs in out:
        ts = [when[rel(p)] for p in fs if rel(p) in when]
        spans.append((min(ts) if ts else 0, max(ts) if ts else 0))
    return out, spans


def shard_pairs(root: str, files: Sequence[str]) -> List[Tuple[str, str]]:
    pairs = []
    for p in files:
        ps = pr.passages(p)
        r = os.path.relpath(p, root)
        pairs += [(f"{r}\n{a}", b) for a, b in zip(ps, ps[1:])]
        if len(ps) == 1:
            pairs.append((r, ps[0]))
    return pairs


# ─── the Zeckendorf addresses ──────────────────────────────────────────────────────────────────────────

def addresses(k: int) -> List[str]:
    """The first k store addresses (p4 = 1, .claude = 0; start and end with 1, no 11), shortest first."""
    out, m = [], 1
    while len(out) < k:
        out += sorted(a for a in ("".join(t) for t in itertools.product("01", repeat=m))
                      if a[0] == "1" and a[-1] == "1" and "11" not in a)
        m += 1
    return out[:k]


def parent(a: str) -> Optional[str]:
    """The address a store at `a` was grown under: its last run of .claude's and the p4 after it dropped."""
    return None if a == "1" else a[:-1].rstrip("0")


# ─── build ─────────────────────────────────────────────────────────────────────────────────────────────

def node_rows(models: Sequence[pr.Ranker]) -> List[torch.Tensor]:
    """Each node's row weights, the route's matrix: zero on every row its mask pruned."""
    return [m.bag.weight.detach().norm(dim=1) for m in models]


def route(rows: Sequence[torch.Tensor], task: str) -> int:
    """The node whose kept rows carry the most weight on the task's features."""
    ids = torch.tensor(pr.featurize(task, rows[0].shape[0]))
    return int(torch.stack([r[ids].sum() for r in rows]).argmax())


@torch.no_grad()
def answer(models: Sequence[pr.Ranker], rows: Sequence[torch.Tensor], task: str, texts: Sequence[str],
           effort: str) -> Tuple[List[float], int]:
    """(scores, the routed node at medium or -1 at high)."""
    if effort == "medium":
        i = route(rows, task)
        return pr.score_pool(models[i], task, texts), i
    total = None
    for m in models:
        sc = pr.score_pool(m, task, texts)
        mu = sum(sc) / len(sc)
        sd = math.sqrt(sum((x - mu) ** 2 for x in sc) / len(sc)) or 1.0
        z = [(x - mu) / sd for x in sc]
        total = z if total is None else [a + b for a, b in zip(total, z)]
    return [x / len(models) for x in total], -1


MASK = False                  # phi_mask the nodes' feature rows (measured to cost 0.05; --mask turns it on)
MINUTES = 0.0                 # pretraining per tree, split over its nodes (measured to lower ranking; --minutes N)


def mask_rows(m: pr.Ranker) -> torch.Tensor:
    """phi_prune's adjacency mask over the node's feature rows, applied in place; the mask is returned."""
    mask = phi_mask(m.bag.weight.detach(), axis=0, density=0.5)
    with torch.no_grad():
        m.bag.weight.mul_(mask)
    return mask


def build(p4: str, pools: Sequence[Dict], k: int = 4, minutes: float = MINUTES, epochs: int = 10,
          bits: int = pr.NB.bit_length() - 1, dim: int = pr.DIM, now: str = "", log=print, mask: bool = MASK) -> Dict:
    files, spans = shard_files(p4, k)
    nb = 1 << (bits - int(round(math.log2(k))))
    order = sorted(addresses(k), key=lambda a: (-len(a), a))     # deepest first: the oldest shard
    trained: Dict[str, Dict] = {}
    out_dir = os.path.join(pr.models_dir(p4), TREE_DIR)
    os.makedirs(out_dir, exist_ok=True)
    nodes = []
    for i, a in enumerate(order):
        m = pr.Ranker(nb, dim)
        kids = [trained[c] for c in trained if parent(c) == a]
        if kids:
            m.load_state_dict({key: sum(s[key] for s in kids) / len(kids) for key in kids[0]})
        torch.manual_seed(i)
        t0 = time.time()
        log(f"tree - node {a} ({i + 1}/{k}): " + (f"pretraining {minutes / k * 60:.0f} s on shard {i + 1}, then "
                                                    if minutes > 0 else "") + "fine-tuning")
        steps = pr.pretrain(m, shard_pairs(p4, files[i]), minutes / k, seed=i, log=lambda *_: None) if minutes > 0 else 0
        trained[a] = {key: v.clone() for key, v in m.state_dict().items()}   # what a parent starts from
        pr.finetune(m, pools, epochs=epochs, seed=i)
        if mask:
            mask_rows(m)
        pr.save(m, {"nb": nb, "dim": dim, "stage": "tree", "address": a, "trained": now, "masked": mask}, os.path.join(out_dir, f"{a}.pt"))
        nodes.append({"address": a, "shard": i, "files": len(files[i]), "span": list(spans[i]), "steps": steps})
        log(f"tree - node {a}: shard {i + 1}/{k}, {len(files[i])} files, {steps} steps, {time.time() - t0:.0f} s")
    meta = {"k": k, "nb": nb, "dim": dim, "trained": now, "nodes": nodes, "pools": [p["id"] for p in pools],
            "masked": mask, "minutes": minutes}
    pr.write_text_atomic(json.dumps(meta, indent=1) + "\n", os.path.join(pr.models_dir(p4), TREE_JSON))
    return meta

