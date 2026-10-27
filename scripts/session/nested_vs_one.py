"""
Many nested small models against one large model, at equal total parameters: the experiment
.docs/p4/nested-store.md names to test the claim that a recursive tree of small models outperforms one
large model.

Data: a p4 store's tracked text, ordered by when git first added each file and cut into K shards of equal
bytes (shard 1, the oldest, is the deepest node), and the store's labelled pools (pick_model.py pairs).

Arms, all scored by the same FOLDS-fold held-out r-precision on the pools, each fine-tuned on the same folds:
  big       one 2^B x DIM ranker pretrained on all the text for T minutes
  small     one 2^(B - log2 K) x DIM ranker on all the text for T/K minutes (a single node, for scale)
  chain     K small rankers: node 1 pretrains on shard 1; node n starts from node n-1's weights and continues on
            shard n, so each node's own training carries the whole subtree below it (rvalue(n) = lvalue(n+1),
            by design); the root (node K) is scored
  ensemble  the chain's K nodes, each fine-tuned, their z-scored scores averaged per pool (K x the small size =
            the big model's parameters)

Feature hashing is cached while fine-tuning (functools.lru_cache over the pools' texts), which changes no
result, only the time.

Run (2026-10-27):
  python scripts/session/nested_vs_one.py --p4 <store> --pairs <pairs.jsonl> --k 4 --minutes 8 --epochs 15 --seeds 0 1
"""

import argparse
import functools
import json
import math
import os
import random
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import phi_rank as pr  # noqa: E402

RAW_FEATURIZE = pr.featurize
CACHED_FEATURIZE = functools.lru_cache(maxsize=None)(pr.featurize)   # the same features, computed once a text


def first_added(root):
    """{path: unix time git first added it}: one git log, oldest commit last."""
    out = subprocess.run(["git", "-C", root, "log", "--format=@%at", "--name-only", "--diff-filter=A"],
                         capture_output=True, text=True).stdout
    when, t = {}, 0
    for line in out.split("\n"):
        if line.startswith("@"):
            t = int(line[1:])
        elif line.strip():
            when[line.strip()] = t            # later lines are older commits: keep overwriting, oldest wins
    return when


def shards(root, k):
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
    return out


def pairs_of(root, files):
    pairs, total = [], 0
    for p in files:
        ps = pr.passages(p)
        total += sum(len(x) for x in ps)
        r = os.path.relpath(p, root)
        pairs += [(f"{r}\n{a}", b) for a, b in zip(ps, ps[1:])]
        if len(ps) == 1:
            pairs.append((r, ps[0]))
    return pairs, total


def heldout(models_init, pools, folds, epochs, seed, combine):
    """Mean held-out r-precision: for each fold, every model in models_init (state dicts) is fine-tuned on the
    other folds and scores the held-out pools; combine merges the models' per-pool scores."""
    got = []
    pr.featurize = CACHED_FEATURIZE        # fine-tuning repeats the pools' texts every epoch: hash each once
    for f in range(folds):
        train = [p for j, p in enumerate(pools) if j % folds != f]
        held = [p for j, p in enumerate(pools) if j % folds == f]
        tuned = []
        for st in models_init:
            m = pr.ranker_like(st)
            m.load_state_dict(st)
            torch.manual_seed(seed * 100 + f)
            pr.finetune(m, train, epochs=epochs, seed=seed * 100 + f)
            tuned.append(m)
        for p in held:
            texts = [r["text"] for r in p["rows"]]
            sc = combine([pr.score_pool(m, p["task"], texts) for m in tuned])
            got.append(pr.pool_metrics([r["label"] for r in p["rows"]], sc, [])["r-precision"])
    pr.featurize = RAW_FEATURIZE
    return sum(got) / len(got)


def zmean(score_lists):
    out = None
    for sc in score_lists:
        mu = sum(sc) / len(sc)
        sd = math.sqrt(sum((x - mu) ** 2 for x in sc) / len(sc)) or 1.0
        z = [(x - mu) / sd for x in sc]
        out = z if out is None else [a + b for a, b in zip(out, z)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bits", type=int, default=15, help="the big model's 2^bits buckets")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--minutes", type=float, default=8, help="the big model's pretraining; each node gets 1/k")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    pools = pr.read_pools(args.pairs)
    sh = shards(args.p4, args.k)
    shard_pairs = [pairs_of(args.p4, s) for s in sh]
    all_pairs = [x for ps, _ in shard_pairs for x in ps]
    small_bits = args.bits - int(round(math.log2(args.k)))
    report = {"pools": len(pools), "k": args.k, "big": f"2^{args.bits} x {args.dim}",
              "small": f"2^{small_bits} x {args.dim}", "minutes_big": args.minutes, "epochs": args.epochs,
              "folds": args.folds, "shards": [{"files": len(s), "pairs": len(p), "mb": round(b / 1e6, 2)}
                                               for s, (p, b) in zip(sh, shard_pairs)], "runs": []}
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=1), flush=True)
    for seed in args.seeds:
        t0, run = time.time(), {"seed": seed}
        torch.manual_seed(seed)
        random.seed(seed)
        big = pr.Ranker(1 << args.bits, args.dim)
        run["big_steps"] = pr.pretrain(big, all_pairs, args.minutes, seed=seed, log=lambda *_: None)
        small = pr.Ranker(1 << small_bits, args.dim)
        run["small_steps"] = pr.pretrain(small, all_pairs, args.minutes / args.k, seed=seed, log=lambda *_: None)
        nodes, prev, steps = [], None, []
        for i, (ps, _b) in enumerate(shard_pairs):
            m = pr.Ranker(1 << small_bits, args.dim)
            if prev is not None:
                m.load_state_dict(prev)                     # node n starts where node n-1 ended
            steps.append(pr.pretrain(m, ps, args.minutes / args.k, seed=seed * 10 + i, log=lambda *_: None))
            prev = {k: v.clone() for k, v in m.state_dict().items()}
            nodes.append(prev)
        run["chain_steps"] = steps
        sd = lambda m: {k: v.clone() for k, v in m.state_dict().items()}  # noqa: E731
        run["big"] = heldout([sd(big)], pools, args.folds, args.epochs, seed, lambda s: s[0])
        run["small"] = heldout([sd(small)], pools, args.folds, args.epochs, seed, lambda s: s[0])
        run["chain_root"] = heldout([nodes[-1]], pools, args.folds, args.epochs, seed, lambda s: s[0])
        run["ensemble"] = heldout(nodes, pools, args.folds, args.epochs, seed, zmean)
        run["seconds"] = round(time.time() - t0)
        report["runs"].append(run)
        print(json.dumps(run), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
