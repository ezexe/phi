"""
"Outperform" beyond a fixed held-out score: one large ranker against a nested tree of small ones, replayed
through a store's history in time order, the way the p4 models actually live. Stage 2 of the experiment
.docs/p4/nested-store.md names; stage 1 is nested_vs_one.py.

Every arm is pretrained, then the labelled pools are replayed in the order they were pooled. Each pool is
scored first by the arm as it stands (prequential: the pool is fresh, never trained on) and then learned (a
short fine-tune on it plus a replay of the pools before it). The update the arm ships is encoded as a packed
phi_prune.delta, the format the p4 rounds use. Measured per arm:

  freshness   prequential r-precision over the pools after the first WARM (each scored before it is learned)
  forgetting  r-precision on the first WARM pools after the whole replay, against right after warm-up
  update      seconds to absorb one new pool (the fine-tune), mean over the replay
  message     bytes of the update a node ships to its peers: the packed phi-delta (keep 10% of what moved), and
              for the tree the node's Fibonacci address, which a reader needs to know which model it is
  query       milliseconds to score one pool, mean
  size        the model's bytes at fp16 (its checkpoint) and its parameter count

Arms (equal total parameters):
  big       one 2^B x DIM ranker
  node      the nested tree's root (2^(B-2) x DIM, the root of a 4-node chain pretrained shard by shard as in stage
            1); a new pool updates the root alone, the way a session updates only its own store's model
  ensemble  the 4 chain nodes, z-scored and averaged; a new pool updates every node (4 messages)

Run (2026-10-27):
  python scripts/session/nested_realtime.py --p4 <store> --pairs <pairs.jsonl> --minutes 8 --warm 8 --seeds 0 1
"""

import argparse
import functools
import json
import math
import os
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phi_rank as pr  # noqa: E402
from nested_vs_one import pairs_of, shards, zmean  # noqa: E402
from phi_prune.delta import encode_delta, save_delta  # noqa: E402

RAW = pr.featurize
CACHED = functools.lru_cache(maxsize=None)(pr.featurize)


def pools_in_time(pairs_jsonl):
    pooled = {}
    with open(pairs_jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pooled.setdefault(r.get("pool_id"), r.get("pooled", ""))
    pools = pr.read_pools(pairs_jsonl)
    return sorted(pools, key=lambda p: pooled.get(p["id"], ""))


def rprec(models, pool, combine):
    texts = [r["text"] for r in pool["rows"]]
    t0 = time.perf_counter()
    sc = combine([pr.score_pool(m, pool["task"], texts) for m in models])
    ms = (time.perf_counter() - t0) * 1000
    return pr.pool_metrics([r["label"] for r in pool["rows"]], sc, [])["r-precision"], ms


def delta_bytes(before, after):
    header, payload = encode_delta(before, after, keep=pr.KEEP, n_digits=pr.N_DIGITS, changed_only=True)
    with tempfile.NamedTemporaryFile(suffix=".phid", delete=False) as f:
        path = f.name
    save_delta(path, header, payload, packed=True)
    n = os.path.getsize(path)
    os.remove(path)
    return n


def replay(models, pools, warm, epochs, combine, address_bits):
    """address_bits: one per model, the bits of its address carried by each message it ships."""
    """Warm up on the first `warm` pools, then score-then-learn each later pool in time order."""
    state = lambda m: {k: v.clone() for k, v in m.state_dict().items()}  # noqa: E731
    pr.featurize = CACHED
    for m in models:
        torch.manual_seed(0)
        pr.finetune(m, pools[:warm], epochs=epochs)
    early_before = sum(rprec(models, p, combine)[0] for p in pools[:warm]) / warm
    fresh, upd_s, msg, q_ms = [], [], [], []
    for t in range(warm, len(pools)):
        r, ms = rprec(models, pools[t], combine)
        fresh.append(r)
        q_ms.append(ms)
        t0 = time.perf_counter()
        for m, bits in zip(models, address_bits):
            before = state(m)
            pr.finetune(m, pools[max(0, t - 4):t + 1], epochs=max(1, epochs // 3), seed=t)
            msg.append(delta_bytes(before, m.state_dict()) + bits / 8)
        upd_s.append(time.perf_counter() - t0)
    early_after = sum(rprec(models, p, combine)[0] for p in pools[:warm]) / warm
    pr.featurize = RAW
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    params = sum(p.numel() for m in models for p in m.parameters())
    return {"freshness": round(mean(fresh), 4), "forgetting": round(early_before - early_after, 4),
            "early_before": round(early_before, 4), "early_after": round(early_after, 4),
            "update_s": round(mean(upd_s), 2), "message_bytes_per_update": round(sum(msg) / max(1, len(upd_s))),
            "query_ms": round(mean(q_ms), 1), "params": params, "checkpoint_mb_fp16": round(params * 2 / 1e6, 2)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bits", type=int, default=15)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    pools = pools_in_time(args.pairs)
    shard_pairs = [pairs_of(args.p4, s) for s in shards(args.p4, args.k)]
    all_pairs = [x for ps, _ in shard_pairs for x in ps]
    small = args.bits - int(round(math.log2(args.k)))
    # a node's Fibonacci address in the tree code: the chain's node i (1 = deepest) sits at depth i, 2i bits
    address = [2 * (i + 1) for i in range(args.k)]
    report = {"pools": len(pools), "warm": args.warm, "k": args.k, "runs": []}
    for seed in args.seeds:
        torch.manual_seed(seed)
        big = pr.Ranker(1 << args.bits, args.dim)
        pr.pretrain(big, all_pairs, args.minutes, seed=seed, log=lambda *_: None)
        nodes, prev = [], None
        for i, (ps, _b) in enumerate(shard_pairs):
            m = pr.Ranker(1 << small, args.dim)
            if prev is not None:
                m.load_state_dict(prev)
            pr.pretrain(m, ps, args.minutes / args.k, seed=seed * 10 + i, log=lambda *_: None)
            prev = {k: v.clone() for k, v in m.state_dict().items()}
            nodes.append(m)
        root = pr.ranker_like(nodes[-1].state_dict())
        root.load_state_dict(nodes[-1].state_dict())
        run = {"seed": seed,
               "big": replay([big], pools, args.warm, args.epochs, lambda s: s[0], [0]),
               "node": replay([root], pools, args.warm, args.epochs, lambda s: s[0], address[-1:]),
               "ensemble": replay(nodes, pools, args.warm, args.epochs, zmean, address)}
        report["runs"].append(run)
        print(json.dumps(run), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
