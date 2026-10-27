"""
Every angle of "a nested tree of small models outperforms one large model", measured on one store, one run.
Supersedes nested_realtime.py (stage 2); nested_vs_one.py was stage 1. For .docs/p4/nested-store.md.

Data: a p4 store's tracked text cut into K shards of equal bytes by when git first added each file (shard 1, the
oldest, is the deepest node), and the store's labelled pools (pick_model.py pairs --jsonl), in date order.

Arms (models are pretrained once per seed and shared between the arms that use them):
  big         one 2^B x D ranker, all the text, T minutes
  small       one 2^(B-log2 K) x D ranker, all the text, T/K minutes
  chain       K small nodes; node n starts from node n-1's weights and continues on shard n (T/K each)
                chain_root (node K scores) and chain_ens (all K, z-scored and averaged)
  indep       K small nodes, each from scratch on its own shard only (T/K each): no handoff
                indep_ens (all K averaged) and routed (each candidate scored only by the node whose time span
                holds the entry: the query goes to the data's owner)
  zeck        K small nodes at the first K Zeckendorf addresses (1, 101, 1001, 10101, 10001, 100101, ...): a node
              starts from the average of its children's weights, so a node with two children merges them
                zeck_root and zeck_ens
  grown       K nodes each 2^B x D (as large as big), chain handoff, T/K each: total parameters grow with the
              nodes, the "many models outgrow one" angle
                grown_root and grown_ens
  zeckq       the zeck nodes pruned by phi_prune's adjacency mask (phi_mask, density 0.5) over their feature table's
              rows, the mask re-applied after every fine-tune
                zeckq_ens, and qrouted: the matrix that pruned each node also routes the query. A task goes to the
                one node whose kept rows carry the most weight on the task's features, and only that node searches
                and only that node learns
  zeck24      the same nodes pruned 2:4 (two_four_mask), the control phi has been measured against
                zeck24_ens and q24routed

Axes, per arm:
  quality       FOLDS-fold held-out r-precision (every model fine-tuned on the same train folds)
  deployed      the arm's out-of-fold scores as the JSON ranker's .pt feature: its leave-one-pool-out
                r-precision with the feature against without (the plugin's own pick_model functions)
  drop1         ensembles: the mean held-out r-precision with one node dropped, over every node
  route_hit     matrix-routed arms: how often the routed node is the node that ranks that pool best
  fresh         prequential: after WARM pools, each later pool scored before it is learned
  forget        r-precision on the WARM pools right after warm-up minus after the whole stream
  update_s      seconds to absorb one new pool (routed: only the newest node, which owns new data)
  msg_bytes     bytes shipped per new pool: each updated model's packed phi-delta plus its Fibonacci address
  query_ms      milliseconds to score one pool
  params, mb    parameters and fp16 checkpoint size
  pretrain_s    pretraining seconds, sequential sum and parallel (independent nodes only: the longest one)

Run (2026-10-27):
  python scripts/session/nested_suite.py --p4 <store> --pairs <pairs.jsonl> --plugin <vlds> --k 4 --minutes 6 --seeds 0 1
"""

import argparse
import datetime
import functools
import json
import math
import os
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import phi_rank as pr  # noqa: E402
from nested_vs_one import first_added, pairs_of, shards, zmean  # noqa: E402
from phi_prune.delta import encode_delta, save_delta  # noqa: E402
from phi_prune.masks import phi_mask, two_four_mask  # noqa: E402

RAW = pr.featurize
CACHED = functools.lru_cache(maxsize=None)(pr.featurize)


# ─── data ──────────────────────────────────────────────────────────────────────────────────────────────

def load_pools(path):
    """Pools in date order, each row with its entry's unix time (None when the entry has none)."""
    pools, order = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("pool_id")
            if pid not in pools:
                pools[pid] = {"id": pid, "task": r["task"], "pooled": r.get("pooled", ""), "rows": []}
                order.append(pid)
            t = None
            if r.get("time"):
                try:
                    t = datetime.datetime.strptime(r["time"][:16], "%Y-%m-%d %H:%M").timestamp()
                except ValueError:
                    t = None
            pools[pid]["rows"].append({"file": r["file"], "text": r["text"], "label": int(r["label"]), "t": t})
    out = [pools[p] for p in order if any(x["label"] for x in pools[p]["rows"])]
    return sorted(out, key=lambda p: p["pooled"])


def shard_spans(root, shard_files):
    when = first_added(root)
    spans = []
    for files in shard_files:
        ts = [when.get(os.path.relpath(p, root).replace(os.sep, "/")) for p in files]
        ts = [t for t in ts if t]
        spans.append((min(ts) if ts else 0, max(ts) if ts else 0))
    return spans


def owner(t, spans):
    """The node whose shard's time span holds t; an entry with no time, or newer than all, is the newest node's."""
    if t is None:
        return len(spans) - 1
    for i, (_a, b) in enumerate(spans):
        if t <= b:
            return i
    return len(spans) - 1


def zeck_addresses(k):
    import itertools
    out = []
    m = 1
    while len(out) < k:
        out += sorted("".join(t) for t in itertools.product("01", repeat=m)
                      if "".join(t)[0] == "1" and "".join(t)[-1] == "1" and "11" not in "".join(t))
        m += 1
    return out[:k]


def zeck_parent(a):
    """The address a store at `a` was grown under: drop its last run of .claude's and the p4 after it."""
    if a == "1":
        return None
    b = a[:-1].rstrip("0")
    return b


# ─── models ────────────────────────────────────────────────────────────────────────────────────────────

def state(m):
    return {k: v.clone() for k, v in m.state_dict().items()}


def pretrain_timed(m, pairs, minutes, seed):
    t0 = time.perf_counter()
    pr.pretrain(m, pairs, minutes, seed=seed, log=lambda *_: None)
    return time.perf_counter() - t0


def build_models(args, shard_pairs, all_pairs, seed):
    K, B, D, T = args.k, args.bits, args.dim, args.minutes
    sb = B - int(round(math.log2(K)))
    M, times = {}, {}
    torch.manual_seed(seed)
    m = pr.Ranker(1 << B, D)
    times["big"] = [pretrain_timed(m, all_pairs, T, seed)]
    M["big"] = [state(m)]
    m = pr.Ranker(1 << sb, D)
    times["small"] = [pretrain_timed(m, all_pairs, T / K, seed)]
    M["small"] = [state(m)]
    for name, bits in (("chain", sb), ("grown", B)):
        prev, M[name], times[name] = None, [], []
        for i, (ps, _b) in enumerate(shard_pairs):
            m = pr.Ranker(1 << bits, D)
            if prev is not None:
                m.load_state_dict(prev)
            times[name].append(pretrain_timed(m, ps, T / K, seed * 10 + i))
            prev = state(m)
            M[name].append(prev)
    M["indep"], times["indep"] = [], []
    for i, (ps, _b) in enumerate(shard_pairs):
        m = pr.Ranker(1 << sb, D)
        times["indep"].append(pretrain_timed(m, ps, T / K, seed * 10 + i))
        M["indep"].append(state(m))
    # the Zeckendorf tree: deepest addresses train first on the oldest shards; a node starts from the mean of its
    # children's weights; the root is the newest
    addrs = zeck_addresses(K)
    order = sorted(addrs, key=lambda a: (-len(a), a))          # deepest first
    trained, M["zeck"], times["zeck"] = {}, [], []
    for i, a in enumerate(order):
        m = pr.Ranker(1 << sb, D)
        kids = [trained[c] for c in trained if zeck_parent(c) == a]
        if kids:
            m.load_state_dict({k: sum(s[k] for s in kids) / len(kids) for k in kids[0]})
        times["zeck"].append(pretrain_timed(m, shard_pairs[i][0], T / K, seed * 10 + i))
        trained[a] = state(m)
    M["zeck"] = [trained[a] for a in order]                      # index K-1 is the root ("1")
    M["zeck_addresses"] = order
    for fam, maker in (("zeckq", lambda w: phi_mask(w, axis=0, density=0.5)), ("zeck24", lambda w: two_four_mask(w, axis=0))):
        M[fam], MASKS[fam], ROWNORM[fam] = [], [], []
        for st in M["zeck"]:
            w = st["bag.weight"]
            mask = maker(w)
            st2 = {k: v.clone() for k, v in st.items()}
            st2["bag.weight"] = w * mask
            M[fam].append(st2)
            MASKS[fam].append(mask)
            ROWNORM[fam].append(st2["bag.weight"].norm(dim=1))
    return M, times


MASKS, ROWNORM = {}, {}


def route(fam, task):
    """The node whose kept rows carry the most weight on the task's features: the pruning matrix as the route."""
    nb = ROWNORM[fam][0].shape[0]
    ids = torch.tensor(RAW(task, nb))
    return int(torch.stack([rn[ids].sum() for rn in ROWNORM[fam]]).argmax())


def remask(fam, i, m):
    if fam in MASKS:
        with torch.no_grad():
            m.bag.weight.mul_(MASKS[fam][i])


ARMS = {  # arm: (model family, which models, how their scores combine)
    "big": ("big", "all", "one"), "small": ("small", "all", "one"),
    "chain_root": ("chain", "last", "one"), "chain_ens": ("chain", "all", "zmean"),
    "indep_ens": ("indep", "all", "zmean"), "routed": ("indep", "all", "route"),
    "zeck_root": ("zeck", "last", "one"), "zeck_ens": ("zeck", "all", "zmean"),
    "grown_root": ("grown", "last", "one"), "grown_ens": ("grown", "all", "zmean"),
    "zeckq_ens": ("zeckq", "all", "zmean"), "qrouted": ("zeckq", "all", "qroute"),
    "zeck24_ens": ("zeck24", "all", "zmean"), "q24routed": ("zeck24", "all", "qroute"),
}


def combine(how, score_lists, pool, spans, idx, fam=None):
    if how == "one":
        return score_lists[0]
    if how == "qroute":
        return score_lists[route(fam, pool["task"])]
    if how == "zmean":
        return zmean(score_lists)
    # route: each candidate takes the z-score its owning node gives it
    zs = [zmean([s]) for s in score_lists]
    own = [owner(r["t"], spans) for r in pool["rows"]]
    return [zs[idx.index(o)][j] if o in idx else zs[-1][j] for j, o in enumerate(own)]


def load(st):
    m = pr.ranker_like(st)
    m.load_state_dict(st)
    return m


def rp(pool, scores):
    return pr.pool_metrics([r["label"] for r in pool["rows"]], scores, [])["r-precision"]


# ─── the axes ──────────────────────────────────────────────────────────────────────────────────────────

def quality(M, pools, args, seed, spans):
    """Held-out r-precision per arm, the out-of-fold score lists per arm (for the deployed axis), and drop-one."""
    folds = args.folds
    per_model = {}                                     # (family, i) -> {pool id: scores}, out of fold
    pr.featurize = CACHED
    for fam in ("big", "small", "chain", "indep", "zeck", "grown", "zeckq", "zeck24"):
        for i, st in enumerate(M[fam]):
            got = {}
            for f in range(folds):
                train = [p for j, p in enumerate(pools) if j % folds != f]
                m = load(st)
                torch.manual_seed(seed * 100 + f)
                pr.finetune(m, train, epochs=args.epochs, seed=seed * 100 + f)
                remask(fam, i, m)
                for j, p in enumerate(pools):
                    if j % folds == f:
                        got[p["id"]] = pr.score_pool(m, p["task"], [r["text"] for r in p["rows"]])
            per_model[(fam, i)] = got
    pr.featurize = RAW
    out, oof = {}, {}
    for arm, (fam, which, how) in ARMS.items():
        idx = list(range(len(M[fam]))) if which == "all" else [len(M[fam]) - 1]
        scores = {p["id"]: combine(how, [per_model[(fam, i)][p["id"]] for i in idx], p, spans, idx, fam) for p in pools}
        out[arm] = sum(rp(p, scores[p["id"]]) for p in pools) / len(pools)
        oof[arm] = scores
        if how == "qroute":
            hits = 0
            for p in pools:
                per = [rp(p, per_model[(fam, i)][p["id"]]) for i in idx]
                hits += per[route(fam, p["task"])] >= max(per)
            out[arm + "_route_hit"] = round(hits / len(pools), 3)
        if how == "zmean" and len(idx) > 1:
            drops = []
            for d in idx:
                keep = [i for i in idx if i != d]
                drops.append(sum(rp(p, zmean([per_model[(fam, i)][p["id"]] for i in keep])) for p in pools) / len(pools))
            out[arm + "_drop1"] = round(sum(drops) / len(drops), 4)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}, oof


def deployed(oof, pools, plugin_scripts, store):
    """Each arm's out-of-fold scores as the JSON ranker's .pt feature: leave-one-pool-out with against without."""
    sys.path.insert(0, plugin_scripts)
    import pick_model as pm
    jpools, data = pm.build_pairs(store)
    feats = [pm.pool_examples(p, data, False) for p in jpools]
    base = pm.loo_rprecision(jpools, feats, data)
    texts_of = {p["id"]: [r["text"] for r in p["rows"]] for p in pools}
    out = {"json_without": round(base, 4)}
    for arm, scores in oof.items():
        as_oof = {pid: {pm.text_key(t): s for t, s in zip(texts_of[pid], sc)} for pid, sc in scores.items()}
        pts = [pm.pool_pt(p, data, as_oof) for p in jpools]
        with_pt = [(pm.add_pt([dict(x) for x in f[0]], sc), f[1], f[2]) if sc else f for f, sc in zip(feats, pts)]
        out[arm] = round(pm.loo_rprecision(jpools, with_pt, data), 4)
    return out


def delta_bytes(before, after):
    header, payload = encode_delta(before, after, keep=pr.KEEP, n_digits=pr.N_DIGITS, changed_only=True)
    fd, path = tempfile.mkstemp(suffix=".phid")
    os.close(fd)
    save_delta(path, header, payload, packed=True)
    n = os.path.getsize(path)
    os.remove(path)
    return n


def stream(M, pools, args, spans):
    """Prequential replay in date order: score each pool, then learn it. Per arm."""
    out = {}
    warm = args.warm
    pr.featurize = CACHED
    for arm, (fam, which, how) in ARMS.items():
        idx = list(range(len(M[fam]))) if which == "all" else [len(M[fam]) - 1]
        models = [load(M[fam][i]) for i in idx]
        addr_bits = [2 * (i + 1) for i in idx]
        if fam in ("zeck", "zeckq", "zeck24"):
            addr_bits = [len(M["zeck_addresses"][i]) + 1 for i in idx]   # the address digits plus the comma
        if fam in ("big", "small"):
            addr_bits = [0]

        def score(p):
            t0 = time.perf_counter()
            if how == "qroute":                      # only the routed node searches
                j = route(fam, p["task"])
                sc = pr.score_pool(models[j], p["task"], [r["text"] for r in p["rows"]])
            else:
                sc = combine(how, [pr.score_pool(m, p["task"], [r["text"] for r in p["rows"]]) for m in models], p,
                             spans, idx, fam)
            return rp(p, sc), (time.perf_counter() - t0) * 1000
        for i, m in zip(idx, models):
            torch.manual_seed(0)
            pr.finetune(m, pools[:warm], epochs=args.epochs)
            remask(fam, i, m)
        early0 = sum(score(p)[0] for p in pools[:warm]) / warm
        fresh, q, upd, msg = [], [], [], []
        for t in range(warm, len(pools)):
            r, ms = score(pools[t])
            fresh.append(r)
            q.append(ms)
            targets = ([len(models) - 1] if how == "route"              # time-routed: the newest node owns new data
                       else [route(fam, pools[t]["task"])] if how == "qroute"   # matrix-routed: the node it routes to
                       else list(range(len(models))))
            t0 = time.perf_counter()
            b = 0
            for j in targets:
                before = state(models[j])
                pr.finetune(models[j], pools[max(0, t - 4):t + 1], epochs=max(1, args.epochs // 3), seed=t)
                remask(fam, idx[j], models[j])
                b += delta_bytes(before, models[j].state_dict()) + addr_bits[j] / 8
            upd.append(time.perf_counter() - t0)
            msg.append(b)
        early1 = sum(score(p)[0] for p in pools[:warm]) / warm
        params = sum(p.numel() for m in models for p in m.parameters())
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        out[arm] = {"fresh": round(mean(fresh), 4), "forget": round(early0 - early1, 4), "update_s": round(mean(upd), 2),
                    "msg_bytes": round(mean(msg)), "query_ms": round(mean(q), 1), "params": params,
                    "mb": round(params * 2 / 1e6, 2)}
    pr.featurize = RAW
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--plugin", required=True, help="the vlds plugin root, for the deployed axis")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bits", type=int, default=15)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--minutes", type=float, default=6)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    pools = load_pools(args.pairs)
    shard_files = shards(args.p4, args.k)
    spans = shard_spans(args.p4, shard_files)
    shard_pairs = [pairs_of(args.p4, s) for s in shard_files]
    all_pairs = [x for ps, _ in shard_pairs for x in ps]
    store = next(os.path.join(args.p4, "sessions", d, "vlds") for d in sorted(os.listdir(os.path.join(args.p4, "sessions")))
                 if os.path.isdir(os.path.join(args.p4, "sessions", d, "vlds")))
    report = {"k": args.k, "pools": len(pools), "minutes": args.minutes, "epochs": args.epochs, "folds": args.folds,
              "warm": args.warm, "shard_mb": [round(b / 1e6, 1) for _p, b in shard_pairs], "runs": []}
    print(json.dumps(report), flush=True)
    for seed in args.seeds:
        t0 = time.time()
        M, times = build_models(args, shard_pairs, all_pairs, seed)
        run = {"seed": seed, "zeck_addresses": M["zeck_addresses"],
               "pretrain_s": {k: {"sequential": round(sum(v)), "parallel": round(max(v))} for k, v in times.items()}}
        run["quality"], oof = quality(M, pools, args, seed, spans)
        print(json.dumps({"seed": seed, "quality": run["quality"]}), flush=True)
        run["deployed"] = deployed(oof, pools, os.path.join(args.plugin, "scripts"), store)
        print(json.dumps({"seed": seed, "deployed": run["deployed"]}), flush=True)
        run["stream"] = stream(M, pools, args, spans)
        run["seconds"] = round(time.time() - t0)
        print(json.dumps({"seed": seed, "stream": run["stream"], "pretrain_s": run["pretrain_s"],
                          "seconds": run["seconds"]}), flush=True)
        report["runs"].append(run)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
