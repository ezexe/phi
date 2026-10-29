"""Tests for phi_rank and phi_serve: the .pt ranker trained on a store in φ-delta rounds, served live, reloaded."""

import json
import os
import subprocess
import threading
import time
import urllib.request

import pytest
import torch

import phi_rank as p4rank
from phi_serve import serve

TOPICS = ["ring buffer drain", "midi transport playhead", "toolbar latency readout", "crash dump symbols",
          "plan document decisions", "render thread priority"]


def make_store(root):
    """A git repository with text per topic, a binary capture beside it, and pairs: pool k's task names topic k."""
    sess = root / "sessions" / "s1" / "vlds"
    sess.mkdir(parents=True)
    for i, t in enumerate(TOPICS):
        (sess / f"notes-{i}.md").write_text("\n".join(f"- ruling: the {t} rule number {j}\n  detail: {t} detail {j}"
                                                      for j in range(30)), encoding="utf-8")
    (root / "scratchpad").mkdir()
    (root / "scratchpad" / "take.wav").write_bytes(b"RIFF\0\0\0\0WAVE")
    (root / "scratchpad" / "run.log").write_text("ring buffer drained in 3 ms\n", encoding="utf-8")
    (root / ".gitignore").write_text("*.wav\nmodels/.*\nmodels/*.log\n", encoding="utf-8")
    if subprocess.run(["git", "init", "-q", str(root)]).returncode != 0:
        pytest.skip("git is not available")
    pairs = root / "pairs.jsonl"
    with open(pairs, "w", encoding="utf-8") as f:
        for k, t in enumerate(TOPICS):
            for i, u in enumerate(TOPICS):
                for j in range(3):
                    f.write(json.dumps({"pool_id": f"p{k}", "task": f"a plan about the {t}", "file": f"notes-{i}.md",
                                        "text": f"the {u} rule number {j}", "label": int(i == k)}) + "\n")
    return pairs


def test_corpus_follows_git(tmp_path):
    make_store(tmp_path)
    (tmp_path / "sessions" / "s1" / "vlds" / "recall-pool.md").write_text("task: x\n## steering\n- picks", "utf-8")
    (tmp_path / "scratchpad" / "pool-picks-s1.md").write_text("picks — x", "utf-8")
    files = [os.path.relpath(p, tmp_path).replace(os.sep, "/") for p in p4rank.text_files(str(tmp_path))]
    assert "scratchpad/run.log" in files and "scratchpad/take.wav" not in files
    assert all(not f.startswith("models") for f in files)
    # the labels stay out of pretraining
    assert not any(f.endswith("recall-pool.md") or "pool-picks-" in f for f in files), files


def test_train_eval_and_live_reload(tmp_path):
    pairs = make_store(tmp_path)
    args = ["--p4", str(tmp_path), "--pairs", str(pairs), "--minutes", "0.05", "--epochs", "8", "--now", "2026-10-27"]
    assert p4rank.main(["train"] + args) == 0
    mdir = tmp_path / "models"
    side = json.loads((mdir / "ranker.json").read_text())
    assert side["round"] == 0 and len(side["pools"]) == len(TOPICS)
    model, meta = p4rank.load(str(mdir / "ranker.pt"))
    # git keeps the anchor at fp16; ranker.pt is the local fp32 build a next round continues from
    assert next(iter(torch.load(mdir / "ranker-anchor.pt", weights_only=True)["state"].values())).dtype == torch.float16
    assert next(iter(torch.load(mdir / "ranker.pt", weights_only=True)["state"].values())).dtype == torch.float32
    assert side["anchor_round"] == 0 and side["chain"] == []
    assert side["threshold_from"].startswith("held-out") and 0 <= side["heldout_rprecision"] <= 1
    oof = json.loads((mdir / "ranker-oof.json").read_text())
    assert set(oof["pools"]) == {f"p{k}" for k in range(len(TOPICS))}
    # it learned the topics: each task ranks its own rules first
    for k, t in enumerate(TOPICS):
        texts = [f"the {u} rule number 0" for u in TOPICS]
        sc = p4rank.score_pool(model, f"a plan about the {t}", texts)
        assert max(range(len(sc)), key=sc.__getitem__) == k, (t, sc)
    assert p4rank.main(["eval", "--folds", "3"] + args) == 0

    ready = threading.Event()
    holder = {}

    def run():
        holder["server"] = None
        serve(str(tmp_path), 0, idle_hours=1, ready=ready)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    assert ready.wait(30)
    info = json.loads((mdir / "ranker.server.json").read_text())

    def call(path, body=None):
        req = urllib.request.Request(f"http://127.0.0.1:{info['port']}{path}",
                                     data=json.dumps(body).encode() if body else None,
                                     headers={"Content-Type": "application/json"}, method="POST" if body else "GET")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    health = call("/health")
    assert health["ok"] and health["round"] == 0 and health["threshold"] == meta["threshold"]
    assert health["p4"] == os.path.realpath(tmp_path) and info["p4"] == os.path.realpath(tmp_path)
    # many requests at once share the embedding cache without a 400
    errors = []

    def hammer(n):
        try:
            for j in range(5):
                call("/score", {"task": f"a plan about the {TOPICS[n % 6]}",
                                "texts": [f"the {u} rule number {(n + j) % 30}" for u in TOPICS]})
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    workers = [threading.Thread(target=hammer, args=(n,)) for n in range(8)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert not errors, errors[:1]
    out = call("/score", {"task": "a plan about the toolbar latency readout",
                          "texts": ["the toolbar latency readout rule number 1", "the crash dump symbols rule number 1"]})
    assert out["round"] == 0 and out["scores"][0] > out["scores"][1]

    # a retrain swaps the checkpoint: the running server answers from round 1 at its next request, no restart
    time.sleep(1.1)
    assert p4rank.main(["train"] + args) == 0
    assert call("/health")["round"] == 1 and json.loads((mdir / "ranker.json").read_text())["round"] == 1


def test_rounds_ship_as_phi_deltas_and_rebuild(tmp_path, monkeypatch):
    """A round after the anchor ships only its change; a clone without ranker.pt rebuilds the same model from the
    chain, and the server serves that rebuild."""
    monkeypatch.setattr(p4rank, "REBASE_GAP", 1.0)        # keep the cut, whatever it costs on a toy store
    pairs = make_store(tmp_path)
    args = ["--p4", str(tmp_path), "--pairs", str(pairs), "--minutes", "0", "--epochs", "6", "--now", "2026-10-27"]
    assert p4rank.main(["train"] + args) == 0
    assert p4rank.main(["train"] + args) == 0
    mdir = tmp_path / "models"
    side = json.loads((mdir / "ranker.json").read_text())
    assert side["round"] == 1 and side["anchor_round"] == 0 and side["chain"] == ["ranker.r001.phid"]
    raw = (mdir / "ranker.r001.phid").read_bytes()
    header = json.loads(raw.split(b"\n", 1)[0])
    assert header["packed_bits"] > 0 and header["changed_only"] and header["round"] == 1 and header["parent"] == [0]
    built, meta = p4rank.load(str(mdir / "ranker.pt"))
    assert meta["chain_key"] == [0, "ranker.r001.phid"]
    (mdir / "ranker.pt").unlink()
    rebuilt, meta2 = p4rank.load_current(str(tmp_path))
    assert meta2["round"] == 1 and (mdir / "ranker.pt").exists()
    for k, v in built.state_dict().items():
        assert torch.equal(v, rebuilt.state_dict()[k]), k
    # the threshold folds were reused (the pools did not grow): no second refold
    assert side["threshold_pools"] == len(TOPICS)


def test_legacy_checkpoint_becomes_the_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr(p4rank, "REBASE_GAP", 1.0)
    pairs = make_store(tmp_path)
    args = ["--p4", str(tmp_path), "--pairs", str(pairs), "--minutes", "0", "--epochs", "4", "--now", "2026-10-27"]
    assert p4rank.main(["train"] + args) == 0
    mdir = tmp_path / "models"
    # a store from before the chain: a whole fp16 ranker.pt and a sidecar with no anchor
    (mdir / "ranker-anchor.pt").replace(mdir / "ranker.pt")
    side = json.loads((mdir / "ranker.json").read_text())
    for k in ("anchor_round", "chain", "anchor_base", "heldout_rprecision"):
        side.pop(k, None)
    (mdir / "ranker.json").write_text(json.dumps(side))
    assert p4rank.main(["train"] + args) == 0
    side = json.loads((mdir / "ranker.json").read_text())
    assert (mdir / "ranker-anchor.pt").exists() and side["anchor_round"] == 0 and side["chain"] == ["ranker.r001.phid"]


def test_bad_request(tmp_path):
    pairs = make_store(tmp_path)
    p4rank.main(["train", "--p4", str(tmp_path), "--pairs", str(pairs), "--minutes", "0", "--epochs", "1"])
    ready = threading.Event()
    threading.Thread(target=lambda: serve(str(tmp_path), 0, 1, ready), daemon=True).start()
    assert ready.wait(30)
    port = json.loads((tmp_path / "models" / "ranker.server.json").read_text())["port"]
    req = urllib.request.Request(f"http://127.0.0.1:{port}/score", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 400
    req = urllib.request.Request(f"http://127.0.0.1:{port}/score", data=b"x", method="POST",
                                 headers={"Content-Length": str(64 << 20)})
    with pytest.raises((urllib.error.HTTPError, ConnectionError, urllib.error.URLError)) as e:
        urllib.request.urlopen(req, timeout=10)
    if isinstance(e.value, urllib.error.HTTPError):
        assert e.value.code == 413


def test_shared_base_seeds_a_store_without_one(tmp_path):
    """`phi_rank shared` pretrains on a marketplace's public text (legacy/ left out); a store with no base of its
    own starts from it and continues on its own text; a git LFS pointer in its place is ignored."""
    market = tmp_path / "market"
    (market / "lbiza" / "plugins" / "vlds").mkdir(parents=True)
    (market / "lbiza" / "plugins" / "legacy").mkdir(parents=True)
    (market / "lbiza" / "plugins" / "vlds" / "operator.md").write_text(
        "\n".join(f"the operator pours the cold entry {j} through the barrier" for j in range(200)), encoding="utf-8")
    (market / "lbiza" / "plugins" / "legacy" / "old.md").write_text("a retired contract\n" * 50, encoding="utf-8")
    if subprocess.run(["git", "init", "-q", str(market)]).returncode != 0:
        pytest.skip("git is not available")
    assert p4rank.text_files(str(market), p4rank.DEFAULT_SHARED_EXCLUDE) == [
        str(market / "lbiza" / "plugins" / "vlds" / "operator.md")]
    out = tmp_path / "p4-base.pt"
    assert p4rank.main(["shared", "--corpus", str(market), "--out", str(out), "--minutes", "0.02", "--buckets", "4096",
                        "--dim", "16"]) == 0
    _m, smeta = p4rank.load(str(out))
    assert smeta["stage"] == "shared" and smeta["exclude"] == list(p4rank.DEFAULT_SHARED_EXCLUDE)

    store = tmp_path / "store"
    pairs = make_store(store)
    _base, meta = p4rank.base_model(str(store), 0.02, "2026-10-27", shared=str(out))
    assert meta["from_shared"]["steps"] == smeta["steps"] and (meta["nb"], meta["dim"]) == (4096, 16)
    assert meta["steps"] >= smeta["steps"]
    # a clone without git-lfs holds a pointer: no shared base, the store starts from nothing as before
    pointer = tmp_path / "pointer.pt"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n")
    other = tmp_path / "other"
    make_store(other)
    _b, meta2 = p4rank.base_model(str(other), 0, "2026-10-27", shared=str(pointer))
    assert "from_shared" not in meta2 and meta2["nb"] == p4rank.NB


def test_node_tree_answers_by_effort(tmp_path):
    """phi_rank.tree builds K nodes at the first K Zeckendorf addresses, each pruned by phi_mask; phi_serve answers
    medium from the one node the mask routes to, high from every node, and no effort from the single ranker."""
    import phi_rank.tree as tree
    assert tree.addresses(4) == ["1", "101", "1001", "10001"]          # lengths 1, 3, 4, 5 then 10101
    assert tree.addresses(5) == ["1", "101", "1001", "10001", "10101"]
    assert [tree.parent(a) for a in ("1", "101", "1001", "10101")] == [None, "1", "1", "101"]
    pairs = make_store(tmp_path)
    pools = p4rank.read_pools(str(pairs))
    meta = tree.build(str(tmp_path), pools, k=4, minutes=0.02, epochs=4, bits=13, dim=16, now="2026-10-28",
                      log=lambda *_: None)
    mdir = tmp_path / "models"
    assert [n["address"] for n in meta["nodes"]][-1] == "1"              # the root trains last
    for n in meta["nodes"]:
        m, _ = p4rank.load(str(mdir / "tree" / f"{n['address']}.pt"))
        kept = m.bag.weight.norm(dim=1) > 0
        assert not bool((kept[1:] & kept[:-1]).any()), n["address"]     # no two adjacent rows kept
    t = tree.Tree(str(tmp_path))
    assert 0 <= t.route("a plan about the ring buffer drain") < 4
    # the JSON ranker's .pt weight for the tree is fitted on these: every pool, each effort, out of fold
    oof = json.loads((mdir / tree.TREE_OOF).read_text())
    assert set(oof["efforts"]) == {"medium", "high"} and set(oof["heldout"]) == {"medium", "high"}
    for effort in ("medium", "high"):
        assert set(oof["efforts"][effort]) == {p["id"] for p in pools}
        for p in pools:
            assert set(oof["efforts"][effort][p["id"]]) == {p4rank.text_key(r["text"]) for r in p["rows"]}
    assert meta["heldout"] == oof["heldout"]
    assert p4rank.main(["train", "--p4", str(tmp_path), "--pairs", str(pairs), "--minutes", "0", "--epochs", "2"]) == 0
    ready = threading.Event()
    threading.Thread(target=lambda: serve(str(tmp_path), 0, 1, ready), daemon=True).start()
    assert ready.wait(30)
    port = json.loads((mdir / "ranker.server.json").read_text())["port"]

    def score(effort):
        body = {"task": "a plan about the toolbar latency readout", "texts": ["the toolbar latency readout rule number 1",
                "the crash dump symbols rule number 1"]}
        if effort:
            body["effort"] = effort
        req = urllib.request.Request(f"http://127.0.0.1:{port}/score", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    assert score("medium")["source"].startswith("tree:medium:")
    assert score("high")["source"] == "tree:high:4" and len(score("high")["scores"]) == 2
    assert score("")["source"] == "ranker" and score("low")["source"] == "ranker"


def test_training_device_and_feature_cache(monkeypatch):
    """Training runs on device() (PHI_DEVICE, else CUDA when present) and leaves the model on the CPU, where its
    checkpoints and deltas are made; the feature cache gives the same features as hashing each time."""
    import numpy as np
    monkeypatch.setenv("PHI_DEVICE", "cpu")
    assert p4rank.device().type == "cpu"
    texts = [f"the ring buffer drain order rule {i} and the toolbar latency {i % 3}" for i in range(9)]
    plain = p4rank.batch_features(texts, 1 << 13)
    cache = p4rank.FeatureCache(1 << 13)
    for _ in range(2):                                   # filled, then read back
        got = p4rank.batch_features(texts, 1 << 13, cache)
        assert torch.equal(plain[0], got[0]) and torch.equal(plain[1], got[1])
    assert cache.dtype == np.uint16 and len(cache.got) == len(texts)
    full = p4rank.FeatureCache(1 << 13, mb=0)            # no room: still right, just not kept
    assert torch.equal(p4rank.batch_features(texts, 1 << 13, full)[0], plain[0]) and not full.got
    m = p4rank.Ranker(1 << 13, 16)
    assert p4rank.pretrain(m, list(zip(texts, texts[1:])), 0.005, log=lambda *_: None) > 0
    pools = [{"task": "a plan about the ring buffer", "rows": [{"text": t, "label": int(i < 2)} for i, t in enumerate(texts)]}]
    p4rank.finetune(m, pools, epochs=1)
    assert all(p.device.type == "cpu" for p in m.parameters())


def test_pretrain_early_stopping():
    """With a select function the pretraining keeps the best-scoring weights it saw, and reports the curve."""
    texts = [f"the ring buffer drain order rule {i} and the toolbar latency {i % 3}" for i in range(12)]
    m = p4rank.Ranker(1 << 12, 16)
    calls = []

    def select(model):                                   # best at the first check: the untrained weights
        calls.append(1)
        return 1.0 if len(calls) == 1 else 0.0
    start = {k: v.clone() for k, v in m.state_dict().items()}
    info = {}
    steps = p4rank.pretrain(m, list(zip(texts, texts[1:])), 0.01, log=lambda *_: None, select=select,
                            select_every=0.1, info=info)
    assert steps > 0 and info["best_steps"] == 0 and info["best_score"] == 1.0 and len(info["curve"]) >= 2
    assert all(torch.equal(start[k], v) for k, v in m.state_dict().items())        # the kept weights are the best
    assert all(p.device.type == "cpu" for p in m.parameters())
    m2 = p4rank.Ranker(1 << 12, 16)
    assert p4rank.pretrain(m2, list(zip(texts, texts[1:])), 0.005, log=lambda *_: None) > 0   # no select: unchanged
