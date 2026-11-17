"""Tests for phi_rank: the dual encoder trained on a store in rounds of incremental checkpoints."""

import json
import os
import subprocess

import pytest
import torch

torch.set_num_threads(1)

import phi_rank
from test_p4rank_labels import TOPICS, labelled_store


def make_store(root):
    """A git repository holding the labelled store of test_p4rank_labels, query k naming topic k, with text per
    topic and a binary capture beside it."""
    labelled_store(root)
    notes = root / "notes"
    notes.mkdir()
    for i, t in enumerate(TOPICS):
        (notes / f"notes-{i}.md").write_text("\n".join(f"- ruling: the {t} rule number {j}\n  detail: {t} detail {j}"
                                                       for j in range(30)), encoding="utf-8")
    (root / "scratchpad").mkdir()
    (root / "scratchpad" / "take.wav").write_bytes(b"RIFF\0\0\0\0WAVE")
    (root / "scratchpad" / "run.log").write_text("ring buffer drained in 3 ms\n", encoding="utf-8")
    (root / ".gitignore").write_text("*.wav\nmodels/.*\nmodels/*.log\n", encoding="utf-8")
    if subprocess.run(["git", "init", "-q", str(root)]).returncode != 0:
        pytest.skip("git is not available")


def test_corpus_follows_git(tmp_path):
    make_store(tmp_path)
    files = [os.path.relpath(p, tmp_path).replace(os.sep, "/") for p in phi_rank.text_files(str(tmp_path))]
    assert "scratchpad/run.log" in files and "scratchpad/take.wav" not in files
    assert all(not f.startswith("models") for f in files)


def test_corpus_of_a_store_its_repository_ignores(tmp_path):
    """A store inside a project repository that ignores .claude/p4 is read by walking it, not left empty."""
    if subprocess.run(["git", "init", "-q", str(tmp_path)]).returncode != 0:
        pytest.skip("git is not available")
    (tmp_path / ".gitignore").write_text(".claude/p4/\n", encoding="utf-8")
    store = tmp_path / ".claude" / "p4" / "phi" / "store"
    store.mkdir(parents=True)
    (store / "rules.md").write_text("- ruling: keep the ring buffer drained\n", encoding="utf-8")
    p4 = tmp_path / ".claude" / "p4"
    # the store's own .gitignore, as the plugin seeds it, still decides: temportal's tmp and the session pages stay out
    (p4 / ".gitignore").write_text("temportal/tmp/\nphi/sessions/*/session.md\n", encoding="utf-8")
    for rel in ("temportal/tmp/x/probe.txt", "phi/sessions/s1/session.md"):
        (p4 / rel).parent.mkdir(parents=True, exist_ok=True)
        (p4 / rel).write_text("a scratch note no model should learn from\n", encoding="utf-8")
    files = sorted(os.path.relpath(p, p4).replace(os.sep, "/") for p in phi_rank.text_files(str(p4)))
    assert files == [".gitignore", "phi/store/rules.md"]


def test_train_and_eval(tmp_path):
    make_store(tmp_path)
    args = ["--p4", str(tmp_path), "--minutes", "0", "--epochs", "4", "--now", "2026-10-27"]
    assert phi_rank.main(["train"] + args) == 0
    mdir = tmp_path / "models"
    side = json.loads((mdir / "ranker-pt.json").read_text())
    assert side["round"] == 0 and len(side["pools"]) == len(TOPICS)
    model, meta = phi_rank.load(str(mdir / "ranker.pt"))
    # git keeps the full checkpoint at fp16; ranker.pt is the local fp32 build a next round continues from
    assert next(iter(torch.load(mdir / "ranker-anchor.pt", weights_only=True)["state"].values())).dtype == torch.float16
    assert next(iter(torch.load(mdir / "ranker.pt", weights_only=True)["state"].values())).dtype == torch.float32
    assert side["anchor_round"] == 0 and side["chain"] == []
    assert side["threshold_from"].startswith("held-out") and 0 <= side["heldout_rprecision"] <= 1
    # it learned the query groups: each query ranks its labelled rules above its group's other candidates
    assert phi_rank.mean_rprecision(model, phi_rank.read_label_pools(str(tmp_path))) >= 0.8
    assert phi_rank.main(["eval", "--folds", "3"] + args + ["--epochs", "1"]) == 0



def test_rounds_ship_incremental_checkpoints_and_rebuild(tmp_path, monkeypatch):
    """A round after the full checkpoint ships only its change; a clone without ranker.pt rebuilds the same model
    from the checkpoints."""
    monkeypatch.setattr(phi_rank, "REBASE_GAP", 1.0)        # keep the cut, whatever it costs on a toy store
    monkeypatch.setattr(phi_rank, "CONTINUE_EPOCHS", 2)
    make_store(tmp_path)
    args = ["--p4", str(tmp_path), "--minutes", "0", "--epochs", "2", "--now", "2026-10-27", "--buckets", "4096",
            "--dim", "16"]
    assert phi_rank.main(["train"] + args) == 0
    assert phi_rank.main(["train"] + args) == 0
    mdir = tmp_path / "models"
    side = json.loads((mdir / "ranker-pt.json").read_text())
    assert side["round"] == 1 and side["anchor_round"] == 0 and side["chain"] == ["ranker.r001.phid"]
    raw = (mdir / "ranker.r001.phid").read_bytes()
    assert raw.startswith(b"PHID1 ")
    header = json.loads(raw.split(b"\n", 2)[1])
    assert header["packed_bits"] > 0 and header["changed_only"] and header["round"] == 1 and header["parent"] == [0]
    built, meta = phi_rank.load(str(mdir / "ranker.pt"))
    assert meta["chain_key"] == [0, "ranker.r001.phid"]
    (mdir / "ranker.pt").unlink()
    rebuilt, meta2 = phi_rank.load_current(str(tmp_path))
    assert meta2["round"] == 1 and (mdir / "ranker.pt").exists()
    for k, v in built.state_dict().items():
        assert torch.equal(v, rebuilt.state_dict()[k]), k
    # the threshold's folds were reused, since the query groups did not grow: no second refold
    assert side["threshold_pools"] == len(TOPICS)


def test_legacy_checkpoint_becomes_the_full_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(phi_rank, "REBASE_GAP", 1.0)
    monkeypatch.setattr(phi_rank, "CONTINUE_EPOCHS", 1)
    make_store(tmp_path)
    args = ["--p4", str(tmp_path), "--minutes", "0", "--epochs", "1", "--now", "2026-10-27", "--buckets", "4096",
            "--dim", "16"]
    assert phi_rank.main(["train"] + args) == 0
    mdir = tmp_path / "models"
    # a store from before the incremental checkpoints: a whole fp16 ranker.pt and a sidecar with no full checkpoint
    (mdir / "ranker-anchor.pt").replace(mdir / "ranker.pt")
    side = json.loads((mdir / "ranker-pt.json").read_text())
    for k in ("anchor_round", "chain", "anchor_base", "heldout_rprecision"):
        side.pop(k, None)
    (mdir / "ranker-pt.json").write_text(json.dumps(side))
    assert phi_rank.main(["train"] + args) == 0
    side = json.loads((mdir / "ranker-pt.json").read_text())
    assert (mdir / "ranker-anchor.pt").exists() and side["anchor_round"] == 0 and side["chain"] == ["ranker.r001.phid"]


def test_ensemble_defaults_are_the_measured_best(tmp_path):
    """By default the ensemble's members keep every row and skip pretraining, since phi_depth measured both to cost
    held-out R-precision."""
    import phi_rank.tree as tree
    make_store(tmp_path)
    pools = phi_rank.read_label_pools(str(tmp_path))
    meta = tree.build(str(tmp_path), pools, k=2, epochs=2, bits=13, dim=16, now="2026-11-09", log=lambda *_: None)
    assert meta["masked"] is False and meta["minutes"] == 0
    assert all(n["steps"] == 0 for n in meta["nodes"]), "no pretraining steps"
    for n in meta["nodes"]:
        m, saved = phi_rank.load(str(tmp_path / "models" / "tree" / f"{n['address']}.pt"))
        assert saved["masked"] is False
        rows = m.bag.weight.norm(dim=1) > 0
        assert bool((rows[1:] & rows[:-1]).any()), "unmasked: adjacent rows are kept"


def test_ensemble_members_mask_and_selection(tmp_path):
    """phi_rank.tree builds K members at the first K Zeckendorf addresses, each pruned by phi_mask when asked, and
    the selection chooses one member for a query."""
    import phi_rank.tree as tree
    assert tree.addresses(4) == ["1", "101", "1001", "10001"]          # lengths 1, 3, 4, 5 then 10101
    assert tree.addresses(5) == ["1", "101", "1001", "10001", "10101"]
    assert [tree.parent(a) for a in ("1", "101", "1001", "10101")] == [None, "1", "1", "101"]
    make_store(tmp_path)
    pools = phi_rank.read_label_pools(str(tmp_path))
    meta = tree.build(str(tmp_path), pools, k=4, minutes=0.002, epochs=1, bits=13, dim=16, now="2026-10-28",
                      log=lambda *_: None, mask=True)
    mdir = tmp_path / "models"
    assert [n["address"] for n in meta["nodes"]][-1] == "1"              # the root trains last
    assert meta["masked"] is True
    for n in meta["nodes"]:
        m, _ = phi_rank.load(str(mdir / "tree" / f"{n['address']}.pt"))
        kept = m.bag.weight.norm(dim=1) > 0
        assert not bool((kept[1:] & kept[:-1]).any()), n["address"]     # no two adjacent rows kept
    models = [phi_rank.load(str(mdir / "tree" / f"{n['address']}.pt"))[0] for n in meta["nodes"]]
    assert 0 <= tree.route(tree.node_rows(models), "a plan about the ring buffer drain") < 4


def test_training_device_and_feature_cache(monkeypatch):
    """Training runs on device(), which is PHI_DEVICE or else CUDA when present, and leaves the model on the CPU,
    where its checkpoints and deltas are made; the feature cache gives the same features as hashing each time."""
    import numpy as np
    monkeypatch.setenv("PHI_DEVICE", "cpu")
    assert phi_rank.device().type == "cpu"
    texts = [f"the ring buffer drain order rule {i} and the toolbar latency {i % 3}" for i in range(9)]
    plain = phi_rank.batch_features(texts, 1 << 13)
    cache = phi_rank.FeatureCache(1 << 13)
    for _ in range(2):                                   # filled, then read back
        got = phi_rank.batch_features(texts, 1 << 13, cache)
        assert torch.equal(plain[0], got[0]) and torch.equal(plain[1], got[1])
    assert cache.dtype == np.uint16 and len(cache.got) == len(texts)
    full = phi_rank.FeatureCache(1 << 13, mb=0)            # no room: still right, just not kept
    assert torch.equal(phi_rank.batch_features(texts, 1 << 13, full)[0], plain[0]) and not full.got
    m = phi_rank.Ranker(1 << 13, 16)
    assert phi_rank.pretrain(m, list(zip(texts, texts[1:])), 0.0005, log=lambda *_: None) > 0
    pools = [{"task": "a plan about the ring buffer", "rows": [{"text": t, "label": int(i < 2)} for i, t in enumerate(texts)]}]
    phi_rank.finetune(m, pools, epochs=1)
    assert all(p.device.type == "cpu" for p in m.parameters())


def test_pretrain_early_stopping():
    """With a select function the pretraining keeps the best-scoring weights it saw, and reports the curve."""
    texts = [f"the ring buffer drain order rule {i} and the toolbar latency {i % 3}" for i in range(12)]
    m = phi_rank.Ranker(1 << 12, 16)
    calls = []

    def select(model):                                   # best at the first check: the untrained weights
        calls.append(1)
        return 1.0 if len(calls) == 1 else 0.0
    start = {k: v.clone() for k, v in m.state_dict().items()}
    info = {}
    steps = phi_rank.pretrain(m, list(zip(texts, texts[1:])), 0.001, log=lambda *_: None, select=select,
                            select_every=0.01, info=info)
    assert steps > 0 and info["best_steps"] == 0 and info["best_score"] == 1.0 and len(info["curve"]) >= 2
    assert all(torch.equal(start[k], v) for k, v in m.state_dict().items())        # the kept weights are the best
    assert all(p.device.type == "cpu" for p in m.parameters())
    m2 = phi_rank.Ranker(1 << 12, 16)
    assert phi_rank.pretrain(m2, list(zip(texts, texts[1:])), 0.0005, log=lambda *_: None) > 0   # no select: unchanged


def test_the_base_reads_the_text_once_to_continue_and_never_to_keep_itself(tmp_path, monkeypatch):
    """base_model keeps its base while the text has not grown `repretrain` times and continues when it has, as it
    did, and decides from the files' sizes: it cuts the files into passages only to continue, once each."""
    make_store(tmp_path)
    monkeypatch.setattr(phi_rank, "pretrain", lambda model, pairs, minutes, **kw: phi_rank.MIN_PRETRAIN_STEPS)
    cut = []
    real = phi_rank.passages
    monkeypatch.setattr(phi_rank, "passages", lambda path: cut.append(path) or real(path))
    p4, now = str(tmp_path), "2026-10-27"
    _, built = phi_rank.base_model(p4, 0.01, now, nb=4096, dim=16)
    assert len(cut) == len(phi_rank.text_files(p4)) and "continued_from_steps" not in built
    cut.clear()
    grown = tmp_path / "notes" / "grown.md"
    grown.write_bytes(b"x" * int(built["corpus_bytes"] * 0.2))           # grown by a fifth: the base is kept
    _, kept = phi_rank.base_model(p4, 0.01, now, nb=4096, dim=16)
    assert kept == built and cut == []
    grown.write_bytes(b"x" * int(built["corpus_bytes"] * 0.3))           # by three tenths: it continues
    _, continued = phi_rank.base_model(p4, 0.01, now, nb=4096, dim=16)
    assert continued["continued_from_steps"] == built["steps"] and continued["corpus_bytes"] > built["corpus_bytes"]
    assert sorted(cut) == sorted(phi_rank.text_files(p4))
