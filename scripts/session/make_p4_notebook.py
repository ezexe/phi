"""Writes .docs/p4/continual_p4.ipynb: small models trained in rounds on a .claude/p4 store, each round shipped as a Fibonacci-coded sparse delta."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / ".docs" / "p4" / "continual_p4.ipynb"

cells = [
    ("markdown", """# Continual training on a `.claude/p4` store (sketch)

A small character-level model trained in rounds on what a p4 store holds (sessions' vlds stores, frags, flights, memory), with each round's update stored in the store as a sparse delta.

Where the φ work comes in, and where it doesn't, follows `.docs/register.md` section 6:

| Stage | Number system | Why |
|---|---|---|
| Corpus, sharding, rounds | plain sorted keys and times | vlds retired its φ-register placement for a keyspace: the Fibonacci capacities made debt no judgment had |
| Training, optimizer, activations | binary (fp32 / bf16) | the φ accumulator loses to self-timed binary (0.82×) and fires more rules per sum |
| Pruning | none (2:4 if needed) | φ masks lose to 2:4 at 50% on transformers and tie at 25% and 75% |
| **Delta positions** | **Fibonacci gap codes** (`phi_prune.gaps`) | beat Elias-gamma, Elias-delta and fixed-width at 1–10% kept, no parameter (K8) |
| **Delta values, 4–6 bits** | **Zeckendorf levels** (`phi_prune.delta`) | 7–10% fewer 1 digits than CSD, ~16% fewer than binary; gone past ~7 bits, and only shift-and-add hardware cashes it |
| Stored deltas | self-delimiting, checked on load | every code ends in `11`; counts, positions and shapes are checked before a weight is written |

This is a sketch: the model is the 0.8M-parameter GPT of `scripts/transformer_check.py`, and the numbers it prints are a pipeline check until the steps are raised and the run is repeated over seeds."""),
    ("code", """# Colab: !pip -q install "phi-prune @ git+https://github.com/ezexe/phi@main"
# and clone the store's repository, then point P4 at its .claude/p4 (or the repo root of a store repo like AudioPlugin-p4).
import math, sys, time, copy
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

P4 = Path("../../../AudioPlugin-p4")    # the store
OUT = P4 / "models"                     # where the rounds' deltas land, beside the sessions that made them
CUTOFF = "2026-10-26 00:00"             # records from here on are held out
ROUNDS = ["2026-10-01 00:00", "2026-10-15 00:00", CUTOFF]  # round r trains on records timed before ROUNDS[r]
STEPS_BASE, STEPS_ROUND = 300, 100      # raise for a real run
KEEP, N_DIGITS = 0.01, 6                # 1% of each update kept, 6 Fibonacci digits (20 levels a side)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, ".")                 # p4_corpus.py sits beside this notebook
from p4_corpus import read_p4, split
from phi_prune.delta import encode_delta, apply_delta, delta_stats, save_delta, load_delta
from phi_prune.gaps import gap_bits, pooled_gaps"""),
    ("markdown", """## 1. The store as records

Pretrain records are the store's own text; pair records are one per flight, the ask and plan as the prompt and the close note and diff as the target, landed flights weighed above the rest; mind-reader scores weigh each session's records."""),
    ("code", """records = list(read_p4(P4))
train_recs, held = split(records, CUTOFF)
print(f"{len(records)} records, {len(train_recs)} train, {len(held)} held out after {CUTOFF}")
print(f"untimed (always train): {sum(not r.time for r in records)}")

CHARS = [chr(c) for c in range(32, 127)] + ["\\n", "\\t"]
STOI = {c: i for i, c in enumerate(CHARS)}
ASK, DONE = "\\n### ask\\n", "\\n### done\\n"

def as_text(r):
    return ASK + r.text + DONE + r.target if r.kind == "pair" else r.text

def encode(text):
    return torch.tensor([STOI[c] for c in text if c in STOI], dtype=torch.long)

def corpus(recs):
    \"\"\"Concatenated encodings, and a sampling weight per character from each record's weight.\"\"\"
    parts = [(encode(as_text(r) + "\\n\\n"), r.weight) for r in recs]
    data = torch.cat([p for p, _ in parts])
    w = torch.cat([torch.full((len(p),), wt) for p, wt in parts])
    return data, w

held_data, _ = corpus(held)
print(f"held-out characters: {len(held_data):,}")"""),
    ("markdown", """## 2. The model and its training

The GPT of `scripts/transformer_check.py`; batches start at positions drawn by record weight."""),
    ("code", """CTX, D, LAYERS, HEADS, BATCH, LR = 128, 128, 4, 4, 32, 3e-3

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.fc1, self.fc2 = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)
    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))

class GPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, D), nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, vocab)
    def forward(self, idx):
        T = idx.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), 1)
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))

def batch(data, starts_w, g):
    i = torch.multinomial(starts_w, BATCH, replacement=True, generator=g)
    x = torch.stack([data[j:j + CTX] for j in i])
    y = torch.stack([data[j + 1:j + CTX + 1] for j in i])
    return x.to(DEVICE), y.to(DEVICE)

def train(model, data, w, steps, seed, lr=LR):
    g = torch.Generator().manual_seed(seed)
    starts_w = w[: len(data) - CTX - 1]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    model.train()
    for _ in range(steps):
        x, y = batch(data, starts_w, g)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

@torch.no_grad()
def bpc(model, data, batches=20):
    \"\"\"Bits per character on contiguous windows of data.\"\"\"
    model.eval()
    n = min(batches * BATCH, (len(data) - 1) // CTX)
    x = torch.stack([data[k * CTX:(k + 1) * CTX] for k in range(n)]).to(DEVICE)
    y = torch.stack([data[k * CTX + 1:(k + 1) * CTX + 1] for k in range(n)]).to(DEVICE)
    return F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item() / math.log(2)"""),
    ("markdown", """## 3. Rounds

Round 0 trains a base on everything before `ROUNDS[0]` and saves it whole. Each later round continues from the shipped model on everything before its own time, then ships only its delta: the top `KEEP` of each tensor's update, positions in Fibonacci gap codes, values on the Zeckendorf grid. The next round starts from the base plus the deltas so far, which is what a reader of the store can rebuild, not from the unshipped trained model."""),
    ("code", """OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
model = GPT(len(CHARS)).to(DEVICE)
data, w = corpus([r for r in train_recs if not r.time or r.time < ROUNDS[0]])
train(model, data, w, STEPS_BASE, seed=0)
torch.save(model.state_dict(), OUT / "round-00.pt")
print(f"round 0: {len(data):,} chars, held-out bpc {bpc(model, held_data):.4f}")

log = []
for r, until in enumerate(ROUNDS[1:], start=1):
    t0 = time.time()
    shipped = {k: v.detach().clone() for k, v in model.state_dict().items()}
    data, w = corpus([x for x in train_recs if not x.time or x.time < until])
    tuned = copy.deepcopy(model)
    train(tuned, data, w, STEPS_ROUND, seed=r, lr=LR / 3)

    header, payload = encode_delta(shipped, tuned.state_dict(), keep=KEEP, n_digits=N_DIGITS)
    path = save_delta(str(OUT / f"round-{r:02d}.phid"), header, payload)
    state = {k: v.clone() for k, v in shipped.items()}
    apply_delta(state, *load_delta(path))           # what a reader of the store rebuilds
    model.load_state_dict(state)

    deltas = [tuned.state_dict()[t["name"]] - shipped[t["name"]] for t in header["tensors"]]
    gaps, n = pooled_gaps(deltas, KEEP)
    codes = gap_bits(gaps, n)
    row = {"round": r, "chars": len(data), "bpc tuned": bpc(tuned, held_data), "bpc shipped": bpc(model, held_data),
           **{f"pos bits {k}": codes[k] for k in ("fibonacci", "elias-gamma", "fixed-width")},
           "bits/kept": delta_stats(header, payload)["bits_per_kept"], "s": time.time() - t0}
    log.append(row)
    print(", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))"""),
    ("markdown", """## 4. What to read off

- **bpc tuned vs bpc shipped**: what keeping only `KEEP` of each update costs. If shipped trails tuned by more than the round gains, raise `KEEP`, or train the round sparsely so the update is sparse to begin with.
- **pos bits**: Fibonacci gap codes against Elias-gamma and fixed-width positions on this store's own updates, the K8 check on real fine-tuning data rather than shifted standard-library files (the open item in `.docs/register.md` section 6).
- **bits/kept**: positions plus a sign digit and a Fibonacci-coded magnitude. The payload is `0`/`1` text, the layout `export_bitstream` uses; packing it 8 to a byte is a file-format change, not a coding one.
- Rebuilding any round: `round-00.pt`, then `apply_delta` of each `round-NN.phid` in order. A truncated or mismatched file raises before writing a weight."""),
]

nb = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": src.splitlines(keepends=True),
         **({"outputs": [], "execution_count": None} if t == "code" else {})}
        for t, src in cells
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1) + "\n")
print(OUT)
