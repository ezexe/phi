# The nested p4 store: what of phi it shows, measured

A p4 session started with a store as its working directory grows a store inside the store. The vlds hooks follow their one rule, that a store lives at `<working dir>/.claude/p4`, so they produce:

```
E:\gritz\AudioPlugin\.claude\p4\.claude\p4\
```

Nothing asked for this, and a session started in the inner store grows the next level the same way. The owner's thesis is that this is phi's structure appearing unasked in a directory tree, and that binary cannot express it.

This note sets the claim out in phi's terms and tests each part on a fixture grown the way it happened. Each claim below is followed by its measurement, its limits and a verdict. The p4 plugins and their rules were not changed.

## The fixture

`scripts/session/nested_store.py` grows the nesting with the plugins' own hooks. Nothing is copied. At each level it starts a session in the previous level's store:

- the vlds and src-fragger `SessionStart` hooks
- one prompt (`prompt-open`)
- the turn's close (`turn-close`)

Each session therefore writes a real store: `sessions/<id>/vlds/` (dispatch record, keyspace, index, briefs), `models/`, `vindex/` and the src-fragger seeds. mind-reader and p4-memory were left out because they spawn a background agent and link the home directory.

```
python scripts/session/nested_store.py --plugins <p4 checkout>/lbiza/plugins --depth 8                      # run A
python scripts/session/nested_store.py --plugins <p4 checkout>/lbiza/plugins --depth 8 --sessions-outer 3  # run B
python scripts/session/nested_store.py --plugins <p4 checkout>/lbiza/plugins --zeckendorf 8                 # run Z
```

| Run | What grows | Sessions | Bytes per store (own content) |
|---|---|---|---|
| A | a chain 8 deep: each session starts in the previous store | 1 per level | 9,047 B in 16 files |
| B | the same chain | 3 per level above the deepest, 1 at it | 22,605 B in 32 files (outer); 9,047 B (deepest) |
| Z | every Zeckendorf address up to 8 digits. Each session starts either in a store (giving `.claude/p4`) or in a `.claude` directory (giving `.claude/.claude/p4`) | 1 per store, 21 stores | 9,022–9,029 B (each prompt names its own address) |

Measured 2026-10-27, vlds 0.0.72.

## The mapping, in phi's terms

Read a store node's path from the outermost node as digits: `p4` is 1 and `.claude` is 0.

```
p4                         → 1
p4/.claude/p4              → 101
p4/.claude/p4/.claude/p4   → 10101
```

| Claim | phi's term | Verdict |
|---|---|---|
| No two store nodes are adjacent; every path has no `11` | Zeckendorf adjacency (`phi_mask`, `cassini_check`'s `11` scan) | **Holds, exactly as Zeckendorf.** The unchanged hooks grow every Zeckendorf address and never a `p4/p4`. See 1. |
| A path is a Zeckendorf number | Zeckendorf digits | **Exact.** Every address is a distinct integer, those whose Zeckendorf form ends in 1. See 1. |
| The paths are self-delimiting with no length headers | Fibonacci codes (`phi_prune.gaps`) | **Holds, and the comma is the tree's forbidden pair.** One more `p4` (`11`) ends an address. The digits alone are refused. See 2. |
| Binary needs a width or a length header and costs more | the README's 9.7–10.3 vs 7.2 bits | **Mixed.** Binary with a length field costs more; a binary or Fibonacci code of the depth costs less than half. See 3. |
| Each node splits as `M` = [[1,1],[1,0]] steps (a, b) → (a+b, a) | `M`, F(n−1) : F(n−2) | **Holds, for the store counts of the Zeckendorf tree.** In a single chain (runs A and B) only the first row holds. See 4. |
| lvalue : rvalue approaches φ as the tree grows | φ | **Holds for the Zeckendorf tree's store counts** (exact Fibonacci numbers, ratio → 1.618034). **Fails in a chain**, where it is linear in depth. Content bytes follow the counts only while each store holds about the same. See 4. |
| A binary split cuts one half off; this keeps both | — | **Analogy only.** A binary tree keeps both children too. See 4. |

## 1. The digits: Zeckendorf, grown by the unchanged hooks

The hooks have one rule: a session's store is `<working dir>/.claude/p4`.

- **A session started in a store** grows a `.claude/p4` under it: the digits `01`.
- **A session started in a store's `.claude` directory** grows a `.claude/.claude/p4`: the digits `001`. One more `.claude` gives `0001`, and so on.
- **Nothing ever creates a `p4` directly inside a `p4`.** Every store the hooks make is preceded by a `.claude`.

So the hooks forbid exactly `11` and allow any run of `0`s, which is the Zeckendorf constraint, with every address beginning and ending at a store (`1`).

**Runs A and B** only started sessions in stores, so they grew one address per length: `1`, `101`, `10101`, … (`(10)*1`). That was the fixture's choice, not the hooks'. An earlier draft of this note took it for the tree's whole language. Read as Zeckendorf digits, those chain addresses are 1 + 3 + 8 + … = F(2n+1) − 1, the alternate-sum identity, confirmed for all 8 depths.

**Run Z** started its 21 sessions wherever each Zeckendorf address of up to 8 digits needed one, with the hooks unchanged:

- **Every store landed at its intended address.** All 21 were found, and their digits match the intended addresses exactly.
- **The rule held everywhere.** No address contains `11`, and no directory anywhere in the fixture is a `p4` directly inside a `p4` (0).
- **The counts by address length are 1, 0, 1, 1, 2, 3, 5, 8**, the Fibonacci numbers. Enumerating the grammar further gives …, 13, 21, 34, 55, 89, 144 at lengths 9–14.

**Each address is a distinct integer.** Read as Zeckendorf digits, the 144 addresses of up to 12 digits are 144 different integers, with no collisions across lengths because every address starts with `1`. They are exactly the integers whose Zeckendorf form ends in a 1: 1, 4, 6, 9, 12, 14, 17, 19, 22, … (the upper Wythoff numbers minus one).

## 2. Self-delimiting: the comma is the pair the tree forbids

A stream of node paths was built with one item per file, naming the node that owns it: 128 items in run A, 240 in run B.

- **Concatenated bare, the digits do not parse.** `phi_prune.gaps.read_codes` refuses the stream: "bitstream ends inside a codeword (14 digits without a closing 1)". A path ends in `1`, and the next begins with `1`, so the `11` that marks a codeword's end lands on the boundary. It takes the next path's first digit and the stream falls out of step.
- **With one `1` appended to each path, every path becomes a Fibonacci codeword**, the code of F(2n+1) − 1. `read_codes` then parses the whole stream back with no length header, and every item comes back exactly (round trip true in both runs).

The appended `1` is a `p4` written directly after a `p4`, the one pair the hooks never produce. Because `11` cannot occur inside an address, it can mark where one ends, exactly as it ends a Fibonacci codeword. Run Z's addresses, any mix of `0` runs, parse the same way: a stream of 7 of them, written least-significant digit first with one `1` appended to each, comes back exactly from `read_codes`.

So the tree supplies the digits, and its forbidden pair supplies the comma. The comma is one bit a path, which is what Fibonacci coding adds. In the filesystem, where a store ends is read from the directory: no further `.claude/p4` means this is the last level. That is a structural ending, not a property of the string.

## 3. The bit cost against binary

The same streams, encoded five ways. Every encoding was built and decoded, not estimated.

| Encoding | Run A (bits/item) | Run B (bits/item) |
|---|---|---|
| tree code: path digits + comma (Fibonacci code of F(2n+1) − 1) | **9.00** | **8.53** |
| path digits + a 4-bit length field each (binary, headed) | 12.00 | 11.53 |
| Fibonacci code of the depth *n* | 4.25 | 4.13 |
| Elias-gamma of the depth | 4.25 | 4.07 |
| fixed 4-bit depth + a 40-bit header (width, count) | 4.31 | 4.17 |

- **Against binary that carries the same path digits**, the tree code wins: 9.0 bits against 12.0. Its comma costs 1 bit a path; a length field costs 4.
- **The tree code is unary in depth**: 2n bits for depth *n*. Any code of the depth itself, Fibonacci, Elias-gamma or fixed width, takes less than half the tree code's bits.
- **Among those three, Fibonacci and Elias-gamma tie.** Run A is 4.25 for both; in run B, Elias-gamma is 0.06 bits less.
- **Fixed width plus a header** is within 0.1 bits of both. With only 8 depths, a 40-bit header spread over 128–240 items is nearly free.

This is the README's finding again. The Fibonacci stream buys self-delimiting and the `11` check, not smaller files (9.7–10.3 bits against 7.2 there).

## 4. lvalue and rvalue by depth, against `M`

- **rvalue** is a node's own content (sessions, models, indexes, seeds), without its nested `.claude/`.
- **lvalue** is the nested `.claude/` subtree.

| Depth | rvalue A | lvalue A | l/r A | r(n) / l(n+1) A | rvalue B | lvalue B | l/r B | r(n) / l(n+1) B |
|---|---|---|---|---|---|---|---|---|
| 1 | 9,047 | 63,329 | 7.0 | 0.167 | 22,605 | 144,677 | 6.4 | 0.185 |
| 2 | 9,047 | 54,282 | 6.0 | 0.200 | 22,605 | 122,072 | 5.4 | 0.227 |
| 3 | 9,047 | 45,235 | 5.0 | 0.250 | 22,605 | 99,467 | 4.4 | 0.294 |
| 4 | 9,047 | 36,188 | 4.0 | 0.333 | 22,605 | 76,862 | 3.4 | 0.417 |
| 5 | 9,047 | 27,141 | 3.0 | 0.500 | 22,605 | 54,257 | 2.4 | 0.714 |
| 6 | 9,047 | 18,094 | 2.0 | 1.000 | 22,605 | 31,652 | 1.4 | 2.499 |
| 7 | 9,047 | 9,047 | 1.0 | — | 22,605 | 9,047 | 0.4 | — |
| 8 | 9,047 | 0 | 0.0 | — | 9,047 | 0 | 0.0 | — |

The `M` step (a, b) → (a + b, a) asks for two things.

- **Row 1: lvalue(n) = lvalue(n+1) + rvalue(n+1). Holds exactly** at all 7 levels of both runs. That is additivity: a subtree is its child's content plus the child's subtree. It holds for every tree, binary ones included, so it does not tell this tree apart.
- **Row 2: rvalue(n) = lvalue(n+1). Fails.** The ratio wanders from 0.17 to 2.5 and is never 1. Each level's own content is what its own sessions wrote, a constant per session (9,047 B), unrelated to how much is nested below it.

So **l/r does not approach φ.** It falls in a straight line, l/r = (levels below) × (inner content ÷ own content): exactly 7, 6, … 0 in run A. The tree's size grows linearly with depth, not as Fⁿ.

**The binary contrast.** A binary *search* descends one half and discards the other. A binary *tree* keeps both children, just as this tree keeps both branches. What sets this tree apart is that its split is uneven (one branch ends, one recurses). The measured unevenness is set by how many sessions each level hosts, not by φ.

## What would make the tree follow `M`, and the many-small-models proposal

### The Zeckendorf tree follows `M`

Run Z's tree has a second kind of child, the store reached through two `.claude`s. A store with *h* digits of address left therefore holds N(h) = 1 + N(h−2) + N(h−3) + … stores, and that recurrence reduces to N(h) − N(h−1) = N(h−2): the `M` step.

The root's two nearest child subtrees (through `.claude/p4` and through `.claude/.claude/p4`) are consecutive Fibonacci numbers in a tree grown to a total address length H:

| H | via `.claude/p4` | via `.claude/.claude/p4` | ratio |
|---|---|---|---|
| 10 | 34 | 21 | 1.619048 |
| 16 | 610 | 377 | 1.618037 |
| 22 | 10,946 | 6,765 | 1.618034 |
| 28 | 196,418 | 121,393 | 1.618034 |

**So rows 1 and 2 both hold for store counts, and the ratio converges on φ.** The chain of runs A and B is the degenerate case: only one child per store, so the second row can't hold.

**For bytes, the counts carry over only while each store holds about the same content.** In run Z every store is one session's 9,022–9,029 bytes, so bytes track the counts to within 0.1%. In a real store they follow whatever its sessions wrote, as runs A and B showed.

### The many-small-models proposal

The owner's proposal: instead of one large model (where phi has been measured against 2:4), put one small model in every store node's `models/`, nested as an endlessly growing recursive tree that mirrors the matrix it is trained on, so that many small models outgrow and outperform one large one.

The measurement says what that tree needs.

- **Row 2 must be made true by design: rvalue(n) = lvalue(n+1).** Each node's own content (here, what its model trains on) must equal everything nested below its child. Concretely, a node's model trains on its own sessions plus a distillation of the subtree under it: its child's text, or its child's model as a feature, the way vlds 0.0.72 already feeds the `.pt` score into the JSON ranker. With row 1 already exact, the sizes then step by `M` and grow as Fⁿ, with l/r → φ. Without it, as measured, they grow linearly.
- **"Outgrow" follows by construction once row 2 holds.** "Outperform" is a separate, untested claim. The experiment that would decide it:
  - the same store text, split into a nested chain of *k* small models (each trained on its node plus its child's distillation) against one model of the same total parameter count trained on everything
  - both scored by held-out r-precision on the store's labelled pools, the measure the p4 rankers already report
  - on AudioPlugin-p4 today, one small model scores 0.200 and the JSON ranker 0.380
- **Self-delimiting holds for this design**, with the comma bit (section 2). A stream of model addresses in such a tree parses with no headers, at the cost section 3 measures.

### Stage 1 measured: equal parameters, equal pretraining time

`scripts/session/nested_vs_one.py` ran on AudioPlugin-p4 (`d59a5f3`):
- **Data.** 91 MB of the store's tracked text, cut by when git first added each file into 4 shards of about 23 MB (oldest deepest), and its 22 labelled pools.
- **Arms.** One large ranker (2^15 × 64, 8 minutes over all the text) against a chain of 4 small ones (2^13 × 64 each, 2 minutes each on its own shard, each node starting from the previous node's weights).
- **Scoring.** 4-fold held-out r-precision, 15 fine-tuning epochs, 2 seeds.

The owner's framing is one model sifting 8 GB against 8 nodes of 1 GB each, at equal parameters and time. This run is that, scaled to 91 MB and 4 nodes. It differs in three ways:
- the chain's nodes hand their weights on
- every arm scores the same candidates (no routing)
- shards are cut by time, not topic

| Arm | Seed 0 | Seed 1 | Mean |
|---|---|---|---|
| one large (2^15 × 64) | 0.213 | 0.195 | **0.204** |
| ensemble of the 4 nodes (same total parameters) | 0.199 | 0.202 | **0.200** |
| chain root (one 2^13 × 64 node that has seen every shard in turn) | 0.190 | 0.185 | 0.187 |
| one small node on all the text (2^13 × 64, 2 minutes) | 0.167 | 0.150 | 0.158 |

**Verdict on ranking quality: a tie, not a win.**
- **The ensemble ties the one large model.** The large model's two seeds differ by 0.018, more than the 0.004 between the two means.
- **The chain root is 0.017 below.**
- **Nesting is what rescues a small model.** A single small node is 0.046 below. The chain root, one small node that inherited the shards below it, is 0.029 above that small node on its own.
- **Neither nested arm beats the large model** on held-out r-precision at equal size and time.
- **Every arm is far below the store's JSON ranker (0.380)**, so this compares `.pt` models with each other, not with the ranker the store uses.

"Outperform" on the other axes the owner names (fresh data absorbed, update messages between agents, forgetting) is stage 2: `scripts/session/nested_realtime.py`.

### Stage 3 measured: every angle (`scripts/session/nested_suite.py`)

The owner asked for every angle, not only held-out quality. This includes:
- the matrix angle: the matrix that pruned and trained a node also defines the route to it, so a query searches one node instead of all of them
- the 2:4 control that phi has been measured against

Setup:
- **Arms.** 14, pretrained once per seed and shared between the arms that use them.
- **Data.** The same store and pools as stage 1. k = 4 shards of 23 MB and k = 8 shards of 11 MB, scaling the owner's "one model on 8 GB against 8 nodes of 1 GB".
- **Training.** One large model gets 6 minutes of pretraining; each node gets 6/k.
- **Measurement.** Everything below is measured, with 10 fine-tuning epochs. k = 4 ran twice (seeds 0 and 1) and k = 8 once.
- **Seed spread.** Pretraining runs on a time budget, so step counts vary run to run, and some arms move by 0.05 between seeds. A difference smaller than an arm's own seed spread is not a result.

The arms:
- **Equal total parameters** (one 2^15 × 64, or k nodes of 2^15/k × 64):
  - **one large** and **one small**
  - **chain**: each node starts from the previous node's weights
  - **independent**: no handoff
  - **time-routed**: each candidate is scored only by the node whose era holds it
  - **Zeckendorf**: nodes at the tree's real addresses; a node with two children starts from their average
  - **φ-pruned** and **2:4-pruned**: the Zeckendorf nodes pruned along their feature rows
  - **matrix-routed**: the pruning mask sends each query to one node, and only that node searches and learns
- **grown**: every node full size, k× the parameters, the "many models outgrow one" angle.

**Held-out quality** (r-precision, 4 folds, 22 pools)

| Arm | k=4 s0 | k=4 s1 | k=4 mean | k=8 s0 |
|---|---|---|---|---|
| one large | 0.224 | 0.215 | **0.220** | 0.222 |
| one small | 0.178 | 0.152 | **0.165** | 0.201 |
| chain root | 0.203 | 0.202 | **0.202** | 0.191 |
| chain ensemble | 0.203 | 0.209 | **0.206** | 0.210 |
| independent ensemble | 0.224 | 0.186 | **0.205** | 0.202 |
| time-routed | 0.210 | 0.240 | **0.225** | 0.174 |
| Zeckendorf root | 0.206 | 0.212 | **0.209** | 0.218 |
| Zeckendorf ensemble | 0.232 | 0.238 | **0.235** | 0.207 |
| grown root (k× params) | 0.193 | 0.255 | **0.224** | 0.222 |
| grown ensemble (k× params) | 0.183 | 0.248 | **0.215** | 0.258 |
| φ-pruned ensemble | 0.204 | 0.262 | **0.233** | 0.208 |
| matrix-routed, φ | 0.192 | 0.188 | **0.190** | 0.171 |
| 2:4-pruned ensemble | 0.218 | 0.205 | **0.212** | 0.187 |
| matrix-routed, 2:4 | 0.178 | 0.152 | **0.165** | 0.164 |

**Deployed** (leave-one-out r-precision of the JSON ranker with the arm's out-of-fold score as its `.pt` feature; without any: 0.380)

| Arm | k=4 s0 | k=4 s1 | k=4 mean | k=8 s0 |
|---|---|---|---|---|
| one large | 0.383 | 0.391 | **0.387** | 0.376 |
| one small | 0.381 | 0.373 | **0.377** | 0.364 |
| chain root | 0.352 | 0.384 | **0.368** | 0.390 |
| chain ensemble | 0.387 | 0.390 | **0.389** | 0.379 |
| independent ensemble | 0.366 | 0.373 | **0.369** | 0.371 |
| time-routed | 0.384 | 0.375 | **0.380** | 0.377 |
| Zeckendorf root | 0.377 | 0.388 | **0.383** | 0.384 |
| Zeckendorf ensemble | 0.385 | 0.371 | **0.378** | 0.384 |
| grown root (k× params) | 0.378 | 0.396 | **0.387** | 0.397 |
| grown ensemble (k× params) | 0.373 | 0.399 | **0.386** | 0.391 |
| φ-pruned ensemble | 0.378 | 0.388 | **0.383** | 0.377 |
| matrix-routed, φ | 0.375 | 0.408 | **0.392** | 0.357 |
| 2:4-pruned ensemble | 0.374 | 0.375 | **0.375** | 0.372 |
| matrix-routed, 2:4 | 0.379 | 0.387 | **0.383** | 0.339 |

**Fresh** (prequential: the 14 pools after the first 8, each scored before it is learned)

| Arm | k=4 s0 | k=4 s1 | k=4 mean | k=8 s0 |
|---|---|---|---|---|
| one large | 0.151 | 0.131 | **0.141** | 0.141 |
| one small | 0.129 | 0.134 | **0.132** | 0.117 |
| chain root | 0.120 | 0.147 | **0.134** | 0.175 |
| chain ensemble | 0.154 | 0.134 | **0.144** | 0.198 |
| independent ensemble | 0.160 | 0.161 | **0.161** | 0.142 |
| time-routed | 0.118 | 0.116 | **0.117** | 0.087 |
| Zeckendorf root | 0.113 | 0.169 | **0.141** | 0.182 |
| Zeckendorf ensemble | 0.144 | 0.136 | **0.140** | 0.147 |
| grown root (k× params) | 0.115 | 0.129 | **0.122** | 0.136 |
| grown ensemble (k× params) | 0.113 | 0.150 | **0.132** | 0.131 |
| φ-pruned ensemble | 0.150 | 0.132 | **0.141** | 0.147 |
| matrix-routed, φ | 0.120 | 0.094 | **0.107** | 0.141 |
| 2:4-pruned ensemble | 0.136 | 0.098 | **0.117** | 0.142 |
| matrix-routed, 2:4 | 0.102 | 0.108 | **0.105** | 0.129 |

**Forgetting** (r-precision lost on the first 8 pools over the stream; lower is better)

| Arm | k=4 s0 | k=4 s1 | k=4 mean | k=8 s0 |
|---|---|---|---|---|
| one large | 0.252 | 0.460 | **0.356** | 0.204 |
| one small | 0.443 | 0.248 | **0.346** | 0.271 |
| chain root | 0.244 | 0.462 | **0.353** | 0.245 |
| chain ensemble | 0.245 | 0.430 | **0.338** | 0.255 |
| independent ensemble | 0.195 | 0.191 | **0.193** | 0.134 |
| time-routed | 0.000 | 0.000 | **0.000** | 0.000 |
| Zeckendorf root | 0.458 | 0.453 | **0.455** | 0.320 |
| Zeckendorf ensemble | 0.248 | 0.230 | **0.239** | 0.229 |
| grown root (k× params) | 0.266 | 0.305 | **0.285** | 0.232 |
| grown ensemble (k× params) | 0.285 | 0.221 | **0.253** | 0.232 |
| φ-pruned ensemble | 0.262 | 0.224 | **0.243** | 0.223 |
| matrix-routed, φ | 0.201 | 0.333 | **0.267** | 0.173 |
| 2:4-pruned ensemble | 0.269 | 0.284 | **0.277** | 0.181 |
| matrix-routed, 2:4 | 0.291 | 0.426 | **0.359** | 0.467 |

**Cost per new pool, and size** (k=4 mean / k=8)

| Arm | update s | message KB | query ms | params (fp16 MB) |
|---|---|---|---|---|
| one large | 1.63 / 1.75 | 133 / 133 | 9.2 / 11.0 | 4.23 / 4.23 |
| one small | 1.56 / 1.76 | 72 / 42 | 8.5 / 8.8 | 1.08 / 0.56 |
| chain root | 1.58 / 1.64 | 73 / 43 | 8.2 / 9.6 | 1.08 / 0.56 |
| chain ensemble | 5.91 / 11.78 | 291 / 346 | 32.2 / 61.4 | 4.33 / 4.46 |
| independent ensemble | 5.59 / 11.72 | 289 / 340 | 29.9 / 57.7 | 4.33 / 4.46 |
| time-routed | 1.35 / 1.55 | 73 / 43 | 31.6 / 62.4 | 4.33 / 4.46 |
| Zeckendorf root | 1.44 / 1.62 | 72 / 42 | 9.1 / 9.2 | 1.08 / 0.56 |
| Zeckendorf ensemble | 5.93 / 12.36 | 289 / 340 | 32.7 / 57.9 | 4.33 / 4.46 |
| grown root (k× params) | 1.64 / 1.67 | 133 / 133 | 8.9 / 9.3 | 4.23 / 4.23 |
| grown ensemble (k× params) | 6.02 / 12.36 | 534 / 1063 | 32.1 / 63.1 | 16.91 / 33.82 |
| φ-pruned ensemble | 5.62 / 11.73 | 152 / 183 | 30.4 / 61.1 | 4.33 / 4.46 |
| matrix-routed, φ | 1.47 / 1.49 | 37 / 23 | 8.7 / 9.2 | 4.33 / 4.46 |
| 2:4-pruned ensemble | 5.81 / 11.55 | 153 / 185 | 32.5 / 64.3 | 4.33 / 4.46 |
| matrix-routed, 2:4 | 1.52 / 1.40 | 38 / 23 | 8.8 / 9.9 | 4.33 / 4.46 |

**Fault tolerance and routing** (k=4 s0, s1 / k=8 s0)

| Arm | quality | one node dropped | route finds the best node |
|---|---|---|---|
| chain ensemble | 0.203, 0.209 / 0.210 | 0.206, 0.210 / 0.203 | — |
| independent ensemble | 0.224, 0.186 / 0.202 | 0.216, 0.190 / 0.206 | — |
| Zeckendorf ensemble | 0.232, 0.238 / 0.207 | 0.233, 0.228 / 0.205 | — |
| grown ensemble (k× params) | 0.183, 0.248 / 0.258 | 0.194, 0.244 / 0.266 | — |
| φ-pruned ensemble | 0.204, 0.262 / 0.208 | 0.203, 0.214 / 0.206 | — |
| 2:4-pruned ensemble | 0.218, 0.205 / 0.187 | 0.221, 0.202 / 0.191 | — |
| matrix-routed, φ | 0.192, 0.188 / 0.171 | — | 0.455, 0.545 / 0.318 |
| matrix-routed, 2:4 | 0.178, 0.152 / 0.164 | — | 0.455, 0.409 / 0.409 |

**Pretraining wall time** (seconds, sequential / with the nodes in parallel): one large 361; k=4 nested 360 / 90; k=8 nested 361 / 45. Chain and Zeckendorf nodes hand weights on, so only independent nodes truly run in parallel; the chain's parallel figure is its per-node time.


**Verdicts, stage 3**

- **Held-out quality at equal parameters: no reliable win.**
  - At k = 4, the Zeckendorf ensemble beat the one large model on both seeds: 0.232 and 0.238 against 0.224 and 0.215, a mean of 0.235 against 0.220.
  - At k = 8 it fell below: 0.207 against 0.222.
  - The φ-pruned ensemble's k = 4 mean (0.233) rests on one seed's 0.262.
  - No arm at equal parameters beats one large model at both sizes.
- **Outgrowing works.** With k× the parameters, the grown ensemble reached 0.258 at k = 8 against one large model's 0.222, its best result. At k = 4 its two seeds disagree (0.183 and 0.248).
- **The deployed effect is mostly a wash.** As the JSON ranker's `.pt` feature, most arms land within ±0.01 of the JSON ranker alone (0.380). The best k = 4 mean is matrix-routed φ at 0.392 (0.375 and 0.408), but at k = 8 it drops to 0.357. The grown root helps at both sizes: 0.387 mean at k = 4, 0.397 at k = 8.
- **Freshness: some nested arms win, not all.**
  - At k = 8, the chain ensemble (0.198) and the Zeckendorf root (0.182) score fresh pools well above one large model (0.141).
  - At k = 4, the independent ensemble leads (0.161 against 0.141) and the rest are level or below.
  - Routing new data to one node does not help freshness. Time-routed scored 0.117 and 0.087.
- **Forgetting: routing protects what is old.**
  - Time-routed forgets nothing (0.000 everywhere), because new pools only ever reach the newest node. That's structural.
  - The independent ensemble forgets about half as much as one large model: 0.193 against 0.356 at k = 4, and 0.134 against 0.204 at k = 8.
  - Matrix-routed φ forgets less than one large model at both sizes (0.267 and 0.173). Matrix-routed 2:4 forgets the most of any arm at k = 8 (0.467).
- **Update messages: much smaller, at both sizes.** A routed update ships one node's φ-delta: 37 KB at k = 4 and 23 KB at k = 8, against 133 KB for one large model, 3.6× and 5.8× smaller. Any single-node update is 72 KB and 42 KB. Ensembles that update every node ship more than one model does (289–346 KB).
- **Update and query time: equal when one node acts, slower when all do.**
  - One node updates in about 1.5 s and answers in about 9 ms, the same as one large model. At these sizes the time goes to hashing features, not to the table size.
  - Ensembles take 4–7× longer to update and 3–6× longer to answer.
- **Pretraining wall time.** Independent nodes run in parallel: 90 s at k = 4 and 45 s at k = 8, against 361 s for one large model on the same total budget.
- **Fault tolerance.** Dropping any one node moves an ensemble by under 0.01 in 11 of 12 cases. The exception is the φ-pruned ensemble's best seed (0.262 to 0.214).
- **The matrix as the route works as a pointer, not yet as an answer.**
  - The node the pruning mask picks is the best node for the pool 50% of the time with φ at k = 4 (45.5% and 54.5%) and 32% at k = 8. Chance is 25% and 12.5%.
  - 2:4 masks: 43% and 41%.
  - The routed node alone ranks below every ensemble.
  - φ routing ranks better than 2:4 routing at both sizes (0.190 against 0.165; 0.171 against 0.164), forgets less (0.267 against 0.359; 0.173 against 0.467), and deploys better (0.392 against 0.383; 0.357 against 0.339).
  - The limit: the hashed features scatter meaning over rows at random, so "no two adjacent kept rows" carries no meaning of its own in this ranker. That routing still finds the right node at twice to two and a half times chance says the masks keep each node's heavy rows, which is where its data lives.

### The realtime loop through the same stream

The owner adds that such a tree also allows one continuous realtime training loop through the same Fibonacci stream the p4 models already retrain on.

**What exists today, per store.**
- The turn close retrains in the background, and each `.pt` round after an anchor ships as `models/ranker.rNNN.phid`: the round's change, positions in Fibonacci gap codes and values as Fibonacci-coded levels, packed eight bits to a byte (`phi_prune.delta`).
- `phi_serve` reloads at its next request.
- That loop already runs on AudioPlugin-p4, one store at a time.

**What one stream through a nested tree would add.** Each round would be one item:

```
[node address][round]
```

The node address is the tree code of section 2, a Fibonacci codeword, so a reader knows which node's model a round belongs to with no header. Two parts of today's format stop it from being a single bitstream:

- **The `.phid` header is a JSON line**: tensor names, shapes, kept counts and scales. Its payload frames itself only because the header gives each tensor's count.
- **The scales are floats.**

The counts and shapes can be Fibonacci-coded the same way. The scales cannot, so one stream needs a fixed float width for them, the same kind of cost as the binary width of section 3.

**Untested.** No such stream has been built or measured here. The numbers that would decide it:
- its bits per kept weight against today's per-node `.phid` files (10.6 bits per kept entry on AudioPlugin-p4's round 1)
- the cost of the address codewords, which section 3 puts at 2*n* bits for a node at depth *n*

## 5. Agreed in advance against logic-free flow

The owner's point, 2026-10-27:
- A boundary "agreed in advance" (a fixed width, a length field) is a wall for continual retraining. It has to be renegotiated when the tree or the model outgrows it.
- A boundary that just happens, because the code never contains `11`, lets each node hand the stream on with no per-item if/else.
- The Cassini check is the same scan: finding where a codeword ends and catching a corrupt one are one comparison, not a separate validation step.

`scripts/session/stream_logic.py` measures this on 2,000 random Zeckendorf store addresses of up to 12 digits (the 144 addresses run Z grows), with every flip sampled from 1,500 single-bit flips per code. Decisions are counted by instrumenting each decoder: every comparison it makes to know where it is.

| Code | Bits/item | Decisions/item | Flips caught by the code's own structure | …with a table of every valid address | Items lost per flip (sequences aligned) | Parses after the tree grows to 17 digits |
|---|---|---|---|---|---|---|
| **Fibonacci** (the comma is the forbidden `11`) | **11.34** | 12.34 | **46.3%** | (no table needed) | **1.22** | **yes** |
| Elias-gamma (universal, no width agreed) | 14.21 | 9.61 | 0% | 81.1% | 2.56 | yes |
| fixed 9-bit (width agreed from the largest address then known) | 9.00 | 0 | 0% | 65.3% | 1.00 | **no** |
| 4-bit length field, then the digits | 14.34 | 4.00 | 74.1%* | (no table) | **169.3** | **no** |

\* only because its decoder also checks each address's digits for `11`: the Zeckendorf rule doing the catching, not the length field.

**What holds**

- **Agreed in advance is a wall.** When the tree grew past the agreement (addresses of up to 17 digits), the fixed width and the length field could not carry the new addresses at all. Fibonacci and Elias-gamma parsed them unchanged: neither agrees a width in advance. For a model that retrains continually and a tree that keeps nesting, that rules out the fixed codes. It does not rule out binary as such: Elias-gamma is binary and grows too.
- **One scan does both jobs.** The Fibonacci decoder's only rule, "`11` ends a codeword", is also its integrity check. It caught 46% of single-bit flips with no table, no parity and no second pass, the README's 41–52% for weights. Elias-gamma and fixed width caught none on their own. They caught 81% and 65% only by looking every value up in a table of all valid addresses, which is one more thing agreed in advance and one more check per item.
- **It falls back into step on its own.** After a flip, Fibonacci lost 1.22 items on average and then resynchronised at the next `11`. Elias-gamma lost 2.56. A flip in a length field derailed 169 items, because nothing in the data marks where the next item starts.
- **It is smaller than the open-ended binary code.** 11.34 bits an address against Elias-gamma's 14.21 and the length field's 14.34. Fixed width is smallest (9.0) but cannot grow and catches nothing on its own.

**What does not hold**

- **"Zero logic" is not what was measured.** Fibonacci's decoder makes one comparison per bit, 12.34 per address, more than Elias-gamma (9.61), and fixed width makes none. What it does not have is any other kind of logic:
  - no width or length to agree on, no header to read
  - no mode switch between reading a length and reading data
  - no table to check against
  - no separate validation pass

  It is one uniform test, repeated, that both delimits and verifies. "100 fires with no if/else" is right in that sense, one rule at every fire, but every fire still makes that one test.

## Verdicts

- **Holds.**
  - The hooks, unchanged, enforce exactly the Zeckendorf constraint: a `p4` in a `.claude` and a `.claude` in a `.claude`, never a `p4` in a `p4`. They grow every Zeckendorf address (run Z: 21 stores, counts 1, 0, 1, 1, 2, 3, 5, 8).
  - Every address is a distinct integer.
  - The paths parse with no headers, using the forbidden `11` as the comma.
  - Both rows of `M` hold for the Zeckendorf tree's store counts, and the child-subtree ratio converges on φ (1.618034 at H = 22).
- **Holds only with a qualifier.**
  - The φ ratio is in store counts. Bytes follow it only while stores hold about the same content.
  - A single chain (runs A and B) is the degenerate case. Its language is `(10)*1`, its depth reads as F(2n+1) − 1, and only row 1 of `M` holds, with l/r linear in depth.
  - The bit cost in section 3 was measured on the chain. There the tree code (9.0 bits) beats binary carrying the same digits with length fields (12.0), but not a code of the depth itself (about 4.3).
- **Holds (section 5).** For a stream that keeps growing, only codes with no agreed width parse what comes later: Fibonacci and Elias-gamma did, fixed width and length fields did not. Fibonacci's one rule both delimits and catches 46% of flips with no table, falls back into step after 1.2 items, and costs fewer bits than Elias-gamma (11.3 against 14.2).
- **Fails.**
  - "Binary cannot express it." Binary expresses the same nesting, with a fixed width or a length field, or open-ended with Elias-gamma. What binary cannot do is delimit and verify with one rule and no agreement in advance.
  - "Zero logic." Fibonacci makes one test per bit. What it lacks is any other logic.
  - The binary-search contrast is an analogy: a binary tree also keeps both children.
- **Measured, stages 1 and 3.**
  - On held-out ranking quality at equal parameters, nesting ties one large model: stage 1's ensemble 0.200 against 0.204. It does not reliably beat it: the Zeckendorf ensemble wins at k = 4 on both seeds and loses at k = 8.
  - With more total parameters (grown), nesting beats it at k = 8 (0.258 against 0.222).
  - On the realtime axes, nested and routed trees beat one large model:
    - update messages 3.6–5.8× smaller
    - pretraining 4–8× faster when the nodes run in parallel
    - forgetting cut by half (independent nodes) or to zero (time-routed)
    - fresh pools scored better by some arms at k = 8
    - one node dropped costs under 0.01
  - The pruning matrix routes to the best node at about twice chance or better, and φ routes better than 2:4 on every axis but route accuracy at k = 8.
- **Untested.** The single realtime stream of every node's rounds; more seeds, which several arms need before their differences count; and a ranker whose features have an order, where φ's adjacency rule would mean something about the data.
