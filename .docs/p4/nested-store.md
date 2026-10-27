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
- **Fails.**
  - "Binary cannot express it." Binary expresses the same nesting, with a fixed width or a length field.
  - The binary-search contrast is an analogy: a binary tree also keeps both children.
- **Untested.** That many nested small models outperform one large one, and the single realtime stream of every node's rounds. The experiment and the numbers above would decide them.
