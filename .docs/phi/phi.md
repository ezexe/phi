# φ-Register

## The Recursive Tree

```python
def fib(n):
    if n <= 0: return 0
    if n == 1: return 1
    return fib(n-1) + fib(n-2)
```

Every time you go up one number (e.g., from 5 to 6), you aren't just adding one branch—you are re-growing the entire tree from two steps ago as a side branch. (6) contains a full (5) tree on the left and a full (4) tree on the right.

Exactly how many **(1)**s and **(0)**s (the base cases) it takes to build the final number? Here is the full, un-cached expansion:

```
* fib(0) --> 0
* fib(1) --> 1

* fib(2) --> 1
  (2)
  / \
 1 + 0

* fib(3) --> 2
    (3)
    / \
  (2)+ 1
  / \
 1 + 0

* fib(4) --> 3
       _(4)_
      /     \
    (3)  +  (2)
    / \     / \
  (2)+ 1   1 + 0
  / \
 1 + 0

* fib(5) --> 5
             ___(5)___
            /         \
       _(4)_     +    (3)
      /     \         / \
    (3)  +  (2)     (2)+ 1
    / \     / \     / \
  (2)+ 1   1 + 0   1 + 0
  / \
 1 + 0

* fib(6) --> 8
                   _______(6)_______
                  /                 \
             ___(5)___     +       _(4)_
            /         \           /     \
       _(4)_     +    (3)       (3)  +  (2)
      /     \         / \       / \     / \
    (3)  +  (2)     (2)+ 1    (2)+ 1   1 + 0
    / \     / \     / \       / \
  (2)+ 1   1 + 0   1 + 0     1 + 0
  / \
 1 + 0
```

## Two Ciphers on One Table

| Decimal | Binary Weights (8,4,2,1) | Binary Code | Fibonacci Weights (13,8,5,3,2,1) | Zeckendorf Code |
| ------- | ------------------------ | ----------- | -------------------------------- | --------------- |
| 1       | 1                        | 0001        | 1                                | 000001          |
| 2       | 2                        | 0010        | 2                                | 000010          |
| 3       | 2+1                      | 0011        | 3                                | 000100          |
| 4       | 4                        | 0100        | 3+1                              | 000101          |
| 5       | 4+1                      | 0101        | 5                                | 001000          |
| 6       | 4+2                      | 0110        | 5+1                              | 001001          |
| 7       | 4+2+1                    | 0111        | 5+2                              | 001010          |
| 8       | 8                        | 1000        | 8                                | 010000          |
| 9       | 8+1                      | 1001        | 8+1                              | 010001          |
| 10      | 8+2                      | 1010        | 8+2                              | 010010          |
| 11      | 8+2+1                    | 1011        | 8+3                              | 010100          |
| 12      | 8+4                      | 1100        | 8+3+1                            | 010101          |
| 13      | 8+4+1                    | 1101        | 13                               | 100000          |
| 14      | 8+4+2                    | 1110        | 13+1                             | 100001          |
| 15      | 8+4+2+1                  | 1111        | 13+2                             | 100010          |
| 16      | 16                       | 10000       | 13+3                             | 100100          |

These aren't competing systems. They're both projections of the same structure.

## The Native Object

The Fibonacci recurrence is a linear map. Applied once, it advances a pair of consecutive values by one step:

```
[F(k+1)]   [1 1]   [F(k)  ]
[F(k)  ] = [1 0] × [F(k-1)]
```

Applied n times, it produces:

```
M^n = [F(n+1)  F(n)  ]
      [F(n)    F(n-1)]
```

Four entries. Three distinct values (the matrix is symmetric: `[0,1] = [1,0]`). One recurrence linking them: `[0][0] = [0][1] + [1][1]`, i.e. `F(n+1) = F(n) + F(n-1)`. The recurrence isn't a rule imposed on the entries — it's readable directly off the geometry. Row/column structure _is_ the recurrence.

This matrix is the native object. Everything else — the 1D register, binary, the adjacency constraint, the Fibonacci delimiter — is a cipher on top of it.

## Six Projections of One Fact

**1. The Adjacency Constraint.** In a flat Zeckendorf register, each cell is an independent switch with a Fibonacci weight. The weights have a dependency — `F(k) + F(k+1) = F(k+2)` — but the cells don't know about it. Independent addressing + dependent weights = redundancy. The no-adjacent-1s rule is duct tape over a dimensional mismatch: a 2D relationship forced into a 1D address space.

**2. The Matrix Symmetry.** In the matrix, `[0,1]` and `[1,0]` both give `F(n)`. Two addresses, one value. The redundancy that the 1D register forbids, the matrix expresses as symmetry. Four states, three distinct outputs. In 1D, one of four states is _forbidden_. In 2D, one of four states is _redundant_.

**3. The Fibonacci Coding Delimiter.** The bit pattern `11` can never appear inside a valid Zeckendorf representation. So it's repurposed as an end-of-number marker in Fibonacci codes. This works because `11` corresponds to the redundant matrix state — `[1,1]` selects `F(n-1)`, which carries no information that `[0,0]` and `[0,1]` don't already determine. The delimiter falls out of the matrix's symmetry.

**4. The Weight Sequence Fork.** Let `11` live as a valid state and the adjacency constraint vanishes. With it goes the Fibonacci weight sequence — the next position no longer needs weight `F(k+2)` because the carry that would have produced it was absorbed. The weights collapse to powers of 2.

```
Allow [1,1]:   weights = 1, 2, 4, 8, 16...    → binary
Forbid [1,1]:  weights = 1, 2, 3, 5, 8, 13... → Fibonacci
Same matrix. Different cipher.
```

**5. The Pythagorean Norm.** In the matrix pair `(a, b) = (F(k), F(k+1))`, the sum of squares `a² + b² = F(2k+1)` — always a Fibonacci number. The squared distance from the origin of the state vector lives on the sequence. In the 1D register, corruption is caught by adjacent 1s. In the matrix pair, corruption is caught by `a² + b²` falling off the Fibonacci sequence. Same structural error detection, different geometry. A second invariant confirms it: `b² - ab - a² = (-1)^k` (Cassini's identity), meaning the matrix has determinant ±1. Two scalar checks on a two-value state — the system is fully determined.

**6. The NAND Gate.** The base matrix M = [[1,1],[1,0]] is the NAND truth table: two binary inputs, one binary output, and the only zero sits at [1,1]. The position all five other projections converge on — forbidden, redundant, delimiter, fork point, norm anchor — is the gate's only rejection. The Fibonacci sequence is what falls out when NAND's truth table composes with itself under matrix multiplication. (The connection is structural, not computational — matrix self-multiplication isn't gate chaining, so NAND's universality as a logic primitive doesn't carry over. The truth table shape grounds _why_ [1,1] is special; it doesn't extend the framework.)

## The Axiom

One identity generates everything:

**F(n) = F(n-1) + F(n-2)**

This isn't a formula you derive — it's the law of the machine. The carry rule, the borrow rule, addition, subtraction, normalization, and error detection all fall out of reading this identity in different directions.

## The Register

A register is an ordered sequence of cells. Each cell holds 0 or 1. Cell _k_ has weight F(k). A register is **valid** when no two adjacent cells both hold 1.

The first self-referential property: the number of valid states in an _n_-cell register is F(n+2). The machine counts its own capacity in its own number system.

## Arithmetic from the Single Identity

Three operations, all just the axiom rewritten:

**Upward (carry):** Cells k and k+1 both hold 1? Clear both, set cell k+2. That's just F(k) + F(k+1) = F(k+2) — the axiom read left to right.

**Downward (borrow):** Cell k holds 1 and you need to decompose it? Clear k, set cells k-1 and k-2. That's F(k) = F(k-1) + F(k-2) — the axiom read right to left.

**Doubling (resolve a 2):** Raw addition can produce a cell holding "2." Since 2·F(k) = F(k+1) + F(k-2), clear the 2 and set those two cells. This is the axiom applied to F(k) + F(k) = F(k) + F(k-1) + F(k-2) = F(k+1) + F(k-2).

That's the complete arithmetic engine. Addition is raw cell-wise sum followed by sweeping these three rules until the register is valid. Subtraction is alignment plus downward expansion plus normalization. Every operation terminates, though not because carries only move up: the doubling rule sends one carry two cells down. `phi_prune/register.py` checks the result on every sum of two 6-cell registers and on random rule orders.

## The Clock is Convergence

There's no external clock. You pour values into a register, and the normalization rules cascade like a wavefront until the system settles into a valid state. **Computation is settling.** The settling time is bounded by register width, so it always halts — but the machine's natural rhythm is the propagation speed of carries, not an imposed tick.

Settling also announces itself. A register still settling holds a `11` or a 2 somewhere; a settled one holds neither, so "done" is the same local scan as the error check below, with no worst-case timer. A binary adder's wires show an ordinary number while a carry is still in flight, which is why a CPU latches its sum on a clock tick sized for the longest carry.

Measured with every cell rewriting in parallel (`normalize_rounds` in `phi_prune/register.py`, every sum of two values below 144): the φ-register settles in 2.7 rounds on average and 8 at worst; binary in 2.1 on average and 8 at worst. A self-timed φ-register pays its average, a clocked binary register its worst case, about 3× apart. Two limits: binary can be self-timed too, with carry-completion wires, and then pays its lower average; and a CPU's carry-lookahead adder settles in about log₂(width) gate delays, not one cell per round, so rounds compare the rule systems, not circuits.

This is closer to how physical systems actually compute. A crystal grows until it's stable. A river finds its channel. The φ-register normalizes.

## Error Detection is Free

In any positional system with unconstrained digits, every bit pattern is valid and corruption is silent. Here, the "no adjacent 1s" invariant means any corruption that creates an adjacency is **immediately visible**. The representation itself is a checksum.

Detection is not correction. Applying the carry rule to a corrupted register makes it valid again at the corrupted value, which hides the error, so check before you normalize. Only a 0→1 flip next to a 1 shows: 40.9% of single flips over all 8-cell registers. A flip that lands between two 1s (`101 → 111`) has one valid repair, 8.6% of flips; a lone `11` could have come from either cell. You don't bolt on parity bits; structural integrity is native.

## Multiplication

Left-shift by one position scales a value by approximately φ. Not exactly — the result may need normalization — but the _structure_ is that shifting is φ-scaling, and the normalization cleanup is bounded.

Multiplying by F(k) specifically: decompose via the recurrence. F(k) = F(k-1) + F(k-2), so `x · F(k) = x · F(k-1) + x · F(k-2)`. This recurses down to `x · F(1) = x` and `x · F(2) = x` (shift-by-one). Multiplication is a tree of additions shaped exactly like the Fibonacci call tree. **The algorithm's structure mirrors the number system's structure.** General multiplication between two arbitrary values decomposes the multiplier into its Zeckendorf digits and sums the shifted copies — analogous to long multiplication, but with φ-spaced shifts instead of uniform ones.

## Comparison, Branching, Division

**Comparison** is scan-from-top: first cell that differs determines order. Same principle as any positional system; the non-adjacency constraint doesn't interfere.

**Division** is repeated subtraction with φ-scaled shifting. Long division, but each trial subtraction shifts by Fibonacci positions. The quotient accumulates in Zeckendorf form naturally.

**The native gate** that doesn't exist elsewhere: **NORMALIZE.** Every other logic operation (AND, OR, NOT on raw cells) can produce invalid states. NORMALIZE sweeps the register back to validity. It's the primitive. It _is_ the computation.

## What This Machine is Good At

The φ-machine isn't a worse version of something else. It has intrinsic properties:

**Self-similar scaling.** Every substructure at scale n contains the structure at n-1 and n-2. This maps naturally to problems that are themselves recursive or fractal — tree traversals, branching processes, growth models.

**Built-in redundancy detection.** The adjacency constraint means the representation space is sparser than the raw bit space. This is a feature — it means encoding carries more structure per symbol, at the cost of more symbols per value.

**Natural compression delimiter.** The sequence `11` can never appear inside a valid number, so it's a free end-of-number marker. You can concatenate φ-encoded values in a stream with zero framing overhead. This is why Fibonacci codes already exist in data compression — they fell out of this system, not out of the legacy one.

**Gentle overflow.** When a register fills, it grows by one cell, increasing capacity by a factor of φ ≈ 1.618 rather than doubling. Growth is spiral, not exponential.

## The Dual Cost Structure

The matrix preserves its invariants under multiplication automatically. No normalization, no sweep — the dot product structure makes invalid states unreachable. The 1D register preserves its invariants via normalization — an explicit sweep because the flat address space doesn't enforce structure.

```
1D:      valid in + valid in → possibly invalid → normalize → valid out
Matrix:  valid in × valid in → valid out (guaranteed by dot product)
```

The normalization sweep in the 1D register is doing from the outside what the dot product does from the inside. The two representations are duals — each native where the other is expensive:

```
                        1D φ-Register         Matrix Register
Value addition          native (pour+norm)    expensive (decompose)
Index composition       expensive (matrix)    native (multiply)
Normalization           required              impossible to violate
Arbitrary integers      direct                only Fibonacci-indexed
Inspectable digits      yes                   no (opaque pair)
Settling                self-timed            none (valid by construction)
Done signal             no 11, no 2           not needed
Integrity check         no 11 (same scan)     a²+b² on sequence, Cassini ±1
```

## The Switching Strategy

The matrix identity's practical consequence isn't a hybrid architecture — it's a decision framework. Since both representations are projections of the same object, converting between them is lossless and cheap. Stay in whichever projection fits the current operation.

**Value work → stay in 1D.** Addition, subtraction, comparison, reading individual digits. The normalization sweep is the right cost to pay. Converting to matrix form would require decomposition and reconstruction for no benefit.

**Composition work → switch to matrix.** Exponentiation, fast doubling, multiplying by Fibonacci numbers, advancing indices. Build the `(F(k), F(k+1))` pair from the Zeckendorf form, O(width), then compose via matrix multiplication with no further normalization. Convert back to 1D when done.

**Multiplication decomposes naturally.** The multiplier determines the structure: Zeckendorf-decompose it (1D work), then for each active digit, shift and add (1D work). The matrix enters only if you need to compute F(n) for large n, or if the multiplication is by a Fibonacci number (which is a single index shift in matrix form).

The startup cost of building the matrix pair from Zeckendorf form when you actually need it is O(width): a walk over every cell, where a self-timed addition settles in a few rounds whatever the width (2.7 on average at 10 cells). You don't pre-pay it — you convert on demand, when the composition work saves more than that walk.

### Self-timing in the switching strategy

Self-timing belongs to the 1D side only, so it sharpens the rule for when to stay there.

**The 1D side is where the clock disappears.** Each value operation settles locally and signals its own end with the no-`11` scan, the same scan that checks for corruption and frames codewords in a stream: one check, three jobs. A run of additions, such as accumulating the shift-and-add partial products of a Fibonacci-coded weight, pays each sum's own settling time, not the widest carry's.

**The matrix side never settles.** Products are valid by construction, so there is nothing to wait for and nothing to signal. Its entries are ordinary integers, though, and in hardware its multiplies run on binary multipliers, clocked like any other. Its checks (a²+b² on the sequence, Cassini's ±1) catch corruption; they say nothing about completion.

**So switching projection also switches timing discipline:** value work self-timed in 1D, composition work clocked in the matrix pair, with the O(width) conversion as the boundary between them. Long runs of additions are where staying in 1D pays most; each switch to the matrix gives the self-timing up until you convert back.

```
Operation           →  Representation  →  Why
Addition            →  1D register     →  pour and normalize
Subtraction         →  1D register     →  borrow and normalize
Comparison          →  1D register     →  scan from top
Fast doubling       →  matrix pair     →  square: (a,b) → (a(2b-a), a²+b²)
Exponentiation      →  matrix pair     →  compose via multiply
Multiply by F(k)    →  matrix pair     →  single index shift
General multiply    →  both            →  decompose in 1D, shift-add, matrix for F(k) lookup
```

## Matrix Exponentiation: Two Traversals

The recurrence `F(n) = F(n-1) + F(n-2)` expressed as a matrix power:

```
[F(n+1)]   [1 1]^n   [1]
[F(n)  ] = [1 0]   × [0]
```

Two methods to compute M^n — same matrix, different cipher driving the traversal:

**Binary traversal (fast doubling).** Decompose n in binary. At each bit, square the pair. If the bit is 1, also multiply by M. This exploits the fact that squaring doubles the index: `(F(k), F(k+1)) → (F(2k), F(2k+1))`.

```
F(2k)   = F(k) · [2·F(k+1) - F(k)]
F(2k+1) = F(k)² + F(k+1)²
```

**Fibonacci traversal (ladder).** Maintain two consecutive Fibonacci-indexed powers `M^F(k-1)` and `M^F(k)`. Combine them to get `M^F(k+1)`. Climb the ladder to the largest Zeckendorf term, then assemble remaining terms.

```
M^F(k) = M^F(k-1) × M^F(k-2)    — the recurrence in the exponent
```

Both are O(log n). Fast doubling uses fewer scalar operations (~3 muls per step × log₂(n) steps). The ladder uses ~1 matrix multiply per step × log_φ(n) ≈ 1.44 log₂(n) steps. Same matrix underneath, different decomposition of the same exponent.

## The Core Insight

The φ-register isn't an alternative to binary. Both are lossy projections of a 2D structure onto a 1D bit string, each preserving different properties and paying different costs. The matrix is where they meet — binary-indexed rows and columns, Fibonacci-valued entries. Binary as the address bus, Fibonacci as the data bus, the matrix as the interface.
