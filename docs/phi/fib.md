# φ-Patterns

Since the choice of method reveals something about _which structural property of the recurrence you're exploiting_.

Here's a comprehensive breakdown by the fundamental computational strategy, not just the syntax pattern.

---

## 1. Naive Recursion

```python
def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)
```

This is the method you have in your project docs — the one that produces the self-similar call tree where fib(6) contains a full fib(5) and fib(4) as subtrees. It's the _definitional_ method: it treats the recurrence as a literal evaluation strategy.

**Time:** O(φⁿ) — the call count itself follows the Fibonacci sequence. **Space:** O(n) stack depth. **Insight:** The explosion is the point for your project. The tree's leaf count _is_ fib(n), so the algorithm's cost structurally encodes the answer. This is the method where computation mirrors the number system most transparently — and why your call-tree visualization works.

---

## 2. Memoized Recursion (Top-Down DP)

```python
def fib(n, memo={}):
    if n in memo: return memo[n]
    if n <= 1: return memo.setdefault(n, n)
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]
```

Same tree structure, but you cache each node on first visit and short-circuit on revisit. This collapses the exponential tree into a linear chain — you visit each value 0..n exactly once.

**Time:** O(n). **Space:** O(n) for both memo table and stack. **Insight:** Memoization is essentially "normalizing" the redundant subtrees. There's a parallel to your normalization engine — the naive tree has "violations" (redundant recomputation), and memoization is the sweep that eliminates them.

---

## 3. Bottom-Up Iteration (Tabulation)

```python
def fib(n):
    if n <= 1: return n
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b
```

Instead of decomposing top-down and caching, you build forward from the base cases. No recursion, no hash lookups.

**Time:** O(n). **Space:** O(1) — you only ever hold two values. **Insight:** This is the "register machine" approach. You maintain a 2-cell sliding window, and each step is a single addition plus a shift. It's essentially what your φ-register's addition circuit does at each position — propagate forward from known state.

---

## 4. Matrix Exponentiation

The recurrence F(n) = F(n-1) + F(n-2) can be expressed as:

```
[F(n+1)]   [1 1]^n   [1]
[F(n)  ] = [1 0]   × [0]
```

Using exponentiation by squaring on the 2×2 matrix:

```python
import numpy as np

def fib(n):
    if n <= 1: return n
    M = np.array([[1,1],[1,0]], dtype=object)
    result = matrix_pow(M, n)
    return result[0][1]

def matrix_pow(M, p):
    if p == 1: return M
    if p % 2 == 0:
        half = matrix_pow(M, p//2)
        return half @ half
    return M @ matrix_pow(M, p-1)
```

**Time:** O(log n) matrix multiplications (each is O(1) for 2×2, or O(M(k)) where M(k) is the cost of multiplying k-digit numbers). **Space:** O(log n) stack, or O(1) iteratively. **Insight:** This is the method that breaks out of linear time. It exploits the _algebraic_ structure — the recurrence is a linear map, so repeated application is exponentiation, and exponentiation admits squaring shortcuts. For your project: this is the binary method of exponentiation. A φ-native version would decompose the exponent in Zeckendorf form — using your multiplication-as-tree-of-additions structure instead of repeated squaring.

---

## 5. Fast Doubling

Derived from the matrix method but avoids full matrix multiplication by exploiting the identities:

```
F(2k)   = F(k) · [2·F(k+1) - F(k)]
F(2k+1) = F(k)² + F(k+1)²
```

```python
def fib(n):
    def _fib(n):
        if n == 0: return (0, 1)
        a, b = _fib(n >> 1)
        c = a * (2*b - a)
        d = a*a + b*b
        if n & 1: return (d, c+d)
        return (c, d)
    return _fib(n)[0]
```

**Time:** O(log n) — same asymptotic as matrix exponentiation but with lower constants (no matrix overhead). **Space:** O(log n) recursion depth. **Insight:** This is the most efficient practical method for large n. It's essentially the matrix method with the symmetry of the Q-matrix baked in. The bit-decomposition of n drives the recursion — again, binary-native. A Zeckendorf-native fast-doubling would need analogous identities for F(k+F(j)) compositions.

---

## 6. Closed-Form (Binet's Formula)

```python
from math import sqrt

def fib(n):
    phi = (1 + sqrt(5)) / 2
    psi = (1 - sqrt(5)) / 2
    return round((phi**n - psi**n) / sqrt(5))
```

**Time:** O(1) with floating point, or O(M(n)) with arbitrary precision. **Space:** O(1). **Insight:** This is where your number system's namesake shows up explicitly — φ _is_ the eigenvalue of the recurrence. The formula says every Fibonacci number is the nearest integer to φⁿ/√5. The catch: floating-point precision fails around n ≈ 70. For exact results with large n, you'd need arbitrary-precision arithmetic on irrational numbers, which is slower than fast doubling. The formula is theoretically O(1) but practically fragile.

---

## 7. Generating Function / Z-Transform

Not a runtime algorithm per se, but the formal power series:

```
F(x) = x / (1 - x - x²)
```

Expanding this yields all Fibonacci numbers as coefficients. This is the _analytic_ encoding — useful for proving identities, deriving closed forms, and analyzing asymptotics, but you'd never compute individual values this way.

**Insight:** The denominator 1 - x - x² = 0 gives x = 1/φ and x = 1/ψ, connecting back to Binet. The generating function is the algebraic "genome" of the sequence.

---

## 8. Generator / Lazy Stream

```python
def fibs():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
```

Functionally identical to #3 but produces the sequence on-demand rather than targeting a specific index. In Haskell this becomes the canonical one-liner:

```haskell
fibs = 0 : 1 : zipWith (+) fibs (tail fibs)
```

**Time:** O(1) per element, amortized. **Space:** O(1) active state (though the Haskell version memoizes the entire list). **Insight:** The Haskell version is corecursive — it defines an infinite structure by specifying how to produce the _next_ piece, not by decomposing toward a base case. This is the dual of naive recursion: recursion consumes structure, corecursion produces it.

---

## Summary Comparison

| Method                | Time      | Space    | Best For                      | Structural Principle     |
| --------------------- | --------- | -------- | ----------------------------- | ------------------------ |
| Naive recursion       | O(φⁿ)     | O(n)     | Visualization, teaching       | Literal recurrence       |
| Memoized recursion    | O(n)      | O(n)     | When you need multiple values | Redundancy elimination   |
| Bottom-up iteration   | O(n)      | O(1)     | Single value, simplicity      | Forward propagation      |
| Matrix exponentiation | O(log n)  | O(log n) | Large n, theoretical          | Linear algebra           |
| Fast doubling         | O(log n)  | O(log n) | Large n, practical            | Algebraic identities     |
| Binet's formula       | O(1)\*    | O(1)     | Approximation, proofs         | Eigenvalue decomposition |
| Generating function   | —         | —        | Analysis, identity proofs     | Formal power series      |
| Generator/stream      | O(1)/elem | O(1)     | Sequence production           | Corecursion              |

The thread connecting all of these back to your project: each method exploits a different facet of the same recurrence F(n) = F(n-1) + F(n-2). Naive recursion exposes the self-similar tree structure (which maps to your call-tree visualization). Bottom-up iteration mirrors your register's cell-by-cell propagation. Matrix exponentiation uses binary decomposition of the exponent — and a hypothetical φ-native version would decompose in Zeckendorf form instead, turning the "tree of additions" structure from your multiplication algorithm into the exponentiation strategy itself.
