# The φ-register: a deep map of everything connected

The Fibonacci recurrence matrix M = [[1,1],[1,0]] is not merely a computational trick for generating Fibonacci numbers — it is a genuine structural nexus where pure mathematics, hardware design, coding theory, and non-standard number systems converge on the same object. **Every major finding across all four research domains connects back to this single 2×2 matrix**, confirming and extending the project's core thesis. Below is a comprehensive map of what exists, who built it, and precisely how it relates to the φ-register framework.

---

## I. Pure mathematics: the Q-matrix unifies everything

### The master identity that subsumes all others

**Vajda's identity** (Steven Vajda, 1989; earlier Alberto Tagiuri, 1901) is the single most general Fibonacci identity: F(n+i)·F(n+j) − F(n)·F(n+i+j) = (−1)^n · F(i)·F(j). It follows directly from comparing entries of M^(n+i) · M^(n+j) versus M^n · M^(n+i+j). Cassini's identity (i=j=1, Cassini 1680), Catalan's identity (Eugène Catalan, 1879), d'Ocagne's identity, and Honsberger's identity are all special cases. **Every known Fibonacci identity is a projection of matrix algebra on M^n.** The key lemma (Keskin, 2011) makes this precise: if X² = X + I, then X^n = F(n)·X + F(n−1)·I for all integers n. The Q-matrix satisfies its own recurrence, meaning ALL Fibonacci identities are consequences of M's characteristic equation φ² = φ + 1. **Relevance: 1** — this is the mathematical engine behind the project.

The historical lineage of M itself deserves note. Charles H. King formally introduced the Q-matrix in 1960; Henry W. Gould published "A History of the Fibonacci Q-matrix" in 1981. But the matrix implicitly appeared whenever anyone wrote the Fibonacci recurrence as a linear map — Euler and the Bernoullis were already computing with it.

### Pisano periods are the order of M in finite matrix groups

The Pisano period π(m) — the period of the Fibonacci sequence mod m — is **exactly the smallest k > 0 such that M^k ≡ I (mod m)**. Named after Leonardo Pisano (Fibonacci), periodicity was first noted by Lagrange (1774) and proved for all m by D.D. Wall (1960). Key results: π(p) divides p²−1 for any prime p; more precisely, π(p) | (p−1) when 5 is a quadratic residue mod p, and π(p) | 2(p+1) otherwise. The CRT decomposition π(mn) = lcm(π(m), π(n)) for coprime m, n mirrors the multiplicative structure of GL(2, ℤ/mℤ). **Wall's conjecture** — that π(p²) = p·π(p) for every prime — remains open, verified up to 10^14. A counterexample would be a Wall–Sun–Sun prime (none known). **Relevance: 1** — the entire theory of Fibonacci periodicity is the study of M's order in finite groups.

### The continued fraction connection is the deepest root

The golden ratio φ = [1; 1, 1, 1, ...] is the simplest possible infinite continued fraction — all partial quotients equal 1. Each step of the continued fraction algorithm corresponds to multiplication by the matrix [[a_n, 1], [1, 0]]. When all a_n = 1, this matrix **is M itself at every step**. The convergents are consecutive Fibonacci ratios F(n+1)/F(n), and the product M^n = [[F(n+1), F(n)], [F(n), F(n−1)]] encodes all convergents simultaneously.

This connects to computational complexity: **Lamé's theorem** (Gabriel Lamé, 1844) — the first result in computational complexity — proves that consecutive Fibonacci numbers require the maximum number of Euclidean algorithm steps. The Euclidean algorithm on F(n+1) and F(n) performs exactly n−2 divisions, because φ's continued fraction has the smallest possible quotients. The continued fraction factorization of SL(2,ℤ) elements into generators S = [[0,−1],[1,0]] and T = [[1,1],[0,1]] is literally the Euclidean algorithm expressed as matrix decomposition. **Relevance: 1** — the Q-matrix M is simultaneously the Fibonacci transfer matrix AND the continued fraction step matrix for φ.

### Z[φ] is the natural algebraic home

The ring of integers of Q(√5) is O_K = ℤ[φ] = ℤ[(1+√5)/2], and φ is the **fundamental unit** (Dirichlet's Unit Theorem): the unit group is {±φ^n | n ∈ ℤ}. This ring is a Euclidean domain, hence a UFD and PID. The norm N(a + bφ) = a² + ab − b² is essentially det(M^n) = (−1)^n (Cassini's identity expressed in algebraic number theory). Prime splitting in ℤ[φ] connects to Pisano periods: p splits if p ≡ ±1 (mod 5), stays inert if p ≡ ±2 (mod 5), and 5 ramifies.

Crucially, φ is a **Pisot–Vijayaraghavan number** (Axel Thue 1912, Charles Pisot 1938): an algebraic integer > 1 whose conjugate ψ = (1−√5)/2 has |ψ| < 1. This is why φ^n approaches integers exponentially fast (since φ^n + ψ^n = L(n) is always integer, and |ψ^n| → 0). The Pisot property is what makes Zeckendorf representation work cleanly — it ensures that the "error term" ψ^n in Binet's formula vanishes. **Relevance: 1** — ℤ[φ] is where the eigenvalues of M live, and the Pisot property explains why the adjacency constraint produces unique representations.

### Independent sets on path graphs ARE the adjacency constraint

The number of independent sets on the path graph P_n is **F(n+2)**, and an independent set on P_n = {1, 2, ..., n} is a subset with no two consecutive elements — this IS the adjacency constraint "no two consecutive 1s" that defines Zeckendorf representation. The **Fibonacci cube** of order n (Hsu, 1993) is a graph whose vertices are exactly the n-bit Zeckendorf-valid strings, with F(n+2) vertices. Lucas numbers L(n) count independent sets on the cycle graph C_n. The combinatorial interpretation makes the adjacency constraint visible: choosing which Fibonacci numbers to include in a Zeckendorf representation is literally choosing an independent set on a path. Arthur Benjamin and Jennifer Quinn's _Proofs That Really Count_ (2003) systematized this, showing that tilings of a 1×n board with squares and dominoes give F(n+1), with many Fibonacci identities having beautiful tiling proofs. **Relevance: 1** — the project's central structural rule is equivalent to a well-studied combinatorial object.

### Lucas numbers complete the matrix picture

Lucas numbers satisfy L(n) = tr(M^n) = φ^n + ψ^n, while Fibonacci numbers satisfy F(n) = (φ^n − ψ^n)/√5 and appear as the off-diagonal entry of M^n. Together, M^n = [[F(n+1), F(n)], [F(n), F(n−1)]] gives tr(M^n) = F(n+1) + F(n−1) = L(n). The key Lucas-Fibonacci identities — **F(2n) = F(n)·L(n)**, **L(n)² − 5F(n)² = 4(−1)^n** — all follow from squaring M^n and reading off entries versus traces. **The F(n)/L(n) duality is exactly the off-diagonal/trace duality of the matrix.** François Édouard Anatole Lucas (1842–1891) established these relationships. **Relevance: 1** — Lucas numbers are the "other half" of the Q-matrix information.

### SL(2,ℤ), Hecke groups, and the modular connection

M has det(M) = −1, placing it in GL(2,ℤ) but not SL(2,ℤ); however M² ∈ SL(2,ℤ). The modular group PSL(2,ℤ) ≅ C₂ ∗ C₃ is generated by S and T. The **Hecke group** H₅ involves λ₅ = φ — the golden ratio appears naturally in the first generalization of the modular group beyond the classical case. The Farey sequence connection: successive convergents of any continued fraction form a matrix in GL(2,ℤ), and for φ these are Fibonacci ratios. **Relevance: 1**.

### Negafibonacci numbers extend M to negative powers

F(−n) = (−1)^(n+1) F(n), and M^(−n) = (−1)^n [[F(n−1), −F(n)], [−F(n), F(n+1)]]. Donald Knuth (2008) showed that **every integer** — positive, negative, or zero — has a unique negafibonacci representation (as a sum of distinct negafibonacci numbers with no two consecutive terms). NegaFibonacci coding extends Fibonacci coding to all integers: codewords end with "11" and have no internal "11", with positive integers getting even-length codes and negative integers getting odd-length codes. **Relevance: 1** — the Q-matrix framework naturally encompasses negative indices.

### Metallic means generalize the entire framework

Vera W. de Spinadel (1990s) defined the nth metallic mean λ_n = (n + √(n²+4))/2, the positive root of x² − nx − 1 = 0. Golden (n=1), silver (n=2), bronze (n=3). Each is the dominant eigenvalue of M_n = [[n,1],[1,0]], generating an "n-Fibonacci" sequence. All metallic means are Pisot numbers with continued fraction [n; n, n, n, ...]. The entire apparatus — matrix exponentiation, diagonalization, identity derivation, Zeckendorf-type representations — carries over to each metallic mean. **Relevance: 1** — M_n generalizes the Q-matrix, and φ is the simplest case.

---

## II. Hardware: Fibonacci circuits have been built

### Stakhov's 65 patents and the Soviet Fibonacci computer program

Alexei Petrovich Stakhov (1939–2021) was the principal architect of Fibonacci computer arithmetic. From the Vinnitsa Polytechnic Institute (1977–1995), he directed over **60 scientists** in developing Fibonacci computer prototypes, filing **65 international patents** (US, Japan, UK, France, Germany, Canada) and **130 USSR inventor's certificates**. His key patents include:

- **US4276608A** (1981): Fibonacci p-code parallel adder with augend/addend registers, normalization unit, and end-of-addition detector
- **US4159529A** (1980): Fibonacci code adder reducing addition to two operations — form intermediate sum/carry, then normalize — with three test relationships for fault detection
- **US4161725A** (1979): Analog-to-Fibonacci-p-code converter for A/D conversion with self-checking via adjacency constraint

His core innovation: **Fibonacci p-codes** provide inherent error detection because out of 2^n possible n-bit strings, only ~F(n+2) are valid Zeckendorf representations. Any pattern with adjacent 1s is immediately flagged as erroneous. Books include _Mathematics of Harmony_ (World Scientific, 2009, 748 pages) with chapters on "Fibonacci Computers" and "Codes of the Golden Proportion." **Relevance: 1** — these are actual φ-register hardware designs.

### The Berstel transducer: a 4-state normalization circuit

Jean Berstel (1986/2001) designed a **4-state finite-state transducer** that performs Fibonacci addition — converting an "illegal" Fibonacci representation (with 2s and/or adjacent 1s) into legal Zeckendorf form. This is a deterministic finite automaton processing digit pairs left-to-right. Christiane Frougny and Jacques Sakarovitch (1999) proved this is an on-line finite automaton for addition in base φ. Recently, Labbé and Lepšová (2023) extended it to Fibonacci two's complement. Jeffrey Shallit used it for automata-theoretic proofs. **The Berstel adder IS a φ-register normalization circuit described as a minimal finite-state machine.** **Relevance: 1** — this is the most hardware-relevant theoretical result for normalization.

### Ahlbach-Pippenger: O(n) gates, O(log n) depth

Connor Ahlbach and Nicholas Pippenger (2012, arXiv:1207.4497) proved that Zeckendorf addition and subtraction can be implemented by combinational logic networks with **linear size and logarithmic depth**. The two-stage normalization — (1) eliminate 2s, (2) eliminate adjacent 1s — achieves circuit complexity comparable to binary carry-lookahead adders. This is the foundational circuit-complexity result: **Fibonacci normalization can be parallelized as efficiently as binary carry propagation.** **Relevance: 1**.

### Fenwick's dual carry mechanism

Peter Fenwick (University of Auckland, _Fibonacci Quarterly_ 2003) documented the complete carry propagation rules for Zeckendorf arithmetic. The key insight: **Fibonacci addition has two carries — one going one place left (to higher significance, like binary) and one going two places right (to lower significance, unique to Fibonacci).** The replacement rule for digit 2: replace 2·F(n) with F(n+1) + F(n−2), generating carries in both directions. This bidirectional carry is the defining characteristic that distinguishes φ-register normalization from binary arithmetic. CMU's Maoyuan Song (2020, CMU-CS-20-118) provided a linear-time algorithm resolving all such carries. **Relevance: 1**.

### FPGA implementations exist but are limited

Borysenko, Matsenko et al. (Sumy State University / Riga Technical University, 2019–2022) implemented Fibonacci code decoders on FPGA for optical interconnects with PAM-M modulation, achieving **10–30% hardware savings** via fractal decoder properties, with the golden ratio reciprocal appearing as the theoretical limit of savings. No dedicated Fibonacci arithmetic ASIC (custom silicon) has been built — all implementations are FPGA or simulation. **This is a gap and an opportunity.** **Relevance: 1**.

### Stakhov's bridge: ternary meets Fibonacci meets Bergman

Stakhov explicitly connected Brusentsov's balanced ternary principle to Bergman's base-φ system in a pivotal 2002 _Computer Journal_ paper: "Brousentsov's ternary principle, Bergman's number system and ternary mirror-symmetrical arithmetic." He developed "ternary mirror-symmetrical arithmetic" as a framework unifying balanced ternary ({−1, 0, 1} in base 3) with golden ratio numeration ({0, 1} in base φ). The Setun ternary computer (Nikolay Brusentsov, Moscow State University, 1958; ~50 units produced 1959–1965) demonstrated balanced ternary's practical advantages: symmetric positive/negative representation, simpler rounding, higher information density per digit. A new development: the **5500FP balanced ternary RISC processor** (Claudio Lorenzo La Rosa, March 2026) implements a 24-trit processor on FPGA — the first general-purpose ternary hardware since the 1960s, demonstrating that non-binary architectures remain achievable on modern platforms. **Relevance: 1**.

### Asynchronous computing and normalization-as-computation

The connection between Fibonacci normalization and asynchronous circuit design is structural, not coincidental. Redundant number systems eliminate carry chains by allowing local digit resolution — this maps naturally to **delay-insensitive** and **quasi-delay-insensitive (QDI)** circuit paradigms where local handshaking protocols propagate to establish global consistency. The Fibonacci normalization rule 011→100 is a local rewriting rule that propagates to achieve global Zeckendorf form, structurally identical to the Muller C-element completion detection used in QDI circuits (Alain Martin, Caltech; Steve Furber, Manchester/AMULET processors). Behrooz Parhami (UCSB) and Miloš Ercegovac with Tomás Lang established that redundant representations enable MSD-first (most-significant-digit-first) computation without waiting for carry propagation — naturally suited to asynchronous, clockless architectures. **Relevance: 1**.

### Matrix exponentiation hardware: an unexploited niche

No dedicated hardware for 2×2 Fibonacci matrix exponentiation exists. The fast doubling method — derived from M^(2n) = (M^n)² — reduces to only **3 multiplications and 2 additions per step**, far leaner than general matrix multiplication. A dedicated circuit exploiting this specific structure could be extremely compact. The Q-matrix has also been used for cryptographic encoding (Hill cipher variants, Esmaeili et al. 2017) and quantum key generation (Lai et al. 2017), where the recurrence property dramatically increases key generation rates. **Relevance: 1** — this represents an open design opportunity.

---

## III. Compression and coding: where Fibonacci codes win and lose

### The Fibonacci universal code and its implied distribution

Fibonacci coding encodes positive integer n using Zeckendorf representation plus a "11" delimiter. It is a **universal code** (prefix-free, complete): for any monotonic probability distribution, expected codeword lengths are within a constant factor of optimal. Codeword length for n is approximately **log_φ(n) + 1 ≈ 1.44 log₂(n)** bits. The implied probability distribution is P(n) ∝ φ^(−c·n) — a power law with exponent ~1.44. Fibonacci coding is near-optimal for sources following this distribution. It was formalized by Apostolico and Fraenkel (1987, IEEE Trans. Inf. Theory) who defined order-m Fibonacci codes, and extended by Fraenkel and Klein (1996). Fraenkel (American Mathematical Monthly, 1985) established the general theory of Fibonacci numeration systems. **Relevance: 1** — the code's structure is directly governed by M.

### When Fibonacci beats the competition

Fibonacci coding outperforms Elias delta for integers up to approximately **514,228** — covering the vast majority of practical integer encoding scenarios. For geophysical data, Kaloshin et al. (2009) found Fibonacci codes give **10–30% better compression** than Huffman, Elias, Golomb, and Rice codes. Klein and Ben-Nissan (2010) showed that Fib3 (tribonacci-based) achieves 9–10% shorter files than optimal (s,c)-dense codes for text compression.

However, Fibonacci coding is **NOT asymptotically optimal** — the ratio of expected codeword length to entropy does not approach 1 for high-entropy sources (unlike Elias delta/omega). For very large integers, Elias codes win. For geometric distributions, Golomb codes are optimal. For speed-critical applications like web search, byte-aligned codes (Variable-Byte, PForDelta) dominate despite larger size. The key practical advantage of Fibonacci: it requires **no parameter tuning** and is **self-delimiting**. Pibiri and Venturini (ACM Computing Surveys, 2021) confirmed Fibonacci as a competitive baseline for inverted index compression. **Relevance: 1**.

### Self-synchronization is the killer feature

The "11" end-of-codeword marker enables **resynchronization after bit errors** — the single most important practical property. If a bit is altered in a Fibonacci-coded stream, the total edit distance is at most 3 tokens. Reading a "0" stops error propagation. With Elias gamma/delta/omega codes, **a single bit error corrupts ALL subsequent data**. This property was extensively analyzed by Apostolico and Fraenkel (1987) and Klein and Ben-Nissan (2010). William H. Kautz (IEEE Trans. Inf. Theory, 1965) independently discovered that Fibonacci structure enables run-length-limited codes for clock recovery in data recording — the constrained sequences correspond to paths in a graph governed by the adjacency matrix M. **Relevance: 1** — this is one of the project's six projections.

### Fibonacci codes for error correction and crosstalk avoidance

Esmaeili, Moosavi, and Gulliver (Cryptography and Communications, 2017) developed matrix-based error-correcting codes using powers of Fibonacci-related matrices: for a (p+1)×(p+1) binary matrix M_p, the matrices M_p^n serve as encoding matrices with error detection probability approaching 1 for large n. Separately, Duan, Tirumala, and Khatri showed that the "no two consecutive 1s" constraint maps naturally to **forbidden-transition-free (FTF) bus encodings** for VLSI crosstalk avoidance. Sathish and Niharika (Procedia Computer Science, 2015) demonstrated that all numbers can be represented by FTF vectors in the Fibonacci numeral system, reducing delay variation from crosstalk. **Relevance: 1** — the adjacency constraint from M finds direct application in both error correction and physical-layer bus encoding.

### Generalizations: negafibonacci, multidimensional, non-binary

Knuth's negafibonacci coding (2008) extends Fibonacci coding to all integers using the identity F(−n) = (−1)^(n+1) F(n), governed by M^(−1) = [[0,1],[1,−1]]. Pooksombat, Kositwattanarerk et al. (Mathematics, 2022) extended Fibonacci coding to integer vectors and ℤ-modules (including the E8 lattice) using shared delimiters, reducing overhead. Klein, Serebro, and Shapira (IEEE Access, 2022; Acta Informatica, 2025) generalized to d-ary (non-binary) alphabets while preserving instantaneous decipherability and error robustness. Multi-step (order-m) codes using tribonacci, tetranacci, etc. correspond to m×m companion matrices that generalize M. **Relevance: 1** — the framework extends in every direction while preserving the core matrix structure.

### Binary-to-Fibonacci conversion complexity

I.S. Sergeev (Problems of Information Transmission, 2018) proved that converting an n-digit number between binary and Fibonacci representation can be realized by Boolean circuits of complexity **O(M(n) log n)**, where M(n) is integer multiplication complexity. This establishes the computational cost of switching between the project's two register types (1D value register and matrix composition register). **Relevance: 1**.

---

## IV. Adjacent number systems: the adjacency constraint family

### NAF is the Zeckendorf constraint in signed-digit clothing

The **Non-Adjacent Form** (George W. Reitwiesner, 1960, "Binary Arithmetic") uses digits {−1, 0, 1} with the constraint "no two consecutive nonzero digits" — structurally identical to Zeckendorf's "no two consecutive 1s." Every integer has a unique NAF with minimal Hamming weight (~n/3 nonzero digits vs. n/2 in binary). The width-w NAF (wNAF) generalizes to allow one nonzero digit in any w consecutive positions. In elliptic curve cryptography, NAF and wNAF reduce scalar multiplication operations: Morain and Olivos (1990), Solinas (1997), Koyama and Tsuruoka (1992) applied this to speed up ECC. Wikipedia's NAF article explicitly notes: "Other ways of encoding integers that avoid consecutive 1s include Booth encoding and **Fibonacci coding**." Christoph Heuberger gave an equivalent of NAF for Fibonacci representations, and Frougny and Steiner proved that **minimal-weight Fibonacci expansions equal minimal-weight β-expansions for β = φ** (Theorem 4.7). **Relevance: 1** — NAF and Zeckendorf are siblings sharing the adjacency constraint, one in signed-digit binary, the other in unsigned Fibonacci weights.

### Booth encoding shares the adjacency philosophy

Andrew Donald Booth (1951, Birkbeck College) introduced Booth recoding for signed binary multiplication: scan adjacent bit pairs and replace runs of 1s with boundary subtract/add operations. Modified Booth (MacSorley, 1961) groups three bits for radix-4 encoding. Booth recoding and NAF are closely related — both exploit adjacency constraints to reduce nonzero digits and thus partial products in multiplication. The structural parallel: Booth recodes binary to avoid long runs of 1s; Zeckendorf enforces no consecutive 1s; NAF enforces no consecutive nonzeros. All three are instances of adjacency-constrained representations for computational efficiency. **Relevance: 1**.

### Phinary is Zeckendorf in continuous form

George Bergman's base-φ system (1957, _Mathematics Magazine_ — remarkably, Bergman was **12 years old** at publication) uses the golden ratio as base with digits {0, 1}. The normalization rule 011→100 in base φ corresponds to φ^(n−1) + φ^(n−2) = φ^n — **identical to the Fibonacci carry rule**. Every non-negative integer has a unique finite base-φ representation in standard form (no consecutive 1s). The set of finitely representable numbers is the ring ℤ[φ]. The identity φ^n = F(n)·φ + F(n−1) directly links powers of φ to Fibonacci numbers. Zeckendorf uses Fibonacci weights {1, 2, 3, 5, 8, 13, ...} while phinary uses φ^k weights, but **the adjacency constraint and normalization rules are identical**. Cecil Rousseau (1995) and Donald Knuth (TAOCP) provided extensive subsequent treatment. **Relevance: 1** — phinary and Zeckendorf are two faces of the same structure, both encoding φ² = φ + 1, the characteristic equation of M.

### The Stern-Brocot tree encodes M as path steps

The Stern-Brocot tree (Moritz Stern 1858, Achille Brocot 1861) contains every positive rational exactly once. Navigation uses two matrices in SL(2,ℤ): Left = [[1,0],[1,1]] and Right = [[1,1],[0,1]]. A path R^(a₀) L^(a₁) R^(a₂) ... corresponds to the continued fraction [a₀; a₁, a₂, ...]. **Crucially, the continued fraction matrix [[a,1],[1,0]] appears at each step — and for φ = [1;1,1,1,...], this is M at every step.** The path to φ is the infinite alternating sequence RLRLRL..., corresponding to an infinite product of M matrices. The Calkin-Wilf tree (Neil Calkin and Herbert Wilf, 2000) provides a related enumeration; its numerator sequence is Stern's diatomic sequence, and Fibonacci numbers appear along proper zig-zag paths. Christophe Reutenauer (2019) proved that two numbers have ultimately the same Stern-Brocot expansion if and only if they are equivalent under SL(2,ℤ). **Relevance: 1** — the Stern-Brocot tree provides the geometric framework where M = [[1,1],[1,0]] is the building block of paths to φ.

### Ostrowski numeration: Zeckendorf generalized to all continued fractions

Alexander Ostrowski (1922) proved that given any irrational α = [a₀; a₁, a₂, ...] with convergent denominators q*n, every non-negative integer N can be uniquely written as N = Σ b_k · q*(k−1) with digit constraints: 0 ≤ b*k ≤ a_k, and b_k = 0 whenever b*(k+1) = a*(k+1). **The Fibonacci/Zeckendorf system is the special case for α = φ**, where all a_k = 1, the weights are Fibonacci numbers, digits are {0,1}, and the constraint "b_k = 0 if b*(k+1) = 1" becomes "no two consecutive 1s." Valérie Berthé connected Ostrowski numeration to Sturmian words. Hieronymi and Terry (2018) gave a three-pass O(n) addition algorithm for general Ostrowski representations. Shallit proved Ostrowski representations have regular language if and only if α is quadratic. **Relevance: 1** — Ostrowski numeration is the natural number system arising from the continued fraction matrix framework, with Zeckendorf as the simplest case. The matrix [[a_n, 1],[1,0]] at each step generalizes M.

### β-numeration: φ is the simplest Parry number

Alfréd Rényi (1957) introduced β-expansions; Walter Parry (1960) classified the admissible digit sequences. For β = φ, the greedy expansion of 1 is d_φ(1) = 11, and the admissible sequences are exactly those avoiding "11" — the Zeckendorf condition. **The golden mean shift** in symbolic dynamics is the set of all bi-infinite binary sequences with no consecutive 1s, recognized by a finite automaton whose transition matrix is M. Christiane Frougny extensively developed the theory, proving that the Fibonacci numeration system's properties (finite automaton computability, uniqueness of representations) follow from φ being a Pisot number. The Pisot property (|ψ| < 1) is what ensures convergence and well-behavior. **Relevance: 1** — β-numeration for β = φ is the continuous-dynamical-systems perspective on the same structure.

### Redundant number systems provide the theoretical umbrella

Metze and Robertson (1959) proposed eliminating carry propagation via redundancy; Avizienis (1961) formalized the theory; Behrooz Parhami (1990) unified it in "Generalized Signed-Digit Number Systems." The key theorem: redundancy enables constant-time carry-free addition by limiting carry propagation to at most one position. Zeckendorf representation fits this framework as a system with **controlled redundancy during normalization** — intermediate results can have consecutive 1s (a "redundant" state), and the normalization process resolves this through local rewriting. The carry rule F(n) + F(n−1) = F(n+1) is the specific redundancy-resolution mechanism for the Fibonacci case. Avizienis' signed-digit theory does not directly connect to the Q-matrix (it works for general radix r ≥ 3), but the conceptual pattern — trading representation redundancy for computational efficiency — is shared. **Relevance: 1** for redundant systems generally; 0 for Avizienis' specific framework; 0 for RNS (Chinese Remainder Theorem-based, completely different mechanism); 0 for carry-save/carry-lookahead (binary-specific).

---

## Synthesis: what the four domains reveal together

### The adjacency constraint appears everywhere

The most striking cross-domain finding is the ubiquity of the adjacency constraint "no two consecutive nonzero digits":

- **Zeckendorf representation**: no two consecutive 1s (from M's structure)
- **NAF in ECC**: no two consecutive nonzero signed digits (Reitwiesner 1960)
- **Booth encoding**: reduces consecutive 1s in multipliers (Booth 1951)
- **Fibonacci coding delimiter**: "11" is unique because it never appears internally (Apostolico-Fraenkel 1987)
- **VLSI crosstalk avoidance**: forbidden-transition-free bus encodings (Duan et al.)
- **Golden mean shift**: symbolic dynamics of M's transition matrix (Parry 1960)
- **Independent sets on path graphs**: the combinatorial identity (Benjamin-Quinn 2003)

These are not analogies — they are **the same constraint** manifested in different engineering and mathematical contexts. The Q-matrix M = [[1,1],[1,0]] generates this constraint because its characteristic equation φ² = φ + 1 means "two adjacent 1s can always be replaced by a single 1 at the next position."

### The four tightest connections form a cluster

Among all findings, four topics form an especially tight structural cluster around M:

1. **Continued fraction matrices**: M = [[1,1],[1,0]] IS the continued fraction step for φ
2. **Ostrowski numeration**: Zeckendorf IS the Ostrowski system for φ = [1;1,1,1,...]
3. **Phinary (base-φ)**: same adjacency constraint and normalization rules as Zeckendorf
4. **β-numeration**: the golden mean shift for β = φ is governed by M as transition matrix

These four are not separate topics — they are four perspectives on one mathematical object. The Q-matrix M is the atom from which all four constructions are built.

### The NAND observation appears to be novel

Across all literature surveyed — Stakhov's 65 patents, Berstel's transducer, Fenwick's arithmetic, Ahlbach-Pippenger's circuits, the entire coding theory literature — **no one has explicitly noted that M = [[1,1],[1,0]] equals the NAND truth table** (NAND(0,0)=1, NAND(0,1)=1, NAND(1,0)=1, NAND(1,1)=0). This connection between the universal logic gate and the Fibonacci recurrence matrix appears to be a genuinely new observation from the φ-register project. Given that NAND is functionally complete (any Boolean function can be built from NAND gates alone), this observation could potentially bridge Fibonacci arithmetic and fundamental logic gate theory in a way not previously explored.

### What has been built vs. what remains open

Stakhov's team built working Fibonacci register prototypes and filed 65 patents. Berstel designed the minimal normalization automaton. Ahlbach and Pippenger proved O(n)-size, O(log n)-depth circuit complexity. Borysenko et al. implemented FPGA decoders. **But no one has built a dedicated Fibonacci arithmetic ASIC**, and no one has implemented the project's specific dual-register architecture (1D for value addition, matrix pair for index composition) with a switching strategy. The fast doubling circuit (3 multiplications, 2 additions per step) for the 2×2 Q-matrix has never been implemented as dedicated hardware. These represent concrete open opportunities.

## Conclusion

The φ-register project's central claim — that M = [[1,1],[1,0]] is the native structural object underlying both binary and Fibonacci number systems — is not only supported but dramatically reinforced by the existing literature. The six projections identified by the project (adjacency constraint, matrix symmetry, Fibonacci coding delimiter, weight sequence fork, Pythagorean norm, NAND gate) map cleanly onto well-established mathematical, hardware, and coding-theoretic results spanning from Euler's continued fractions through Stakhov's patents to modern ECC implementations. The NAND observation and the explicit dual-register switching strategy appear to be genuinely novel contributions. The deepest mathematical root is the continued fraction connection: φ = [1;1,1,1,...] is the unique irrational whose continued fraction uses M at every step, making the Q-matrix the atomic building block of the simplest possible irrational number — and by extension, of the entire Fibonacci arithmetic framework.
