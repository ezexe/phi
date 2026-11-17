# Where Fibonacci meets machine learning: a structural survey

**The φ-register's core properties — Zeckendorf sparsity, ternary normalization, redundant arithmetic, and Q-matrix recurrence — connect to real, published work across AI hardware, neural architecture design, and learning theory, though most intersections remain lightly explored.** The strongest established bridges run through ternary/quantized neural networks (where {−1,0,+1} weights are structurally identical to balanced ternary), Canonical Signed Digit accelerator designs (which enforce the exact same "no adjacent nonzeros" constraint as Zeckendorf), and neural cellular automata (where learned local rewrite rules mirror Fibonacci normalization). Several connections are classical information-theoretic results awaiting their first application to deep learning. What follows maps each avenue of this project to the literature, distinguishing established results from speculative territory.

---

## Zeckendorf sparsity has one direct paper and a structural gap worth filling

The most explicit bridge between Zeckendorf representation and deep learning is **Z pooling** (Vigneron, Maaref & Syed, _Entropy_ 2021), a CNN pooling operator that encodes features via non-consecutive Fibonacci weights. It replaces max pooling with a Zeckendorf-decomposition-based operator yielding rotation-invariant neighborhoods, evaluated on BSD500 and glioblastoma imaging with improvements over standard pooling. This directly exploits the unique decomposition property and adjacency constraint.

The structural gap is more interesting. NVIDIA's Ampere architecture supports **2:4 structured sparsity** (exactly 2 nonzeros per 4-element block), and the broader N:M sparsity literature is large (Hoefler et al., JMLR 2021, surveyed 300+ papers). Zeckendorf's "no consecutive 1s" constraint is formally a **(1,∞)-RLL constraint** — at most 1 nonzero in every 2 consecutive positions, enforcing ≤50% density. This is a different and potentially useful structured sparsity pattern, but **no paper frames Zeckendorf's constraint as a neural network pruning mask**. The lottery ticket hypothesis literature (Frankle & Carlin, ICLR 2019) likewise contains no Fibonacci-structured analysis. Whether "winning tickets" discovered by magnitude pruning tend toward non-consecutive patterns is an open empirical question.

A 2024 preprint fills part of this gap. **Fibonacci Codeword Quantization (FCQ)** (arXiv:2511.01921) rounds neural network weights to their nearest Fibonacci codeword — binary values containing no consecutive 1s — then exploits the resulting redundancy for lossless compression. Applied to neural radio receivers, FCQ combines with Incremental Network Quantization to mitigate accuracy loss while enabling two novel compression algorithms. This is the most direct demonstration that Fibonacci number properties deliver practical benefits in quantized neural networks.

---

## CSD accelerators enforce the same constraint as Zeckendorf and deliver real speedups

The deepest hardware connection runs through **Canonical Signed Digit (CSD) representation** — a ternary {−1,0,+1} system where no two adjacent digits are nonzero. This is mathematically identical to Zeckendorf's constraint transposed into signed-digit arithmetic. The payoff is concrete:

**CAxCNN** (Riaz et al., IEEE Access 2020) applies CSD approximation to CNN filter weights, achieving **77% logic area reduction** in multipliers with only 5.63% top-1 accuracy drop on VGG-16/ImageNet — no retraining required. **MA4C** (ScienceDirect 2022) converts pretrained CNN parameters to Minimum Signed Digit representations, delivering 4.2× reduction in FPGA logic circuits and 1.2× multiplier latency reduction versus 8-bit Booth. An RNS-CSD CNN accelerator at 22nm demonstrated **1.54× energy efficiency** and 1.88× throughput versus binary.

The CSD-RISC-V work (Reichenbach et al., Springer 2025) achieved **2.41× clock frequency increase** over standard binary in a RISC-V pipeline using CSD encoding, with >20% attributable to the encoding itself. The companion project "RISC-V3" built an entire datapath on redundant number systems.

Google's TPU patent (US20230015148A1, 2023) explicitly describes **carry-save adders** — a redundant binary representation — within systolic array MAC units, passing carries between MAC units in partially redundant form. This is redundant number arithmetic inside the most widely deployed AI accelerator architecture.

**The φ-register's normalization-as-settling maps cleanly to QDI neuromorphic circuits.** IBM's TrueNorth neurosynaptic core (ASYNC 2012) uses quasi-delay-insensitive design where circuits switch only during neural updates. SynSense's Speck SoC (_Nature Communications_) implements 4-phase handshake QDI dual-rail encoding throughout its asynchronous spiking pipeline at 0.42mW resting power. In both QDI circuits and Fibonacci normalization: no global clock exists, local operations resolve local constraint violations, the system signals completion when all constraints are satisfied, and settling time is data-dependent.

---

## Ternary weight networks sit exactly at the Stakhov bridge

**Ternary Weight Networks** (Li & Liu, 2016; ICASSP 2023) constrain weights to {−1,0,+1} — structurally identical to balanced ternary digits. TWN achieves 16× model compression with minimal accuracy loss; **Trained Ternary Quantization** (Zhu et al., ICLR 2017) refines this with asymmetric learned scaling. The field is mature for edge deployment, with memristor-based TWN architectures (IEEE 2024-2025) achieving 96% accuracy for in-memory computing.

Stakhov's foundational work provides the mathematical bridge this project needs. In "Brousentsov's Ternary Principle, Bergman's Number System and Ternary Mirror-Symmetrical Arithmetic" (_The Computer Journal_ 2002), he explicitly connects Brusentsov's balanced ternary (from the Soviet Setun computer) to Bergman's base-φ system. His ternary mirror-symmetrical arithmetic — developed for pipeline digital signal processors — derives directly from golden-proportion codes. His later _Computer Journal_ paper (2018) argues binary has an inherent "Trojan horse" vulnerability that Fibonacci p-codes resolve through built-in error detection via representation constraints.

The **5500FP** balanced ternary RISC processor (La Rosa, March 2026) implements a 24-trit data bus on FPGA using 2-bit-per-trit binary encoding internally. Coverage appeared in Hackaday and The Register. Its AI applications remain speculative — the processor is general-purpose — but a native ternary processor could execute TWN inference without encoding overhead. The separate "Triton TPU" ternary processing unit on GitHub is an independent effort in the same direction.

**The "Exact Neural Networks from Inexact Multipliers via Fibonacci Weight Encoding"** (IEEE/ACM DATE 2021) directly bridges these worlds: it retrains neural network weights to Fibonacci-encoded values (no consecutive 1s in binary), then uses an inexact multiplier that computes _exact_ results for Fibonacci-encoded inputs. On SqueezeNet 1.0 the accuracy degradation is just 0.4%, while the multiplier achieves **73% area reduction** and 43% power-delay-product reduction.

---

## The Q-matrix is a specific SSM that no one has studied as one

The Q-matrix M=[[1,1],[1,0]] is the companion matrix for x(n) = x(n-1) + x(n-2). Setting A=M, B=[1,0]ᵀ, C=[1,0] yields a state space model computing a "running Fibonacci" of the input signal. **This has not been explicitly noted in the S4/Mamba literature**, though the mathematical connection is immediate. S4 uses HiPPO matrices (DPLR form) chosen for long-range memory; the critical contrast is that the Q-matrix has eigenvalues **φ ≈ 1.618** and **−1/φ ≈ −0.618**, making it an _unstable_ system (|φ| > 1). SSMs require eigenvalues on or inside the unit circle. The Fibonacci recurrence blows up — which is a feature for number generation but disqualifying for signal processing without modification.

The matrix exponentiation trick (computing M^n in O(log n)) is analogous to the **parallel scan algorithms** used in linear RNN training (LRU, S5). Both exploit associativity of matrix multiplication. The diagonalization of M into eigenvalues φ and −1/φ parallels the Diagonal State Space approach. But the specific eigenvalue structure has not been studied in deep learning spectral theory — random matrix analyses of neural networks (Marchenko-Pastur, Wigner semicircle) have not produced the golden ratio as a significant spectral feature.

The **Fibonacci Network** (arXiv:2411.05052, 2024) is the one architecture that directly implements Fibonacci recurrence: each block takes input from the two previous blocks plus the original, mirroring f(n) = f(n-1) + f(n-2). Used for signal reconstruction, it outperforms positional encoding in noisy settings. Cassini's identity (det(M^n) = (−1)^n) has no established parallel in ML error detection.

---

## Fractal structures in deep learning are well-established but φ-disconnected

**FractalNet** (Larsson, Maire & Shakhnarovich, ICLR 2017) is the seminal self-similar architecture: its expansion rule f*{n+1}(z) = (f_n ∘ f_n)(z) ⊕ f₁(z) creates networks with subpaths of lengths 1, 2, 4, 8, 16 — achieving 22.85% error on CIFAR-100 matching ResNets without residual connections. The expansion is **binary doubling**, not Fibonacci recurrence (it merges two copies of f_n, not f*{n-1} and f*{n-2}). A genuine Fibonacci variant would merge f*{n-1} and f\_{n-2} — this has not been published.

The fractal dynamics literature is richer and more rigorous. **Multifractal loss landscapes** (Ly & Gong, _Nature Communications_ 2025) model neural network loss surfaces as multifractals with varying Hölder exponents, unifying clustered minima, edge-of-stability, and anomalous diffusion in a single framework. **Fractal trainability boundaries** (Sohl-Dickstein, arXiv 2024, Google DeepMind) show the boundary between successful and failed training in hyperparameter space has genuine Mandelbrot/Julia-set fractal structure. The **Hausdorff dimension of SGD trajectories** (Şimşekli et al., NeurIPS 2020) provides rigorous generalization bounds where heavier-tailed processes (lower fractal dimension) generalize better. None of these involve φ specifically — the fractal dimensions are empirical, not golden-ratio-valued.

---

## The golden mean shift gives you a clean information-theoretic foundation

The strongest classical result connecting Fibonacci to information theory: the number of binary strings of length n with no consecutive 1s is **F(n+2)**, giving the golden mean shift a channel capacity of exactly **log₂(φ) ≈ 0.694 bits/symbol**. This is computed directly from the Q-matrix's largest eigenvalue. MacKay's _Information Theory, Inference, and Learning Algorithms_ covers this in Chapter 17.

For a network layer of width n under the Zeckendorf sparsity constraint, at most **n · log₂(φ) ≈ 0.694n bits** of information can be encoded in the activation mask. This is a 30.6% capacity reduction from unconstrained binary — a precise, rigorous bound on the information cost of the adjacency constraint. Fibonacci codes for integers achieve rate ≈ 1/log₂(φ) ≈ **1.44 bits per source symbol** for geometric distributions, with the critical advantage of self-synchronization (a single bit error affects at most 3 codewords).

The unexplored connection: **PAC-Bayes compression bounds** (Lotfi et al., NeurIPS 2022) derive generalization bounds proportional to compressed model size in bits. The framework explicitly requires choosing a coding scheme — currently standard Huffman or arithmetic coding. Fibonacci coding of quantized weight indices would yield different (likely looser but more robust) bounds. The MDL framework (Hinton & Van Camp 1993) similarly involves description length choices where Fibonacci's self-synchronization might offer advantages in distributed or fault-prone settings. **No paper has explored either direction.**

---

## Neural cellular automata are your closest architectural analogue

**Growing Neural Cellular Automata** (Mordvintsev et al., Distill 2020) replaces hand-designed CA rules with learned differentiable update rules (~8,000 parameters). Each cell runs the same shared update function based on its local neighborhood — functionally equivalent to string rewriting rules. The 011→100 normalization rule operates on a 3-cell neighborhood; NCA update rules operate on similar local neighborhoods. Both compute through iterated local rule application until a stable state emerges.

This parallel deepens in three directions. **"A Path to Universal Neural Cellular Automata"** (Béna et al., GECCO 2025) trains NCA to perform general computation — matrix multiplication, transposition, rotation — through continuous cellular interactions, and even emulates a neural network solving MNIST _within_ the CA state. **Differentiable Logic Cellular Automata** (Google Research, Miotti et al., 2025) integrates differentiable logic gates with NCA, creating end-to-end differentiable self-organizing discrete systems — the exact class to which Fibonacci normalization's 011→100 belongs. **Neural Rewriting Systems** (Petruzzellis et al., arXiv July 2025) directly learn convergent term rewriting systems with modular components mirroring the match-select-apply cycle, outperforming GPT-4o on formula simplification — and Fibonacci normalization is precisely a convergent rewriting system (Church-Rosser: always terminates in Zeckendorf form regardless of rule application order).

The **Hopfield network parallel** is also structural: both Fibonacci normalization and Hopfield dynamics start from arbitrary states, apply local rules, and converge monotonically to canonical forms. Modern Hopfield networks (Ramsauer et al., 2020) connect to transformer attention, suggesting a chain from φ-register settling to attention-as-associative-memory.

---

## Fibonacci lattice sampling and hashing are quietly deployed

**Super-Fibonacci Spirals** (Alexa, CVPR 2022) extends golden-ratio-based Fibonacci spiral sampling from S² to SO(3) for rotation sampling in 3D vision. **Machine Learning on Spherical Fibonacci Lattices** (Song et al., _Physical Review Research_ 2022) uses the Fibonacci lattice — the most uniform spherical point distribution, derived from the golden angle — as substrate for GCN-based phase classification. The KIT/ISAS group has published multiple papers on generalized Fibonacci grids for optimal deterministic Gaussian sampling applicable to Bayesian filtering and quasi-Monte Carlo ML pipelines. Google's Deep Learning Tuning Playbook recommends quasi-random search using golden-ratio-based sequences for hyperparameter optimization.

**Fibonacci hashing** (Knuth's multiplicative method, using floor(2^w/φ)) is implemented in TensorFlow's hash table operations. The golden ratio's property as the "most irrational number" ensures optimal distribution of consecutive keys — a small but real deployment in ML infrastructure.

---

## What's substantive, what's speculative, and where the gaps are

Three tiers of connection emerge from this survey. **Established and deployed**: CSD/signed-digit accelerators enforcing Zeckendorf-like constraints (77% area reduction in CNNs), ternary weight networks, carry-save arithmetic in TPU systolic arrays, QDI neuromorphic circuits, Fibonacci lattice sampling, and the golden mean shift capacity result. **Published but niche**: Fibonacci weight encoding for hardware-efficient inference (DATE 2021), Z pooling, FCQ quantization, the Fibonacci Network architecture, Jaeger's golden-ratio learning rate derivation, and Neural Rewriting Systems. **Genuinely unexplored**: Zeckendorf-constrained pruning masks as a structured sparsity pattern, Fibonacci coding for PAC-Bayes generalization bounds, the Q-matrix as an SSM (requires stabilization), Fibonacci-variant FractalNet merging f*{n-1} and f*{n-2}, and the information-theoretic capacity cost (0.694 bits/symbol) as a neural network capacity bound.

The φ-register project sits at a genuine intersection. The "no adjacent nonzeros" constraint is already proven in silicon (CSD accelerators). The normalization-as-settling paradigm is already proven in neuromorphic hardware (QDI circuits). The rewrite-rule-as-computation framework is already proven in learned systems (NCA, NRS). What hasn't been done is connecting these established threads through a unified Fibonacci/golden-ratio framework — that appears to be the open contribution this project can make.
