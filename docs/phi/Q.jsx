import { useState, useMemo, useCallback, useRef, useEffect } from "react";

const MONO = "'JetBrains Mono', 'Fira Code', monospace";
const DISPLAY = "'Playfair Display', Georgia, serif";
const GOLD = "#d4a017";
const RED = "#e04040";
const GREEN = "#40c878";
const CYAN = "#40b8c8";
const PURPLE = "#a070e0";
const ORANGE = "#ffa028";
const BG = "#08080a";

const DOMAIN_COLORS = {
  math: GOLD,
  hardware: GREEN,
  coding: CYAN,
  systems: PURPLE
};

const DOMAIN_LABELS = {
  math: "PURE MATH",
  hardware: "HARDWARE",
  coding: "CODING",
  systems: "ADJACENT SYSTEMS"
};

// ═══════════════════════════════════════════════════════════════
// DATA: Every finding as a node
// ═══════════════════════════════════════════════════════════════

const FINDINGS = [
  // MATH
  {
    id: "vajda",
    domain: "math",
    name: "Vajda's Identity",
    rel: 1,
    year: "1901/1989",
    who: "Tagiuri (1901), Vajda (1989)",
    what: "F(n+i)·F(n+j) − F(n)·F(n+i+j) = (−1)^n · F(i)·F(j). Every known Fibonacci identity is a special case.",
    connection: "All identities are projections of M^n algebra. Cassini, Catalan, d'Ocagne — all subsumed.",
    links: ["cassini_norm", "lucas", "sl2z"]
  },
  {
    id: "pisano",
    domain: "math",
    name: "Pisano Periods",
    rel: 1,
    year: "1774/1960",
    who: "Lagrange (1774), D.D. Wall (1960)",
    what: "F(n) mod m repeats with period π(m) = smallest k where M^k ≡ I (mod m).",
    connection: "Periodicity IS the order of M in finite matrix groups. Wall's conjecture (π(p²)=p·π(p)) still open.",
    links: ["sl2z", "contfrac"]
  },
  {
    id: "contfrac",
    domain: "math",
    name: "Continued Fractions",
    rel: 1,
    year: "1844",
    who: "Euler, Lamé (1844)",
    what: "φ = [1;1,1,1...]. Each CF step multiplies by [[a_n,1],[1,0]]. For φ, this is M at every step.",
    connection: "M IS the CF step matrix for φ. Lamé's theorem: Fibonacci pairs are the worst case for Euclid's algorithm.",
    links: ["stern_brocot", "ostrowski", "zphi"]
  },
  {
    id: "zphi",
    domain: "math",
    name: "ℤ[φ] Ring",
    rel: 1,
    year: "1838/1938",
    who: "Dirichlet, Pisot (1938)",
    what: "Ring of integers of Q(√5). φ is the fundamental unit. Euclidean domain. Norm = det(M^n) = (−1)^n.",
    connection: "φ is a Pisot number (conjugate |ψ|<1), explaining why Zeckendorf representation works cleanly.",
    links: ["vajda", "phinary", "metallic"]
  },
  {
    id: "indep_sets",
    domain: "math",
    name: "Independent Sets",
    rel: 1,
    year: "1993/2003",
    who: "Hsu (1993), Benjamin & Quinn (2003)",
    what: "Valid Zeckendorf strings = independent sets on path graph P_n. Count = F(n+2). Fibonacci cube = the graph of valid states.",
    connection:
      "The adjacency constraint IS choosing an independent set. Tiling proofs give combinatorial Fibonacci identities.",
    links: ["golden_shift", "naf", "crosstalk"]
  },
  {
    id: "lucas",
    domain: "math",
    name: "Lucas Numbers",
    rel: 1,
    year: "1878",
    who: "François Édouard Anatole Lucas",
    what: "L(n) = tr(M^n) = φ^n + ψ^n. F(2n) = F(n)·L(n). L(n)² − 5F(n)² = 4(−1)^n.",
    connection: "Lucas = trace of M^n, Fibonacci = off-diagonal. They're the two independent readouts of one matrix.",
    links: ["vajda", "cassini_norm"]
  },
  {
    id: "sl2z",
    domain: "math",
    name: "SL(2,ℤ) & Modular Group",
    rel: 1,
    year: "1960/1981",
    who: "King (1960), Gould (1981), Hecke",
    what: "M ∈ GL(2,ℤ), M² ∈ SL(2,ℤ). Hecke group H₅ involves λ₅ = φ. Modular group PSL(2,ℤ) ≅ C₂∗C₃.",
    connection: "The Q-matrix lives in the modular group. Fibonacci ratios are Farey sequence convergents.",
    links: ["contfrac", "stern_brocot", "pisano"]
  },
  {
    id: "negafib",
    domain: "math",
    name: "Negafibonacci",
    rel: 1,
    year: "2008",
    who: "Donald Knuth",
    what: "F(−n) = (−1)^(n+1)F(n). Every integer (positive, negative, zero) has a unique negafibonacci representation.",
    connection: "M^(−n) extends the framework to negative indices. Same adjacency constraint applies.",
    links: ["negafib_code", "phinary"]
  },
  {
    id: "metallic",
    domain: "math",
    name: "Metallic Means",
    rel: 1,
    year: "1990s",
    who: "Vera W. de Spinadel",
    what: "λ_n = (n+√(n²+4))/2. Golden (n=1), silver (n=2), bronze (n=3). Eigenvalues of M_n = [[n,1],[1,0]].",
    connection: "The entire apparatus generalizes. φ is the simplest metallic mean.",
    links: ["zphi", "ostrowski"]
  },
  {
    id: "cassini_norm",
    domain: "math",
    name: "Cassini + Pythagorean",
    rel: 1,
    year: "1680",
    who: "Giovanni Cassini",
    what: "det(M^n) = (−1)^n (Cassini). F(k)²+F(k+1)² = F(2k+1) (Pythagorean). Two invariants on the state pair.",
    connection: "Project's 5th projection. Error detection in matrix form: if a²+b² ∉ Fibonacci, state is corrupt.",
    links: ["vajda", "lucas", "stakhov"]
  },

  // HARDWARE
  {
    id: "stakhov",
    domain: "hardware",
    name: "Stakhov's 65 Patents",
    rel: 1,
    year: "1977–2009",
    who: "Alexei Stakhov, Vinnitsa Polytechnic",
    what: "65 international patents, 130 USSR certificates. Fibonacci p-code adders, A/D converters, error-detecting registers.",
    connection: "Actual φ-register hardware. Adjacency constraint used for built-in error detection in every design.",
    links: ["berstel", "fenwick", "cassini_norm"]
  },
  {
    id: "berstel",
    domain: "hardware",
    name: "Berstel Transducer",
    rel: 1,
    year: "1986/2001",
    who: "Jean Berstel, Frougny & Sakarovitch",
    what: "4-state finite-state transducer that normalizes Fibonacci addition. Processes digit pairs left-to-right.",
    connection: "THE minimal normalization circuit. A φ-register normalizer described as a finite automaton.",
    links: ["ahlbach", "fenwick", "async"]
  },
  {
    id: "ahlbach",
    domain: "hardware",
    name: "O(n) Gates, O(log n) Depth",
    rel: 1,
    year: "2012",
    who: "Ahlbach & Pippenger",
    what: "Zeckendorf add/sub via combinational logic: linear size, logarithmic depth. Comparable to binary carry-lookahead.",
    connection: "Fibonacci normalization parallelizes as efficiently as binary carry propagation.",
    links: ["berstel", "fenwick", "redundant"]
  },
  {
    id: "fenwick",
    domain: "hardware",
    name: "Dual Carry Mechanism",
    rel: 1,
    year: "2003",
    who: "Peter Fenwick, U. Auckland",
    what: "Fibonacci has TWO carries: one left (+1 position) and one right (−2 positions). 2·F(n) → F(n+1)+F(n−2).",
    connection: "Bidirectional carry is what makes φ-register normalization fundamentally different from binary.",
    links: ["berstel", "ahlbach", "stakhov"]
  },
  {
    id: "fpga",
    domain: "hardware",
    name: "FPGA Implementations",
    rel: 1,
    year: "2019–2022",
    who: "Borysenko, Matsenko et al.",
    what: "Fibonacci decoders on FPGA for optical interconnects. 10–30% hardware savings. No dedicated ASIC exists yet.",
    connection: "Working silicon (FPGA). The gap: no one has built a dedicated Fibonacci arithmetic chip.",
    links: ["stakhov", "async"]
  },
  {
    id: "async",
    domain: "hardware",
    name: "Async/Clockless Computing",
    rel: 1,
    year: "1990s–2020s",
    who: "Martin (Caltech), Furber (AMULET), Parhami",
    what: "QDI circuits use local handshaking. Fibonacci's 011→100 rewrite IS a local-to-global settling process.",
    connection: "Normalization-as-computation maps structurally to delay-insensitive circuit paradigms.",
    links: ["berstel", "redundant", "ternary"]
  },
  {
    id: "ternary",
    domain: "hardware",
    name: "Ternary Hardware Revival",
    rel: 1,
    year: "1958/2026",
    who: "Brusentsov (Setun), La Rosa (5500FP, 2026)",
    what: "Balanced ternary {−1,0,1}. Setun: ~50 units built. 5500FP: 24-trit RISC on FPGA (March 2026).",
    connection: "Stakhov explicitly bridged ternary ↔ Fibonacci ↔ Bergman. Non-binary architectures are buildable.",
    links: ["stakhov", "phinary", "naf"]
  },

  // CODING
  {
    id: "fib_code",
    domain: "coding",
    name: "Fibonacci Universal Code",
    rel: 1,
    year: "1987",
    who: "Apostolico & Fraenkel",
    what: "Zeckendorf + '11' delimiter. Length ≈ 1.44·log₂(n). Universal, prefix-free, self-delimiting.",
    connection: "The code's structure is directly governed by M. The delimiter IS the matrix's [1,1] redundancy.",
    links: ["self_sync", "negafib_code", "inverted_idx"]
  },
  {
    id: "self_sync",
    domain: "coding",
    name: "Self-Synchronization",
    rel: 1,
    year: "1965/1987",
    who: "Kautz (1965), Apostolico-Fraenkel (1987)",
    what: "After a bit error, max edit distance = 3 tokens. Reading '0' stops error propagation. Elias codes: total corruption.",
    connection: "The killer practical feature. Falls directly from the adjacency constraint / delimiter property.",
    links: ["fib_code", "crosstalk", "indep_sets"]
  },
  {
    id: "inverted_idx",
    domain: "coding",
    name: "Search Index Compression",
    rel: 1,
    year: "2021",
    who: "Pibiri & Venturini, Klein & Ben-Nissan",
    what: "Fibonacci competitive for inverted index compression. Beats Elias delta for n < 514,228. No parameter tuning needed.",
    connection: "Practical deployment. Fibonacci codes are a competitive baseline for real-world search engines.",
    links: ["fib_code", "self_sync"]
  },
  {
    id: "crosstalk",
    domain: "coding",
    name: "VLSI Crosstalk Avoidance",
    rel: 1,
    year: "2015",
    who: "Duan et al., Sathish & Niharika",
    what: "'No consecutive 1s' = forbidden-transition-free bus encoding. Reduces delay variation from crosstalk.",
    connection: "The adjacency constraint applied to physical-layer bus design. Same rule, different substrate.",
    links: ["indep_sets", "self_sync", "stakhov"]
  },
  {
    id: "error_correct",
    domain: "coding",
    name: "Fibonacci Error Correction",
    rel: 1,
    year: "2017",
    who: "Esmaeili, Moosavi & Gulliver",
    what: "Powers of Fibonacci-related matrices as encoding matrices. Error detection probability → 1 for large n.",
    connection: "M^n used directly for error-correcting codes. The matrix IS the encoder.",
    links: ["cassini_norm", "fib_code", "stakhov"]
  },
  {
    id: "negafib_code",
    domain: "coding",
    name: "NegaFibonacci Coding",
    rel: 1,
    year: "2008",
    who: "Knuth",
    what: "Extends Fibonacci coding to all integers. Even-length = positive, odd-length = negative.",
    connection: "M^(−n) governs the encoding. Complete integer coverage with same delimiter property.",
    links: ["negafib", "fib_code"]
  },
  {
    id: "conversion",
    domain: "coding",
    name: "Binary↔Fibonacci Conversion",
    rel: 1,
    year: "2018",
    who: "I.S. Sergeev",
    what: "Conversion complexity: O(M(n)·log n) Boolean circuit, where M(n) = integer multiplication complexity.",
    connection: "Establishes the cost of switching between the project's two register representations.",
    links: ["ahlbach", "fib_code"]
  },

  // ADJACENT SYSTEMS
  {
    id: "naf",
    domain: "systems",
    name: "Non-Adjacent Form (NAF)",
    rel: 1,
    year: "1960",
    who: "George W. Reitwiesner",
    what: "Digits {−1,0,1}, constraint: no consecutive nonzero digits. Unique, minimal Hamming weight. Used in ECC.",
    connection: "Structurally identical constraint to Zeckendorf. Siblings: one signed-digit binary, one unsigned Fibonacci.",
    links: ["indep_sets", "booth", "golden_shift"]
  },
  {
    id: "booth",
    domain: "systems",
    name: "Booth Encoding",
    rel: 1,
    year: "1951",
    who: "Andrew Donald Booth",
    what: "Recode binary multiplier to reduce partial products. Scan adjacent pairs, replace runs of 1s.",
    connection: "Same adjacency philosophy. Booth, NAF, and Zeckendorf all exploit 'no consecutive nonzeros' for efficiency.",
    links: ["naf", "ahlbach"]
  },
  {
    id: "phinary",
    domain: "systems",
    name: "Phinary (Base-φ)",
    rel: 1,
    year: "1957",
    who: "George Bergman (age 12!)",
    what: "Base φ, digits {0,1}. Normalization: 011→100 = φ^(n-1)+φ^(n-2)=φ^n. Identical to Fibonacci carry.",
    connection: "Zeckendorf and phinary are two faces of the same structure. Both encode φ²=φ+1.",
    links: ["contfrac", "zphi", "golden_shift"]
  },
  {
    id: "stern_brocot",
    domain: "systems",
    name: "Stern-Brocot Tree",
    rel: 1,
    year: "1858/1861",
    who: "Moritz Stern, Achille Brocot",
    what: "Every positive rational exactly once. Navigation via L=[[1,0],[1,1]] and R=[[1,1],[0,1]] in SL(2,ℤ).",
    connection: "Path to φ is RLRLRL... = infinite product of M matrices. M is the building block of paths to φ.",
    links: ["sl2z", "contfrac", "ostrowski"]
  },
  {
    id: "ostrowski",
    domain: "systems",
    name: "Ostrowski Numeration",
    rel: 1,
    year: "1922",
    who: "Alexander Ostrowski",
    what: "Generalized Zeckendorf for any irrational α. Weights = CF denominators. Zeckendorf = special case for α=φ.",
    connection: "The natural number system from the CF matrix framework. M is the atomic case (all a_k=1).",
    links: ["contfrac", "stern_brocot", "metallic"]
  },
  {
    id: "golden_shift",
    domain: "systems",
    name: "Golden Mean Shift",
    rel: 1,
    year: "1960",
    who: "Walter Parry, Alfréd Rényi",
    what: "β-expansion for β=φ. Admissible sequences avoid '11'. The shift space's transition matrix IS M.",
    connection: "Symbolic dynamics perspective. The adjacency constraint is the grammar of the golden mean shift.",
    links: ["indep_sets", "phinary", "naf"]
  },
  {
    id: "redundant",
    domain: "systems",
    name: "Redundant Number Systems",
    rel: 1,
    year: "1959/1961",
    who: "Metze-Robertson, Avizienis, Parhami",
    what: "Redundancy enables carry-free addition. Zeckendorf fits as controlled redundancy during normalization.",
    connection: "Theoretical umbrella. Fibonacci normalization is a specific instance of redundancy resolution.",
    links: ["ahlbach", "async", "booth"]
  }
];

// Cross-domain connections for the web view
const CROSS_LINKS = [
  { from: "indep_sets", to: "naf", label: "Same constraint" },
  { from: "indep_sets", to: "golden_shift", label: "Same constraint" },
  { from: "indep_sets", to: "crosstalk", label: "Same constraint" },
  { from: "contfrac", to: "ostrowski", label: "Generalization" },
  { from: "contfrac", to: "stern_brocot", label: "Matrix paths" },
  { from: "cassini_norm", to: "stakhov", label: "Error detection" },
  { from: "cassini_norm", to: "error_correct", label: "M^n as encoder" },
  { from: "berstel", to: "async", label: "Local→global settling" },
  { from: "fib_code", to: "self_sync", label: "Delimiter property" },
  { from: "negafib", to: "negafib_code", label: "M^(-n) extension" },
  { from: "zphi", to: "phinary", label: "Same ring" },
  { from: "naf", to: "booth", label: "Adjacency family" },
  { from: "conversion", to: "ahlbach", label: "Switching cost" },
  { from: "stakhov", to: "ternary", label: "Stakhov's bridge" }
];

// ═══════════════════════════════════════════════════════════════
// TREE STRUCTURE: recursive build-up
// ═══════════════════════════════════════════════════════════════

const TREE_LEVELS = [
  {
    level: 0,
    label: "Base Cases",
    desc: "Each finding evaluated independently: relevant (1) or not (0)?",
    filter: () => true,
    mode: "base"
  },
  {
    level: 1,
    label: "Domain Clusters",
    desc: "Findings within each domain reinforce each other. Does clustering change any 0→1?",
    filter: () => true,
    mode: "domain"
  },
  {
    level: 2,
    label: "Cross-Domain Connections",
    desc: "The adjacency constraint appears in ALL four domains. What was independent becomes structural.",
    filter: () => true,
    mode: "cross"
  },
  {
    level: 3,
    label: "The Sum Total",
    desc: "Everything converges on M = [[1,1],[1,0]]. The matrix is the atom. All domains are projections.",
    filter: () => true,
    mode: "synthesis"
  }
];

// ═══════════════════════════════════════════════════════════════
// FIBONACCI TREE INFRASTRUCTURE
// ═══════════════════════════════════════════════════════════════

const FIB = [1, 2];
for (let i = 2; i < 25; i++) FIB.push(FIB[i - 1] + FIB[i - 2]);

function buildTree(n) {
  if (n <= 1) return { n, left: null, right: null, leafCount: 1, id: Math.random().toString(36).slice(2, 8) };
  const left = buildTree(n - 1);
  const right = buildTree(n - 2);
  return { n, left, right, leafCount: left.leafCount + right.leafCount, id: Math.random().toString(36).slice(2, 8) };
}

function layoutTree(node, depth = 0, leafIndex = { current: 0 }) {
  if (!node) return [];
  if (!node.left && !node.right) {
    const pos = { ...node, x: leafIndex.current, y: depth };
    leafIndex.current += 1;
    return [pos];
  }
  const leftNodes = layoutTree(node.left, depth + 1, leafIndex);
  const rightNodes = layoutTree(node.right, depth + 1, leafIndex);
  const lRoot = leftNodes.find(nd => nd.id === node.left.id);
  const rRoot = rightNodes.find(nd => nd.id === node.right.id);
  const pos = { ...node, x: (lRoot.x + rRoot.x) / 2, y: depth, leftRoot: lRoot, rightRoot: rRoot };
  return [pos, ...leftNodes, ...rightNodes];
}

// ═══════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════

function NodeCard({ finding, isSelected, onClick, compact = false, dimmed = false }) {
  const color = DOMAIN_COLORS[finding.domain];
  return (
    <button
      onClick={onClick}
      style={{
        border: "none",
        borderRadius: compact ? 6 : 8,
        cursor: "pointer",
        padding: compact ? "4px 8px" : "10px 14px",
        textAlign: "left",
        width: "100%",
        background:
          isSelected ? `${color}20`
          : dimmed ? "rgba(255,255,255,0.01)"
          : "rgba(255,255,255,0.03)",
        border:
          isSelected ? `2px solid ${color}` : `1px solid ${dimmed ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.06)"}`,
        opacity: dimmed ? 0.3 : 1,
        transition: "all 0.2s ease"
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
        <div
          style={{
            fontFamily: MONO,
            fontSize: compact ? 10 : 12,
            color: isSelected ? color : "rgba(255,255,255,0.5)",
            fontWeight: isSelected ? 700 : 400
          }}
        >
          {finding.name}
        </div>
        <div
          style={{
            fontFamily: MONO,
            fontSize: 9,
            color: finding.rel ? GREEN : RED,
            padding: "1px 6px",
            borderRadius: 4,
            background: finding.rel ? "rgba(64,200,120,0.12)" : "rgba(224,64,64,0.12)",
            flexShrink: 0
          }}
        >
          {finding.rel ? "1" : "0"}
        </div>
      </div>
      {!compact && (
        <div
          style={{
            fontFamily: MONO,
            fontSize: 9,
            color: "rgba(255,255,255,0.2)",
            marginTop: 2
          }}
        >
          {finding.who} · {finding.year}
        </div>
      )}
    </button>
  );
}

function DetailPanel({ finding }) {
  if (!finding)
    return (
      <div
        style={{
          padding: 20,
          textAlign: "center",
          fontFamily: MONO,
          fontSize: 11,
          color: "rgba(255,255,255,0.15)"
        }}
      >
        Select a node to see details
      </div>
    );

  const color = DOMAIN_COLORS[finding.domain];
  return (
    <div
      style={{
        padding: 16,
        background: `${color}08`,
        borderRadius: 10,
        border: `1px solid ${color}30`
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
        <div>
          <div style={{ fontFamily: MONO, fontSize: 14, color, fontWeight: 700 }}>{finding.name}</div>
          <div style={{ fontFamily: MONO, fontSize: 10, color: "rgba(255,255,255,0.3)", marginTop: 2 }}>
            {finding.who} · {finding.year}
          </div>
        </div>
        <div
          style={{
            fontFamily: MONO,
            fontSize: 9,
            padding: "2px 8px",
            borderRadius: 6,
            letterSpacing: 2,
            background: `${color}20`,
            color
          }}
        >
          {DOMAIN_LABELS[finding.domain]}
        </div>
      </div>

      <div style={{ marginBottom: 12 }}>
        <div style={{ fontFamily: MONO, fontSize: 9, color: "rgba(255,255,255,0.25)", letterSpacing: 2, marginBottom: 4 }}>
          WHAT
        </div>
        <div style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.5)", lineHeight: 1.7 }}>{finding.what}</div>
      </div>

      <div style={{ marginBottom: 12 }}>
        <div style={{ fontFamily: MONO, fontSize: 9, color: "rgba(255,255,255,0.25)", letterSpacing: 2, marginBottom: 4 }}>
          CONNECTION TO M
        </div>
        <div style={{ fontFamily: MONO, fontSize: 11, color, lineHeight: 1.7 }}>{finding.connection}</div>
      </div>

      {finding.links && finding.links.length > 0 && (
        <div>
          <div style={{ fontFamily: MONO, fontSize: 9, color: "rgba(255,255,255,0.25)", letterSpacing: 2, marginBottom: 4 }}>
            CONNECTS TO
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {finding.links.map(lid => {
              const linked = FINDINGS.find(f => f.id === lid);
              if (!linked) return null;
              return (
                <span
                  key={lid}
                  style={{
                    fontFamily: MONO,
                    fontSize: 9,
                    padding: "2px 6px",
                    borderRadius: 4,
                    background: `${DOMAIN_COLORS[linked.domain]}15`,
                    color: DOMAIN_COLORS[linked.domain],
                    border: `1px solid ${DOMAIN_COLORS[linked.domain]}30`
                  }}
                >
                  {linked.name}
                </span>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function TreeDiagram({ level, selectedId, onSelect }) {
  const domains = ["math", "hardware", "coding", "systems"];
  const byDomain = useMemo(() => {
    const m = {};
    domains.forEach(d => {
      m[d] = FINDINGS.filter(f => f.domain === d);
    });
    return m;
  }, []);

  const showCross = level >= 2;
  const showSynthesis = level >= 3;

  return (
    <div style={{ position: "relative" }}>
      {/* Root node */}
      <div style={{ textAlign: "center", marginBottom: 16 }}>
        <div
          style={{
            display: "inline-block",
            padding: "8px 20px",
            borderRadius: 8,
            background: showSynthesis ? "rgba(212,160,23,0.2)" : "rgba(255,255,255,0.04)",
            border: `2px solid ${showSynthesis ? GOLD : "rgba(255,255,255,0.08)"}`,
            fontFamily: MONO,
            fontSize: 12,
            color: showSynthesis ? GOLD : "rgba(255,255,255,0.3)",
            fontWeight: 700,
            letterSpacing: 1,
            transition: "all 0.5s ease"
          }}
        >
          M = [[1,1],[1,0]]
        </div>
        {showSynthesis && (
          <div
            style={{
              fontFamily: MONO,
              fontSize: 10,
              color: "rgba(255,255,255,0.2)",
              marginTop: 4
            }}
          >
            {FINDINGS.filter(f => f.rel).length} findings → all relevant → all connected → one matrix
          </div>
        )}
      </div>

      {/* Domain branches */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: 10
        }}
      >
        {domains.map(domain => {
          const color = DOMAIN_COLORS[domain];
          const items = byDomain[domain];
          const showItems = level >= 1;

          return (
            <div
              key={domain}
              style={{
                borderRadius: 10,
                border: `1px solid ${color}20`,
                background: `${color}04`,
                padding: 10,
                transition: "all 0.3s ease"
              }}
            >
              <div
                style={{
                  fontFamily: MONO,
                  fontSize: 9,
                  color,
                  letterSpacing: 2,
                  marginBottom: 8,
                  textAlign: "center",
                  fontWeight: 700
                }}
              >
                {DOMAIN_LABELS[domain]}
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {items.map(f => (
                  <NodeCard
                    key={f.id}
                    finding={f}
                    isSelected={selectedId === f.id}
                    onClick={() => onSelect(f.id)}
                    compact
                    dimmed={level === 0 && false}
                  />
                ))}
              </div>

              {showItems && (
                <div
                  style={{
                    fontFamily: MONO,
                    fontSize: 9,
                    color: `${color}80`,
                    textAlign: "center",
                    marginTop: 6
                  }}
                >
                  {items.filter(f => f.rel).length}/{items.length} relevant
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Cross-domain connections */}
      {showCross && (
        <div
          style={{
            marginTop: 14,
            padding: 12,
            borderRadius: 8,
            background: "rgba(255,160,40,0.04)",
            border: "1px solid rgba(255,160,40,0.15)"
          }}
        >
          <div
            style={{ fontFamily: MONO, fontSize: 9, color: ORANGE, letterSpacing: 2, marginBottom: 8, textAlign: "center" }}
          >
            CROSS-DOMAIN CONNECTIONS — THE ADJACENCY CONSTRAINT EVERYWHERE
          </div>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 4,
              justifyContent: "center"
            }}
          >
            {CROSS_LINKS.map((cl, i) => {
              const from = FINDINGS.find(f => f.id === cl.from);
              const to = FINDINGS.find(f => f.id === cl.to);
              if (!from || !to) return null;
              const isHighlighted = selectedId === cl.from || selectedId === cl.to;
              return (
                <div
                  key={i}
                  style={{
                    fontFamily: MONO,
                    fontSize: 9,
                    padding: "3px 8px",
                    borderRadius: 6,
                    background: isHighlighted ? "rgba(255,160,40,0.15)" : "rgba(255,255,255,0.02)",
                    border: `1px solid ${isHighlighted ? ORANGE : "rgba(255,255,255,0.06)"}`,
                    color: isHighlighted ? ORANGE : "rgba(255,255,255,0.25)",
                    transition: "all 0.2s ease",
                    cursor: "default"
                  }}
                >
                  <span style={{ color: DOMAIN_COLORS[from.domain] }}>{from.name.split(" ")[0]}</span>
                  <span style={{ margin: "0 4px" }}>↔</span>
                  <span style={{ color: DOMAIN_COLORS[to.domain] }}>{to.name.split(" ")[0]}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function LevelSelector({ level, onSetLevel }) {
  return (
    <div
      style={{
        display: "flex",
        gap: 4,
        marginBottom: 16,
        padding: 4,
        background: "rgba(255,255,255,0.02)",
        borderRadius: 10,
        border: "1px solid rgba(255,255,255,0.04)"
      }}
    >
      {TREE_LEVELS.map((tl, i) => (
        <button
          key={i}
          onClick={() => onSetLevel(i)}
          style={{
            flex: 1,
            border: "none",
            borderRadius: 8,
            cursor: "pointer",
            padding: "8px 4px",
            fontFamily: MONO,
            fontSize: 9,
            letterSpacing: 1,
            fontWeight: 600,
            background: level === i ? "rgba(212,160,23,0.15)" : "transparent",
            color:
              level === i ? GOLD
              : i < level ? GREEN
              : "rgba(255,255,255,0.2)",
            transition: "all 0.2s ease"
          }}
        >
          <div>{tl.label}</div>
          <div style={{ fontSize: 8, fontWeight: 400, marginTop: 2, opacity: 0.6 }}>fib({i})</div>
        </button>
      ))}
    </div>
  );
}

function StatsBar({ level }) {
  const total = FINDINGS.length;
  const relevant = FINDINGS.filter(f => f.rel).length;
  const domains = ["math", "hardware", "coding", "systems"];

  return (
    <div
      style={{
        display: "flex",
        gap: 12,
        justifyContent: "center",
        flexWrap: "wrap",
        padding: "8px 0",
        fontFamily: MONO,
        fontSize: 10
      }}
    >
      <span style={{ color: "rgba(255,255,255,0.3)" }}>
        {total} findings · <span style={{ color: GREEN }}>{relevant} relevant</span>
      </span>
      {level >= 1 &&
        domains.map(d => (
          <span
            key={d}
            style={{ color: DOMAIN_COLORS[d] }}
          >
            {DOMAIN_LABELS[d]}: {FINDINGS.filter(f => f.domain === d).length}
          </span>
        ))}
      {level >= 2 && <span style={{ color: ORANGE }}>{CROSS_LINKS.length} cross-links</span>}
    </div>
  );
}

// Standard Fibonacci & Lucas lookups for annotations
const SF = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233];
const LN = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199];
const sf = k => SF[k] ?? "?";
const ln = k => LN[k] ?? "?";
const PHI = (1 + Math.sqrt(5)) / 2;

function getTreeAnnotation(finding) {
  if (!finding) return null;
  const color = DOMAIN_COLORS[finding.domain];
  const a = {
    vajda: {
      eq: "F(n+i)·F(n+j) − F(n)·F(n+i+j) = (−1)ⁿ·F(i)·F(j)",
      sub: k => `(−1)^${k}=${k % 2 === 0 ? "+1" : "−1"}`,
      insight: "Every node carries the sign factor. Vajda subsumes Cassini, Catalan, d'Ocagne."
    },
    pisano: {
      eq: "F(n) mod m repeats with period π(m)",
      sub: k => `F(${k})≡${sf(k) % 7} mod 7`,
      insight: "π(7)=16. The residues cycle: the tree wraps around modularly."
    },
    contfrac: {
      eq: "φ = [1;1,1,1...] — F(n)/F(n−1) → φ",
      sub: k => (k <= 1 ? `F(${k})=${sf(k)}` : `${sf(k)}/${sf(k - 1)}≈${(sf(k) / sf(k - 1)).toFixed(3)}`),
      insight: "Each depth ratio converges to φ. The tree IS the continued fraction unfolding."
    },
    zphi: {
      eq: "ℤ[φ]: Norm(φⁿ) = det(Mⁿ) = (−1)ⁿ",
      sub: k => `N=${k % 2 === 0 ? "+1" : "−1"}`,
      insight: "The algebraic norm alternates at every level. φ is the fundamental unit of ℤ[φ]."
    },
    indep_sets: {
      eq: "Independent sets on Pₙ = F(n+2)",
      sub: k => `I(P${k})=${sf(k + 2)}`,
      insight: "Each node's count of valid Zeckendorf strings = independent sets = F(k+2)."
    },
    lucas: {
      eq: "L(n) = tr(Mⁿ) = φⁿ + ψⁿ",
      sub: k => `L(${k})=${ln(k)}`,
      insight: "Lucas = trace, Fibonacci = off-diagonal. Two readouts of one matrix at each node."
    },
    sl2z: {
      eq: "M ∈ GL(2,ℤ), det(Mⁿ) = (−1)ⁿ",
      sub: k => `det=${k % 2 === 0 ? "+1" : "−1"}`,
      insight: "The Q-matrix lives in the modular group. Each node's determinant tracks parity."
    },
    negafib: {
      eq: "F(−n) = (−1)^(n+1)·F(n)",
      sub: k => (k === 0 ? "F(0)=0" : `F(−${k})=${(k + 1) % 2 === 0 ? "-" : ""}${sf(k)}`),
      insight: "Negative indices mirror the tree with alternating signs."
    },
    metallic: {
      eq: "φ = (1+√5)/2, eigenvalue of M",
      sub: k => `φ^${k}≈${Math.pow(PHI, k).toFixed(1)}`,
      insight: "φ^n grows at each node. The golden mean is the simplest metallic mean."
    },
    cassini_norm: {
      eq: "F(n−1)·F(n+1) − F(n)² = (−1)ⁿ",
      sub: k => (k === 0 ? "base" : `${sf(k - 1)}·${sf(k + 1)}−${sf(k)}²=${k % 2 === 0 ? "+1" : "−1"}`),
      insight: "Cassini's identity at every node. If the check fails, the state is corrupt."
    },
    stakhov: {
      eq: "φ-register: adjacency → built-in error detection",
      sub: k => (k <= 1 ? `F(${k})=${sf(k)}` : "✓"),
      insight: "Stakhov's 65 patents: every node is a valid φ-register state."
    },
    berstel: {
      eq: "4-state transducer normalizes Fibonacci addition",
      sub: k => ["S₀", "S₁", "S₂", "S₃"][k % 4],
      insight: "Berstel's automaton cycles through 4 states at each node."
    },
    ahlbach: {
      eq: "O(n) gates, O(log n) depth — parallel carry",
      sub: k => `d=${k}`,
      insight: "Tree depth = log structure. Fibonacci carry parallelizes like carry-lookahead."
    },
    fenwick: {
      eq: "Dual carry: F(n) → F(n+1) + F(n−2)",
      sub: k => (k <= 1 ? `F(${k})` : `${sf(k)}→${sf(k + 1)}+${sf(Math.max(0, k - 2))}`),
      insight: "Two carries per node: one forward (+1), one backward (−2)."
    },
    fpga: {
      eq: "FPGA Fibonacci decoders: 10–30% savings",
      sub: k => `${sf(k)}b`,
      insight: "Working silicon. Each node = a Fibonacci-weighted bit position."
    },
    async: {
      eq: "011→100 rewrite = local-to-global settling",
      sub: k => (k <= 1 ? `${sf(k)}` : "settle"),
      insight: "Normalization IS computation. Each node settles independently."
    },
    ternary: {
      eq: "Balanced ternary {−1,0,1} ↔ Fibonacci bridge",
      sub: k => (k === 0 ? "0" : `±${sf(k)}`),
      insight: "Stakhov bridged ternary ↔ Fibonacci. Each node carries signed weight."
    },
    fib_code: {
      eq: "Zeckendorf + '11' delimiter → universal prefix code",
      sub: k => (sf(k) <= 1 ? `${sf(k)}` : `≈${Math.ceil(1.44 * Math.log2(sf(k)))}b`),
      insight: "Code length ≈ 1.44·log₂(n). The '11' delimiter IS the branching redundancy."
    },
    self_sync: {
      eq: "After bit error: max 3 tokens drift, '0' resyncs",
      sub: k => (k <= 1 ? "sync" : "≤3"),
      insight: "Errors don't propagate through the tree. Each subtree resyncs."
    },
    inverted_idx: {
      eq: "Beats Elias δ for n < 514,228",
      sub: k => "✓fib",
      insight: "Every node value in this tree is below the crossover. Fibonacci code wins."
    },
    crosstalk: {
      eq: "'No consecutive 1s' = crosstalk-free bus encoding",
      sub: k => (k <= 1 ? `${sf(k)}` : "no 11"),
      insight: "The adjacency constraint at each node = forbidden-transition-free bus signal."
    },
    error_correct: {
      eq: "Mⁿ as encoding matrix — detection → 1",
      sub: k => `M^${k}`,
      insight: "The matrix at each node IS the encoder. Detection improves with depth."
    },
    negafib_code: {
      eq: "NegaFibonacci: extends coding to all integers",
      sub: k => (k === 0 ? "0" : `F(−${k})=${(k + 1) % 2 === 0 ? "-" : ""}${sf(k)}`),
      insight: "Even-length = positive, odd-length = negative. Complete integer coverage."
    },
    conversion: {
      eq: "Binary↔Fibonacci: O(M(n)·log n) circuit",
      sub: k => `${sf(k)}→${sf(k).toString(2)}₂`,
      insight: "Each node: the conversion cost between two representations."
    },
    naf: {
      eq: "NAF: {−1,0,1}, no consecutive nonzeros",
      sub: k => `w=${sf(k).toString(2).replace(/0/g, "").length}`,
      insight: "Hamming weight through the tree. NAF and Zeckendorf share the same constraint."
    },
    booth: {
      eq: "Booth: recode to reduce partial products",
      sub: k => `${sf(k)}→B`,
      insight: "Same adjacency philosophy. Booth, NAF, Zeckendorf all exploit 'no consecutive nonzeros'."
    },
    phinary: {
      eq: "Base φ: 011→100 = φⁿ⁻¹+φⁿ⁻²=φⁿ",
      sub: k => `φ^${k}≈${Math.pow(PHI, k).toFixed(2)}`,
      insight: "Zeckendorf and phinary are two faces of one structure. The carry IS φ²=φ+1."
    },
    stern_brocot: {
      eq: "Path to φ = RLRLRL... in Stern-Brocot tree",
      sub: k => (k % 2 === 0 ? "R" : "L"),
      insight: "The path alternates R,L,R,L... Each node = one step toward φ."
    },
    ostrowski: {
      eq: "Ostrowski: generalized Zeckendorf for any α",
      sub: k => `q${k}=${sf(k)}`,
      insight: "CF denominators = Fibonacci for α=φ. The tree IS the Ostrowski decomposition."
    },
    golden_shift: {
      eq: "β-expansion for β=φ: admissible sequences avoid '11'",
      sub: k => (k <= 1 ? `${sf(k)}` : "adm."),
      insight: "Symbolic dynamics: each tree path is admissible in the golden mean shift."
    },
    redundant: {
      eq: "Controlled redundancy → carry-free addition",
      sub: k => (k <= 1 ? `${sf(k)}` : "norm"),
      insight: "Fibonacci normalization is a specific instance of redundancy resolution."
    }
  };
  const entry = a[finding.id];
  if (!entry) return { eq: finding.what.slice(0, 60), color, sub: k => `F(${k})`, insight: finding.connection };
  return { ...entry, color };
}

function RecursiveView({ finding }) {
  const [n, setN] = useState(6);
  const [highlight, setHighlight] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);

  const annotation = useMemo(() => getTreeAnnotation(finding), [finding]);
  const accentColor = annotation?.color || GOLD;

  const tree = useMemo(() => buildTree(n), [n]);
  const nodes = useMemo(() => layoutTree(tree), [tree]);

  const maxX = Math.max(...nodes.map(nd => nd.x), 1);
  const maxY = Math.max(...nodes.map(nd => nd.y), 1);
  const padding = 40;
  const nodeR = 16;
  const svgW = Math.max(400, (maxX + 1) * 50 + padding * 2);
  const svgH = (maxY + 1) * (annotation ? 65 : 55) + padding * 2;
  const scaleX = x => padding + (x / Math.max(maxX, 1)) * (svgW - padding * 2);
  const scaleY = y => padding + y * (annotation ? 58 : 50);

  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setHighlight(null);
  }, [n]);

  const handleWheel = useCallback(
    e => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const newZoom = Math.max(0.2, Math.min(5, zoom * delta));
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const scale = newZoom / zoom;
        setPan(p => ({ x: cx - scale * (cx - p.x), y: cy - scale * (cy - p.y) }));
      }
      setZoom(newZoom);
    },
    [zoom]
  );

  const handleMouseDown = useCallback(
    e => {
      if (e.button !== 0) return;
      setDragging(true);
      setDragStart({ x: e.clientX, y: e.clientY });
      setPanStart({ ...pan });
    },
    [pan]
  );

  const handleMouseMove = useCallback(
    e => {
      if (!dragging) return;
      setPan({ x: panStart.x + (e.clientX - dragStart.x), y: panStart.y + (e.clientY - dragStart.y) });
    },
    [dragging, dragStart, panStart]
  );

  const handleMouseUp = useCallback(() => setDragging(false), []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener("wheel", handleWheel, { passive: false });
    return () => el.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

  const collectIds = useCallback(node => {
    if (!node) return new Set();
    const ids = new Set([node.id]);
    collectIds(node.left).forEach(id => ids.add(id));
    collectIds(node.right).forEach(id => ids.add(id));
    return ids;
  }, []);

  const highlightIds = useMemo(() => {
    if (highlight === null) return null;
    const findNode = (node, targetN) => {
      if (!node) return null;
      if (node.n === targetN) return node;
      return findNode(node.left, targetN) || findNode(node.right, targetN);
    };
    const subtreeRoot = findNode(tree, highlight);
    return subtreeRoot ? collectIds(subtreeRoot) : null;
  }, [highlight, tree, collectIds]);

  const leaves = nodes.filter(nd => nd.n <= 1);
  const ones = nodes.filter(nd => nd.n === 1).length;

  return (
    <div
      style={{
        padding: 16,
        borderRadius: 10,
        background: `${accentColor}08`,
        border: `1px solid ${accentColor}1a`
      }}
    >
      <div
        style={{ fontFamily: MONO, fontSize: 9, color: accentColor, letterSpacing: 3, marginBottom: 12, textAlign: "center" }}
      >
        THE RECURSIVE TREE
      </div>

      {/* Equation banner */}
      {annotation && (
        <div
          style={{
            textAlign: "center",
            marginBottom: 10,
            padding: "6px 14px",
            borderRadius: 6,
            background: `${accentColor}10`,
            border: `1px solid ${accentColor}20`
          }}
        >
          <div style={{ fontFamily: MONO, fontSize: 11, color: accentColor, fontWeight: 600 }}>{annotation.eq}</div>
          <div style={{ fontFamily: MONO, fontSize: 9, color: `${accentColor}90`, marginTop: 2 }}>
            {finding.name} — {DOMAIN_LABELS[finding.domain]}
          </div>
        </div>
      )}

      {/* Controls */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 10,
          marginBottom: 10,
          flexWrap: "wrap"
        }}
      >
        <span style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.4)" }}>fib(</span>
        <input
          type="range"
          min={2}
          max={9}
          value={n}
          onChange={e => setN(+e.target.value)}
          style={{ width: 100, accentColor: GOLD }}
        />
        <span style={{ fontFamily: MONO, fontSize: 13, color: accentColor, fontWeight: 700 }}>{n}</span>
        <span style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.4)" }}>) = {FIB[n - 1] || n}</span>
        {annotation && (
          <span style={{ fontFamily: MONO, fontSize: 9, color: accentColor, opacity: 0.6 }}>· {annotation.sub(n)}</span>
        )}
        <span style={{ fontFamily: MONO, fontSize: 10, color: "rgba(255,255,255,0.2)" }}>
          {nodes.length} nodes · {leaves.length} leaves
        </span>
      </div>

      {/* Subtree highlight buttons */}
      <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 8, flexWrap: "wrap" }}>
        <button
          onClick={() => setHighlight(null)}
          style={{
            padding: "4px 10px",
            border: "1px solid rgba(212,160,23,0.3)",
            borderRadius: 5,
            background: highlight === null ? "rgba(212,160,23,0.15)" : "transparent",
            color: GOLD,
            cursor: "pointer",
            fontFamily: MONO,
            fontSize: 9
          }}
        >
          FULL TREE
        </button>
        {n >= 2 && (
          <button
            onClick={() => setHighlight(n - 1)}
            style={{
              padding: "4px 10px",
              border: "1px solid rgba(212,160,23,0.3)",
              borderRadius: 5,
              background: highlight === n - 1 ? "rgba(212,160,23,0.15)" : "transparent",
              color: highlight === n - 1 ? GOLD : "rgba(212,160,23,0.5)",
              cursor: "pointer",
              fontFamily: MONO,
              fontSize: 9
            }}
          >
            LEFT: fib({n - 1})
          </button>
        )}
        {n >= 3 && (
          <button
            onClick={() => setHighlight(n - 2)}
            style={{
              padding: "4px 10px",
              border: "1px solid rgba(64,200,120,0.3)",
              borderRadius: 5,
              background: highlight === n - 2 ? "rgba(64,200,120,0.12)" : "transparent",
              color: highlight === n - 2 ? GREEN : "rgba(64,200,120,0.4)",
              cursor: "pointer",
              fontFamily: MONO,
              fontSize: 9
            }}
          >
            RIGHT: fib({n - 2})
          </button>
        )}
      </div>

      {/* SVG tree viewport */}
      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{
          overflow: "hidden",
          background: "rgba(0,0,0,0.2)",
          borderRadius: 8,
          border: "1px solid rgba(255,255,255,0.05)",
          height: 320,
          cursor: dragging ? "grabbing" : "grab",
          position: "relative"
        }}
      >
        <div
          style={{
            position: "absolute",
            bottom: 6,
            right: 8,
            fontFamily: MONO,
            fontSize: 8,
            color: "rgba(255,255,255,0.12)",
            pointerEvents: "none",
            zIndex: 1
          }}
        >
          scroll to zoom · drag to pan
        </div>
        <svg
          width={svgW}
          height={svgH}
          style={{
            display: "block",
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: "0 0",
            transition: dragging ? "none" : "transform 0.1s ease-out"
          }}
        >
          {/* Edges */}
          {nodes.map(nd => {
            if (!nd.leftRoot && !nd.rightRoot) return null;
            const px = scaleX(nd.x),
              py = scaleY(nd.y);
            const dimmed = highlightIds && !highlightIds.has(nd.id);
            return (
              <g
                key={`e-${nd.id}`}
                opacity={dimmed ? 0.1 : 0.45}
              >
                {nd.leftRoot && (
                  <line
                    x1={px}
                    y1={py + nodeR}
                    x2={scaleX(nd.leftRoot.x)}
                    y2={scaleY(nd.leftRoot.y) - nodeR}
                    stroke={accentColor}
                    strokeWidth={1.5}
                  />
                )}
                {nd.rightRoot && (
                  <line
                    x1={px}
                    y1={py + nodeR}
                    x2={scaleX(nd.rightRoot.x)}
                    y2={scaleY(nd.rightRoot.y) - nodeR}
                    stroke={accentColor}
                    strokeWidth={1.5}
                  />
                )}
              </g>
            );
          })}
          {/* Nodes */}
          {nodes.map(nd => {
            const cx = scaleX(nd.x),
              cy = scaleY(nd.y);
            const isLeaf = nd.n <= 1;
            const dimmed = highlightIds && !highlightIds.has(nd.id);
            const isHighlightRoot = highlight !== null && nd.n === highlight && !dimmed;
            let fill, stroke, textFill;
            if (dimmed) {
              fill = "rgba(255,255,255,0.02)";
              stroke = "rgba(255,255,255,0.06)";
              textFill = "rgba(255,255,255,0.1)";
            } else if (isLeaf) {
              fill = nd.n === 1 ? `${accentColor}33` : "rgba(255,255,255,0.05)";
              stroke = nd.n === 1 ? accentColor : "rgba(255,255,255,0.2)";
              textFill = nd.n === 1 ? accentColor : "rgba(255,255,255,0.4)";
            } else {
              fill = isHighlightRoot ? `${accentColor}40` : `${accentColor}14`;
              stroke = isHighlightRoot ? accentColor : `${accentColor}4d`;
              textFill = "#fff";
            }
            return (
              <g
                key={`n-${nd.id}`}
                style={{ cursor: nd.n >= 2 ? "pointer" : "default" }}
                onClick={() => {
                  if (nd.n >= 2) setHighlight(highlight === nd.n ? null : nd.n);
                }}
              >
                <circle
                  cx={cx}
                  cy={cy}
                  r={nodeR}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={isHighlightRoot ? 2 : 1}
                />
                <text
                  x={cx}
                  y={cy + 1}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill={textFill}
                  fontSize={isLeaf ? 12 : 10}
                  fontFamily={MONO}
                  fontWeight={isLeaf ? 700 : 400}
                >
                  {isLeaf ? nd.n : `(${nd.n})`}
                </text>
                {annotation && !dimmed && (
                  <text
                    x={cx}
                    y={cy + nodeR + 11}
                    textAnchor="middle"
                    fill={accentColor}
                    fontSize={7}
                    fontFamily={MONO}
                    opacity={0.7}
                  >
                    {annotation.sub(nd.n)}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      {/* Insight */}
      <div
        style={{
          marginTop: 12,
          padding: 10,
          background: `${accentColor}0a`,
          borderRadius: 8,
          border: `1px solid ${accentColor}1a`,
          fontFamily: MONO,
          fontSize: 10,
          color: "rgba(255,255,255,0.35)",
          lineHeight: 1.8
        }}
      >
        {annotation ?
          <>
            <div>
              <span style={{ color: accentColor }}>{finding.name}:</span> {annotation.insight}
            </div>
            <div>
              <span style={{ color: accentColor }}>At each node:</span> the annotation shows{" "}
              {annotation.eq.toLowerCase().includes("mod") ?
                "modular residues"
              : annotation.eq.toLowerCase().includes("det") ?
                "determinant values"
              : annotation.eq.toLowerCase().includes("φ") ?
                "golden ratio powers"
              : "the equation evaluated"}{" "}
              for fib({n}).
            </div>
          </>
        : <>
            <div>
              <span style={{ color: GOLD }}>Self-similar:</span> fib({n}) contains a full fib({n - 1}) on the left and a full
              fib({n - 2}) on the right.
            </div>
            <div>
              <span style={{ color: GOLD }}>Leaf count = value:</span> {ones} ones → fib({n}) = {FIB[n - 1] || n}. The number
              of 1-leaves IS the Fibonacci number.
            </div>
          </>
        }
        <div>
          <span style={{ color: accentColor }}>Click any node</span> to highlight its subtree.
          {!annotation && " Select a finding above to see its equation at each node."}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════

export default function ResearchTree() {
  const [level, setLevel] = useState(0);
  const [selectedId, setSelectedId] = useState(null);
  const selected = FINDINGS.find(f => f.id === selectedId);

  return (
    <div
      style={{
        background: BG,
        minHeight: "100vh",
        padding: "20px 12px",
        color: "rgba(255,255,255,0.7)",
        fontFamily: MONO
      }}
    >
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 20 }}>
          <div
            style={{
              fontFamily: DISPLAY,
              fontSize: 24,
              fontWeight: 700,
              color: GOLD,
              letterSpacing: 1
            }}
          >
            The Deep Map
          </div>
          <div
            style={{
              fontFamily: MONO,
              fontSize: 10,
              color: "rgba(255,255,255,0.2)",
              maxWidth: 500,
              margin: "6px auto 0",
              lineHeight: 1.6
            }}
          >
            {FINDINGS.length} findings across 4 domains, built up recursively. Each level reveals how findings change each
            other.
          </div>
        </div>

        {/* Level selector */}
        <LevelSelector
          level={level}
          onSetLevel={setLevel}
        />

        {/* Level description */}
        <div
          style={{
            padding: "8px 14px",
            marginBottom: 14,
            borderRadius: 8,
            background: "rgba(212,160,23,0.04)",
            border: "1px solid rgba(212,160,23,0.1)",
            fontFamily: MONO,
            fontSize: 11,
            color: "rgba(255,255,255,0.35)",
            textAlign: "center"
          }}
        >
          <span style={{ color: GOLD, fontWeight: 700 }}>
            Level {level}: {TREE_LEVELS[level].label}
          </span>
          <span style={{ margin: "0 8px", color: "rgba(255,255,255,0.1)" }}>—</span>
          {TREE_LEVELS[level].desc}
        </div>

        <StatsBar level={level} />

        {/* Tree diagram */}
        <div style={{ marginTop: 14, marginBottom: 14 }}>
          <TreeDiagram
            level={level}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        </div>

        {/* Detail panel */}
        <DetailPanel finding={selected} />

        {/* Recursive view */}
        <div style={{ marginTop: 16 }}>
          <RecursiveView finding={selected} />
        </div>

        {/* Domain legend */}
        <div
          style={{
            display: "flex",
            gap: 16,
            justifyContent: "center",
            marginTop: 20,
            fontFamily: MONO,
            fontSize: 9,
            color: "rgba(255,255,255,0.2)"
          }}
        >
          {Object.entries(DOMAIN_COLORS).map(([domain, color]) => (
            <span key={domain}>
              <span
                style={{
                  display: "inline-block",
                  width: 8,
                  height: 8,
                  borderRadius: 2,
                  background: color,
                  marginRight: 4,
                  verticalAlign: "middle",
                  opacity: 0.7
                }}
              />
              {DOMAIN_LABELS[domain]}
            </span>
          ))}
        </div>

        {/* Footer */}
        <div
          style={{
            marginTop: 20,
            padding: 12,
            borderRadius: 8,
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.04)",
            textAlign: "center",
            fontFamily: MONO,
            fontSize: 10,
            color: "rgba(255,255,255,0.15)",
            lineHeight: 1.8
          }}
        >
          The NAND observation and the dual-register switching strategy appear to be novel.
          <br />
          No dedicated Fibonacci arithmetic ASIC has been built. The gap is open.
        </div>
      </div>
    </div>
  );
}
