/**
 * ═══════════════════════════════════════════════════════════════════
 * φ-REGISTER — Fibonacci-Weighted Arithmetic Machine
 * ═══════════════════════════════════════════════════════════════════
 *
 * A complete interactive exploration of the Zeckendorf number system:
 *   - ARITHMETIC: Addition (carry/resolve) and Subtraction (borrow)
 *   - TREE: Recursive Fibonacci call tree with self-similar highlighting
 *   - PROPERTIES: The four intrinsic advantages of φ-weighted encoding
 *
 * The single axiom: F(n) = F(n-1) + F(n-2)
 *   Read left-to-right  → CARRY   (adjacent 1s merge upward)
 *   Applied to doubles   → RESOLVE (2×F(k) splits apart)
 *   Read right-to-left  → BORROW  (F(k) decomposes downward)
 */

import { useState, useEffect, useRef, useCallback, useMemo } from "react";

// ═══════════════════════════════════════════════════════════════════
// CONSTANTS & FIBONACCI GENERATION
// ═══════════════════════════════════════════════════════════════════

/** Pre-compute Fibonacci numbers. FIB[0]=1, FIB[1]=2, FIB[2]=3, FIB[3]=5, ...
 *  These are the Zeckendorf basis: every positive integer has a unique
 *  representation as a sum of non-consecutive elements from this sequence. */
const FIB = [1, 2];
for (let i = 2; i < 25; i++) FIB.push(FIB[i - 1] + FIB[i - 2]);

/** Working register width — supports values up to FIB[15] = 1597 */
const NUM_DIGITS = 16;

// ═══════════════════════════════════════════════════════════════════
// STYLE CONSTANTS — dark theme with gold/red/green semantics
// ═══════════════════════════════════════════════════════════════════

const GOLD = "#d4a017";
const RED = "#e04040";
const GREEN = "#40c878";
const ORANGE = "#ffa028";
const MONO = "'JetBrains Mono', monospace";
const SERIF = "'Crimson Pro', Georgia, serif";
const BG = "#0a0a0c";

// ═══════════════════════════════════════════════════════════════════
// ZECKENDORF ENCODING / DECODING
// ═══════════════════════════════════════════════════════════════════

/** Convert a decimal integer to its Zeckendorf digit array.
 *  Greedy algorithm: always pick the largest Fibonacci number that fits.
 *  Result is an array where digits[i] = 1 means FIB[i] is included. */
function toZeckendorf(n) {
  const digits = new Array(NUM_DIGITS).fill(0);
  if (n <= 0) return digits;
  let rem = n;
  for (let i = NUM_DIGITS - 1; i >= 0; i--) {
    if (FIB[i] <= rem) {
      digits[i] = 1;
      rem -= FIB[i];
    }
  }
  return digits;
}

/** Convert a digit array back to decimal. Tolerates digits > 1 and < 0
 *  (used during intermediate normalization steps). */
function fromDigits(digits) {
  return digits.reduce((sum, d, i) => sum + d * (FIB[i] || 0), 0);
}

// ═══════════════════════════════════════════════════════════════════
// VIOLATION DETECTION
// ═══════════════════════════════════════════════════════════════════

/** Scan a digit array for all structural violations:
 *  - "double": any cell holding 2 or more
 *  - "adjacent": two consecutive cells both holding 1+
 *  - "negative": any cell below 0 (from raw subtraction) */
function findViolations(digits) {
  const v = [];
  for (let i = 0; i < digits.length; i++) {
    if (digits[i] >= 2) v.push({ type: "double", index: i });
    if (digits[i] < 0) v.push({ type: "negative", index: i });
  }
  for (let i = 0; i < digits.length - 1; i++) {
    if (digits[i] >= 1 && digits[i + 1] >= 1) v.push({ type: "adjacent", indices: [i, i + 1] });
  }
  return v;
}

// ═══════════════════════════════════════════════════════════════════
// NORMALIZATION ENGINE
// ═══════════════════════════════════════════════════════════════════

/** Apply ONE normalization rule, returning the new state + metadata,
 *  or null if the register is already valid.
 *
 *  Rule priority:
 *  1. BORROW — fix negatives (from subtraction). Find nearest higher
 *     positive cell and decompose it downward: F(j) → F(j-1) + F(j-2)
 *  2. RESOLVE — fix doubles (cell >= 2). Split using:
 *     2·F(k) = F(k+1) + F(k-2)  [for k >= 2]
 *     2·F(1) = F(2) + F(0)       [special: 2×2 = 3+1]
 *     2·F(0) = F(1)              [special: 2×1 = 2]
 *  3. CARRY — fix adjacencies. Merge upward:
 *     F(k) + F(k+1) = F(k+2) */
function normalizeStep(digits, maxLen) {
  const d = [...digits];
  while (d.length < maxLen) d.push(0);

  // ─── BORROW: fix negative cells ───
  for (let i = 0; i < d.length; i++) {
    if (d[i] < 0) {
      // Find nearest position j > i that has a positive value to borrow from
      let j = i + 1;
      while (j < d.length && d[j] <= 0) j++;
      if (j >= d.length) return null; // shouldn't happen if a >= b

      // Decompose F(j) downward: F(j) = F(j-1) + F(j-2)
      d[j] -= 1;
      const targets = [];
      if (j === 1) {
        // Special case: FIB[1]=2 = 2×FIB[0]. Add 2 to position 0.
        // The resulting "double" at position 0 will be resolved later if needed.
        d[0] += 2;
        targets.push(0);
      } else {
        d[j - 1] += 1;
        d[j - 2] += 1;
        targets.push(j - 1, j - 2);
      }
      return { digits: d, rule: "borrow", index: j, sources: [j], targets };
    }
  }

  // ─── RESOLVE: fix doubles (cell >= 2) ───
  for (let i = 0; i < d.length; i++) {
    if (d[i] >= 2) {
      const targets = [];
      if (i === 0) {
        // 2×FIB[0] = 2×1 = 2 = FIB[1]. Promote to position 1.
        d[0] -= 2;
        if (d.length <= 1) d.push(0);
        d[1] += 1;
        targets.push(1);
      } else if (i === 1) {
        // 2×FIB[1] = 2×2 = 4 = FIB[2] + FIB[0] = 3 + 1
        d[1] -= 2;
        if (d.length <= 2) d.push(0);
        d[2] += 1;
        d[0] += 1;
        targets.push(2, 0);
      } else {
        // General: 2×FIB[i] = FIB[i+1] + FIB[i-2]
        d[i] -= 2;
        if (d.length <= i + 1) d.push(0);
        d[i + 1] += 1;
        d[i - 2] += 1;
        targets.push(i + 1, i - 2);
      }
      return { digits: d, rule: "double", index: i, sources: [i], targets };
    }
  }

  // ─── CARRY: fix adjacent 1s ───
  for (let i = 0; i < d.length - 1; i++) {
    if (d[i] >= 1 && d[i + 1] >= 1) {
      d[i] -= 1;
      d[i + 1] -= 1;
      if (d.length <= i + 2) d.push(0);
      d[i + 2] += 1;
      return { digits: d, rule: "carry", index: i, sources: [i, i + 1], targets: [i + 2] };
    }
  }

  return null; // No violations — register is settled
}

/** Run the full normalization cascade, collecting every intermediate step.
 *  Returns an array of {digits, rule, index, sources, targets}. */
function fullNormalize(digits) {
  const steps = [{ digits: [...digits], rule: null, index: null, sources: [], targets: [] }];
  let current = [...digits];
  let safety = 300;
  while (safety-- > 0) {
    const result = normalizeStep(current, NUM_DIGITS + 4);
    if (!result) break;
    current = result.digits;
    steps.push({
      digits: [...current],
      rule: result.rule,
      index: result.index,
      sources: result.sources,
      targets: result.targets
    });
  }
  return steps;
}

// ═══════════════════════════════════════════════════════════════════
// RECURSIVE TREE BUILDER
// ═══════════════════════════════════════════════════════════════════

/** Build the full uncached Fibonacci call tree for fib(n).
 *  Each node stores: { n, left, right, leafCount, depth }
 *  Leaf nodes are fib(0) and fib(1) — the base cases.
 *  leafCount tracks how many leaves (base cases) exist in the subtree. */
function buildTree(n) {
  if (n <= 1) return { n, left: null, right: null, leafCount: 1, id: Math.random().toString(36).slice(2, 8) };
  const left = buildTree(n - 1);
  const right = buildTree(n - 2);
  return { n, left, right, leafCount: left.leafCount + right.leafCount, id: Math.random().toString(36).slice(2, 8) };
}

/** Assign x/y positions to every node using a leaf-indexed layout.
 *  Leaves are spaced evenly; internal nodes sit at the midpoint of their children. */
function layoutTree(node, depth = 0, leafIndex = { current: 0 }) {
  if (!node) return [];
  if (!node.left && !node.right) {
    // Leaf node — assign position from leaf counter
    const pos = { ...node, x: leafIndex.current, y: depth };
    leafIndex.current += 1;
    return [pos];
  }
  const leftNodes = layoutTree(node.left, depth + 1, leafIndex);
  const rightNodes = layoutTree(node.right, depth + 1, leafIndex);
  // Internal node sits at midpoint of its leftmost and rightmost descendants
  const leftX = leftNodes[0].x;
  const rightX = rightNodes[rightNodes.length - 1].x;
  const lRoot = leftNodes.find(n => n.id === node.left.id);
  const rRoot = rightNodes.find(n => n.id === node.right.id);
  const pos = { ...node, x: (lRoot.x + rRoot.x) / 2, y: depth, leftRoot: lRoot, rightRoot: rRoot };
  return [pos, ...leftNodes, ...rightNodes];
}

// ═══════════════════════════════════════════════════════════════════
// UI: CELL — a single position in the register
// ═══════════════════════════════════════════════════════════════════

/** Renders one cell of the φ-register with semantic coloring:
 *  - source (red): value was consumed by the current rule
 *  - target (green): value was deposited by the current rule
 *  - violation-double (dashed red): cell holds 2+ — needs RESOLVE
 *  - violation-adjacent (dashed orange): adjacent to another 1 — needs CARRY
 *  - violation-negative (dashed red, italic): cell is negative — needs BORROW
 *  - settled (green glow): register is in valid Zeckendorf form */
function Cell({ value, weight, status }) {
  const styles = {
    source: { bg: "rgba(224,64,64,0.2)", border: `2px solid ${RED}`, glow: `0 0 10px rgba(224,64,64,0.5)`, color: RED },
    target: { bg: "rgba(64,200,120,0.2)", border: `2px solid ${GREEN}`, glow: `0 0 10px rgba(64,200,120,0.5)`, color: GREEN },
    "violation-double": { bg: "rgba(224,64,64,0.12)", border: `1px dashed ${RED}`, glow: "none", color: RED },
    "violation-adjacent": { bg: "rgba(255,160,40,0.12)", border: `1px dashed ${ORANGE}`, glow: "none", color: ORANGE },
    "violation-negative": { bg: "rgba(224,64,64,0.12)", border: `1px dashed ${RED}`, glow: "none", color: RED },
    settled: {
      bg: "rgba(64,200,120,0.1)",
      border: "1px solid rgba(64,200,120,0.4)",
      glow: "0 0 8px rgba(64,200,120,0.2)",
      color: GREEN
    },
    active: { bg: "rgba(212,160,23,0.12)", border: "1px solid rgba(212,160,23,0.35)", glow: "none", color: GOLD },
    empty: {
      bg: "rgba(255,255,255,0.02)",
      border: "1px solid rgba(255,255,255,0.06)",
      glow: "none",
      color: "rgba(255,255,255,0.12)"
    }
  };
  const s = status === "settled" && value <= 0 ? styles.empty : styles[status] || (value > 0 ? styles.active : styles.empty);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3, transition: "all 0.3s ease" }}>
      <div style={{ fontSize: 9, color: "rgba(212,160,23,0.4)", fontFamily: MONO }}>F({weight})</div>
      <div
        style={{
          width: 40,
          height: 40,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: s.bg,
          border: s.border,
          borderRadius: 6,
          fontFamily: MONO,
          fontSize: Math.abs(value) >= 2 ? 18 : 16,
          color: s.color,
          fontWeight: value !== 0 ? 700 : 400,
          transition: "all 0.3s ease",
          boxShadow: s.glow,
          position: "relative"
        }}
      >
        {value}
        {status === "source" && (
          <div
            style={{
              position: "absolute",
              top: -7,
              right: -7,
              width: 14,
              height: 14,
              background: RED,
              borderRadius: "50%",
              fontSize: 9,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#fff",
              fontWeight: 700
            }}
          >
            −
          </div>
        )}
        {status === "target" && (
          <div
            style={{
              position: "absolute",
              top: -7,
              right: -7,
              width: 14,
              height: 14,
              background: GREEN,
              borderRadius: "50%",
              fontSize: 9,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#fff",
              fontWeight: 700
            }}
          >
            +
          </div>
        )}
      </div>
      <div style={{ fontSize: 8, color: "rgba(255,255,255,0.2)", fontFamily: MONO }}>{FIB[weight] ?? ""}</div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// UI: REGISTER — a row of cells with violation/source/target overlays
// ═══════════════════════════════════════════════════════════════════

function Register({ digits, sources = [], targets = [], violations = [], settled = false }) {
  const maxIdx = Math.max(7, digits.reduce((m, v, i) => (v !== 0 ? Math.max(m, i) : m), 0) + 1);
  const visible = digits.slice(0, Math.min(maxIdx + 1, NUM_DIGITS));

  // Build a lookup: cell index → visual status
  const violMap = new Map();
  violations.forEach(v => {
    if (v.type === "double") violMap.set(v.index, "violation-double");
    else if (v.type === "negative") violMap.set(v.index, "violation-negative");
    else if (v.type === "adjacent")
      v.indices.forEach(idx => {
        if (!violMap.has(idx)) violMap.set(idx, "violation-adjacent");
      });
  });
  const srcSet = new Set(sources);
  const tgtSet = new Set(targets);

  const getStatus = i => {
    if (srcSet.has(i)) return "source";
    if (tgtSet.has(i)) return "target";
    if (settled) return "settled";
    if (violMap.has(i)) return violMap.get(i);
    return "normal";
  };

  return (
    <div style={{ display: "flex", flexDirection: "row-reverse", gap: 4, justifyContent: "flex-end", flexWrap: "wrap" }}>
      {visible
        .map((v, i) => (
          <Cell
            key={i}
            value={v}
            weight={i}
            status={getStatus(i)}
          />
        ))
        .reverse()}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// UI: RULE BANNER — prominent display of the current normalization rule
// ═══════════════════════════════════════════════════════════════════

function RuleBanner({ step, isSettled, isRaw }) {
  if (isSettled)
    return (
      <div
        style={{
          padding: "10px 14px",
          background: "rgba(64,200,120,0.08)",
          border: `1px solid rgba(64,200,120,0.3)`,
          borderRadius: 8,
          marginBottom: 12,
          fontFamily: MONO,
          fontSize: 12,
          color: GREEN,
          display: "flex",
          alignItems: "center",
          gap: 10
        }}
      >
        <span style={{ fontSize: 16 }}>⬡</span>
        <span>SETTLED — valid Zeckendorf form</span>
      </div>
    );
  if (isRaw)
    return (
      <div
        style={{
          padding: "10px 14px",
          background: "rgba(212,160,23,0.06)",
          border: "1px solid rgba(212,160,23,0.2)",
          borderRadius: 8,
          marginBottom: 12,
          fontFamily: MONO,
          fontSize: 12,
          color: "rgba(212,160,23,0.7)"
        }}
      >
        RAW RESULT — not yet normalized
      </div>
    );

  const { rule, index } = step;
  const configs = {
    carry: {
      bg: "rgba(212,160,23,0.06)",
      bdr: "rgba(212,160,23,0.25)",
      hdr: GOLD,
      title: "CARRY — adjacent 1s merge upward",
      eq: (
        <>
          <span style={{ color: RED }}>
            F({index}) + F({index + 1})
          </span>
          {" → "}
          <span style={{ color: GREEN }}>F({index + 2})</span>
          <span style={{ color: "rgba(255,255,255,0.3)", marginLeft: 12 }}>
            {FIB[index]}+{FIB[index + 1]}={FIB[index + 2]}
          </span>
        </>
      )
    },
    borrow: {
      bg: "rgba(64,200,120,0.04)",
      bdr: "rgba(64,200,120,0.25)",
      hdr: GREEN,
      title: "BORROW — decompose downward",
      eq:
        index === 1 ?
          <>
            <span style={{ color: RED }}>F(1)</span>
            {" → "}
            <span style={{ color: GREEN }}>2×F(0)</span>
            <span style={{ color: "rgba(255,255,255,0.3)", marginLeft: 12 }}>
              {FIB[1]}=2×{FIB[0]}
            </span>
          </>
        : <>
            <span style={{ color: RED }}>F({index})</span>
            {" → "}
            <span style={{ color: GREEN }}>
              F({index - 1}) + F({index - 2})
            </span>
            <span style={{ color: "rgba(255,255,255,0.3)", marginLeft: 12 }}>
              {FIB[index]}={FIB[index - 1]}+{FIB[index - 2]}
            </span>
          </>
    },
    double: {
      bg: "rgba(224,64,64,0.05)",
      bdr: "rgba(224,64,64,0.25)",
      hdr: RED,
      title: "RESOLVE — double splits apart",
      eq:
        index === 0 ?
          <>
            <span style={{ color: RED }}>2×F(0)</span>
            {" → "}
            <span style={{ color: GREEN }}>F(1)</span>
            <span style={{ color: "rgba(255,255,255,0.3)", marginLeft: 12 }}>
              2×{FIB[0]}={FIB[1]}
            </span>
          </>
        : index === 1 ?
          <>
            <span style={{ color: RED }}>2×F(1)</span>
            {" → "}
            <span style={{ color: GREEN }}>F(2)+F(0)</span>
            <span style={{ color: "rgba(255,255,255,0.3)", marginLeft: 12 }}>
              2×{FIB[1]}={FIB[2]}+{FIB[0]}
            </span>
          </>
        : <>
            <span style={{ color: RED }}>2×F({index})</span>
            {" → "}
            <span style={{ color: GREEN }}>
              F({index + 1})+F({index - 2})
            </span>
            <span style={{ color: "rgba(255,255,255,0.3)", marginLeft: 12 }}>
              2×{FIB[index]}={FIB[index + 1]}+{FIB[index - 2]}
            </span>
          </>
    }
  };
  const c = configs[rule];
  if (!c) return null;
  return (
    <div
      style={{
        padding: "10px 14px",
        background: c.bg,
        border: `1px solid ${c.bdr}`,
        borderRadius: 8,
        marginBottom: 12,
        fontFamily: MONO
      }}
    >
      <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: 1, marginBottom: 3, color: c.hdr }}>{c.title}</div>
      <div style={{ fontSize: 12, color: "rgba(255,255,255,0.5)" }}>{c.eq}</div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB: ARITHMETIC — addition and subtraction with cascade playback
// ═══════════════════════════════════════════════════════════════════

function ArithmeticTab() {
  const [a, setA] = useState(12);
  const [b, setB] = useState(7);
  const [op, setOp] = useState("+");
  const [steps, setSteps] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [mode, setMode] = useState("idle");
  const timerRef = useRef(null);

  const compute = useCallback(() => {
    const zA = toZeckendorf(a);
    const zB = toZeckendorf(b);
    let raw;
    if (op === "+") {
      raw = zA.map((v, i) => v + zB[i]);
    } else {
      // Subtraction: ensure a >= b, then cell-wise subtract
      if (a < b) {
        raw = zB.map((v, i) => v - zA[i]);
      } else {
        raw = zA.map((v, i) => v - zB[i]);
      }
    }
    const normSteps = fullNormalize(raw);
    setSteps(normSteps);
    setCurrentStep(0);
    setMode("result");
    setPlaying(false);
  }, [a, b, op]);

  useEffect(() => {
    if (playing && currentStep < steps.length - 1) {
      timerRef.current = setTimeout(() => setCurrentStep(s => s + 1), 800);
      return () => clearTimeout(timerRef.current);
    } else if (playing) setPlaying(false);
  }, [playing, currentStep, steps.length]);

  const play = () => {
    if (steps.length === 0) return;
    if (currentStep >= steps.length - 1) setCurrentStep(0);
    setPlaying(true);
  };

  const current = steps[currentStep] || { digits: [], rule: null, sources: [], targets: [] };
  const isSettled = currentStep === steps.length - 1 && steps.length > 0;
  const isRaw = currentStep === 0;
  const zA = toZeckendorf(a);
  const zB = toZeckendorf(b);
  const currentViolations = isSettled ? [] : findViolations(current.digits);

  // Count how many of each rule type fired across all steps
  const ruleCounts = useMemo(() => {
    const counts = { carry: 0, double: 0, borrow: 0 };
    steps.forEach(s => {
      if (s.rule) counts[s.rule] = (counts[s.rule] || 0) + 1;
    });
    return counts;
  }, [steps]);

  const inputStyle = {
    width: 68,
    padding: "7px 8px",
    background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(212,160,23,0.3)",
    borderRadius: 6,
    color: GOLD,
    fontFamily: MONO,
    fontSize: 15,
    textAlign: "center",
    outline: "none"
  };
  const btnBase = {
    padding: "7px 16px",
    border: "1px solid rgba(212,160,23,0.4)",
    borderRadius: 6,
    background: "rgba(212,160,23,0.1)",
    color: GOLD,
    cursor: "pointer",
    fontFamily: MONO,
    fontSize: 11,
    letterSpacing: 1,
    transition: "all 0.2s ease"
  };

  return (
    <div>
      {/* Input row */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 10,
          marginBottom: 24,
          flexWrap: "wrap"
        }}
      >
        <input
          type="number"
          min={0}
          max={300}
          value={a}
          onChange={e => {
            setA(Math.min(300, Math.max(0, +e.target.value || 0)));
            setMode("idle");
          }}
          style={inputStyle}
        />
        {/* Operation toggle */}
        <button
          onClick={() => {
            setOp(op === "+" ? "−" : "+");
            setMode("idle");
          }}
          style={{ ...btnBase, width: 36, fontSize: 18, fontWeight: 300, padding: "4px 0", color: op === "−" ? GREEN : GOLD }}
        >
          {op}
        </button>
        <input
          type="number"
          min={0}
          max={300}
          value={b}
          onChange={e => {
            setB(Math.min(300, Math.max(0, +e.target.value || 0)));
            setMode("idle");
          }}
          style={inputStyle}
        />
        <button
          onClick={compute}
          style={{ ...btnBase, background: "rgba(212,160,23,0.2)" }}
        >
          POUR
        </button>
      </div>

      {op === "−" && a < b && mode === "idle" && (
        <div style={{ textAlign: "center", fontSize: 11, color: ORANGE, fontFamily: MONO, marginBottom: 12 }}>
          A {"<"} B — will compute |A−B| = {b - a}
        </div>
      )}

      {/* Source registers */}
      <div
        style={{
          padding: 14,
          background: "rgba(255,255,255,0.02)",
          borderRadius: 10,
          border: "1px solid rgba(255,255,255,0.05)",
          marginBottom: 14
        }}
      >
        <div style={{ fontSize: 11, color: "rgba(212,160,23,0.5)", marginBottom: 5, fontFamily: MONO, letterSpacing: 2 }}>
          A = {a}
        </div>
        <Register digits={zA} />
        <div style={{ height: 10 }} />
        <div style={{ fontSize: 11, color: "rgba(212,160,23,0.5)", marginBottom: 5, fontFamily: MONO, letterSpacing: 2 }}>
          B = {b}
        </div>
        <Register digits={zB} />
      </div>

      {/* Normalization cascade */}
      {mode === "result" && steps.length > 0 && (
        <div
          style={{
            padding: 16,
            background: "rgba(212,160,23,0.02)",
            borderRadius: 10,
            border: "1px solid rgba(212,160,23,0.12)",
            marginBottom: 14
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
            <div style={{ fontSize: 10, fontFamily: MONO, color: "rgba(255,255,255,0.25)", letterSpacing: 2 }}>
              NORMALIZATION CASCADE
            </div>
            <div style={{ fontSize: 11, fontFamily: MONO, color: "rgba(255,255,255,0.3)" }}>
              {currentStep + 1} / {steps.length}
            </div>
          </div>

          <RuleBanner
            step={current}
            isSettled={isSettled}
            isRaw={isRaw}
          />

          <div style={{ fontSize: 11, color: "rgba(212,160,23,0.5)", marginBottom: 5, fontFamily: MONO, letterSpacing: 2 }}>
            {op === "+" ? "A + B" : "|A − B|"}
          </div>
          <Register
            digits={current.digits}
            sources={isRaw ? [] : current.sources}
            targets={isRaw ? [] : current.targets}
            violations={currentViolations}
            settled={isSettled}
          />
          <div style={{ marginTop: 4, fontSize: 13, color: "rgba(255,255,255,0.45)", fontFamily: MONO }}>
            = {fromDigits(current.digits)}
          </div>

          {!isSettled && currentViolations.length > 0 && (
            <div style={{ fontSize: 10, fontFamily: MONO, color: "rgba(255,160,40,0.6)", marginTop: 5 }}>
              {currentViolations.length} violation{currentViolations.length > 1 ? "s" : ""} remaining
            </div>
          )}

          {/* Playback controls */}
          <div style={{ display: "flex", gap: 8, justifyContent: "center", marginTop: 12 }}>
            <button
              onClick={() => {
                setPlaying(false);
                setCurrentStep(s => Math.max(0, s - 1));
              }}
              disabled={currentStep === 0}
              style={{ ...btnBase, opacity: currentStep === 0 ? 0.3 : 1 }}
            >
              ◁
            </button>
            <button
              onClick={playing ? () => setPlaying(false) : play}
              style={{ ...btnBase, minWidth: 72 }}
            >
              {playing ? "PAUSE" : "▷ PLAY"}
            </button>
            <button
              onClick={() => {
                setPlaying(false);
                setCurrentStep(s => Math.min(steps.length - 1, s + 1));
              }}
              disabled={currentStep >= steps.length - 1}
              style={{ ...btnBase, opacity: currentStep >= steps.length - 1 ? 0.3 : 1 }}
            >
              ▷
            </button>
          </div>

          {/* Rule usage summary */}
          {isSettled && steps.length > 1 && (
            <div
              style={{ marginTop: 10, textAlign: "center", fontSize: 10, fontFamily: MONO, color: "rgba(255,255,255,0.3)" }}
            >
              {ruleCounts.carry > 0 && <span style={{ color: GOLD, marginRight: 12 }}>CARRY ×{ruleCounts.carry}</span>}
              {ruleCounts.double > 0 && <span style={{ color: RED, marginRight: 12 }}>RESOLVE ×{ruleCounts.double}</span>}
              {ruleCounts.borrow > 0 && <span style={{ color: GREEN }}>BORROW ×{ruleCounts.borrow}</span>}
            </div>
          )}

          {/* Step log — clickable list of every normalization step */}
          <div
            style={{
              marginTop: 12,
              padding: 8,
              background: "rgba(0,0,0,0.3)",
              borderRadius: 6,
              maxHeight: 170,
              overflowY: "auto"
            }}
          >
            {steps.map((s, i) => {
              const vis = s.digits.slice(
                0,
                Math.max(
                  8,
                  s.digits.reduce((m, v, j) => (v ? Math.max(m, j + 1) : m), 0)
                )
              );
              const isCur = i === currentStep;
              const viols = i < steps.length - 1 ? findViolations(s.digits) : [];
              const ruleColors = { carry: GOLD, double: RED, borrow: GREEN };
              return (
                <div
                  key={i}
                  onClick={() => {
                    setPlaying(false);
                    setCurrentStep(i);
                  }}
                  style={{
                    fontSize: 11,
                    fontFamily: MONO,
                    padding: "3px 8px",
                    cursor: "pointer",
                    color: isCur ? "#fff" : "rgba(255,255,255,0.2)",
                    background: isCur ? "rgba(212,160,23,0.1)" : "transparent",
                    borderLeft: isCur ? `2px solid ${GOLD}` : "2px solid transparent",
                    borderRadius: 2,
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 1
                  }}
                >
                  <span>
                    {i === 0 ?
                      "raw"
                    : <>
                        <span style={{ color: isCur ? ruleColors[s.rule] || "#fff" : "inherit" }}>{s.rule}</span> @F({s.index}
                        )
                      </>
                    }
                  </span>
                  <span style={{ display: "flex", gap: 6, alignItems: "center" }}>
                    <span style={{ color: isCur ? "rgba(255,255,255,0.4)" : "rgba(255,255,255,0.1)" }}>
                      [{vis.slice().reverse().join("")}]
                    </span>
                    {viols.length > 0 && (
                      <span
                        style={{
                          fontSize: 9,
                          padding: "1px 4px",
                          borderRadius: 3,
                          background: "rgba(255,160,40,0.15)",
                          color: ORANGE
                        }}
                      >
                        {viols.length}
                      </span>
                    )}
                    {viols.length === 0 && i === steps.length - 1 && (
                      <span
                        style={{
                          fontSize: 9,
                          padding: "1px 4px",
                          borderRadius: 3,
                          background: "rgba(64,200,120,0.15)",
                          color: GREEN
                        }}
                      >
                        ✓
                      </span>
                    )}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB: TREE — recursive call tree with self-similar highlighting
// ═══════════════════════════════════════════════════════════════════

function TreeTab() {
  const [n, setN] = useState(6);
  const [highlight, setHighlight] = useState(null); // which subtree root to highlight

  // ─── Pan & Zoom state ───
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);

  const tree = useMemo(() => buildTree(n), [n]);
  const nodes = useMemo(() => layoutTree(tree), [tree]);

  // Find max x/y for scaling the SVG coordinate space
  const maxX = Math.max(...nodes.map(n => n.x), 1);
  const maxY = Math.max(...nodes.map(n => n.y), 1);

  const padding = 40;
  const nodeR = 18;
  const svgW = Math.max(500, (maxX + 1) * 50 + padding * 2);
  const svgH = (maxY + 1) * 60 + padding * 2;
  const scaleX = x => padding + (x / Math.max(maxX, 1)) * (svgW - padding * 2);
  const scaleY = y => padding + y * 55;

  // Reset pan/zoom when n changes
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [n]);

  // ─── Mouse wheel → zoom (centered on cursor) ───
  const handleWheel = useCallback(
    e => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1; // scroll down = zoom out, up = zoom in
      const newZoom = Math.max(0.2, Math.min(5, zoom * delta));

      // Zoom toward cursor position within the container
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        // Adjust pan so the point under the cursor stays fixed
        const scale = newZoom / zoom;
        setPan(p => ({
          x: cx - scale * (cx - p.x),
          y: cy - scale * (cy - p.y)
        }));
      }
      setZoom(newZoom);
    },
    [zoom]
  );

  // ─── Mouse drag → pan ───
  const handleMouseDown = useCallback(
    e => {
      // Only start drag on left button, and not if clicking a node
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
      setPan({
        x: panStart.x + (e.clientX - dragStart.x),
        y: panStart.y + (e.clientY - dragStart.y)
      });
    },
    [dragging, dragStart, panStart]
  );

  const handleMouseUp = useCallback(() => {
    setDragging(false);
  }, []);

  // ─── Touch support for mobile ───
  const lastTouchRef = useRef(null);
  const lastPinchRef = useRef(null);

  const handleTouchStart = useCallback(
    e => {
      if (e.touches.length === 1) {
        // Single touch = pan
        lastTouchRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        setPanStart({ ...pan });
      } else if (e.touches.length === 2) {
        // Pinch = zoom
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        lastPinchRef.current = Math.sqrt(dx * dx + dy * dy);
      }
    },
    [pan]
  );

  const handleTouchMove = useCallback(
    e => {
      e.preventDefault();
      if (e.touches.length === 1 && lastTouchRef.current) {
        const dx = e.touches[0].clientX - lastTouchRef.current.x;
        const dy = e.touches[0].clientY - lastTouchRef.current.y;
        setPan({ x: panStart.x + dx, y: panStart.y + dy });
      } else if (e.touches.length === 2 && lastPinchRef.current) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const scale = dist / lastPinchRef.current;
        setZoom(z => Math.max(0.2, Math.min(5, z * scale)));
        lastPinchRef.current = dist;
      }
    },
    [panStart]
  );

  const handleTouchEnd = useCallback(() => {
    lastTouchRef.current = null;
    lastPinchRef.current = null;
  }, []);

  // Attach non-passive wheel listener (React onWheel is passive by default)
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener("wheel", handleWheel, { passive: false });
    return () => el.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

  // Collect all node IDs in a subtree for highlighting
  const collectIds = useCallback(node => {
    if (!node) return new Set();
    const ids = new Set([node.id]);
    collectIds(node.left).forEach(id => ids.add(id));
    collectIds(node.right).forEach(id => ids.add(id));
    return ids;
  }, []);

  // Find the subtree to highlight
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

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };
  const zoomIn = () => setZoom(z => Math.min(5, z * 1.3));
  const zoomOut = () => setZoom(z => Math.max(0.2, z / 1.3));
  const fitToView = () => {
    // Scale so the full SVG fits in the 400px-high viewport
    if (!containerRef.current) return;
    const cw = containerRef.current.clientWidth;
    const ch = 400;
    const fitZoom = Math.min(cw / svgW, ch / svgH, 2);
    setZoom(fitZoom);
    setPan({ x: (cw - svgW * fitZoom) / 2, y: (ch - svgH * fitZoom) / 2 });
  };

  const btnBase = {
    padding: "5px 12px",
    border: "1px solid rgba(212,160,23,0.3)",
    borderRadius: 5,
    background: "rgba(212,160,23,0.1)",
    color: GOLD,
    cursor: "pointer",
    fontFamily: MONO,
    fontSize: 11,
    transition: "all 0.2s ease"
  };
  const zoomBtn = { ...btnBase, padding: "4px 10px", fontSize: 14, lineHeight: 1, minWidth: 32 };

  return (
    <div>
      {/* Controls */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 12,
          marginBottom: 16,
          flexWrap: "wrap"
        }}
      >
        <span style={{ fontFamily: MONO, fontSize: 12, color: "rgba(255,255,255,0.4)" }}>fib(</span>
        <input
          type="range"
          min={2}
          max={9}
          value={n}
          onChange={e => {
            setN(+e.target.value);
            setHighlight(null);
          }}
          style={{ width: 120, accentColor: GOLD }}
        />
        <span style={{ fontFamily: MONO, fontSize: 14, color: GOLD }}>{n}</span>
        <span style={{ fontFamily: MONO, fontSize: 12, color: "rgba(255,255,255,0.4)" }}>)</span>
        <span style={{ fontFamily: MONO, fontSize: 12, color: "rgba(255,255,255,0.25)" }}>
          = {FIB[n - 1] || (n <= 1 ? n : "?")} · {nodes.length} nodes · {nodes.filter(nd => nd.n <= 1).length} leaves
        </span>
      </div>

      {/* Self-similar highlight buttons */}
      <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 10, flexWrap: "wrap" }}>
        <button
          onClick={() => setHighlight(null)}
          style={{ ...btnBase, background: highlight === null ? "rgba(212,160,23,0.2)" : "transparent" }}
        >
          FULL TREE
        </button>
        {n >= 2 && (
          <button
            onClick={() => setHighlight(n - 1)}
            style={{
              ...btnBase,
              background: highlight === n - 1 ? "rgba(212,160,23,0.2)" : "transparent",
              color: highlight === n - 1 ? GOLD : "rgba(212,160,23,0.5)"
            }}
          >
            LEFT: fib({n - 1})
          </button>
        )}
        {n >= 3 && (
          <button
            onClick={() => setHighlight(n - 2)}
            style={{
              ...btnBase,
              background: highlight === n - 2 ? "rgba(64,200,120,0.15)" : "transparent",
              color: highlight === n - 2 ? GREEN : "rgba(64,200,120,0.4)"
            }}
          >
            RIGHT: fib({n - 2})
          </button>
        )}
      </div>

      {/* Zoom controls bar */}
      <div style={{ display: "flex", gap: 4, justifyContent: "center", alignItems: "center", marginBottom: 8 }}>
        <button
          onClick={zoomOut}
          style={zoomBtn}
        >
          −
        </button>
        <div style={{ fontFamily: MONO, fontSize: 10, color: "rgba(255,255,255,0.3)", minWidth: 48, textAlign: "center" }}>
          {(zoom * 100).toFixed(0)}%
        </div>
        <button
          onClick={zoomIn}
          style={zoomBtn}
        >
          +
        </button>
        <button
          onClick={fitToView}
          style={{ ...btnBase, padding: "4px 10px", fontSize: 10 }}
        >
          FIT
        </button>
        <button
          onClick={resetView}
          style={{ ...btnBase, padding: "4px 10px", fontSize: 10 }}
        >
          RESET
        </button>
      </div>

      {/* SVG tree — scrollable & zoomable viewport */}
      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
        style={{
          overflow: "hidden",
          background: "rgba(0,0,0,0.2)",
          borderRadius: 10,
          border: "1px solid rgba(255,255,255,0.05)",
          height: 400,
          cursor: dragging ? "grabbing" : "grab",
          position: "relative",
          touchAction: "none" // prevent browser scroll on touch
        }}
      >
        {/* Hint overlay — fades after interaction */}
        <div
          style={{
            position: "absolute",
            bottom: 8,
            right: 10,
            fontFamily: MONO,
            fontSize: 9,
            color: "rgba(255,255,255,0.15)",
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
                opacity={dimmed ? 0.12 : 0.5}
              >
                {nd.leftRoot && (
                  <line
                    x1={px}
                    y1={py + nodeR}
                    x2={scaleX(nd.leftRoot.x)}
                    y2={scaleY(nd.leftRoot.y) - nodeR}
                    stroke={GOLD}
                    strokeWidth={1.5}
                  />
                )}
                {nd.rightRoot && (
                  <line
                    x1={px}
                    y1={py + nodeR}
                    x2={scaleX(nd.rightRoot.x)}
                    y2={scaleY(nd.rightRoot.y) - nodeR}
                    stroke={GOLD}
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
              fill = nd.n === 1 ? "rgba(212,160,23,0.2)" : "rgba(255,255,255,0.05)";
              stroke = nd.n === 1 ? GOLD : "rgba(255,255,255,0.2)";
              textFill = nd.n === 1 ? GOLD : "rgba(255,255,255,0.4)";
            } else {
              fill = isHighlightRoot ? "rgba(212,160,23,0.25)" : "rgba(212,160,23,0.08)";
              stroke = isHighlightRoot ? GOLD : "rgba(212,160,23,0.3)";
              textFill = "#fff";
            }
            return (
              <g
                key={`n-${nd.id}`}
                style={{ cursor: nd.n >= 2 ? "pointer" : "default" }}
                onClick={e => {
                  // Don't trigger node click if we just finished a drag
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
                  fontSize={isLeaf ? 13 : 11}
                  fontFamily={MONO}
                  fontWeight={isLeaf ? 700 : 400}
                >
                  {isLeaf ? nd.n : `(${nd.n})`}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Insight box */}
      <div
        style={{
          marginTop: 14,
          padding: 12,
          background: "rgba(212,160,23,0.04)",
          borderRadius: 8,
          border: "1px solid rgba(212,160,23,0.12)",
          fontFamily: MONO,
          fontSize: 11,
          color: "rgba(255,255,255,0.4)",
          lineHeight: 1.8
        }}
      >
        <div>
          <span style={{ color: GOLD }}>Self-similar structure:</span> fib({n}) contains a complete fib({n - 1}) tree on the
          left and a complete fib({n - 2}) tree on the right.
        </div>
        <div>
          <span style={{ color: GOLD }}>Leaf count = value:</span> {nodes.filter(nd => nd.n === 1).length} ones +{" "}
          {nodes.filter(nd => nd.n === 0).length} zeros = the tree has {nodes.filter(nd => nd.n <= 1).length} leaf nodes, of
          which {nodes.filter(nd => nd.n === 1).length} are 1-leaves — and fib({n}) = {FIB[n - 1] || n}.
        </div>
        <div>
          <span style={{ color: GOLD }}>Click any node</span> to highlight its subtree. Scroll to zoom, drag to pan.
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB: PROPERTIES — the four intrinsic advantages
// ═══════════════════════════════════════════════════════════════════

function PropertiesTab() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* 1. REDUNDANCY DETECTION */}
      <PropertyCard
        title="Built-In Redundancy Detection"
        color={ORANGE}
        subtitle="The adjacency constraint makes invalid states immediately visible"
      >
        <RedundancyDemo />
      </PropertyCard>

      {/* 2. COMPRESSION DELIMITER */}
      <PropertyCard
        title="Natural Compression Delimiter"
        color={GREEN}
        subtitle="'11' can never appear inside a valid number — free end-of-number marker"
      >
        <CompressionDemo />
      </PropertyCard>

      {/* 3. GENTLE OVERFLOW */}
      <PropertyCard
        title="Gentle Overflow"
        color={GOLD}
        subtitle="Capacity grows by φ ≈ 1.618× per cell, not 2×"
      >
        <OverflowDemo />
      </PropertyCard>

      {/* 4. SELF-SIMILAR SCALING (reference to Tree tab) */}
      <PropertyCard
        title="Self-Similar Scaling"
        color={GOLD}
        subtitle="Every structure at scale n contains n-1 and n-2 as substructures"
      >
        <div style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.4)", lineHeight: 1.8 }}>
          <div>
            Explore this property interactively in the <span style={{ color: GOLD }}>TREE</span> tab.
          </div>
          <div style={{ marginTop: 8 }}>
            The recursion F(n) = F(n-1) + F(n-2) means every Fibonacci structure embeds two smaller copies of itself. This
            maps naturally to fractal geometry, branching processes (botany, river networks), and any domain where growth
            follows the golden ratio spiral.
          </div>
          <div style={{ marginTop: 8 }}>
            In the φ-register, this manifests as: the normalization rules that work at scale n are the <em>same rules</em>{" "}
            that work at scale n-1. The machine is scale-invariant.
          </div>
        </div>
      </PropertyCard>
    </div>
  );
}

/** Reusable card wrapper for property demos */
function PropertyCard({ title, subtitle, color, children }) {
  return (
    <div
      style={{
        padding: 16,
        background: "rgba(255,255,255,0.02)",
        borderRadius: 10,
        border: `1px solid rgba(255,255,255,0.06)`
      }}
    >
      <div style={{ fontSize: 13, fontFamily: MONO, fontWeight: 700, color, letterSpacing: 1, marginBottom: 3 }}>{title}</div>
      <div style={{ fontSize: 10, fontFamily: MONO, color: "rgba(255,255,255,0.3)", marginBottom: 12 }}>{subtitle}</div>
      {children}
    </div>
  );
}

/** REDUNDANCY: Show valid vs total bit patterns for a small register */
function RedundancyDemo() {
  const width = 6;
  const total = 1 << width; // 2^width
  // Generate all bit patterns and check validity (no adjacent 1s)
  const patterns = [];
  for (let i = 0; i < total; i++) {
    const bits = [];
    for (let b = 0; b < width; b++) bits.push((i >> b) & 1);
    let valid = true;
    for (let b = 0; b < width - 1; b++) {
      if (bits[b] === 1 && bits[b + 1] === 1) {
        valid = false;
        break;
      }
    }
    const value = bits.reduce((s, d, j) => s + d * (FIB[j] || 0), 0);
    patterns.push({ bits, valid, value, key: i });
  }
  const validCount = patterns.filter(p => p.valid).length;

  return (
    <div>
      <div style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 10 }}>
        {width}-cell register: <span style={{ color: GREEN }}>{validCount} valid</span> / {total} total patterns
        <span style={{ color: "rgba(255,255,255,0.25)", marginLeft: 8 }}>
          ({((validCount / total) * 100).toFixed(1)}% density — F({width + 2}) = {FIB[width + 1]})
        </span>
      </div>
      {/* Grid of patterns */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(90px, 1fr))",
          gap: 3,
          maxHeight: 200,
          overflowY: "auto"
        }}
      >
        {patterns.map(p => (
          <div
            key={p.key}
            style={{
              fontFamily: MONO,
              fontSize: 10,
              padding: "2px 4px",
              borderRadius: 3,
              background: p.valid ? "rgba(64,200,120,0.08)" : "rgba(255,255,255,0.02)",
              color: p.valid ? GREEN : "rgba(255,255,255,0.12)",
              border: p.valid ? "1px solid rgba(64,200,120,0.2)" : "1px solid rgba(255,255,255,0.04)",
              display: "flex",
              justifyContent: "space-between"
            }}
          >
            <span>{p.bits.slice().reverse().join("")}</span>
            <span style={{ color: p.valid ? "rgba(64,200,120,0.5)" : "rgba(255,255,255,0.08)" }}>={p.value}</span>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 8, fontFamily: MONO, fontSize: 10, color: "rgba(255,255,255,0.25)", lineHeight: 1.7 }}>
        Any bit corruption that creates adjacent 1s is <span style={{ color: RED }}>immediately detectable</span> — the
        representation itself is a checksum. Binary has no such property: every bit pattern is valid, corruption is silent.
      </div>
    </div>
  );
}

/** COMPRESSION: Show Fibonacci universal coding with 11 delimiter */
function CompressionDemo() {
  const [inputStr, setInputStr] = useState("3 7 1 5 12");
  const numbers = inputStr
    .split(/[,\s]+/)
    .map(Number)
    .filter(n => n > 0 && n <= 200);

  // Fibonacci universal code: encode each number as its Zeckendorf rep (LSB first) + "1" terminator
  // This means "11" is the end-of-number signal (the final bit of the number + the terminator)
  const encoded = numbers.map(num => {
    const z = toZeckendorf(num);
    const maxI = z.reduce((m, v, j) => (v ? Math.max(m, j) : m), 0);
    const bits = z.slice(0, maxI + 1); // LSB first, trim trailing zeros
    return { num, bits: [...bits, 1], raw: bits }; // append terminator "1"
  });

  const stream = encoded.flatMap((e, idx) =>
    e.bits.map((b, bi) => ({
      bit: b,
      numIdx: idx,
      // Is this bit part of the "11" delimiter? It's the last two bits of each encoded number.
      isDelimiter: bi >= e.bits.length - 2 && b === 1 && (bi === e.bits.length - 1 || e.bits[bi + 1] === 1)
    }))
  );

  return (
    <div>
      <input
        value={inputStr}
        onChange={e => setInputStr(e.target.value)}
        style={{
          width: "100%",
          padding: "6px 8px",
          background: "rgba(255,255,255,0.05)",
          border: "1px solid rgba(64,200,120,0.3)",
          borderRadius: 5,
          color: GREEN,
          fontFamily: MONO,
          fontSize: 12,
          outline: "none",
          marginBottom: 10
        }}
        placeholder="Enter numbers separated by spaces"
      />
      {/* Bitstream visualization */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 1, marginBottom: 8 }}>
        {stream.map((s, i) => {
          const prevNumIdx = i > 0 ? stream[i - 1].numIdx : -1;
          const isEndBit = i < stream.length - 1 && stream[i + 1]?.numIdx !== s.numIdx;
          const isTerminator = isEndBit || i === stream.length - 1;
          const isPrevEnd = i > 0 && stream[i - 1]?.numIdx !== s.numIdx && s.numIdx !== prevNumIdx;
          return (
            <span
              key={i}
              style={{
                fontFamily: MONO,
                fontSize: 14,
                fontWeight: 700,
                color:
                  isTerminator && s.bit === 1 ? GREEN
                  : (
                    i > 0 &&
                    stream[i - 1]?.numIdx === s.numIdx &&
                    isEndBit === false &&
                    s.bit === 1 &&
                    i + 1 < stream.length &&
                    stream[i + 1]?.bit === 1 &&
                    stream[i + 1]?.numIdx === s.numIdx
                  ) ?
                    GOLD
                  : s.bit === 1 ? GOLD
                  : "rgba(255,255,255,0.2)",
                marginLeft: s.numIdx !== prevNumIdx && i > 0 ? 8 : 0,
                borderBottom: isTerminator ? `2px solid ${GREEN}` : "2px solid transparent",
                padding: "0 1px"
              }}
            >
              {s.bit}
            </span>
          );
        })}
      </div>
      {/* Decoded values */}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {encoded.map((e, i) => (
          <div
            key={i}
            style={{
              fontFamily: MONO,
              fontSize: 10,
              color: "rgba(255,255,255,0.35)",
              padding: "2px 6px",
              background: "rgba(64,200,120,0.06)",
              borderRadius: 4,
              border: "1px solid rgba(64,200,120,0.15)"
            }}
          >
            {e.num} → {e.raw.join("")}
            <span style={{ color: GREEN, fontWeight: 700 }}>|1</span>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 8, fontFamily: MONO, fontSize: 10, color: "rgba(255,255,255,0.25)", lineHeight: 1.7 }}>
        Since no valid Zeckendorf number contains adjacent 1s, appending a <span style={{ color: GREEN }}>1</span> creates a{" "}
        <span style={{ color: GREEN }}>11</span> pattern that unambiguously marks the boundary. Zero framing overhead — the
        delimiter falls out of the representation constraint itself.
      </div>
    </div>
  );
}

/** GENTLE OVERFLOW: Compare φ-growth vs binary doubling */
function OverflowDemo() {
  const cells = Array.from({ length: 12 }, (_, i) => i + 1);

  return (
    <div>
      <div style={{ display: "flex", gap: 2, alignItems: "flex-end", height: 140, marginBottom: 8 }}>
        {cells.map(c => {
          const fibMax = FIB[c] - 1 || 1; // max representable value
          const binMax = (1 << c) - 1;
          const maxVal = Math.max(binMax, 200);
          const fibH = Math.min((fibMax / maxVal) * 120, 120);
          const binH = Math.min((binMax / maxVal) * 120, 120);
          return (
            <div
              key={c}
              style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2, flex: 1 }}
            >
              <div style={{ display: "flex", gap: 1, alignItems: "flex-end", height: 120 }}>
                {/* φ bar */}
                <div
                  style={{
                    width: 12,
                    height: fibH,
                    background: `linear-gradient(to top, rgba(212,160,23,0.3), rgba(212,160,23,0.6))`,
                    borderRadius: "3px 3px 0 0",
                    transition: "height 0.3s"
                  }}
                  title={`φ: max ${fibMax}`}
                />
                {/* Binary bar */}
                <div
                  style={{
                    width: 12,
                    height: binH,
                    background: `linear-gradient(to top, rgba(255,255,255,0.1), rgba(255,255,255,0.25))`,
                    borderRadius: "3px 3px 0 0",
                    transition: "height 0.3s"
                  }}
                  title={`Binary: max ${binMax}`}
                />
              </div>
              <div style={{ fontSize: 8, fontFamily: MONO, color: "rgba(255,255,255,0.25)" }}>{c}</div>
            </div>
          );
        })}
      </div>
      {/* Legend */}
      <div
        style={{ display: "flex", gap: 16, fontSize: 10, fontFamily: MONO, color: "rgba(255,255,255,0.3)", marginBottom: 8 }}
      >
        <span>
          <span
            style={{
              display: "inline-block",
              width: 10,
              height: 10,
              background: "rgba(212,160,23,0.5)",
              borderRadius: 2,
              marginRight: 4,
              verticalAlign: "middle"
            }}
          />
          φ capacity (×1.618/cell)
        </span>
        <span>
          <span
            style={{
              display: "inline-block",
              width: 10,
              height: 10,
              background: "rgba(255,255,255,0.2)",
              borderRadius: 2,
              marginRight: 4,
              verticalAlign: "middle"
            }}
          />
          Binary capacity (×2/cell)
        </span>
      </div>
      {/* Data table */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "40px 1fr 1fr 1fr",
          gap: "2px 8px",
          fontFamily: MONO,
          fontSize: 10,
          color: "rgba(255,255,255,0.3)"
        }}
      >
        <div style={{ color: "rgba(255,255,255,0.15)" }}>cells</div>
        <div style={{ color: GOLD }}>φ max</div>
        <div>bin max</div>
        <div style={{ color: "rgba(255,255,255,0.15)" }}>ratio</div>
        {[4, 6, 8, 10, 12].map(c => (
          <React.Fragment key={c}>
            <div>{c}</div>
            <div style={{ color: GOLD }}>{FIB[c] - 1}</div>
            <div>{(1 << c) - 1}</div>
            <div style={{ color: "rgba(255,255,255,0.15)" }}>{((1 << c) / FIB[c]).toFixed(1)}×</div>
          </React.Fragment>
        ))}
      </div>
      <div style={{ marginTop: 8, fontFamily: MONO, fontSize: 10, color: "rgba(255,255,255,0.25)", lineHeight: 1.7 }}>
        Binary doubles capacity per cell — explosive growth that wastes headroom. The φ-register grows by the golden ratio,
        spiraling gently. Adding one cell increases capacity by ~61.8%, not 100%.
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// MAIN APP — tabbed interface
// ═══════════════════════════════════════════════════════════════════

export default function PhiRegister() {
  const [tab, setTab] = useState("arithmetic");

  const tabs = [
    { id: "arithmetic", label: "ARITHMETIC" },
    { id: "tree", label: "TREE" },
    { id: "properties", label: "PROPERTIES" }
  ];

  return (
    <div style={{ minHeight: "100vh", background: BG, color: "#e8e0d0", padding: "28px 20px", fontFamily: SERIF }}>
      <link
        href="https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@300;400;600&family=JetBrains+Mono:wght@400;700&display=swap"
        rel="stylesheet"
      />

      <div style={{ maxWidth: 760, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 28, textAlign: "center" }}>
          <h1
            style={{
              fontSize: 26,
              fontWeight: 300,
              color: GOLD,
              margin: 0,
              letterSpacing: 6,
              textTransform: "uppercase",
              fontFamily: SERIF
            }}
          >
            φ-Register
          </h1>
          <div style={{ fontSize: 10, color: "rgba(212,160,23,0.4)", marginTop: 5, fontFamily: MONO, letterSpacing: 3 }}>
            FIBONACCI-WEIGHTED ARITHMETIC MACHINE
          </div>
        </div>

        {/* Tab bar */}
        <div style={{ display: "flex", gap: 2, marginBottom: 24, justifyContent: "center" }}>
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                padding: "8px 20px",
                fontFamily: MONO,
                fontSize: 11,
                letterSpacing: 2,
                border: "1px solid rgba(212,160,23,0.2)",
                borderRadius: 5,
                cursor: "pointer",
                background: tab === t.id ? "rgba(212,160,23,0.15)" : "transparent",
                color: tab === t.id ? GOLD : "rgba(212,160,23,0.35)",
                transition: "all 0.2s ease"
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {tab === "arithmetic" && <ArithmeticTab />}
        {tab === "tree" && <TreeTab />}
        {tab === "properties" && <PropertiesTab />}

        {/* Axiom reference — always visible at bottom */}
        <div
          style={{
            marginTop: 28,
            padding: 12,
            background: "rgba(255,255,255,0.02)",
            borderRadius: 8,
            border: "1px solid rgba(255,255,255,0.04)"
          }}
        >
          <div style={{ fontSize: 9, fontFamily: MONO, color: "rgba(212,160,23,0.35)", letterSpacing: 2, marginBottom: 6 }}>
            THE SINGLE AXIOM — THREE READINGS
          </div>
          <div style={{ fontSize: 11, fontFamily: MONO, color: "rgba(255,255,255,0.25)", lineHeight: 2.0 }}>
            <div>
              <span style={{ color: GOLD }}>CARRY</span> F(k) + F(k+1) = F(k+2) — adjacent 1s merge upward
            </div>
            <div>
              <span style={{ color: RED }}>RESOLVE</span> 2·F(k) = F(k+1) + F(k−2) — doubles split apart
            </div>
            <div>
              <span style={{ color: GREEN }}>BORROW</span> F(k) = F(k−1) + F(k−2) — expand downward for subtraction
            </div>
          </div>
        </div>

        {/* Legend */}
        <div
          style={{
            display: "flex",
            gap: 14,
            justifyContent: "center",
            flexWrap: "wrap",
            marginTop: 14,
            fontSize: 9,
            fontFamily: MONO,
            color: "rgba(255,255,255,0.2)"
          }}
        >
          {[
            [`2px solid ${RED}`, "rgba(224,64,64,0.3)", "consumed"],
            [`2px solid ${GREEN}`, "rgba(64,200,120,0.3)", "produced"],
            [`1px dashed ${RED}`, "rgba(224,64,64,0.1)", "double/negative"],
            [`1px dashed ${ORANGE}`, "rgba(255,160,40,0.1)", "adjacent"]
          ].map(([bdr, bg, label]) => (
            <span key={label}>
              <span
                style={{
                  display: "inline-block",
                  width: 9,
                  height: 9,
                  borderRadius: 2,
                  background: bg,
                  border: bdr,
                  marginRight: 3,
                  verticalAlign: "middle",
                  opacity: 0.7
                }}
              />
              {label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
