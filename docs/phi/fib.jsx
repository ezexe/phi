import { useState, useEffect, useRef, useCallback, useMemo } from "react";

// ═══════════════════════════════════════════════════════════════
// FIBONACCI METHOD VISUALIZER
// Step-through every computation method, see what changes when
// ═══════════════════════════════════════════════════════════════

const GOLD = "#c9a227";
const GOLD_DIM = "#6b5a1e";
const RED = "#d94040";
const GREEN = "#3dba6f";
const CYAN = "#4ac0c0";
const PURPLE = "#9d7aed";
const ORANGE = "#e08a30";
const BG = "#08080b";
const SURFACE = "#101016";
const SURFACE2 = "#181820";
const BORDER = "#252530";
const TEXT = "#d4d4d8";
const TEXT_DIM = "#6b6b78";
const MONO = "'JetBrains Mono', 'Fira Code', monospace";
const SANS = "'DM Sans', 'Segoe UI', sans-serif";

// ─── Trace generators for each method ───

function traceNaive(n) {
  const steps = [];
  const tree = {};
  let callId = 0;

  function fib(k, parentId = null, side = null) {
    const id = callId++;
    const node = { id, n: k, parentId, side, result: null, children: [] };
    if (parentId !== null && tree[parentId]) tree[parentId].children.push(id);
    tree[id] = node;

    if (k <= 1) {
      node.result = k;
      steps.push({
        type: "base",
        callId: id,
        n: k,
        result: k,
        desc: `fib(${k}) = ${k}  [base case]`,
        activeNodes: [id],
        tree: JSON.parse(JSON.stringify(tree))
      });
      return k;
    }

    steps.push({
      type: "call",
      callId: id,
      n: k,
      desc: `fib(${k}) → call fib(${k - 1}) + fib(${k - 2})`,
      activeNodes: [id],
      tree: JSON.parse(JSON.stringify(tree))
    });

    const left = fib(k - 1, id, "L");
    const right = fib(k - 2, id, "R");
    node.result = left + right;

    steps.push({
      type: "return",
      callId: id,
      n: k,
      result: left + right,
      left,
      right,
      desc: `fib(${k}) = fib(${k - 1}) + fib(${k - 2}) = ${left} + ${right} = ${left + right}`,
      activeNodes: [id],
      tree: JSON.parse(JSON.stringify(tree))
    });

    return left + right;
  }

  fib(n);
  return steps;
}

function traceMemo(n) {
  const steps = [];
  const memo = {};
  let callCount = 0;

  function fib(k) {
    callCount++;
    if (memo[k] !== undefined) {
      steps.push({
        type: "cache_hit",
        n: k,
        result: memo[k],
        memo: { ...memo },
        callCount,
        desc: `fib(${k}) → CACHE HIT = ${memo[k]}`
      });
      return memo[k];
    }

    if (k <= 1) {
      memo[k] = k;
      steps.push({
        type: "base",
        n: k,
        result: k,
        memo: { ...memo },
        callCount,
        desc: `fib(${k}) = ${k}  [base case, cached]`
      });
      return k;
    }

    steps.push({
      type: "call",
      n: k,
      memo: { ...memo },
      callCount,
      desc: `fib(${k}) → not cached, computing...`
    });

    const left = fib(k - 1);
    const right = fib(k - 2);
    memo[k] = left + right;

    steps.push({
      type: "return",
      n: k,
      result: left + right,
      left,
      right,
      memo: { ...memo },
      callCount,
      desc: `fib(${k}) = ${left} + ${right} = ${left + right}  [cached]`
    });

    return left + right;
  }

  fib(n);
  return steps;
}

function traceIterative(n) {
  const steps = [];
  if (n === 0) {
    steps.push({ i: 0, a: 0, b: 1, result: 0, desc: "fib(0) = 0" });
    return steps;
  }
  if (n === 1) {
    steps.push({ i: 0, a: 0, b: 1, desc: "Start: a=0, b=1" });
    steps.push({ i: 1, a: 0, b: 1, result: 1, desc: "fib(1) = 1" });
    return steps;
  }

  let a = 0,
    b = 1;
  steps.push({ i: 0, a, b, desc: "Initialize: a=0, b=1", cells: [0, 1] });

  for (let i = 2; i <= n; i++) {
    const newB = a + b;
    steps.push({
      i,
      a,
      b,
      newB,
      desc: `Step ${i}: a + b = ${a} + ${b} = ${newB}`,
      op: `${a} + ${b}`,
      cells: [a, b, newB],
      shift: true
    });
    a = b;
    b = newB;
    steps.push({
      i,
      a,
      b,
      desc: `Shift: a←${a}, b←${b}`,
      cells: [a, b],
      shifted: true,
      result: i === n ? b : undefined
    });
  }
  return steps;
}

function traceMatrix(n) {
  const steps = [];
  const matMul = (A, B) => [
    [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
    [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]
  ];

  if (n <= 1) {
    steps.push({
      desc: `fib(${n}) = ${n}`,
      result: n,
      matrix: [
        [1, 0],
        [0, 1]
      ],
      power: 0
    });
    return steps;
  }

  const Q = [
    [1, 1],
    [1, 0]
  ];
  steps.push({ desc: `Q = [[1,1],[1,0]]  — the Fibonacci matrix`, matrix: Q, power: 1 });

  // Decompose n into binary, compute via repeated squaring
  const bits = n.toString(2);
  let result = [
    [1, 0],
    [0, 1]
  ]; // identity
  let base = Q;

  steps.push({
    desc: `n = ${n} = ${bits}₂  — process ${bits.length} bits via repeated squaring`,
    matrix: result,
    base,
    binary: bits,
    power: 0
  });

  for (let i = bits.length - 1; i >= 0; i--) {
    const bit = bits[bits.length - 1 - i];
    if (bit === "1") {
      result = matMul(result, base);
      steps.push({
        desc: `Bit=${bit}: multiply result × base`,
        matrix: result,
        base,
        bitIndex: bits.length - 1 - i,
        bitVal: 1,
        binary: bits,
        power: n - i
      });
    } else {
      steps.push({
        desc: `Bit=${bit}: skip (no multiply)`,
        matrix: result,
        base,
        bitIndex: bits.length - 1 - i,
        bitVal: 0,
        binary: bits,
        power: n - i
      });
    }
    if (i > 0) {
      base = matMul(base, base);
      steps.push({
        desc: `Square the base matrix`,
        matrix: result,
        base,
        squaring: true,
        binary: bits
      });
    }
  }

  steps.push({
    desc: `Q^${n}[0][1] = ${result[0][1]} = fib(${n})`,
    matrix: result,
    result: result[0][1],
    final: true
  });

  return steps;
}

function traceFastDouble(n) {
  const steps = [];

  function fib(k, depth = 0) {
    if (k === 0) {
      steps.push({ n: k, a: 0, b: 1, type: "base", depth, desc: `fib(0) → (0, 1)` });
      return [0, 1];
    }

    const [a, b] = fib(Math.floor(k / 2), depth + 1);
    const c = a * (2 * b - a);
    const d = a * a + b * b;

    steps.push({
      n: k,
      half: Math.floor(k / 2),
      a,
      b,
      c,
      d,
      type: "double",
      depth,
      desc: `n=${k}, half=${Math.floor(k / 2)}: c = ${a}×(2×${b}−${a}) = ${c}, d = ${a}²+${b}² = ${d}`
    });

    if (k % 2 === 1) {
      steps.push({
        n: k,
        type: "odd_adjust",
        c,
        d,
        resultA: d,
        resultB: c + d,
        depth,
        desc: `n=${k} is odd → return (d, c+d) = (${d}, ${c + d})`
      });
      return [d, c + d];
    } else {
      steps.push({
        n: k,
        type: "even_return",
        c,
        d,
        resultA: c,
        resultB: d,
        depth,
        desc: `n=${k} is even → return (c, d) = (${c}, ${d})`
      });
      return [c, d];
    }
  }

  const [res] = fib(n);
  steps.push({ type: "final", result: res, desc: `fib(${n}) = ${res}` });
  return steps;
}

function traceBinet(n) {
  const steps = [];
  const phi = (1 + Math.sqrt(5)) / 2;
  const psi = (1 - Math.sqrt(5)) / 2;

  steps.push({
    desc: `φ = (1+√5)/2 ≈ ${phi.toFixed(10)}`,
    phi,
    psi,
    type: "constants"
  });

  steps.push({
    desc: `ψ = (1−√5)/2 ≈ ${psi.toFixed(10)}`,
    phi,
    psi,
    type: "constants2"
  });

  const phiN = Math.pow(phi, n);
  const psiN = Math.pow(psi, n);

  steps.push({
    desc: `φ^${n} = ${phiN.toFixed(6)}`,
    phiN,
    psiN,
    type: "power_phi"
  });

  steps.push({
    desc: `ψ^${n} = ${psiN.toFixed(6)}  (vanishes for large n)`,
    phiN,
    psiN,
    type: "power_psi"
  });

  const diff = phiN - psiN;
  const result = diff / Math.sqrt(5);

  steps.push({
    desc: `(φ^${n} − ψ^${n}) / √5 = ${diff.toFixed(6)} / ${Math.sqrt(5).toFixed(6)} = ${result.toFixed(6)}`,
    diff,
    result,
    type: "divide"
  });

  steps.push({
    desc: `round(${result.toFixed(6)}) = ${Math.round(result)}`,
    result: Math.round(result),
    type: "final"
  });

  return steps;
}

function traceGenerator(n) {
  const steps = [];
  let a = 0,
    b = 1;
  const sequence = [];

  for (let i = 0; i <= n; i++) {
    sequence.push(a);
    steps.push({
      i,
      a,
      b,
      yielded: a,
      sequence: [...sequence],
      desc: i === 0 ? `yield 0 (initial)` : `yield ${a}`
    });
    const tmp = a;
    a = b;
    b = tmp + b;
  }
  return steps;
}

// ─── Method definitions ───

const METHODS = [
  {
    id: "naive",
    name: "Naive Recursion",
    complexity: "O(φⁿ)",
    space: "O(n)",
    color: RED,
    trace: traceNaive,
    maxN: 8,
    desc: "Literal recurrence — exposes the self-similar call tree"
  },
  {
    id: "memo",
    name: "Memoized",
    complexity: "O(n)",
    space: "O(n)",
    color: ORANGE,
    trace: traceMemo,
    maxN: 14,
    desc: "Same tree, but cached — redundant subtrees collapse"
  },
  {
    id: "iterative",
    name: "Bottom-Up",
    complexity: "O(n)",
    space: "O(1)",
    color: GREEN,
    trace: traceIterative,
    maxN: 14,
    desc: "Two-cell sliding window — forward propagation"
  },
  {
    id: "matrix",
    name: "Matrix Exp",
    complexity: "O(log n)",
    space: "O(log n)",
    color: CYAN,
    trace: traceMatrix,
    maxN: 14,
    desc: "Repeated squaring on the Q-matrix"
  },
  {
    id: "fastdouble",
    name: "Fast Doubling",
    complexity: "O(log n)",
    space: "O(log n)",
    color: PURPLE,
    trace: traceFastDouble,
    maxN: 14,
    desc: "Algebraic identities — halve n each step"
  },
  {
    id: "binet",
    name: "Binet (φ)",
    complexity: "O(1)*",
    space: "O(1)",
    color: GOLD,
    trace: traceBinet,
    maxN: 14,
    desc: "Closed-form via golden ratio — your number system's eigenvalue"
  },
  {
    id: "generator",
    name: "Generator",
    complexity: "O(1)/yield",
    space: "O(1)",
    color: "#e0e0e0",
    trace: traceGenerator,
    maxN: 14,
    desc: "Corecursive — produces the sequence lazily"
  }
];

// ─── Visualization components ───

function NaiveVis({ step }) {
  if (!step.tree) return null;
  const nodes = Object.values(step.tree);
  if (nodes.length === 0) return null;

  // Layout tree
  const positions = {};
  const levelWidths = {};

  function layout(id, depth, order) {
    if (!positions[depth]) positions[depth] = [];
    positions[depth].push(id);
    const node = step.tree[id];
    if (node.children.length > 0) {
      node.children.forEach((c, i) => layout(c, depth + 1, order * 2 + i));
    }
  }
  layout(0, 0, 0);

  const maxDepth = Math.max(...Object.keys(positions).map(Number));
  const leafCount = nodes.filter(n => n.children.length === 0).length;
  const W = Math.max(400, leafCount * 42);
  const H = (maxDepth + 1) * 52 + 20;

  // Assign x positions based on leaf ordering
  let leafIdx = 0;
  const xPos = {};
  function assignX(id) {
    const node = step.tree[id];
    if (node.children.length === 0) {
      xPos[id] = ((leafIdx + 0.5) / Math.max(leafCount, 1)) * W;
      leafIdx++;
    } else {
      node.children.forEach(c => assignX(c));
      const childXs = node.children.map(c => xPos[c]);
      xPos[id] = childXs.reduce((a, b) => a + b, 0) / childXs.length;
    }
  }
  assignX(0);

  const isActive = id => step.activeNodes && step.activeNodes.includes(id);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: Math.min(H, 300) }}
    >
      {nodes.map(node =>
        node.children.map(cid => (
          <line
            key={`${node.id}-${cid}`}
            x1={xPos[node.id]}
            y1={node.id === 0 ? 18 : Object.keys(positions).find(d => positions[d].includes(node.id)) * 52 + 18}
            x2={xPos[cid]}
            y2={Object.keys(positions).find(d => positions[d].includes(cid)) * 52 + 18}
            stroke={BORDER}
            strokeWidth={1}
          />
        ))
      )}
      {nodes.map(node => {
        const depth = Object.keys(positions).find(d => positions[d].includes(node.id));
        const x = xPos[node.id];
        const y = depth * 52 + 18;
        const active = isActive(node.id);
        const hasResult = node.result !== null;
        return (
          <g key={node.id}>
            <circle
              cx={x}
              cy={y}
              r={16}
              fill={
                active ?
                  hasResult ?
                    GREEN
                  : RED
                : hasResult ?
                  SURFACE2
                : SURFACE
              }
              stroke={
                active ?
                  hasResult ?
                    GREEN
                  : RED
                : hasResult ?
                  GREEN + "60"
                : BORDER
              }
              strokeWidth={active ? 2 : 1}
              opacity={
                active ? 1
                : hasResult ?
                  0.8
                : 0.4
              }
            />
            <text
              x={x}
              y={y + 1}
              textAnchor="middle"
              dominantBaseline="middle"
              fill={
                active ? "#fff"
                : hasResult ?
                  GREEN
                : TEXT_DIM
              }
              fontSize={11}
              fontFamily={MONO}
              fontWeight={active ? 700 : 400}
            >
              {node.n}
            </text>
            {hasResult && (
              <text
                x={x}
                y={y + 30}
                textAnchor="middle"
                fill={active ? "#fff" : TEXT_DIM}
                fontSize={9}
                fontFamily={MONO}
              >
                ={node.result}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

function MemoVis({ step }) {
  const memo = step.memo || {};
  const keys = Object.keys(memo)
    .map(Number)
    .sort((a, b) => a - b);
  const maxKey = keys.length > 0 ? Math.max(...keys) : 0;

  return (
    <div>
      <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 12 }}>
        {Array.from({ length: maxKey + 1 }, (_, i) => {
          const cached = memo[i] !== undefined;
          const isCurrentN = step.n === i;
          const isHit = step.type === "cache_hit" && step.n === i;
          return (
            <div
              key={i}
              style={{
                width: 48,
                height: 48,
                border: `2px solid ${
                  isHit ? ORANGE
                  : isCurrentN ? ORANGE + "80"
                  : cached ? GREEN + "40"
                  : BORDER
                }`,
                borderRadius: 6,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                background:
                  isHit ? ORANGE + "20"
                  : cached ? GREEN + "0a"
                  : SURFACE,
                transition: "all 0.3s"
              }}
            >
              <span style={{ fontSize: 9, color: TEXT_DIM, fontFamily: MONO }}>f({i})</span>
              <span
                style={{
                  fontSize: 14,
                  fontFamily: MONO,
                  fontWeight: 700,
                  color:
                    isHit ? ORANGE
                    : cached ? GREEN
                    : TEXT_DIM + "40"
                }}
              >
                {cached ? memo[i] : "·"}
              </span>
            </div>
          );
        })}
      </div>
      <div style={{ fontFamily: MONO, fontSize: 11, color: TEXT_DIM }}>
        Calls: {step.callCount || 0} {step.type === "cache_hit" && <span style={{ color: ORANGE }}>⚡ HIT</span>}
      </div>
    </div>
  );
}

function IterativeVis({ step }) {
  const cells = step.cells || [];
  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
      {cells.map((v, i) => (
        <div
          key={i}
          style={{ textAlign: "center" }}
        >
          <div
            style={{
              fontSize: 9,
              color:
                i === 0 ? CYAN
                : i === 1 ? GREEN
                : GOLD,
              fontFamily: MONO,
              marginBottom: 3
            }}
          >
            {i === 0 ?
              "a"
            : i === 1 ?
              "b"
            : "a+b"}
          </div>
          <div
            style={{
              width: 56,
              height: 56,
              border: `2px solid ${
                i === 2 ? GOLD
                : i === 1 ? GREEN + "60"
                : CYAN + "60"
              }`,
              borderRadius: 8,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontFamily: MONO,
              fontSize: 20,
              fontWeight: 700,
              color: i === 2 ? GOLD : "#fff",
              background: i === 2 ? GOLD + "15" : SURFACE
            }}
          >
            {v}
          </div>
        </div>
      ))}
      {step.shift && <div style={{ fontSize: 22, color: GOLD, margin: "12px 8px 0" }}>→</div>}
      {step.result !== undefined && (
        <div
          style={{
            marginLeft: 12,
            padding: "6px 14px",
            background: GREEN + "18",
            border: `1px solid ${GREEN}40`,
            borderRadius: 6,
            marginTop: 10,
            fontFamily: MONO,
            fontSize: 14,
            color: GREEN,
            fontWeight: 700
          }}
        >
          = {step.result}
        </div>
      )}
    </div>
  );
}

function MatrixVis({ step }) {
  const m = step.matrix;
  const b = step.base;

  const Matrix = ({ mat, label, highlight }) => (
    <div style={{ textAlign: "center" }}>
      {label && <div style={{ fontSize: 9, color: TEXT_DIM, fontFamily: MONO, marginBottom: 4 }}>{label}</div>}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 2,
          border: `2px solid ${highlight || BORDER}`,
          borderRadius: 6,
          padding: 4,
          background: SURFACE
        }}
      >
        {mat.flat().map((v, i) => (
          <div
            key={i}
            style={{
              width: 52,
              height: 30,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontFamily: MONO,
              fontSize: 12,
              color: highlight || TEXT,
              background: SURFACE2,
              borderRadius: 3
            }}
          >
            {typeof v === "number" && v > 999 ? v.toLocaleString() : v}
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
      {m && (
        <Matrix
          mat={m}
          label="Result"
          highlight={step.final ? GREEN : CYAN}
        />
      )}
      {b && (
        <Matrix
          mat={b}
          label="Base"
          highlight={step.squaring ? PURPLE : null}
        />
      )}
      {step.binary && (
        <div style={{ marginTop: 16 }}>
          <div style={{ fontSize: 9, color: TEXT_DIM, fontFamily: MONO, marginBottom: 4 }}>Binary of n</div>
          <div style={{ display: "flex", gap: 2 }}>
            {step.binary.split("").map((bit, i) => (
              <div
                key={i}
                style={{
                  width: 24,
                  height: 24,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontFamily: MONO,
                  fontSize: 13,
                  fontWeight: 700,
                  background:
                    step.bitIndex === i ?
                      bit === "1" ?
                        CYAN + "30"
                      : RED + "20"
                    : SURFACE2,
                  color:
                    step.bitIndex === i ?
                      bit === "1" ?
                        CYAN
                      : RED
                    : TEXT_DIM,
                  border: `1px solid ${
                    step.bitIndex === i ?
                      bit === "1" ?
                        CYAN + "60"
                      : RED + "40"
                    : BORDER
                  }`,
                  borderRadius: 4
                }}
              >
                {bit}
              </div>
            ))}
          </div>
        </div>
      )}
      {step.result !== undefined && (
        <div
          style={{
            marginTop: 16,
            padding: "8px 14px",
            background: GREEN + "18",
            border: `1px solid ${GREEN}40`,
            borderRadius: 6,
            fontFamily: MONO,
            fontSize: 16,
            color: GREEN,
            fontWeight: 700
          }}
        >
          fib = {step.result}
        </div>
      )}
    </div>
  );
}

function FastDoubleVis({ step }) {
  const barColor =
    step.type === "odd_adjust" ? PURPLE
    : step.type === "final" ? GREEN
    : CYAN;
  return (
    <div>
      {step.type !== "final" && (
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-end" }}>
          {[
            { label: "a", val: step.a ?? step.resultA },
            { label: "b", val: step.b ?? step.resultB },
            step.c !== undefined && { label: "c", val: step.c },
            step.d !== undefined && { label: "d", val: step.d }
          ]
            .filter(Boolean)
            .map((item, i) => (
              <div
                key={i}
                style={{ textAlign: "center" }}
              >
                <div style={{ fontSize: 9, color: TEXT_DIM, fontFamily: MONO, marginBottom: 4 }}>{item.label}</div>
                <div
                  style={{
                    minWidth: 48,
                    padding: "8px 10px",
                    border: `1px solid ${barColor}50`,
                    borderRadius: 6,
                    fontFamily: MONO,
                    fontSize: 16,
                    fontWeight: 700,
                    color: barColor,
                    background: barColor + "10",
                    textAlign: "center"
                  }}
                >
                  {item.val}
                </div>
              </div>
            ))}
        </div>
      )}
      {step.depth !== undefined && step.type !== "final" && (
        <div
          style={{
            marginTop: 8,
            fontFamily: MONO,
            fontSize: 10,
            color: TEXT_DIM,
            paddingLeft: step.depth * 16
          }}
        >
          {"  ".repeat(step.depth)}depth {step.depth} · n={step.n}
          {step.half !== undefined && ` → half=${step.half}`}
        </div>
      )}
      {step.type === "final" && (
        <div
          style={{
            padding: "10px 18px",
            background: GREEN + "18",
            border: `1px solid ${GREEN}40`,
            borderRadius: 6,
            fontFamily: MONO,
            fontSize: 18,
            color: GREEN,
            fontWeight: 700
          }}
        >
          fib = {step.result}
        </div>
      )}
    </div>
  );
}

function BinetVis({ step }) {
  const phi = (1 + Math.sqrt(5)) / 2;
  const barMax = step.phiN || Math.pow(phi, 10);

  return (
    <div>
      {step.type === "constants" && (
        <div style={{ display: "flex", gap: 24, alignItems: "center" }}>
          <div
            style={{
              padding: "12px 20px",
              borderRadius: 8,
              background: `linear-gradient(135deg, ${GOLD}20, ${GOLD}08)`,
              border: `1px solid ${GOLD}40`
            }}
          >
            <span style={{ fontFamily: MONO, fontSize: 20, color: GOLD, fontWeight: 700 }}>φ</span>
            <span style={{ fontFamily: MONO, fontSize: 13, color: TEXT_DIM, marginLeft: 8 }}>= {phi.toFixed(10)}</span>
          </div>
        </div>
      )}
      {step.type === "constants2" && (
        <div
          style={{
            padding: "12px 20px",
            borderRadius: 8,
            background: SURFACE2,
            border: `1px solid ${BORDER}`
          }}
        >
          <span style={{ fontFamily: MONO, fontSize: 20, color: TEXT_DIM, fontWeight: 700 }}>ψ</span>
          <span style={{ fontFamily: MONO, fontSize: 13, color: TEXT_DIM, marginLeft: 8 }}>= {step.psi.toFixed(10)}</span>
          <span style={{ fontFamily: MONO, fontSize: 10, color: TEXT_DIM, marginLeft: 8 }}>(shrinks to 0)</span>
        </div>
      )}
      {(step.type === "power_phi" || step.type === "power_psi") && (
        <div style={{ display: "flex", gap: 12, alignItems: "flex-end" }}>
          <div
            style={{
              height: Math.max(20, Math.min(80, (step.phiN / barMax) * 80)),
              width: 80,
              background: GOLD + "30",
              border: `1px solid ${GOLD}60`,
              borderRadius: 4,
              display: "flex",
              alignItems: "flex-end",
              justifyContent: "center",
              padding: 4
            }}
          >
            <span style={{ fontFamily: MONO, fontSize: 10, color: GOLD }}>{step.phiN.toFixed(2)}</span>
          </div>
          <div
            style={{
              height: Math.max(4, Math.min(80, (Math.abs(step.psiN) / barMax) * 80)),
              width: 80,
              background: TEXT_DIM + "20",
              border: `1px solid ${BORDER}`,
              borderRadius: 4,
              display: "flex",
              alignItems: "flex-end",
              justifyContent: "center",
              padding: 4
            }}
          >
            <span style={{ fontFamily: MONO, fontSize: 10, color: TEXT_DIM }}>{step.psiN.toFixed(4)}</span>
          </div>
        </div>
      )}
      {step.type === "divide" && (
        <div style={{ fontFamily: MONO, fontSize: 14, color: GOLD }}>
          {step.result.toFixed(6)} ≈ {Math.round(step.result)}
        </div>
      )}
      {step.type === "final" && (
        <div
          style={{
            padding: "10px 18px",
            background: GREEN + "18",
            border: `1px solid ${GREEN}40`,
            borderRadius: 6,
            fontFamily: MONO,
            fontSize: 18,
            color: GREEN,
            fontWeight: 700
          }}
        >
          fib = {step.result}
        </div>
      )}
    </div>
  );
}

function GeneratorVis({ step }) {
  const seq = step.sequence || [];
  return (
    <div>
      <div style={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
        {seq.map((v, i) => (
          <div
            key={i}
            style={{
              minWidth: 36,
              padding: "6px 8px",
              background: i === seq.length - 1 ? "#fff1" : SURFACE2,
              border: `1px solid ${i === seq.length - 1 ? "#fff3" : BORDER}`,
              borderRadius: 5,
              fontFamily: MONO,
              fontSize: 13,
              color: i === seq.length - 1 ? "#fff" : TEXT_DIM,
              fontWeight: i === seq.length - 1 ? 700 : 400,
              textAlign: "center"
            }}
          >
            {v}
          </div>
        ))}
      </div>
      <div style={{ marginTop: 8, fontFamily: MONO, fontSize: 10, color: TEXT_DIM }}>
        ← yielded {seq.length} value{seq.length !== 1 ? "s" : ""}
      </div>
    </div>
  );
}

const VIS_MAP = {
  naive: NaiveVis,
  memo: MemoVis,
  iterative: IterativeVis,
  matrix: MatrixVis,
  fastdouble: FastDoubleVis,
  binet: BinetVis,
  generator: GeneratorVis
};

// ─── Step cost comparison bar ───

function CostBar({ methods, n }) {
  const costs = methods.map(m => ({ ...m, steps: m.trace(n).length }));
  const maxSteps = Math.max(...costs.map(c => c.steps));

  return (
    <div style={{ marginTop: 16, padding: 12, background: SURFACE, borderRadius: 8, border: `1px solid ${BORDER}` }}>
      <div style={{ fontFamily: MONO, fontSize: 10, color: TEXT_DIM, marginBottom: 8 }}>STEPS TO COMPUTE fib({n})</div>
      {costs.map(c => (
        <div
          key={c.id}
          style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}
        >
          <div style={{ width: 80, fontFamily: MONO, fontSize: 10, color: c.color, textAlign: "right" }}>{c.name}</div>
          <div
            style={{
              height: 14,
              borderRadius: 3,
              width: `${(c.steps / maxSteps) * 100}%`,
              minWidth: 2,
              background: `linear-gradient(90deg, ${c.color}60, ${c.color}30)`,
              border: `1px solid ${c.color}40`,
              transition: "width 0.4s ease"
            }}
          />
          <div style={{ fontFamily: MONO, fontSize: 10, color: TEXT_DIM }}>{c.steps}</div>
        </div>
      ))}
    </div>
  );
}

// ─── Main App ───

export default function FibViz() {
  const [methodIdx, setMethodIdx] = useState(0);
  const [n, setN] = useState(6);
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const playRef = useRef(null);

  const method = METHODS[methodIdx];
  const steps = useMemo(() => method.trace(n), [methodIdx, n]);
  const step = steps[Math.min(stepIdx, steps.length - 1)];
  const Vis = VIS_MAP[method.id];

  useEffect(() => {
    setStepIdx(0);
    setPlaying(false);
  }, [methodIdx, n]);

  useEffect(() => {
    if (playing) {
      playRef.current = setInterval(() => {
        setStepIdx(prev => {
          if (prev >= steps.length - 1) {
            setPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 500);
    }
    return () => clearInterval(playRef.current);
  }, [playing, steps.length]);

  const maxN = method.maxN;

  return (
    <div
      style={{
        background: BG,
        color: TEXT,
        minHeight: "100vh",
        fontFamily: SANS,
        padding: "20px 20px 32px"
      }}
    >
      <link
        href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap"
        rel="stylesheet"
      />

      {/* Title */}
      <div style={{ marginBottom: 20 }}>
        <h1
          style={{
            fontFamily: MONO,
            fontSize: 15,
            fontWeight: 700,
            color: GOLD,
            margin: 0,
            letterSpacing: 2
          }}
        >
          FIBONACCI METHODS
        </h1>
        <p style={{ fontFamily: MONO, fontSize: 11, color: TEXT_DIM, margin: "4px 0 0" }}>
          step through every computation strategy
        </p>
      </div>

      {/* Method tabs */}
      <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 16 }}>
        {METHODS.map((m, i) => (
          <button
            key={m.id}
            onClick={() => setMethodIdx(i)}
            style={{
              padding: "6px 12px",
              borderRadius: 6,
              background: i === methodIdx ? m.color + "20" : SURFACE,
              border: `1px solid ${i === methodIdx ? m.color + "60" : BORDER}`,
              color: i === methodIdx ? m.color : TEXT_DIM,
              fontFamily: MONO,
              fontSize: 11,
              cursor: "pointer",
              fontWeight: i === methodIdx ? 700 : 400,
              transition: "all 0.2s"
            }}
          >
            {m.name}
          </button>
        ))}
      </div>

      {/* Method info */}
      <div
        style={{
          padding: "10px 14px",
          background: SURFACE,
          borderRadius: 8,
          border: `1px solid ${method.color}30`,
          marginBottom: 16
        }}
      >
        <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ fontFamily: MONO, fontSize: 12, color: method.color, fontWeight: 700 }}>
            Time: {method.complexity}
          </span>
          <span style={{ fontFamily: MONO, fontSize: 12, color: TEXT_DIM }}>Space: {method.space}</span>
        </div>
        <div style={{ fontFamily: SANS, fontSize: 12, color: TEXT_DIM, marginTop: 4 }}>{method.desc}</div>
      </div>

      {/* N selector + controls */}
      <div
        style={{
          display: "flex",
          gap: 16,
          alignItems: "center",
          marginBottom: 16,
          flexWrap: "wrap"
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontFamily: MONO, fontSize: 11, color: TEXT_DIM }}>n =</span>
          <input
            type="range"
            min={2}
            max={maxN}
            value={Math.min(n, maxN)}
            onChange={e => setN(parseInt(e.target.value))}
            style={{ width: 120, accentColor: method.color }}
          />
          <span style={{ fontFamily: MONO, fontSize: 16, color: method.color, fontWeight: 700, minWidth: 24 }}>
            {Math.min(n, maxN)}
          </span>
        </div>

        <div style={{ display: "flex", gap: 4 }}>
          <button
            onClick={() => setStepIdx(Math.max(0, stepIdx - 1))}
            style={{
              width: 32,
              height: 32,
              borderRadius: 6,
              background: SURFACE2,
              border: `1px solid ${BORDER}`,
              color: TEXT,
              cursor: "pointer",
              fontFamily: MONO,
              fontSize: 14
            }}
          >
            ◂
          </button>
          <button
            onClick={() => setPlaying(!playing)}
            style={{
              width: 48,
              height: 32,
              borderRadius: 6,
              background: playing ? RED + "30" : GREEN + "20",
              border: `1px solid ${playing ? RED + "60" : GREEN + "50"}`,
              color: playing ? RED : GREEN,
              cursor: "pointer",
              fontFamily: MONO,
              fontSize: 12,
              fontWeight: 700
            }}
          >
            {playing ? "■" : "▶"}
          </button>
          <button
            onClick={() => setStepIdx(Math.min(steps.length - 1, stepIdx + 1))}
            style={{
              width: 32,
              height: 32,
              borderRadius: 6,
              background: SURFACE2,
              border: `1px solid ${BORDER}`,
              color: TEXT,
              cursor: "pointer",
              fontFamily: MONO,
              fontSize: 14
            }}
          >
            ▸
          </button>
          <button
            onClick={() => {
              setStepIdx(0);
              setPlaying(false);
            }}
            style={{
              width: 32,
              height: 32,
              borderRadius: 6,
              background: SURFACE2,
              border: `1px solid ${BORDER}`,
              color: TEXT_DIM,
              cursor: "pointer",
              fontFamily: MONO,
              fontSize: 10
            }}
          >
            ↺
          </button>
        </div>

        <div style={{ fontFamily: MONO, fontSize: 11, color: TEXT_DIM }}>
          step {stepIdx + 1} / {steps.length}
        </div>
      </div>

      {/* Progress bar */}
      <div
        style={{
          height: 3,
          background: SURFACE2,
          borderRadius: 2,
          marginBottom: 16,
          overflow: "hidden"
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${((stepIdx + 1) / steps.length) * 100}%`,
            background: method.color,
            borderRadius: 2,
            transition: "width 0.2s"
          }}
        />
      </div>

      {/* Step description */}
      <div
        style={{
          padding: "10px 14px",
          background: SURFACE2,
          borderRadius: 8,
          border: `1px solid ${BORDER}`,
          marginBottom: 16,
          fontFamily: MONO,
          fontSize: 12,
          color: TEXT,
          minHeight: 36,
          display: "flex",
          alignItems: "center"
        }}
      >
        {step.desc}
      </div>

      {/* Visualization area */}
      <div
        style={{
          padding: 16,
          background: SURFACE,
          borderRadius: 10,
          border: `1px solid ${BORDER}`,
          minHeight: 100,
          overflow: "auto"
        }}
      >
        <Vis step={step} />
      </div>

      {/* Cost comparison */}
      <CostBar
        methods={METHODS}
        n={Math.min(n, 8)}
      />
    </div>
  );
}
