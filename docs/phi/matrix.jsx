/**
 * ═══════════════════════════════════════════════════════════════════
 * THE MATRIX IDENTITY — Interactive Explorer
 * ═══════════════════════════════════════════════════════════════════
 *
 * Binary and Fibonacci aren't competing systems.
 * They're ciphers on the same matrix.
 *
 * Three interactive explorations:
 *   CIPHER FORK — 0-15 encoded both ways from the same structure
 *   ARITHMETIC  — same operation, dual cost on each register
 *   COMPOSITION — building M^20 step by step, two traversals
 */

import { useState, useMemo, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════

const GOLD = "#d4a017";
const RED = "#e04040";
const GREEN = "#40c878";
const ORANGE = "#ffa028";
const CYAN = "#40b8c8";
const BG = "#08080a";
const SURFACE = "rgba(255,255,255,0.02)";
const BORDER = "rgba(255,255,255,0.06)";
const MONO = "'JetBrains Mono', 'Fira Code', monospace";
const DISPLAY = "'Playfair Display', Georgia, serif";

const FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946];

// ═══════════════════════════════════════════════════════════════════
// MATRIX UTILITIES
// ═══════════════════════════════════════════════════════════════════

function matMul(A, B) {
  return [
    [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
    [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]
  ];
}

function matPow(n) {
  if (n === 0)
    return [
      [1, 0],
      [0, 1]
    ];
  if (n === 1)
    return [
      [1, 1],
      [1, 0]
    ];
  let result = [
    [1, 0],
    [0, 1]
  ];
  let base = [
    [1, 1],
    [1, 0]
  ];
  let p = n;
  while (p > 0) {
    if (p % 2 === 1) result = matMul(result, base);
    base = matMul(base, base);
    p = Math.floor(p / 2);
  }
  return result;
}

function toZeckendorf(n) {
  if (n === 0) return [];
  const fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597];
  const bits = [];
  let rem = n;
  for (let i = fibs.length - 1; i >= 0; i--) {
    if (fibs[i] <= rem) {
      bits.push(i);
      rem -= fibs[i];
    }
  }
  return bits;
}

function toBinary(n, width = 4) {
  const bits = [];
  for (let i = width - 1; i >= 0; i--) bits.push((n >> i) & 1);
  return bits;
}

function toZeckBits(n, width = 6) {
  const fibs = [1, 2, 3, 5, 8, 13];
  const bits = new Array(width).fill(0);
  let rem = n;
  for (let i = width - 1; i >= 0; i--) {
    if (fibs[i] <= rem) {
      bits[i] = 1;
      rem -= fibs[i];
    }
  }
  return bits.slice().reverse();
}

// ═══════════════════════════════════════════════════════════════════
// SHARED UI COMPONENTS
// ═══════════════════════════════════════════════════════════════════

function MatrixDisplay({ m, label, highlight, size = "normal", showIndices = false }) {
  const s =
    size === "small" ? 11
    : size === "tiny" ? 9
    : 13;
  const p =
    size === "small" ? 6
    : size === "tiny" ? 4
    : 10;
  const highlightCell = highlight || [-1, -1];

  return (
    <div style={{ display: "inline-flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
      {label && (
        <div style={{ fontSize: s - 2, fontFamily: MONO, color: "rgba(212,160,23,0.5)", letterSpacing: 1 }}>{label}</div>
      )}
      <div
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 2,
          background: "rgba(212,160,23,0.03)",
          borderRadius: 6,
          border: "1px solid rgba(212,160,23,0.1)",
          padding: `${p}px ${p + 2}px`
        }}
      >
        <div style={{ color: "rgba(212,160,23,0.2)", fontSize: s + 4, fontWeight: 100, marginRight: 2 }}>[</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: size === "tiny" ? 2 : 4 }}>
          {[0, 1].map(r =>
            [0, 1].map(c => {
              const isHl = highlightCell[0] === r && highlightCell[1] === c;
              return (
                <div
                  key={`${r}${c}`}
                  style={{
                    fontFamily: MONO,
                    fontSize: s,
                    fontWeight: isHl ? 700 : 400,
                    color: isHl ? GOLD : "rgba(255,255,255,0.6)",
                    background: isHl ? "rgba(212,160,23,0.15)" : "transparent",
                    borderRadius: 3,
                    padding: `2px ${size === "tiny" ? 4 : 6}px`,
                    textAlign: "right",
                    minWidth:
                      size === "tiny" ? 20
                      : size === "small" ? 28
                      : 36,
                    transition: "all 0.3s ease",
                    position: "relative"
                  }}
                >
                  {m[r][c]}
                  {showIndices && (
                    <span
                      style={{
                        position: "absolute",
                        top: -6,
                        right: -2,
                        fontSize: 7,
                        color: "rgba(255,255,255,0.2)",
                        fontFamily: MONO
                      }}
                    >
                      [{r},{c}]
                    </span>
                  )}
                </div>
              );
            })
          )}
        </div>
        <div style={{ color: "rgba(212,160,23,0.2)", fontSize: s + 4, fontWeight: 100, marginLeft: 2 }}>]</div>
      </div>
    </div>
  );
}

function DotProductDetail({ A, B, row, col, color = GOLD }) {
  const terms = [
    { a: A[row][0], b: B[0][col] },
    { a: A[row][1], b: B[1][col] }
  ];
  const result = terms[0].a * terms[0].b + terms[1].a * terms[1].b;

  return (
    <div
      style={{
        fontFamily: MONO,
        fontSize: 11,
        color: "rgba(255,255,255,0.5)",
        padding: "6px 10px",
        background: "rgba(255,255,255,0.02)",
        borderRadius: 6,
        border: `1px solid rgba(255,255,255,0.05)`
      }}
    >
      <span style={{ color: "rgba(255,255,255,0.25)" }}>
        [{row},{col}]:{" "}
      </span>
      <span style={{ color: RED }}>{terms[0].a}</span>
      <span style={{ color: "rgba(255,255,255,0.3)" }}>×</span>
      <span style={{ color: CYAN }}>{terms[0].b}</span>
      <span style={{ color: "rgba(255,255,255,0.3)" }}> + </span>
      <span style={{ color: RED }}>{terms[1].a}</span>
      <span style={{ color: "rgba(255,255,255,0.3)" }}>×</span>
      <span style={{ color: CYAN }}>{terms[1].b}</span>
      <span style={{ color: "rgba(255,255,255,0.3)" }}> = </span>
      <span style={{ color, fontWeight: 700 }}>{result}</span>
    </div>
  );
}

function SectionLabel({ children, color = GOLD }) {
  return (
    <div
      style={{
        fontSize: 9,
        fontFamily: MONO,
        color: color,
        letterSpacing: 3,
        marginBottom: 8,
        opacity: 0.6
      }}
    >
      {children}
    </div>
  );
}

function Card({ children, border = BORDER, bg = SURFACE, style = {} }) {
  return (
    <div
      style={{
        padding: 16,
        background: bg,
        borderRadius: 10,
        border: `1px solid ${border}`,
        ...style
      }}
    >
      {children}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB A: CIPHER FORK
// ═══════════════════════════════════════════════════════════════════

function CipherForkTab() {
  const [selectedValue, setSelectedValue] = useState(3);

  const fibWeights = [13, 8, 5, 3, 2, 1];
  const binWeights = [8, 4, 2, 1];

  const data = useMemo(() => {
    return Array.from({ length: 16 }, (_, v) => {
      const zBits = toZeckBits(v, 6);
      const bBits = toBinary(v, 4);
      const zWeights = [];
      const bWeights = [];
      const fibs = [1, 2, 3, 5, 8, 13];
      for (let i = 5; i >= 0; i--) if (zBits[5 - i] === 1) zWeights.push(fibs[i]);
      for (let i = 3; i >= 0; i--) if (bBits[3 - i] === 1) bWeights.push(1 << i);
      return { value: v, zBits, bBits, zWeights, bWeights };
    });
  }, []);

  const sel = data[selectedValue];
  const isDivergence = selectedValue >= 3;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <Card>
        <SectionLabel>THE FORK POINT</SectionLabel>
        <div
          style={{
            fontFamily: MONO,
            fontSize: 12,
            color: "rgba(255,255,255,0.4)",
            lineHeight: 1.8,
            marginBottom: 12
          }}
        >
          At value <span style={{ color: GOLD, fontWeight: 700 }}>3</span>, the ciphers diverge. Fibonacci says{" "}
          <span style={{ color: RED }}>11 is forbidden</span> — carry up. Binary says{" "}
          <span style={{ color: GREEN }}>11 is fine</span> — absorb it.
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 12,
            padding: "12px 0"
          }}
        >
          <div
            style={{
              padding: 12,
              borderRadius: 8,
              background: "rgba(212,160,23,0.04)",
              border: "1px solid rgba(212,160,23,0.15)"
            }}
          >
            <div style={{ fontSize: 9, fontFamily: MONO, color: GOLD, letterSpacing: 2, marginBottom: 8 }}>
              CIPHER A — FIBONACCI
            </div>
            <div style={{ fontSize: 10, fontFamily: MONO, color: "rgba(255,255,255,0.3)", marginBottom: 4 }}>
              Forbid [1,1] → weights spiral: 1,2,3,5,8,13
            </div>
            <div style={{ fontSize: 10, fontFamily: MONO, color: "rgba(255,255,255,0.3)" }}>
              6 bits → 16 valid / 64 total states
            </div>
          </div>
          <div
            style={{
              padding: 12,
              borderRadius: 8,
              background: "rgba(64,200,120,0.04)",
              border: "1px solid rgba(64,200,120,0.15)"
            }}
          >
            <div style={{ fontSize: 9, fontFamily: MONO, color: GREEN, letterSpacing: 2, marginBottom: 8 }}>
              CIPHER B — BINARY
            </div>
            <div style={{ fontSize: 10, fontFamily: MONO, color: "rgba(255,255,255,0.3)", marginBottom: 4 }}>
              Allow [1,1] → weights double: 1,2,4,8
            </div>
            <div style={{ fontSize: 10, fontFamily: MONO, color: "rgba(255,255,255,0.3)" }}>
              4 bits → 16 valid / 16 total states
            </div>
          </div>
        </div>
      </Card>

      {/* Value selector */}
      <Card>
        <SectionLabel>SELECT A VALUE</SectionLabel>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 16 }}>
          {data.map(d => (
            <button
              key={d.value}
              onClick={() => setSelectedValue(d.value)}
              style={{
                width: 36,
                height: 36,
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
                fontFamily: MONO,
                fontSize: 13,
                fontWeight: selectedValue === d.value ? 700 : 400,
                background:
                  selectedValue === d.value ? "rgba(212,160,23,0.25)"
                  : d.value === 3 ? "rgba(224,64,64,0.08)"
                  : "rgba(255,255,255,0.04)",
                color:
                  selectedValue === d.value ? GOLD
                  : d.value === 3 ? ORANGE
                  : "rgba(255,255,255,0.4)",
                border:
                  selectedValue === d.value ? `2px solid ${GOLD}`
                  : d.value === 3 ? `1px solid rgba(224,64,64,0.2)`
                  : `1px solid rgba(255,255,255,0.06)`,
                transition: "all 0.2s ease"
              }}
            >
              {d.value}
            </button>
          ))}
        </div>

        {/* Comparison display */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          {/* Fibonacci encoding */}
          <div>
            <div style={{ fontSize: 10, fontFamily: MONO, color: GOLD, letterSpacing: 2, marginBottom: 8 }}>FIBONACCI</div>
            <div style={{ display: "flex", gap: 3, marginBottom: 8 }}>
              {sel.zBits.map((b, i) => {
                const hasAdj = i < sel.zBits.length - 1 && sel.zBits[i] === 1 && sel.zBits[i + 1] === 1;
                return (
                  <div
                    key={i}
                    style={{
                      width: 28,
                      height: 34,
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "center",
                      borderRadius: 4,
                      background: b ? "rgba(212,160,23,0.2)" : "rgba(255,255,255,0.03)",
                      border: b ? `1px solid ${GOLD}` : `1px solid rgba(255,255,255,0.06)`
                    }}
                  >
                    <div
                      style={{ fontFamily: MONO, fontSize: 13, fontWeight: 700, color: b ? GOLD : "rgba(255,255,255,0.15)" }}
                    >
                      {b}
                    </div>
                    <div style={{ fontFamily: MONO, fontSize: 7, color: "rgba(255,255,255,0.2)" }}>{fibWeights[i]}</div>
                  </div>
                );
              })}
            </div>
            <div style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.4)" }}>
              = {sel.zWeights.length > 0 ? sel.zWeights.join(" + ") : "0"} = <span style={{ color: GOLD }}>{sel.value}</span>
            </div>
          </div>

          {/* Binary encoding */}
          <div>
            <div style={{ fontSize: 10, fontFamily: MONO, color: GREEN, letterSpacing: 2, marginBottom: 8 }}>BINARY</div>
            <div style={{ display: "flex", gap: 3, marginBottom: 8 }}>
              {sel.bBits.map((b, i) => (
                <div
                  key={i}
                  style={{
                    width: 28,
                    height: 34,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    borderRadius: 4,
                    background: b ? "rgba(64,200,120,0.2)" : "rgba(255,255,255,0.03)",
                    border: b ? `1px solid ${GREEN}` : `1px solid rgba(255,255,255,0.06)`
                  }}
                >
                  <div
                    style={{ fontFamily: MONO, fontSize: 13, fontWeight: 700, color: b ? GREEN : "rgba(255,255,255,0.15)" }}
                  >
                    {b}
                  </div>
                  <div style={{ fontFamily: MONO, fontSize: 7, color: "rgba(255,255,255,0.2)" }}>{binWeights[i]}</div>
                </div>
              ))}
            </div>
            <div style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.4)" }}>
              = {sel.bWeights.length > 0 ? sel.bWeights.join(" + ") : "0"} = <span style={{ color: GREEN }}>{sel.value}</span>
            </div>
          </div>
        </div>

        {/* Divergence annotation */}
        {selectedValue === 3 && (
          <div
            style={{
              marginTop: 14,
              padding: 12,
              borderRadius: 8,
              background: "rgba(224,64,64,0.06)",
              border: "1px solid rgba(224,64,64,0.2)",
              fontFamily: MONO,
              fontSize: 11,
              color: ORANGE,
              lineHeight: 1.8
            }}
          >
            <strong>THE FORK:</strong> Fibonacci uses <span style={{ color: GOLD }}>000100</span> (weight 3 at position 2).
            Binary uses <span style={{ color: GREEN }}>0011</span> (weights 2+1, adjacent). Same value, different
            decomposition. From here the weight sequences diverge permanently.
          </div>
        )}

        {selectedValue >= 4 && (
          <div
            style={{
              marginTop: 14,
              padding: 10,
              borderRadius: 8,
              background: "rgba(255,255,255,0.02)",
              border: "1px solid rgba(255,255,255,0.06)",
              fontFamily: MONO,
              fontSize: 10,
              color: "rgba(255,255,255,0.3)",
              lineHeight: 1.8
            }}
          >
            Fibonacci's 3rd position carries weight <span style={{ color: GOLD }}>5</span> (F+2 from the carry). Binary's 3rd
            position carries weight <span style={{ color: GREEN }}>4</span> (doubled, because 3 was absorbed below).
          </div>
        )}
      </Card>

      {/* Matrix underneath */}
      <Card
        border="rgba(212,160,23,0.12)"
        bg="rgba(212,160,23,0.02)"
      >
        <SectionLabel>THE MATRIX UNDERNEATH</SectionLabel>
        <div
          style={{
            fontFamily: MONO,
            fontSize: 11,
            color: "rgba(255,255,255,0.35)",
            lineHeight: 1.8,
            marginBottom: 12
          }}
        >
          Both ciphers project from the same object. Select an index to see what the matrix holds:
        </div>
        <MatrixIndexExplorer n={selectedValue <= 1 ? 2 : selectedValue} />
      </Card>
    </div>
  );
}

function MatrixIndexExplorer({ n }) {
  const [selIdx, setSelIdx] = useState([0, 1]);
  const m = matPow(n);
  const indices = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ];
  const labels = ["F(n+1)", "F(n)", "F(n)", "F(n-1)"];
  const isDup = selIdx[0] === 1 && selIdx[1] === 0;

  return (
    <div style={{ display: "flex", gap: 20, alignItems: "flex-start", flexWrap: "wrap" }}>
      <MatrixDisplay
        m={m}
        label={`M^${n}`}
        highlight={selIdx}
        showIndices
      />
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {indices.map(([r, c], i) => {
          const isSel = selIdx[0] === r && selIdx[1] === c;
          const isRedundant = r === 1 && c === 0;
          return (
            <button
              key={i}
              onClick={() => setSelIdx([r, c])}
              style={{
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
                padding: "6px 12px",
                fontFamily: MONO,
                fontSize: 11,
                textAlign: "left",
                background: isSel ? "rgba(212,160,23,0.15)" : "rgba(255,255,255,0.03)",
                color:
                  isSel ? GOLD
                  : isRedundant ? "rgba(255,255,255,0.2)"
                  : "rgba(255,255,255,0.4)",
                border: isSel ? `1px solid ${GOLD}` : `1px solid rgba(255,255,255,0.06)`,
                transition: "all 0.2s ease",
                textDecoration: isRedundant ? "line-through" : "none",
                opacity: isRedundant && !isSel ? 0.5 : 1
              }}
            >
              [{r},{c}] → {labels[i]} = {m[r][c]}
              {isRedundant && (
                <span style={{ fontSize: 9, marginLeft: 6, color: ORANGE, textDecoration: "none" }}>redundant</span>
              )}
            </button>
          );
        })}
      </div>
      {isDup && (
        <div
          style={{
            padding: 10,
            borderRadius: 8,
            background: "rgba(255,160,40,0.06)",
            border: "1px solid rgba(255,160,40,0.2)",
            fontFamily: MONO,
            fontSize: 10,
            color: ORANGE,
            lineHeight: 1.8,
            maxWidth: 200
          }}
        >
          [1,0] = [0,1] = {m[1][0]}. This degeneracy is the adjacency constraint (1D), the delimiter (Fibonacci coding), and
          the symmetry (matrix) — same fact, three views.
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB B: ARITHMETIC
// ═══════════════════════════════════════════════════════════════════

function ArithmeticTab() {
  const [mode, setMode] = useState("value"); // "value" | "index"

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <Card>
        <SectionLabel>CHOOSE AN OPERATION</SectionLabel>
        <div
          style={{
            fontFamily: MONO,
            fontSize: 11,
            color: "rgba(255,255,255,0.35)",
            lineHeight: 1.8,
            marginBottom: 12
          }}
        >
          Each register is native where the other is expensive. Pick an operation to see the cost asymmetry.
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {[
            { id: "value", label: "F(5) + F(4) = 5 + 3", sub: "Value Addition" },
            { id: "index", label: "M⁵ × M⁴ = M⁹", sub: "Index Composition" }
          ].map(o => (
            <button
              key={o.id}
              onClick={() => setMode(o.id)}
              style={{
                flex: 1,
                border: "none",
                borderRadius: 8,
                cursor: "pointer",
                padding: "12px 16px",
                textAlign: "left",
                background: mode === o.id ? "rgba(212,160,23,0.12)" : "rgba(255,255,255,0.03)",
                border: mode === o.id ? `1px solid ${GOLD}` : `1px solid rgba(255,255,255,0.06)`,
                transition: "all 0.2s ease"
              }}
            >
              <div
                style={{
                  fontFamily: MONO,
                  fontSize: 12,
                  color: mode === o.id ? GOLD : "rgba(255,255,255,0.4)",
                  fontWeight: 600
                }}
              >
                {o.label}
              </div>
              <div style={{ fontFamily: MONO, fontSize: 9, color: "rgba(255,255,255,0.25)", marginTop: 4, letterSpacing: 1 }}>
                {o.sub}
              </div>
            </button>
          ))}
        </div>
      </Card>

      {mode === "value" ?
        <ValueAdditionExample />
      : <IndexCompositionExample />}
    </div>
  );
}

function ValueAdditionExample() {
  const [step, setStep] = useState(0);
  const maxStep = 4;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      {/* 1D Register side */}
      <Card
        border="rgba(64,200,120,0.15)"
        bg="rgba(64,200,120,0.02)"
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <SectionLabel color={GREEN}>1D φ-REGISTER — NATIVE</SectionLabel>
          <div
            style={{
              fontFamily: MONO,
              fontSize: 9,
              color: GREEN,
              padding: "2px 8px",
              borderRadius: 10,
              background: "rgba(64,200,120,0.15)"
            }}
          >
            CHEAP
          </div>
        </div>

        <div style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.4)", lineHeight: 2.2 }}>
          <div>
            <span style={{ color: "rgba(255,255,255,0.25)", width: 30, display: "inline-block" }}>A:</span>
            <BitRow
              bits={[0, 0, 1, 0, 0, 0]}
              weights={[13, 8, 5, 3, 2, 1]}
              active={step >= 0}
              color={GOLD}
            />
            <span style={{ color: "rgba(255,255,255,0.25)", marginLeft: 8 }}>= 5</span>
          </div>
          <div>
            <span style={{ color: "rgba(255,255,255,0.25)", width: 30, display: "inline-block" }}>B:</span>
            <BitRow
              bits={[0, 0, 0, 1, 0, 0]}
              weights={[13, 8, 5, 3, 2, 1]}
              active={step >= 0}
              color={GOLD}
            />
            <span style={{ color: "rgba(255,255,255,0.25)", marginLeft: 8 }}>= 3</span>
          </div>

          {step >= 1 && (
            <div style={{ marginTop: 8, padding: "6px 0", borderTop: "1px solid rgba(255,255,255,0.06)" }}>
              <span style={{ color: "rgba(255,255,255,0.25)", width: 30, display: "inline-block" }}>Σ:</span>
              <BitRow
                bits={[0, 0, 1, 1, 0, 0]}
                weights={[13, 8, 5, 3, 2, 1]}
                active
                color={ORANGE}
              />
              <span style={{ color: ORANGE, marginLeft: 8 }}>← adjacent!</span>
            </div>
          )}

          {step >= 2 && (
            <div style={{ marginTop: 4 }}>
              <span style={{ color: "rgba(255,255,255,0.25)", width: 30, display: "inline-block" }}></span>
              <span style={{ color: "rgba(255,255,255,0.25)" }}>carry: F(2)+F(3)=F(4) → 3+5=8</span>
            </div>
          )}

          {step >= 3 && (
            <div style={{ marginTop: 4 }}>
              <span style={{ color: "rgba(255,255,255,0.25)", width: 30, display: "inline-block" }}>R:</span>
              <BitRow
                bits={[0, 1, 0, 0, 0, 0]}
                weights={[13, 8, 5, 3, 2, 1]}
                active
                color={GREEN}
              />
              <span style={{ color: GREEN, marginLeft: 8 }}>= 8 ✓</span>
            </div>
          )}
        </div>

        {step >= 3 && (
          <div
            style={{
              marginTop: 12,
              padding: 8,
              borderRadius: 6,
              background: "rgba(64,200,120,0.06)",
              fontFamily: MONO,
              fontSize: 10,
              color: GREEN
            }}
          >
            Cost: 1 pour + 1 carry step
          </div>
        )}
      </Card>

      {/* Matrix Register side */}
      <Card
        border="rgba(224,64,64,0.15)"
        bg="rgba(224,64,64,0.02)"
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <SectionLabel color={RED}>MATRIX REGISTER — EXPENSIVE</SectionLabel>
          <div
            style={{
              fontFamily: MONO,
              fontSize: 9,
              color: RED,
              padding: "2px 8px",
              borderRadius: 10,
              background: "rgba(224,64,64,0.15)"
            }}
          >
            COSTLY
          </div>
        </div>

        <div style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.4)", lineHeight: 1.8 }}>
          {step >= 0 && (
            <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: 8 }}>
              <MatrixDisplay
                m={matPow(5)}
                label="M⁵"
                size="small"
              />
              <MatrixDisplay
                m={matPow(4)}
                label="M⁴"
                size="small"
              />
            </div>
          )}

          {step >= 1 && (
            <div style={{ color: "rgba(255,255,255,0.3)", marginBottom: 4 }}>⚠ M⁵ × M⁴ = M⁹ → F(9)=34 — wrong operation!</div>
          )}

          {step >= 2 && (
            <div style={{ color: "rgba(255,255,255,0.3)", marginBottom: 4 }}>
              Must: read values → sum → decompose → rebuild
            </div>
          )}

          {step >= 2 && <div style={{ color: "rgba(255,255,255,0.3)" }}>5 + 3 = 8 = F(6) → need M⁶ = M⁵ × M¹</div>}

          {step >= 3 && (
            <div style={{ marginTop: 8 }}>
              <MatrixDisplay
                m={matPow(6)}
                label="M⁶ = result"
                size="small"
                highlight={[0, 1]}
              />
              <div style={{ marginTop: 6, color: GREEN }}>READ = M⁶[0,1] = 8 ✓</div>
            </div>
          )}
        </div>

        {step >= 3 && (
          <div
            style={{
              marginTop: 12,
              padding: 8,
              borderRadius: 6,
              background: "rgba(224,64,64,0.06)",
              fontFamily: MONO,
              fontSize: 10,
              color: RED
            }}
          >
            Cost: 1 add + 1 decompose + 1 matrix multiply
          </div>
        )}
      </Card>

      {/* Step control */}
      <div style={{ gridColumn: "1 / -1", display: "flex", justifyContent: "center", gap: 8 }}>
        <button
          onClick={() => setStep(Math.max(0, step - 1))}
          style={{
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            padding: "8px 20px",
            fontFamily: MONO,
            fontSize: 11,
            background: "rgba(255,255,255,0.06)",
            color: "rgba(255,255,255,0.4)",
            border: `1px solid rgba(255,255,255,0.08)`
          }}
        >
          ← PREV
        </button>
        <div
          style={{
            fontFamily: MONO,
            fontSize: 11,
            color: "rgba(255,255,255,0.3)",
            padding: "8px 16px",
            display: "flex",
            alignItems: "center"
          }}
        >
          step {step} / {maxStep - 1}
        </div>
        <button
          onClick={() => setStep(Math.min(maxStep - 1, step + 1))}
          style={{
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            padding: "8px 20px",
            fontFamily: MONO,
            fontSize: 11,
            background: "rgba(212,160,23,0.15)",
            color: GOLD,
            border: `1px solid rgba(212,160,23,0.3)`
          }}
        >
          NEXT →
        </button>
      </div>
    </div>
  );
}

function IndexCompositionExample() {
  const [step, setStep] = useState(0);
  const maxStep = 4;
  const M5 = matPow(5);
  const M4 = matPow(4);
  const M9 = matPow(9);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      {/* Matrix side — native */}
      <Card
        border="rgba(64,200,120,0.15)"
        bg="rgba(64,200,120,0.02)"
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <SectionLabel color={GREEN}>MATRIX REGISTER — NATIVE</SectionLabel>
          <div
            style={{
              fontFamily: MONO,
              fontSize: 9,
              color: GREEN,
              padding: "2px 8px",
              borderRadius: 10,
              background: "rgba(64,200,120,0.15)"
            }}
          >
            CHEAP
          </div>
        </div>

        <div style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.4)", lineHeight: 1.8 }}>
          {step >= 0 && (
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 8 }}>
              <MatrixDisplay
                m={M5}
                label="M⁵"
                size="small"
              />
              <span style={{ color: "rgba(255,255,255,0.2)", fontSize: 16 }}>×</span>
              <MatrixDisplay
                m={M4}
                label="M⁴"
                size="small"
              />
            </div>
          )}

          {step >= 1 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 8 }}>
              <DotProductDetail
                A={M5}
                B={M4}
                row={0}
                col={0}
                color={GREEN}
              />
              <DotProductDetail
                A={M5}
                B={M4}
                row={0}
                col={1}
                color={GOLD}
              />
              <DotProductDetail
                A={M5}
                B={M4}
                row={1}
                col={0}
              />
              <DotProductDetail
                A={M5}
                B={M4}
                row={1}
                col={1}
              />
            </div>
          )}

          {step >= 2 && (
            <div>
              <MatrixDisplay
                m={M9}
                label="M⁹ — done"
                size="small"
                highlight={[0, 1]}
              />
              <div style={{ marginTop: 6, color: GREEN }}>READ = F(9) = 34 ✓ — no normalization</div>
            </div>
          )}
        </div>

        {step >= 2 && (
          <div
            style={{
              marginTop: 12,
              padding: 8,
              borderRadius: 6,
              background: "rgba(64,200,120,0.06)",
              fontFamily: MONO,
              fontSize: 10,
              color: GREEN
            }}
          >
            Cost: 1 matrix multiply (8 muls + 4 adds). Done.
          </div>
        )}
      </Card>

      {/* 1D side — expensive */}
      <Card
        border="rgba(224,64,64,0.15)"
        bg="rgba(224,64,64,0.02)"
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <SectionLabel color={RED}>1D φ-REGISTER — EXPENSIVE</SectionLabel>
          <div
            style={{
              fontFamily: MONO,
              fontSize: 9,
              color: RED,
              padding: "2px 8px",
              borderRadius: 10,
              background: "rgba(224,64,64,0.15)"
            }}
          >
            COSTLY
          </div>
        </div>

        <div style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.4)", lineHeight: 1.8 }}>
          {step >= 0 && <div>Want: F(5+4) = F(9) = ?</div>}
          {step >= 1 && <div style={{ color: "rgba(255,255,255,0.3)" }}>No native operation for index addition.</div>}
          {step >= 1 && <div style={{ color: "rgba(255,255,255,0.3)" }}>Must build M⁵, M⁴ from scratch and multiply...</div>}
          {step >= 2 && (
            <div style={{ color: "rgba(255,255,255,0.3)" }}>
              ...or climb the entire Fibonacci ladder with value additions.
            </div>
          )}
          {step >= 2 && (
            <div style={{ marginTop: 8, color: "rgba(255,255,255,0.25)", fontSize: 10, lineHeight: 1.6 }}>
              F(0)=0, F(1)=1, F(2)=1, F(3)=2, F(4)=3,
              <br />
              F(5)=5, F(6)=8, F(7)=13, F(8)=21, F(9)=34
              <br />
              <span style={{ color: RED }}>9 addition+normalize steps</span>
            </div>
          )}
        </div>

        {step >= 2 && (
          <div
            style={{
              marginTop: 12,
              padding: 8,
              borderRadius: 6,
              background: "rgba(224,64,64,0.06)",
              fontFamily: MONO,
              fontSize: 10,
              color: RED
            }}
          >
            Cost: 9 additions + normalizations, OR build matrix machinery
          </div>
        )}
      </Card>

      <div style={{ gridColumn: "1 / -1", display: "flex", justifyContent: "center", gap: 8 }}>
        <button
          onClick={() => setStep(Math.max(0, step - 1))}
          style={{
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            padding: "8px 20px",
            fontFamily: MONO,
            fontSize: 11,
            background: "rgba(255,255,255,0.06)",
            color: "rgba(255,255,255,0.4)",
            border: `1px solid rgba(255,255,255,0.08)`
          }}
        >
          ← PREV
        </button>
        <div
          style={{
            fontFamily: MONO,
            fontSize: 11,
            color: "rgba(255,255,255,0.3)",
            padding: "8px 16px",
            display: "flex",
            alignItems: "center"
          }}
        >
          step {step} / {maxStep - 2}
        </div>
        <button
          onClick={() => setStep(Math.min(maxStep - 2, step + 1))}
          style={{
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            padding: "8px 20px",
            fontFamily: MONO,
            fontSize: 11,
            background: "rgba(212,160,23,0.15)",
            color: GOLD,
            border: `1px solid rgba(212,160,23,0.3)`
          }}
        >
          NEXT →
        </button>
      </div>
    </div>
  );
}

function BitRow({ bits, weights, active, color = GOLD }) {
  return (
    <span style={{ display: "inline-flex", gap: 2 }}>
      {bits.map((b, i) => (
        <span
          key={i}
          style={{
            display: "inline-block",
            width: 16,
            textAlign: "center",
            fontFamily: MONO,
            fontSize: 12,
            fontWeight: b ? 700 : 400,
            color: b && active ? color : "rgba(255,255,255,0.15)"
          }}
        >
          {b}
        </span>
      ))}
    </span>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB C: COMPOSITION CHAIN
// ═══════════════════════════════════════════════════════════════════

function CompositionTab() {
  const [method, setMethod] = useState("ladder"); // "ladder" | "doubling"
  const [step, setStep] = useState(0);

  const ladderSteps = useMemo(() => {
    const steps = [];
    steps.push({ label: "M¹ (base)", m: matPow(1), exp: 1, op: "base case", a: null, b: null });
    steps.push({ label: "M² = M¹×M¹", m: matPow(2), exp: 2, op: "1+1=2", a: matPow(1), b: matPow(1) });
    steps.push({ label: "M³ = M²×M¹", m: matPow(3), exp: 3, op: "2+1=3", a: matPow(2), b: matPow(1) });
    steps.push({ label: "M⁵ = M³×M²", m: matPow(5), exp: 5, op: "3+2=5", a: matPow(3), b: matPow(2) });
    steps.push({ label: "M⁸ = M⁵×M³", m: matPow(8), exp: 8, op: "5+3=8", a: matPow(5), b: matPow(3) });
    steps.push({ label: "M¹³ = M⁸×M⁵", m: matPow(13), exp: 13, op: "8+5=13", a: matPow(8), b: matPow(5) });
    steps.push({
      label: "M¹⁸ = M¹³×M⁵",
      m: matPow(18),
      exp: 18,
      op: "assemble: 13+5",
      a: matPow(13),
      b: matPow(5),
      assemble: true
    });
    steps.push({
      label: "M²⁰ = M¹⁸×M²",
      m: matPow(20),
      exp: 20,
      op: "assemble: 18+2",
      a: matPow(18),
      b: matPow(2),
      assemble: true
    });
    return steps;
  }, []);

  const doublingSteps = useMemo(() => {
    // 20 in binary: 10100, process MSB to LSB
    const steps = [];
    steps.push({ label: "Start", a: 0, b: 1, k: 0, bit: null, op: "(F(0), F(1)) = (0, 1)" });
    // bit 1 (MSB)
    let a = 0,
      b = 1;
    let c = a * (2 * b - a);
    let d = a * a + b * b;
    steps.push({ label: "Double k=0", a: c, b: d, k: 0, bit: null, op: `c=${c}, d=${d}` });
    a = d;
    b = c + d; // odd
    steps.push({ label: "Bit=1, odd", a, b, k: 1, bit: 1, op: `(a,b)=(${a},${b}) → k=1` });
    // bit 0
    c = a * (2 * b - a);
    d = a * a + b * b;
    a = c;
    b = d; // even
    steps.push({ label: "Bit=0, double", a, b, k: 2, bit: 0, op: `c=${c}, d=${d} → k=2` });
    // bit 1
    c = a * (2 * b - a);
    d = a * a + b * b;
    a = d;
    b = c + d; // odd
    steps.push({ label: "Bit=1, odd", a, b, k: 5, bit: 1, op: `c=${c}, d=${d}, odd → k=5` });
    // bit 0
    c = a * (2 * b - a);
    d = a * a + b * b;
    a = c;
    b = d;
    steps.push({ label: "Bit=0, double", a, b, k: 10, bit: 0, op: `c=${c}, d=${d} → k=10` });
    // bit 0
    c = a * (2 * b - a);
    d = a * a + b * b;
    a = c;
    b = d;
    steps.push({ label: "Bit=0, double", a, b, k: 20, bit: 0, op: `c=${c}, d=${d} → k=20` });
    return steps;
  }, []);

  const steps = method === "ladder" ? ladderSteps : doublingSteps;
  const curStep = Math.min(step, steps.length - 1);
  const cur = steps[curStep];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <Card>
        <SectionLabel>BUILDING M²⁰ — TWO PATHS TO F(20) = 6765</SectionLabel>
        <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
          {[
            { id: "ladder", label: "Fibonacci Ladder", sub: "Zeckendorf traversal" },
            { id: "doubling", label: "Fast Doubling", sub: "Binary traversal" }
          ].map(o => (
            <button
              key={o.id}
              onClick={() => {
                setMethod(o.id);
                setStep(0);
              }}
              style={{
                flex: 1,
                border: "none",
                borderRadius: 8,
                cursor: "pointer",
                padding: "10px 14px",
                textAlign: "left",
                background: method === o.id ? "rgba(212,160,23,0.12)" : "rgba(255,255,255,0.03)",
                border: method === o.id ? `1px solid ${GOLD}` : `1px solid rgba(255,255,255,0.06)`,
                transition: "all 0.2s ease"
              }}
            >
              <div
                style={{
                  fontFamily: MONO,
                  fontSize: 12,
                  color: method === o.id ? GOLD : "rgba(255,255,255,0.4)",
                  fontWeight: 600
                }}
              >
                {o.label}
              </div>
              <div style={{ fontFamily: MONO, fontSize: 9, color: "rgba(255,255,255,0.25)", marginTop: 2, letterSpacing: 1 }}>
                {o.sub}
              </div>
            </button>
          ))}
        </div>

        {/* Exponent decomposition */}
        <div
          style={{
            padding: 10,
            borderRadius: 8,
            background: method === "ladder" ? "rgba(212,160,23,0.04)" : "rgba(64,200,120,0.04)",
            border: `1px solid ${method === "ladder" ? "rgba(212,160,23,0.15)" : "rgba(64,200,120,0.15)"}`,
            fontFamily: MONO,
            fontSize: 11,
            color: "rgba(255,255,255,0.4)"
          }}
        >
          {method === "ladder" ?
            <>
              20 in Zeckendorf: <span style={{ color: GOLD, fontWeight: 700 }}>13 + 5 + 2</span>
              <span style={{ color: "rgba(255,255,255,0.2)", marginLeft: 8 }}>→ climb ladder to 13, then assemble</span>
            </>
          : <>
              20 in binary: <span style={{ color: GREEN, fontWeight: 700 }}>10100</span>
              <span style={{ color: "rgba(255,255,255,0.2)", marginLeft: 8 }}>→ scan bits, double-and-multiply</span>
            </>
          }
        </div>
      </Card>

      {/* Step visualization */}
      <Card>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <SectionLabel>
            STEP {curStep} / {steps.length - 1}
          </SectionLabel>
          <div style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.25)" }}>{cur.label}</div>
        </div>

        {method === "ladder" ?
          <LadderStepView
            step={cur}
            stepIdx={curStep}
          />
        : <DoublingStepView
            step={cur}
            stepIdx={curStep}
          />
        }
      </Card>

      {/* Timeline */}
      <Card>
        <SectionLabel>TIMELINE</SectionLabel>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {steps.map((s, i) => (
            <button
              key={i}
              onClick={() => setStep(i)}
              style={{
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
                padding: "6px 10px",
                fontFamily: MONO,
                fontSize: 10,
                background:
                  i === curStep ? "rgba(212,160,23,0.2)"
                  : i < curStep ? "rgba(64,200,120,0.08)"
                  : "rgba(255,255,255,0.03)",
                color:
                  i === curStep ? GOLD
                  : i < curStep ? GREEN
                  : "rgba(255,255,255,0.25)",
                border: i === curStep ? `1px solid ${GOLD}` : `1px solid rgba(255,255,255,0.06)`,
                transition: "all 0.2s ease"
              }}
            >
              {s.label}
            </button>
          ))}
        </div>
      </Card>

      {/* Navigation */}
      <div style={{ display: "flex", justifyContent: "center", gap: 8 }}>
        <button
          onClick={() => setStep(Math.max(0, step - 1))}
          style={{
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            padding: "8px 20px",
            fontFamily: MONO,
            fontSize: 11,
            background: "rgba(255,255,255,0.06)",
            color: "rgba(255,255,255,0.4)",
            border: `1px solid rgba(255,255,255,0.08)`
          }}
        >
          ← PREV
        </button>
        <button
          onClick={() => setStep(Math.min(steps.length - 1, step + 1))}
          style={{
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            padding: "8px 20px",
            fontFamily: MONO,
            fontSize: 11,
            background: "rgba(212,160,23,0.15)",
            color: GOLD,
            border: `1px solid rgba(212,160,23,0.3)`
          }}
        >
          NEXT →
        </button>
      </div>

      {/* Final result */}
      {curStep === steps.length - 1 && (
        <Card
          border="rgba(64,200,120,0.2)"
          bg="rgba(64,200,120,0.04)"
        >
          <div style={{ textAlign: "center" }}>
            <div style={{ fontFamily: MONO, fontSize: 11, color: GREEN, letterSpacing: 2, marginBottom: 8 }}>RESULT</div>
            <div style={{ fontFamily: DISPLAY, fontSize: 32, color: GOLD, fontWeight: 700 }}>F(20) = 6765</div>
            <div style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.3)", marginTop: 8 }}>
              {method === "ladder" ?
                "7 matrix multiplies via Fibonacci ladder"
              : "5 doubling steps (15 scalar multiplies) via binary scan"}
            </div>
            <div style={{ fontFamily: MONO, fontSize: 10, color: "rgba(255,255,255,0.2)", marginTop: 4 }}>
              Same matrix. Same result. Different cipher driving the traversal.
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}

function LadderStepView({ step, stepIdx }) {
  if (stepIdx === 0) {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <MatrixDisplay
          m={step.m}
          label={`M^${step.exp}`}
          highlight={[0, 1]}
        />
        <div style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.35)" }}>
          Base case. READ = {step.m[0][1]}
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <MatrixDisplay
          m={step.a}
          label=""
          size="small"
        />
        <span style={{ fontFamily: MONO, fontSize: 16, color: "rgba(255,255,255,0.2)" }}>×</span>
        <MatrixDisplay
          m={step.b}
          label=""
          size="small"
        />
        <span style={{ fontFamily: MONO, fontSize: 16, color: "rgba(255,255,255,0.2)" }}>=</span>
        <MatrixDisplay
          m={step.m}
          label={`M^${step.exp}`}
          highlight={[0, 1]}
        />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4 }}>
        <DotProductDetail
          A={step.a}
          B={step.b}
          row={0}
          col={0}
        />
        <DotProductDetail
          A={step.a}
          B={step.b}
          row={0}
          col={1}
          color={GOLD}
        />
        <DotProductDetail
          A={step.a}
          B={step.b}
          row={1}
          col={0}
        />
        <DotProductDetail
          A={step.a}
          B={step.b}
          row={1}
          col={1}
        />
      </div>

      <div style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.3)" }}>
        {step.assemble ?
          <span>
            ASSEMBLE: <span style={{ color: ORANGE }}>collecting Zeckendorf term</span>
          </span>
        : <span>
            LADDER: <span style={{ color: GOLD }}>{step.op}</span>
          </span>
        }
        <span style={{ marginLeft: 12, color: GREEN }}>
          READ = F({step.exp}) = {step.m[0][1]}
        </span>
      </div>
    </div>
  );
}

function DoublingStepView({ step, stepIdx }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div
        style={{
          display: "flex",
          gap: 20,
          alignItems: "center",
          padding: 12,
          borderRadius: 8,
          background: "rgba(255,255,255,0.02)"
        }}
      >
        <div style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.3)" }}>
          <div style={{ marginBottom: 4 }}>
            <span style={{ color: "rgba(255,255,255,0.2)", width: 16, display: "inline-block" }}>a</span>={" "}
            <span style={{ color: GOLD, fontWeight: 700, fontSize: 14 }}>{step.a}</span>
            <span style={{ color: "rgba(255,255,255,0.15)", marginLeft: 8 }}>F(k)</span>
          </div>
          <div style={{ marginBottom: 4 }}>
            <span style={{ color: "rgba(255,255,255,0.2)", width: 16, display: "inline-block" }}>b</span>={" "}
            <span style={{ color: CYAN, fontWeight: 700, fontSize: 14 }}>{step.b}</span>
            <span style={{ color: "rgba(255,255,255,0.15)", marginLeft: 8 }}>F(k+1)</span>
          </div>
          <div>
            <span style={{ color: "rgba(255,255,255,0.2)", width: 16, display: "inline-block" }}>k</span>={" "}
            <span style={{ color: GREEN, fontWeight: 700, fontSize: 14 }}>{step.k}</span>
          </div>
        </div>

        {step.bit !== null && (
          <div
            style={{
              padding: "6px 12px",
              borderRadius: 6,
              background: step.bit ? "rgba(212,160,23,0.12)" : "rgba(255,255,255,0.04)",
              border: `1px solid ${step.bit ? "rgba(212,160,23,0.3)" : "rgba(255,255,255,0.08)"}`,
              fontFamily: MONO,
              fontSize: 12,
              color: step.bit ? GOLD : "rgba(255,255,255,0.3)"
            }}
          >
            bit = {step.bit} → {step.bit ? "double + multiply" : "double only"}
          </div>
        )}
      </div>

      <div style={{ fontFamily: MONO, fontSize: 10, color: "rgba(255,255,255,0.25)" }}>{step.op}</div>

      {/* Binary display showing which bit we're at */}
      <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
        <span style={{ fontFamily: MONO, fontSize: 9, color: "rgba(255,255,255,0.2)", marginRight: 6 }}>20 =</span>
        {[1, 0, 1, 0, 0].map((b, i) => {
          const bitPosition = 4 - i;
          const processed = stepIdx > 0 && i < stepIdx;
          const current = i === stepIdx - 1 && stepIdx > 0;
          return (
            <div
              key={i}
              style={{
                width: 24,
                height: 28,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontFamily: MONO,
                fontSize: 13,
                fontWeight: current ? 700 : 400,
                borderRadius: 4,
                background:
                  current ? "rgba(212,160,23,0.2)"
                  : processed ? "rgba(64,200,120,0.08)"
                  : "rgba(255,255,255,0.03)",
                color:
                  current ? GOLD
                  : processed ? GREEN
                  : "rgba(255,255,255,0.15)",
                border: current ? `1px solid ${GOLD}` : `1px solid rgba(255,255,255,0.06)`
              }}
            >
              {b}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════════

const TABS = [
  { id: "cipher", label: "CIPHER FORK" },
  { id: "arithmetic", label: "ARITHMETIC" },
  { id: "composition", label: "COMPOSITION" }
];

export default function MatrixIdentityExplorer() {
  const [tab, setTab] = useState("cipher");

  return (
    <div
      style={{
        background: BG,
        minHeight: "100vh",
        padding: "24px 16px",
        color: "rgba(255,255,255,0.7)",
        fontFamily: MONO
      }}
    >
      <div style={{ maxWidth: 760, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 28 }}>
          <div
            style={{
              fontFamily: DISPLAY,
              fontSize: 28,
              fontWeight: 700,
              color: GOLD,
              letterSpacing: 1,
              marginBottom: 6
            }}
          >
            The Matrix Identity
          </div>
          <div
            style={{
              fontFamily: MONO,
              fontSize: 11,
              color: "rgba(255,255,255,0.25)",
              maxWidth: 480,
              margin: "0 auto",
              lineHeight: 1.6
            }}
          >
            Binary and Fibonacci are ciphers on the same matrix. The adjacency constraint, the symmetry, and the delimiter are
            one structural fact.
          </div>
        </div>

        {/* Tab bar */}
        <div
          style={{
            display: "flex",
            gap: 4,
            marginBottom: 20,
            padding: 4,
            background: "rgba(255,255,255,0.02)",
            borderRadius: 10,
            border: "1px solid rgba(255,255,255,0.04)"
          }}
        >
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                flex: 1,
                border: "none",
                borderRadius: 8,
                cursor: "pointer",
                padding: "10px 0",
                fontFamily: MONO,
                fontSize: 10,
                letterSpacing: 2,
                fontWeight: 600,
                background: tab === t.id ? "rgba(212,160,23,0.15)" : "transparent",
                color: tab === t.id ? GOLD : "rgba(212,160,23,0.3)",
                transition: "all 0.2s ease"
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Content */}
        {tab === "cipher" && <CipherForkTab />}
        {tab === "arithmetic" && <ArithmeticTab />}
        {tab === "composition" && <CompositionTab />}

        {/* Footer axiom */}
        <div
          style={{
            marginTop: 28,
            padding: 14,
            background: "rgba(255,255,255,0.02)",
            borderRadius: 8,
            border: "1px solid rgba(255,255,255,0.04)",
            textAlign: "center"
          }}
        >
          <div style={{ fontSize: 9, fontFamily: MONO, color: "rgba(212,160,23,0.3)", letterSpacing: 3, marginBottom: 8 }}>
            THE NATIVE OBJECT
          </div>
          <div style={{ fontFamily: MONO, fontSize: 11, color: "rgba(255,255,255,0.2)", lineHeight: 1.8 }}>
            <span style={{ color: "rgba(212,160,23,0.5)" }}>M</span> = [[1,1],[1,0]]
            <span style={{ color: "rgba(255,255,255,0.1)", margin: "0 12px" }}>·</span>
            <span style={{ color: GOLD }}>Allow [1,1]</span> → binary
            <span style={{ color: "rgba(255,255,255,0.1)", margin: "0 12px" }}>·</span>
            <span style={{ color: GREEN }}>Forbid [1,1]</span> → Fibonacci
          </div>
        </div>
      </div>
    </div>
  );
}
