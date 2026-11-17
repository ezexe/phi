import { useState, useMemo, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════
// PRECOMPUTED DATA from scipy analysis
// ═══════════════════════════════════════════════════════════════

const PHI = 1.618033988749895;
const PSI = -0.618033988749895;
const LOG2PHI = 0.6942419136306174;

const EIGENVALUES = {
  raw: [
    { re: 1.618034, im: 0 },
    { re: -0.618034, im: 0 }
  ],
  m1: [
    { re: 1.0, im: 0 },
    { re: -0.381966, im: 0 }
  ],
  proj: [
    { re: 1.0, im: 0 },
    { re: -1.0, im: 0 }
  ],
  m4: [
    { re: 0.95, im: 0 },
    { re: -0.362868, im: 0 }
  ],
  hippo: [
    { re: 0, im: 0.778801 },
    { re: 0, im: -0.778801 }
  ]
};

// Impulse responses (40 steps)
const IMP_RAW = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765];
const IMP_M1 = [
  1, 0.618034, 0.763932, 0.708204, 0.72949, 0.72136, 0.724465, 0.723279, 0.723733, 0.72356, 0.723633, 0.723601, 0.723613,
  0.723609, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361,
  0.72361, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361, 0.72361
];
const IMP_M4 = [
  1, 0.587134, 0.689454, 0.607197, 0.594171, 0.558171, 0.532545, 0.505093, 0.480136, 0.456024, 0.433256, 0.411482, 0.390595,
  0.370818, 0.351889, 0.333831, 0.316607, 0.300176, 0.284499, 0.269538, 0.255258, 0.241625, 0.228607, 0.216173, 0.204292,
  0.192937, 0.18208, 0.171696, 0.161761, 0.15225, 0.143141, 0.134413, 0.126044, 0.118016, 0.11031, 0.103206, 0.096393,
  0.089968, 0.083723, 0.077849
];
const IMP_HIPPO = [
  0.291181, -0.211808, -0.176614, 0.128466, 0.107124, -0.077918, -0.064972, 0.047261, 0.03941, -0.028669, -0.023902, 0.017389,
  0.014501, -0.010549, -0.008798, 0.006401, 0.005338, -0.003883, -0.003238, 0.002355, 0.001964, -0.001429, -0.001192,
  0.000867, 0.000723, -0.000526, -0.000438, 0.000319, 0.000266, -0.000193, -0.000161, 0.000117, 0.000098, -0.000071,
  -0.000059, 0.000043, 0.000036, -0.000026, -0.000022, 0.000016
];

// Frequency responses (80 points, ω/π from ~0 to 1)
const FREQ_OMEGA = [
  0.003, 0.016, 0.029, 0.041, 0.054, 0.066, 0.079, 0.092, 0.104, 0.117, 0.129, 0.142, 0.155, 0.167, 0.18, 0.192, 0.205, 0.217,
  0.23, 0.243, 0.255, 0.268, 0.28, 0.293, 0.305, 0.318, 0.331, 0.343, 0.356, 0.368, 0.381, 0.393, 0.406, 0.419, 0.431, 0.444,
  0.456, 0.469, 0.481, 0.494, 0.507, 0.519, 0.532, 0.544, 0.557, 0.57, 0.582, 0.595, 0.607, 0.62, 0.632, 0.645, 0.658, 0.67,
  0.683, 0.695, 0.708, 0.72, 0.733, 0.746, 0.758, 0.771, 0.783, 0.796, 0.808, 0.821, 0.834, 0.846, 0.859, 0.871, 0.884, 0.896,
  0.909, 0.922, 0.934, 0.947, 0.959, 0.972, 0.984, 0.997
];
const FREQ_M1_MAG = [
  72.362, 14.606, 7.512, 5.078, 3.856, 3.119, 2.622, 2.267, 2.001, 1.794, 1.629, 1.493, 1.381, 1.286, 1.206, 1.137, 1.077,
  1.025, 0.98, 0.94, 0.905, 0.874, 0.846, 0.822, 0.8, 0.78, 0.763, 0.747, 0.734, 0.722, 0.711, 0.702, 0.694, 0.687, 0.682,
  0.677, 0.674, 0.671, 0.669, 0.668, 0.668, 0.668, 0.67, 0.671, 0.674, 0.677, 0.68, 0.685, 0.689, 0.695, 0.7, 0.707, 0.713,
  0.72, 0.728, 0.736, 0.744, 0.753, 0.762, 0.771, 0.781, 0.791, 0.801, 0.811, 0.821, 0.832, 0.843, 0.853, 0.864, 0.875, 0.885,
  0.896, 0.906, 0.916, 0.926, 0.936, 0.946, 0.955, 0.964, 0.973
];
const FREQ_M4_MAG = [
  14.404, 7.695, 4.967, 3.668, 2.914, 2.415, 2.065, 1.807, 1.61, 1.455, 1.329, 1.225, 1.138, 1.063, 1.0, 0.944, 0.896, 0.854,
  0.818, 0.785, 0.757, 0.732, 0.71, 0.691, 0.674, 0.66, 0.647, 0.636, 0.627, 0.619, 0.613, 0.608, 0.604, 0.601, 0.599, 0.598,
  0.598, 0.598, 0.6, 0.602, 0.604, 0.607, 0.611, 0.615, 0.62, 0.625, 0.631, 0.637, 0.643, 0.65, 0.657, 0.664, 0.672, 0.68,
  0.688, 0.696, 0.704, 0.713, 0.721, 0.73, 0.738, 0.747, 0.756, 0.764, 0.773, 0.781, 0.79, 0.798, 0.806, 0.814, 0.822, 0.83,
  0.837, 0.845, 0.852, 0.859, 0.866, 0.872, 0.879, 0.885
];
const FREQ_HIPPO_MAG = [
  0.049, 0.051, 0.055, 0.062, 0.072, 0.085, 0.101, 0.121, 0.144, 0.17, 0.199, 0.232, 0.268, 0.306, 0.348, 0.392, 0.437, 0.484,
  0.532, 0.58, 0.627, 0.672, 0.716, 0.756, 0.793, 0.826, 0.853, 0.876, 0.893, 0.905, 0.912, 0.914, 0.912, 0.906, 0.897, 0.885,
  0.871, 0.856, 0.839, 0.822, 0.804, 0.786, 0.768, 0.751, 0.734, 0.718, 0.702, 0.687, 0.673, 0.66, 0.648, 0.637, 0.626, 0.617,
  0.608, 0.6, 0.593, 0.586, 0.581, 0.576, 0.571, 0.567, 0.564, 0.562, 0.56, 0.558, 0.557, 0.557, 0.557, 0.557, 0.558, 0.559,
  0.56, 0.562, 0.564, 0.567, 0.57, 0.573, 0.577, 0.581
];

// ═══════════════════════════════════════════════════════════════
// DESIGN TOKENS
// ═══════════════════════════════════════════════════════════════

const GOLD = "#D4A843";
const CYAN = "#5CC8D4";
const RED = "#E85454";
const GREEN = "#4ADE80";
const PURPLE = "#A78BFA";
const ORANGE = "#FB923C";
const DIM = "rgba(255,255,255,0.25)";
const DIMMER = "rgba(255,255,255,0.08)";
const BG = "#0A0E14";
const SURFACE = "#111821";
const MONO = "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace";
const SANS = "'DM Sans', 'Inter', system-ui, sans-serif";

// ═══════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════

const TABS = ["EIGENVALUES", "IMPULSE", "FREQUENCY", "SCAN TREE"];

export default function QMatrixSSM() {
  const [tab, setTab] = useState(0);

  return (
    <div style={{ background: BG, color: "#E0E0E0", minHeight: "100vh", fontFamily: SANS, fontSize: 13 }}>
      {/* Header */}
      <div style={{ padding: "20px 24px 0", borderBottom: `1px solid ${DIMMER}` }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 4 }}>
          <span style={{ fontFamily: MONO, fontSize: 18, fontWeight: 800, color: GOLD, letterSpacing: 2 }}>
            M = [[1,1],[1,0]]
          </span>
          <span style={{ fontSize: 11, color: DIM, letterSpacing: 1 }}>AS STATE SPACE MODEL</span>
        </div>
        <div style={{ fontSize: 11, color: DIM, marginBottom: 14, lineHeight: 1.5 }}>
          The Q-matrix is a companion matrix with eigenvalues φ and −1/φ. As-is it's an unstable SSM. Four stabilization
          methods. One structural winner.
        </div>
        {/* Tab bar */}
        <div style={{ display: "flex", gap: 0 }}>
          {TABS.map((t, i) => (
            <button
              key={t}
              onClick={() => setTab(i)}
              style={{
                background: tab === i ? SURFACE : "transparent",
                color: tab === i ? GOLD : DIM,
                border: "none",
                borderBottom: tab === i ? `2px solid ${GOLD}` : "2px solid transparent",
                padding: "8px 16px",
                fontFamily: MONO,
                fontSize: 10,
                letterSpacing: 1.5,
                cursor: "pointer",
                fontWeight: tab === i ? 700 : 400,
                transition: "all 0.2s"
              }}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div style={{ padding: 24 }}>
        {tab === 0 && <EigenvalueTab />}
        {tab === 1 && <ImpulseTab />}
        {tab === 2 && <FrequencyTab />}
        {tab === 3 && <ScanTreeTab />}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// TAB 1: EIGENVALUE PLANE
// ═══════════════════════════════════════════════════════════════

function EigenvalueTab() {
  const [hovered, setHovered] = useState(null);

  const W = 600,
    H = 400;
  const cx = W / 2,
    cy = H / 2;
  const scale = 130; // pixels per unit

  const methods = [
    { key: "raw", label: "Raw M", color: RED, vals: EIGENVALUES.raw, desc: "Unstable: |φ| = 1.618 > 1" },
    { key: "m1", label: "M / φ", color: GOLD, vals: EIGENVALUES.m1, desc: "Spectral norm: dominant λ → 1.0" },
    { key: "proj", label: "Unit proj", color: CYAN, vals: EIGENVALUES.proj, desc: "Both on unit circle: oscillatory" },
    { key: "m4", label: "Damped", color: GREEN, vals: EIGENVALUES.m4, desc: "γ = 0.95/φ: both |λ| < 1" },
    { key: "hippo", label: "HiPPO 2×2", color: PURPLE, vals: EIGENVALUES.hippo, desc: "Complex conjugate pair, |λ| ≈ 0.78" }
  ];

  const toX = re => cx + re * scale;
  const toY = im => cy - im * scale;

  return (
    <div>
      <div style={{ fontFamily: MONO, fontSize: 11, color: DIM, marginBottom: 16 }}>
        Eigenvalue plane — unit circle is the stability boundary. Inside = stable, outside = blowup.
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", maxWidth: W, background: SURFACE, borderRadius: 8 }}
      >
        {/* Grid */}
        {[-2, -1, 0, 1, 2].map(v => (
          <g key={`grid-${v}`}>
            <line
              x1={toX(v)}
              y1={0}
              x2={toX(v)}
              y2={H}
              stroke={DIMMER}
              strokeWidth={v === 0 ? 1 : 0.5}
            />
            <line
              x1={0}
              y1={toY(v)}
              x2={W}
              y2={toY(v)}
              stroke={DIMMER}
              strokeWidth={v === 0 ? 1 : 0.5}
            />
          </g>
        ))}

        {/* Unit circle */}
        <circle
          cx={cx}
          cy={cy}
          r={scale}
          fill="none"
          stroke="rgba(255,255,255,0.15)"
          strokeWidth={1.5}
          strokeDasharray="4 3"
        />
        <text
          x={cx + scale + 4}
          y={cy - 4}
          fill={DIM}
          fontSize={9}
          fontFamily={MONO}
        >
          |λ|=1
        </text>

        {/* Eigenvalue points */}
        {methods.map(m =>
          m.vals.map((v, vi) => {
            const x = toX(v.re);
            const y = toY(v.im);
            const isHov = hovered === m.key;
            return (
              <g
                key={`${m.key}-${vi}`}
                onMouseEnter={() => setHovered(m.key)}
                onMouseLeave={() => setHovered(null)}
                style={{ cursor: "pointer" }}
              >
                <circle
                  cx={x}
                  cy={y}
                  r={isHov ? 8 : 6}
                  fill={m.color}
                  opacity={isHov ? 1 : 0.8}
                  stroke={isHov ? "#fff" : "none"}
                  strokeWidth={1.5}
                />
                {vi === 0 && (
                  <text
                    x={x + 10}
                    y={y - 8}
                    fill={m.color}
                    fontSize={9}
                    fontFamily={MONO}
                    fontWeight={700}
                  >
                    {m.label}
                  </text>
                )}
              </g>
            );
          })
        )}

        {/* φ marker on real axis */}
        <line
          x1={toX(PHI)}
          y1={cy - 8}
          x2={toX(PHI)}
          y2={cy + 8}
          stroke={RED}
          strokeWidth={1}
          opacity={0.5}
        />
        <text
          x={toX(PHI) - 2}
          y={cy + 20}
          fill={RED}
          fontSize={8}
          fontFamily={MONO}
          textAnchor="middle"
          opacity={0.6}
        >
          φ
        </text>

        {/* -1/φ marker */}
        <line
          x1={toX(-1 / PHI)}
          y1={cy - 8}
          x2={toX(-1 / PHI)}
          y2={cy + 8}
          stroke={RED}
          strokeWidth={1}
          opacity={0.5}
        />
        <text
          x={toX(-1 / PHI)}
          y={cy + 20}
          fill={RED}
          fontSize={8}
          fontFamily={MONO}
          textAnchor="middle"
          opacity={0.6}
        >
          −1/φ
        </text>

        {/* Axis labels */}
        <text
          x={W - 20}
          y={cy - 6}
          fill={DIM}
          fontSize={9}
          fontFamily={MONO}
        >
          Re
        </text>
        <text
          x={cx + 6}
          y={14}
          fill={DIM}
          fontSize={9}
          fontFamily={MONO}
        >
          Im
        </text>
      </svg>

      {/* Legend / detail panel */}
      <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        {methods.map(m => (
          <div
            key={m.key}
            onMouseEnter={() => setHovered(m.key)}
            onMouseLeave={() => setHovered(null)}
            style={{
              padding: "10px 12px",
              borderRadius: 6,
              background: hovered === m.key ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.02)",
              border: `1px solid ${hovered === m.key ? m.color + "40" : DIMMER}`,
              cursor: "pointer",
              transition: "all 0.15s"
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: m.color }} />
              <span style={{ fontFamily: MONO, fontSize: 11, fontWeight: 700, color: m.color }}>{m.label}</span>
            </div>
            <div style={{ fontSize: 11, color: DIM, lineHeight: 1.4 }}>{m.desc}</div>
            <div style={{ fontFamily: MONO, fontSize: 10, color: "rgba(255,255,255,0.35)", marginTop: 4 }}>
              λ = [
              {m.vals
                .map(v => {
                  if (Math.abs(v.im) < 0.001) return v.re.toFixed(3);
                  return `${v.re.toFixed(1)}${v.im > 0 ? "+" : ""}${v.im.toFixed(3)}i`;
                })
                .join(", ")}
              ]
            </div>
          </div>
        ))}
      </div>

      {/* Key insight callout */}
      <div
        style={{
          marginTop: 16,
          padding: 14,
          borderRadius: 8,
          background: `${GOLD}08`,
          border: `1px solid ${GOLD}25`
        }}
      >
        <div style={{ fontFamily: MONO, fontSize: 10, color: GOLD, fontWeight: 700, letterSpacing: 1, marginBottom: 6 }}>
          ★ STRUCTURAL WINNER: M / φ
        </div>
        <div style={{ fontSize: 12, lineHeight: 1.6, color: "rgba(255,255,255,0.6)" }}>
          Dividing by the spectral radius places the dominant eigenvalue exactly on the unit circle (marginally stable) while
          the secondary eigenvalue lands at −1/φ² ≈ −0.382. The eigenvalue <em>ratio</em> λ₁/λ₂ = −φ² is preserved — the
          golden-ratio spacing survives stabilization. The recurrence becomes x(n) = x(n−1)/φ + x(n−2)/φ: a{" "}
          <strong style={{ color: GOLD }}>decaying Fibonacci filter</strong>.
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// TAB 2: IMPULSE RESPONSE
// ═══════════════════════════════════════════════════════════════

function ImpulseTab() {
  const [showRaw, setShowRaw] = useState(false);
  const [steps, setSteps] = useState(30);

  const series = [
    { key: "m1", label: "M / φ (spectral)", color: GOLD, data: IMP_M1 },
    { key: "m4", label: "Damped (γ=0.95/φ)", color: GREEN, data: IMP_M4 },
    { key: "hippo", label: "HiPPO 2×2", color: PURPLE, data: IMP_HIPPO }
  ];

  const W = 600,
    H = 300;
  const pad = { l: 50, r: 20, t: 10, b: 30 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  const visData = series.map(s => s.data.slice(0, steps));
  const allVals = visData.flat();
  const yMin = Math.min(0, ...allVals) * 1.1;
  const yMax = Math.max(...allVals) * 1.1;

  const xScale = i => pad.l + (i / (steps - 1)) * pw;
  const yScale = v => pad.t + ph - ((v - yMin) / (yMax - yMin)) * ph;

  const makePath = data =>
    data.map((v, i) => `${i === 0 ? "M" : "L"}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`).join(" ");

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div style={{ fontFamily: MONO, fontSize: 11, color: DIM }}>
          h(k) = C · Aᵏ · B — how each kernel responds to a single input pulse
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <label style={{ fontFamily: MONO, fontSize: 10, color: DIM, display: "flex", gap: 6, alignItems: "center" }}>
            <input
              type="range"
              min={8}
              max={40}
              value={steps}
              onChange={e => setSteps(+e.target.value)}
              style={{ width: 80, accentColor: GOLD }}
            />
            {steps} steps
          </label>
        </div>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", maxWidth: W, background: SURFACE, borderRadius: 8 }}
      >
        {/* Zero line */}
        <line
          x1={pad.l}
          y1={yScale(0)}
          x2={W - pad.r}
          y2={yScale(0)}
          stroke="rgba(255,255,255,0.1)"
          strokeWidth={1}
        />

        {/* Y-axis labels */}
        {[yMin, 0, yMax * 0.5, yMax].map((v, i) => (
          <text
            key={i}
            x={pad.l - 6}
            y={yScale(v) + 3}
            fill={DIM}
            fontSize={8}
            fontFamily={MONO}
            textAnchor="end"
          >
            {v.toFixed(2)}
          </text>
        ))}

        {/* X-axis labels */}
        {[0, Math.floor(steps / 4), Math.floor(steps / 2), Math.floor((3 * steps) / 4), steps - 1].map(i => (
          <text
            key={i}
            x={xScale(i)}
            y={H - 5}
            fill={DIM}
            fontSize={8}
            fontFamily={MONO}
            textAnchor="middle"
          >
            {i}
          </text>
        ))}

        {/* Series */}
        {series.map((s, si) => (
          <path
            key={s.key}
            d={makePath(s.data.slice(0, steps))}
            fill="none"
            stroke={s.color}
            strokeWidth={1.5}
            opacity={0.9}
          />
        ))}

        {/* Dots at key points */}
        {series.map(s =>
          [0, 1, 2, 3, 5, 8]
            .filter(i => i < steps)
            .map(i => (
              <circle
                key={`${s.key}-${i}`}
                cx={xScale(i)}
                cy={yScale(s.data[i])}
                r={2.5}
                fill={s.color}
                opacity={0.7}
              />
            ))
        )}

        {/* Convergence annotation for M/φ */}
        {steps >= 15 && (
          <g>
            <line
              x1={xScale(12)}
              y1={yScale(IMP_M1[12])}
              x2={xScale(steps - 1)}
              y2={yScale(IMP_M1[12])}
              stroke={GOLD}
              strokeWidth={0.5}
              strokeDasharray="3 2"
              opacity={0.4}
            />
            <text
              x={xScale(steps - 1) + 2}
              y={yScale(IMP_M1[12]) + 3}
              fill={GOLD}
              fontSize={8}
              fontFamily={MONO}
              opacity={0.6}
            >
              → 1/√5
            </text>
          </g>
        )}

        <text
          x={pad.l + 5}
          y={H - 14}
          fill={DIM}
          fontSize={8}
          fontFamily={MONO}
        >
          k (time step)
        </text>
      </svg>

      {/* Legend */}
      <div style={{ display: "flex", gap: 20, marginTop: 12 }}>
        {series.map(s => (
          <div
            key={s.key}
            style={{ display: "flex", alignItems: "center", gap: 6 }}
          >
            <div style={{ width: 16, height: 2, background: s.color, borderRadius: 1 }} />
            <span style={{ fontFamily: MONO, fontSize: 10, color: s.color }}>{s.label}</span>
          </div>
        ))}
      </div>

      {/* Analysis cards */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginTop: 16 }}>
        <AnalysisCard
          color={GOLD}
          title="M / φ impulse"
          body={`Fibonacci numbers divided by φᵏ. Converges to 1/√5 ≈ 0.4472. The golden-ratio decay creates an exponentially-weighted moving average where the weight at lag k is F(k+1)/φᵏ — a unique filter shape that no other 2×2 matrix produces.`}
        />
        <AnalysisCard
          color={GREEN}
          title="Damped impulse"
          body={`Same Fibonacci structure but with strictly decaying envelope (γ < 1). For SSM use: this is the practical choice. The decay rate γ is a tunable hyperparameter controlling memory length. At γ = 0.95/φ, effective memory ≈ 20 steps.`}
        />
        <AnalysisCard
          color={PURPLE}
          title="HiPPO 2×2"
          body={`Complex conjugate eigenvalues produce oscillatory decay — fundamentally different character. HiPPO memorizes with Legendre polynomial basis (all frequencies). The Q-matrix memorizes with Fibonacci basis (low-pass). Different signal priors.`}
        />
      </div>
    </div>
  );
}

function AnalysisCard({ color, title, body }) {
  return (
    <div
      style={{
        padding: 12,
        borderRadius: 6,
        background: `${color}06`,
        border: `1px solid ${color}20`
      }}
    >
      <div style={{ fontFamily: MONO, fontSize: 10, fontWeight: 700, color, letterSpacing: 0.5, marginBottom: 6 }}>
        {title}
      </div>
      <div style={{ fontSize: 11, lineHeight: 1.5, color: "rgba(255,255,255,0.5)" }}>{body}</div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// TAB 3: FREQUENCY RESPONSE
// ═══════════════════════════════════════════════════════════════

function FrequencyTab() {
  const [logScale, setLogScale] = useState(false);

  const W = 600,
    H = 320;
  const pad = { l: 55, r: 20, t: 15, b: 35 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  const series = [
    { label: "M / φ", color: GOLD, mag: FREQ_M1_MAG },
    { label: "Damped", color: GREEN, mag: FREQ_M4_MAG },
    { label: "HiPPO", color: PURPLE, mag: FREQ_HIPPO_MAG }
  ];

  // Cap display for log scale sanity
  const maxMag = logScale ? 2 : 20;
  const minMag = logScale ? -1.5 : 0;

  const xScale = i => pad.l + (i / (FREQ_OMEGA.length - 1)) * pw;
  const yScale = v => {
    const val = logScale ? Math.log10(Math.max(v, 0.001)) : Math.min(v, maxMag);
    const range = logScale ? 2 - -1.5 : maxMag;
    const base = logScale ? -1.5 : 0;
    return pad.t + ph - ((val - base) / range) * ph;
  };

  const makePath = data =>
    data.map((v, i) => `${i === 0 ? "M" : "L"}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`).join(" ");

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div style={{ fontFamily: MONO, fontSize: 11, color: DIM }}>|H(e^jω)| — magnitude response, ω from 0 to π</div>
        <button
          onClick={() => setLogScale(!logScale)}
          style={{
            background: logScale ? `${GOLD}20` : DIMMER,
            border: `1px solid ${logScale ? GOLD + "40" : DIMMER}`,
            color: logScale ? GOLD : DIM,
            fontFamily: MONO,
            fontSize: 10,
            padding: "4px 10px",
            borderRadius: 4,
            cursor: "pointer"
          }}
        >
          {logScale ? "LOG" : "LINEAR"}
        </button>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", maxWidth: W, background: SURFACE, borderRadius: 8 }}
      >
        {/* Grid */}
        {[0, 0.25, 0.5, 0.75, 1.0].map(f => {
          const x = pad.l + f * pw;
          return (
            <line
              key={f}
              x1={x}
              y1={pad.t}
              x2={x}
              y2={H - pad.b}
              stroke={DIMMER}
              strokeWidth={0.5}
            />
          );
        })}

        {/* Y-axis ticks */}
        {(logScale ? [-1, 0, 1, 2] : [0, 5, 10, 15, 20]).map(v => {
          const y = yScale(logScale ? Math.pow(10, v) : v);
          if (y < pad.t || y > H - pad.b) return null;
          return (
            <g key={v}>
              <line
                x1={pad.l}
                y1={y}
                x2={W - pad.r}
                y2={y}
                stroke={DIMMER}
                strokeWidth={0.5}
              />
              <text
                x={pad.l - 6}
                y={y + 3}
                fill={DIM}
                fontSize={8}
                fontFamily={MONO}
                textAnchor="end"
              >
                {logScale ? `10^${v}` : v}
              </text>
            </g>
          );
        })}

        {/* Unity gain line */}
        <line
          x1={pad.l}
          y1={yScale(1)}
          x2={W - pad.r}
          y2={yScale(1)}
          stroke="rgba(255,255,255,0.2)"
          strokeWidth={1}
          strokeDasharray="4 3"
        />
        <text
          x={W - pad.r + 2}
          y={yScale(1) + 3}
          fill={DIM}
          fontSize={8}
          fontFamily={MONO}
        >
          0 dB
        </text>

        {/* Series */}
        {series.map(s => (
          <path
            key={s.label}
            d={makePath(s.mag)}
            fill="none"
            stroke={s.color}
            strokeWidth={1.5}
            opacity={0.9}
          />
        ))}

        {/* X-axis */}
        {[0, 0.25, 0.5, 0.75, 1.0].map(f => {
          const idx = Math.round(f * (FREQ_OMEGA.length - 1));
          return (
            <text
              key={f}
              x={pad.l + f * pw}
              y={H - 8}
              fill={DIM}
              fontSize={8}
              fontFamily={MONO}
              textAnchor="middle"
            >
              {f === 0 ?
                "DC"
              : f === 1 ?
                "π"
              : `${f}π`}
            </text>
          );
        })}

        {/* Crossover annotation */}
        <text
          x={pad.l + pw * 0.38}
          y={pad.t + 10}
          fill={PURPLE}
          fontSize={8}
          fontFamily={MONO}
          opacity={0.7}
        >
          HiPPO peaks mid-band
        </text>
        <text
          x={pad.l + 10}
          y={pad.t + 10}
          fill={GOLD}
          fontSize={8}
          fontFamily={MONO}
          opacity={0.7}
        >
          φ-kernel peaks at DC
        </text>
      </svg>

      <div style={{ display: "flex", gap: 20, marginTop: 12 }}>
        {series.map(s => (
          <div
            key={s.label}
            style={{ display: "flex", alignItems: "center", gap: 6 }}
          >
            <div style={{ width: 16, height: 2, background: s.color, borderRadius: 1 }} />
            <span style={{ fontFamily: MONO, fontSize: 10, color: s.color }}>{s.label}</span>
          </div>
        ))}
      </div>

      {/* Interpretation */}
      <div
        style={{
          marginTop: 16,
          padding: 14,
          borderRadius: 8,
          background: `${CYAN}06`,
          border: `1px solid ${CYAN}20`
        }}
      >
        <div style={{ fontFamily: MONO, fontSize: 10, color: CYAN, fontWeight: 700, letterSpacing: 1, marginBottom: 8 }}>
          THE FUNDAMENTAL TRADEOFF
        </div>
        <div style={{ fontSize: 12, lineHeight: 1.7, color: "rgba(255,255,255,0.55)" }}>
          The stabilized Q-matrix is a <strong style={{ color: GOLD }}>low-pass filter</strong> with massive DC gain and
          monotonic rolloff. It emphasizes recent history exponentially, weighted by Fibonacci numbers. HiPPO's complex
          eigenvalues produce a <strong style={{ color: PURPLE }}>bandpass/broadband</strong> response — it spreads memory
          across all frequencies via Legendre polynomial basis.
          <br />
          <br />
          This means the Q-matrix kernel is structurally suited to <em>signals with the adjacency constraint</em> — signals
          where local correlations dominate, exactly the regime where the golden mean shift capacity (log₂(φ) ≈ 0.694
          bits/symbol) is the information-theoretic bound. HiPPO is better for long-range dependencies across diverse
          frequency content.
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// TAB 4: PARALLEL SCAN TREE
// ═══════════════════════════════════════════════════════════════

function ScanTreeTab() {
  const [mode, setMode] = useState("binary"); // "binary" | "fibonacci"

  const W = 620,
    H = 380;

  // Binary doubling tree for 8 elements
  const binaryTree = {
    levels: [
      [
        { label: "M¹", w: 1 },
        { label: "M¹", w: 1 },
        { label: "M¹", w: 1 },
        { label: "M¹", w: 1 },
        { label: "M¹", w: 1 },
        { label: "M¹", w: 1 },
        { label: "M¹", w: 1 },
        { label: "M¹", w: 1 }
      ],
      [
        { label: "M²", w: 2, op: "1+1" },
        { label: "M²", w: 2, op: "1+1" },
        { label: "M²", w: 2, op: "1+1" },
        { label: "M²", w: 2, op: "1+1" }
      ],
      [
        { label: "M⁴", w: 4, op: "2+2" },
        { label: "M⁴", w: 4, op: "2+2" }
      ],
      [{ label: "M⁸", w: 8, op: "4+4" }]
    ],
    depth: 3,
    desc: "Binary reduction: same-sized operands at every level. Powers of 2."
  };

  // Fibonacci ladder tree for 8
  const fibTree = {
    levels: [
      [{ label: "M¹", w: 1 }],
      [{ label: "M²", w: 2, op: "1+1" }],
      [{ label: "M³", w: 3, op: "2+1" }],
      [{ label: "M⁵", w: 5, op: "3+2" }],
      [{ label: "M⁸", w: 8, op: "5+3" }]
    ],
    depth: 4,
    desc: "Fibonacci ladder: consecutive Fibonacci products. Each step reuses both previous results."
  };

  const tree = mode === "binary" ? binaryTree : fibTree;
  const nodeColor = mode === "binary" ? CYAN : GOLD;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div style={{ fontFamily: MONO, fontSize: 11, color: DIM }}>
          Two ways to compute M⁸: binary reduction vs. Fibonacci ladder
        </div>
        <div style={{ display: "flex", gap: 0, borderRadius: 4, overflow: "hidden", border: `1px solid ${DIMMER}` }}>
          {["binary", "fibonacci"].map(m => (
            <button
              key={m}
              onClick={() => setMode(m)}
              style={{
                background:
                  mode === m ?
                    m === "binary" ?
                      `${CYAN}20`
                    : `${GOLD}20`
                  : "transparent",
                color:
                  mode === m ?
                    m === "binary" ?
                      CYAN
                    : GOLD
                  : DIM,
                border: "none",
                padding: "6px 14px",
                fontFamily: MONO,
                fontSize: 10,
                cursor: "pointer",
                letterSpacing: 1
              }}
            >
              {m.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", maxWidth: W, background: SURFACE, borderRadius: 8 }}
      >
        {tree.levels.map((level, li) => {
          const y = 30 + li * (mode === "binary" ? 80 : 65);
          const totalW = level.reduce((s, n) => s + n.w, 0);
          const gap = (W - 60) / (mode === "binary" ? 8 : 1);

          return level.map((node, ni) => {
            let x;
            if (mode === "binary") {
              // Evenly space across width based on position
              const positions = level.length;
              const spacing = (W - 80) / positions;
              x = 40 + spacing * ni + spacing / 2;
            } else {
              // Center single chain
              x = W / 2;
            }

            const boxW = mode === "binary" ? Math.max(40, node.w * 12 + 24) : 80;
            const boxH = 28;

            return (
              <g key={`${li}-${ni}`}>
                {/* Connection lines to children */}
                {li > 0 && mode === "binary" && (
                  <g>
                    {/* Lines from previous level */}
                    {[0, 1].map(ci => {
                      const prevLevel = tree.levels[li - 1];
                      const prevSpacing = (W - 80) / prevLevel.length;
                      const childIdx = ni * 2 + ci;
                      if (childIdx >= prevLevel.length) return null;
                      const cx = 40 + prevSpacing * childIdx + prevSpacing / 2;
                      return (
                        <line
                          key={ci}
                          x1={cx}
                          y1={y - 80 + boxH + 4}
                          x2={x}
                          y2={y - 4}
                          stroke={nodeColor}
                          strokeWidth={0.8}
                          opacity={0.3}
                        />
                      );
                    })}
                  </g>
                )}
                {li > 0 && mode === "fibonacci" && (
                  <line
                    x1={x}
                    y1={y - 65 + boxH + 4}
                    x2={x}
                    y2={y - 4}
                    stroke={nodeColor}
                    strokeWidth={0.8}
                    opacity={0.3}
                  />
                )}

                {/* Node box */}
                <rect
                  x={x - boxW / 2}
                  y={y}
                  width={boxW}
                  height={boxH}
                  rx={4}
                  fill={`${nodeColor}15`}
                  stroke={nodeColor}
                  strokeWidth={1}
                  opacity={0.8}
                />
                <text
                  x={x}
                  y={y + boxH / 2 + 4}
                  fill={nodeColor}
                  fontSize={11}
                  fontFamily={MONO}
                  fontWeight={700}
                  textAnchor="middle"
                >
                  {node.label}
                </text>

                {/* Operation label */}
                {node.op && (
                  <text
                    x={x + boxW / 2 + 6}
                    y={y + boxH / 2 + 3}
                    fill={DIM}
                    fontSize={8}
                    fontFamily={MONO}
                  >
                    {node.op}
                  </text>
                )}
              </g>
            );
          });
        })}

        {/* Depth annotation */}
        <text
          x={12}
          y={H - 10}
          fill={DIM}
          fontSize={9}
          fontFamily={MONO}
        >
          depth = {tree.depth} multiplications
        </text>
      </svg>

      {/* Comparison table */}
      <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div
          style={{
            padding: 14,
            borderRadius: 6,
            background: mode === "binary" ? `${CYAN}08` : "rgba(255,255,255,0.02)",
            border: `1px solid ${mode === "binary" ? CYAN + "30" : DIMMER}`
          }}
        >
          <div style={{ fontFamily: MONO, fontSize: 10, fontWeight: 700, color: CYAN, marginBottom: 8 }}>BINARY SCAN</div>
          <div style={{ fontSize: 11, lineHeight: 1.6, color: "rgba(255,255,255,0.5)" }}>
            Standard parallel prefix. All operands at each level have equal size. Depth = log₂(n). This is what S5 and LRU
            use. The reduction tree is a balanced binary tree — every intermediate product is M^(2^k). Maximally parallel: all
            multiplications at each level are independent.
          </div>
          <div style={{ fontFamily: MONO, fontSize: 10, color: CYAN, marginTop: 8 }}>
            Path to M²⁰: 1→2→4→5→10→20 (5 mults)
          </div>
        </div>
        <div
          style={{
            padding: 14,
            borderRadius: 6,
            background: mode === "fibonacci" ? `${GOLD}08` : "rgba(255,255,255,0.02)",
            border: `1px solid ${mode === "fibonacci" ? GOLD + "30" : DIMMER}`
          }}
        >
          <div style={{ fontFamily: MONO, fontSize: 10, fontWeight: 700, color: GOLD, marginBottom: 8 }}>
            FIBONACCI LADDER
          </div>
          <div style={{ fontSize: 11, lineHeight: 1.6, color: "rgba(255,255,255,0.5)" }}>
            Sequential but self-similar. Each step combines the two most recent results — the same recurrence the matrix
            encodes. Intermediate products are M^F(k): every node has Fibonacci-structured entries. This is the only matrix
            where the scan tree is self-similar with its own content.
          </div>
          <div style={{ fontFamily: MONO, fontSize: 10, color: GOLD, marginTop: 8 }}>
            Path to M²⁰: 1→2→3→5→8→13→(assemble via Zeckendorf)
          </div>
        </div>
      </div>

      {/* Key insight */}
      <div
        style={{
          marginTop: 16,
          padding: 14,
          borderRadius: 8,
          background: `${GOLD}06`,
          border: `1px solid ${GOLD}20`
        }}
      >
        <div style={{ fontFamily: MONO, fontSize: 10, color: GOLD, fontWeight: 700, letterSpacing: 1, marginBottom: 6 }}>
          ★ THE SELF-SIMILARITY PROPERTY
        </div>
        <div style={{ fontSize: 12, lineHeight: 1.7, color: "rgba(255,255,255,0.55)" }}>
          For a <em>general</em> SSM matrix A, the intermediate products A^k in a parallel scan have no special structure —
          they're just arbitrary matrices. For the Q-matrix,{" "}
          <strong style={{ color: GOLD }}>every intermediate product M^k = [[F(k+1), F(k)], [F(k), F(k−1)]]</strong> —
          Fibonacci numbers all the way down. The scan tree's content mirrors its own structure.
          <br />
          <br />
          This means a Fibonacci-based parallel scan could exploit the Fibonacci entries for <em>cheaper multiplications</em>
          at each node (since F(k+1) = F(k) + F(k−1), one addition replaces one entry computation), whereas binary scan treats
          each 2×2 product as four independent multiplies and two additions.
        </div>
      </div>
    </div>
  );
}
