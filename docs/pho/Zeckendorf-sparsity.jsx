import { useState, useMemo, useCallback } from "react";

const PHI = (1.0 + Math.sqrt(5)) / 2;
const LOG2PHI = Math.log2(PHI);
const LOG2_6_OVER_4 = Math.log2(6) / 4;

const GOLD = "#D4A843";
const CYAN = "#5CC8D4";
const RED = "#E85454";
const GREEN = "#4ADE80";
const PURPLE = "#A78BFA";
const DIM = "rgba(255,255,255,0.25)";
const DIMMER = "rgba(255,255,255,0.08)";
const BG = "#0A0E14";
const SURFACE = "#111821";
const MONO = "'JetBrains Mono','Fira Code','SF Mono',monospace";
const SANS = "'DM Sans','Inter',system-ui,sans-serif";

// Generate all valid Zeckendorf masks for width n
function genZeckMasks(n) {
  const masks = [];
  for (let i = 0; i < 1 << n; i++) {
    let valid = true;
    for (let b = 0; b < n - 1; b++) {
      if ((i >> b) & 1 && (i >> (b + 1)) & 1) {
        valid = false;
        break;
      }
    }
    if (valid) {
      const bits = [];
      for (let b = 0; b < n; b++) bits.push((i >> b) & 1);
      masks.push(bits);
    }
  }
  return masks;
}

// Generate all valid 2:4 masks for width n (must be mult of 4)
function gen24Masks(n) {
  if (n % 4 !== 0) return [];
  const groupMasks = [];
  for (let i = 0; i < 16; i++) {
    let cnt = 0;
    for (let b = 0; b < 4; b++) if ((i >> b) & 1) cnt++;
    if (cnt === 2) {
      const bits = [];
      for (let b = 0; b < 4; b++) bits.push((i >> b) & 1);
      groupMasks.push(bits);
    }
  }
  // For display: just return masks for a single group repeated
  const masks = [];
  const nGroups = n / 4;
  function recurse(groups, depth) {
    if (depth === nGroups) {
      masks.push(groups.flat());
      return;
    }
    for (const gm of groupMasks) recurse([...groups, gm], depth + 1);
  }
  if (n <= 16) recurse([], 0);
  return masks;
}

const TABS = ["MASKS", "CAPACITY", "DENSITY", "STRUCTURE"];

export default function ZeckSparsity() {
  const [tab, setTab] = useState(0);

  return (
    <div style={{ background: BG, color: "#E0E0E0", minHeight: "100vh", fontFamily: SANS, fontSize: 13 }}>
      <div style={{ padding: "20px 24px 0", borderBottom: `1px solid ${DIMMER}` }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 4 }}>
          <span style={{ fontFamily: MONO, fontSize: 16, fontWeight: 800, color: GOLD, letterSpacing: 1.5 }}>
            ZECKENDORF SPARSITY
          </span>
          <span style={{ fontSize: 11, color: DIM, letterSpacing: 1 }}>vs NVIDIA 2:4</span>
        </div>
        <div style={{ fontSize: 11, color: DIM, marginBottom: 14, lineHeight: 1.5 }}>
          "No consecutive 1s" as a structured pruning mask for neural networks. Tier 3 contribution.
        </div>
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
      <div style={{ padding: 24 }}>
        {tab === 0 && <MasksTab />}
        {tab === 1 && <CapacityTab />}
        {tab === 2 && <DensityTab />}
        {tab === 3 && <StructureTab />}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// TAB 1: VISUAL MASK COMPARISON
// ═══════════════════════════════════════════════════════════════

function MasksTab() {
  const [width, setWidth] = useState(12);
  const [page, setPage] = useState(0);
  const perPage = 24;

  const zeckMasks = useMemo(() => genZeckMasks(width), [width]);
  const t24Masks = useMemo(() => (width % 4 === 0 ? gen24Masks(width) : []), [width]);

  const zeckPage = zeckMasks.slice(page * perPage, (page + 1) * perPage);
  const t24Page = t24Masks.slice(page * perPage, (page + 1) * perPage);

  const cellSize = width <= 12 ? 14 : 10;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div style={{ fontFamily: MONO, fontSize: 11, color: DIM }}>
          All valid masks for width n — showing page {page + 1}
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <label style={{ fontFamily: MONO, fontSize: 10, color: DIM, display: "flex", gap: 6, alignItems: "center" }}>
            n=
            <select
              value={width}
              onChange={e => {
                setWidth(+e.target.value);
                setPage(0);
              }}
              style={{
                background: SURFACE,
                color: GOLD,
                border: `1px solid ${DIMMER}`,
                borderRadius: 4,
                padding: "2px 6px",
                fontFamily: MONO,
                fontSize: 10
              }}
            >
              {[4, 8, 12, 16].map(v => (
                <option
                  key={v}
                  value={v}
                >
                  {v}
                </option>
              ))}
            </select>
          </label>
          <div style={{ display: "flex", gap: 4 }}>
            <button
              onClick={() => setPage(Math.max(0, page - 1))}
              disabled={page === 0}
              style={{
                background: DIMMER,
                border: "none",
                color: DIM,
                padding: "2px 8px",
                borderRadius: 3,
                cursor: "pointer",
                fontFamily: MONO,
                fontSize: 10
              }}
            >
              ←
            </button>
            <button
              onClick={() => setPage(page + 1)}
              disabled={(page + 1) * perPage >= zeckMasks.length}
              style={{
                background: DIMMER,
                border: "none",
                color: DIM,
                padding: "2px 8px",
                borderRadius: 3,
                cursor: "pointer",
                fontFamily: MONO,
                fontSize: 10
              }}
            >
              →
            </button>
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        {/* Zeckendorf column */}
        <div>
          <div style={{ fontFamily: MONO, fontSize: 10, fontWeight: 700, color: GOLD, letterSpacing: 1, marginBottom: 8 }}>
            ZECKENDORF — {zeckMasks.length} masks
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {zeckPage.map((mask, mi) => (
              <div
                key={mi}
                style={{ display: "flex", gap: 1 }}
              >
                {mask.map((b, bi) => (
                  <div
                    key={bi}
                    style={{
                      width: cellSize,
                      height: cellSize,
                      borderRadius: 2,
                      background: b ? GOLD : "rgba(255,255,255,0.04)",
                      border: `1px solid ${b ? GOLD + "60" : "rgba(255,255,255,0.06)"}`
                    }}
                  />
                ))}
                <span style={{ fontFamily: MONO, fontSize: 8, color: DIM, marginLeft: 4, lineHeight: `${cellSize}px` }}>
                  {((mask.reduce((s, b) => s + b, 0) / width) * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* 2:4 column */}
        <div>
          <div style={{ fontFamily: MONO, fontSize: 10, fontWeight: 700, color: CYAN, letterSpacing: 1, marginBottom: 8 }}>
            2:4 — {t24Masks.length} masks
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {t24Page.map((mask, mi) => (
              <div
                key={mi}
                style={{ display: "flex", gap: 1 }}
              >
                {mask.map((b, bi) => (
                  <div
                    key={bi}
                    style={{
                      width: cellSize,
                      height: cellSize,
                      borderRadius: 2,
                      background: b ? CYAN : "rgba(255,255,255,0.04)",
                      border: `1px solid ${b ? CYAN + "60" : "rgba(255,255,255,0.06)"}`,
                      // Show group boundaries
                      marginRight: bi % 4 === 3 && bi < mask.length - 1 ? 4 : 0
                    }}
                  />
                ))}
                <span style={{ fontFamily: MONO, fontSize: 8, color: DIM, marginLeft: 4, lineHeight: `${cellSize}px` }}>
                  50%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ marginTop: 16, padding: 12, borderRadius: 6, background: `${GOLD}06`, border: `1px solid ${GOLD}20` }}>
        <div style={{ fontSize: 11, lineHeight: 1.6, color: "rgba(255,255,255,0.5)" }}>
          <strong style={{ color: GOLD }}>Visual pattern:</strong> Zeckendorf masks show organic variation — density ranges
          from 0% to 50%, with most masks around 25-35% density. Every mask is shift-invariant (no block alignment). The 2:4
          masks are rigidly grouped, always exactly 50% dense, with visible block boundaries every 4 positions.
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// TAB 2: CAPACITY COMPARISON
// ═══════════════════════════════════════════════════════════════

function CapacityTab() {
  const W = 580,
    H = 300;
  const pad = { l: 55, r: 20, t: 15, b: 35 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  // Compute capacity at various n
  function fibCount(n) {
    if (n <= 0) return 1;
    let a = 1,
      b = 2;
    for (let i = 1; i < n; i++) {
      [a, b] = [b, a + b];
    }
    return b;
  }

  const ns = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32];
  const zCaps = ns.map(n => Math.log2(fibCount(n)) / n);
  const tCaps = ns.map(n => (n % 4 === 0 ? LOG2_6_OVER_4 : null));

  const xScale = i => pad.l + (i / (ns.length - 1)) * pw;
  const yScale = v => pad.t + ph - ((v - 0.5) / 0.55) * ph;

  const zPath = ns.map((n, i) => `${i === 0 ? "M" : "L"}${xScale(i).toFixed(1)},${yScale(zCaps[i]).toFixed(1)}`).join(" ");

  return (
    <div>
      <div style={{ fontFamily: MONO, fontSize: 11, color: DIM, marginBottom: 16 }}>
        Bits of information per mask position — higher = more expressiveness per element
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", maxWidth: W, background: SURFACE, borderRadius: 8 }}
      >
        {/* Grid */}
        {[0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map(v => (
          <g key={v}>
            <line
              x1={pad.l}
              y1={yScale(v)}
              x2={W - pad.r}
              y2={yScale(v)}
              stroke={DIMMER}
              strokeWidth={0.5}
            />
            <text
              x={pad.l - 6}
              y={yScale(v) + 3}
              fill={DIM}
              fontSize={8}
              fontFamily={MONO}
              textAnchor="end"
            >
              {v.toFixed(1)}
            </text>
          </g>
        ))}

        {/* Asymptotic lines */}
        <line
          x1={pad.l}
          y1={yScale(LOG2PHI)}
          x2={W - pad.r}
          y2={yScale(LOG2PHI)}
          stroke={GOLD}
          strokeWidth={1}
          strokeDasharray="6 3"
          opacity={0.5}
        />
        <text
          x={W - pad.r + 2}
          y={yScale(LOG2PHI) + 3}
          fill={GOLD}
          fontSize={8}
          fontFamily={MONO}
        >
          log₂φ
        </text>

        <line
          x1={pad.l}
          y1={yScale(LOG2_6_OVER_4)}
          x2={W - pad.r}
          y2={yScale(LOG2_6_OVER_4)}
          stroke={CYAN}
          strokeWidth={1}
          strokeDasharray="6 3"
          opacity={0.5}
        />
        <text
          x={W - pad.r + 2}
          y={yScale(LOG2_6_OVER_4) + 3}
          fill={CYAN}
          fontSize={8}
          fontFamily={MONO}
        >
          2:4
        </text>

        <line
          x1={pad.l}
          y1={yScale(1.0)}
          x2={W - pad.r}
          y2={yScale(1.0)}
          stroke="rgba(255,255,255,0.2)"
          strokeWidth={1}
          strokeDasharray="3 2"
        />
        <text
          x={W - pad.r + 2}
          y={yScale(1.0) + 3}
          fill={DIM}
          fontSize={8}
          fontFamily={MONO}
        >
          binary
        </text>

        {/* Zeckendorf curve */}
        <path
          d={zPath}
          fill="none"
          stroke={GOLD}
          strokeWidth={2}
        />
        {ns.map((n, i) => (
          <circle
            key={`z-${n}`}
            cx={xScale(i)}
            cy={yScale(zCaps[i])}
            r={3}
            fill={GOLD}
          />
        ))}

        {/* 2:4 points */}
        {ns.map((n, i) =>
          tCaps[i] !== null ?
            <circle
              key={`t-${n}`}
              cx={xScale(i)}
              cy={yScale(tCaps[i])}
              r={3}
              fill={CYAN}
            />
          : null
        )}

        {/* X labels */}
        {ns
          .filter((n, i) => i % 2 === 0)
          .map((n, i) => {
            const idx = ns.indexOf(n);
            return (
              <text
                key={n}
                x={xScale(idx)}
                y={H - 8}
                fill={DIM}
                fontSize={8}
                fontFamily={MONO}
                textAnchor="middle"
              >
                {n}
              </text>
            );
          })}
        <text
          x={pad.l + pw / 2}
          y={H - 0}
          fill={DIM}
          fontSize={8}
          fontFamily={MONO}
          textAnchor="middle"
        >
          layer width n
        </text>
      </svg>

      <div style={{ display: "flex", gap: 20, marginTop: 12 }}>
        {[
          { label: "Zeckendorf", color: GOLD },
          { label: "NVIDIA 2:4", color: CYAN }
        ].map(s => (
          <div
            key={s.label}
            style={{ display: "flex", alignItems: "center", gap: 6 }}
          >
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: s.color }} />
            <span style={{ fontFamily: MONO, fontSize: 10, color: s.color }}>{s.label}</span>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
        <InfoCard
          color={GOLD}
          title="ZECKENDORF"
          value="0.6942"
          unit="bits/pos"
          body="Converges from above to log₂(φ). Proven capacity of the golden mean shift. 60+ years of information-theoretic foundation."
        />
        <InfoCard
          color={CYAN}
          title="NVIDIA 2:4"
          value="0.6462"
          unit="bits/pos"
          body="Constant at log₂(6)/4 regardless of n. Block-aligned constraint with no classical information-theoretic name."
        />
        <InfoCard
          color={GREEN}
          title="DELTA"
          value="+7.4%"
          unit="more capacity"
          body="Zeckendorf encodes more information per position. More valid masks means more expressiveness for the pruning search."
        />
      </div>
    </div>
  );
}

function InfoCard({ color, title, value, unit, body }) {
  return (
    <div style={{ padding: 12, borderRadius: 6, background: `${color}06`, border: `1px solid ${color}20` }}>
      <div style={{ fontFamily: MONO, fontSize: 9, fontWeight: 700, color, letterSpacing: 1, marginBottom: 4 }}>{title}</div>
      <div style={{ fontFamily: MONO, fontSize: 22, fontWeight: 800, color, marginBottom: 2 }}>{value}</div>
      <div style={{ fontFamily: MONO, fontSize: 9, color: `${color}80`, marginBottom: 8 }}>{unit}</div>
      <div style={{ fontSize: 10, lineHeight: 1.5, color: "rgba(255,255,255,0.4)" }}>{body}</div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// TAB 3: DENSITY DISTRIBUTION
// ═══════════════════════════════════════════════════════════════

function DensityTab() {
  const n = 16;
  const masks = useMemo(() => genZeckMasks(n), []);
  const densityDist = useMemo(() => {
    const counts = {};
    masks.forEach(m => {
      const k = m.reduce((s, b) => s + b, 0);
      counts[k] = (counts[k] || 0) + 1;
    });
    return counts;
  }, [masks]);

  const maxCount = Math.max(...Object.values(densityDist));
  const W = 580,
    H = 260;
  const pad = { l: 45, r: 20, t: 15, b: 35 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const ks = Object.keys(densityDist)
    .map(Number)
    .sort((a, b) => a - b);
  const barW = pw / (n / 2 + 2);

  return (
    <div>
      <div style={{ fontFamily: MONO, fontSize: 11, color: DIM, marginBottom: 16 }}>
        How many active weights per mask? (n={n}, {masks.length} total Zeckendorf masks)
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", maxWidth: W, background: SURFACE, borderRadius: 8 }}
      >
        {/* Bars */}
        {ks.map((k, i) => {
          const x = pad.l + (k + 0.5) * barW;
          const h = (densityDist[k] / maxCount) * ph;
          const pct = ((densityDist[k] / masks.length) * 100).toFixed(1);
          return (
            <g key={k}>
              <rect
                x={x}
                y={pad.t + ph - h}
                width={barW * 0.7}
                height={h}
                fill={GOLD}
                opacity={0.7}
                rx={2}
              />
              <text
                x={x + barW * 0.35}
                y={pad.t + ph - h - 4}
                fill={GOLD}
                fontSize={8}
                fontFamily={MONO}
                textAnchor="middle"
              >
                {pct}%
              </text>
              <text
                x={x + barW * 0.35}
                y={H - 10}
                fill={DIM}
                fontSize={8}
                fontFamily={MONO}
                textAnchor="middle"
              >
                {k}
              </text>
            </g>
          );
        })}

        {/* 2:4 density line (always n/2) */}
        <line
          x1={pad.l + (n / 2 + 0.5) * barW + barW * 0.35}
          y1={pad.t}
          x2={pad.l + (n / 2 + 0.5) * barW + barW * 0.35}
          y2={pad.t + ph}
          stroke={CYAN}
          strokeWidth={1.5}
          strokeDasharray="4 3"
          opacity={0.7}
        />
        <text
          x={pad.l + (n / 2 + 0.5) * barW + barW * 0.35 + 4}
          y={pad.t + 12}
          fill={CYAN}
          fontSize={8}
          fontFamily={MONO}
        >
          2:4 (always 8)
        </text>

        {/* Average density line */}
        {(() => {
          const avgK = masks.reduce((s, m) => s + m.reduce((a, b) => a + b, 0), 0) / masks.length;
          const x = pad.l + (avgK + 0.5) * barW + barW * 0.35;
          return (
            <>
              <line
                x1={x}
                y1={pad.t}
                x2={x}
                y2={pad.t + ph}
                stroke={GOLD}
                strokeWidth={1}
                strokeDasharray="3 2"
                opacity={0.5}
              />
              <text
                x={x + 4}
                y={pad.t + 24}
                fill={GOLD}
                fontSize={8}
                fontFamily={MONO}
              >
                avg={avgK.toFixed(1)}
              </text>
            </>
          );
        })()}

        <text
          x={pad.l + pw / 2}
          y={H - 0}
          fill={DIM}
          fontSize={8}
          fontFamily={MONO}
          textAnchor="middle"
        >
          active weights (k)
        </text>
      </svg>

      <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div style={{ padding: 14, borderRadius: 6, background: `${GOLD}06`, border: `1px solid ${GOLD}20` }}>
          <div style={{ fontFamily: MONO, fontSize: 10, fontWeight: 700, color: GOLD, marginBottom: 8 }}>
            ZECKENDORF DENSITY
          </div>
          <div style={{ fontSize: 11, lineHeight: 1.6, color: "rgba(255,255,255,0.5)" }}>
            Average density ≈ <strong style={{ color: GOLD }}>28.6%</strong> for n=16 (converges to 1/(1+φ) ≈ 38.2%). The
            distribution is approximately binomial, peaked near n/φ² ≈ 0.38n. Most masks keep 30-40% of weights active — more
            aggressive pruning than 2:4 by default. Maximum possible density is exactly 50% (alternating 1-0-1-0...).
          </div>
        </div>
        <div style={{ padding: 14, borderRadius: 6, background: `${CYAN}06`, border: `1px solid ${CYAN}20` }}>
          <div style={{ fontFamily: MONO, fontSize: 10, fontWeight: 700, color: CYAN, marginBottom: 8 }}>2:4 DENSITY</div>
          <div style={{ fontSize: 11, lineHeight: 1.6, color: "rgba(255,255,255,0.5)" }}>
            Always exactly <strong style={{ color: CYAN }}>50%</strong>. No variance. This is both a strength (predictable
            compute) and a limitation (no flexibility). The 2:4 constraint cannot express sparser patterns even when sparsity
            less that 50% would improve accuracy. Zeckendorf naturally supports the full range from 0% to 50%.
          </div>
        </div>
      </div>

      <div style={{ marginTop: 12, padding: 14, borderRadius: 6, background: `${GREEN}06`, border: `1px solid ${GREEN}20` }}>
        <div style={{ fontFamily: MONO, fontSize: 10, fontWeight: 700, color: GREEN, letterSpacing: 1, marginBottom: 6 }}>
          ★ THE DENSITY TRADEOFF
        </div>
        <div style={{ fontSize: 12, lineHeight: 1.7, color: "rgba(255,255,255,0.5)" }}>
          Zeckendorf is <em>sparser on average</em> than 2:4 (38% vs 50% active weights), which means more compute savings but
          potentially more accuracy loss from aggressive pruning. However, it has <em>more valid masks</em> (higher capacity),
          meaning the pruning search has more candidates to find accuracy-preserving patterns. The question is empirical: does
          the larger search space compensate for the lower average density? The DATE 2021 paper suggests yes —
          Fibonacci-encoded weights lost only 0.4% accuracy on SqueezeNet while achieving 73% area reduction.
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// TAB 4: STRUCTURAL COMPARISON
// ═══════════════════════════════════════════════════════════════

function StructureTab() {
  const [highlight, setHighlight] = useState(null);

  const rows = [
    {
      prop: "Constraint type",
      zeck: "Local (pairwise)",
      t24: "Block-global (group of 4)",
      winner: "zeck",
      why: "Local = shift-invariant, no alignment artifacts"
    },
    {
      prop: "Capacity",
      zeck: "log₂(φ) ≈ 0.694 bits/pos",
      t24: "log₂(6)/4 ≈ 0.646 bits/pos",
      winner: "zeck",
      why: "7.4% more information per position"
    },
    {
      prop: "Average density",
      zeck: "≈38.2% (variable)",
      t24: "50% (exact)",
      winner: "t24",
      why: "Higher density = fewer pruned weights"
    },
    {
      prop: "Density range",
      zeck: "0% to 50%",
      t24: "50% only",
      winner: "zeck",
      why: "Flexibility to be sparser where beneficial"
    },
    {
      prop: "Error detection",
      zeck: "'11' pattern = invalid (free)",
      t24: "Popcount ≠ 2 per group",
      winner: "zeck",
      why: "O(1) check per position vs O(4) per group"
    },
    {
      prop: "Shift invariance",
      zeck: "Yes",
      t24: "No (group-aligned)",
      winner: "zeck",
      why: "No boundary artifacts across layer width"
    },
    {
      prop: "Mask perturbation",
      zeck: "Flip 1 bit, check 2 neighbors",
      t24: "Flip 1 bit, rebalance group",
      winner: "zeck",
      why: "Smoother gradient-based mask search"
    },
    {
      prop: "Mask space topology",
      zeck: "Fibonacci cube (studied)",
      t24: "Product of K₆ graphs",
      winner: "zeck",
      why: "Known diameter, connectivity, distance properties"
    },
    {
      prop: "Hardware maturity",
      zeck: "RLL codes (1970s+)",
      t24: "NVIDIA A100 (2020+)",
      winner: "t24",
      why: "2:4 has shipping silicon with Tensor Core support"
    },
    {
      prop: "Existing ML papers",
      zeck: "DATE 2021, FCQ 2025, Z-pool",
      t24: "100+ papers, production use",
      winner: "t24",
      why: "2:4 has massive ecosystem"
    },
    {
      prop: "Theory depth",
      zeck: "60+ years, information theory",
      t24: "~5 years, empirical",
      winner: "zeck",
      why: "Golden mean shift is a classical result"
    }
  ];

  const zWins = rows.filter(r => r.winner === "zeck").length;
  const tWins = rows.filter(r => r.winner === "t24").length;

  return (
    <div>
      <div style={{ fontFamily: MONO, fontSize: 11, color: DIM, marginBottom: 16 }}>
        Head-to-head comparison — hover any row for explanation
      </div>

      <div style={{ borderRadius: 8, overflow: "hidden", border: `1px solid ${DIMMER}` }}>
        {/* Header */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "180px 1fr 1fr",
            gap: 0,
            background: "rgba(255,255,255,0.03)",
            padding: "8px 12px",
            borderBottom: `1px solid ${DIMMER}`
          }}
        >
          <div style={{ fontFamily: MONO, fontSize: 9, color: DIM, letterSpacing: 1 }}>PROPERTY</div>
          <div style={{ fontFamily: MONO, fontSize: 9, color: GOLD, letterSpacing: 1 }}>ZECKENDORF</div>
          <div style={{ fontFamily: MONO, fontSize: 9, color: CYAN, letterSpacing: 1 }}>NVIDIA 2:4</div>
        </div>

        {/* Rows */}
        {rows.map((r, i) => (
          <div
            key={i}
            onMouseEnter={() => setHighlight(i)}
            onMouseLeave={() => setHighlight(null)}
            style={{
              display: "grid",
              gridTemplateColumns: "180px 1fr 1fr",
              gap: 0,
              padding: "7px 12px",
              cursor: "pointer",
              background: highlight === i ? "rgba(255,255,255,0.04)" : "transparent",
              borderBottom: i < rows.length - 1 ? `1px solid ${DIMMER}` : "none",
              transition: "background 0.15s"
            }}
          >
            <div style={{ fontFamily: MONO, fontSize: 10, color: "rgba(255,255,255,0.5)" }}>{r.prop}</div>
            <div
              style={{
                fontFamily: MONO,
                fontSize: 10,
                color: r.winner === "zeck" ? GOLD : "rgba(255,255,255,0.4)",
                fontWeight: r.winner === "zeck" ? 700 : 400
              }}
            >
              {r.winner === "zeck" ? "★ " : ""}
              {r.zeck}
            </div>
            <div
              style={{
                fontFamily: MONO,
                fontSize: 10,
                color: r.winner === "t24" ? CYAN : "rgba(255,255,255,0.4)",
                fontWeight: r.winner === "t24" ? 700 : 400
              }}
            >
              {r.winner === "t24" ? "★ " : ""}
              {r.t24}
            </div>
          </div>
        ))}
      </div>

      {/* Explanation tooltip */}
      {highlight !== null && (
        <div
          style={{
            marginTop: 8,
            padding: 10,
            borderRadius: 6,
            background: `${rows[highlight].winner === "zeck" ? GOLD : CYAN}08`,
            border: `1px solid ${rows[highlight].winner === "zeck" ? GOLD : CYAN}25`
          }}
        >
          <span
            style={{
              fontFamily: MONO,
              fontSize: 10,
              color: rows[highlight].winner === "zeck" ? GOLD : CYAN
            }}
          >
            {rows[highlight].why}
          </span>
        </div>
      )}

      {/* Score */}
      <div style={{ display: "flex", gap: 16, marginTop: 16, justifyContent: "center" }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontFamily: MONO, fontSize: 28, fontWeight: 800, color: GOLD }}>{zWins}</div>
          <div style={{ fontFamily: MONO, fontSize: 9, color: GOLD, letterSpacing: 1 }}>ZECKENDORF</div>
        </div>
        <div style={{ fontFamily: MONO, fontSize: 28, color: DIM, lineHeight: "36px" }}>:</div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontFamily: MONO, fontSize: 28, fontWeight: 800, color: CYAN }}>{tWins}</div>
          <div style={{ fontFamily: MONO, fontSize: 9, color: CYAN, letterSpacing: 1 }}>NVIDIA 2:4</div>
        </div>
      </div>

      <div style={{ marginTop: 16, padding: 14, borderRadius: 8, background: `${GOLD}06`, border: `1px solid ${GOLD}20` }}>
        <div style={{ fontFamily: MONO, fontSize: 10, color: GOLD, fontWeight: 700, letterSpacing: 1, marginBottom: 6 }}>
          ★ ASSESSMENT
        </div>
        <div style={{ fontSize: 12, lineHeight: 1.7, color: "rgba(255,255,255,0.55)" }}>
          Zeckendorf wins on structural and theoretical properties: higher capacity, shift invariance, local constraint
          checking, and a mathematically characterized mask space (Fibonacci cube). NVIDIA 2:4 wins on{" "}
          <em>deployment reality</em>: shipping silicon with Tensor Core acceleration and a large empirical literature. The
          Zeckendorf contribution isn't "replace 2:4" — it's
          <strong style={{ color: GOLD }}>
            {" "}
            "here's a structured sparsity pattern with 60 years of information-theoretic foundation, a precise capacity bound,
            and properties that 2:4 lacks."
          </strong>{" "}
          Whether those properties translate to better pruned networks is the open empirical question.
        </div>
      </div>
    </div>
  );
}
