import { useState, useCallback, useMemo, useEffect, useRef } from "react";

const BITS = 8;
const WEIGHTS = Array.from({ length: BITS }, (_, i) => Math.pow(2, BITS - 1 - i));

// ─── Helpers ───
function toBits(n, width = BITS) {
  const b = [];
  for (let i = width - 1; i >= 0; i--) b.push((n >> i) & 1);
  return b;
}
function fromBits(bits) {
  return bits.reduce((acc, b, i) => acc + b * Math.pow(2, bits.length - 1 - i), 0);
}
function addWithCarry(a, b) {
  const steps = [];
  const aBits = [...toBits(a)];
  const bBits = [...toBits(b)];
  let result = aBits.map((v, i) => v + bBits[i]);
  steps.push({ bits: [...result], label: "Sum digits", carries: [] });
  for (let pass = 0; pass < BITS; pass++) {
    let carried = false;
    const carries = [];
    for (let i = result.length - 1; i >= 1; i--) {
      if (result[i] >= 2) {
        const overflow = Math.floor(result[i] / 2);
        result[i] = result[i] % 2;
        result[i - 1] += overflow;
        carries.push(i - 1);
        carried = true;
      }
    }
    if (!carried) break;
    steps.push({ bits: [...result], label: `Carry pass ${pass + 1}`, carries });
  }
  if (result[0] >= 2) {
    steps.push({ bits: [...result], label: "Overflow!", carries: [], overflow: true });
  }
  return steps;
}
function subtractWithBorrow(a, b) {
  if (a < b) return [{ bits: toBits(0), label: "Underflow (a < b)", carries: [], overflow: true }];
  const steps = [];
  const aBits = [...toBits(a)];
  const bBits = [...toBits(b)];
  let result = aBits.map((v, i) => v - bBits[i]);
  steps.push({ bits: [...result], label: "Subtract digits", carries: [] });
  for (let pass = 0; pass < BITS; pass++) {
    let borrowed = false;
    const carries = [];
    for (let i = result.length - 1; i >= 1; i--) {
      if (result[i] < 0) {
        result[i] += 2;
        result[i - 1] -= 1;
        carries.push(i - 1);
        borrowed = true;
      }
    }
    if (!borrowed) break;
    steps.push({ bits: [...result], label: `Borrow pass ${pass + 1}`, carries });
  }
  return steps;
}

// ─── Tree builder for to_bin(n) ───
function buildTree(n) {
  if (n === 0) return { val: 0, label: "to_bin(0)", result: 0, base: true };
  if (n === 1) return { val: 1, label: "to_bin(1)", result: 1, base: true };
  const left = buildTree(Math.floor(n / 2));
  const right = buildTree(n % 2);
  return {
    val: n,
    label: `to_bin(${n})`,
    result: left.result * 10 + right.result,
    left,
    right,
    base: false
  };
}
function treeDepth(node) {
  if (!node || node.base) return 0;
  return 1 + Math.max(treeDepth(node.left), treeDepth(node.right));
}
function layoutTree(node, x, y, spread) {
  if (!node) return [];
  const items = [{ ...node, x, y }];
  if (!node.base) {
    const childY = y + 80;
    if (node.left) items.push(...layoutTree(node.left, x - spread, childY, spread * 0.55));
    if (node.right) items.push(...layoutTree(node.right, x + spread, childY, spread * 0.55));
  }
  return items;
}

// ─── Styles ───
const fonts = `@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');`;

const C = {
  bg: "#0c0e12",
  surface: "#12151b",
  surfaceAlt: "#181c24",
  border: "#262c38",
  borderHi: "#3a4258",
  text: "#d4dae8",
  textMute: "#6b7a96",
  textDim: "#3e4d66",
  accent: "#22d68a",
  accentDim: "rgba(34,214,138,0.12)",
  accentMid: "rgba(34,214,138,0.25)",
  warm: "#f0873a",
  warmDim: "rgba(240,135,58,0.12)",
  red: "#f05454",
  redDim: "rgba(240,84,84,0.12)",
  cyan: "#38bdf8",
  cyanDim: "rgba(56,189,248,0.12)",
  purple: "#a78bfa",
  purpleDim: "rgba(167,139,250,0.12)"
};

// ─── Components ───
function BitCell({ value, weight, highlight, glow, label, onClick, dim }) {
  const active = value === 1;
  return (
    <div
      onClick={onClick}
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 4,
        cursor: onClick ? "pointer" : "default",
        userSelect: "none",
        opacity: dim ? 0.3 : 1,
        transition: "all 0.3s ease"
      }}
    >
      {label && (
        <div style={{ fontFamily: "DM Sans", fontSize: 10, color: C.textMute, letterSpacing: 1, textTransform: "uppercase" }}>
          {label}
        </div>
      )}
      <div style={{ fontFamily: "JetBrains Mono", fontSize: 11, color: C.textDim, fontWeight: 400 }}>{weight}</div>
      <div
        style={{
          width: 44,
          height: 44,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: "JetBrains Mono",
          fontSize: 20,
          fontWeight: 600,
          color: active ? C.bg : C.textDim,
          background:
            glow === "carry" ? C.warm
            : glow === "borrow" ? C.cyan
            : active ? C.accent
            : "transparent",
          border: `1px solid ${
            highlight ? C.accent
            : active ? C.accentMid
            : C.border
          }`,
          borderRadius: 6,
          transition: "all 0.25s ease",
          boxShadow:
            active ? `0 0 16px ${C.accentDim}`
            : glow === "carry" ? `0 0 16px ${C.warmDim}`
            : "none"
        }}
      >
        {typeof value === "number" ? value : "·"}
      </div>
    </div>
  );
}

function Register({ bits, weights, carries = [], label, overflow, highlight }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
      {label && (
        <div
          style={{
            fontFamily: "DM Sans",
            fontSize: 11,
            color: C.textMute,
            letterSpacing: 1.5,
            textTransform: "uppercase",
            marginBottom: 2
          }}
        >
          {label}
        </div>
      )}
      <div style={{ display: "flex", gap: 6, alignItems: "flex-start" }}>
        {bits.map((b, i) => (
          <BitCell
            key={i}
            value={b}
            weight={weights[i]}
            glow={carries.includes(i) ? "carry" : null}
            highlight={highlight === i}
            dim={overflow && i === 0 && b >= 2}
          />
        ))}
      </div>
    </div>
  );
}

function Tab({ active, label, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        fontFamily: "JetBrains Mono",
        fontSize: 11,
        fontWeight: active ? 600 : 400,
        letterSpacing: 2,
        textTransform: "uppercase",
        color: active ? C.accent : C.textMute,
        background: active ? C.accentDim : "transparent",
        border: `1px solid ${active ? C.accentMid : "transparent"}`,
        borderRadius: 6,
        padding: "8px 18px",
        cursor: "pointer",
        transition: "all 0.2s ease"
      }}
    >
      {label}
    </button>
  );
}

function NumInput({ value, onChange, max = 255, label }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      {label && <span style={{ fontFamily: "DM Sans", fontSize: 12, color: C.textMute }}>{label}</span>}
      <input
        type="number"
        min={0}
        max={max}
        value={value}
        onChange={e => onChange(Math.max(0, Math.min(max, parseInt(e.target.value) || 0)))}
        style={{
          fontFamily: "JetBrains Mono",
          fontSize: 15,
          fontWeight: 500,
          width: 72,
          padding: "6px 10px",
          textAlign: "center",
          color: C.text,
          background: C.surfaceAlt,
          border: `1px solid ${C.border}`,
          borderRadius: 6,
          outline: "none"
        }}
      />
    </div>
  );
}

function SmallBtn({ label, onClick, active, color = C.accent }) {
  return (
    <button
      onClick={onClick}
      style={{
        fontFamily: "JetBrains Mono",
        fontSize: 10,
        fontWeight: 500,
        letterSpacing: 1.5,
        textTransform: "uppercase",
        color: active ? C.bg : color,
        background: active ? color : "transparent",
        border: `1px solid ${active ? color : C.border}`,
        borderRadius: 5,
        padding: "6px 14px",
        cursor: "pointer",
        transition: "all 0.2s ease"
      }}
    >
      {label}
    </button>
  );
}

// ─── ARITHMETIC TAB ───
function ArithmeticTab() {
  const [a, setA] = useState(13);
  const [b, setB] = useState(6);
  const [op, setOp] = useState("add");
  const [step, setStep] = useState(0);

  const steps = useMemo(() => {
    return op === "add" ? addWithCarry(a, b) : subtractWithBorrow(a, b);
  }, [a, b, op]);

  useEffect(() => setStep(0), [a, b, op]);

  const current = steps[Math.min(step, steps.length - 1)];
  const isLast = step >= steps.length - 1;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 28 }}>
      <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap", justifyContent: "center" }}>
        <NumInput
          label="A"
          value={a}
          onChange={setA}
        />
        <div style={{ display: "flex", gap: 4 }}>
          <SmallBtn
            label="+"
            onClick={() => setOp("add")}
            active={op === "add"}
          />
          <SmallBtn
            label="−"
            onClick={() => setOp("sub")}
            active={op === "sub"}
          />
        </div>
        <NumInput
          label="B"
          value={b}
          onChange={setB}
        />
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 18, alignItems: "center", width: "100%" }}>
        <Register
          bits={toBits(a)}
          weights={WEIGHTS}
          label={`A = ${a}`}
        />
        <div style={{ fontFamily: "JetBrains Mono", fontSize: 18, color: C.textMute }}>{op === "add" ? "+" : "−"}</div>
        <Register
          bits={toBits(b)}
          weights={WEIGHTS}
          label={`B = ${b}`}
        />
        <div style={{ width: "80%", maxWidth: 400, height: 1, background: C.border }} />
        <Register
          bits={current.bits}
          weights={WEIGHTS}
          carries={current.carries}
          label={current.label}
          overflow={current.overflow}
        />
      </div>

      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <SmallBtn
          label="◂ Prev"
          onClick={() => setStep(s => Math.max(0, s - 1))}
          color={C.cyan}
        />
        <span style={{ fontFamily: "JetBrains Mono", fontSize: 11, color: C.textDim, minWidth: 70, textAlign: "center" }}>
          {step + 1} / {steps.length}
        </span>
        <SmallBtn
          label="Next ▸"
          onClick={() => setStep(s => Math.min(steps.length - 1, s + 1))}
          color={C.cyan}
        />
      </div>

      {isLast && !current.overflow && (
        <div
          style={{
            fontFamily: "JetBrains Mono",
            fontSize: 14,
            fontWeight: 500,
            color: C.accent,
            padding: "8px 20px",
            background: C.accentDim,
            borderRadius: 6,
            border: `1px solid ${C.accentMid}`
          }}
        >
          Result: {fromBits(current.bits)} ({current.bits.join("")})₂
        </div>
      )}
      {current.overflow && (
        <div
          style={{
            fontFamily: "JetBrains Mono",
            fontSize: 13,
            fontWeight: 500,
            color: C.red,
            padding: "8px 20px",
            background: C.redDim,
            borderRadius: 6,
            border: `1px solid rgba(240,84,84,0.25)`
          }}
        >
          {op === "sub" ?
            "Underflow — result is negative"
          : `Overflow — exceeds ${BITS}-bit register (max ${Math.pow(2, BITS) - 1})`}
        </div>
      )}
    </div>
  );
}

// ─── TREE TAB ───
function TreeTab() {
  const [n, setN] = useState(13);
  const [hovered, setHovered] = useState(null);
  const [containerH, setContainerH] = useState(380);
  const svgRef = useRef(null);
  const panState = useRef({ isPanning: false, startX: 0, startY: 0, vbX: 0, vbY: 0 });
  const resizeState = useRef({ active: false, startY: 0, startH: 0 });
  const [, forceRender] = useState(0);

  const tree = useMemo(() => buildTree(Math.min(n, 255)), [n]);
  const depth = treeDepth(tree);
  const spread = Math.max(120, 60 + depth * 60);
  const cx = Math.max(340, spread * 1.2 + 60);
  const nodes = useMemo(() => layoutTree(tree, cx, 40, spread), [tree, cx, spread]);

  const allX = nodes.map(nd => nd.x);
  const allY = nodes.map(nd => nd.y);
  const pad = 80;
  const contentMinX = Math.min(...allX) - pad;
  const contentMaxX = Math.max(...allX) + pad;
  const contentMaxY = Math.max(...allY) + 50;
  const contentW = contentMaxX - Math.min(0, contentMinX);
  const contentH = contentMaxY + 20;

  // viewBox state stored in ref for perf (no re-render per mousemove)
  const vb = useRef({ x: 0, y: 0, w: contentW, h: contentH });

  // Reset viewBox when n changes
  useEffect(() => {
    vb.current = { x: 0, y: 0, w: contentW, h: contentH };
    forceRender(c => c + 1);
  }, [n, contentW, contentH]);

  const applyViewBox = useCallback(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const v = vb.current;
    svg.setAttribute("viewBox", `${v.x} ${v.y} ${v.w} ${v.h}`);
  }, []);

  // Zoom via buttons
  const doZoom = useCallback(
    factor => {
      const v = vb.current;
      const cxMid = v.x + v.w / 2;
      const cyMid = v.y + v.h / 2;
      const nw = Math.max(100, Math.min(contentW * 4, v.w * factor));
      const nh = Math.max(60, Math.min(contentH * 4, v.h * factor));
      vb.current = { x: cxMid - nw / 2, y: cyMid - nh / 2, w: nw, h: nh };
      applyViewBox();
      forceRender(c => c + 1);
    },
    [contentW, contentH, applyViewBox]
  );

  const resetView = useCallback(() => {
    vb.current = { x: 0, y: 0, w: contentW, h: contentH };
    applyViewBox();
    forceRender(c => c + 1);
  }, [contentW, contentH, applyViewBox]);

  // Mouse pan on SVG
  const onSvgMouseDown = useCallback(e => {
    if (e.button !== 0) return;
    e.preventDefault();
    panState.current = { isPanning: true, startX: e.clientX, startY: e.clientY, vbX: vb.current.x, vbY: vb.current.y };
  }, []);

  useEffect(() => {
    const onMove = e => {
      if (!panState.current.isPanning) return;
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const scaleX = vb.current.w / rect.width;
      const scaleY = vb.current.h / rect.height;
      const dx = (e.clientX - panState.current.startX) * scaleX;
      const dy = (e.clientY - panState.current.startY) * scaleY;
      vb.current.x = panState.current.vbX - dx;
      vb.current.y = panState.current.vbY - dy;
      applyViewBox();
    };
    const onUp = () => {
      panState.current.isPanning = false;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [applyViewBox]);

  // Wheel zoom centered on cursor
  const onWheel = useCallback(
    e => {
      e.preventDefault();
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const mx = ((e.clientX - rect.left) / rect.width) * vb.current.w + vb.current.x;
      const my = ((e.clientY - rect.top) / rect.height) * vb.current.h + vb.current.y;
      const factor = e.deltaY > 0 ? 1.12 : 0.89;
      const nw = Math.max(100, Math.min(contentW * 4, vb.current.w * factor));
      const nh = Math.max(60, Math.min(contentH * 4, vb.current.h * factor));
      vb.current = {
        x: mx - (mx - vb.current.x) * (nw / vb.current.w),
        y: my - (my - vb.current.y) * (nh / vb.current.h),
        w: nw,
        h: nh
      };
      applyViewBox();
      forceRender(c => c + 1);
    },
    [contentW, contentH, applyViewBox]
  );

  // Resize handle
  const onResizeDown = useCallback(
    e => {
      e.preventDefault();
      resizeState.current = { active: true, startY: e.clientY, startH: containerH };
      const move = ev => {
        if (!resizeState.current.active) return;
        setContainerH(Math.max(160, resizeState.current.startH + (ev.clientY - resizeState.current.startY)));
      };
      const up = () => {
        resizeState.current.active = false;
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    },
    [containerH]
  );

  const zoomPct = useMemo(() => {
    if (!contentW) return 100;
    return Math.round((contentW / vb.current.w) * 100);
  }, [contentW, forceRender]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
      <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap", justifyContent: "center" }}>
        <NumInput
          label="n"
          value={n}
          onChange={setN}
        />
        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
          <SmallBtn
            label="−"
            onClick={() => doZoom(1.3)}
            color={C.cyan}
          />
          <span style={{ fontFamily: "JetBrains Mono", fontSize: 10, color: C.textDim, minWidth: 40, textAlign: "center" }}>
            {zoomPct}%
          </span>
          <SmallBtn
            label="+"
            onClick={() => doZoom(0.7)}
            color={C.cyan}
          />
          <SmallBtn
            label="Fit"
            onClick={resetView}
            color={C.textMute}
          />
        </div>
      </div>

      {/* SVG viewport — viewBox does all pan/zoom, no clipping */}
      <div
        style={{
          width: "100%",
          height: containerH,
          background: C.bg,
          borderRadius: 8,
          border: `1px solid ${C.border}`,
          overflow: "hidden",
          position: "relative"
        }}
      >
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          viewBox={`${vb.current.x} ${vb.current.y} ${vb.current.w} ${vb.current.h}`}
          preserveAspectRatio="xMidYMid meet"
          onMouseDown={onSvgMouseDown}
          onWheel={onWheel}
          style={{ cursor: panState.current.isPanning ? "grabbing" : "grab", display: "block" }}
        >
          {/* Edges */}
          {nodes.map((nd, i) => {
            if (nd.base && nd === tree) return null;
            return nodes.map((parent, j) => {
              if (parent.base) return null;
              const isLeft = parent.left && parent.left.val === nd.val && parent.left.result === nd.result;
              const isRight = parent.right && parent.right.val === nd.val && parent.right.result === nd.result;
              if (!isLeft && !isRight) return null;
              if (Math.abs(parent.y + 80 - nd.y) > 1) return null;
              return (
                <line
                  key={`e-${j}-${i}`}
                  x1={parent.x}
                  y1={parent.y + 20}
                  x2={nd.x}
                  y2={nd.y - 20}
                  stroke={C.border}
                  strokeWidth={1}
                />
              );
            });
          })}

          {/* Nodes */}
          {nodes.map((nd, i) => {
            const isHov = hovered === i;
            const isBase = nd.base;
            const fill = isBase ? C.surfaceAlt : C.surface;
            const stroke = isBase ? C.warm : C.accentMid;
            const textCol = isBase ? C.warm : C.text;
            const w = Math.max(90, nd.label.length * 9 + 24);
            return (
              <g
                key={i}
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered(null)}
                style={{ cursor: "pointer" }}
              >
                <rect
                  x={nd.x - w / 2}
                  y={nd.y - 18}
                  width={w}
                  height={36}
                  rx={6}
                  fill={isHov ? C.surfaceAlt : fill}
                  stroke={isHov ? C.accent : stroke}
                  strokeWidth={isHov ? 1.5 : 0.8}
                />
                <text
                  x={nd.x}
                  y={nd.y + 1}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontFamily="JetBrains Mono"
                  fontSize={11}
                  fontWeight={isBase ? 600 : 400}
                  fill={textCol}
                >
                  {nd.label}
                  {isBase ? ` → ${nd.result}` : ""}
                </text>
                {!isBase && isHov && (
                  <text
                    x={nd.x}
                    y={nd.y + 28}
                    textAnchor="middle"
                    fontFamily="JetBrains Mono"
                    fontSize={10}
                    fill={C.accent}
                  >
                    = {nd.result}
                  </text>
                )}
                {!isBase && (
                  <>
                    <text
                      x={
                        nd.x -
                        (nd.x - (nodes.find(c => c.val === Math.floor(nd.val / 2) && c.y === nd.y + 80)?.x || nd.x)) / 2
                      }
                      y={nd.y + 30}
                      textAnchor="middle"
                      fontFamily="JetBrains Mono"
                      fontSize={9}
                      fill={C.textDim}
                    >
                      n//2
                    </text>
                    <text
                      x={nd.x + ((nodes.find(c => c.val === nd.val % 2 && c.y === nd.y + 80)?.x || nd.x) - nd.x) / 2}
                      y={nd.y + 30}
                      textAnchor="middle"
                      fontFamily="JetBrains Mono"
                      fontSize={9}
                      fill={C.textDim}
                    >
                      n%2
                    </text>
                  </>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      {/* Resize handle */}
      <div
        onMouseDown={onResizeDown}
        style={{
          width: 64,
          height: 6,
          borderRadius: 3,
          marginTop: -10,
          background: C.borderHi,
          cursor: "ns-resize",
          transition: "background 0.2s"
        }}
        onMouseEnter={e => (e.currentTarget.style.background = C.accent)}
        onMouseLeave={e => (e.currentTarget.style.background = C.borderHi)}
      />

      <div
        style={{
          fontFamily: "JetBrains Mono",
          fontSize: 13,
          color: C.textMute,
          padding: "8px 16px",
          background: C.surfaceAlt,
          borderRadius: 6,
          border: `1px solid ${C.border}`
        }}
      >
        to_bin({n}) = <span style={{ color: C.accent, fontWeight: 600 }}>{tree.result}</span>
        <span style={{ color: C.textDim, marginLeft: 8 }}>
          ({depth + 1} levels, chain depth {depth})
        </span>
      </div>

      <div style={{ fontFamily: "JetBrains Mono", fontSize: 9, color: C.textDim, textAlign: "center" }}>
        Drag to pan · scroll to zoom · drag handle to resize
      </div>
    </div>
  );
}

// ─── PROPERTIES TAB ───
function PropertiesTab() {
  const [n, setN] = useState(42);
  const bits = toBits(n);
  const setBits = bits.reduce((a, b) => a + b, 0);
  const maxVal = Math.pow(2, BITS) - 1;
  const density = setBits / BITS;
  const minBits = n === 0 ? 1 : Math.floor(Math.log2(n)) + 1;

  // Detect runs (groups of consecutive 1s or 0s)
  const runs = [];
  let current = bits[0],
    count = 1;
  for (let i = 1; i < bits.length; i++) {
    if (bits[i] === current) count++;
    else {
      runs.push({ val: current, len: count });
      current = bits[i];
      count = 1;
    }
  }
  runs.push({ val: current, len: count });

  // Two's complement
  const inverted = bits.map(b => (b === 0 ? 1 : 0));
  const twoComp = fromBits(inverted) + 1;

  const props = [
    {
      title: "Unique representation",
      desc: "Every integer 0–255 maps to exactly one 8-bit pattern. No redundancy — unlike Zeckendorf, which must forbid adjacent 1s to achieve uniqueness.",
      color: C.accent,
      bg: C.accentDim,
      detail: `${n} → ${bits.join("")}  (the only encoding)`
    },
    {
      title: "Hard overflow",
      desc: `At ${BITS} bits the ceiling is ${maxVal}. Adding 1 to ${maxVal} wraps to 0 — a cliff, not a graceful degrade. Fibonacci registers overflow gently through the Zeckendorf theorem.`,
      color: C.red,
      bg: C.redDim,
      detail: n > maxVal ? "Overflow!" : `${n} uses ${minBits} of ${BITS} bits — ${maxVal - n} from overflow`
    },
    {
      title: "Bit density",
      desc: "Popcount / register width. Binary has no structural constraint on density — any pattern of 1s and 0s is valid.",
      color: C.cyan,
      bg: C.cyanDim,
      detail: `${setBits} set bits / ${BITS} = ${(density * 100).toFixed(0)}% density`
    },
    {
      title: "Run-length structure",
      desc: "Consecutive identical bits form runs. Run patterns reveal nothing self-similar — unlike Fibonacci where adjacent-1 prohibition creates fractal-like gaps.",
      color: C.purple,
      bg: C.purpleDim,
      detail: runs.map(r => `${r.val}×${r.len}`).join(" ")
    },
    {
      title: "Two's complement",
      desc: "Negation via invert-and-add-1. A mechanical trick with no analog in Fibonacci encoding — binary's linearity makes it work.",
      color: C.warm,
      bg: C.warmDim,
      detail: `−${n} = ~${bits.join("")} + 1 = ${toBits(twoComp).join("")} (${twoComp})`
    }
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 24 }}>
      <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap", justifyContent: "center" }}>
        <NumInput
          label="n"
          value={n}
          onChange={setN}
        />
        <div style={{ display: "flex", gap: 6 }}>
          <Register
            bits={bits}
            weights={WEIGHTS}
          />
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12, width: "100%", maxWidth: 560 }}>
        {props.map((p, i) => (
          <div
            key={i}
            style={{
              padding: "14px 18px",
              borderRadius: 8,
              background: C.surface,
              border: `1px solid ${C.border}`,
              transition: "border-color 0.2s"
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: p.color, flexShrink: 0 }} />
              <span
                style={{ fontFamily: "JetBrains Mono", fontSize: 12, fontWeight: 600, color: p.color, letterSpacing: 0.5 }}
              >
                {p.title}
              </span>
            </div>
            <div style={{ fontFamily: "DM Sans", fontSize: 12, color: C.textMute, lineHeight: 1.5, marginBottom: 8 }}>
              {p.desc}
            </div>
            <div
              style={{
                fontFamily: "JetBrains Mono",
                fontSize: 11,
                color: p.color,
                padding: "5px 10px",
                background: p.bg,
                borderRadius: 4,
                display: "inline-block"
              }}
            >
              {p.detail}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Main ───
export default function BinaryRegister() {
  const [tab, setTab] = useState("arithmetic");

  return (
    <>
      <style>
        {fonts}
        {`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        input::-webkit-outer-spin-button,
        input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
        input[type=number] { -moz-appearance: textfield; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 2px; }
      `}
      </style>
      <div
        style={{
          minHeight: "100vh",
          background: C.bg,
          color: C.text,
          fontFamily: "DM Sans, sans-serif",
          padding: "32px 16px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center"
        }}
      >
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <div
            style={{
              fontFamily: "JetBrains Mono",
              fontSize: 10,
              letterSpacing: 4,
              color: C.textDim,
              textTransform: "uppercase",
              marginBottom: 8
            }}
          >
            2-Register
          </div>
          <h1
            style={{
              fontFamily: "JetBrains Mono",
              fontSize: 20,
              fontWeight: 700,
              letterSpacing: -0.5,
              color: C.text,
              margin: 0
            }}
          >
            Power-of-2 Weighted Arithmetic Machine
          </h1>
          <div
            style={{
              fontFamily: "JetBrains Mono",
              fontSize: 11,
              color: C.textDim,
              marginTop: 8,
              display: "flex",
              gap: 16,
              justifyContent: "center",
              flexWrap: "wrap"
            }}
          >
            <span>P(n) = 2 · P(n−1)</span>
            <span style={{ color: C.border }}>│</span>
            <span>weights: {WEIGHTS.join(", ")}</span>
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 6, marginBottom: 28 }}>
          <Tab
            label="Arithmetic"
            active={tab === "arithmetic"}
            onClick={() => setTab("arithmetic")}
          />
          <Tab
            label="Tree"
            active={tab === "tree"}
            onClick={() => setTab("tree")}
          />
          <Tab
            label="Properties"
            active={tab === "properties"}
            onClick={() => setTab("properties")}
          />
        </div>

        {/* Content */}
        <div
          style={{
            width: "100%",
            maxWidth: 620,
            padding: "28px 20px",
            background: C.surface,
            border: `1px solid ${C.border}`,
            borderRadius: 12
          }}
        >
          {tab === "arithmetic" && <ArithmeticTab />}
          {tab === "tree" && <TreeTab />}
          {tab === "properties" && <PropertiesTab />}
        </div>

        {/* Footer */}
        <div
          style={{
            fontFamily: "JetBrains Mono",
            fontSize: 10,
            color: C.textDim,
            marginTop: 24,
            textAlign: "center",
            letterSpacing: 1
          }}
        >
          Binary counterpart to φ-Register · single recurrence P(n) = 2·P(n−1) · chain, not tree
        </div>
      </div>
    </>
  );
}
