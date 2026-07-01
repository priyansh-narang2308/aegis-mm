import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import { useAegisBackend } from "../hooks/useAegisBackend";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  Shield,
  Eye,
  Mic2,
  Brain,
  Link2,
  Cpu,
  Gem,
  Zap,
  Database,
  ChevronLeft,
  ChevronRight,
  Radio,
  Play,
  RotateCcw,
  Layers,
  Waves,
  Grid3x3,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  ZAxis,
  CartesianGrid,
  Legend,
} from "recharts";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Aegis-MM Control Room · Live Proctoring HUD" },
      {
        name: "description",
        content:
          "Real-time multimodal AI interview guardrail. Live vision, audio, PEFT/LoRA, late-fusion and quantization telemetry.",
      },
    ],
  }),
  component: AegisApp,
});

// ---------------- Nav model ----------------
type ViewId =
  "control" | "vision" | "audio" | "peft" | "fusion" | "engine" | "quant" | "bench" | "data";

const NAV: {
  id: ViewId;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  hint: string;
}[] = [
  { id: "control", label: "Live Control Room", icon: Shield, hint: "Master Proctoring HUD" },
  { id: "vision", label: "Vision Inspector", icon: Eye, hint: "src/vision" },
  { id: "audio", label: "Audio Spectral Studio", icon: Mic2, hint: "src/audio" },
  { id: "peft", label: "PEFT · AI Fluency", icon: Brain, hint: "src/peft" },
  { id: "fusion", label: "Multimodal Fusion", icon: Link2, hint: "src/fusion" },
  { id: "engine", label: "Streaming Engine", icon: Cpu, hint: "src/engine" },
  { id: "quant", label: "Quantization Deck", icon: Gem, hint: "src/optimization" },
  { id: "bench", label: "SLA Stress Harness", icon: Zap, hint: "src/benchmarks" },
  { id: "data", label: "Benchmark Datasets", icon: Database, hint: "src/data" },
];

// ---------------- Simulated telemetry ----------------
type Anomaly = "normal" | "gaze" | "tts" | "prompt";
type Packet = {
  t: number;
  risk: number;
  fluency: number;
  lipsync: number;
  confidence: number;
  gazeDrift: number;
  spoof: number;
  keystroke: number;
  pitch: number;
  yaw: number;
};

function useTelemetry(anomaly: Anomaly) {
  const [buf, setBuf] = useState<Packet[]>([]);
  const tRef = useRef(0);
  useEffect(() => {
    const id = setInterval(() => {
      tRef.current += 1;
      const t = tRef.current;
      const base = {
        gaze: {
          risk: 78,
          fluency: 42,
          lipsync: 22,
          confidence: 38,
          gazeDrift: 0.82,
          spoof: 0.08,
          keystroke: 0.3,
          pitch: 18,
          yaw: 34,
        },
        tts: {
          risk: 84,
          fluency: 55,
          lipsync: 71,
          confidence: 30,
          gazeDrift: 0.15,
          spoof: 0.91,
          keystroke: 0.25,
          pitch: 4,
          yaw: 6,
        },
        prompt: {
          risk: 69,
          fluency: 18,
          lipsync: 12,
          confidence: 44,
          gazeDrift: 0.1,
          spoof: 0.05,
          keystroke: 0.94,
          pitch: 3,
          yaw: 5,
        },
        normal: {
          risk: 12,
          fluency: 78,
          lipsync: 6,
          confidence: 92,
          gazeDrift: 0.08,
          spoof: 0.04,
          keystroke: 0.35,
          pitch: 2,
          yaw: 4,
        },
      }[anomaly];
      const jitter = (a: number, s = 4) => a + (Math.random() - 0.5) * s;
      const p: Packet = {
        t,
        risk: Math.max(0, Math.min(100, jitter(base.risk, 6))),
        fluency: Math.max(0, Math.min(100, jitter(base.fluency, 5))),
        lipsync: Math.max(0, Math.min(100, jitter(base.lipsync, 5))),
        confidence: Math.max(0, Math.min(100, jitter(base.confidence, 4))),
        gazeDrift: Math.max(0, Math.min(1, jitter(base.gazeDrift, 0.05))),
        spoof: Math.max(0, Math.min(1, jitter(base.spoof, 0.04))),
        keystroke: Math.max(0, Math.min(1, jitter(base.keystroke, 0.05))),
        pitch: jitter(base.pitch, 3),
        yaw: jitter(base.yaw, 4),
      };
      setBuf((b) => [...b.slice(-59), p]);
    }, 500);
    return () => clearInterval(id);
  }, [anomaly]);
  return buf;
}

// ---------------- Primitives ----------------
function Card({
  children,
  className = "",
  title,
  subtitle,
  tag,
}: {
  children: React.ReactNode;
  className?: string;
  title?: string;
  subtitle?: string;
  tag?: string;
}) {
  return (
    <div className={`glass rounded-xl p-5 ${className}`}>
      {(title || tag) && (
        <div className="flex items-start justify-between mb-4">
          <div>
            {title && <h3 className="text-sm font-semibold tracking-tight">{title}</h3>}
            {subtitle && <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>}
          </div>
          {tag && (
            <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-secondary text-primary border border-border">
              {tag}
            </span>
          )}
        </div>
      )}
      {children}
    </div>
  );
}

function Radial({
  value,
  label,
  color = "var(--color-cyan)",
  suffix = "%",
}: {
  value: number;
  label: string;
  color?: string;
  suffix?: string;
}) {
  const r = 42;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-28 h-28">
        <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
          <circle cx="50" cy="50" r={r} fill="none" stroke="var(--color-border)" strokeWidth="6" />
          <motion.circle
            cx="50"
            cy="50"
            r={r}
            fill="none"
            stroke={color}
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={c}
            initial={false}
            animate={{ strokeDashoffset: c - (c * pct) / 100 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="font-mono text-xl font-semibold">
            {pct.toFixed(0)}
            <span className="text-xs text-muted-foreground">{suffix}</span>
          </span>
        </div>
      </div>
      <span className="text-[11px] text-muted-foreground uppercase tracking-wider text-center">
        {label}
      </span>
    </div>
  );
}

// ---------------- Header ----------------
function Header({
  precision,
  setPrecision,
  wsConnected,
  latencyP95,
  throughput,
}: {
  precision: string;
  setPrecision: (v: string) => void;
  wsConnected: boolean;
  latencyP95: number;
  throughput: number;
}) {
  return (
    <header className="h-14 border-b border-border/70 flex items-center px-4 gap-4 sticky top-0 z-40 backdrop-blur bg-background/70">
      <div className="flex items-center gap-2">
        <div
          className="w-7 h-7 rounded-md bg-linear-to-br from-cyan to-magenta grid place-items-center"
          style={{ background: "linear-gradient(135deg, var(--color-cyan), var(--color-magenta))" }}
        >
          <Shield className="w-4 h-4 text-background" />
        </div>
        <div className="leading-tight">
          <p className="text-sm font-semibold">Aegis-MM</p>
        </div>
      </div>

      <div className="ml-6 flex items-center gap-2 px-3 h-8 rounded-full glass">
        <span
          className={`w-2 h-2 rounded-full pulse-dot ${wsConnected ? "bg-(--color-lime)" : "bg-destructive"}`}
        />
        <span className="text-[11px] font-mono">
          {wsConnected ? "ws://localhost:8000/ws/stream · CONNECTED" : "DISCONNECTED"}
        </span>
      </div>

      <div className="ml-auto flex items-center gap-3">
        <div className="flex items-center gap-2 px-3 h-8 rounded-full glass font-mono text-[11px]">
          <Activity className="w-3.5 h-3.5 text-[color:var(--color-cyan)]" />
          p95: <span className="text-[color:var(--color-cyan)]">{latencyP95.toFixed(2)} ms</span>
          <span className="text-border">|</span>
          <span>{throughput} req/s</span>
        </div>

        <label className="relative">
          <select
            value={precision}
            onChange={(e) => setPrecision(e.target.value)}
            className="appearance-none glass rounded-full h-8 pl-3 pr-8 text-[11px] font-mono cursor-pointer outline-none"
          >
            <option value="fp32">FP32 · Baseline</option>
            <option value="int8">INT8 · 4× · 0.35 ms</option>
            <option value="int4">INT4 · 8× · 0.34 ms</option>
          </select>
          <ChevronRight className="w-3 h-3 absolute right-2 top-1/2 -translate-y-1/2 rotate-90 pointer-events-none text-muted-foreground" />
        </label>
      </div>
    </header>
  );
}

// ---------------- Sidebar ----------------
function Sidebar({
  view,
  setView,
  collapsed,
  setCollapsed,
}: {
  view: ViewId;
  setView: (v: ViewId) => void;
  collapsed: boolean;
  setCollapsed: (v: boolean) => void;
}) {
  return (
    <aside
      className={`shrink-0 border-r border-border/70 transition-[width] duration-300 ${collapsed ? "w-16" : "w-64"} sticky top-14 h-[calc(100vh-3.5rem)]`}
    >
      <div className="p-2 flex flex-col h-full">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="ml-auto mb-2 h-8 w-8 grid place-items-center rounded-md hover:bg-secondary text-muted-foreground"
          aria-label="Toggle sidebar"
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
        <nav className="flex-1 space-y-1">
          {NAV.map((n) => {
            const active = view === n.id;
            const Icon = n.icon;
            return (
              <button
                key={n.id}
                onClick={() => setView(n.id)}
                title={collapsed ? n.label : undefined}
                className={`group w-full flex items-center gap-3 px-3 h-10 rounded-lg text-left transition-colors relative ${
                  active
                    ? "bg-secondary text-foreground"
                    : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
                }`}
              >
                {active && (
                  <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-r bg-(--color-cyan)" />
                )}
                <Icon className={`w-4 h-4 shrink-0 ${active ? "text-(--color-cyan)" : ""}`} />
                {!collapsed && (
                  <span className="flex-1 min-w-0">
                    <span className="block text-sm truncate">{n.label}</span>
                    <span className="block text-[10px] font-mono text-muted-foreground truncate">
                      {n.hint}
                    </span>
                  </span>
                )}
              </button>
            );
          })}
        </nav>
        {!collapsed && (
          <div className="mt-2 p-3 rounded-lg glass">
            <p className="text-[10px] font-mono text-muted-foreground">SESSION</p>
            <p className="text-xs font-mono mt-1 text-[color:var(--color-cyan)]">
              candidate_sherlock_001
            </p>
          </div>
        )}
      </div>
    </aside>
  );
}

// ---------------- View: Control Room ----------------
function ControlRoom({
  anomaly,
  setAnomaly,
  buf,
}: {
  anomaly: Anomaly;
  setAnomaly: (a: Anomaly) => void;
  buf: Packet[];
}) {
  const last = buf[buf.length - 1];
  const injectors: { id: Anomaly; label: string; icon: React.ReactNode; tone: string }[] = [
    {
      id: "normal",
      label: "Normal Stream",
      icon: <RotateCcw className="w-3.5 h-3.5" />,
      tone: "text-[color:var(--color-lime)]",
    },
    {
      id: "gaze",
      label: "Inject Gaze Drift",
      icon: <Eye className="w-3.5 h-3.5" />,
      tone: "text-[color:var(--color-amber)]",
    },
    {
      id: "tts",
      label: "TTS Deepfake Audio",
      icon: <Mic2 className="w-3.5 h-3.5" />,
      tone: "text-[color:var(--color-magenta)]",
    },
    {
      id: "prompt",
      label: "Prompt Injection Burst",
      icon: <Zap className="w-3.5 h-3.5" />,
      tone: "text-destructive",
    },
  ];

  return (
    <div className="grid grid-cols-12 gap-4">
      <Card
        className="col-span-12 lg:col-span-8"
        title="Candidate Visualizer Feed"
        subtitle="Live 30 FPS · head pose · gaze vector · landmarks"
        tag="HUD"
      >
        <div className="relative aspect-video rounded-lg overflow-hidden bg-linear-to-br from-(--color-surface-2) to-background border border-border scan-line">
          {/* Simulated silhouette */}
          <svg viewBox="0 0 640 360" className="absolute inset-0 w-full h-full">
            <defs>
              <radialGradient id="g" cx="50%" cy="45%" r="45%">
                <stop offset="0%" stopColor="oklch(0.42 0.05 260)" />
                <stop offset="100%" stopColor="transparent" />
              </radialGradient>
            </defs>
            <rect width="640" height="360" fill="url(#g)" />
            {/* Head ellipse */}
            <ellipse
              cx="320"
              cy="170"
              rx="72"
              ry="92"
              fill="oklch(0.28 0.03 260)"
              stroke="var(--color-cyan)"
              strokeOpacity="0.5"
            />
            {/* Landmarks */}
            {[
              [290, 150],
              [350, 150],
              [320, 180],
              [300, 210],
              [340, 210],
              [320, 220],
            ].map(([x, y], i) => (
              <circle key={i} cx={x} cy={y} r="2" fill="var(--color-cyan)" />
            ))}
            {/* Gaze vector */}
            <line
              x1="320"
              y1="165"
              x2={320 + (last?.yaw ?? 4) * 3}
              y2={165 + (last?.pitch ?? 2) * 3}
              stroke="var(--color-lime)"
              strokeWidth="2"
              strokeLinecap="round"
            />
            {/* Bounding box */}
            <rect
              x="240"
              y="70"
              width="160"
              height="200"
              fill="none"
              stroke="var(--color-cyan)"
              strokeWidth="1.5"
              strokeDasharray="6 4"
            />
            <text x="242" y="64" fill="var(--color-cyan)" fontSize="10" fontFamily="JetBrains Mono">
              FACE · conf 0.98
            </text>
          </svg>

          <div className="absolute top-3 left-3 flex items-center gap-2 text-[10px] font-mono">
            <span className="w-2 h-2 rounded-full bg-destructive pulse-dot" />
            <span>REC · candidate_sherlock_001</span>
          </div>
          <div className="absolute bottom-3 right-3 font-mono text-[10px] text-muted-foreground grid gap-0.5 text-right">
            <span>
              pitch {last?.pitch.toFixed(1) ?? "--"}° yaw {last?.yaw.toFixed(1) ?? "--"}°
            </span>
            <span>gaze_drift {(last?.gazeDrift ?? 0).toFixed(2)}</span>
          </div>
        </div>

        <div className="mt-4">
          <p className="text-[10px] font-mono text-muted-foreground mb-2">
            RED TEAM · ANOMALY INJECTOR
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {injectors.map((i) => {
              const active = anomaly === i.id;
              return (
                <button
                  key={i.id}
                  onClick={() => setAnomaly(i.id)}
                  className={`flex items-center gap-2 px-3 h-10 rounded-lg border transition-all text-xs font-medium ${
                    active
                      ? "border-(--color-cyan) bg-secondary glow-cyan"
                      : "border-border glass hover:border-muted-foreground"
                  }`}
                >
                  <span className={i.tone}>{i.icon}</span>
                  <span className="truncate">{i.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </Card>

      <Card
        className="col-span-12 lg:col-span-4"
        title="Risk Gauges"
        subtitle="Fused telemetry heads"
        tag="LIVE"
      >
        <div className="grid grid-cols-2 gap-4">
          <Radial value={last?.risk ?? 0} label="Cheating Risk" color="var(--color-magenta)" />
          <Radial value={last?.fluency ?? 0} label="AI Fluency Index" color="var(--color-cyan)" />
          <Radial value={last?.lipsync ?? 0} label="Lip-Sync Mismatch" color="var(--color-amber)" />
          <Radial
            value={last?.confidence ?? 0}
            label="Session Confidence"
            color="var(--color-lime)"
          />
        </div>
      </Card>

      <Card
        className="col-span-12"
        title="Rolling 30s Telemetry"
        subtitle="Risk · Fluency · Lip-Sync · Confidence"
        tag="STREAM"
      >
        <div className="h-64">
          <ResponsiveContainer>
            <LineChart data={buf} margin={{ top: 5, right: 10, bottom: 0, left: -20 }}>
              <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" opacity={0.4} />
              <XAxis
                dataKey="t"
                tick={{
                  fill: "var(--color-muted-foreground)",
                  fontSize: 10,
                  fontFamily: "JetBrains Mono",
                }}
              />
              <YAxis
                tick={{
                  fill: "var(--color-muted-foreground)",
                  fontSize: 10,
                  fontFamily: "JetBrains Mono",
                }}
                domain={[0, 100]}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--color-surface-2)",
                  border: "1px solid var(--color-border)",
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line
                type="monotone"
                dataKey="risk"
                stroke="var(--color-magenta)"
                dot={false}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="fluency"
                stroke="var(--color-cyan)"
                dot={false}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="lipsync"
                stroke="var(--color-amber)"
                dot={false}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="confidence"
                stroke="var(--color-lime)"
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
}

// ---------------- View: Vision ----------------
function VisionView({ buf }: { buf: Packet[] }) {
  const last = buf[buf.length - 1];
  return (
    <div className="grid grid-cols-12 gap-4">
      <Card
        className="col-span-12 lg:col-span-7"
        title="Spatial Attention Heatmap"
        subtitle="Input frame vs multi-head attention activation"
        tag="VISION"
      >
        <div className="grid grid-cols-2 gap-3">
          <div className="aspect-square rounded-lg bg-(--color-surface-2) border border-border grid place-items-center">
            <Eye className="w-12 h-12 text-muted-foreground" />
          </div>
          <div className="aspect-square rounded-lg border border-border relative overflow-hidden">
            <div
              className="absolute inset-0"
              style={{
                background: `radial-gradient(circle at 55% 40%, oklch(0.7 0.25 25 / .9), oklch(0.7 0.22 75 / .5) 30%, oklch(0.5 0.2 200 / .3) 55%, transparent 75%)`,
              }}
            />
            <div
              className="absolute inset-0 mix-blend-overlay"
              style={{
                background:
                  "repeating-linear-gradient(45deg, transparent 0 6px, oklch(1 0 0 / .04) 6px 7px)",
              }}
            />
            <span className="absolute bottom-2 left-2 font-mono text-[10px] text-primary-foreground bg-black/40 px-1.5 py-0.5 rounded">
              attn_head_3
            </span>
          </div>
        </div>
      </Card>
      <Card
        className="col-span-12 lg:col-span-5"
        title="Gaze Euler Angles"
        subtitle="Pitch / Yaw saccade telemetry"
        tag="EULER"
      >
        <div className="h-56">
          <ResponsiveContainer>
            <LineChart data={buf}>
              <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" opacity={0.4} />
              <XAxis dataKey="t" tick={{ fill: "var(--color-muted-foreground)", fontSize: 10 }} />
              <YAxis tick={{ fill: "var(--color-muted-foreground)", fontSize: 10 }} />
              <Tooltip
                contentStyle={{
                  background: "var(--color-surface-2)",
                  border: "1px solid var(--color-border)",
                  fontSize: 12,
                }}
              />
              <Line type="monotone" dataKey="pitch" stroke="var(--color-cyan)" dot={false} />
              <Line type="monotone" dataKey="yaw" stroke="var(--color-magenta)" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Card className="col-span-12" title="Pipeline Stage Metrics" tag="ResNet+MobileNet">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { k: "Backbone", v: "0.46 ms" },
            { k: "Attention", v: "0.12 ms" },
            { k: "Gaze Head", v: "0.08 ms" },
            { k: "Anomaly σ", v: (last?.gazeDrift ?? 0).toFixed(3) },
          ].map((m) => (
            <div key={m.k} className="p-3 rounded-lg bg-secondary/50">
              <p className="text-[10px] font-mono text-muted-foreground">{m.k}</p>
              <p className="font-mono text-lg mt-1 text-[color:var(--color-cyan)]">{m.v}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ---------------- View: Audio ----------------
function AudioView({ buf }: { buf: Packet[] }) {
  const spectrogram = useMemo(() => {
    const rows = 32;
    const cols = 60;
    return Array.from({ length: rows }, (_, r) =>
      Array.from({ length: cols }, (_, c) => {
        const base = Math.sin((r / rows) * 3.14 + c / 6) * 0.4 + 0.5;
        return Math.max(0, Math.min(1, base + (Math.random() - 0.5) * 0.3));
      }),
    );
  }, [buf.length]);
  const last = buf[buf.length - 1];
  return (
    <div className="grid grid-cols-12 gap-4">
      <Card
        className="col-span-12 lg:col-span-8"
        title="Log-Mel Spectrogram · 64ch"
        subtitle="16 kHz waterfall · time → frequency"
        tag="AUDIO"
      >
        <div className="rounded-lg overflow-hidden border border-border bg-black">
          <div className="grid" style={{ gridTemplateRows: `repeat(32, 1fr)` }}>
            {spectrogram.map((row, ri) => (
              <div key={ri} className="flex h-2">
                {row.map((v, ci) => (
                  <div
                    key={ci}
                    className="flex-1"
                    style={{
                      background: `oklch(${0.2 + v * 0.55} ${0.05 + v * 0.2} ${210 + v * 100})`,
                    }}
                  />
                ))}
              </div>
            ))}
          </div>
        </div>
      </Card>
      <Card
        className="col-span-12 lg:col-span-4"
        title="Spectral Anomaly Detector"
        subtitle="Vocoder harmonics · ASVspoof"
        tag="SPOOF"
      >
        <div className="flex flex-col items-center gap-3">
          <Radial
            value={(last?.spoof ?? 0) * 100}
            label="Spoof Probability"
            color="var(--color-magenta)"
          />
          <div className="w-full p-3 rounded-lg bg-secondary/50 space-y-1 font-mono text-[11px]">
            <div className="flex justify-between">
              <span className="text-muted-foreground">3 kHz peak</span>
              <span>{((last?.spoof ?? 0) * 42).toFixed(1)} dB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">HNR</span>
              <span>{(22 - (last?.spoof ?? 0) * 10).toFixed(1)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">verdict</span>
              <span
                className={
                  (last?.spoof ?? 0) > 0.5 ? "text-destructive" : "text-[color:var(--color-lime)]"
                }
              >
                {(last?.spoof ?? 0) > 0.5 ? "SYNTHETIC" : "BONAFIDE"}
              </span>
            </div>
          </div>
        </div>
      </Card>
      <Card className="col-span-12" title="Acoustic Embedding · 128-D" tag="EMBED">
        <div
          className="grid grid-cols-16 md:grid-cols-32 gap-0.5"
          style={{ gridTemplateColumns: "repeat(32, minmax(0, 1fr))" }}
        >
          {Array.from({ length: 128 }, (_, i) => {
            const v = Math.abs(Math.sin(i * 0.3 + buf.length * 0.1));
            return (
              <div
                key={i}
                className="h-6 rounded-sm"
                style={{ background: `oklch(${0.3 + v * 0.5} ${0.05 + v * 0.18} ${200 + v * 80})` }}
              />
            );
          })}
        </div>
      </Card>
    </div>
  );
}

// ---------------- View: PEFT ----------------
function PeftView({ buf }: { buf: Packet[] }) {
  const [merged, setMerged] = useState(false);
  const scatter = useMemo(
    () =>
      Array.from({ length: 40 }, (_, i) => ({
        x: 20 + Math.random() * 80,
        y: 20 + Math.random() * 80,
        z: 50 + Math.random() * 50,
        cheat: Math.random() > 0.6,
      })),
    [],
  );
  return (
    <div className="grid grid-cols-12 gap-4">
      <Card
        className="col-span-12 lg:col-span-7"
        title="LoRA Low-Rank Decomposition"
        subtitle="W = W₀ + A·B  ·  rank r=8"
        tag="PEFT"
      >
        <div className="flex items-center justify-center gap-3 py-4">
          <MatrixBlock label="W₀" w={12} h={12} tone="cyan" />
          <span className="text-2xl font-mono text-muted-foreground">+</span>
          <MatrixBlock label="A" w={12} h={3} tone="magenta" />
          <span className="text-2xl font-mono text-muted-foreground">×</span>
          <MatrixBlock label="B" w={3} h={12} tone="magenta" />
          <span className="text-2xl font-mono text-muted-foreground">=</span>
          <MatrixBlock
            label={merged ? "W_merged" : "ΔW"}
            w={12}
            h={12}
            tone={merged ? "lime" : "amber"}
          />
        </div>
        <div className="flex items-center justify-between mt-2">
          <p className="text-[11px] font-mono text-muted-foreground">
            params: {merged ? "0 (fused)" : "12·3 + 3·12 = 72"} · base: 144
          </p>
          <button
            onClick={() => setMerged(!merged)}
            className="px-4 h-9 rounded-lg glass hover:border-[color:var(--color-cyan)] text-xs font-mono flex items-center gap-2"
          >
            <Layers className="w-3.5 h-3.5" />
            {merged ? "unmerge_weights()" : "merge_weights()"}
          </button>
        </div>
      </Card>
      <Card
        className="col-span-12 lg:col-span-5"
        title="AI Fluency vs Cheating Quadrant"
        tag="SCATTER"
      >
        <div className="h-64">
          <ResponsiveContainer>
            <ScatterChart>
              <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" opacity={0.4} />
              <XAxis
                type="number"
                dataKey="x"
                name="Interaction Speed"
                domain={[0, 100]}
                tick={{ fill: "var(--color-muted-foreground)", fontSize: 10 }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name="Conceptual Structure"
                domain={[0, 100]}
                tick={{ fill: "var(--color-muted-foreground)", fontSize: 10 }}
              />
              <ZAxis type="number" dataKey="z" range={[40, 200]} />
              <Tooltip
                contentStyle={{
                  background: "var(--color-surface-2)",
                  border: "1px solid var(--color-border)",
                  fontSize: 12,
                }}
                cursor={{ strokeDasharray: "3 3" }}
              />
              <Scatter data={scatter.filter((s) => !s.cheat)} fill="var(--color-cyan)" />
              <Scatter data={scatter.filter((s) => s.cheat)} fill="var(--color-magenta)" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 flex gap-4 text-[11px] font-mono">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[color:var(--color-cyan)]" /> Legal AI Leverage
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[color:var(--color-magenta)]" /> Copy-Paste
            Fraud
          </span>
        </div>
      </Card>
    </div>
  );
}

function MatrixBlock({
  label,
  w,
  h,
  tone,
}: {
  label: string;
  w: number;
  h: number;
  tone: "cyan" | "magenta" | "amber" | "lime";
}) {
  const color = {
    cyan: "var(--color-cyan)",
    magenta: "var(--color-magenta)",
    amber: "var(--color-amber)",
    lime: "var(--color-lime)",
  }[tone];
  return (
    <div className="flex flex-col items-center gap-1.5">
      <div
        className="grid gap-px p-1 rounded"
        style={{
          gridTemplateColumns: `repeat(${w}, 6px)`,
          background: "var(--color-surface-2)",
          border: `1px solid ${color}55`,
        }}
      >
        {Array.from({ length: w * h }, (_, i) => (
          <div
            key={i}
            className="w-1.5 h-1.5"
            style={{ background: `${color}${Math.random() > 0.5 ? "cc" : "44"}` }}
          />
        ))}
      </div>
      <span className="font-mono text-[10px]" style={{ color }}>
        {label}
      </span>
    </div>
  );
}

// ---------------- View: Fusion ----------------
function FusionView({ buf }: { buf: Packet[] }) {
  const last = buf[buf.length - 1];
  return (
    <div className="grid grid-cols-12 gap-4">
      <Card
        className="col-span-12 lg:col-span-8"
        title="Cross-Modal Attention"
        subtitle="vision_audio_attn ⇄ audio_vision_attn"
        tag="FUSION"
      >
        <div className="grid grid-cols-2 gap-4">
          {["Vision → Audio", "Audio → Vision"].map((label, idx) => (
            <div key={label}>
              <p className="text-[10px] font-mono text-muted-foreground mb-2">{label}</p>
              <div className="grid gap-px" style={{ gridTemplateColumns: "repeat(16, 1fr)" }}>
                {Array.from({ length: 16 * 16 }, (_, i) => {
                  const v = Math.abs(Math.sin(i * (idx ? 0.13 : 0.19) + buf.length * 0.05));
                  return (
                    <div
                      key={i}
                      className="aspect-square"
                      style={{
                        background: `oklch(${0.2 + v * 0.55} ${0.05 + v * 0.2} ${idx ? 330 : 200})`,
                      }}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </Card>
      <Card
        className="col-span-12 lg:col-span-4"
        title="Lip-Sync Discrepancy"
        subtitle="Deepfake voice-video drift"
        tag="SYNC"
      >
        <Radial value={last?.lipsync ?? 0} label="AV Drift" color="var(--color-amber)" />
        <div className="mt-4 p-3 rounded-lg bg-secondary/50 font-mono text-[11px] space-y-1">
          <div className="flex justify-between">
            <span className="text-muted-foreground">phoneme_lag</span>
            <span>{((last?.lipsync ?? 0) * 1.8).toFixed(1)} ms</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">viseme_conf</span>
            <span>{(1 - (last?.lipsync ?? 0) / 100).toFixed(2)}</span>
          </div>
        </div>
      </Card>
      <Card
        className="col-span-12"
        title="Joint Vector · 416-D"
        subtitle="[ vision:256 | audio:128 | cognitive:32 ] → risk heads"
        tag="CONCAT"
      >
        <div className="flex gap-1 h-8">
          <SegBar label="vision · 256" width={256} color="var(--color-cyan)" />
          <SegBar label="audio · 128" width={128} color="var(--color-magenta)" />
          <SegBar label="cognitive · 32" width={32} color="var(--color-lime)" />
        </div>
      </Card>
    </div>
  );
}

function SegBar({ label, width, color }: { label: string; width: number; color: string }) {
  return (
    <div
      className="relative rounded-md overflow-hidden flex items-center justify-center"
      style={{ flex: width, background: `${color}22`, border: `1px solid ${color}66` }}
    >
      <span className="font-mono text-[10px]" style={{ color }}>
        {label}
      </span>
    </div>
  );
}

// ---------------- View: Engine ----------------
function EngineView() {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 200);
    return () => clearInterval(id);
  }, []);
  const cells = 60;
  const filled = 24 + Math.floor(Math.sin(tick / 5) * 8 + 8);
  const batches = [4, 8, 12, 16, 6, 11, 15, 9];
  return (
    <div className="grid grid-cols-12 gap-4">
      <Card
        className="col-span-12 lg:col-span-6"
        title="StreamRingBuffer"
        subtitle="300-frame circular · occupancy + evictions"
        tag="RING"
      >
        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${cells}, 1fr)` }}>
          {Array.from({ length: cells }, (_, i) => (
            <div
              key={i}
              className="aspect-square rounded-sm"
              style={{
                background: i < filled ? "var(--color-cyan)" : "var(--color-surface-2)",
                opacity: i === filled - 1 ? 1 : 0.7,
                boxShadow: i === filled - 1 ? "0 0 8px var(--color-cyan)" : "none",
              }}
            />
          ))}
        </div>
        <div className="mt-3 grid grid-cols-3 gap-2 font-mono text-[11px]">
          <Stat k="occupancy" v={`${filled}/${cells}`} />
          <Stat k="evictions/s" v={String(4 + (tick % 3))} />
          <Stat k="write_ptr" v={String(tick % 300)} />
        </div>
      </Card>
      <Card
        className="col-span-12 lg:col-span-6"
        title="DynamicBatchingScheduler"
        subtitle="20 ms window · max_batch=16 · unified GPU pass"
        tag="BATCH"
      >
        <div className="flex items-end gap-2 h-40">
          {batches.map((b, i) => (
            <motion.div
              key={i}
              className="flex-1 rounded-t-md relative"
              style={{
                background: "linear-gradient(to top, var(--color-magenta), var(--color-cyan))",
              }}
              initial={{ height: 0 }}
              animate={{ height: `${(b / 16) * 100}%` }}
              transition={{ delay: i * 0.05 }}
            >
              <span className="absolute -top-5 left-0 right-0 text-center font-mono text-[10px]">
                {b}
              </span>
            </motion.div>
          ))}
        </div>
        <p className="mt-2 text-[11px] font-mono text-muted-foreground">
          avg batch: {(batches.reduce((a, b) => a + b, 0) / batches.length).toFixed(1)} · window: 20
          ms
        </p>
      </Card>
    </div>
  );
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="p-2 rounded bg-secondary/50">
      <p className="text-[9px] text-muted-foreground">{k}</p>
      <p className="text-[color:var(--color-cyan)]">{v}</p>
    </div>
  );
}

// ---------------- View: Quantization ----------------
function QuantView() {
  const rows = [
    { name: "FP32", size: 1632.26, ratio: "1×", mae: "0.00000", latency: 0.2 },
    { name: "INT8", size: 404.09, ratio: "4×", mae: "0.00018", latency: 0.35 },
    { name: "INT4", size: 202.05, ratio: "8×", mae: "0.00334", latency: 0.34 },
  ];
  return (
    <div className="grid grid-cols-12 gap-4">
      <Card
        className="col-span-12 lg:col-span-7"
        title="PTQ Tradeoff Matrix"
        subtitle="Quality vs Speed vs Footprint"
        tag="BENCH"
      >
        <div className="overflow-hidden rounded-lg border border-border">
          <table className="w-full text-sm font-mono">
            <thead className="bg-secondary/60 text-[10px] uppercase text-muted-foreground">
              <tr>
                {["mode", "size KB", "ratio", "MAE", "latency"].map((h) => (
                  <th key={h} className="text-left px-3 py-2">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name} className="border-t border-border">
                  <td className="px-3 py-3">
                    <span className="px-2 py-0.5 rounded text-[color:var(--color-cyan)] bg-secondary text-xs">
                      {r.name}
                    </span>
                  </td>
                  <td className="px-3 py-3">{r.size.toFixed(2)}</td>
                  <td className="px-3 py-3 text-[color:var(--color-lime)]">{r.ratio}</td>
                  <td className="px-3 py-3">{r.mae}</td>
                  <td className="px-3 py-3">{r.latency.toFixed(2)} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <Card className="col-span-12 lg:col-span-5" title="Memory Bandwidth Savings" tag="MEM">
        <div className="h-64">
          <ResponsiveContainer>
            <BarChart data={rows}>
              <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" opacity={0.4} />
              <XAxis
                dataKey="name"
                tick={{ fill: "var(--color-muted-foreground)", fontSize: 11 }}
              />
              <YAxis tick={{ fill: "var(--color-muted-foreground)", fontSize: 10 }} />
              <Tooltip
                contentStyle={{
                  background: "var(--color-surface-2)",
                  border: "1px solid var(--color-border)",
                  fontSize: 12,
                }}
              />
              <Bar dataKey="size" fill="var(--color-cyan)" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
}

// ---------------- View: Benchmarks ----------------
function BenchView() {
  const [sessions, setSessions] = useState<1 | 4 | 8 | 16>(8);
  const [running, setRunning] = useState(false);
  const profiles: Record<number, { rps: number; p50: number; p95: number }> = {
    1: { rps: 36, p50: 21.3, p95: 24.4 },
    4: { rps: 114, p50: 26.5, p95: 36.4 },
    8: { rps: 191, p50: 31.2, p95: 43.2 },
    16: { rps: 132, p50: 114.8, p95: 125.1 },
  };
  const active = profiles[sessions];
  const curve = useMemo(() => {
    // Log-normal-ish latency histogram
    const p50 = active.p50,
      p95 = active.p95;
    return Array.from({ length: 40 }, (_, i) => {
      const x = (i / 40) * (p95 * 1.6);
      const y = Math.exp(-Math.pow((x - p50) / (p95 - p50 + 6), 2)) * 100;
      return { x: +x.toFixed(1), y: +y.toFixed(2) };
    });
  }, [active]);
  return (
    <div className="grid grid-cols-12 gap-4">
      <Card className="col-span-12 lg:col-span-4" title="Concurrency Load Launcher" tag="STRESS">
        <div className="grid grid-cols-2 gap-2">
          {[1, 4, 8, 16].map((s) => (
            <button
              key={s}
              onClick={() => setSessions(s as 1 | 4 | 8 | 16)}
              className={`h-16 rounded-lg border transition-all font-mono ${
                sessions === s
                  ? "border-(--color-cyan) bg-secondary glow-cyan"
                  : "border-border glass hover:border-muted-foreground"
              }`}
            >
              <span className="block text-xs text-muted-foreground">sessions</span>
              <span className="text-lg text-(--color-cyan)">{s}</span>
            </button>
          ))}
        </div>
        <button
          onClick={() => {
            setRunning(true);
            setTimeout(() => setRunning(false), 1400);
          }}
          className="mt-4 w-full h-11 rounded-lg bg-[color:var(--color-cyan)] text-primary-foreground font-medium flex items-center justify-center gap-2 hover:brightness-110"
        >
          <Play className="w-4 h-4" />
          {running ? "RUNNING · wrk -t8 -c16" : "Launch Benchmark"}
        </button>
        <div className="mt-4 grid grid-cols-3 gap-2 font-mono text-[11px]">
          <Stat k="req/s" v={String(active.rps)} />
          <Stat k="p50 ms" v={active.p50.toFixed(1)} />
          <Stat k="p95 ms" v={active.p95.toFixed(1)} />
        </div>
      </Card>
      <Card
        className="col-span-12 lg:col-span-8"
        title="Latency SLA Distribution"
        subtitle={`Sessions: ${sessions} · ${active.rps} req/s`}
        tag="SLA"
      >
        <div className="h-64">
          <ResponsiveContainer>
            <AreaChart data={curve}>
              <defs>
                <linearGradient id="sla" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--color-cyan)" stopOpacity={0.7} />
                  <stop offset="100%" stopColor="var(--color-cyan)" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" opacity={0.4} />
              <XAxis
                dataKey="x"
                tick={{ fill: "var(--color-muted-foreground)", fontSize: 10 }}
                label={{
                  value: "latency (ms)",
                  fill: "var(--color-muted-foreground)",
                  fontSize: 10,
                  position: "insideBottom",
                  offset: -5,
                }}
              />
              <YAxis tick={{ fill: "var(--color-muted-foreground)", fontSize: 10 }} />
              <Tooltip
                contentStyle={{
                  background: "var(--color-surface-2)",
                  border: "1px solid var(--color-border)",
                  fontSize: 12,
                }}
              />
              <Area
                type="monotone"
                dataKey="y"
                stroke="var(--color-cyan)"
                strokeWidth={2}
                fill="url(#sla)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
}

// ---------------- View: Data ----------------
function DataView() {
  return (
    <div className="grid grid-cols-12 gap-4">
      <Card
        className="col-span-12 lg:col-span-6"
        title="ASVspoof 2019 · Logical Access"
        subtitle="Bonafide vs synthetic waveforms"
        tag="AUDIO"
      >
        <div className="space-y-3">
          {[
            { id: "LA_T_0001", verdict: "bonafide" },
            { id: "LA_T_1042", verdict: "spoof-tts" },
            { id: "LA_T_2311", verdict: "spoof-vc" },
          ].map((s) => (
            <div key={s.id} className="p-3 rounded-lg bg-secondary/50 flex items-center gap-3">
              <Waves className="w-4 h-4 text-[color:var(--color-cyan)]" />
              <div className="flex-1">
                <p className="font-mono text-xs">{s.id}</p>
                <div className="flex gap-0.5 mt-1.5 h-6 items-end">
                  {Array.from({ length: 50 }, (_, i) => (
                    <div
                      key={i}
                      className="flex-1 rounded-sm"
                      style={{
                        height: `${20 + Math.abs(Math.sin(i * 0.4 + s.id.length)) * 80}%`,
                        background:
                          s.verdict === "bonafide" ? "var(--color-lime)" : "var(--color-magenta)",
                        opacity: 0.7,
                      }}
                    />
                  ))}
                </div>
              </div>
              <span
                className={`text-[10px] font-mono px-2 py-1 rounded ${s.verdict === "bonafide" ? "text-[color:var(--color-lime)] bg-[color:var(--color-lime)]/10" : "text-[color:var(--color-magenta)] bg-[color:var(--color-magenta)]/10"}`}
              >
                {s.verdict}
              </span>
            </div>
          ))}
        </div>
      </Card>
      <Card
        className="col-span-12 lg:col-span-6"
        title="FaceForensics++"
        subtitle="Real vs manipulated frames"
        tag="VIDEO"
      >
        <div className="grid grid-cols-3 gap-2">
          {[
            { l: "REAL", c: "var(--color-lime)" },
            { l: "DEEP", c: "var(--color-magenta)" },
            { l: "FS2F", c: "var(--color-amber)" },
            { l: "REAL", c: "var(--color-lime)" },
            { l: "NT", c: "var(--color-magenta)" },
            { l: "FS", c: "var(--color-amber)" },
          ].map((f, i) => (
            <div
              key={i}
              className="relative aspect-square rounded-lg overflow-hidden border border-border"
            >
              <div
                className="absolute inset-0"
                style={{
                  background: `radial-gradient(circle at 50% 40%, ${f.c}44, transparent 60%), var(--color-surface-2)`,
                }}
              />
              <Grid3x3 className="absolute inset-0 m-auto w-8 h-8 text-muted-foreground/40" />
              <span
                className="absolute bottom-1 left-1 font-mono text-[9px] px-1 py-0.5 rounded bg-black/60"
                style={{ color: f.c }}
              >
                {f.l}
              </span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ---------------- Root App ----------------
function AegisApp() {
  const [view, setView] = useState<ViewId>("control");
  const [collapsed, setCollapsed] = useState(false);
  const [precision, setPrecision] = useState("int8");
  const [anomaly, setAnomaly] = useState<Anomaly>("normal");
  const { wsConnected, packets: buf, metrics } = useAegisBackend("candidate_sherlock_001", anomaly);

  const p95 = metrics?.stages?.total_ms?.p95
    ? Number(metrics.stages.total_ms.p95.toFixed(2))
    : precision === "fp32"
      ? 43.1
      : precision === "int8"
        ? 24.45
        : 22.1;
  const throughput =
    metrics?.throughput || (precision === "fp32" ? 118 : precision === "int8" ? 191 : 214);

  return (
    <div className="min-h-screen">
      <Header
        precision={precision}
        setPrecision={setPrecision}
        wsConnected={wsConnected}
        latencyP95={p95}
        throughput={throughput}
      />
      <div className="flex">
        <Sidebar view={view} setView={setView} collapsed={collapsed} setCollapsed={setCollapsed} />
        <main className="flex-1 min-w-0 p-4 md:p-6">
          <div className="mb-4 flex items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-tight">
              {NAV.find((n) => n.id === view)?.label}
            </h1>
            <span className="font-mono text-[10px] text-muted-foreground">
              {NAV.find((n) => n.id === view)?.hint}
            </span>
            <span className="ml-auto flex items-center gap-1.5 text-[11px] font-mono text-muted-foreground">
              <Radio className="w-3 h-3 text-[color:var(--color-cyan)]" />
              GET /health · 200 · active_sessions=1 · mode={precision.toUpperCase()}
            </span>
          </div>

          <AnimatePresence mode="wait">
            <motion.div
              key={view}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.25 }}
            >
              {view === "control" && (
                <ControlRoom anomaly={anomaly} setAnomaly={setAnomaly} buf={buf} />
              )}
              {view === "vision" && <VisionView buf={buf} />}
              {view === "audio" && <AudioView buf={buf} />}
              {view === "peft" && <PeftView buf={buf} />}
              {view === "fusion" && <FusionView buf={buf} />}
              {view === "engine" && <EngineView />}
              {view === "quant" && <QuantView />}
              {view === "bench" && <BenchView />}
              {view === "data" && <DataView />}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
