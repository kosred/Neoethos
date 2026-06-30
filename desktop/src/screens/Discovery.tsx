import { useEffect, useState, useSyncExternalStore } from "react";
import { enginesStatus, settings, dataImport, pickDataFile } from "../api";
import { usePoll } from "../hooks";
import { useSymbolOptions, useTimeframeOptions, TimeframeSelect } from "../components/Select";
import { HelpPanel, HelpStep } from "../components/Help";
import {
  subscribe,
  getSnapshot,
  setQueue,
  startQueue,
  stopQueue,
  clearQueue,
  drive,
  labelFor,
  type QItem,
} from "../discoveryQueue";

// Fast-first TF order so the cheap, high-yield timeframes run before the
// dense ones (M5/M3 take hours) — you get strategies quickly and the slow
// units land last. Lower index = runs earlier.
const TF_SPEED = ["MN1", "W1", "D1", "H12", "H4", "H1", "M30", "M15", "M5", "M3", "M1"];
const tfRank = (t: string) => {
  const i = TF_SPEED.indexOf(t);
  return i < 0 ? 99 : i;
};

const num = (s: string) => (s.trim() === "" ? undefined : Number(s));
const statusIcon: Record<QItem["status"], string> = {
  pending: "⏳",
  running: "▶",
  done: "✓",
  failed: "✗",
};

function Chips({
  opts,
  sel,
  onToggle,
}: {
  opts: string[];
  sel: string[];
  onToggle: (v: string) => void;
}) {
  return (
    <div className="chip-row">
      {opts.map((o) => (
        <button
          key={o}
          type="button"
          className={`chip ${sel.includes(o) ? "on" : ""}`}
          onClick={() => onToggle(o)}
        >
          {o}
        </button>
      ))}
    </div>
  );
}

export default function Discovery() {
  const { data: st, error } = usePoll(enginesStatus, 2000);
  const { data: cfg } = usePoll(settings, 0);
  const q = useSyncExternalStore(subscribe, getSnapshot);
  const symOpts = useSymbolOptions();
  const tfOpts = useTimeframeOptions();

  const [selSyms, setSelSyms] = useState<string[]>([]);
  const [selTfs, setSelTfs] = useState<string[]>([]);
  const [adv, setAdv] = useState(false);
  const [population, setPopulation] = useState("");
  const [generations, setGenerations] = useState("");
  const [targets, setTargets] = useState("");
  const [portfolio, setPortfolio] = useState("");
  const [msg, setMsg] = useState("");

  // Import data file (lives here because data is only for search + training).
  const [impSrc, setImpSrc] = useState("");
  const [impSym, setImpSym] = useState("EURUSD");
  const [impTf, setImpTf] = useState("H1");
  const [impMsg, setImpMsg] = useState("");
  const [impBusy, setImpBusy] = useState(false);

  const browse = async () => {
    try {
      const p = await pickDataFile();
      if (p) setImpSrc(p);
    } catch (e) {
      setImpMsg(String(e));
    }
  };
  const doImport = async () => {
    if (!impSrc) { setImpMsg("Choose a file first (Browse…)."); return; }
    setImpBusy(true);
    setImpMsg("Importing…");
    try {
      const r: any = await dataImport(impSrc, impSym.toUpperCase(), impTf.toUpperCase());
      setImpMsg(`✓ Imported → ${r?.writtenPath ?? r?.written_path ?? "done"}`);
    } catch (e) {
      setImpMsg(`Import failed: ${e}`);
    } finally {
      setImpBusy(false);
    }
  };

  const state = st?.discovery ?? "…";
  const running = state.toLowerCase().startsWith("running");
  const stage = st?.discoveryStage ?? st?.discovery_stage ?? "";
  const percent = st?.discoveryPercent ?? st?.discovery_percent ?? 0;
  const summary = st?.discoverySummary ?? st?.discovery_summary ?? "";
  const counters = st?.discoveryCounters ?? st?.discovery_counters ?? [];

  // RAM / disk readout (operator visibility).
  const ramTotal = st?.ramTotalGb ?? 0;
  const ramAvail = st?.ramAvailableGb ?? 0;
  const ramUsedPct = ramTotal > 0 ? ((ramTotal - ramAvail) / ramTotal) * 100 : 0;
  const diskMb = st?.featureStoreMb ?? 0;

  // Drive the queue forward on every poll tick.
  useEffect(() => {
    if (st) void drive(running, summary);
  }, [st]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggle =
    (set: React.Dispatch<React.SetStateAction<string[]>>) => (v: string) =>
      set((cur) => (cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v]));

  const queued = q.items.length;
  const done = q.items.filter((i) => i.status === "done").length;
  const failed = q.items.filter((i) => i.status === "failed").length;

  const launch = () => {
    const syms = selSyms.length ? selSyms : [""]; // "" → resolve from config
    const tfs = (selTfs.length ? selTfs : [""])
      .slice()
      .sort((a, b) => tfRank(a) - tfRank(b));
    // symbol-major: finish a pair's TFs before moving on.
    const pairs = syms.flatMap((s) => tfs.map((t) => ({ symbol: s, tf: t })));
    setQueue(pairs, {
      population: num(population),
      generations: num(generations),
      target_candidates: num(targets),
      portfolio_size: num(portfolio),
    });
    startQueue();
    setMsg(`Queued ${pairs.length} run${pairs.length === 1 ? "" : "s"}.`);
  };

  const stop = async () => {
    await stopQueue();
    setMsg("Stopped — current run cancelled, queue cleared.");
  };

  return (
    <div className="screen">
      <h1>
        Discovery{" "}
        {cfg?.tradingMode && (
          <span className={`badge ${cfg.tradingMode === "risky" ? "live" : "demo"}`}>
            {cfg.tradingMode === "risky" ? "🚀 RISKY MODE" : "🛡 PROP-FIRM MODE"}
          </span>
        )}
      </h1>
      <p className="sub">
        Genetic strategy search · queue many pairs · <b>mode + tuning in Settings</b>
      </p>

      <HelpPanel id="discovery">
        <p>
          Discovery is the <b>strategy factory</b>. Pick one or more <b>symbols</b> and{" "}
          <b>timeframes</b>, press <b>Start queue</b>, and it runs each combination in turn — breeding
          and testing thousands of rules, keeping only the ones that survive out-of-sample validation.
        </p>
        <HelpStep n={1}>
          Tick the symbols and timeframes you want (or leave both empty to use your config defaults).
          Each symbol × timeframe becomes one queued run.
        </HelpStep>
        <HelpStep n={2}>
          Timeframes run <b>fastest-first</b> (H1 in minutes, M3 can take ~hours on 11-year data), so
          you get results early. Watch the live bar, stage and counters for whatever is running now.
        </HelpStep>
        <HelpStep n={3}>
          The <b>RAM / disk</b> strip shows what the run is consuming: cubes that fit in RAM use no
          disk; large ones stream to disk and are freed as each timeframe finishes. Results appear in{" "}
          <b>Strategy Lab</b> / <b>Autopilot</b>.
        </HelpStep>
        <p className="muted small">
          The engine runs in-process, so keep the app open while a queue runs. Leaving this screen is
          fine — the queue resumes when you return.
        </p>
      </HelpPanel>

      {/* ── Live machine-resource strip ── */}
      <div className="res-strip">
        <div className="res-item">
          <div className="res-label">
            RAM {ramAvail.toFixed(1)} GB free of {ramTotal.toFixed(0)} GB
          </div>
          <div className="res-bar">
            <div className="res-fill" style={{ width: `${Math.min(100, ramUsedPct)}%` }} />
          </div>
        </div>
        <div className="res-item res-disk">
          <div className="res-label">Discovery disk</div>
          <div className="res-value">
            {diskMb > 0 ? `${(diskMb / 1024).toFixed(2)} GB` : "0 (all in RAM)"}
          </div>
        </div>
      </div>

      {/* ── Currently running ── */}
      <div className="engine-status">
        <span className={`badge ${running ? "live" : "demo"}`}>
          {running ? "RUNNING" : state.toUpperCase()}
        </span>
        {stage && <span className="muted">{stage}</span>}
        {running && (
          <div className="progress">
            <div className="progress-bar" style={{ width: `${Math.min(100, percent)}%` }} />
            <span className="progress-label">{percent.toFixed(0)}%</span>
          </div>
        )}
      </div>
      {summary && <div className="banner info">{summary}</div>}
      {error && <div className="banner warn">{error}</div>}

      {counters.length > 0 && (
        <div
          className="cards"
          style={{ gridTemplateColumns: `repeat(${Math.min(4, counters.length)}, 1fr)` }}
        >
          {counters.map((c) => (
            <div className="card" key={c.name}>
              <div className="card-label">{c.name.toUpperCase()}</div>
              <div className="card-value">{c.value.toLocaleString()}</div>
            </div>
          ))}
        </div>
      )}

      {/* ── Queue ── */}
      {queued > 0 && (
        <>
          <h2>
            Queue{" "}
            <span className="muted">
              — {done} done · {queued - done - failed} left
              {failed ? ` · ${failed} failed` : ""}
            </span>
          </h2>
          <div className="queue-list">
            {q.items.map((it) => (
              <div className={`q-item q-${it.status}`} key={it.id}>
                <span className="q-icon">{statusIcon[it.status]}</span>
                <span className="q-name">{labelFor(it.symbol, it.tf)}</span>
                {it.status === "running" && (
                  <span className="q-prog">
                    <span className="q-bar">
                      <span className="q-fill" style={{ width: `${Math.min(100, percent)}%` }} />
                    </span>
                    {stage || "running"} · {percent.toFixed(0)}%
                  </span>
                )}
                {it.note && it.status !== "running" && (
                  <span className="q-note muted">{it.note}</span>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {/* ── Build a queue ── */}
      <h2>Build a queue</h2>
      <div className="ticket">
        <label className="picker-label">
          Symbols <span className="muted">({selSyms.length || "config default"})</span>
          <div className="picker-actions">
            <button type="button" className="link" onClick={() => setSelSyms(symOpts)}>all</button>
            <button type="button" className="link" onClick={() => setSelSyms([])}>none</button>
          </div>
        </label>
        <Chips opts={symOpts} sel={selSyms} onToggle={toggle(setSelSyms)} />

        <label className="picker-label" style={{ marginTop: 12 }}>
          Timeframes <span className="muted">({selTfs.length ? "fastest-first" : "config default"})</span>
          <div className="picker-actions">
            <button type="button" className="link" onClick={() => setSelTfs(["H1", "M30", "M15", "M5"])}>productive</button>
            <button type="button" className="link" onClick={() => setSelTfs(tfOpts)}>all</button>
            <button type="button" className="link" onClick={() => setSelTfs([])}>none</button>
          </div>
        </label>
        <Chips opts={tfOpts} sel={selTfs} onToggle={toggle(setSelTfs)} />

        <label style={{ flexDirection: "row", alignItems: "center", gap: 6, marginTop: 12 }}>
          <input type="checkbox" checked={adv} onChange={(e) => setAdv(e.target.checked)} /> Advanced knobs
        </label>
        {adv && (
          <div className="ticket-row" style={{ marginTop: 8 }}>
            <label>Population<input type="number" min="0" step="50" value={population} placeholder="default" onChange={(e) => setPopulation(e.target.value)} /></label>
            <label>Generations<input type="number" min="0" step="10" value={generations} placeholder="default" onChange={(e) => setGenerations(e.target.value)} /></label>
            <label>Target candidates<input type="number" min="0" step="10" value={targets} placeholder="default" onChange={(e) => setTargets(e.target.value)} /></label>
            <label>Portfolio size<input type="number" min="0" step="1" value={portfolio} placeholder="default" onChange={(e) => setPortfolio(e.target.value)} /></label>
          </div>
        )}

        <div className="muted small" style={{ marginTop: 8 }}>
          {(() => {
            const ns = selSyms.length || 1;
            const nt = selTfs.length || 1;
            return `${ns} symbol${ns === 1 ? "" : "s"} × ${nt} timeframe${nt === 1 ? "" : "s"} = ${ns * nt} run${ns * nt === 1 ? "" : "s"}`;
          })()}
        </div>

        <div className="btn-row">
          <button className="primary" disabled={q.active} onClick={launch}>
            {q.active ? "Queue running…" : "Start queue"}
          </button>
          <button className="danger" disabled={!q.active && !running} onClick={stop}>
            Stop
          </button>
          {queued > 0 && !q.active && (
            <button className="ghost" onClick={() => { clearQueue(); setMsg(""); }}>Clear</button>
          )}
        </div>
        {msg && <div className="banner info">{msg}</div>}
      </div>

      {/* ── Import data file (data is only for search + training) ── */}
      <h2>Import data file</h2>
      <div className="ticket">
        <p className="muted small">Bring in a CSV / Parquet / TSV you already have — it's converted into the engine's format so you can search + train on it.</p>
        <div className="ticket-row">
          <button onClick={browse} disabled={impBusy}>Browse…</button>
          <label style={{ flex: 1 }}>
            File
            <input value={impSrc} onChange={(e) => setImpSrc(e.target.value)} placeholder="(choose a file with Browse…)" style={{ width: "100%" }} />
          </label>
          <label>Symbol<input value={impSym} onChange={(e) => setImpSym(e.target.value)} style={{ width: 90 }} placeholder="EURUSD" /></label>
          <label>TF<TimeframeSelect value={impTf} onChange={setImpTf} style={{ width: 80 }} /></label>
          <button className="primary" disabled={impBusy || !impSrc} onClick={doImport}>Import</button>
        </div>
        {impMsg && <div className="banner info">{impMsg}</div>}
      </div>
    </div>
  );
}
