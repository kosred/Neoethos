import { useState } from "react";
import {
  enginesStatus,
  discoveryStart,
  discoveryStop,
  type StartJob,
} from "../api";
import { usePoll } from "../hooks";
import { SymbolSelect, TimeframeSelect } from "../components/Select";
import { HelpPanel, HelpStep } from "../components/Help";

const pick = <T,>(...vals: (T | undefined)[]) => vals.find((v) => v !== undefined);

export default function Discovery() {
  const { data: st, error, reload } = usePoll(enginesStatus, 2000);
  const [symbol, setSymbol] = useState("");
  const [baseTf, setBaseTf] = useState("");
  const [adv, setAdv] = useState(false);
  const [population, setPopulation] = useState<string>("");
  const [generations, setGenerations] = useState<string>("");
  const [targets, setTargets] = useState<string>("");
  const [portfolio, setPortfolio] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const state = st?.discovery ?? "…";
  const running = state.toLowerCase().startsWith("running");
  const stage = pick(st?.discoveryStage, st?.discovery_stage) ?? "";
  const percent = pick(st?.discoveryPercent, st?.discovery_percent) ?? 0;
  const summary = pick(st?.discoverySummary, st?.discovery_summary) ?? "";
  const counters = pick(st?.discoveryCounters, st?.discovery_counters) ?? [];

  const num = (s: string) => (s.trim() === "" ? undefined : Number(s));

  const start = async () => {
    setBusy(true);
    setMsg("Starting discovery…");
    const body: StartJob = {
      symbol: symbol.trim() || undefined,
      base_tf: baseTf.trim() || undefined,
      population: num(population),
      generations: num(generations),
      target_candidates: num(targets),
      portfolio_size: num(portfolio),
    };
    try {
      await discoveryStart(body);
      setMsg("✓ Discovery started.");
      reload();
    } catch (e) {
      setMsg(`Start failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };
  const stop = async () => {
    setBusy(true);
    setMsg("Stopping…");
    try {
      await discoveryStop();
      setMsg("Stop requested.");
      reload();
    } catch (e) {
      setMsg(`Stop failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <h1>Discovery</h1>
      <p className="sub">Genetic strategy search · symbol/base resolved from config when blank</p>

      <HelpPanel id="discovery">
        <p>Discovery is the <b>strategy factory</b>. It runs a genetic algorithm that breeds and tests thousands of trading rules on your downloaded history, keeps only the ones that survive out-of-sample validation, and saves them as a portfolio you can later replay or trade.</p>
        <HelpStep n={1}>Pick a <b>Symbol</b> and <b>Base TF</b> (or leave them on <i>(from config)</i> to use your defaults). The base timeframe is the bar size the rules trade on; higher timeframes are added automatically for context.</HelpStep>
        <HelpStep n={2}>Optional <b>Advanced</b> knobs: <b>Population</b>/<b>Generations</b> = how wide/deep the search goes (bigger = slower, more thorough). <b>Target candidates</b> = how many survivors to aim for. <b>Portfolio size</b> = how many to keep in the final set. Leave blank for sensible defaults.</HelpStep>
        <HelpStep n={3}>Press <b>Start discovery</b> and watch the progress bar + live counters. It can run for minutes to hours depending on data + settings. Results land in the cache and show up in <b>Strategy Lab</b> / <b>Autopilot</b>.</HelpStep>
        <p className="muted small">Validation is automatic (80/20 in-sample split + walk-forward + CPCV). Only strategies that hold up out-of-sample are exported.</p>
      </HelpPanel>

      <div className="engine-status">
        <span className={`badge ${running ? "live" : "demo"}`}>{running ? "RUNNING" : state.toUpperCase()}</span>
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
        <div className="cards" style={{ gridTemplateColumns: `repeat(${Math.min(4, counters.length)}, 1fr)` }}>
          {counters.map((c) => (
            <div className="card" key={c.name}>
              <div className="card-label">{c.name.toUpperCase()}</div>
              <div className="card-value">{c.value.toLocaleString()}</div>
            </div>
          ))}
        </div>
      )}

      <h2>Launch a search</h2>
      <div className="ticket">
        <div className="ticket-row">
          <label>
            Symbol
            <SymbolSelect value={symbol} onChange={setSymbol} allowConfig style={{ width: 120 }} />
          </label>
          <label>
            Base TF
            <TimeframeSelect value={baseTf} onChange={setBaseTf} allowConfig style={{ width: 90 }} />
          </label>
          <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={adv} onChange={(e) => setAdv(e.target.checked)} /> Advanced
          </label>
        </div>
        {adv && (
          <div className="ticket-row" style={{ marginTop: 10 }}>
            <label>Population<input type="number" min="0" step="50" value={population} placeholder="default" onChange={(e) => setPopulation(e.target.value)} /></label>
            <label>Generations<input type="number" min="0" step="10" value={generations} placeholder="default" onChange={(e) => setGenerations(e.target.value)} /></label>
            <label>Target candidates<input type="number" min="0" step="10" value={targets} placeholder="default" onChange={(e) => setTargets(e.target.value)} /></label>
            <label>Portfolio size<input type="number" min="0" step="1" value={portfolio} placeholder="default" onChange={(e) => setPortfolio(e.target.value)} /></label>
          </div>
        )}
        <div className="btn-row">
          <button className="primary" disabled={busy || running} onClick={start}>
            {running ? "Running…" : "Start discovery"}
          </button>
          <button className="danger" disabled={busy || !running} onClick={stop}>
            Stop
          </button>
        </div>
        {msg && <div className="banner info">{msg}</div>}
      </div>
    </div>
  );
}
