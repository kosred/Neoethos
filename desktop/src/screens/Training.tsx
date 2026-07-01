import { useEffect, useState } from "react";
import { enginesStatus, trainingStart, trainingStop, dataCoverage, type StartJob, type SymbolCoverage } from "../api";
import { usePoll } from "../hooks";
import { SymbolSelect, TimeframeSelect } from "../components/Select";
import { HelpPanel, HelpStep, Tip } from "../components/Help";

const pick = <T,>(...vals: (T | undefined)[]) => vals.find((v) => v !== undefined);

export default function Training() {
  const { data: st, error, reload } = usePoll(enginesStatus, 2000);
  const [symbol, setSymbol] = useState("");
  const [baseTf, setBaseTf] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [cov, setCov] = useState<SymbolCoverage | null>(null);

  const state = st?.training ?? "…";
  const running = state.toLowerCase().startsWith("running");
  const summary = pick(st?.trainingSummary, st?.training_summary) ?? "";

  // Pre-flight coverage for the picked pair/TF (only when both are chosen).
  useEffect(() => {
    if (!symbol.trim() || !baseTf.trim()) { setCov(null); return; }
    let live = true;
    dataCoverage([symbol.trim().toUpperCase()], baseTf.trim().toUpperCase())
      .then((c) => { if (live) setCov(c[0] ?? null); })
      .catch(() => { if (live) setCov(null); });
    return () => { live = false; };
  }, [symbol, baseTf]);

  const start = async () => {
    setBusy(true);
    setMsg("Starting training…");
    const body: StartJob = {
      symbol: symbol.trim() || undefined,
      base_tf: baseTf.trim() || undefined,
    };
    try {
      await trainingStart(body);
      setMsg("✓ Training started.");
      reload();
    } catch (e) {
      setMsg(`Start failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };
  const stop = async () => {
    setBusy(true);
    try {
      await trainingStop();
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
      <h1>Training</h1>
      <p className="sub">Train the model ensemble on discovered features · symbol/base from config when blank</p>

      <HelpPanel id="training">
        <p>Training fits the <b>machine-learning ensemble</b> (the models that act as a regime filter on top of the discovered rules). It learns from the same features discovery used and is validated on an 80/20 hold-out so it can't just memorise the past.</p>
        <HelpStep n={1}>Pick a <b>Symbol</b> and <b>Base TF</b> (or leave <i>(from config)</i>). Use the same pair/timeframe you ran discovery on.</HelpStep>
        <HelpStep n={2}>Check the <b>pre-flight</b> below — it shows exactly what will train (pair, timeframe, and how much history) before you commit.</HelpStep>
        <HelpStep n={3}>Press <b>Start training</b>. Progress shows here; trained models are saved to the model store (see <b>Files &amp; Storage</b> / <b>Intelligence</b>).</HelpStep>
        <p className="muted small">Models never decide direction on their own — the discovered rules do. The ensemble only down-weights trades in unfavourable regimes.</p>
      </HelpPanel>

      <div className="engine-status">
        <span className={`badge ${running ? "live" : "demo"}`}>{running ? "RUNNING" : state.toUpperCase()}</span>
      </div>
      {summary && <div className="banner info">{summary}</div>}
      {error && <div className="banner warn">{error}</div>}

      <h2>Launch training</h2>
      <div className="ticket">
        <div className="ticket-row">
          <label>
            Symbol <Tip text="The pair to train on. Leave (from config) to use your configured default. Best practice: train on the SAME pair + timeframe you discovered strategies on, so the ensemble sees the same features." />
            <SymbolSelect value={symbol} onChange={setSymbol} allowConfig style={{ width: 120 }} />
          </label>
          <label>
            Base TF <Tip text="The base timeframe. The ensemble also reads the higher timeframes automatically for regime context — same as discovery." />
            <TimeframeSelect value={baseTf} onChange={setBaseTf} allowConfig style={{ width: 90 }} />
          </label>
        </div>

        {/* ── Pre-flight: exactly what will train ── */}
        <div className="banner info" style={{ marginTop: 4 }}>
          <b>Before you start</b> —{" "}
          {symbol.trim() && baseTf.trim() ? (
            <>
              training <b>{symbol.trim().toUpperCase()} {baseTf.trim().toUpperCase()}</b> on{" "}
              {cov ? (
                cov.bars > 0 ? (
                  <><b>{cov.years.toFixed(1)} years</b> ({cov.bars.toLocaleString()} bars){cov.years < 2 ? " ⚠ low history" : ""}</>
                ) : (
                  <span className="sell">⚠ no local data — download it in Data first</span>
                )
              ) : "…"}
              {" · "}fits the discovered-feature ensemble, validated on an 80/20 hold-out.
            </>
          ) : (
            <span className="muted">pair/timeframe from config defaults. Pick both above to preview years of data + bar count.</span>
          )}
        </div>

        <div className="btn-row">
          <button className="primary" disabled={busy || running} onClick={start}>
            {running ? "Running…" : "Start training"}
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
