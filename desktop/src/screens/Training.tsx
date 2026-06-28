import { useState } from "react";
import { enginesStatus, trainingStart, trainingStop, type StartJob } from "../api";
import { usePoll } from "../hooks";
import { SymbolSelect, TimeframeSelect } from "../components/Select";
import { HelpPanel, HelpStep } from "../components/Help";

const pick = <T,>(...vals: (T | undefined)[]) => vals.find((v) => v !== undefined);

export default function Training() {
  const { data: st, error, reload } = usePoll(enginesStatus, 2000);
  const [symbol, setSymbol] = useState("");
  const [baseTf, setBaseTf] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const state = st?.training ?? "…";
  const running = state.toLowerCase().startsWith("running");
  const summary = pick(st?.trainingSummary, st?.training_summary) ?? "";

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
        <HelpStep n={2}>Press <b>Start training</b>. Progress shows here; trained models are saved to the model store (see <b>Files &amp; Storage</b> / <b>Intelligence</b>).</HelpStep>
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
            Symbol
            <SymbolSelect value={symbol} onChange={setSymbol} allowConfig style={{ width: 120 }} />
          </label>
          <label>
            Base TF
            <TimeframeSelect value={baseTf} onChange={setBaseTf} allowConfig style={{ width: 90 }} />
          </label>
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
