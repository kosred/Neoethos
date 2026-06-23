import { useState } from "react";
import { enginesStatus, trainingStart, trainingStop, type StartJob } from "../api";
import { usePoll } from "../hooks";

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
            <input value={symbol} placeholder="(config)" onChange={(e) => setSymbol(e.target.value)} style={{ width: 110 }} />
          </label>
          <label>
            Base TF
            <input value={baseTf} placeholder="(config)" onChange={(e) => setBaseTf(e.target.value)} style={{ width: 80 }} />
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
