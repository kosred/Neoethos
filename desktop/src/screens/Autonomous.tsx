import { useState } from "react";
import { autonomousStatus, autonomousStart, autonomousStop, autonomousReplay } from "../api";
import { usePoll } from "../hooks";

function StatGrid({ data }: { data: any }) {
  if (!data || typeof data !== "object") return null;
  const entries = Object.entries(data).filter(([, v]) => typeof v !== "object" || v === null);
  return (
    <table className="tbl">
      <tbody>
        {entries.map(([k, v]) => (
          <tr key={k}>
            <td style={{ color: "#9ca3af" }}>{k.replace(/_/g, " ")}</td>
            <td>{typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : (v as number).toFixed(4)) : String(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function Autonomous() {
  const { data: status, error, reload } = usePoll(autonomousStatus, 3000);
  const [symbol, setSymbol] = useState("");
  const [baseTf, setBaseTf] = useState("");
  const [replayResult, setReplayResult] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const running = !!(status?.running ?? status?.live ?? status?.active);

  const act = async (fn: () => Promise<any>, label: string) => {
    setBusy(true);
    setMsg(`${label}…`);
    try {
      const r = await fn();
      if (label === "Replaying") setReplayResult(r);
      setMsg(`✓ ${label} done.`);
      await reload();
    } catch (e) {
      setMsg(`${label} failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const body = { symbol: symbol.trim() || undefined, base_tf: baseTf.trim() || undefined };

  return (
    <div className="screen">
      <h1>Autonomous Trader</h1>
      <p className="sub">Bar → gene signal → risk → execution loop · live or dry-run over history</p>

      <div className="engine-status">
        <span className={`badge ${running ? "live" : "demo"}`}>{running ? "LIVE RUNNING" : "STOPPED"}</span>
      </div>

      <div className="ticket">
        <div className="ticket-row">
          <label>Symbol<input value={symbol} placeholder="(config)" onChange={(e) => setSymbol(e.target.value)} style={{ width: 110 }} /></label>
          <label>Base TF<input value={baseTf} placeholder="(config)" onChange={(e) => setBaseTf(e.target.value)} style={{ width: 80 }} /></label>
        </div>
        <div className="btn-row">
          <button className="primary" disabled={busy || running} onClick={() => act(() => autonomousStart(body), "Starting live")}>Start live</button>
          <button className="danger" disabled={busy || !running} onClick={() => act(() => autonomousStop(), "Stopping")}>Stop</button>
          <button disabled={busy} onClick={() => act(() => autonomousReplay(body), "Replaying")}>Replay (dry-run)</button>
        </div>
        {msg && <div className="banner info">{msg}</div>}
      </div>

      {error && <div className="banner warn">{error}</div>}

      <h2>Live status</h2>
      {status ? <StatGrid data={status} /> : <p className="muted">No status yet.</p>}

      {replayResult && (
        <>
          <h2>Replay result</h2>
          <StatGrid data={replayResult} />
        </>
      )}
    </div>
  );
}
