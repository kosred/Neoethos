import { useState } from "react";
import {
  portfoliosList,
  autonomousStatus,
  autonomousStart,
  autonomousStop,
  autonomousReplay,
  openPath,
  type PortfolioEntry,
} from "../api";
import { usePoll } from "../hooks";

function StatGrid({ data }: { data: any }) {
  if (!data || typeof data !== "object") return null;
  const rows = Object.entries(data).filter(([, v]) => typeof v !== "object" || v === null);
  return (
    <table className="tbl">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <td style={{ color: "#9ca3af" }}>{k.replace(/_/g, " ")}</td>
            <td>{typeof v === "number" ? (Number.isInteger(v) ? (v as number).toLocaleString() : (v as number).toFixed(4)) : String(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function Autopilot() {
  const { data: list, error, reload } = usePoll(portfoliosList, 0);
  const { data: status, reload: reloadStatus } = usePoll(autonomousStatus, 3000);
  const [sel, setSel] = useState<PortfolioEntry | null>(null);
  const [replay, setReplay] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const running = !!(status?.running ?? status?.live ?? status?.active);

  const act = async (fn: () => Promise<any>, label: string) => {
    if (!sel) { setMsg("Pick a strategy first."); return; }
    setBusy(true);
    setMsg(`${label} ${sel.symbol ?? ""} ${sel.baseTf ?? ""}…`);
    try {
      const r = await fn();
      if (label === "Replaying") setReplay(r);
      setMsg(`✓ ${label} done.`);
      await reloadStatus();
    } catch (e) {
      setMsg(`${label} failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const body = sel ? { symbol: sel.symbol ?? undefined, base_tf: sel.baseTf ?? undefined } : {};

  return (
    <div className="screen">
      <h1>Autopilot <span className={`badge ${running ? "live" : "demo"}`}>{running ? "LIVE RUNNING" : "STOPPED"}</span></h1>
      <p className="sub">Run an EXISTING discovered strategy — dry-run on history or live — with clear file provenance</p>

      <div className="btn-row">
        <button onClick={reload} disabled={busy}>Refresh strategies</button>
        <span className="muted small">{list?.count ?? 0} portfolios found</span>
      </div>
      {error && <div className="banner warn">{error}</div>}

      {(list?.portfolios.length ?? 0) === 0 ? (
        <p className="muted">No discovered strategies yet — run Discovery, then promote in Strategy Lab.</p>
      ) : (
        <table className="tbl">
          <thead><tr><th></th><th>Symbol</th><th>Base TF</th><th>Genes</th><th>File</th><th></th></tr></thead>
          <tbody>
            {list!.portfolios.map((p) => (
              <tr key={p.path} className={sel?.path === p.path ? "row-sel" : ""}>
                <td><input type="radio" checked={sel?.path === p.path} onChange={() => setSel(p)} /></td>
                <td><b>{p.symbol ?? "?"}</b></td>
                <td>{p.baseTf ?? "?"}</td>
                <td>{p.geneCount ?? "—"}</td>
                <td style={{ fontFamily: "monospace", fontSize: 11, color: "#9ca3af", maxWidth: 320, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={p.path}>{p.fileName}</td>
                <td><button onClick={() => openPath(p.path).catch(() => {})}>Open</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <div className="ticket" style={{ marginTop: 14 }}>
        <h2>{sel ? `${sel.symbol ?? "?"} ${sel.baseTf ?? ""}` : "Select a strategy above"}</h2>
        {sel && <p className="muted small">{sel.path}</p>}
        <div className="btn-row">
          <button disabled={busy || !sel} onClick={() => act(() => autonomousReplay(body), "Replaying")}>Replay (dry-run)</button>
          <button className="primary" disabled={busy || !sel || running} onClick={() => act(() => autonomousStart(body), "Starting live")}>Start live</button>
          <button className="danger" disabled={busy || !running} onClick={() => act(() => autonomousStop(), "Stopping")}>Stop</button>
        </div>
        {msg && <div className="banner info">{msg}</div>}
      </div>

      <h2>Live engine status</h2>
      {status ? <StatGrid data={status} /> : <p className="muted">No status yet.</p>}
      {replay && (<><h2>Replay result</h2><StatGrid data={replay} /></>)}
    </div>
  );
}
