import { useState } from "react";
import { promotionStatus, promoteStrategy } from "../api";
import { SymbolSelect, TimeframeSelect } from "../components/Select";
import { HelpPanel, HelpStep } from "../components/Help";

export default function StrategyLab() {
  const [symbol, setSymbol] = useState("");
  const [baseTf, setBaseTf] = useState("");
  const [status, setStatus] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const check = async () => {
    setBusy(true);
    setMsg("Checking promotion gate…");
    try {
      const s = await promotionStatus(symbol, baseTf);
      setStatus(s);
      setMsg("");
    } catch (e) {
      setMsg(String(e));
      setStatus(null);
    } finally {
      setBusy(false);
    }
  };

  const promote = async () => {
    setBusy(true);
    setMsg("Promoting to live…");
    try {
      const r = await promoteStrategy(symbol, baseTf);
      setMsg(`${r?.promoted ? "✓" : "✗"} ${r?.message ?? ""} ${r?.filesCopied ? `(${r.filesCopied} files)` : ""}`);
      await check();
    } catch (e) {
      setMsg(`Promote failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const decision = status?.decision;
  const eligible = !!decision?.promoted;
  const summary: string = decision?.summary ?? "";

  return (
    <div className="screen">
      <h1>Strategy Lab</h1>
      <p className="sub">Promotion gate — validate a discovered portfolio and promote it to live</p>

      <HelpPanel id="strategylab">
        <p>Strategy Lab is the <b>quality checkpoint</b> between discovery and trading. It re-checks a discovered portfolio against the promotion criteria (profit factor, Sharpe, drawdown, trade count) and, if it passes, copies it into the live set.</p>
        <HelpStep n={1}>Pick the <b>Symbol</b> + <b>Base TF</b> of a portfolio you discovered, then press <b>Check gate</b>. You'll see each criterion and a <b>PROMOTE</b> / <b>HOLD</b> verdict.</HelpStep>
        <HelpStep n={2}>If it says PROMOTE, press <b>Promote to live</b> to copy it into the live portfolio set. <b>HOLD</b> means the strategy isn't good enough yet — run a deeper discovery.</HelpStep>
        <p className="muted small">This is the backtest gate. Before risking real money there's a second, stricter <b>demo forward-test</b> gate in Autopilot.</p>
      </HelpPanel>

      <div className="ticket">
        <div className="ticket-row">
          <label>Symbol<SymbolSelect value={symbol} onChange={setSymbol} allowConfig style={{ width: 120 }} /></label>
          <label>Base TF<TimeframeSelect value={baseTf} onChange={setBaseTf} allowConfig style={{ width: 90 }} /></label>
        </div>
        <div className="btn-row">
          <button disabled={busy} onClick={check}>Check gate</button>
          <button className="primary" disabled={busy || !status} onClick={promote}>Promote to live</button>
        </div>
        {msg && <div className="banner info">{msg}</div>}
      </div>

      {status && (
        <>
          <div className="cards" style={{ marginTop: 14 }}>
            <div className="card"><div className="card-label">SYMBOL</div><div className="card-value">{status.symbol}</div></div>
            <div className="card"><div className="card-label">BASE TF</div><div className="card-value">{status.baseTf}</div></div>
            <div className="card"><div className="card-label">PORTFOLIO</div><div className="card-value">{status.portfolioSize}</div></div>
            <div className="card accent"><div className="card-label">DECISION</div><div className="card-value" style={{ color: eligible ? "#22c55e" : "#fca5a5", fontSize: 16 }}>{eligible ? "PROMOTE" : "HOLD"}</div></div>
          </div>
          {summary && <div className="banner info">{summary}</div>}
          {status.aggregate && (
            <>
              <h2>Aggregate metrics</h2>
              <table className="tbl">
                <tbody>
                  {Object.entries(status.aggregate).map(([k, v]) => (
                    <tr key={k}><td style={{ color: "#9ca3af" }}>{k}</td><td>{typeof v === "number" ? (v as number).toFixed(4) : String(v)}</td></tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </>
      )}
    </div>
  );
}
