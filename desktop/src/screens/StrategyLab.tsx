import { useState } from "react";
import { promotionStatus, promoteStrategy, type PromotionStatus } from "../api";
import { SymbolSelect, TimeframeSelect } from "../components/Select";
import { HelpPanel, HelpStep } from "../components/Help";

export default function StrategyLab() {
  const [symbol, setSymbol] = useState("");
  const [baseTf, setBaseTf] = useState("");
  const [status, setStatus] = useState<PromotionStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const check = async () => {
    if (busy) return;
    setBusy(true);
    setStatus(null);
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
    if (busy || status?.decision?.promoted !== true) return;
    const checked = status;
    setBusy(true);
    setMsg("Promoting to live…");
    try {
      const r = await promoteStrategy(checked.symbol, checked.baseTf);
      setMsg(`${r.promoted ? "✓" : "✗"} ${r.message} ${r.filesCopied ? `(${r.filesCopied} files)` : ""}`);
      setStatus(null);
      try {
        setStatus(await promotionStatus(checked.symbol, checked.baseTf));
      } catch (refreshError) {
        setMsg((message) => `${message}\nThe follow-up authority check failed: ${refreshError}`);
      }
    } catch (e) {
      setMsg(`Promote failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const decision = status?.decision;
  const eligible = decision?.promoted === true;
  const summary = decision?.summary ?? "";

  return (
    <div className="screen">
      <h1>Promotion readiness</h1>
      <p className="sub">Read-only authority check before any model can enter the live set</p>

      <HelpPanel id="strategylab">
        <p>This screen asks the backend whether one exact portfolio has enough sealed evidence to be copied into the live model set. Metric thresholds alone are not authorization.</p>
        <HelpStep n={1}>Pick the <b>Symbol</b> and <b>Base TF</b>, then run the read-only authority check.</HelpStep>
        <HelpStep n={2}>The promote action becomes available only when the server returns an explicit <b>PROMOTE</b> decision. A missing receipt, quote replay, composite scope, or broker-truth permit remains a hard block.</HelpStep>
        <p className="muted small">The current backend deliberately fails closed when exact composite promotion evidence is unavailable. That is a safety state, not a failed UI request.</p>
      </HelpPanel>

      <div className="banner warn">
        Promotion is not inferred from attractive backtest metrics. NeoEthos requires the exact receipt,
        composite validation authority and broker financial truth for this portfolio.
      </div>

      <div className="ticket">
        <fieldset className="ticket-row" disabled={busy} style={{ border: 0, padding: 0, margin: 0 }}>
          <label>Symbol<SymbolSelect value={symbol} onChange={(value) => { setSymbol(value); setStatus(null); setMsg(""); }} allowConfig style={{ width: 120 }} /></label>
          <label>Base TF<TimeframeSelect value={baseTf} onChange={(value) => { setBaseTf(value); setStatus(null); setMsg(""); }} allowConfig style={{ width: 90 }} /></label>
        </fieldset>
        <div className="btn-row">
          <button disabled={busy} onClick={check}>Check gate</button>
          <button className="primary" disabled={busy || !eligible} onClick={promote}>
            {eligible ? "Promote authorized portfolio" : "Promotion not authorized"}
          </button>
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
                    <tr key={k}><td style={{ color: "#9ca3af" }}>{k}</td><td>{typeof v === "number" ? v.toFixed(4) : String(v)}</td></tr>
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
