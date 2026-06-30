import { useState } from "react";
import { strategyList, strategyReport, type StrategyEntry, type StrategyReport as Report } from "../api";
import { usePoll } from "../hooks";

const eur = (v: number) => `€${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
const Badge = ({ ok, label }: { ok: boolean | null; label: string }) => (
  <span className={`badge ${ok ? "demo" : "live"}`} title={label}>{ok ? "✓" : "✗"} {label}</span>
);

export default function StrategyReport() {
  const { data, error } = usePoll(strategyList, 0);
  const [rep, setRep] = useState<Report | null>(null);
  const [busy, setBusy] = useState(false);

  const open = async (s: StrategyEntry) => {
    setBusy(true);
    try {
      setRep(await strategyReport(s.dir, s.base));
    } catch {
      setRep(null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <h1>Strategy Report</h1>
      <p className="sub">Monthly journal · €1000 growth · validation verdict · honest flags — from the stored backtests</p>
      {error && <div className="banner warn">{error}</div>}

      <table className="tbl">
        <thead>
          <tr><th>Mode</th><th>Symbol</th><th>TF</th><th>Trades</th><th>Win%</th><th>CAGR%</th><th>maxDD%</th><th>€1k→</th><th>Validation</th><th></th></tr>
        </thead>
        <tbody>
          {(data?.strategies ?? []).map((s) => (
            <tr key={s.dir + s.base} className={rep?.base === s.base && rep?.dir === s.dir ? "row-sel" : ""}>
              <td><span className={`badge ${s.mode === "risky" ? "live" : "demo"}`}>{s.mode}</span></td>
              <td><b>{s.symbol}</b></td>
              <td>{s.timeframe}</td>
              <td>{s.trades}</td>
              <td>{s.winRate != null ? (s.winRate * 100).toFixed(1) : "—"}</td>
              <td className={s.cagrPct >= 0 ? "buy" : "sell"}>{Math.abs(s.cagrPct) > 1000 ? "🚩" : s.cagrPct.toFixed(1)}</td>
              <td>{s.maxDdPct.toFixed(1)}</td>
              <td>{Math.abs(s.cagrPct) > 1000 ? "—" : eur(s.finalFrom1000)}</td>
              <td>
                <span className={`badge ${s.cpcvPassed ? "demo" : "live"}`} title="CPCV">C{s.cpcvPassed ? "✓" : "✗"}</span>{" "}
                <span className={`badge ${s.walkforwardPassed ? "demo" : "live"}`} title="Walkforward (OOS)">W{s.walkforwardPassed ? "✓" : "✗"}</span>
                {s.flags.length > 0 && <span className="sell small"> 🚩{s.flags.length}</span>}
              </td>
              <td><button disabled={busy} onClick={() => open(s)}>Report</button></td>
            </tr>
          ))}
        </tbody>
      </table>

      {rep && (
        <>
          <h2>{rep.symbol} {rep.timeframe} <span className={`badge ${rep.mode === "risky" ? "live" : "demo"}`}>{rep.mode}</span></h2>
          <p className="muted small">{rep.spanStart} → {rep.spanEnd} · {rep.years}y · {rep.trades} trades</p>

          {(() => {
            const oos = !!rep.cpcvPassed && !!rep.walkforwardPassed;
            const full = oos && !!rep.validationComplete;
            const txt = full
              ? "✅ FULLY VALIDATED — passed CPCV + Walkforward, evidence complete"
              : oos
                ? "✅ PASSED out-of-sample (CPCV + Walkforward) — full-evidence check still pending"
                : "⚠️ NOT validated out-of-sample — not safe to trade live";
            return (
              <div className={`banner ${oos ? "info" : "warn"}`} style={{ fontWeight: 600 }}>
                {txt}
              </div>
            );
          })()}

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", margin: "8px 0" }}>
            <Badge ok={rep.cpcvPassed} label="CPCV" />
            <Badge ok={rep.walkforwardPassed} label="Walkforward (out-of-sample)" />
            <Badge ok={rep.validationComplete} label="Full evidence" />
          </div>
          {rep.flags.map((f, i) => <div className="banner warn" key={i}>🚩 {f}</div>)}

          <div className="cards">
            <div className="card"><div className="card-label">CAGR</div><div className="card-value">{Math.abs(rep.cagrPct) > 1000 ? "🚩 bug" : `${rep.cagrPct.toFixed(1)}%`}</div></div>
            <div className="card accent"><div className="card-label">€1000 →</div><div className="card-value" style={{ fontSize: 18 }}>{Math.abs(rep.cagrPct) > 1000 ? "—" : eur(rep.finalFrom1000)}</div></div>
            <div className="card"><div className="card-label">MAX DD</div><div className="card-value">{rep.maxDdPct.toFixed(1)}%</div></div>
            <div className="card"><div className="card-label">WIN RATE</div><div className="card-value">{rep.winRate != null ? `${(rep.winRate * 100).toFixed(1)}%` : "—"}</div></div>
          </div>

          {Math.abs(rep.cagrPct) <= 1000 && rep.yearly.length > 0 && (
            <>
              <h2>Year-end balance (from €1000)</h2>
              <div className="ticker" style={{ flexWrap: "wrap" }}>
                {rep.yearly.map((y) => <span className="tick" key={y.month}>{y.month}: <b>{eur(y.balance)}</b></span>)}
              </div>
              <h2>Monthly journal</h2>
              <table className="tbl">
                <thead><tr><th>Month</th><th>Return%</th><th>Balance</th><th>Trades</th></tr></thead>
                <tbody>
                  {rep.monthly.slice(-24).reverse().map((m) => (
                    <tr key={m.month}>
                      <td>{m.month}</td>
                      <td className={m.returnPct >= 0 ? "buy" : "sell"}>{m.returnPct >= 0 ? "+" : ""}{m.returnPct.toFixed(1)}</td>
                      <td className="mono">{eur(m.balance)}</td>
                      <td>{m.trades}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted small">Showing last 24 months of {rep.monthly.length}.</p>
            </>
          )}
        </>
      )}
    </div>
  );
}
