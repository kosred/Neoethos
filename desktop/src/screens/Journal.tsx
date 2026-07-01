import { journalStats, journalTrades } from "../api";
import { usePoll } from "../hooks";

const fmt = (v: unknown) =>
  typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2)) : v == null ? "—" : String(v);

const num = (v: any, d = 2) => (typeof v === "number" && isFinite(v) ? v.toFixed(d) : "—");
const price = (v: any) => (typeof v === "number" && isFinite(v) ? v.toString() : "—");
const fmtTime = (ms: any) => (typeof ms === "number" && ms > 0 ? new Date(ms).toLocaleString() : "—");

// "winRatePct" / "max_drawdown_pct" → "WIN RATE PCT"
const label = (k: string) =>
  k.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/_/g, " ").toUpperCase();

export default function Journal() {
  const { data: stats, error: e1 } = usePoll(journalStats, 0);
  const { data: trades, error: e2 } = usePoll(journalTrades, 0);

  const statEntries =
    stats && typeof stats === "object"
      ? Object.entries(stats).filter(([, v]) => typeof v !== "object" || v === null)
      : [];
  const tradeRows: any[] = Array.isArray(trades) ? trades : (trades?.trades ?? []);
  // newest first
  const rows = [...tradeRows].sort(
    (a, b) => (b.exitTsMs ?? b.recordedAtUnixMs ?? 0) - (a.exitTsMs ?? a.recordedAtUnixMs ?? 0),
  );

  return (
    <div className="screen">
      <h1>Trade Journal</h1>
      <p className="sub">Closed-trade log &amp; computed stats (MyFxbook-style)</p>
      {(e1 || e2) && <div className="banner warn">{e1 || e2}</div>}

      {statEntries.length > 0 && (
        <div className="cards" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
          {statEntries.slice(0, 8).map(([k, v]) => (
            <div className="card" key={k}>
              <div className="card-label">{label(k)}</div>
              <div className="card-value" style={{ fontSize: 18 }}>{fmt(v)}</div>
            </div>
          ))}
        </div>
      )}

      <h2>Trades ({rows.length})</h2>
      {rows.length === 0 ? (
        <p className="muted">No closed trades recorded yet.</p>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>Closed</th>
              <th>Symbol</th>
              <th>Side</th>
              <th>Lots</th>
              <th>Entry</th>
              <th>Exit</th>
              <th>Costs</th>
              <th>Net P/L</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 300).map((r, i) => {
              const net = Number(r.netProfit ?? 0);
              const costs = Number(r.commission ?? 0) + Number(r.swap ?? 0);
              const buy = String(r.side ?? "").toUpperCase().includes("BUY");
              const cls = net >= 0 ? "buy" : "sell";
              return (
                <tr key={r.positionId ?? i}>
                  <td className="muted">{fmtTime(r.exitTsMs ?? r.recordedAtUnixMs)}</td>
                  <td><b>{r.symbol ?? "?"}</b></td>
                  <td className={buy ? "buy" : "sell"}>{r.side ?? "—"}</td>
                  <td>{num(r.lots)}</td>
                  <td>{price(r.entryPrice)}</td>
                  <td>{price(r.exitPrice)}</td>
                  <td className="muted">{num(costs)}</td>
                  <td className={cls}><b>{net >= 0 ? "+" : ""}{num(net)}</b></td>
                  <td>{net > 0 ? "✓ win" : net < 0 ? "✗ loss" : "— BE"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
