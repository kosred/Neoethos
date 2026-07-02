import { useState } from "react";
import {
  brokerProfile,
  brokerVersion,
  ordersHistory,
  cashFlow,
  expectedMargin,
  journalStats,
  journalTrades,
} from "../api";
import { usePoll } from "../hooks";

const fmt = (v: unknown) =>
  typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(5)) : v == null ? "—" : String(v);

const fmt2 = (v: unknown) =>
  typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2)) : v == null ? "—" : String(v);

const num = (v: any, d = 2) => (typeof v === "number" && isFinite(v) ? v.toFixed(d) : "—");
const price = (v: any) => (typeof v === "number" && isFinite(v) ? v.toString() : "—");
const fmtTime = (ms: any) => (typeof ms === "number" && ms > 0 ? new Date(ms).toLocaleString() : "—");

// "winRatePct" / "max_drawdown_pct" → "WIN RATE PCT"
const label = (k: string) =>
  k.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/_/g, " ").toUpperCase();

export default function Account() {
  // One screen for everything account-shaped: the closed-trade journal
  // (day-to-day view) plus broker identity, order history, cash flow, margin.
  const [tab, setTab] = useState<"journal" | "broker">("journal");

  const { data: profile } = usePoll(brokerProfile, 0);
  const { data: version } = usePoll(brokerVersion, 0);
  const { data: hist, error: he } = usePoll(ordersHistory, 0);
  const { data: cash } = usePoll(cashFlow, 0);
  const { data: stats, error: e1 } = usePoll(journalStats, 0);
  const { data: trades, error: e2 } = usePoll(journalTrades, 0);

  const [symId, setSymId] = useState("1");
  const [vol, setVol] = useState("100000");
  const [margin, setMargin] = useState<any>(null);
  const [mErr, setMErr] = useState("");

  const calcMargin = async () => {
    setMErr("");
    try {
      setMargin(await expectedMargin(Number(symId), Number(vol)));
    } catch (e) {
      setMErr(String(e));
      setMargin(null);
    }
  };

  const orders: any[] = hist?.orders ?? [];
  const ocols = orders.length ? ["orderId", "side", "orderType", "orderStatus", "volumeLots", "limitPrice", "stopPrice"] : [];
  const entries: any[] = cash?.entries ?? [];

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
      <h1>Account &amp; Journal</h1>
      <p className="sub">Closed-trade log &amp; stats · broker identity · order history · cash flow · margin</p>

      <div className="settings-grid">
        <div className="kv"><span>cTID user</span><b>{profile?.userId ?? "—"}</b></div>
        <div className="kv"><span>Broker API</span><b>v{version?.version ?? "—"}</b></div>
        <div className="kv"><span>Account</span><b>{hist?.accountId ?? "—"}</b></div>
      </div>

      <div className="seg" style={{ margin: "12px 0" }}>
        <button className={tab === "journal" ? "on" : ""} onClick={() => setTab("journal")}>Journal</button>
        <button className={tab === "broker" ? "on" : ""} onClick={() => setTab("broker")}>Broker &amp; history</button>
      </div>

      {tab === "journal" ? (
        <>
          {(e1 || e2) && <div className="banner warn">{e1 || e2}</div>}

          {statEntries.length > 0 && (
            <div className="cards" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
              {statEntries.slice(0, 8).map(([k, v]) => (
                <div className="card" key={k}>
                  <div className="card-label">{label(k)}</div>
                  <div className="card-value" style={{ fontSize: 18 }}>{fmt2(v)}</div>
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
        </>
      ) : (
        <>
          <h2>Margin calculator</h2>
          <div className="ticket">
            <div className="ticket-row">
              <label>Symbol id<input value={symId} onChange={(e) => setSymId(e.target.value)} style={{ width: 80 }} /></label>
              <label>Volume (units)<input value={vol} onChange={(e) => setVol(e.target.value)} style={{ width: 120 }} /></label>
              <button className="primary" onClick={calcMargin}>Compute</button>
            </div>
            {mErr && <div className="banner warn">{mErr}</div>}
            {margin && (
              <table className="tbl">
                <tbody>
                  {Object.entries(margin).map(([k, v]) => (
                    <tr key={k}><td style={{ color: "#9ca3af" }}>{k}</td><td>{fmt(v)}</td></tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <h2>Order history ({orders.length})</h2>
          {he && <div className="banner warn">{he}</div>}
          {orders.length === 0 ? (
            <p className="muted">No order history.</p>
          ) : (
            <table className="tbl">
              <thead><tr>{ocols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>
                {orders.slice(0, 200).map((o, i) => (
                  <tr key={i}>{ocols.map((c) => <td key={c}>{fmt(o[c])}</td>)}</tr>
                ))}
              </tbody>
            </table>
          )}

          <h2>Cash flow ({entries.length})</h2>
          {entries.length === 0 ? (
            <p className="muted">No deposits / withdrawals / swaps recorded.</p>
          ) : (
            <table className="tbl">
              <thead><tr>{Object.keys(entries[0]).map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>
                {entries.slice(0, 200).map((e, i) => (
                  <tr key={i}>{Object.keys(entries[0]).map((c) => <td key={c}>{fmt(e[c])}</td>)}</tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  );
}
