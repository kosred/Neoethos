import { useEffect, useState } from "react";
import {
  accountSnapshot,
  brokerStatus,
  type AccountSnapshot,
  type BrokerStatus,
} from "../api";

export default function Dashboard() {
  const [acct, setAcct] = useState<AccountSnapshot | null>(null);
  const [status, setStatus] = useState<BrokerStatus | null>(null);
  const [err, setErr] = useState<string>("");

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const s = await brokerStatus();
        if (alive) setStatus(s);
      } catch {
        /* ignore */
      }
      try {
        const a = await accountSnapshot();
        if (alive) {
          setAcct(a);
          setErr("");
        }
      } catch (e) {
        if (alive) setErr(String(e));
      }
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const cur = acct?.currency ?? "";
  const fmt = (v: number | undefined) =>
    v === undefined ? "—" : `${v.toLocaleString(undefined, { maximumFractionDigits: 2 })} ${cur}`;

  return (
    <div className="screen">
      <h1>Dashboard</h1>
      <p className="sub">Account equity · open positions · engine status</p>

      {acct && (
        <div className="acct-identity">
          <span className={`badge ${acct.live ? "live" : "demo"}`}>{acct.live ? "LIVE" : "DEMO"}</span>
          <b className="acct-label">{acct.label}</b>
          <span className="muted small">
            {acct.brokerName ?? ""} · login {acct.login ?? "—"} · {acct.accountType ?? ""} · #{acct.accountId}
          </span>
        </div>
      )}

      {!status?.configured && (
        <div className="banner warn">
          Broker not configured yet — go to <b>Settings</b> and add cTrader credentials, then
          Re-authenticate once.
        </div>
      )}
      {status?.configured && err && (
        <div className="banner warn">
          Connecting to broker… {status.hasToken ? "(auto-refreshing token)" : "(needs re-auth)"} —{" "}
          {err.slice(0, 120)}
        </div>
      )}

      <div className="cards">
        <Card label="BALANCE" value={fmt(acct?.balance)} />
        <Card label="EQUITY" value={fmt(acct?.equity)} accent />
        <Card label="UNREALIZED P/L" value={fmt(acct?.unrealizedPnl)} pnl={acct?.unrealizedPnl} />
        <Card label="OPEN POSITIONS" value={acct ? String(acct.openPositions) : "—"} />
      </div>

      <h2>Open positions</h2>
      {acct && acct.positions.length > 0 ? (
        <table className="tbl">
          <thead>
            <tr>
              <th>Side</th>
              <th>Symbol</th>
              <th>Volume</th>
              <th>Entry</th>
              <th>SL</th>
              <th>TP</th>
            </tr>
          </thead>
          <tbody>
            {acct.positions.map((p) => (
              <tr key={p.positionId}>
                <td className={p.side.toLowerCase().includes("buy") ? "buy" : "sell"}>{p.side}</td>
                <td>#{p.symbolId}</td>
                <td>{p.volume}</td>
                <td>{p.price ?? "—"}</td>
                <td>{p.stopLoss ?? "—"}</td>
                <td>{p.takeProfit ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="muted">No open positions.</p>
      )}
    </div>
  );
}

function Card({
  label,
  value,
  accent,
  pnl,
}: {
  label: string;
  value: string;
  accent?: boolean;
  pnl?: number;
}) {
  const cls = pnl === undefined ? "" : pnl >= 0 ? "buy" : "sell";
  return (
    <div className={`card${accent ? " accent" : ""}`}>
      <div className="card-label">{label}</div>
      <div className={`card-value ${cls}`}>{value}</div>
    </div>
  );
}
