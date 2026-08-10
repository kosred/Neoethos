import { useEffect, useState } from "react";
import {
  accountSnapshot,
  brokerStatus,
  type AccountSnapshot,
  type BrokerStatus,
} from "../api";
import { useAccountStream } from "../hooks";
import PositionsTable from "../components/PositionsTable";

export default function Dashboard() {
  const [acct, setAcct] = useState<AccountSnapshot | null>(null);
  const [status, setStatus] = useState<BrokerStatus | null>(null);
  const [err, setErr] = useState<string>("");
  const { snap, connected } = useAccountStream(); // live balance/equity/PnL (push)

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
    const id = setInterval(tick, 8000); // identity + positions table (slow); numbers stream live
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  // Live numbers prefer the SSE stream; fall back to the Tauri snapshot.
  const balance = snap?.balance ?? acct?.balance;
  const equity = snap?.equity ?? acct?.equity;
  const pnl = snap ? snap.equity - snap.balance : acct?.unrealizedPnl;
  const openCount = snap?.positions?.length ?? acct?.openPositions;
  const cur = snap?.currency ?? acct?.currency ?? "";
  const fmt = (v: number | undefined) =>
    v === undefined ? "—" : `${v.toLocaleString(undefined, { maximumFractionDigits: 2 })} ${cur}`;

  return (
    <div className="screen">
      <h1>Dashboard <span className={`stream-pill ${connected ? "on" : ""}`}>{connected ? "● LIVE" : "○ polling"}</span></h1>
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

      {/* Audit #119: this banner used to send the operator to a credentials
          form that had never existed anywhere in `desktop/src`. The form now
          exists — Settings → "cTrader API credentials" — so the instruction is
          true for the first time. Keep the two steps in the order the backend
          requires: credentials first, then the one-time OAuth. */}
      {!status?.configured && (
        <div className="banner warn">
          Broker not configured yet — open <b>Settings → cTrader API credentials</b>, paste the
          Client ID and Secret from your cTrader Open API application, save, then press{" "}
          <b>Authenticate cTrader</b> once.
        </div>
      )}
      {status?.configured && err && (
        <div className="banner warn">
          Connecting to broker… {status.hasToken ? "(auto-refreshing token)" : "(needs re-auth)"} —{" "}
          {err.slice(0, 120)}
        </div>
      )}

      <div className="cards">
        <Card label="BALANCE" value={fmt(balance)} />
        <Card label="EQUITY" value={fmt(equity)} accent />
        <Card label="UNREALIZED P/L" value={fmt(pnl)} pnl={pnl} />
        <Card label="OPEN POSITIONS" value={openCount !== undefined ? String(openCount) : "—"} />
      </div>

      <h2>Open positions</h2>
      <PositionsTable live={snap?.positions ?? []} currency={cur} />
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
