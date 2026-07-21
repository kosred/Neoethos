import { useEffect, useState } from "react";
import KChart, { KLINE_INDICATORS } from "../components/KChart";
import PositionsTable from "../components/PositionsTable";
import {
  serverSymbols,
  brokerTimeframes,
  placeOrder,
  closePosition,
  refreshAccount,
  amendProtection,
  type BrokerSymbol,
  type ExecResult,
} from "../api";
import { useSpotStream, useAccountStream } from "../hooks";

const fmt = (v: number | undefined, d = 2) =>
  v === undefined ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: d });

export default function Cockpit() {
  const { ticks, connected } = useSpotStream();
  const { snap } = useAccountStream();
  const [universe, setUniverse] = useState<BrokerSymbol[]>([]);
  const [symbol, setSymbol] = useState("EURUSD");
  const [tf, setTf] = useState("H1");
  const [tfs, setTfs] = useState<string[]>(["M1", "M5", "M15", "M30", "H1", "H4", "D1"]);
  const [indicator, setIndicator] = useState("");
  const [filter, setFilter] = useState("");

  // order ticket
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [lots, setLots] = useState(0.01);
  const [sl, setSl] = useState<number | "">(20);
  const [tp, setTp] = useState<number | "">(40);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  // Modify-protection editor (merged from the old Positions screen): click
  // Edit on an open position → set SL/TP as PRICE LEVELS (breakeven, trailing).
  const [editId, setEditId] = useState<number | null>(null);
  const [editSl, setEditSl] = useState<number | "">("");
  const [editTp, setEditTp] = useState<number | "">("");
  const [editTrail, setEditTrail] = useState(false);
  const positions = snap?.positions ?? [];

  useEffect(() => {
    serverSymbols().then((u) => setUniverse(u.symbols)).catch(() => {});
    brokerTimeframes().then((r) => r.timeframes.length && setTfs(r.timeframes)).catch(() => {});
  }, []);

  // The account stream only delivers snapshots when the server PUSHES one.
  // Kick a refresh on mount + every 5s so balance/equity and the positions
  // strip are live even if nothing else triggers a push (this was the old
  // "No open positions" bug when landing directly on the cockpit).
  useEffect(() => {
    const kick = () => refreshAccount().catch(() => {});
    kick();
    const id = setInterval(kick, 5000);
    return () => clearInterval(id);
  }, []);

  const place = async () => {
    setBusy(true); setMsg("");
    try {
      const r: ExecResult = await placeOrder(symbol, side, lots, sl === "" ? undefined : Number(sl), tp === "" ? undefined : Number(tp));
      setMsg(`${r.status}${r.positionId ? ` · #${r.positionId}` : ""}${r.message ? ` · ${r.message}` : ""}`);
      refreshAccount().catch(() => {});
    } catch (e) { setMsg(`Error: ${e}`); } finally { setBusy(false); }
  };
  const onClose = async (id: number, vol: number) => {
    setBusy(true);
    try {
      await closePosition(id, Math.round(vol));
      if (editId === id) setEditId(null);
      refreshAccount().catch(() => {});
    }
    catch (e) { setMsg(`Close error: ${e}`); } finally { setBusy(false); }
  };

  // Selecting a position pre-fills the inline editor with its current stops.
  const onEdit = (positionId: number) => {
    const p = positions.find((x) => x.positionId === positionId);
    if (!p) return;
    setEditId(positionId);
    setEditSl(p.stopLoss ?? "");
    setEditTp(p.takeProfit ?? "");
    setEditTrail(false);
  };

  const saveProtection = async () => {
    if (editId == null) return;
    setBusy(true);
    setMsg("Updating SL/TP…");
    try {
      const r: any = await amendProtection(
        editId,
        editSl === "" ? null : Number(editSl),
        editTp === "" ? null : Number(editTp),
        editTrail,
      );
      setMsg(`✓ Protection updated${r?.message ? ` · ${r.message}` : ""}`);
      setEditId(null);
      refreshAccount().catch(() => {});
    } catch (e) {
      setMsg(`SL/TP update failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const editPos = editId != null ? positions.find((p) => p.positionId === editId) : undefined;

  const groups: Record<string, BrokerSymbol[]> = {};
  for (const s of universe) {
    if (filter && !s.symbolName.toUpperCase().includes(filter.toUpperCase())) continue;
    (groups[s.assetClass || "Other"] ??= []).push(s);
  }
  const cur = snap?.currency ?? "";
  const pnl = snap ? snap.equity - snap.balance : undefined;

  return (
    <div className="cockpit">
      <div className="ck-grid">
        {/* Market Watch */}
        <div className="ck-watch">
          <input className="ck-filter" placeholder="filter…" value={filter} onChange={(e) => setFilter(e.target.value)} />
          <div className="ck-watch-list">
            {Object.keys(groups).sort().map((cls) => (
              <div key={cls}>
                <div className="ck-group">{cls}</div>
                {groups[cls].map((s) => {
                  const t = ticks[s.symbolName];
                  return (
                    <button key={s.symbolId} className={`ck-sym${symbol === s.symbolName ? " on" : ""}`} onClick={() => setSymbol(s.symbolName)}>
                      <span>{s.symbolName}</span>
                      <span className="mono">{t ? t.midPrice?.toFixed(5) : "—"}</span>
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        {/* Chart */}
        <div className="ck-chart">
          <div className="ck-chart-bar">
            <b>{symbol}</b>
            <select value={tf} onChange={(e) => setTf(e.target.value)}>
              {tfs.map((t) => <option key={t}>{t}</option>)}
            </select>
            <select value={indicator} onChange={(e) => setIndicator(e.target.value)}>
              <option value="">indicator</option>
              {KLINE_INDICATORS.map((n) => <option key={n.v} value={n.v}>{n.l}</option>)}
            </select>
            <span className="spacer" />
            <span className={`stream-pill ${connected ? "on" : ""}`}>{connected ? "● live" : "○"}</span>
          </div>
          <div className="ck-chart-host">
            {!symbol || !tf
              ? <div className="empty">Select a symbol…</div>
              : <KChart symbol={symbol} timeframe={tf} indicator={indicator} liveTick={ticks[symbol] ?? null} />}
          </div>
        </div>

        {/* Order + Account */}
        <div className="ck-side">
          <div className="ck-order">
            <div className="ck-label">Order</div>
            <div className="seg" style={{ width: "100%" }}>
              <button className={side === "buy" ? "on buy" : ""} style={{ flex: 1 }} onClick={() => setSide("buy")}>BUY</button>
              <button className={side === "sell" ? "on sell" : ""} style={{ flex: 1 }} onClick={() => setSide("sell")}>SELL</button>
            </div>
            {/* min guards: without one the spinner arrows step below zero, and
                this ticket places REAL orders on the live account. */}
            <label className="ck-field">Lots<input type="number" min="0.01" step="0.01" value={lots} onChange={(e) => setLots(Math.max(0, Number(e.target.value)))} /></label>
            <label className="ck-field">SL pips<input type="number" min="0" value={sl} onChange={(e) => setSl(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))} /></label>
            <label className="ck-field">TP pips<input type="number" min="0" value={tp} onChange={(e) => setTp(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))} /></label>
            <button className="primary" style={{ width: "100%", marginTop: 6 }} disabled={busy} onClick={place}>
              {busy ? "…" : `${side.toUpperCase()} ${symbol} ${lots}`}
            </button>
            {msg && <div className="ck-msg">{msg}</div>}
          </div>
          <div className="ck-account">
            <div className="ck-label">Account</div>
            <div className="ck-kv"><span>Balance</span><b className="mono">{fmt(snap?.balance)} {cur}</b></div>
            <div className="ck-kv"><span>Equity</span><b className="mono">{fmt(snap?.equity)} {cur}</b></div>
            <div className="ck-kv"><span>Used margin</span><b className="mono">{fmt(snap?.usedMargin)} {cur}</b></div>
            <div className="ck-kv"><span>Free margin</span><b className="mono">{fmt(snap?.freeMargin)} {cur}</b></div>
            <div className="ck-kv"><span>P/L</span><b className={`mono ${pnl !== undefined && pnl < 0 ? "sell" : "buy"}`}>{fmt(pnl)} {cur}</b></div>
          </div>
        </div>
      </div>

      {/* Trade Watch — open positions with Edit (SL/TP as price levels) + Close */}
      <div className="ck-tradewatch">
        <div className="ck-label">Positions {positions.length > 0 && <span className="badge live">{positions.length}</span>}</div>
        {editPos && (
          <div className="ticket" style={{ marginBottom: 8 }}>
            <div className="ticket-row">
              <b style={{ alignSelf: "center" }}>{editPos.symbol} {editPos.side} #{editPos.positionId}</b>
              <label>SL price<input type="number" min="0" step="0.00001" value={editSl} onChange={(e) => setEditSl(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))} /></label>
              <label>TP price<input type="number" min="0" step="0.00001" value={editTp} onChange={(e) => setEditTp(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))} /></label>
              <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
                <input type="checkbox" checked={editTrail} onChange={(e) => setEditTrail(e.target.checked)} /> Trailing
              </label>
              <button className="primary" disabled={busy} onClick={saveProtection}>Update SL/TP</button>
              <button disabled={busy} onClick={() => setEditId(null)}>Cancel</button>
            </div>
          </div>
        )}
        <PositionsTable live={positions} currency={cur} onClose={onClose} onEdit={onEdit} busy={busy} />
      </div>
    </div>
  );
}
