import { useEffect, useState } from "react";
import KChart, { KLINE_INDICATORS } from "../components/KChart";
import PositionsTable from "../components/PositionsTable";
import {
  serverSymbols,
  brokerTimeframes,
  placeOrder,
  closePosition,
  refreshAccount,
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

  useEffect(() => {
    serverSymbols().then((u) => setUniverse(u.symbols)).catch(() => {});
    brokerTimeframes().then((r) => r.timeframes.length && setTfs(r.timeframes)).catch(() => {});
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
    try { await closePosition(id, Math.round(vol)); refreshAccount().catch(() => {}); }
    catch (e) { setMsg(`Close error: ${e}`); } finally { setBusy(false); }
  };

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
            <label className="ck-field">Lots<input type="number" step="0.01" value={lots} onChange={(e) => setLots(Number(e.target.value))} /></label>
            <label className="ck-field">SL pips<input type="number" value={sl} onChange={(e) => setSl(e.target.value === "" ? "" : Number(e.target.value))} /></label>
            <label className="ck-field">TP pips<input type="number" value={tp} onChange={(e) => setTp(e.target.value === "" ? "" : Number(e.target.value))} /></label>
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

      {/* Trade Watch */}
      <div className="ck-tradewatch">
        <div className="ck-label">Positions {snap && snap.positions.length > 0 && <span className="badge live">{snap.positions.length}</span>}</div>
        <PositionsTable live={snap?.positions ?? []} currency={cur} onClose={onClose} busy={busy} />
      </div>
    </div>
  );
}
