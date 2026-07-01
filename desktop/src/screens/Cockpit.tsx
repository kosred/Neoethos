import { useCallback, useEffect, useRef, useState } from "react";
import Chart, { type Overlay } from "../components/Chart";
import PositionsTable from "../components/PositionsTable";
import {
  serverSymbols,
  brokerChart,
  brokerTimeframes,
  chartHistory,
  indicators as fetchIndicators,
  placeOrder,
  closePosition,
  refreshAccount,
  INDICATORS,
  OVERLAY_INDICATORS,
  type Candle,
  type BrokerSymbol,
  type ExecResult,
} from "../api";
import { useSpotStream, useAccountStream } from "../hooks";

const TF_SECONDS: Record<string, number> = {
  M1: 60, M3: 180, M5: 300, M15: 900, M30: 1800,
  H1: 3600, H4: 14400, H12: 43200, D1: 86400, W1: 604800, MN1: 2592000,
};
const IND_COLORS = ["#3b82f6", "#f59e0b", "#a855f7", "#22d3ee", "#ec4899"];
const fmt = (v: number | undefined, d = 2) =>
  v === undefined ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: d });

export default function Cockpit() {
  const { ticks, connected } = useSpotStream();
  const { snap } = useAccountStream();
  const [universe, setUniverse] = useState<BrokerSymbol[]>([]);
  const [symbol, setSymbol] = useState("EURUSD");
  const [tf, setTf] = useState("H1");
  const [tfs, setTfs] = useState<string[]>(["M1", "M5", "M15", "M30", "H1", "H4", "D1"]);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [liveBar, setLiveBar] = useState<Candle | null>(null);
  const [indicator, setIndicator] = useState("");
  const [overlays, setOverlays] = useState<Overlay[]>([]);
  const [filter, setFilter] = useState("");
  const histLoading = useRef(false);
  const noMoreHist = useRef(false);

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

  const load = useCallback(async () => {
    if (!symbol || !tf) return;
    setLiveBar(null);
    noMoreHist.current = false;
    try {
      setCandles(await brokerChart(symbol, tf));
    } catch {
      setCandles([]);
    }
  }, [symbol, tf]);
  useEffect(() => { load(); }, [load]);

  // live forming candle from this symbol's ticks
  const tick = ticks[symbol];
  useEffect(() => {
    if (!tick || candles.length === 0) return;
    const tfSec = TF_SECONDS[tf] ?? 60;
    const price = tick.midPrice;
    const bucket = Math.floor(tick.brokerTimestampMs / 1000 / tfSec) * tfSec;
    if (bucket < candles[candles.length - 1].time) return;
    setLiveBar((prev) =>
      prev && prev.time === bucket
        ? { ...prev, high: Math.max(prev.high, price), low: Math.min(prev.low, price), close: price }
        : { time: bucket, open: price, high: price, low: price, close: price },
    );
  }, [tick, tf, candles]);

  // indicator overlay
  useEffect(() => {
    if (!indicator || candles.length === 0) { setOverlays([]); return; }
    let alive = true;
    fetchIndicators(symbol, tf, indicator)
      .then((res) => {
        if (!alive) return;
        const onPrice = OVERLAY_INDICATORS.includes(indicator);
        setOverlays(res.lines.map((line, li) => {
          const start = candles.length - line.values.length;
          return {
            name: `${indicator}:${line.name}`,
            color: IND_COLORS[li % IND_COLORS.length],
            priceScaleId: onPrice ? "right" : "ind",
            data: line.values
              .map((v, i) => ({ time: candles[start + i]?.time ?? 0, value: v }))
              .filter((d) => d.time > 0 && Number.isFinite(d.value)),
          };
        }));
      })
      .catch(() => setOverlays([]));
    return () => { alive = false; };
  }, [indicator, symbol, tf, candles]);

  const loadOlder = useCallback(async () => {
    if (histLoading.current || noMoreHist.current || candles.length === 0) return;
    const oldest = candles[0];
    histLoading.current = true;
    try {
      const r = await chartHistory(symbol, tf, oldest.time * 1000, 500);
      const older: Candle[] = (r.candles ?? [])
        .filter((c) => c.tsMs != null)
        .map((c) => ({ time: (c.tsMs as number) / 1000, open: c.open, high: c.high, low: c.low, close: c.close }))
        .filter((c) => c.time < oldest.time);
      if (older.length === 0 || !r.hasMore) noMoreHist.current = true;
      if (older.length > 0)
        setCandles((prev) => {
          const seen = new Set(prev.map((c) => c.time));
          return [...older.filter((c) => !seen.has(c.time)), ...prev].sort((a, b) => a.time - b.time);
        });
    } catch { /* keep */ } finally { histLoading.current = false; }
  }, [symbol, tf, candles]);

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
              {INDICATORS.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
            <span className="spacer" />
            <span className={`stream-pill ${connected ? "on" : ""}`}>{connected ? "● live" : "○"}</span>
          </div>
          <div className="ck-chart-host">
            {candles.length === 0
              ? <div className="empty">Loading {symbol} {tf}… (needs broker connection)</div>
              : <Chart candles={candles} liveBar={liveBar} overlays={overlays} onReachStart={loadOlder} resetKey={`${symbol}|${tf}`} />}
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
