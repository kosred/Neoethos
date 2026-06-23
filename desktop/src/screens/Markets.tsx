import { useCallback, useEffect, useState } from "react";
import Chart from "../components/Chart";
import {
  listSymbols,
  listTimeframes,
  localChart,
  brokerSymbols,
  brokerChart,
  type Candle,
} from "../api";
import { useSpotStream } from "../hooks";

const TF_ORDER = ["M1", "M3", "M5", "M15", "M30", "H1", "H4", "H12", "D1", "W1", "MN1"];
const BROKER_TFS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"];
const TF_SECONDS: Record<string, number> = {
  M1: 60, M3: 180, M5: 300, M15: 900, M30: 1800,
  H1: 3600, H4: 14400, H12: 43200, D1: 86400, W1: 604800, MN1: 2592000,
};

type Source = "local" | "broker";

export default function Markets() {
  const [source, setSource] = useState<Source>("local");
  const [symbols, setSymbols] = useState<string[]>([]);
  const [symbol, setSymbol] = useState("");
  const [timeframes, setTimeframes] = useState<string[]>([]);
  const [tf, setTf] = useState("");
  const [candles, setCandles] = useState<Candle[]>([]);
  const [liveBar, setLiveBar] = useState<Candle | null>(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const { ticks, connected } = useSpotStream();

  // symbols when source changes
  useEffect(() => {
    (async () => {
      try {
        const s =
          source === "local" ? await listSymbols() : (await brokerSymbols()).map((x) => x.name);
        setSymbols(s);
        setSymbol((p) => (s.includes(p) ? p : s.includes("EURUSD") ? "EURUSD" : s[0] ?? ""));
        setErr("");
      } catch (e) {
        setErr(String(e));
      }
    })();
  }, [source]);

  // timeframes when symbol/source changes
  useEffect(() => {
    if (!symbol) return;
    (async () => {
      if (source === "broker") {
        setTimeframes(BROKER_TFS);
        setTf((p) => (BROKER_TFS.includes(p) ? p : "H1"));
        return;
      }
      try {
        const tfs = (await listTimeframes(symbol)).sort(
          (a, b) => TF_ORDER.indexOf(a) - TF_ORDER.indexOf(b),
        );
        setTimeframes(tfs);
        setTf((p) => (tfs.includes(p) ? p : tfs.includes("H1") ? "H1" : tfs[0] ?? ""));
      } catch (e) {
        setErr(String(e));
      }
    })();
  }, [symbol, source]);

  const load = useCallback(async () => {
    if (!symbol || !tf) return;
    setLoading(true);
    setErr("");
    setLiveBar(null); // reset the forming candle for the new series
    try {
      const c = source === "local" ? await localChart(symbol, tf) : await brokerChart(symbol, tf);
      setCandles(c);
    } catch (e) {
      setErr(String(e));
      setCandles([]);
    } finally {
      setLoading(false);
    }
  }, [symbol, tf, source]);

  useEffect(() => {
    load();
  }, [load]);

  // Live forming candle: fold the current symbol's ticks into the last bar.
  // Only for broker data (live) — local history ends in the past.
  const tick = ticks[symbol];
  useEffect(() => {
    if (source !== "broker" || !tick || candles.length === 0) return;
    const tfSec = TF_SECONDS[tf] ?? 60;
    const price = tick.midPrice;
    const bucket = Math.floor(tick.brokerTimestampMs / 1000 / tfSec) * tfSec;
    const lastHist = candles[candles.length - 1].time;
    if (bucket < lastHist) return; // tick older than our history tail
    setLiveBar((prev) => {
      if (prev && prev.time === bucket) {
        return {
          ...prev,
          high: Math.max(prev.high, price),
          low: Math.min(prev.low, price),
          close: price,
        };
      }
      return { time: bucket, open: price, high: price, low: price, close: price };
    });
  }, [tick, symbol, tf, source, candles]);

  const tickerRows = Object.values(ticks)
    .sort((a, b) => a.symbolName.localeCompare(b.symbolName))
    .slice(0, 14);

  return (
    <div className="screen markets">
      <div className="markets-head">
        <h1>Markets</h1>
        <div className="controls">
          <span className={`stream-pill ${connected ? "on" : ""}`} title="Live tick stream">
            {connected ? "● LIVE" : "○ offline"}
          </span>
          <div className="seg">
            <button className={source === "local" ? "on" : ""} onClick={() => setSource("local")}>Local</button>
            <button className={source === "broker" ? "on" : ""} onClick={() => setSource("broker")}>Broker</button>
          </div>
          <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
            {symbols.map((s) => <option key={s}>{s}</option>)}
          </select>
          <select value={tf} onChange={(e) => setTf(e.target.value)}>
            {timeframes.map((t) => <option key={t}>{t}</option>)}
          </select>
          <button onClick={load} disabled={loading}>{loading ? "…" : "Reload"}</button>
        </div>
      </div>

      {tickerRows.length > 0 && (
        <div className="ticker">
          {tickerRows.map((t) => (
            <span key={t.symbolId} className="tick">
              <b>{t.symbolName}</b> {t.midPrice?.toFixed(5) ?? "—"}
            </span>
          ))}
        </div>
      )}

      {err && <div className="banner warn">{err.slice(0, 160)}</div>}

      <div className="chart-host">
        {candles.length === 0 && !loading ? (
          <div className="empty">No candles for {symbol} {tf}.</div>
        ) : (
          <Chart candles={candles} liveBar={source === "broker" ? liveBar : null} />
        )}
      </div>
      <div className="muted small">
        {candles.length} candles · source: {source} · {symbol} {tf}
        {source === "broker" && liveBar ? ` · live ${liveBar.close.toFixed(5)}` : ""}
      </div>
    </div>
  );
}
