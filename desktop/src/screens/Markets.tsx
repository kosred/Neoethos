import { useCallback, useEffect, useRef, useState } from "react";
import Chart, { type Overlay } from "../components/Chart";
import {
  listSymbols,
  listTimeframes,
  localChart,
  serverSymbols,
  brokerChart,
  brokerTimeframes,
  indicators as fetchIndicators,
  chartHistory,
  getWatchlist,
  setWatchlist,
  INDICATORS,
  OVERLAY_INDICATORS,
  type Candle,
  type BrokerSymbol,
} from "../api";
import { useSpotStream } from "../hooks";

const IND_COLORS = ["#3b82f6", "#f59e0b", "#a855f7", "#22d3ee", "#ec4899"];

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
  const [brokerSyms, setBrokerSyms] = useState<BrokerSymbol[]>([]);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [subMsg, setSubMsg] = useState("");
  const [indicator, setIndicator] = useState("");
  const [overlays, setOverlays] = useState<Overlay[]>([]);
  const histLoading = useRef(false);
  const noMoreHist = useRef(false);
  const { ticks, connected } = useSpotStream();

  // symbols when source changes
  useEffect(() => {
    (async () => {
      try {
        if (source === "local") {
          const s = await listSymbols();
          setSymbols(s);
          setSymbol((p) => (s.includes(p) ? p : s.includes("EURUSD") ? "EURUSD" : s[0] ?? ""));
        } else {
          const u = await serverSymbols();
          setBrokerSyms(u.symbols);
          const s = u.symbols.map((x) => x.symbolName);
          setSymbols(s);
          setSymbol((p) => (s.includes(p) ? p : s.includes("EURUSD") ? "EURUSD" : s[0] ?? ""));
        }
        setErr("");
      } catch (e) {
        setErr(String(e));
      }
    })();
  }, [source]);

  // Subscribe the selected broker symbol to the live stream (union with the
  // current watchlist) so its forming candle ticks — for ANY of the dozens.
  const streamThis = async () => {
    setSubMsg(`Subscribing ${symbol}…`);
    try {
      const w = await getWatchlist();
      const cur: string[] = Array.isArray(w) ? w : (w?.symbols ?? []);
      const next = Array.from(new Set([...cur.map((x) => x.toUpperCase()), symbol.toUpperCase()]));
      await setWatchlist(next);
      setSubMsg(`✓ ${symbol} subscribed — live candle within ~5s.`);
    } catch (e) {
      setSubMsg(`Subscribe failed: ${e}`);
    }
  };

  // timeframes when symbol/source changes
  useEffect(() => {
    if (!symbol) return;
    (async () => {
      if (source === "broker") {
        try {
          const tfs = (await brokerTimeframes()).timeframes;
          setTimeframes(tfs.length ? tfs : BROKER_TFS);
          setTf((p) => (tfs.includes(p) ? p : tfs.includes("H1") ? "H1" : tfs[0] ?? "H1"));
        } catch {
          setTimeframes(BROKER_TFS);
          setTf((p) => (BROKER_TFS.includes(p) ? p : "H1"));
        }
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
    noMoreHist.current = false; // allow scroll-back again for the new series
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

  // Indicator overlays: fetch the chosen indicator, align each line's values to
  // the tail of the loaded candles, draw on the price scale (sma/ema/bbands/vwap)
  // or a separate scale (oscillators).
  useEffect(() => {
    if (!indicator || !symbol || !tf || candles.length === 0) {
      setOverlays([]);
      return;
    }
    let alive = true;
    fetchIndicators(symbol, tf, indicator)
      .then((res) => {
        if (!alive) return;
        const onPrice = OVERLAY_INDICATORS.includes(indicator);
        const ov: Overlay[] = res.lines.map((line, li) => {
          const vals = line.values;
          const start = candles.length - vals.length;
          const data = vals
            .map((v, i) => ({ time: candles[start + i]?.time ?? 0, value: v }))
            .filter((d) => d.time > 0 && Number.isFinite(d.value));
          return {
            name: `${indicator}:${line.name}`,
            color: IND_COLORS[li % IND_COLORS.length],
            priceScaleId: onPrice ? "right" : "ind",
            data,
          };
        });
        setOverlays(ov);
      })
      .catch(() => setOverlays([]));
    return () => {
      alive = false;
    };
  }, [indicator, symbol, tf, candles]);

  // Scroll-back: when the chart pans near the left edge, prepend older bars.
  const loadOlder = useCallback(async () => {
    if (histLoading.current || noMoreHist.current || candles.length === 0) return;
    const oldest = candles[0];
    const beforeMs = oldest.time * 1000;
    histLoading.current = true;
    try {
      const r = await chartHistory(symbol, tf, beforeMs, 500);
      const older: Candle[] = (r.candles ?? [])
        .filter((c) => c.tsMs != null)
        .map((c) => ({ time: (c.tsMs as number) / 1000, open: c.open, high: c.high, low: c.low, close: c.close }))
        .filter((c) => c.time < oldest.time);
      if (older.length === 0 || !r.hasMore) noMoreHist.current = true;
      if (older.length > 0) {
        setCandles((prev) => {
          const seen = new Set(prev.map((c) => c.time));
          const merged = [...older.filter((c) => !seen.has(c.time)), ...prev];
          merged.sort((a, b) => a.time - b.time);
          return merged;
        });
      }
    } catch {
      /* ignore — keep what we have */
    } finally {
      histLoading.current = false;
    }
  }, [symbol, tf, candles]);

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
            {source === "broker" && brokerSyms.length > 0
              ? Object.entries(
                  brokerSyms.reduce<Record<string, BrokerSymbol[]>>((g, s) => {
                    (g[s.assetClass || "Other"] ??= []).push(s);
                    return g;
                  }, {}),
                )
                  .sort()
                  .map(([cls, syms]) => (
                    <optgroup key={cls} label={cls}>
                      {syms
                        .slice()
                        .sort((a, b) => a.symbolName.localeCompare(b.symbolName))
                        .map((s) => <option key={s.symbolId}>{s.symbolName}</option>)}
                    </optgroup>
                  ))
              : symbols.map((s) => <option key={s}>{s}</option>)}
          </select>
          <select value={tf} onChange={(e) => setTf(e.target.value)}>
            {timeframes.map((t) => <option key={t}>{t}</option>)}
          </select>
          <select value={indicator} onChange={(e) => setIndicator(e.target.value)} title="Indicator overlay">
            <option value="">— indicator —</option>
            {INDICATORS.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
          <button onClick={load} disabled={loading}>{loading ? "…" : "Reload"}</button>
          {source === "broker" && symbol && !ticks[symbol] && (
            <button onClick={streamThis} title="Subscribe this symbol to the live stream">📡 Stream</button>
          )}
        </div>
      </div>
      {subMsg && <div className="banner info">{subMsg}</div>}

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
          <Chart
            candles={candles}
            liveBar={source === "broker" ? liveBar : null}
            overlays={overlays}
            onReachStart={loadOlder}
          />
        )}
      </div>
      <div className="muted small">
        {candles.length} candles · source: {source} · {symbol} {tf}
        {source === "broker" && liveBar ? ` · live ${liveBar.close.toFixed(5)}` : ""}
      </div>
    </div>
  );
}
