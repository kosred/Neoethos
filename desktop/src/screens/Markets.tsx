import { useEffect, useState } from "react";
import KChart, { KLINE_INDICATORS } from "../components/KChart";
import {
  serverSymbols,
  brokerTimeframes,
  getWatchlist,
  setWatchlist,
  type BrokerSymbol,
} from "../api";
import { useSpotStream } from "../hooks";

const BROKER_TFS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"];

export default function Markets() {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [symbol, setSymbol] = useState("");
  const [timeframes, setTimeframes] = useState<string[]>([]);
  const [tf, setTf] = useState("H1");
  const [brokerSyms, setBrokerSyms] = useState<BrokerSymbol[]>([]);
  const [err, setErr] = useState("");
  const [subMsg, setSubMsg] = useState("");
  const [indicator, setIndicator] = useState("");
  const { ticks, connected } = useSpotStream();

  // broker symbol universe
  useEffect(() => {
    (async () => {
      try {
        const u = await serverSymbols();
        setBrokerSyms(u.symbols);
        const s = u.symbols.map((x) => x.symbolName);
        setSymbols(s);
        setSymbol((p) => (s.includes(p) ? p : s.includes("EURUSD") ? "EURUSD" : s[0] ?? ""));
        setErr("");
      } catch (e) {
        setErr(String(e));
      }
    })();
  }, []);

  // broker timeframes
  useEffect(() => {
    (async () => {
      try {
        const tfs = (await brokerTimeframes()).timeframes;
        setTimeframes(tfs.length ? tfs : BROKER_TFS);
        setTf((p) => (tfs.includes(p) ? p : tfs.includes("H1") ? "H1" : tfs[0] ?? "H1"));
      } catch {
        setTimeframes(BROKER_TFS);
        setTf((p) => (BROKER_TFS.includes(p) ? p : "H1"));
      }
    })();
  }, []);

  // Subscribe the selected symbol to the live tick stream (union with watchlist)
  // so its forming candle ticks.
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
          <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
            {brokerSyms.length > 0
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
          <select value={indicator} onChange={(e) => setIndicator(e.target.value)} title="Technical indicator">
            <option value="">— indicator —</option>
            {KLINE_INDICATORS.map((n) => <option key={n.v} value={n.v}>{n.l}</option>)}
          </select>
          {symbol && !ticks[symbol] && (
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
        {!symbol || !tf ? (
          <div className="empty">Select a symbol and timeframe.</div>
        ) : (
          <KChart symbol={symbol} timeframe={tf} indicator={indicator} liveTick={ticks[symbol] ?? null} />
        )}
      </div>
      <div className="muted small">
        {symbol} {tf} · drag the toolbar tools to draw · scroll to zoom, drag to pan
      </div>
    </div>
  );
}
