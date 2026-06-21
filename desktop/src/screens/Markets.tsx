import { useCallback, useEffect, useState } from "react";
import Chart from "../components/Chart";
import {
  listSymbols,
  listTimeframes,
  localChart,
  brokerSymbols,
  brokerChart,
  spotPrices,
  type Candle,
  type SpotPrice,
} from "../api";

const TF_ORDER = ["M1", "M3", "M5", "M15", "M30", "H1", "H4", "H12", "D1", "W1", "MN1"];
const BROKER_TFS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"];

type Source = "local" | "broker";

export default function Markets() {
  const [source, setSource] = useState<Source>("local");
  const [symbols, setSymbols] = useState<string[]>([]);
  const [symbol, setSymbol] = useState("");
  const [timeframes, setTimeframes] = useState<string[]>([]);
  const [tf, setTf] = useState("");
  const [candles, setCandles] = useState<Candle[]>([]);
  const [spots, setSpots] = useState<SpotPrice[]>([]);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  // symbols when source changes
  useEffect(() => {
    (async () => {
      try {
        if (source === "local") {
          const s = await listSymbols();
          setSymbols(s);
          setSymbol((p) => (s.includes(p) ? p : s.includes("EURUSD") ? "EURUSD" : s[0] ?? ""));
        } else {
          const s = (await brokerSymbols()).map((x) => x.name);
          setSymbols(s);
          setSymbol((p) => (s.includes(p) ? p : s.includes("EURUSD") ? "EURUSD" : s[0] ?? ""));
        }
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

  // live spot prices poll
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const s = await spotPrices();
        if (alive) setSpots(s);
      } catch {
        /* ignore */
      }
    };
    tick();
    const id = setInterval(tick, 1500);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="screen markets">
      <div className="markets-head">
        <h1>Markets</h1>
        <div className="controls">
          <div className="seg">
            <button className={source === "local" ? "on" : ""} onClick={() => setSource("local")}>
              Local
            </button>
            <button className={source === "broker" ? "on" : ""} onClick={() => setSource("broker")}>
              Broker
            </button>
          </div>
          <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
            {symbols.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
          <select value={tf} onChange={(e) => setTf(e.target.value)}>
            {timeframes.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
          <button onClick={load} disabled={loading}>
            {loading ? "…" : "Reload"}
          </button>
        </div>
      </div>

      {spots.length > 0 && (
        <div className="ticker">
          {spots.slice(0, 12).map((s) => (
            <span key={s.symbolId} className="tick">
              <b>{s.name}</b> {s.mid?.toFixed(5) ?? "—"}
            </span>
          ))}
        </div>
      )}

      {err && <div className="banner warn">{err.slice(0, 160)}</div>}

      <div className="chart-host">
        {candles.length === 0 && !loading ? (
          <div className="empty">No candles for {symbol} {tf}.</div>
        ) : (
          <Chart candles={candles} />
        )}
      </div>
      <div className="muted small">
        {candles.length} candles · source: {source} · {symbol} {tf}
      </div>
    </div>
  );
}
