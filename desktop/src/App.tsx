import { useEffect, useRef, useState, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  createChart,
  CandlestickSeries,
  ColorType,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from "lightweight-charts";
import "./App.css";

type AppInfo = { version: string; data_root: string; data_root_exists: boolean };
type Candle = { time: number; open: number; high: number; low: number; close: number };

const TF_ORDER = ["M1", "M3", "M5", "M15", "M30", "H1", "H4", "H12", "D1", "W1", "MN1"];

export default function App() {
  const [info, setInfo] = useState<AppInfo | null>(null);
  const [symbols, setSymbols] = useState<string[]>([]);
  const [symbol, setSymbol] = useState<string>("");
  const [timeframes, setTimeframes] = useState<string[]>([]);
  const [tf, setTf] = useState<string>("");
  const [count, setCount] = useState<number>(0);
  const [err, setErr] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  const chartElRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  // ── Create the chart once ───────────────────────────────────────────────
  useEffect(() => {
    if (!chartElRef.current) return;
    const chart = createChart(chartElRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0f1117" },
        textColor: "#9ca3af",
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: "#1b2030" },
        horzLines: { color: "#1b2030" },
      },
      rightPriceScale: { borderColor: "#2a3142" },
      timeScale: { borderColor: "#2a3142", timeVisible: true, secondsVisible: false },
      crosshair: { mode: 0 },
      autoSize: true,
    });
    const series = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });
    chartRef.current = chart;
    seriesRef.current = series;
    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  // ── Initial load: app info + symbols ────────────────────────────────────
  useEffect(() => {
    (async () => {
      try {
        const ai = await invoke<AppInfo>("app_info");
        setInfo(ai);
        const syms = await invoke<string[]>("list_symbols");
        setSymbols(syms);
        if (syms.length) setSymbol(syms.includes("EURUSD") ? "EURUSD" : syms[0]);
      } catch (e) {
        setErr(String(e));
      }
    })();
  }, []);

  // ── Symbol change → load its timeframes ─────────────────────────────────
  useEffect(() => {
    if (!symbol) return;
    (async () => {
      try {
        const tfs = await invoke<string[]>("list_timeframes", { symbol });
        const ordered = tfs
          .slice()
          .sort((a, b) => TF_ORDER.indexOf(a) - TF_ORDER.indexOf(b));
        setTimeframes(ordered);
        setTf((prev) =>
          ordered.includes(prev) ? prev : ordered.includes("H1") ? "H1" : ordered[0] ?? "",
        );
      } catch (e) {
        setErr(String(e));
      }
    })();
  }, [symbol]);

  // ── (symbol, tf) change → load candles ──────────────────────────────────
  const loadChart = useCallback(async () => {
    if (!symbol || !tf || !seriesRef.current) return;
    setLoading(true);
    setErr("");
    try {
      const candles = await invoke<Candle[]>("chart", {
        symbol,
        timeframe: tf,
        limit: 1500,
      });
      seriesRef.current.setData(
        candles.map((c) => ({
          time: c.time as UTCTimestamp,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        })),
      );
      chartRef.current?.timeScale().fitContent();
      setCount(candles.length);
    } catch (e) {
      setErr(String(e));
      setCount(0);
    } finally {
      setLoading(false);
    }
  }, [symbol, tf]);

  useEffect(() => {
    loadChart();
  }, [loadChart]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          NeoEthos <span className="pill">TAURI</span>
        </div>
        <div className="pickers">
          <label>
            Symbol
            <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
              {symbols.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </label>
          <label>
            Timeframe
            <select value={tf} onChange={(e) => setTf(e.target.value)}>
              {timeframes.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </label>
          <button onClick={loadChart} disabled={loading}>
            {loading ? "Loading…" : "Reload"}
          </button>
        </div>
      </header>

      {err && <div className="error">⚠ {err}</div>}

      <main className="chart-wrap">
        <div ref={chartElRef} className="chart" />
      </main>

      <footer className="statusbar">
        <span>{symbol || "—"} · {tf || "—"}</span>
        <span>{count} candles</span>
        <span className="spacer" />
        <span className="muted">{info?.data_root ?? "resolving data root…"}</span>
        <span className="ver">v{info?.version ?? "…"}</span>
      </footer>
    </div>
  );
}
