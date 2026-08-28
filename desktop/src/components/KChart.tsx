import { useEffect, useRef } from "react";
import { init, dispose, type Chart, type KLineData, type DeepPartial, type Styles } from "klinecharts";
import { brokerChart, chartHistory, type Candle, type Tick } from "../api";
import {
  CANONICAL_BROKER_TIMEFRAMES,
  isCanonicalBrokerTimeframe,
  type CanonicalBrokerTimeframe,
} from "../timeframes";

// Exact direct broker timeframe to klinecharts display period.
const PERIOD: Record<CanonicalBrokerTimeframe, { type: "minute" | "hour" | "day" | "week" | "month"; span: number }> = {
  M1: { type: "minute", span: 1 }, M2: { type: "minute", span: 2 },
  M3: { type: "minute", span: 3 }, M4: { type: "minute", span: 4 },
  M5: { type: "minute", span: 5 }, M10: { type: "minute", span: 10 },
  M15: { type: "minute", span: 15 }, M30: { type: "minute", span: 30 },
  H1: { type: "hour", span: 1 }, H4: { type: "hour", span: 4 }, H12: { type: "hour", span: 12 },
  D1: { type: "day", span: 1 }, W1: { type: "week", span: 1 }, MN1: { type: "month", span: 1 },
};
// FX price precision heuristic (JPY pairs 3, metals 2, majors 5).
const precisionFor = (sym: string) => {
  const s = sym.toUpperCase();
  if (s.endsWith("JPY")) return 3;
  if (s.startsWith("XAU") || s.startsWith("XAG")) return 2;
  return 5;
};

const toKline = (c: Candle): KLineData => ({
  timestamp: c.time * 1000, // our Candle.time is UTC seconds → klinecharts wants ms
  open: c.open, high: c.high, low: c.low, close: c.close,
});

// Indicators drawn ON the candles vs. in their own sub-pane (oscillators).
const PRICE_OVERLAY = new Set(["MA", "EMA", "SMA", "BOLL", "SAR"]);

// KLineChart built-in indicators offered in the screens' dropdowns.
export const KLINE_INDICATORS: { v: string; l: string }[] = [
  { v: "MA", l: "MA · Moving Average" },
  { v: "EMA", l: "EMA" },
  { v: "BOLL", l: "Bollinger Bands" },
  { v: "SAR", l: "Parabolic SAR" },
  { v: "MACD", l: "MACD" },
  { v: "RSI", l: "RSI" },
  { v: "KDJ", l: "KDJ · Stochastic" },
  { v: "CCI", l: "CCI" },
  { v: "DMI", l: "DMI / ADX" },
  { v: "WR", l: "Williams %R" },
];

// Drawing tools exposed in the toolbar (klinecharts built-in overlay names).
const DRAW_TOOLS: { name: string; label: string; title: string }[] = [
  { name: "segment", label: "／", title: "Trend line" },
  { name: "rayLine", label: "→", title: "Ray" },
  { name: "horizontalStraightLine", label: "―", title: "Horizontal line" },
  { name: "priceLine", label: "$", title: "Price line" },
  { name: "fibonacciLine", label: "fib", title: "Fibonacci retracement" },
  { name: "rect", label: "▭", title: "Rectangle" },
  { name: "simpleAnnotation", label: "T", title: "Text note" },
];

const DARK_STYLES: DeepPartial<Styles> = {
  grid: { horizontal: { color: "#161b26" }, vertical: { color: "#161b26" } },
  candle: {
    bar: {
      upColor: "#26a69a", downColor: "#ef5350", noChangeColor: "#888888",
      upBorderColor: "#26a69a", downBorderColor: "#ef5350",
      upWickColor: "#26a69a", downWickColor: "#ef5350",
    },
    priceMark: {
      high: { color: "#9ca3af" }, low: { color: "#9ca3af" },
    },
    tooltip: { title: { show: true }, legend: { color: "#cbd5e1" } as any },
  },
  indicator: {
    tooltip: { legend: { color: "#cbd5e1" } as any },
  },
  xAxis: { axisLine: { color: "#2a3142" }, tickLine: { color: "#2a3142" }, tickText: { color: "#9ca3af" } },
  yAxis: { axisLine: { color: "#2a3142" }, tickLine: { color: "#2a3142" }, tickText: { color: "#9ca3af" } },
  separator: { color: "#2a3142" },
  crosshair: {
    horizontal: { line: { color: "#3b4358" }, text: { backgroundColor: "#2a3142" } },
    vertical: { line: { color: "#3b4358" }, text: { backgroundColor: "#2a3142" } },
  },
};

export default function KChart({
  symbol,
  timeframe,
  indicator,
  liveTick,
}: {
  symbol: string;
  timeframe: string;
  indicator?: string; // klinecharts built-in name (e.g. "MACD"); "" = none
  liveTick?: Tick | null;
}) {
  const elRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<Chart | null>(null);
  const symRef = useRef(symbol);
  const tfRef = useRef(timeframe);
  symRef.current = symbol;
  tfRef.current = timeframe;

  const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";

  // create once
  useEffect(() => {
    if (!elRef.current) return;
    const chart = init(elRef.current, { locale: "en-US", timezone: tz, styles: DARK_STYLES });
    if (!chart) return;
    chartRef.current = chart;

    chart.setDataLoader({
      getBars: async ({ type, timestamp, callback }) => {
        const sym = symRef.current;
        const tf = tfRef.current;
        if (!sym || !isCanonicalBrokerTimeframe(tf)) { callback([], false); return; }
        try {
          if (type === "init") {
            const c = await brokerChart(sym, tf, 800);
            callback(c.map(toKline), { forward: true, backward: false });
          } else if (type === "forward" && timestamp != null) {
            const res = await chartHistory(sym, tf, timestamp, 500);
            const bars = res.candles
              .filter((b) => b.tsMs != null)
              .map((b) => ({ timestamp: b.tsMs as number, open: b.open, high: b.high, low: b.low, close: b.close }));
            callback(bars, { forward: res.hasMore, backward: false });
          } else {
            callback([], false);
          }
        } catch {
          callback([], false);
        }
      },
    });

    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(elRef.current);

    return () => {
      ro.disconnect();
      chartRef.current = null;
      dispose(elRef.current!);
    };
  }, []);

  // symbol / timeframe → reload (triggers a fresh getBars("init"))
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !symbol || !isCanonicalBrokerTimeframe(timeframe)) return;
    chart.setSymbol({ ticker: symbol, pricePrecision: precisionFor(symbol), volumePrecision: 0 });
    chart.setPeriod(PERIOD[timeframe]);
  }, [symbol, timeframe]);

  // indicator selection → single indicator at a time (price overlay or sub-pane)
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.removeIndicator(); // clear any previous
    const name = (indicator ?? "").toUpperCase();
    if (!name) return;
    if (PRICE_OVERLAY.has(name)) {
      chart.createIndicator(name, { isStack: true, pane: { id: "candle_pane" } });
    } else {
      chart.createIndicator(name, { pane: { id: "ind_pane", height: 110 } });
    }
  }, [indicator]);

  const draw = (name: string) => chartRef.current?.createOverlay(name);
  const clearDrawings = () => chartRef.current?.removeOverlay();
  const invalidTimeframe =
    timeframe.length > 0 && !isCanonicalBrokerTimeframe(timeframe);
  const currentPriceMarker =
    liveTick &&
    liveTick.symbolName === symbol &&
    Number.isFinite(liveTick.midPrice)
      ? liveTick.midPrice
      : null;

  return (
    <div style={{ position: "absolute", inset: 0, background: "#0f1117" }}>
      {invalidTimeframe && (
        <div
          className="banner warn"
          style={{ position: "absolute", zIndex: 2, inset: 12, bottom: "auto" }}
        >
          Unsupported broker timeframe {timeframe}. Select one exact direct period:{" "}
          {CANONICAL_BROKER_TIMEFRAMES.join(", ")}.
        </div>
      )}
      {currentPriceMarker !== null && (
        <div
          className="kchart-live-price-marker"
          style={{
            position: "absolute",
            zIndex: 2,
            right: 12,
            top: 48,
            pointerEvents: "none",
          }}
        >
          LIVE {currentPriceMarker.toFixed(precisionFor(symbol))}
        </div>
      )}
      <div className="kchart-tools">
        {DRAW_TOOLS.map((t) => (
          <button key={t.name} title={t.title} onClick={() => draw(t.name)}>{t.label}</button>
        ))}
        <button title="Clear all drawings" className="danger" onClick={clearDrawings}>✕</button>
      </div>
      <div
        ref={elRef}
        style={{
          position: "absolute",
          inset: 0,
          visibility: invalidTimeframe ? "hidden" : "visible",
        }}
      />
    </div>
  );
}
