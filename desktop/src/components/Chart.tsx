import { useEffect, useRef } from "react";
import {
  createChart,
  CandlestickSeries,
  LineSeries,
  ColorType,
  CrosshairMode,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from "lightweight-charts";
import type { Candle } from "../api";

export type Overlay = {
  name: string;
  color: string;
  priceScaleId?: string; // "right" = on the price pane; anything else = own oscillator pane
  data: { time: number; value: number }[];
};

export default function Chart({
  candles,
  liveBar,
  overlays,
  onReachStart,
  resetKey,
}: {
  candles: Candle[];
  liveBar?: Candle | null;
  overlays?: Overlay[];
  onReachStart?: () => void;
  /** When this changes (e.g. "EURUSD|H1") the view refits; otherwise the
   *  user's zoom/scroll is preserved across data + live updates. */
  resetKey?: string;
}) {
  const elRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const lineRefs = useRef<Map<string, ISeriesApi<"Line">>>(new Map());
  const reachCb = useRef<(() => void) | undefined>(undefined);
  reachCb.current = onReachStart;

  // Zoom-preservation bookkeeping (see the candle-data effect).
  const resetKeyRef = useRef<string | undefined>(undefined);
  const prevLenRef = useRef(0);
  const prevFirstRef = useRef(0);

  // Lightweight-charts renders UTCTimestamps in UTC. Shift by the local
  // timezone offset so the axis shows the operator's WALL-CLOCK time (e.g.
  // Berlin/CEST = UTC+2 → bars no longer look "2h behind"). Display-only; the
  // underlying data stays correct UTC.
  const tzOffSec = new Date().getTimezoneOffset() * 60;
  const toLocal = (t: number) => (t - tzOffSec) as UTCTimestamp;

  // create once
  useEffect(() => {
    if (!elRef.current) return;
    const chart = createChart(elRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0f1117" },
        textColor: "#9ca3af",
        attributionLogo: false,
        fontFamily: "Inter, system-ui, sans-serif",
        panes: { separatorColor: "#2a3142", separatorHoverColor: "#3b4358", enableResize: true },
      } as any,
      grid: { vertLines: { color: "#161b26" }, horzLines: { color: "#161b26" } },
      rightPriceScale: { borderColor: "#2a3142", scaleMargins: { top: 0.08, bottom: 0.12 } },
      timeScale: {
        borderColor: "#2a3142",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 6,
        barSpacing: 9,
        minBarSpacing: 0.5, // allow zooming OUT far; zoom-IN is uncapped for close inspection
      },
      crosshair: { mode: CrosshairMode.Normal },
      autoSize: true,
    });
    seriesRef.current = chart.addSeries(CandlestickSeries, {
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderVisible: false,
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
    });
    chartRef.current = chart;

    // scroll-back: fire when the user pans near the left edge
    chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (range && range.from < 6) reachCb.current?.();
    });

    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      lineRefs.current.clear();
    };
  }, []);

  // candle data — preserve the user's zoom/scroll unless the symbol/TF changed
  useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart || !series) return;

    const data = candles.map((c) => ({
      time: toLocal(c.time),
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));

    const ts = chart.timeScale();
    const range = ts.getVisibleLogicalRange();
    const prevLen = prevLenRef.current;
    const prevFirst = prevFirstRef.current;
    const isReset = resetKeyRef.current !== resetKey;

    series.setData(data);

    if (isReset || prevLen === 0) {
      // New instrument/timeframe (or first load): frame the whole series.
      ts.fitContent();
      resetKeyRef.current = resetKey;
    } else if (range && candles.length > prevLen && (candles[0]?.time ?? 0) < prevFirst) {
      // Scroll-back prepended older bars — shift the logical range by the
      // number added so the SAME bars stay under the viewport (no jump).
      const delta = candles.length - prevLen;
      ts.setVisibleLogicalRange({ from: range.from + delta, to: range.to + delta });
    }
    // else: live tick / same dataset → setData preserves the current range.

    prevLenRef.current = candles.length;
    prevFirstRef.current = candles[0]?.time ?? 0;
  }, [candles, resetKey]);

  // live forming candle — updates in place, never refits (zoom preserved)
  useEffect(() => {
    if (!seriesRef.current || !liveBar) return;
    seriesRef.current.update({
      time: toLocal(liveBar.time),
      open: liveBar.open,
      high: liveBar.high,
      low: liveBar.low,
      close: liveBar.close,
    });
  }, [liveBar]);

  // indicator overlays — price-scale indicators (SMA/EMA/BBands/VWAP) draw on
  // the main pane; oscillators (RSI/MACD/ATR/Stoch) get their OWN pane below so
  // they read at their true scale instead of being squashed onto the price axis.
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const want = new Map((overlays ?? []).map((o) => [o.name, o]));

    // remove stale series (empty panes auto-remove in v5)
    for (const [name, s] of lineRefs.current) {
      if (!want.has(name)) {
        chart.removeSeries(s);
        lineRefs.current.delete(name);
      }
    }

    let usesOscillatorPane = false;
    for (const o of overlays ?? []) {
      const onPrice = !o.priceScaleId || o.priceScaleId === "right";
      const paneIndex = onPrice ? 0 : 1;
      if (!onPrice) usesOscillatorPane = true;
      let s = lineRefs.current.get(o.name);
      if (!s) {
        s = chart.addSeries(
          LineSeries,
          {
            color: o.color,
            lineWidth: 2,
            priceScaleId: "right", // each pane has its own right scale
            priceLineVisible: false,
            lastValueVisible: onPrice,
          },
          paneIndex,
        );
        lineRefs.current.set(o.name, s);
      }
      s.setData(o.data.map((d) => ({ time: toLocal(d.time), value: d.value })));
    }

    // Keep the price pane dominant; give the oscillator pane a slim slice.
    if (usesOscillatorPane) {
      const panes = chart.panes();
      if (panes.length > 1) {
        try {
          panes[0].setStretchFactor(4);
          panes[1].setStretchFactor(1);
        } catch {
          /* stretch API guarded — ignore if unavailable */
        }
      }
    }
  }, [overlays]);

  return <div ref={elRef} style={{ position: "absolute", inset: 0 }} />;
}
