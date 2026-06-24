import { useEffect, useRef } from "react";
import {
  createChart,
  CandlestickSeries,
  LineSeries,
  ColorType,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from "lightweight-charts";
import type { Candle } from "../api";

export type Overlay = {
  name: string;
  color: string;
  priceScaleId?: string; // "right" = on price; anything else = own scale (oscillators)
  data: { time: number; value: number }[];
};

export default function Chart({
  candles,
  liveBar,
  overlays,
  onReachStart,
}: {
  candles: Candle[];
  liveBar?: Candle | null;
  overlays?: Overlay[];
  onReachStart?: () => void;
}) {
  const elRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const lineRefs = useRef<Map<string, ISeriesApi<"Line">>>(new Map());
  const reachCb = useRef<(() => void) | undefined>(undefined);
  reachCb.current = onReachStart;

  // create once
  useEffect(() => {
    if (!elRef.current) return;
    const chart = createChart(elRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0f1117" },
        textColor: "#9ca3af",
        attributionLogo: false,
      },
      grid: { vertLines: { color: "#1b2030" }, horzLines: { color: "#1b2030" } },
      rightPriceScale: { borderColor: "#2a3142" },
      timeScale: { borderColor: "#2a3142", timeVisible: true, secondsVisible: false },
      crosshair: { mode: 0 },
      autoSize: true,
    });
    seriesRef.current = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
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

  // candle data
  useEffect(() => {
    if (!seriesRef.current) return;
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
  }, [candles]);

  // live forming candle
  useEffect(() => {
    if (!seriesRef.current || !liveBar) return;
    seriesRef.current.update({
      time: liveBar.time as UTCTimestamp,
      open: liveBar.open,
      high: liveBar.high,
      low: liveBar.low,
      close: liveBar.close,
    });
  }, [liveBar]);

  // indicator overlays — create/update/remove line series by name
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const want = new Map((overlays ?? []).map((o) => [o.name, o]));
    // remove stale
    for (const [name, s] of lineRefs.current) {
      if (!want.has(name)) {
        chart.removeSeries(s);
        lineRefs.current.delete(name);
      }
    }
    // create/update
    for (const o of overlays ?? []) {
      let s = lineRefs.current.get(o.name);
      if (!s) {
        s = chart.addSeries(LineSeries, {
          color: o.color,
          lineWidth: 2,
          priceScaleId: o.priceScaleId ?? "right",
          priceLineVisible: false,
          lastValueVisible: false,
        });
        lineRefs.current.set(o.name, s);
      }
      s.setData(o.data.map((d) => ({ time: d.time as UTCTimestamp, value: d.value })));
    }
  }, [overlays]);

  return <div ref={elRef} style={{ position: "absolute", inset: 0 }} />;
}
