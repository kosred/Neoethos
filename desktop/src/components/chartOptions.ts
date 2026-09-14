// These are KLineChart display overlays only. They are deliberately separate
// from the indicator/SMC search universe used by the research engine.
export const KLINE_DISPLAY_INDICATORS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "MA", label: "MA · Moving Average" },
  { value: "EMA", label: "EMA" },
  { value: "BOLL", label: "Bollinger Bands" },
  { value: "SAR", label: "Parabolic SAR" },
  { value: "MACD", label: "MACD" },
  { value: "RSI", label: "RSI" },
  { value: "KDJ", label: "KDJ · Stochastic" },
  { value: "CCI", label: "CCI" },
  { value: "DMI", label: "DMI / ADX" },
  { value: "WR", label: "Williams %R" },
];
