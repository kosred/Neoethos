/**
 * Exact cTrader Open API trendbar periods in protocol order.
 *
 * Every label names its own direct broker/source artifact. A successful broker
 * response remains authoritative; this static contract keeps controls complete
 * while the broker timeframe endpoint is unavailable.
 */
export const CANONICAL_BROKER_TIMEFRAMES = [
  "M1",
  "M2",
  "M3",
  "M4",
  "M5",
  "M10",
  "M15",
  "M30",
  "H1",
  "H4",
  "H12",
  "D1",
  "W1",
  "MN1",
] as const;

export type CanonicalBrokerTimeframe =
  (typeof CANONICAL_BROKER_TIMEFRAMES)[number];

export function isCanonicalBrokerTimeframe(
  value: string,
): value is CanonicalBrokerTimeframe {
  return (CANONICAL_BROKER_TIMEFRAMES as readonly string[]).includes(value);
}
