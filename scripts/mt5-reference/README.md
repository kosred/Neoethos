# MT5 reference venue — EURUSD export monthly sampling

> **Venue label: MT5/Dukascopy reference venue — not NeoEthos execution truth**

This dataset is a reference/historical venue sample used for spread and quote-economics
research only. It is **not** NeoEthos execution truth and must never be mixed into the
sealed cost model.

## Source file

- Path: 'C:\Users\konst\OneDrive\Desktop\EURUSD_201601040000_202610231254.csv'
- Size: 14,486,972,042 bytes (13.49 GiB)
- Processed at: 2026-10-23T11:44:33Z (UTC)
- Delimiter: '\t'

## Detected schema (first 3 lines, verbatim; tabs shown as \t)

    L1: <DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>
    L2: 2016.01.04\t00:00:00.000\t0.00000\t0.00000\t0.00000\t0.00000000\t0
    L3: 2016.01.04\t00:00:38.000\t\t\t1.08696\t1000000.00000000\t56

- Header row detected: 'True'
- Columns (normalized): 'DATE, TIME, BID, ASK, LAST, VOLUME, FLAGS'
- Granularity: 'ticks' (stat mode 'bidask_spread')

## Column meaning assumptions

- DATE / TIME: tick timestamp as exported by the MT5 terminal (yyyy.mm.dd, HH:mm:ss.mmm);
  timezone is the export terminal's own and is not normalized.
- BID / ASK: best bid/ask at the tick; rows where either is empty or zero carry no quote
  (deal/session rows) and are excluded from spread stats but still counted as rows.
- LAST: last deal price (empty on quote-only rows).
- VOLUME: deal volume; FLAGS: MT5 tick flags bitmask.
- OPEN/HIGH/LOW/CLOSE (bar exports): OHLC prices.
- SPREAD (bar exports): spread in integer points.
- TICKVOL / VOL (bar exports): tick volume / real volume.

## Statistics (per month, in monthly-summary.json)

- spread_n: rows used for spread stats (both BID and ASK > 0).
- spread_mean_points / spread_median_points / spread_p95_points / spread_p99_points.
- Spread unit: points, derived as (ASK-BID) * 100000 (EURUSD point = 0.00001; 1 point = 0.1 pip).
- Median is the exact middle value (average of the two middle values when even); p95/p99 use
  exact nearest-rank over per-month value counts — no sampling, no approximation.
- rows: number of rows (tick records or bars) in the month.
- bars: row count for bar exports, null for tick exports (no bars exist in a tick export).
- time_start / time_end: first/last timestamp of the month (ISO-ish, as exported).

- Total rows: 332,752,151 across 130 months (2016-01 .. 2026-10).
- Rows counted but excluded from stats (empty/zero quotes or unparsable numbers): 186051377
- Median of monthly spread means: 3.54275 points

## Method

- Python stdlib csv only; row-streamed over an 8 MiB buffered text stream. The large source
  file is never loaded into memory and never copied.
- Memory use is O(number of months) plus a small per-month value counter (exact quantiles).
- Output: 'C:\Users\konst\AppData\Local\neoethos\mt5-reference\monthly-summary.json'
- Elapsed: 1610.0 s

## Verification limits

- Statistics were not cross-checked against MT5 terminal totals or an independent source.
- The file was exported at 2026-10-23 ~13:03 local (LastWriteTime); if the export was still
  running, this snapshot covers only what was on disk when the scan started.
- Derived spread (BID/ASK mode) assumes 5-digit EURUSD quotes; if the terminal exported
  3-digit quotes the point scale would be off by 100x.
