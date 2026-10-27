# Dukascopy pilot — labeled research reference (NOT execution truth)

> **Venue rule:** Dukascopy is a *reference venue, not execution truth*.
> NeoEthos execution truth is the **cTrader demo** broker. Dukascopy data
> is labeled research reference data only and must **never** be silently
> mixed into the sealed cost model.

Bounded pilot: EURUSD BID/ASK ticks for exactly four UTC days
(2016-06-01, 2018-02-01, 2022-07-01, 2026-10-20), downloaded with
`dukascopy-downloader` (jetta.dukascopy.com) with a direct
`datafeed.dukascopy.com` `.bi5` fallback when the jetta API lacks depth.
Computed per day (nothing else): tick count, mean/median/p99 spread in
pips (1 pip = 0.0001), share of the day with spread <= 1.0 pip
(time-weighted over the tick ms-span).

## Per-day results

| date | status | ticks | mean pip | median pip | p99 pip | share<=1.0 pip | bytes | source |
|---|---|---|---|---|---|---|---|---|
| 2016-06-01 | ok | 92,646 | 0.252 | 0.2 | 0.9 | 98.5% | 1,168,710 | dukascopy-downloader 1.1.0 (jetta.dukascopy.com) |
| 2018-02-01 | ok | 120,523 | 0.336 | 0.3 | 1.0 | 97.8% | 1,594,110 | dukascopy-downloader 1.1.0 (jetta.dukascopy.com) |
| 2022-07-01 | ok | 115,505 | 0.364 | 0.3 | 0.7 | 99.7% | 1,439,733 | dukascopy-downloader 1.1.0 (jetta.dukascopy.com) |
| 2026-10-20 | ok | 77,088 | 0.329 | 0.3 | 3.5 | 95.8% | 929,053 | dukascopy-downloader 1.1.0 (jetta.dukascopy.com) |

Full method notes and per-day JSON: `summary.json` in this folder.

## Data location

- Raw ticks + per-day stats: `%LOCALAPPDATA%/neoethos/dukascopy-ref/<YYYY-MM-DD>/`
  (`EURUSD_ticks.csv`, `summary.json`)
- Downloader store (resumable cache): `%LOCALAPPDATA%/neoethos/dukascopy-ref/store/`
- jetta tick history start for EURUSD: `2003-05-04 19:00:00+00:00`

## Full-range feasibility (2016-01-01..2024-12-31) — estimate, not executed

- Trading days (weekdays): **2,348**
- Mean ticks/day from in-range pilot days: **109558.0**
- Estimated total ticks: **257,242,184**
- Bytes/tick measured: **12.7864**
- Estimated size: **3.29 GB**
- Estimated download time at measured throughput: **11.5 h**
- Request floor: 78,912 hourly chunks; at the
  package's 8 req/s rate limit the absolute floor is
  **2.74 h** of request time
  (resumable across sessions).

See `summary.json` -> `feasibility_estimate_full_range_2016_2024` for
assumptions.
