#!/usr/bin/env python3
"""Dukascopy historical tick pilot for NeoEthos (bounded: 4 UTC days).

Purpose
-------
Verify Dukascopy as a LABELED RESEARCH REFERENCE for historical EURUSD
BID/ASK tick spread statistics.

IMPORTANT VENUE RULE: Dukascopy is a reference venue, NOT the execution
venue. NeoEthos execution truth is the cTrader demo broker. Dukascopy
data must never be silently mixed into the sealed cost model. These
artifacts are research-only labels.

Scope (bounded pilot, not full history)
---------------------------------------
Exactly four UTC days: 2016-06-01, 2018-02-01, 2022-07-01, 2026-10-20.
Per day this script computes ONLY:
  * tick count
  * mean / median / p99 spread in pips (1 pip = 0.0001 for EURUSD)
  * share of the day with spread <= 1.0 pip (time-weighted over ms span)
plus the bytes/tick needed for the 2016-2024 full-range feasibility
estimate. No other statistics are computed. No orders, no broker access.

Sources
-------
Primary : dukascopy-downloader 1.1.0 (jetta.dukascopy.com API) -> local store.
Fallback: direct https://datafeed.dukascopy.com .bi5 hourly tick files
          (only if the jetta API serves no data for a day, e.g. outside
          its tick-history depth).

Outputs
-------
%LOCALAPPDATA%/neoethos/dukascopy-ref/<YYYY-MM-DD>/EURUSD_ticks.csv   raw ticks
%LOCALAPPDATA%/neoethos/dukascopy-ref/<YYYY-MM-DD>/summary.json       per-day stats
%LOCALAPPDATA%/neoethos/dukascopy-ref/summary.json                    combined report
<repo>/scripts/dukascopy-pilot/<YYYY-MM-DD>.json                      per-day copy
<repo>/scripts/dukascopy-pilot/summary.json                           combined copy
<repo>/scripts/dukascopy-pilot/README.md                              generated notes
"""
from __future__ import annotations

import json
import lzma
import logging
import os
import struct
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from dukascopy_downloader import Downloader

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("pilot")

INSTRUMENT = "EURUSD"
CODE = None  # resolved jetta API instrument code, set in main()
PIP = 0.0001  # one EURUSD pip
DAYS = ["2016-06-01", "2018-02-01", "2022-07-01", "2026-10-20"]
MAX_RETRIES = 2  # "do not retry more than twice" -> initial attempt + 2 retries
VENUE_ROLE = ("labeled research reference only; NOT the execution venue "
              "(execution truth = cTrader demo); never mixed into the sealed cost model")
SPREAD_METHOD = ("spread_pips=(ask-bid)/0.0001; p99=linear-interpolation quantile; "
                 "share_of_day=time-weighted: ms-span of ticks with spread<=1.0 pip "
                 "divided by ms-span from first to last tick of the UTC day")

LOCALAPPDATA = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
DATA_ROOT = Path(LOCALAPPDATA) / "neoethos" / "dukascopy-ref"
STORE_DIR = DATA_ROOT / "store"
REPO_DIR = Path(__file__).resolve().parent


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ stats

def compute_stats(df: pd.DataFrame) -> dict:
    """The agreed per-day statistics. Nothing else."""
    n = len(df)
    if n == 0:
        return {
            "tick_count": 0,
            "mean_spread_pips": None,
            "median_spread_pips": None,
            "p99_spread_pips": None,
            "share_of_day_spread_le_1pip": None,
            "ticks_with_ask_lt_bid": 0,
        }
    ask = df["ask"].to_numpy(dtype="float64")
    bid = df["bid"].to_numpy(dtype="float64")
    spread = (ask - bid) / PIP
    bad = int((spread < 0).sum())
    share = None
    if n >= 2:
        ms = df["time"].astype("int64").to_numpy(dtype="int64") // 1_000_000
        dur = np.diff(ms)
        total = int(ms[-1] - ms[0])
        if total > 0:
            share = float(dur[spread[:-1] <= 1.0].sum() / total)
    return {
        "tick_count": n,
        "mean_spread_pips": float(spread.mean()),
        "median_spread_pips": float(np.median(spread)),
        "p99_spread_pips": float(np.quantile(spread, 0.99)),
        "share_of_day_spread_le_1pip": share,
        "ticks_with_ask_lt_bid": bad,
    }


def df_to_csv_rows(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "time_ms_utc": df["time"].astype("int64") // 1_000_000,
        "bid": df["bid"],
        "ask": df["ask"],
        "bid_volume": df["bid_volume"],
        "ask_volume": df["ask_volume"],
    })


# ------------------------------------------------------------------ jetta source

def stored_day_size(store, day: str) -> int:
    try:
        cd = store.code_dir("tick", CODE)
        for p in cd.glob(day + ".*"):
            if p.is_file():
                return p.stat().st_size
    except Exception:
        pass
    return 0


def jetta_attempt(d, day: str):
    """One jetta download attempt. Returns (res, df, seconds, bytes_stored)."""
    t0 = time.monotonic()
    res = d.download_series(CODE, "tick", start=day, end=day)
    secs = time.monotonic() - t0
    df = d.store.read("tick", CODE, start=day, end=day)
    size = stored_day_size(d.store, day)
    return res, df, secs, size


def jetta_day(d, day: str):
    """Try jetta up to MAX_RETRIES+1 times. Returns dict or None if unusable."""
    if CODE is None:
        return None
    last = None
    for attempt in range(MAX_RETRIES + 1):
        res, df, secs, size = jetta_attempt(d, day)
        errs = [f"{k}: {m}" for k, m in res.errors]
        info = {
            "planned": res.planned, "fetched": res.fetched,
            "cached": res.cached, "failed": len(res.errors),
        }
        log.info("%s jetta attempt %d: planned=%d fetched=%d rows=%d errors=%d secs=%.1f",
                 day, attempt + 1, res.planned, res.fetched, len(df), len(res.errors), secs)
        if not errs:
            return {"res": res, "df": df, "secs": secs, "bytes": size,
                    "requests": info, "errors": [], "attempts": attempt + 1}
        last = {"res": res, "df": df, "secs": secs, "bytes": size,
                "requests": info, "errors": errs, "attempts": attempt + 1}
        if len(df):
            return last  # partial data; keep it, report errors
        time.sleep(2 * (attempt + 1))
    return last


# ------------------------------------------------------------------ bi5 fallback

def fetch_bi5_day(day: str, session: requests.Session):
    """Direct datafeed.dukascopy.com hourly .bi5 ticks (LZMA, 20-byte records).

    Record layout (big-endian): >I f f f f = ms offset in hour, ask, bid,
    ask_volume, bid_volume. Month in URL path is 0-based.
    """
    d = date.fromisoformat(day)
    hour_base = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)
    rows, wire, missing = [], 0, []
    for hour in range(24):
        url = (f"https://datafeed.dukascopy.com/datafeed/{INSTRUMENT}/"
               f"{d.year}/{d.month - 1:02d}/{d.day:02d}/{hour:02d}h_ticks.bi5")
        r = session.get(url, timeout=30)
        if r.status_code == 404:
            missing.append(hour)
            continue
        r.raise_for_status()
        raw = r.content
        wire += len(raw)
        try:
            data = lzma.decompress(raw)
        except lzma.LZMAError:
            data = lzma.decompress(raw, format=lzma.FORMAT_ALONE)
        n = len(data) // 20
        fmt = ">iffff"
        base = hour_base + hour * 3_600_000
        for i in range(n):
            ms_off, ask, bid, ask_v, bid_v = struct.unpack_from(fmt, data, i * 20)
            rows.append((base + ms_off, bid, ask, bid_v, ask_v))
    if not rows:
        return (pd.DataFrame(columns=["time", "bid", "ask", "bid_volume", "ask_volume"]),
                wire, missing, "no .bi5 data for any hour of this UTC day")
    df = pd.DataFrame(rows, columns=["time_ms_utc", "bid", "ask", "bid_volume", "ask_volume"])
    df["time"] = pd.to_datetime(df["time_ms_utc"], unit="ms", utc=True)
    df = df[["time", "bid", "ask", "bid_volume", "ask_volume"]].sort_values("time").reset_index(drop=True)
    err = (f"missing hours (404): {missing}" if missing else None)
    return df, wire, missing, err


def bi5_day(day: str):
    session = requests.Session()
    for attempt in range(MAX_RETRIES + 1):
        t0 = time.monotonic()
        try:
            df, wire, missing, err = fetch_bi5_day(day, session)
            secs = time.monotonic() - t0
        except Exception as e:  # network/parse failure -> retry
            time.sleep(2 * (attempt + 1))
            if attempt == MAX_RETRIES:
                raise
            continue
        info = {"fetched": 24 - len(missing), "missing": missing}
        return df, wire, secs, info, err, attempt + 1
    raise RuntimeError("unreachable")


# ------------------------------------------------------------------ artifacts

def write_day_artifacts(day: str, df: pd.DataFrame, entry: dict):
    day_dir = DATA_ROOT / day
    day_dir.mkdir(parents=True, exist_ok=True)
    csv_path = day_dir / "EURUSD_ticks.csv"
    if len(df):
        df_to_csv_rows(df).to_csv(csv_path, index=False)
    else:
        csv_path.write_text("time_ms_utc,bid,ask,bid_volume,ask_volume" + chr(10), encoding="utf-8")
    summary = dict(entry)
    summary["file"] = str(csv_path)
    (day_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPO_DIR / (day + ".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path


# ------------------------------------------------------------------ estimate

def build_estimate(day_entries, history_start):
    in_range_ok = [e for e in day_entries if e["status"] == "ok" and e["date"] < "2025-01-01"]
    pool = in_range_ok or [e for e in day_entries if e["status"] == "ok"]
    est = {"status": "estimate-only; not executed", "range_utc": "2016-01-01..2024-12-31 (inclusive)"}
    start = date(2016, 1, 1)
    calendar_days = (date(2024, 12, 31) - start).days + 1
    weekdays = 0
    for i in range(calendar_days):
        if (start + timedelta(days=i)).weekday() < 5:
            weekdays += 1
    est["calendar_days"] = calendar_days
    est["trading_days_weekdays"] = weekdays
    est["request_chunks_estimate"] = calendar_days * 24
    est["estimated_hours_at_8_req_per_s_floor"] = round(calendar_days * 24 / 8 / 3600, 2)
    if not pool:
        est["error"] = "no successful day to extrapolate from"
        return est
    est["sample_days_used"] = [e["date"] for e in pool]
    ticks = sum(e["tick_count"] for e in pool)
    bytes_ = sum(e["bytes_measured"] for e in pool)
    secs = sum(e["download_seconds"] for e in pool)
    est["mean_ticks_per_day"] = round(ticks / len(pool), 1)
    est["bytes_per_tick_measured"] = round(bytes_ / ticks, 4) if ticks else None
    est["estimated_total_ticks"] = int(est["mean_ticks_per_day"] * weekdays)
    est["estimated_size_gb"] = round(est["estimated_total_ticks"] * (bytes_ / ticks) / 1e9, 2) if ticks else None
    est["measured_throughput_bytes_per_second"] = round(bytes_ / secs, 1) if secs else None
    if secs and est["estimated_size_gb"] is not None:
        est["estimated_download_hours_at_measured_throughput"] = round(
            est["estimated_size_gb"] * 1e9 / (bytes_ / secs) / 3600, 1)
    est["assumptions"] = [
        "bytes_measured = compressed stored bytes (csv.gz) for jetta days, raw .bi5 wire bytes for fallback days",
        "ticks/day extrapolated from the in-range pilot days only (2026-10-20 excluded; it is the probe-overlap day)",
        "weekend/holiday days cost a cheap empty response but are still counted in request_chunks_estimate",
        "jetta API rate limit 8 req/s (package default) puts a hard floor on full-range wall time",
        "dukascopy-downloader is resumable (per-chunk manifest), so the range can be fetched across sessions",
        "storage size assumes csv.gz; parquet would differ slightly",
    ]
    return est


# ------------------------------------------------------------------ main

def main() -> int:
    global CODE
    print(f"pilot start {utcnow_iso()} instrument={INSTRUMENT}")
    print(f"DATA_ROOT={DATA_ROOT}")
    print(f"STORE_DIR={STORE_DIR}")

    d = Downloader(str(STORE_DIR), progress=False)
    history_start = None
    try:
        CODE = d.resolve_symbols([INSTRUMENT])[0]
        print(f"resolved {INSTRUMENT} -> jetta code {CODE!r}")
        history_start = d.instruments.history_start(CODE, "tick")
        print(f"jetta tick history_start({CODE}) = {history_start}")
    except Exception as e:
        CODE = None
        print(f"WARN: jetta catalogue/history unavailable: {type(e).__name__}: {e}; using .bi5 fallback")

    day_entries = []
    for day in DAYS:
        entry = {
            "date": day, "instrument": INSTRUMENT, "venue": "dukascopy",
            "venue_role": VENUE_ROLE, "status": "failed", "source": None,
            "attempts": 0, "error": None, "detail": None,
            "requests": None, "download_seconds": 0.0,
            "bytes_measured": 0, "bytes_wire_bi5": 0,
            "spread_stats_method": SPREAD_METHOD,
        }
        df = pd.DataFrame(columns=["time", "bid", "ask", "bid_volume", "ask_volume"])
        try:
            try:
                j = jetta_day(d, day)
            except Exception as e:
                entry["error"] = f"jetta: {type(e).__name__}: {e}"
                entry["requests"] = None
                entry["source"] = "dukascopy-downloader 1.1.0 (jetta.dukascopy.com) FAILED"
                j = None
            if j is not None:
                entry.update({
                    "source": "dukascopy-downloader 1.1.0 (jetta.dukascopy.com)",
                    "attempts": j["attempts"],
                    "requests": j["requests"],
                    "download_seconds": round(j["secs"], 1),
                    "bytes_measured": j["bytes"],
                })
                df = j["df"]
                if j["errors"]:
                    entry["error"] = "; ".join(j["errors"])
                if len(df) and j["errors"]:
                    entry["status"] = "failed"
                    entry["detail"] = "partial data; some hour chunks failed"
                elif len(df):
                    entry["status"] = "ok"
                elif j["errors"]:
                    entry["status"] = "failed"
                    entry["detail"] = "jetta served no rows and reported errors"
                else:
                    entry["status"] = "empty"
                    entry["detail"] = "jetta served no ticks for this UTC day"
            if len(df) == 0:
                entry["error"] = ((entry["error"] + "; ") if entry["error"] else "") + (
                    f"jetta unusable (history_start={history_start}, requests={entry['requests']})")
                entry["source"] = "datafeed.dukascopy.com .bi5 (fallback)"
                bdf, wire, secs, info, err, att = bi5_day(day)
                df = bdf
                entry["attempts"] += att
                entry["download_seconds"] = round(entry["download_seconds"] + secs, 1)
                entry["bytes_wire_bi5"] = wire
                entry["bytes_measured"] = wire
                entry["requests"] = {"fetched": info["fetched"], "missing": info["missing"]}
                if err:
                    entry["error"] = ((entry["error"] + "; ") if entry["error"] else "") + f"bi5: {err}"
                if len(df):
                    entry["status"] = "ok" if not err else "failed"
                    if err and entry["status"] == "failed":
                        entry["detail"] = "partial data from .bi5 fallback"
                else:
                    entry["status"] = "empty"
                    entry["detail"] = "no ticks available from either source"
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["status"] = "failed"

        stats = compute_stats(df)
        entry.update(stats)
        write_day_artifacts(day, df, entry)
        day_entries.append(entry)
        print(f"  {day}: status={entry['status']} source={entry['source']} "
              f"ticks={entry['tick_count']} err={bool(entry['error'])} "
              f"bytes={entry['bytes_measured']} secs={entry['download_seconds']}")

    estimate = build_estimate(day_entries, history_start)
    combined = {
        "generated_at_utc": utcnow_iso(),
        "pilot_scope": "bounded pilot: exactly 4 UTC days (no full-history download)",
        "instrument": INSTRUMENT,
        "jetta_code": CODE,
        "venue": "dukascopy",
        "venue_role": VENUE_ROLE,
        "data_root": str(DATA_ROOT),
        "store_dir": str(STORE_DIR),
        "jetta_tick_history_start_utc": str(history_start),
        "days": day_entries,
        "feasibility_estimate_full_range_2016_2024": estimate,
        "method_notes": [SPREAD_METHOD],
    }
    (DATA_ROOT / "summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    (REPO_DIR / "summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    write_readme(combined, estimate, history_start)
    print("combined summary written:", DATA_ROOT / "summary.json")
    print("JSON " + json.dumps(combined, default=str)[:4000])
    return 0


def write_readme(combined, estimate, history_start):
    def gfmt(key, nd=None):
        v = estimate.get(key)
        return "n/a" if v is None else (f"{v:,}" if nd is None else str(round(v, nd)))

    rows = []
    for e in combined["days"]:
        def fmt(x, nd=3):
            return "" if x is None else str(round(x, nd))
        rows.append(
            f"| {e['date']} | {e['status']} | {e['tick_count']:,} | "
            f"{fmt(e['mean_spread_pips'])} | {fmt(e['median_spread_pips'])} | "
            f"{fmt(e['p99_spread_pips'])} | "
            f"{'' if e['share_of_day_spread_le_1pip'] is None else str(round(100 * e['share_of_day_spread_le_1pip'], 1)) + '%'} | "
            f"{e['bytes_measured']:,} | {e['source']} |")
    est_line = ""
    hrs = estimate.get("estimated_download_hours_at_measured_throughput")
    if hrs is not None:
        est_line = chr(10) + "- Estimated download time at measured throughput: **" + str(hrs) + " h**"
    text = f"""# Dukascopy pilot — labeled research reference (NOT execution truth)

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
{chr(10).join(rows)}

Full method notes and per-day JSON: `summary.json` in this folder.

## Data location

- Raw ticks + per-day stats: `%LOCALAPPDATA%/neoethos/dukascopy-ref/<YYYY-MM-DD>/`
  (`EURUSD_ticks.csv`, `summary.json`)
- Downloader store (resumable cache): `%LOCALAPPDATA%/neoethos/dukascopy-ref/store/`
- jetta tick history start for EURUSD: `{history_start}`

## Full-range feasibility (2016-01-01..2024-12-31) — estimate, not executed

- Trading days (weekdays): **{gfmt('trading_days_weekdays')}**
- Mean ticks/day from in-range pilot days: **{gfmt('mean_ticks_per_day', 1)}**
- Estimated total ticks: **{gfmt('estimated_total_ticks')}**
- Bytes/tick measured: **{gfmt('bytes_per_tick_measured', 4)}**
- Estimated size: **{gfmt('estimated_size_gb', 2)} GB**{est_line}
- Request floor: {gfmt('request_chunks_estimate')} hourly chunks; at the
  package's 8 req/s rate limit the absolute floor is
  **{gfmt('estimated_hours_at_8_req_per_s_floor', 2)} h** of request time
  (resumable across sessions).

See `summary.json` -> `feasibility_estimate_full_range_2016_2024` for
assumptions.
"""
    (REPO_DIR / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
