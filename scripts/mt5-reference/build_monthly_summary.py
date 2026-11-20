#!/usr/bin/env python3
"""
Streaming per-month sampler for the EURUSD MT5 reference export (~14.5 GB CSV/TSV).

Requirements honored:
  * Python stdlib only (csv module), row-streamed over a large buffered text
    stream - the file is never loaded into memory and never copied.
  * One compact entry per month: rows, time range, spread stats (or OHLC range
    stats for bar exports), bar/row counts.
  * Writes monthly-summary.json into %LOCALAPPDATA%/neoethos/mt5-reference/
    and a README.md documenting schema, assumptions and the venue label.

Stat selection (task spec):
  * SPREAD column present            -> mean/median/p95/p99 spread in points
  * else BID+ASK columns present     -> derived (ask-bid) spread in points
                                        (only rows with both quotes > 0)
  * else OHLC columns present        -> mean((high-low)/close) in pips and
                                        mean true range in pips
  * row counts and time range are always reported per month.
"""
import collections
import csv
import json
import math
import os
import sys
import time
import traceback

SOURCE = r"C:\Users\konst\OneDrive\Desktop\EURUSD_201601040000_202610231254.csv"
OUT_DIR = os.path.join(os.environ.get("LOCALAPPDATA", ""), "neoethos", "mt5-reference")
OUT_JSON = os.path.join(OUT_DIR, "monthly-summary.json")
README_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "README.md")

PIP = 10000.0     # EURUSD pip = 0.0001
POINT = 100000.0  # EURUSD point = 0.00001 (1 point = 0.1 pip)
PROGRESS_EVERY = 2000000
IO_BUFFER = 8 * 1024 * 1024  # 8 MiB chunked buffered reads

VENUE_LABEL = "MT5/Dukascopy reference venue \u2014 not NeoEthos execution truth"


def norm(name):
    return (name or "").strip().strip('<>"').strip().upper()


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def detect_delimiter(path):
    with open(path, "rb") as f:
        head = f.read(65536).decode("utf-8", "replace")
    first = head.splitlines()[0] if head.splitlines() else ""
    counts = {d: first.count(d) for d in ("\t", ",", ";")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def first_lines(path, k=3):
    out = []
    with open(path, "rb") as f:
        for _ in range(k):
            b = f.readline()
            if not b:
                break
            out.append(b.decode("utf-8", "replace").rstrip("\r\n"))
    return out


def month_key(date_str, time_str=""):
    d = (date_str or "").strip()
    digits = "".join(ch for ch in d if ch.isdigit())
    if len(digits) >= 8:
        return digits[0:4] + "-" + digits[4:6]
    for sep in ("-", "/", "."):
        if sep in d:
            parts = [p for p in d.split(sep) if p]
            if len(parts) >= 2:
                return parts[0] + "-" + parts[1].zfill(2)
    return d or "unknown"


def ts_iso(date_str, time_str=""):
    d = (date_str or "").strip()
    t = (time_str or "").strip()
    digits = "".join(ch for ch in d if ch.isdigit())
    if len(digits) >= 8:
        s = digits[0:4] + "-" + digits[4:6] + "-" + digits[6:8]
        if t:
            s += "T" + t
        return s
    return (d + " " + t).strip()


def ts_iso_from_raw(raw):
    parts = (raw or "").split(None, 1)
    return ts_iso(parts[0] if parts else "", parts[1] if len(parts) > 1 else "")


def median_from_counter(counter, n):
    if n == 0:
        return None
    ordered = sorted(counter.items())
    if n % 2 == 1:
        target = n // 2 + 1
        cum = 0
        for v, c in ordered:
            cum += c
            if cum >= target:
                return v
    else:
        t1, t2 = n // 2, n // 2 + 1
        cum, v1 = 0, None
        for v, c in ordered:
            cum += c
            if v1 is None and cum >= t1:
                v1 = v
            if cum >= t2:
                return (v1 + v) / 2.0
        return v1
    return None


def quantile_from_counter(counter, n, q):
    if n == 0:
        return None
    target = math.ceil(q * n)  # exact nearest-rank
    cum = 0
    for v, c in sorted(counter.items()):
        cum += c
        if cum >= target:
            return v
    return None


def write_readme(report, raw_first3):
    lines = []
    a = lines.append
    a("# MT5 reference venue \u2014 EURUSD export monthly sampling")
    a("")
    a("> **Venue label: " + VENUE_LABEL + "**")
    a("")
    a("This dataset is a reference/historical venue sample used for spread and quote-economics")
    a("research only. It is **not** NeoEthos execution truth and must never be mixed into the")
    a("sealed cost model.")
    a("")
    a("## Source file")
    a("")
    a("- Path: '" + report["source_file"] + "'")
    a("- Size: %s bytes (%.2f GiB)" % (format(report["file_size_bytes"], ","), report["file_size_bytes"] / 2 ** 30))
    a("- Processed at: " + report["generated_at"] + " (UTC)")
    a("- Delimiter: '" + report["delimiter"] + "'")
    a("")
    a("## Detected schema (first 3 lines, verbatim; tabs shown as \\t)")
    a("")
    for i, ln in enumerate(raw_first3, 1):
        a("    L%d: %s" % (i, ln.replace("\t", "\\t")))
    a("")
    a("- Header row detected: '%s'" % report["header_detected"])
    a("- Columns (normalized): '%s'" % ", ".join(report["schema_columns"]))
    a("- Granularity: '%s' (stat mode '%s')" % (report["granularity"], report["stat_mode"]))
    a("")
    a("## Column meaning assumptions")
    a("")
    a("- DATE / TIME: tick timestamp as exported by the MT5 terminal (yyyy.mm.dd, HH:mm:ss.mmm);")
    a("  timezone is the export terminal's own and is not normalized.")
    a("- BID / ASK: best bid/ask at the tick; rows where either is empty or zero carry no quote")
    a("  (deal/session rows) and are excluded from spread stats but still counted as rows.")
    a("- LAST: last deal price (empty on quote-only rows).")
    a("- VOLUME: deal volume; FLAGS: MT5 tick flags bitmask.")
    a("- OPEN/HIGH/LOW/CLOSE (bar exports): OHLC prices.")
    a("- SPREAD (bar exports): spread in integer points.")
    a("- TICKVOL / VOL (bar exports): tick volume / real volume.")
    a("")
    a("## Statistics (per month, in monthly-summary.json)")
    a("")
    if report["stat_mode"] in ("spread", "bidask_spread"):
        a("- spread_n: rows used for spread stats (both BID and ASK > 0).")
        a("- spread_mean_points / spread_median_points / spread_p95_points / spread_p99_points.")
        a("- Spread unit: points, derived as (ASK-BID) * 100000 (EURUSD point = 0.00001; 1 point = 0.1 pip).")
        a("- Median is the exact middle value (average of the two middle values when even); p95/p99 use")
        a("  exact nearest-rank over per-month value counts \u2014 no sampling, no approximation.")
    elif report["stat_mode"] == "ohlc":
        a("- range_pips_mean: mean of (high-low)/close * 10000 (EURUSD pip = 0.0001).")
        a("- true_range_pips_mean: mean true range = max(high-low, |high-prev_close|, |low-prev_close|) / close * 10000.")
        a("- First bar of the file uses high-low as its true range (no previous close available).")
    a("- rows: number of rows (tick records or bars) in the month.")
    a("- bars: row count for bar exports, null for tick exports (no bars exist in a tick export).")
    a("- time_start / time_end: first/last timestamp of the month (ISO-ish, as exported).")
    a("")
    a("- Total rows: %s across %d months (%s .. %s)." % (format(report["rows_total"], ","), report["months_covered"], report["month_first"], report["month_last"]))
    a("- Rows counted but excluded from stats (empty/zero quotes or unparsable numbers): %d" % report["stat_skipped_rows"])
    if report["median_monthly_spread_mean_points"] is not None:
        a("- Median of monthly spread means: %s points" % report["median_monthly_spread_mean_points"])
    a("")
    a("## Method")
    a("")
    a("- Python stdlib csv only; row-streamed over an 8 MiB buffered text stream. The large source")
    a("  file is never loaded into memory and never copied.")
    a("- Memory use is O(number of months) plus a small per-month value counter (exact quantiles).")
    a("- Output: '%s'" % report["output_json"])
    a("- Elapsed: %.1f s" % report["elapsed_seconds"])
    a("")
    a("## Verification limits")
    a("")
    a("- Statistics were not cross-checked against MT5 terminal totals or an independent source.")
    a("- The file was exported at 2026-10-23 ~13:03 local (LastWriteTime); if the export was still")
    a("  running, this snapshot covers only what was on disk when the scan started.")
    a("- Derived spread (BID/ASK mode) assumes 5-digit EURUSD quotes; if the terminal exported")
    a("  3-digit quotes the point scale would be off by 100x.")
    os.makedirs(os.path.dirname(README_PATH), exist_ok=True)
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    if not os.path.exists(SOURCE):
        raise FileNotFoundError("file missing: " + SOURCE)
    size = os.path.getsize(SOURCE)
    delim = detect_delimiter(SOURCE)
    raw_first3 = first_lines(SOURCE, 3)

    with open(SOURCE, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        first_row = next(csv.reader(f, delimiter=delim), None)
    header_detected = False
    if first_row:
        names = [norm(c) for c in first_row]
        known = {"DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "BID", "ASK",
                 "TICKVOL", "SPREAD", "VOL", "LAST", "VOLUME", "FLAGS"}
        if any(n in known for n in names):
            header_detected = True
            header_row = names
        else:
            defaults = ["DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "TICKVOL", "SPREAD", "VOL"]
            header_row = [(defaults[i] if i < len(defaults) else "COL%d" % (i + 1)) for i in range(len(first_row))]
    else:
        header_row = []

    idx = {name: i for i, name in enumerate(header_row)}
    cols = list(header_row)
    has_spread = "SPREAD" in idx
    has_ohlc = all(c in idx for c in ("OPEN", "HIGH", "LOW", "CLOSE"))
    has_bidask = "BID" in idx and "ASK" in idx
    if has_spread:
        mode = "spread"
    elif has_bidask:
        mode = "bidask_spread"
    elif has_ohlc:
        mode = "ohlc"
    else:
        mode = "counts"
    granularity = "bars" if (has_ohlc or "TICKVOL" in idx) else ("ticks" if has_bidask else "unknown")

    months = {}
    total = 0
    stat_skip = 0  # rows counted but excluded from stats
    prev_close = None
    started = time.time()

    with open(SOURCE, "r", encoding="utf-8-sig", errors="replace", newline="", buffering=IO_BUFFER) as f:
        reader = csv.reader(f, delimiter=delim)
        if header_detected:
            next(reader, None)
        for row in reader:
            total += 1
            if total % PROGRESS_EVERY == 0:
                el = time.time() - started
                print("[progress] rows=%d elapsed_s=%.0f rate=%.0f rows/s" % (total, el, total / el), flush=True)

            date_str = row[idx["DATE"]] if "DATE" in idx and len(row) > idx["DATE"] else ""
            time_str = row[idx["TIME"]] if "TIME" in idx and len(row) > idx["TIME"] else ""
            mk = month_key(date_str, time_str)
            m = months.get(mk)
            if m is None:
                m = {"month": mk, "rows": 0, "time_start": None, "time_end": None,
                     "s_sum": 0.0, "s_n": 0, "s_counter": collections.Counter(),
                     "r_sum": 0.0, "tr_sum": 0.0, "o_n": 0}
                months[mk] = m
            m["rows"] += 1
            raw_ts = (date_str + " " + time_str).strip()
            if m["time_start"] is None or raw_ts < m["time_start"]:
                m["time_start"] = raw_ts
            if m["time_end"] is None or raw_ts > m["time_end"]:
                m["time_end"] = raw_ts

            if has_spread:
                sv = fnum(row[idx["SPREAD"]]) if len(row) > idx["SPREAD"] else None
                if sv is not None:
                    m["s_sum"] += sv
                    m["s_n"] += 1
                    if float(sv).is_integer():
                        m["s_counter"][int(sv)] += 1
                    else:
                        m["s_counter"][round(sv, 2)] += 1
                else:
                    stat_skip += 1
            elif has_bidask:
                b = fnum(row[idx["BID"]]) if len(row) > idx["BID"] else None
                a = fnum(row[idx["ASK"]]) if len(row) > idx["ASK"] else None
                if b is not None and a is not None and b > 0 and a > 0 and a >= b:
                    sp = int(round((a - b) * POINT))
                    m["s_sum"] += sp
                    m["s_n"] += 1
                    m["s_counter"][sp] += 1
                else:
                    stat_skip += 1

            if has_ohlc:
                h = fnum(row[idx["HIGH"]]) if len(row) > idx["HIGH"] else None
                l = fnum(row[idx["LOW"]]) if len(row) > idx["LOW"] else None
                c = fnum(row[idx["CLOSE"]]) if len(row) > idx["CLOSE"] else None
                if h is not None and l is not None and c is not None and c > 0:
                    m["r_sum"] += (h - l) / c * PIP
                    tr = h - l
                    if prev_close is not None:
                        tr = max(tr, abs(h - prev_close), abs(l - prev_close))
                    m["tr_sum"] += tr / c * PIP
                    m["o_n"] += 1
                    prev_close = c
                else:
                    stat_skip += 1

    elapsed = time.time() - started

    def build_entry(m):
        e = {
            "month": m["month"],
            "rows": m["rows"],
            "bars": m["rows"] if granularity == "bars" else None,
            "time_start": ts_iso_from_raw(m["time_start"]),
            "time_end": ts_iso_from_raw(m["time_end"]),
        }
        if mode in ("spread", "bidask_spread"):
            n = m["s_n"]
            e["spread_n"] = n
            e["spread_mean_points"] = round(m["s_sum"] / n, 4) if n else None
            e["spread_median_points"] = median_from_counter(m["s_counter"], n)
            e["spread_p95_points"] = quantile_from_counter(m["s_counter"], n, 0.95)
            e["spread_p99_points"] = quantile_from_counter(m["s_counter"], n, 0.99)
        if mode == "ohlc":
            n = m["o_n"]
            e["range_pips_mean"] = round(m["r_sum"] / n, 6) if n else None
            e["true_range_pips_mean"] = round(m["tr_sum"] / n, 6) if n else None
        return e

    monthly = [build_entry(months[k]) for k in sorted(months)]

    med_monthly_spread = None
    if mode in ("spread", "bidask_spread"):
        vals = sorted(e["spread_mean_points"] for e in monthly if e.get("spread_mean_points") is not None)
        if vals:
            n = len(vals)
            med_monthly_spread = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_file": SOURCE,
        "file_size_bytes": size,
        "delimiter": "\\t" if delim == "\t" else delim,
        "header_detected": header_detected,
        "schema_columns": cols,
        "granularity": granularity,
        "stat_mode": mode,
        "rows_total": total,
        "months_covered": len(months),
        "month_first": min(months) if months else None,
        "month_last": max(months) if months else None,
        "stat_skipped_rows": stat_skip,
        "median_monthly_spread_mean_points": med_monthly_spread,
        "elapsed_seconds": round(elapsed, 1),
        "output_json": OUT_JSON,
        "monthly": monthly,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    write_readme(report, raw_first3)
    print("SUMMARY " + json.dumps({
        "months": report["months_covered"], "first": report["month_first"],
        "last": report["month_last"], "rows": report["rows_total"],
        "granularity": granularity, "mode": mode,
        "median_monthly_spread_mean_points": med_monthly_spread,
        "out_json": OUT_JSON, "readme": README_PATH,
        "elapsed_s": round(elapsed, 1),
    }), flush=True)
    return report


def main():
    try:
        run()
    except Exception as exc:
        err_text = "%s: %s\n%s" % (type(exc).__name__, exc, traceback.format_exc())
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump({"error": str(exc), "traceback": err_text, "source_file": SOURCE,
                       "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=1)
        os.makedirs(os.path.dirname(README_PATH), exist_ok=True)
        with open(README_PATH, "w", encoding="utf-8") as f:
            f.write("# MT5 reference venue \u2014 sampling FAILED\n\n")
            f.write("> **Venue label: " + VENUE_LABEL + "**\n\n")
            f.write("Source file: '%s'\n\n" % SOURCE)
            f.write("The scan could not complete. Exact error:\n\n")
            for ln in err_text.splitlines():
                f.write("    " + ln + "\n")
        print("ERROR " + json.dumps({"error": str(exc), "out_json": OUT_JSON, "readme": README_PATH}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
