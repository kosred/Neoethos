#!/usr/bin/env python3
"""Build a versioned NeoEthos GPU benchmark snapshot from canonical CSV.

Required columns: timestamp, high, low, close. Every additional numeric column is
used as a feature in header order. Timestamp may be unix seconds or milliseconds.
The deterministic candidate population is generated from the feature set so the
same input and arguments always produce byte-identical JSON.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
from typing import Any

SCHEMA_VERSION = 1
SMC_WIDTH = 11


def finite(value: str, label: str, row: int) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise SystemExit(f"row {row}: {label} is not numeric: {value!r}") from error
    if not math.isfinite(parsed):
        raise SystemExit(f"row {row}: {label} is not finite")
    return parsed


def timestamp_ms(value: str, row: int) -> int:
    parsed = int(finite(value, "timestamp", row))
    if abs(parsed) < 10_000_000_000:
        parsed *= 1000
    return parsed


def month_id(timestamp: int) -> int:
    instant = dt.datetime.fromtimestamp(timestamp / 1000, tz=dt.timezone.utc)
    return instant.year * 12 + instant.month - 1


def deterministic_genes(population: int, feature_count: int, terms: int) -> dict[str, Any]:
    terms = max(1, min(terms, feature_count))
    offsets = [0]
    indices: list[int] = []
    weights: list[float] = []
    long_thresholds: list[float] = []
    short_thresholds: list[float] = []
    for candidate in range(population):
        for term in range(terms):
            indices.append((candidate + term * 3) % feature_count)
            magnitude = 0.35 + ((candidate + term) % 5) * 0.11
            weights.append(magnitude if (candidate + term) % 2 == 0 else -magnitude)
        offsets.append(len(indices))
        threshold = 0.20 + (candidate % 3) * 0.03
        long_thresholds.append(threshold)
        short_thresholds.append(-threshold)
    return {
        "gene_offsets": offsets,
        "gene_indices": indices,
        "gene_weights": weights,
        "long_thresholds": long_thresholds,
        "short_thresholds": short_thresholds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--terms-per-gene", type=int, default=4)
    parser.add_argument("--stop-pips", type=float, default=18.0)
    parser.add_argument("--target-pips", type=float, default=36.0)
    parser.add_argument("--max-hold-bars", type=int, default=12)
    parser.add_argument("--max-trades-per-day", type=int, default=20)
    parser.add_argument("--pip-value", type=float, default=0.0001)
    parser.add_argument("--spread-pips", type=float, default=0.0)
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument("--pip-value-per-lot", type=float, default=10.0)
    args = parser.parse_args()

    if args.population <= 0:
        raise SystemExit("--population must be positive")
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    timestamps: list[int] = []
    high: list[float] = []
    low: list[float] = []
    close: list[float] = []
    feature_columns: list[str] | None = None
    feature_rows: list[list[float]] = []
    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("CSV has no header")
        canonical = {name.strip().lower(): name for name in reader.fieldnames}
        required = ("timestamp", "high", "low", "close")
        missing = [name for name in required if name not in canonical]
        if missing:
            raise SystemExit(f"CSV missing required columns: {', '.join(missing)}")
        excluded = {canonical[name] for name in required}
        feature_columns = [name for name in reader.fieldnames if name not in excluded]
        if not feature_columns:
            raise SystemExit("CSV must contain at least one numeric feature column")
        for row_number, row in enumerate(reader, start=2):
            timestamps.append(timestamp_ms(row[canonical["timestamp"]], row_number))
            high.append(finite(row[canonical["high"]], "high", row_number))
            low.append(finite(row[canonical["low"]], "low", row_number))
            close.append(finite(row[canonical["close"]], "close", row_number))
            feature_rows.append(
                [finite(row[name], name, row_number) for name in feature_columns]
            )

    if len(close) < 64:
        raise SystemExit("snapshot CSV must contain at least 64 bars")
    if any(timestamps[index] <= timestamps[index - 1] for index in range(1, len(timestamps))):
        raise SystemExit("timestamps must be strictly increasing")
    if any(lo > hi for lo, hi in zip(low, high, strict=True)):
        raise SystemExit("low exceeds high in at least one row")

    feature_count = len(feature_columns)
    indicators = [
        feature_rows[bar][feature]
        for feature in range(feature_count)
        for bar in range(len(close))
    ]
    genes = deterministic_genes(args.population, feature_count, args.terms_per_gene)
    months = [month_id(value) for value in timestamps]
    days = [value // 86_400_000 for value in timestamps]
    source_hash = hashlib.sha256(args.csv.read_bytes()).hexdigest()

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timeframe": args.timeframe.upper(),
        "source_description": f"{args.csv.name} sha256={source_hash}",
        "close": close,
        "high": high,
        "low": low,
        "indicators": indicators,
        "feature_count": feature_count,
        **genes,
        "months": months,
        "days": days,
        "timestamps": timestamps,
        "stop_pips": [args.stop_pips] * args.population,
        "target_pips": [args.target_pips] * args.population,
        "stop_vol_multipliers": [0.0] * args.population,
        "smc_data": [[0] * SMC_WIDTH for _ in close],
        "gene_smc_flags": [[0] * SMC_WIDTH for _ in range(args.population)],
        "smc_weights": [0.0] * SMC_WIDTH,
        "settings": {
            "max_hold_bars": args.max_hold_bars,
            "min_hold_bars": 0,
            "max_trades_per_day": args.max_trades_per_day,
            "gap_threshold_ms": 0,
            "trailing_enabled": False,
            "trailing_atr_multiplier": 0.0,
            "trailing_be_trigger_r": 0.0,
            "pip_value": args.pip_value,
            "spread_pips": args.spread_pips,
            "commission_per_trade": args.commission,
            "pip_value_per_lot": args.pip_value_per_lot,
            "swap_long_pips_per_day": 0.0,
            "swap_short_pips_per_day": 0.0,
            "pnl_conversion_fee_rate": 0.0,
            "risk_based_sizing": True,
            "risk_per_trade_min": 0.005,
            "risk_per_trade_max": 0.01,
            "high_quality_confidence": 0.65,
            "adaptive_base_pips": None,
            "adaptive_rr": 2.0,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    args.out.write_bytes(encoded)
    snapshot_hash = hashlib.sha256(encoded).hexdigest()
    print(json.dumps({
        "snapshot": str(args.out),
        "sha256": snapshot_hash,
        "bars": len(close),
        "features": feature_count,
        "population": args.population,
        "timeframe": args.timeframe.upper(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
