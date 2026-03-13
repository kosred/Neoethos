#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]


TF_TO_MT5_CONST = {
    "M1": "TIMEFRAME_M1",
    "M2": "TIMEFRAME_M2",
    "M3": "TIMEFRAME_M3",
    "M4": "TIMEFRAME_M4",
    "M5": "TIMEFRAME_M5",
    "M6": "TIMEFRAME_M6",
    "M10": "TIMEFRAME_M10",
    "M12": "TIMEFRAME_M12",
    "M15": "TIMEFRAME_M15",
    "M20": "TIMEFRAME_M20",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H2": "TIMEFRAME_H2",
    "H3": "TIMEFRAME_H3",
    "H4": "TIMEFRAME_H4",
    "H6": "TIMEFRAME_H6",
    "H8": "TIMEFRAME_H8",
    "H12": "TIMEFRAME_H12",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
    "MN1": "TIMEFRAME_MN1",
}

TF_TO_MINUTES = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M4": 4,
    "M5": 5,
    "M6": 6,
    "M10": 10,
    "M12": 12,
    "M15": 15,
    "M20": 20,
    "M30": 30,
    "H1": 60,
    "H2": 120,
    "H3": 180,
    "H4": 240,
    "H6": 360,
    "H8": 480,
    "H12": 720,
    "D1": 1440,
    "W1": 10080,
    "MN1": 43200,
}


def _parse_csv(raw: str) -> list[str]:
    return [p.strip().upper() for p in str(raw or "").split(",") if p.strip()]


class _OhlcvFrame:
    def __init__(self, data: dict[str, Any] | None = None, *, index: Any | None = None) -> None:
        self.index = _to_datetime64_ns(index)
        self._data: dict[str, np.ndarray] = {}
        if data:
            for key, value in data.items():
                self._data[str(key)] = np.asarray(value).reshape(-1)

    @property
    def columns(self) -> list[str]:
        return list(self._data.keys())

    @property
    def empty(self) -> bool:
        return len(self.index) <= 0

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[str(key)]

    def copy(self) -> _OhlcvFrame:
        return _OhlcvFrame(
            {k: np.asarray(v).copy() for k, v in self._data.items()},
            index=np.asarray(self.index).copy(),
        )


def _empty_ohlcv_frame() -> _OhlcvFrame:
    return _OhlcvFrame(
        {
            "open": np.zeros(0, dtype=np.float64),
            "high": np.zeros(0, dtype=np.float64),
            "low": np.zeros(0, dtype=np.float64),
            "close": np.zeros(0, dtype=np.float64),
            "volume": np.zeros(0, dtype=np.float64),
        },
        index=np.zeros(0, dtype="datetime64[ns]"),
    )


def _to_datetime64_ns(values: Any, *, unit: str | None = None) -> np.ndarray:
    if values is None:
        return np.zeros(0, dtype="datetime64[ns]")
    try:
        arr = np.asarray(values).reshape(-1)
    except Exception:
        arr = np.asarray([values])
    if arr.size <= 0:
        return np.zeros(0, dtype="datetime64[ns]")
    try:
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype("datetime64[ns]")
        if unit == "s":
            ints = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
            return ints.astype("datetime64[s]").astype("datetime64[ns]")
        if arr.dtype.kind in {"i", "u"}:
            return arr.astype("datetime64[ns]")
    except Exception:
        pass
    out = np.empty(arr.size, dtype="datetime64[ns]")
    for i, value in enumerate(arr.tolist()):
        try:
            text = str(value or "").strip()
            if not text:
                out[i] = np.datetime64("NaT", "ns")
                continue
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            else:
                dt = dt.astimezone(UTC)
            out[i] = np.datetime64(dt.replace(tzinfo=None), "ns")
        except Exception:
            try:
                out[i] = np.datetime64(value, "ns")
            except Exception:
                out[i] = np.datetime64("NaT", "ns")
    return out


def _to_float64(values: Any, *, fill: float = np.nan) -> np.ndarray:
    try:
        arr = np.asarray(values).reshape(-1)
    except Exception:
        arr = np.asarray([values], dtype=object)
    if arr.size <= 0:
        return np.zeros(0, dtype=np.float64)
    if arr.dtype.kind in {"b", "i", "u", "f"}:
        out = arr.astype(np.float64, copy=False)
        if np.isfinite(fill):
            out = np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill)
        return out
    out = np.empty(arr.size, dtype=np.float64)
    for i, value in enumerate(arr.tolist()):
        try:
            out[i] = float(value)
        except Exception:
            out[i] = float(fill)
    return out


def _frame_mapping(frame: Any) -> tuple[dict[str, np.ndarray], Any | None]:
    if frame is None:
        return {}, None
    if isinstance(frame, dict):
        data = {str(k): np.asarray(v).reshape(-1) for k, v in frame.items()}
        idx = data.get("timestamp")
        return data, idx
    cols = getattr(frame, "columns", None)
    data: dict[str, np.ndarray] = {}
    if cols is not None:
        for col in list(cols):
            try:
                data[str(col)] = np.asarray(frame[col]).reshape(-1)  # type: ignore[index]
            except Exception:
                continue
    idx = getattr(frame, "index", None)
    return data, idx


def _ensure_ohlcv(frame: Any) -> _OhlcvFrame:
    data, idx_values = _frame_mapping(frame)
    if not data and idx_values is None:
        return _empty_ohlcv_frame()

    if "timestamp" in data:
        idx = _to_datetime64_ns(data.get("timestamp"))
    else:
        idx = _to_datetime64_ns(idx_values)
    if idx.size <= 0:
        return _empty_ohlcv_frame()

    open_arr = _to_float64(data.get("open"))
    high_arr = _to_float64(data.get("high"))
    low_arr = _to_float64(data.get("low"))
    close_arr = _to_float64(data.get("close"))
    volume_arr = _to_float64(data.get("volume"), fill=0.0)

    rows = min(idx.size, open_arr.size, high_arr.size, low_arr.size, close_arr.size)
    if rows <= 0:
        return _empty_ohlcv_frame()

    idx = idx[:rows]
    open_arr = open_arr[:rows]
    high_arr = high_arr[:rows]
    low_arr = low_arr[:rows]
    close_arr = close_arr[:rows]
    volume_arr = volume_arr[:rows]

    idx_ns = idx.astype("datetime64[ns]").astype(np.int64, copy=False)
    nat = np.iinfo(np.int64).min
    valid_idx = idx_ns != nat
    valid_ohlc = (
        np.isfinite(open_arr)
        & np.isfinite(high_arr)
        & np.isfinite(low_arr)
        & np.isfinite(close_arr)
    )
    valid = valid_idx & valid_ohlc
    if not np.any(valid):
        return _empty_ohlcv_frame()

    order = np.argsort(idx_ns, kind="mergesort")
    idx = idx[order]
    idx_ns = idx_ns[order]
    open_arr = open_arr[order]
    high_arr = high_arr[order]
    low_arr = low_arr[order]
    close_arr = close_arr[order]
    volume_arr = np.nan_to_num(volume_arr[order], nan=0.0, posinf=0.0, neginf=0.0)
    valid = valid[order]

    keep = np.ones(idx_ns.shape[0], dtype=bool)
    if idx_ns.shape[0] > 1:
        keep[:-1] = idx_ns[:-1] != idx_ns[1:]
    take = valid & keep
    if not np.any(take):
        return _empty_ohlcv_frame()

    return _OhlcvFrame(
        {
            "open": open_arr[take],
            "high": high_arr[take],
            "low": low_arr[take],
            "close": close_arr[take],
            "volume": volume_arr[take],
        },
        index=idx[take],
    )


def _concat_frames(frames: list[_OhlcvFrame]) -> _OhlcvFrame:
    parts = [frame for frame in frames if frame is not None and not frame.empty]
    if not parts:
        return _empty_ohlcv_frame()
    return _ensure_ohlcv(
        {
            "timestamp": np.concatenate([np.asarray(frame.index) for frame in parts], axis=0),
            "open": np.concatenate([np.asarray(frame["open"], dtype=np.float64) for frame in parts], axis=0),
            "high": np.concatenate([np.asarray(frame["high"], dtype=np.float64) for frame in parts], axis=0),
            "low": np.concatenate([np.asarray(frame["low"], dtype=np.float64) for frame in parts], axis=0),
            "close": np.concatenate([np.asarray(frame["close"], dtype=np.float64) for frame in parts], axis=0),
            "volume": np.concatenate([np.asarray(frame["volume"], dtype=np.float64) for frame in parts], axis=0),
        }
    )


def _to_utc(raw: str, *, default_now: bool = False) -> datetime:
    txt = str(raw or "").strip()
    if not txt and default_now:
        return datetime.now(UTC)
    dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _rates_to_df(rates: Any) -> _OhlcvFrame:
    if rates is None:
        return _empty_ohlcv_frame()
    arr = np.asarray(rates)
    if arr.size <= 0:
        return _empty_ohlcv_frame()
    names = list(arr.dtype.names or [])
    if not names:
        return _empty_ohlcv_frame()
    mapping = {str(name): np.asarray(arr[name]).reshape(-1) for name in names}
    real_volume = _to_float64(mapping.get("real_volume"), fill=0.0)
    tick_volume = _to_float64(mapping.get("tick_volume"), fill=0.0)
    volume = real_volume if real_volume.size > 0 and float(np.sum(real_volume)) > 0.0 else tick_volume
    if volume.size <= 0:
        volume = np.zeros(int(arr.shape[0]), dtype=np.float64)
    return _ensure_ohlcv(
        {
            "timestamp": mapping.get("time"),
            "open": mapping.get("open"),
            "high": mapping.get("high"),
            "low": mapping.get("low"),
            "close": mapping.get("close"),
            "volume": volume,
        }
    )


def _read_existing(parquet_path: Path, csv_path: Path) -> _OhlcvFrame:
    if parquet_path.exists():
        if pq is not None:
            try:
                table = pq.read_table(parquet_path)
                return _ensure_ohlcv(table.to_pydict())
            except Exception:
                pass
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            if rows:
                cols = {str(k): [row.get(k) for row in rows] for k in rows[0].keys()}
                return _ensure_ohlcv(cols)
        except Exception:
            pass
    return _empty_ohlcv_frame()


def _write_frame(df: _OhlcvFrame, parquet_path: Path, csv_path: Path, fmt: str) -> tuple[Path, str]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": np.asarray(df.index).astype("datetime64[ns]"),
        "open": np.asarray(df["open"], dtype=np.float64),
        "high": np.asarray(df["high"], dtype=np.float64),
        "low": np.asarray(df["low"], dtype=np.float64),
        "close": np.asarray(df["close"], dtype=np.float64),
        "volume": np.asarray(df["volume"], dtype=np.float64),
    }
    prefer_parquet = fmt in {"parquet", "auto"}
    if prefer_parquet and pq is not None and pa is not None:
        try:
            table = pa.table(payload)
            pq.write_table(table, parquet_path)
            return parquet_path, "parquet"
        except Exception:
            if fmt == "parquet":
                raise
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        for i in range(int(len(df))):
            writer.writerow(
                {
                    "timestamp": np.datetime_as_string(payload["timestamp"][i], unit="ns"),
                    "open": float(payload["open"][i]),
                    "high": float(payload["high"][i]),
                    "low": float(payload["low"][i]),
                    "close": float(payload["close"][i]),
                    "volume": float(payload["volume"][i]),
                }
            )
    return csv_path, "csv"


def _fetch_mt5_range(
    mt5: Any,
    *,
    symbol: str,
    mt5_tf: Any,
    tf_minutes: int,
    start_utc: datetime,
    end_utc: datetime,
    chunk_target_bars: int,
    sleep_ms: int,
) -> _OhlcvFrame:
    bars_per_day = max(1.0, 1440.0 / max(1, tf_minutes))
    chunk_days = max(1, int(chunk_target_bars / bars_per_day))
    step = timedelta(days=chunk_days)
    overlap = timedelta(minutes=max(1, tf_minutes))
    cursor = start_utc
    chunks: list[_OhlcvFrame] = []
    while cursor < end_utc:
        chunk_end = min(end_utc, cursor + step)
        try:
            rates = mt5.copy_rates_range(symbol, mt5_tf, cursor, chunk_end)
        except Exception:
            rates = None
        part = _rates_to_df(rates)
        if not part.empty:
            chunks.append(part)
        if chunk_end >= end_utc:
            break
        next_cursor = chunk_end - overlap
        if next_cursor <= cursor:
            next_cursor = chunk_end
        cursor = next_cursor
        if sleep_ms > 0:
            time.sleep(float(sleep_ms) / 1000.0)
    if not chunks:
        return _empty_ohlcv_frame()
    return _concat_frames(chunks)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync MT5 history into local data folder for training/forward tests.")
    p.add_argument("--symbols", required=True, help="Comma-separated symbols (e.g. EURUSD,GBPUSD,XAUUSD).")
    p.add_argument(
        "--timeframes",
        default="M1,M3,M5,M15,M30,H1,H2,H4,D1,W1,MN1",
        help="Comma-separated timeframes.",
    )
    p.add_argument("--from-date", default="2025-08-01", help="UTC cutoff start (e.g. 2025-08-01).")
    p.add_argument("--to-date", default="", help="UTC end (default: now).")
    p.add_argument("--data-dir", default="data", help="Local data root.")
    p.add_argument("--format", default="auto", choices=["auto", "parquet", "csv"], help="Output file format.")
    p.add_argument("--replace", type=int, default=0, help="Replace existing files (1) instead of merge append (0).")
    p.add_argument("--chunk-target-bars", type=int, default=120000, help="Approx bars per MT5 request chunk.")
    p.add_argument("--sleep-ms", type=int, default=0, help="Pause between MT5 chunk requests.")
    p.add_argument("--mt5-terminal-path", default="", help="Optional MT5 terminal executable path.")
    p.add_argument("--mt5-login", type=int, default=0, help="Optional MT5 account login.")
    p.add_argument("--mt5-password", default="", help="Optional MT5 account password.")
    p.add_argument("--mt5-server", default="", help="Optional MT5 server name.")
    p.add_argument("--mt5-timeout-ms", type=int, default=60000, help="MT5 initialize timeout in milliseconds.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    symbols = _parse_csv(args.symbols)
    timeframes = _parse_csv(args.timeframes)
    if not symbols or not timeframes:
        print("No symbols/timeframes provided.")
        return 2

    start_utc = _to_utc(args.from_date)
    end_utc = _to_utc(args.to_date, default_now=True)
    if end_utc <= start_utc:
        print("Invalid range: --to-date must be after --from-date.")
        return 2

    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception as exc:
        print(f"MetaTrader5 import failed: {exc}")
        return 2

    init_kwargs: dict[str, Any] = {
        "timeout": int(max(1000, args.mt5_timeout_ms)),
    }
    terminal_path = str(args.mt5_terminal_path or "").strip()
    if terminal_path:
        init_kwargs["path"] = terminal_path
    login_val = int(args.mt5_login or 0)
    if login_val > 0:
        init_kwargs["login"] = login_val
    pwd = str(args.mt5_password or "").strip()
    if pwd:
        init_kwargs["password"] = pwd
    server = str(args.mt5_server or "").strip()
    if server:
        init_kwargs["server"] = server

    print("Initializing MT5...", flush=True)
    if not mt5.initialize(**init_kwargs):
        try:
            last_err = mt5.last_error()
        except Exception:
            last_err = None
        print(f"MT5 initialize failed. last_error={last_err}")
        return 2
    print("MT5 initialized.", flush=True)

    data_dir = Path(args.data_dir)
    total_updated = 0
    total_rows_added = 0
    try:
        for symbol in symbols:
            try:
                mt5.symbol_select(symbol, True)
            except Exception:
                pass
            for tf in timeframes:
                const_name = TF_TO_MT5_CONST.get(tf)
                tf_minutes = TF_TO_MINUTES.get(tf)
                if not const_name or not tf_minutes:
                    print(f"[SKIP] {symbol} {tf}: unsupported timeframe")
                    continue
                mt5_tf = getattr(mt5, const_name, None)
                if mt5_tf is None:
                    print(f"[SKIP] {symbol} {tf}: MT5 constant missing ({const_name})")
                    continue

                print(f"[FETCH] {symbol} {tf}: {start_utc.isoformat()} -> {end_utc.isoformat()}", flush=True)
                fetched = _fetch_mt5_range(
                    mt5,
                    symbol=symbol,
                    mt5_tf=mt5_tf,
                    tf_minutes=tf_minutes,
                    start_utc=start_utc,
                    end_utc=end_utc,
                    chunk_target_bars=int(max(1000, args.chunk_target_bars)),
                    sleep_ms=int(max(0, args.sleep_ms)),
                )
                if fetched.empty:
                    print(f"[MISS] {symbol} {tf}: no bars in range")
                    continue

                parquet_path = data_dir / f"symbol={symbol}" / f"timeframe={tf}" / "data.parquet"
                csv_path = data_dir / f"symbol={symbol}" / f"timeframe={tf}" / "data.csv"
                before = _empty_ohlcv_frame()
                if int(args.replace) <= 0:
                    before = _read_existing(parquet_path, csv_path)
                merged = fetched if int(args.replace) > 0 else _concat_frames([before, fetched])
                rows_added = int(max(0, len(merged) - len(before)))
                out_path, out_fmt = _write_frame(merged, parquet_path, csv_path, args.format)
                first_ts = str(merged.index.min()) if not merged.empty else "n/a"
                last_ts = str(merged.index.max()) if not merged.empty else "n/a"
                print(
                    f"[OK] {symbol} {tf}: rows_total={len(merged)} rows_added={rows_added} "
                    f"first={first_ts} last={last_ts} file={out_path} ({out_fmt})"
                )
                total_updated += 1
                total_rows_added += rows_added
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

    print(f"Done. updated_series={total_updated} total_rows_added={total_rows_added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
