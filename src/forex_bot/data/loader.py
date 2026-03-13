from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import ALL_TIMEFRAMES, Settings

logger = logging.getLogger(__name__)
_RUST_DATA_BACKEND_OK: bool | None = None
_RUST_DATA_WARNED_UNAVAILABLE = False
_OHLCV_COLUMN_ALIASES = {
    "timestamp",
    "time",
    "datetime",
    "date",
    "o",
    "open",
    "h",
    "high",
    "l",
    "low",
    "c",
    "close",
    "v",
    "vol",
    "volume",
    "tick_volume",
    "real_volume",
}
_FRAME_IO_WARNED_UNKNOWN = False


class _RustFrame:
    """Lightweight frame for strict frame-native Rust payloads."""

    def __init__(self, data: dict[str, np.ndarray], index: np.ndarray):
        self._data = {str(k): np.asarray(v) for k, v in data.items()}
        self.index = np.asarray(index)
        self.columns = list(self._data.keys())
        self.attrs: dict[str, Any] = {}

    @property
    def empty(self) -> bool:
        return len(self.index) <= 0

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[str(key)]

    def copy(self) -> "_RustFrame":
        out = _RustFrame({k: np.asarray(v).copy() for k, v in self._data.items()}, np.asarray(self.index).copy())
        out.attrs = dict(self.attrs)
        return out

    def tail(self, n: int) -> "_RustFrame":
        take = max(0, int(n))
        if take <= 0:
            out = _RustFrame({k: v[:0] for k, v in self._data.items()}, self.index[:0])
            out.attrs = dict(self.attrs)
            return out
        out = _RustFrame({k: v[-take:] for k, v in self._data.items()}, self.index[-take:])
        out.attrs = dict(self.attrs)
        return out


def _to_datetime64_ns(values: Any) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if arr.size <= 0:
        return np.zeros(0, dtype="datetime64[ns]")
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]", copy=False)
    if arr.dtype.kind in {"i", "u"}:
        vals = arr.astype(np.int64, copy=False)
        vmax = int(np.max(np.abs(vals))) if vals.size > 0 else 0
        if vmax > 10**14:
            return vals.astype("datetime64[ns]")
        if vmax > 10**11:
            return vals.astype("datetime64[ms]").astype("datetime64[ns]")
        return vals.astype("datetime64[s]").astype("datetime64[ns]")
    if arr.dtype.kind == "f":
        vals = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
        return vals.astype("datetime64[s]").astype("datetime64[ns]")
    with np.errstate(all="ignore"):
        try:
            return arr.astype("datetime64[ns]")
        except Exception:
            return np.arange(arr.size, dtype=np.int64).astype("datetime64[s]").astype("datetime64[ns]")


def _normalize_rust_payload_columns(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rename: dict[str, str] = {}
    for col in data.keys():
        low = str(col).lower()
        if low in {"o", "open"}:
            rename[col] = "open"
        elif low in {"h", "high"}:
            rename[col] = "high"
        elif low in {"l", "low"}:
            rename[col] = "low"
        elif low in {"c", "close"}:
            rename[col] = "close"
        elif low in {"v", "vol", "volume"}:
            rename[col] = "volume"
        elif low in {"tick_volume", "real_volume"}:
            rename[col] = "volume"
        elif low in {"timestamp", "time", "datetime", "date"}:
            rename[col] = "timestamp"
    out: dict[str, np.ndarray] = {}
    for src, vals in data.items():
        out[rename.get(src, src)] = np.asarray(vals)
    return out


def _is_rust_frame(obj: Any) -> bool:
    return isinstance(obj, _RustFrame)


def _is_frame_like(obj: Any) -> bool:
    return bool(
        _is_dataframe(obj)
        or _is_rust_frame(obj)
        or (hasattr(obj, "columns") and hasattr(obj, "index") and hasattr(obj, "__getitem__"))
    )


def _frame_empty(obj: Any) -> bool:
    try:
        return bool(obj is None or obj.empty)
    except Exception:
        return True


def _fit_len_array(values: Any, n: int) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    target = max(0, int(n))
    if arr.size == target:
        return arr
    if arr.size <= 0:
        return np.zeros(target, dtype=np.float64)
    if arr.size > target:
        return arr[:target]
    pad = np.full(target - arr.size, arr[-1], dtype=arr.dtype)
    return np.concatenate([arr, pad])


def _to_rust_frame(df: Any) -> _RustFrame | None:
    if _is_rust_frame(df):
        return df
    if df is None or _frame_empty(df):
        return None
    if not _is_frame_like(df):
        return None

    cols = getattr(df, "columns", None)
    try:
        col_list = [str(c) for c in list(cols or [])]
    except Exception:
        col_list = []
    if not col_list:
        return None

    raw: dict[str, np.ndarray] = {}
    for col in col_list:
        try:
            src = df[col]  # type: ignore[index]
        except Exception:
            continue
        try:
            vals = src.to_numpy(copy=False) if hasattr(src, "to_numpy") else np.asarray(src)
            raw[str(col)] = np.asarray(vals).reshape(-1)
        except Exception:
            continue
    if not raw:
        return None

    data = _normalize_rust_payload_columns(raw)
    if "close" not in data:
        return None

    close_arr = np.asarray(data["close"]).reshape(-1)
    n = int(close_arr.shape[0])
    if n <= 0:
        return None

    ts_src = None
    for key in ("timestamp", "time", "datetime", "date"):
        if key in data:
            ts_src = data.get(key)
            break
    idx_src = ts_src if ts_src is not None else getattr(df, "index", None)
    idx_np = _to_datetime64_ns(idx_src if idx_src is not None else np.arange(n, dtype=np.int64))
    idx_np = _fit_len_array(idx_np, n).astype("datetime64[ns]")

    clean = {
        k: _fit_len_array(v, n)
        for k, v in data.items()
        if str(k).lower() not in {"timestamp", "time", "datetime", "date"}
    }
    clean = _ensure_ohlcv_arrays(clean)
    out = _RustFrame(clean, idx_np)
    try:
        attrs = getattr(df, "attrs", None)
        if isinstance(attrs, dict):
            out.attrs = dict(attrs)
    except Exception:
        pass
    return out


def _build_frame_from_arrays(
    data: dict[str, np.ndarray],
    *,
    index: np.ndarray,
    allow_tabular_module: bool = False,
) -> Any:
    _ = allow_tabular_module
    idx_np = _to_datetime64_ns(index)
    clean = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
    return _RustFrame(clean, idx_np)


def _ensure_ohlcv_arrays(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
    close = out.get("close")
    if close is None:
        return out
    close_arr = np.asarray(close).reshape(-1)
    n = int(close_arr.shape[0])
    if n <= 0:
        return out
    for col in ("open", "high", "low"):
        if col not in out:
            out[col] = close_arr.copy()
    if "volume" not in out:
        out["volume"] = np.zeros(n, dtype=np.float64)
    return out


def _strict_rust_data_mode_enabled() -> bool:
    rust_only = str(os.environ.get("FOREX_BOT_RUST_ONLY", "") or "").strip().lower()
    if rust_only in {"1", "true", "yes", "on"}:
        return True
    mode = str(os.environ.get("FOREX_BOT_DATA_BACKEND", "") or "").strip().lower()
    if mode in {"rust_strict", "strict_rust", "rust_only", "rust-only"}:
        return True
    pandas_free = str(os.environ.get("FOREX_BOT_PANDAS_FREE", "1") or "1").strip().lower()
    if pandas_free in {"1", "true", "yes", "on"}:
        return True
    runtime_profile = str(os.environ.get("FOREX_BOT_RUNTIME_PROFILE", "") or "").strip().lower()
    if runtime_profile.startswith("rust"):
        return True
    return False


def _strict_tabular_free_enabled() -> bool:
    raw = str(os.environ.get("FOREX_BOT_PANDAS_FREE_STRICT", "1") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _tabular_module(*, required: bool = True):
    _ = required
    return None


def _is_dataframe(obj: Any) -> bool:
    return bool(hasattr(obj, "columns") and hasattr(obj, "index"))


def _is_datetime_index(obj: Any) -> bool:
    return bool(
        hasattr(obj, "tz")
        and hasattr(obj, "tz_localize")
        and hasattr(obj, "tz_convert")
        and (hasattr(obj, "year") or hasattr(obj, "asi8"))
    )


def _rust_data_backend_available(*, force_log: bool = False) -> bool:
    global _RUST_DATA_BACKEND_OK, _RUST_DATA_WARNED_UNAVAILABLE
    if _RUST_DATA_BACKEND_OK is None:
        try:
            import forex_bindings  # type: ignore

            _RUST_DATA_BACKEND_OK = hasattr(forex_bindings, "load_symbol_frames")
        except Exception:
            _RUST_DATA_BACKEND_OK = False
    if force_log and not _RUST_DATA_BACKEND_OK and not _RUST_DATA_WARNED_UNAVAILABLE:
        logger.warning(
            "Rust data backend requested but forex_bindings.load_symbol_frames is unavailable."
        )
        _RUST_DATA_WARNED_UNAVAILABLE = True
    return bool(_RUST_DATA_BACKEND_OK)


def _disable_rust_data_backend() -> None:
    global _RUST_DATA_BACKEND_OK
    _RUST_DATA_BACKEND_OK = False


def _frame_io_backend() -> str:
    """
    Select the frame I/O backend for local history files.

    Env (preferred): FOREX_BOT_FRAME_IO_BACKEND=auto|polars|pyarrow|python
    Back-compat:     FOREX_BOT_DATA_IO_BACKEND=...
    """
    global _FRAME_IO_WARNED_UNKNOWN
    raw = os.environ.get("FOREX_BOT_FRAME_IO_BACKEND")
    if raw is None or str(raw).strip() == "":
        raw = os.environ.get("FOREX_BOT_DATA_IO_BACKEND", "auto")
    mode = str(raw).strip().lower()
    aliases = {
        "auto": "auto",
        "detect": "auto",
        "pl": "polars",
        "polars": "polars",
        "arrow": "pyarrow",
        "pa": "pyarrow",
        "pyarrow": "pyarrow",
        "python": "python",
        "legacy": "python",
    }
    resolved = aliases.get(mode)
    if resolved is not None:
        if _strict_tabular_free_enabled() and resolved == "python":
            return "auto"
        return resolved
    if not _FRAME_IO_WARNED_UNKNOWN:
        logger.warning("Unknown frame I/O backend '%s'; using auto.", mode)
        _FRAME_IO_WARNED_UNKNOWN = True
    return "auto"


def _use_rust_data_backend() -> bool:
    raw = os.environ.get("FOREX_BOT_RUST_DATA")
    if raw is not None and str(raw).strip() != "":
        mode = str(raw).strip().lower()
        if mode in {"auto", "detect"}:
            return _rust_data_backend_available()
        enabled = mode in {"1", "true", "yes", "on", "rust"}
        return enabled and _rust_data_backend_available(force_log=True)
    mode = str(os.environ.get("FOREX_BOT_DATA_BACKEND", "auto")).strip().lower()
    if mode in {"rust", "rs", "1", "true", "yes", "on"}:
        return _rust_data_backend_available(force_log=True)
    if mode in {"python", "py", "0", "false", "no", "off"}:
        return False
    return _rust_data_backend_available()


def _timeframe_to_freq(tf: str) -> str | None:
    tf = str(tf or "").upper()
    return {
        "M1": "1min",
        "M2": "2min",
        "M3": "3min",
        "M4": "4min",
        "M5": "5min",
        "M6": "6min",
        "M10": "10min",
        "M12": "12min",
        "M15": "15min",
        "M20": "20min",
        "M30": "30min",
        # Pandas >= 2.2/4 rejects legacy uppercase hour alias ("H").
        "H1": "1h",
        "H2": "2h",
        "H3": "3h",
        "H4": "4h",
        "H6": "6h",
        "H8": "8h",
        "H12": "12h",
        "D1": "1D",
        "W1": "1W",
        # "M" was removed; "ME" is the direct replacement for month-end bars.
        "MN1": "1ME",
    }.get(tf)


def _timeframe_to_minutes(tf: str) -> int | None:
    tf = str(tf or "").upper()
    return {
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
    }.get(tf)


def _ordered_timeframes(tfs: list[str]) -> list[str]:
    uniq = [str(tf or "").upper() for tf in tfs if str(tf or "").strip()]
    seen: set[str] = set()
    out: list[str] = []
    for tf in uniq:
        if tf in seen:
            continue
        seen.add(tf)
        out.append(tf)

    def _key(tf: str) -> tuple[int, str]:
        mins = _timeframe_to_minutes(tf)
        if mins is None:
            return (10**9, tf)
        return (int(mins), tf)

    return sorted(out, key=_key)


def _resample_missing_from_available(frames: dict[str, Any], required_tfs: list[str]) -> dict[str, Any]:
    if not frames:
        return frames
    required = _ordered_timeframes(required_tfs)
    if not required:
        return frames

    available = {str(k).upper(): v for k, v in frames.items() if _is_frame_like(v) and not _frame_empty(v)}
    if not available:
        return frames

    # Prefer the finest available frame as source for upsampling to larger bars.
    source_order = _ordered_timeframes(list(available.keys()))
    if not source_order:
        return frames
    source_tf = source_order[0]
    source_df = available.get(source_tf)
    if source_df is None or _frame_empty(source_df):
        return frames

    for tf in required:
        if tf in available:
            continue
        resampled = _resample_ohlcv(source_df, tf)
        if resampled is None or _frame_empty(resampled):
            continue
        try:
            resampled.attrs["source"] = str(source_df.attrs.get("source", "resampled"))
        except Exception:
            pass
        frames[tf] = resampled
        available[tf] = resampled

    return frames


def _normalize_columns(df: Any) -> Any:
    if df is None or df.empty:
        return df
    if _is_rust_frame(df):
        return _to_rust_frame(df) or df
    if not _is_dataframe(df):
        if _is_frame_like(df):
            return _to_rust_frame(df) or df
        return df
    out = df.copy()
    rename = {}
    for col in out.columns:
        low = str(col).lower()
        if low in {"o", "open"}:
            rename[col] = "open"
        elif low in {"h", "high"}:
            rename[col] = "high"
        elif low in {"l", "low"}:
            rename[col] = "low"
        elif low in {"c", "close"}:
            rename[col] = "close"
        elif low in {"v", "vol", "volume"}:
            rename[col] = "volume"
        elif low in {"tick_volume", "real_volume"}:
            rename[col] = "volume"
        elif low in {"timestamp", "time", "datetime", "date"}:
            rename[col] = "timestamp"
    if rename:
        out = out.rename(columns=rename)
    return out


def _ensure_ohlcv(df: Any) -> Any:
    if df is None or df.empty:
        return df
    if not _is_dataframe(df):
        if _is_frame_like(df):
            return _to_rust_frame(df) or df
        return df
    out = df.copy()
    if "close" not in out.columns:
        return out
    for col in ("open", "high", "low"):
        if col not in out.columns:
            out[col] = out["close"]
    if "volume" not in out.columns:
        out["volume"] = 0.0
    return out


def _ensure_datetime_index(df: Any) -> Any:
    if df is None or df.empty:
        return df
    if not _is_dataframe(df):
        if _is_frame_like(df):
            return _to_rust_frame(df) or df
        return df
    out = df.copy()
    if "timestamp" in out.columns:
        try:
            idx = _to_datetime64_ns(out["timestamp"])
            out = out.set_index(idx)
        except Exception:
            pass
    if not _is_datetime_index(out.index):
        try:
            idx = _to_datetime64_ns(out.index)
            out.index = idx
        except Exception:
            # Keep original index as a last resort; caller can still proceed
            # without a strict DatetimeIndex, but we must avoid crashing here.
            return out
    if _is_datetime_index(out.index):
        try:
            if out.index.tz is None:
                out.index = out.index.tz_localize("UTC")
            else:
                out.index = out.index.tz_convert("UTC")
        except Exception:
            pass
    return out


def _resample_ohlcv(df: Any, tf: str) -> Any | None:
    if _is_rust_frame(df):
        return _resample_ohlcv_rust(df, tf)
    if df is None or df.empty:
        return None
    if _strict_tabular_free_enabled():
        rust_src = _to_rust_frame(df)
        return _resample_ohlcv_rust(rust_src, tf) if rust_src is not None else None
    if not _is_dataframe(df):
        rust_src = _to_rust_frame(df)
        return _resample_ohlcv_rust(rust_src, tf) if rust_src is not None else None
    freq = _timeframe_to_freq(tf)
    if freq is None:
        return None
    if not _is_datetime_index(df.index):
        return None
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(freq).agg(agg).dropna()
    return out


def _resample_ohlcv_rust(df: _RustFrame, tf: str) -> _RustFrame | None:
    if _frame_empty(df):
        return None
    tf_min = _timeframe_to_minutes(tf)
    if tf_min is None or tf_min <= 0:
        return None

    idx_ns = _to_datetime64_ns(df.index).astype(np.int64, copy=False)
    if idx_ns.size <= 0:
        return None

    close = np.asarray(df["close"], dtype=np.float64).reshape(-1)
    n = int(close.shape[0])
    if n <= 0:
        return None
    open_ = np.asarray(df["open"], dtype=np.float64).reshape(-1)[:n]
    high = np.asarray(df["high"], dtype=np.float64).reshape(-1)[:n]
    low = np.asarray(df["low"], dtype=np.float64).reshape(-1)[:n]
    close = close[:n]
    volume = np.asarray(df["volume"], dtype=np.float64).reshape(-1)[:n]
    idx_ns = idx_ns[:n]

    bucket_ns = np.int64(int(tf_min) * 60 * 1_000_000_000)
    if bucket_ns <= 0:
        return None
    bucket = idx_ns // bucket_ns
    if bucket.size <= 0:
        return None

    starts = np.flatnonzero(np.r_[True, bucket[1:] != bucket[:-1]])
    ends = np.r_[starts[1:], bucket.size]
    m = int(starts.size)
    if m <= 0:
        return None

    out_open = np.empty(m, dtype=np.float64)
    out_high = np.empty(m, dtype=np.float64)
    out_low = np.empty(m, dtype=np.float64)
    out_close = np.empty(m, dtype=np.float64)
    out_volume = np.empty(m, dtype=np.float64)
    out_idx = np.empty(m, dtype=np.int64)

    for i in range(m):
        s = int(starts[i])
        e = int(ends[i])
        out_open[i] = float(open_[s])
        out_high[i] = float(np.max(high[s:e]))
        out_low[i] = float(np.min(low[s:e]))
        out_close[i] = float(close[e - 1])
        out_volume[i] = float(np.sum(volume[s:e]))
        out_idx[i] = int(idx_ns[s])

    out = _RustFrame(
        {
            "open": out_open,
            "high": out_high,
            "low": out_low,
            "close": out_close,
            "volume": out_volume,
        },
        out_idx.astype("datetime64[ns]"),
    )
    out.attrs = dict(getattr(df, "attrs", {}) or {})
    return out


class _StubMT5Connection:
    async def get_account_information(self) -> dict[str, Any]:
        return {}

    async def positions_get(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return []

    async def get_history_deals(self, _from: int, _to: int) -> list[dict[str, Any]]:
        return []

    async def get_symbol_info(self, _symbol: str) -> dict[str, Any]:
        return {}

    async def get_symbol_price(self, _symbol: str) -> dict[str, Any]:
        return {}

    async def create_market_buy_order(self, *args, **kwargs) -> dict[str, Any]:
        return {"success": False, "reason": "MT5 not connected"}

    async def create_market_sell_order(self, *args, **kwargs) -> dict[str, Any]:
        return {"success": False, "reason": "MT5 not connected"}

    async def close_position_by_ticket(self, *args, **kwargs) -> dict[str, Any]:
        return {"success": False, "reason": "MT5 not connected"}


def _mt5_to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "_asdict"):
        try:
            return obj._asdict()
        except Exception:
            return {}
    try:
        return dict(obj)
    except Exception:
        return {}


def _mt5_list_to_dict(items: Any) -> list[dict[str, Any]]:
    if items is None:
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        out.append(_mt5_to_dict(item))
    return out


class _MT5Connection:
    def __init__(self, mt5_module: Any) -> None:
        self.mt5 = mt5_module

    async def get_account_information(self) -> dict[str, Any]:
        return _mt5_to_dict(self.mt5.account_info())

    async def positions_get(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol:
            return _mt5_list_to_dict(self.mt5.positions_get(symbol=symbol))
        return _mt5_list_to_dict(self.mt5.positions_get())

    async def get_history_deals(self, _from: int, _to: int) -> list[dict[str, Any]]:
        return _mt5_list_to_dict(self.mt5.history_deals_get(_from, _to))

    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        return _mt5_to_dict(self.mt5.symbol_info(symbol))

    async def get_symbol_price(self, symbol: str) -> dict[str, Any]:
        return _mt5_to_dict(self.mt5.symbol_info_tick(symbol))

    async def create_market_buy_order(self, symbol: str, volume: float, sl: float, tp: float, deviation: int = 20, magic: int = 0, comment: str | None = None) -> dict[str, Any]:
        price = self.mt5.symbol_info_tick(symbol).ask
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": self.mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": magic,
            "comment": comment or "forex_bot",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        result = self.mt5.order_send(request)
        return _mt5_to_dict(result)

    async def create_market_sell_order(self, symbol: str, volume: float, sl: float, tp: float, deviation: int = 20, magic: int = 0, comment: str | None = None) -> dict[str, Any]:
        price = self.mt5.symbol_info_tick(symbol).bid
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": self.mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": magic,
            "comment": comment or "forex_bot",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        result = self.mt5.order_send(request)
        return _mt5_to_dict(result)

    async def close_position_by_ticket(self, ticket: int, symbol: str, volume: float | None = None) -> dict[str, Any]:
        positions = self.mt5.positions_get(ticket=ticket)
        if not positions:
            return {"success": False, "reason": "Position not found"}
        pos = positions[0]
        pos_dict = _mt5_to_dict(pos)
        vol = float(volume if volume is not None else pos_dict.get("volume", 0.0))
        order_type = self.mt5.ORDER_TYPE_SELL if pos_dict.get("type", 0) == 0 else self.mt5.ORDER_TYPE_BUY
        price = self.mt5.symbol_info_tick(symbol).bid if order_type == self.mt5.ORDER_TYPE_SELL else self.mt5.symbol_info_tick(symbol).ask
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": vol,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": pos_dict.get("magic", 0),
            "comment": "forex_bot_close",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        result = self.mt5.order_send(request)
        return _mt5_to_dict(result)


class MT5Adapter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.connection = _StubMT5Connection()
        self._connected = False

    async def connect(self) -> bool:
        backend = str(getattr(self.settings.system, "broker_backend", ""))
        if backend != "mt5_local":
            self._connected = False
            return False
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception as exc:
            logger.warning("MetaTrader5 not available: %s", exc)
            self._connected = False
            return False
        try:
            if not mt5.initialize():
                self._connected = False
                return False
            self._connected = True
            self.connection = _MT5Connection(mt5)
        except Exception as exc:
            logger.warning("MT5 initialize failed: %s", exc)
            self._connected = False
            return False
        return True

    async def disconnect(self) -> None:
        try:
            import MetaTrader5 as mt5  # type: ignore

            mt5.shutdown()
        except Exception:
            pass
        self._connected = False
        self.connection = _StubMT5Connection()

    def is_connected(self) -> bool:
        return bool(self._connected)

    async def get_live_frames(self, symbol: str, *, timeframes: list[str], tail: int = 500) -> dict[str, Any]:
        frames: dict[str, Any] = {}
        if not self._connected:
            return frames
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception:
            return frames
        tf_const_name = {
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
        tf_map = {tf: getattr(mt5, const, None) for tf, const in tf_const_name.items()}
        for tf in timeframes:
            mt5_tf = tf_map.get(tf)
            if mt5_tf is None:
                continue
            try:
                rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, tail)
            except Exception:
                rates = None
            if rates is None or len(rates) == 0:
                continue
            try:
                rates_arr = np.asarray(rates)
                data_raw: dict[str, np.ndarray] = {}
                if getattr(rates_arr, "dtype", None) is not None and getattr(rates_arr.dtype, "names", None):
                    for name in rates_arr.dtype.names or ():
                        data_raw[str(name)] = np.asarray(rates_arr[name])
                elif rates_arr.ndim == 2 and rates_arr.shape[1] >= 4:
                    cols = ("open", "high", "low", "close", "tick_volume")
                    for i, name in enumerate(cols):
                        if i >= rates_arr.shape[1]:
                            break
                        data_raw[name] = np.asarray(rates_arr[:, i])
                else:
                    continue
                data = _normalize_rust_payload_columns(data_raw)
                if "close" not in data:
                    continue
                data = _ensure_ohlcv_arrays(
                    {k: v for k, v in data.items() if str(k).lower() not in {"timestamp", "time", "datetime", "date"}}
                )
                n = int(np.asarray(data["close"]).reshape(-1).shape[0])
                if n <= 0:
                    continue
                ts_src = data_raw.get("time")
                idx_np = _to_datetime64_ns(ts_src if ts_src is not None else np.arange(n, dtype=np.int64))
                for key, vals in list(data.items()):
                    data[key] = np.asarray(vals).reshape(-1)[:n]
                df = _build_frame_from_arrays(data, index=idx_np, allow_tabular_module=not _strict_tabular_free_enabled())
            except Exception:
                logger.debug("Skipping MT5 frame build for %s/%s because conversion failed.", symbol, tf)
                continue
            if _is_frame_like(df):
                df = _normalize_columns(df)
                df = _ensure_datetime_index(df)
                df = _ensure_ohlcv(df)
            try:
                df.attrs["source"] = "mt5"
            except Exception:
                pass
            frames[tf] = df
        # Fill unavailable MT5 timeframes via deterministic resampling from available data.
        frames = _resample_missing_from_available(frames, [str(tf).upper() for tf in timeframes])
        return frames


class DataLoader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_dir = Path(getattr(self.settings.system, "data_dir", "data"))
        self.mt5_adapter = MT5Adapter(settings)
        self._connected = False

    async def connect(self) -> bool:
        self._connected = await self.mt5_adapter.connect()
        return self._connected

    async def disconnect(self) -> None:
        await self.mt5_adapter.disconnect()
        self._connected = False

    def is_connected(self) -> bool:
        return bool(self._connected)

    def _timeframes(self) -> list[str]:
        base_tf = str(getattr(self.settings.system, "base_timeframe", "M1") or "M1")
        tfs = [base_tf]
        use_multires = bool(getattr(self.settings.system, "multi_resolution_enabled", True))
        if use_multires:
            for tf in list(getattr(self.settings.system, "multi_resolution_timeframes", []) or []):
                tfu = str(tf or "").upper()
                if tfu and tfu not in tfs:
                    tfs.append(tfu)
        for tf in list(getattr(self.settings.system, "required_timeframes", []) or []):
            tfu = str(tf or "").upper()
            if tfu and tfu not in tfs:
                tfs.append(tfu)
        for tf in list(getattr(self.settings.system, "higher_timeframes", []) or []):
            tfu = str(tf or "").upper()
            if tfu and tfu not in tfs:
                tfs.append(tfu)
        use_all_tfs = str(os.environ.get("FOREX_BOT_USE_ALL_TIMEFRAMES", "0") or "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if use_all_tfs:
            for tf in ALL_TIMEFRAMES:
                tfu = str(tf or "").upper()
                if tfu and tfu not in tfs:
                    tfs.append(tfu)
        return _ordered_timeframes(tfs)

    def _load_frames(self, symbol: str, *, tail: int | None = None, allow_resample: bool = True) -> dict[str, Any]:
        if not _use_rust_data_backend():
            logger.error("Rust data backend unavailable or disabled; skipping local data load for symbol %s.", symbol)
            return {}
        try:
            rust_frames = self._load_frames_rust(symbol, tail=tail, allow_resample=allow_resample)
        except Exception as exc:
            _disable_rust_data_backend()
            logger.error("Rust data loader failed for symbol %s; Python fallback removed: %s", symbol, exc)
            return {}
        if not rust_frames:
            logger.error("Rust data loader returned no frames for symbol %s; Python fallback removed.", symbol)
            return {}
        return rust_frames

    def _load_frames_rust(
        self,
        symbol: str,
        *,
        tail: int | None = None,
        allow_resample: bool = True,
    ) -> dict[str, Any]:
        try:
            import forex_bindings  # type: ignore
        except Exception:
            _disable_rust_data_backend()
            return {}
        if not hasattr(forex_bindings, "load_symbol_frames"):
            _disable_rust_data_backend()
            return {}

        if not self.data_dir.exists():
            return {}

        base_tf = str(getattr(self.settings.system, "base_timeframe", "M1") or "M1")
        tfs = self._timeframes()

        payload = forex_bindings.load_symbol_frames(
            root=str(self.data_dir),
            symbol=symbol,
            timeframes=tfs,
            resample_missing=bool(allow_resample),
            base_tf=base_tf,
        )
        if not isinstance(payload, dict):
            return {}

        frames: dict[str, Any] = {}
        for tf, frame in payload.items():
            if not isinstance(frame, dict):
                continue
            try:
                data_raw = {str(k): np.asarray(v) for k, v in frame.items()}
                data = _normalize_rust_payload_columns(data_raw)
                if "close" not in data:
                    continue
                n = int(np.asarray(data["close"]).reshape(-1).shape[0])
                if n <= 0:
                    continue
                for col in ("open", "high", "low"):
                    if col not in data:
                        data[col] = np.asarray(data["close"])
                if "volume" not in data:
                    data["volume"] = np.zeros(n, dtype=np.float64)
                ts_src = None
                for key in ("timestamp", "time", "datetime", "date"):
                    if key in data:
                        ts_src = data.get(key)
                        break
                idx_np = _to_datetime64_ns(ts_src if ts_src is not None else np.arange(n, dtype=np.int64))
                data = {k: np.asarray(v).reshape(-1)[:n] for k, v in data.items() if str(k).lower() not in {"timestamp", "time", "datetime", "date"}}
                df = _build_frame_from_arrays(
                    data,
                    index=idx_np,
                    allow_tabular_module=not _strict_tabular_free_enabled(),
                )
            except Exception:
                continue
            if _is_frame_like(df):
                df = _normalize_columns(df)
                df = _ensure_datetime_index(df)
                df = _ensure_ohlcv(df)
            try:
                df.attrs["source"] = "disk"
            except Exception:
                pass
            if tail is not None and tail > 0:
                df = df.tail(tail)
            frames[str(tf)] = df
        return frames

    async def ensure_history(self, symbol: str) -> bool:
        frames = self._load_frames(symbol)
        return bool(frames)

    async def ensure_all_history(self, symbols: list[str]) -> bool:
        if not self.data_dir.exists():
            logger.warning(
                "Data directory '%s' not found. Expected layout: "
                "data/symbol=EURUSD/timeframe=M1/data.parquet (or data/EURUSD_M1.csv).",
                self.data_dir,
            )
            return False
        any_found = False
        for sym in symbols:
            try:
                if await self.ensure_history(sym):
                    any_found = True
                else:
                    logger.warning("No local history found for symbol %s.", sym)
            except Exception as exc:
                logger.warning("History check failed for %s: %s", sym, exc)
        return any_found

    async def get_training_data(self, symbol: str) -> dict[str, Any]:
        return self._load_frames(symbol)

    async def get_live_data(self, symbol: str) -> dict[str, Any]:
        tfs = self._timeframes()
        if self.mt5_adapter.is_connected():
            frames = await self.mt5_adapter.get_live_frames(symbol, timeframes=tfs, tail=500)
            if frames:
                frames = _resample_missing_from_available(frames, tfs)
                return frames
        return self._load_frames(symbol, tail=2000)


__all__ = ["DataLoader"]


