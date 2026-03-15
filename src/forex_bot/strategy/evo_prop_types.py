from __future__ import annotations

import contextlib
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from .fast_backtest import infer_pip_metrics

logger = logging.getLogger(__name__)

class NumpyFrame:
    """Minimal frame-like object used for frame-native slicing/alignment in strategy search."""
    def __init__(self, data: dict[str, Any], index: Any, attrs: dict[str, Any] | None = None) -> None:
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.index = np.asarray(index).reshape(-1)
        self.columns = list(self._data.keys())
        self.attrs: dict[str, Any] = dict(attrs or {})

    @property
    def empty(self) -> bool:
        return int(self.index.size) <= 0

    def __len__(self) -> int:
        return int(self.index.size)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[str(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        col = str(key)
        arr = np.asarray(value).reshape(-1)
        n = int(len(self))
        if arr.size != n:
            if arr.size <= 0:
                arr = np.zeros(n, dtype=np.float32)
            elif arr.size > n:
                arr = arr[:n]
            else:
                arr = np.concatenate([arr, np.full(n - arr.size, arr[-1], dtype=arr.dtype)])
        self._data[col] = arr
        if col not in self.columns:
            self.columns.append(col)

    def copy(self, deep: bool = False) -> "NumpyFrame":
        _ = deep
        return NumpyFrame(
            {k: np.asarray(v).copy() for k, v in self._data.items()},
            np.asarray(self.index).copy(),
            attrs=dict(self.attrs),
        )

    def tail(self, n: int) -> "NumpyFrame":
        take = max(0, int(n))
        if take <= 0:
            return NumpyFrame(
                {k: v[:0] for k, v in self._data.items()},
                self.index[:0],
                attrs=dict(self.attrs),
            )
        return NumpyFrame(
            {k: v[-take:] for k, v in self._data.items()},
            self.index[-take:],
            attrs=dict(self.attrs),
        )

def frame_empty(df: Any) -> bool:
    if df is None:
        return True
    with contextlib.suppress(Exception):
        return bool(df.empty)
    with contextlib.suppress(Exception):
        return int(len(df)) <= 0
    return True

def frame_len(df: Any) -> int:
    with contextlib.suppress(Exception):
        return int(len(df))
    return 0

def frame_index(df: Any) -> Any:
    return getattr(df, "index", None)

def frame_columns(df: Any) -> list[str]:
    cols = getattr(df, "columns", None)
    if cols is None:
        return []
    try:
        return [str(c) for c in list(cols)]
    except Exception:
        return []

def frame_attr(df: Any, key: str, default: Any = None) -> Any:
    attrs = getattr(df, "attrs", None)
    if isinstance(attrs, dict):
        return attrs.get(key, default)
    return default

def frame_has_column(df: Any, name: str) -> bool:
    target = str(name).strip().lower()
    for col in frame_columns(df):
        if str(col).strip().lower() == target:
            return True
    return False

def frame_resolve_column(df: Any, name: str) -> str | None:
    target = str(name).strip().lower()
    for col in frame_columns(df):
        if str(col).strip().lower() == target:
            return col
    return None

def to_numpy_1d(values: Any, *, dtype: Any) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        with contextlib.suppress(Exception):
            out = values.to_numpy(dtype=dtype, copy=False)
            return np.asarray(out, dtype=dtype).reshape(-1)
    return np.asarray(values, dtype=dtype).reshape(-1)

def frame_column_numpy(df: Any, name: str, *, dtype: Any = np.float64) -> np.ndarray:
    col = frame_resolve_column(df, name)
    if col is None:
        raise KeyError(name)
    return to_numpy_1d(df[col], dtype=dtype)

def frame_copy(df: Any) -> Any:
    if df is None:
        return None
    with contextlib.suppress(Exception):
        return df.copy(deep=True)
    with contextlib.suppress(Exception):
        return df.copy()
    return df

def copy_attrs(src: Any, dst: Any) -> Any:
    try:
        dst.attrs.update(dict(getattr(src, "attrs", {}) or {}))
    except Exception:
        pass
    return dst

def frame_slice(df: Any, start: int, end: int) -> Any:
    if df is None:
        return None
    s = max(0, int(start))
    e = max(s, int(end))
    if hasattr(df, "take"):
        with contextlib.suppress(Exception):
            idx_take = np.arange(s, e, dtype=np.int64)
            out = df.take(idx_take)
            out = out.copy() if hasattr(out, "copy") else out
            return copy_attrs(df, out)
    if hasattr(df, "loc"):
        with contextlib.suppress(Exception):
            idx_src = frame_index(df)
            if idx_src is not None:
                idx_arr = np.asarray(idx_src).reshape(-1)
                out = df.loc[idx_arr[s:e]]
                out = out.copy() if hasattr(out, "copy") else out
                return copy_attrs(df, out)
    idx = frame_index(df)
    idx_arr = np.asarray(idx).reshape(-1) if idx is not None else np.arange(frame_len(df), dtype=np.int64)
    cols = frame_columns(df)
    data: dict[str, np.ndarray] = {}
    for col in cols:
        with contextlib.suppress(Exception):
            vals = np.asarray(df[col]).reshape(-1)
            data[str(col)] = vals[s:e]
    out = NumpyFrame(data, idx_arr[s:e], attrs=dict(getattr(df, "attrs", {}) or {}))
    return out

def frame_filter_mask(df: Any, mask: Any) -> Any:
    if df is None:
        return None
    m = np.asarray(mask).reshape(-1).astype(bool, copy=False)
    if hasattr(df, "loc"):
        with contextlib.suppress(Exception):
            out = df.loc[m].copy()
            return copy_attrs(df, out)
    idx = frame_index(df)
    idx_arr = np.asarray(idx).reshape(-1) if idx is not None else np.arange(frame_len(df), dtype=np.int64)
    cols = frame_columns(df)
    data: dict[str, np.ndarray] = {}
    for col in cols:
        with contextlib.suppress(Exception):
            vals = np.asarray(df[col]).reshape(-1)
            if vals.size == m.size:
                data[str(col)] = vals[m]
            else:
                k = min(vals.size, m.size)
                data[str(col)] = vals[:k][m[:k]]
    idx_out = idx_arr[m] if idx_arr.size == m.size else idx_arr[: m.size][m[: idx_arr[: m.size].size]]
    out = NumpyFrame(data, idx_out, attrs=dict(getattr(df, "attrs", {}) or {}))
    return out

def is_datetime_index(idx: Any) -> bool:
    if idx is None:
        return False
    if hasattr(idx, "year") and hasattr(idx, "month") and hasattr(idx, "day"):
        return True
    try:
        arr = np.asarray(idx)
        if arr.size <= 0:
            return False
        return np.issubdtype(arr.dtype, np.datetime64)
    except Exception:
        return False

def index_to_ns_int64(idx: Any) -> np.ndarray:
    if idx is None:
        return np.zeros(0, dtype=np.int64)
    try:
        if hasattr(idx, "asi8"):
            vals = np.asarray(idx.asi8, dtype=np.int64).reshape(-1)
            # Normalize to nanoseconds if it looks like microseconds or milliseconds
            if vals.size > 0:
                if vals[-1] < 10**13: # Milliseconds (e.g. 1.7e12)
                    vals = vals * 1_000_000
                elif vals[-1] < 10**16: # Microseconds (e.g. 1.7e15)
                    vals = vals * 1_000
            return vals
    except Exception:
        pass
    with np.errstate(all="ignore"):
        arr = np.asarray(idx).reshape(-1)
    if arr.size <= 0:
        return np.zeros(0, dtype=np.int64)
    try:
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
        if arr.dtype.kind in {"i", "u"}:
            return arr.astype(np.int64, copy=False)
        if arr.dtype.kind == "f":
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
        if hasattr(idx, "view"):
            viewed = idx.view("int64")
            if hasattr(viewed, "to_numpy"):
                return np.asarray(viewed.to_numpy(dtype=np.int64, copy=False), dtype=np.int64).reshape(-1)
            return np.asarray(viewed, dtype=np.int64).reshape(-1)
        if arr.dtype.kind == "O":
            out = np.zeros(arr.size, dtype=np.int64)
            for i, value in enumerate(arr.tolist()):
                try:
                    ns = getattr(value, "value", None)
                    if ns is not None:
                        out[i] = int(ns)
                    else:
                        out[i] = int(np.datetime64(value, "ns").astype(np.int64))
                except Exception:
                    out[i] = 0
            return out
    except Exception:
        pass
    try:
        return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
    except Exception:
        return np.zeros(arr.size, dtype=np.int64)

def env_float(name: str, default: float = 0.0) -> float:
    try:
        val = os.environ.get(name)
        if val is None or str(val).strip() == "":
            return float(default)
        return float(val)
    except Exception:
        return float(default)

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)

def df_reference_prices(df: Any) -> dict[str, float] | None:
    raw = frame_attr(df, "pip_reference_prices")
    if not isinstance(raw, dict):
        return None
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            px = float(value)
        except Exception:
            continue
        if np.isfinite(px) and px > 0.0:
            out[str(key).upper()] = px
    return out or None

def df_pip_metrics(df: Any, close: np.ndarray | None = None) -> tuple[float, float]:
    pip_size = safe_float(frame_attr(df, "pip_size"), 0.0)
    pip_val = safe_float(frame_attr(df, "pip_value_per_lot"), 0.0)
    if pip_size > 0.0 and pip_val > 0.0:
        return float(pip_size), float(pip_val)

    symbol = str(frame_attr(df, "symbol", "") or "")
    last_close: float | None = None
    if close is not None and close.size > 0:
        last_close = safe_float(close[-1], 0.0)
    elif frame_has_column(df, "close") and frame_len(df) > 0:
        with contextlib.suppress(Exception):
            close_arr = frame_column_numpy(df, "close", dtype=np.float64)
            if close_arr.size > 0:
                last_close = safe_float(close_arr[-1], 0.0)
    if last_close is not None and (not np.isfinite(last_close) or last_close <= 0.0):
        last_close = None

    ref_prices = df_reference_prices(df)
    pip_size, pip_val = infer_pip_metrics(
        symbol,
        price=last_close,
        account_currency="USD",
        reference_prices=ref_prices,
    )
    return float(pip_size), float(pip_val)

def history_span_days_months(df: Any) -> tuple[float, float]:
    if frame_empty(df):
        return 0.0, 0.0
    idx = frame_index(df)
    if not is_datetime_index(idx) or len(idx) < 2:
        return 0.0, 0.0
    try:
        if hasattr(idx, "tz_localize"):
            if idx.tz is None:
                i2 = idx.tz_localize("UTC")
            else:
                i2 = idx.tz_convert("UTC")
        else:
            i2 = np.asarray(idx).astype("datetime64[ns]")
    except Exception:
        i2 = idx
    try:
        if hasattr(i2, "max") and hasattr(i2, "min") and not isinstance(i2, np.ndarray):
            span_days = float((i2.max() - i2.min()).total_seconds() / 86400.0)
        else:
            arr = np.asarray(i2).astype("datetime64[ns]")
            if arr.size < 2:
                span_days = 0.0
            else:
                delta_ns = (arr.max() - arr.min()).astype("timedelta64[ns]").astype(np.int64)
                span_days = float(delta_ns) / 86_400_000_000_000.0
    except Exception:
        span_days = 0.0
    span_days = max(0.0, span_days)
    span_months = (span_days / 30.4375) if span_days > 0.0 else 0.0
    return float(span_days), float(span_months)

def holdout_cfg(settings: Any) -> tuple[float, int, float, float, float, int, bool, float, float]:
    def _get(name: str, fallback: Any) -> Any:
        env = os.environ.get(name)
        if env is not None and str(env).strip() != "":
            return env
        return fallback

    frac = float(_get("FOREX_BOT_PROP_HOLDOUT_FRACTION", getattr(settings.models, "prop_search_holdout_fraction", 0.0)) or 0.0)
    min_rows = int(_get("FOREX_BOT_PROP_HOLDOUT_MIN_ROWS", getattr(settings.models, "prop_search_holdout_min_rows", 8000)) or 8000)
    min_sharpe = float(_get("FOREX_BOT_PROP_HOLDOUT_MIN_SHARPE", getattr(settings.models, "prop_search_holdout_min_sharpe", 1.0)) or 1.0)
    min_win = float(_get("FOREX_BOT_PROP_HOLDOUT_MIN_WIN_RATE", getattr(settings.models, "prop_search_holdout_min_win_rate", 0.50)) or 0.50)
    min_pf = float(_get("FOREX_BOT_PROP_HOLDOUT_MIN_PROFIT_FACTOR", getattr(settings.models, "prop_search_holdout_min_profit_factor", 1.20)) or 1.20)
    tr_fallback = getattr(settings.models, "prop_search_holdout_min_trades", 15)
    min_tr = int(_get("FOREX_BOT_PROP_HOLDOUT_MIN_TRADES", tr_fallback) or 15)
    
    # Default to 2.0 years for forward testing if not provided
    years_fallback = getattr(settings.models, "prop_search_holdout_years", 2.0)
    years = float(_get("FOREX_BOT_PROP_HOLDOUT_YEARS", years_fallback) or 2.0)
    min_truth = float(
        _get(
            "FOREX_BOT_MIN_TRUTH_PROBABILITY",
            _get(
                "FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY",
                getattr(settings.models, "prop_search_holdout_min_truth_probability", 0.0),
            ),
        )
        or 0.0
    )
    if min_truth > 1.0:
        min_truth *= 0.01
    min_truth = float(min(1.0, max(0.0, min_truth)))
    required = str(
        _get("FOREX_BOT_PROP_HOLDOUT_REQUIRED", getattr(settings.models, "prop_search_holdout_required", False))
    ).strip().lower() in {"1", "true", "yes", "on"}
    return frac, max(0, min_rows), min_sharpe, min_win, min_pf, max(0, min_tr), required, max(0.0, years), min_truth

def train_years_cfg(settings: Any) -> float:
    raw = os.environ.get("FOREX_BOT_PROP_SEARCH_TRAIN_YEARS")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(0.0, float(raw))
        except Exception:
            return 0.0
    try:
        configured = max(0.0, float(getattr(settings.models, "prop_search_train_years", 0.0) or 0.0))
    except Exception:
        configured = 0.0
    if configured > 0.0:
        return configured
    with contextlib.suppress(Exception):
        system_years = float(getattr(getattr(settings, "system", None), "history_years", 0.0) or 0.0)
        if system_years > 0.0:
            return max(0.0, system_years)
    return 0.0

def trim_to_recent_years(df: Any, years: float) -> Any:
    if frame_empty(df) or years <= 0.0:
        return df
    idx = frame_index(df)
    if not is_datetime_index(idx) or len(idx) < 2:
        return df
    try:
        if hasattr(idx, "tz_localize"):
            if idx.tz is None:
                idx2 = idx.tz_localize("UTC")
            else:
                idx2 = idx.tz_convert("UTC")
        else:
            idx2 = np.asarray(idx).astype("datetime64[ns]")
    except Exception:
        idx2 = idx
    try:
        if hasattr(idx2, "max") and hasattr(idx2, "min") and not isinstance(idx2, np.ndarray):
            cutoff = idx2.max() - timedelta(days=float(years) * 365.2425)
            mask = idx2 >= cutoff
        else:
            arr = np.asarray(idx2).astype("datetime64[ns]")
            if arr.size <= 0:
                return df
            cutoff = arr.max() - np.timedelta64(int(float(years) * 365.2425 * 24 * 3600 * 1_000_000_000), "ns")
            mask = arr >= cutoff
    except Exception:
        return df
    keep = int(np.count_nonzero(np.asarray(mask, dtype=bool)))
    if keep <= 0 or keep >= len(df):
        return df
    trimmed = frame_filter_mask(df, mask)
    trimmed = copy_attrs(df, trimmed)
    return trimmed
