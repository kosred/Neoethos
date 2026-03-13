from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..features.talib_mixer import (
    ALL_INDICATORS,
    TALibStrategyGene,
    TALibStrategyMixer,
    signal_to_numpy,
)
from .fast_backtest import (
    infer_pip_metrics,
    infer_sl_tp_pips_auto,
)

logger = logging.getLogger(__name__)

try:
    import forex_bindings as _fb  # type: ignore

    _RUST_SEARCH = hasattr(_fb, "search_evolve_ohlcv")
    _RUST_GPU_SEARCH = hasattr(_fb, "search_evolve_gpu_ohlcv")
    _RUST_TRADE_JOURNAL = hasattr(_fb, "trade_journal_metrics")
    _RUST_TALIB_POP = hasattr(_fb, "evaluate_population_talib_ohlcv")
    _RUST_POP_EVAL = _RUST_TALIB_POP or hasattr(_fb, "batch_evaluate_strategies")
except Exception:
    _fb = None  # type: ignore
    _RUST_SEARCH = False
    _RUST_GPU_SEARCH = False
    _RUST_TRADE_JOURNAL = False
    _RUST_POP_EVAL = False
    _RUST_TALIB_POP = False


class _NumpyFrame:
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

    def copy(self, deep: bool = False) -> "_NumpyFrame":
        _ = deep
        return _NumpyFrame(
            {k: np.asarray(v).copy() for k, v in self._data.items()},
            np.asarray(self.index).copy(),
            attrs=dict(self.attrs),
        )

    def tail(self, n: int) -> "_NumpyFrame":
        take = max(0, int(n))
        if take <= 0:
            return _NumpyFrame(
                {k: v[:0] for k, v in self._data.items()},
                self.index[:0],
                attrs=dict(self.attrs),
            )
        return _NumpyFrame(
            {k: v[-take:] for k, v in self._data.items()},
            self.index[-take:],
            attrs=dict(self.attrs),
        )


def _frame_empty(df: Any) -> bool:
    if df is None:
        return True
    with contextlib.suppress(Exception):
        return bool(df.empty)
    with contextlib.suppress(Exception):
        return int(len(df)) <= 0
    return True


def _frame_len(df: Any) -> int:
    with contextlib.suppress(Exception):
        return int(len(df))
    return 0


def _frame_index(df: Any) -> Any:
    return getattr(df, "index", None)


def _frame_columns(df: Any) -> list[str]:
    cols = getattr(df, "columns", None)
    if cols is None:
        return []
    try:
        return [str(c) for c in list(cols)]
    except Exception:
        return []


def _frame_attr(df: Any, key: str, default: Any = None) -> Any:
    attrs = getattr(df, "attrs", None)
    if isinstance(attrs, dict):
        return attrs.get(key, default)
    return default


def _frame_has_column(df: Any, name: str) -> bool:
    target = str(name).strip().lower()
    for col in _frame_columns(df):
        if str(col).strip().lower() == target:
            return True
    return False


def _frame_resolve_column(df: Any, name: str) -> str | None:
    target = str(name).strip().lower()
    for col in _frame_columns(df):
        if str(col).strip().lower() == target:
            return col
    return None


def _to_numpy_1d(values: Any, *, dtype: Any) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        with contextlib.suppress(Exception):
            out = values.to_numpy(dtype=dtype, copy=False)  # type: ignore[call-arg]
            return np.asarray(out, dtype=dtype).reshape(-1)
    return np.asarray(values, dtype=dtype).reshape(-1)


def _frame_column_numpy(df: Any, name: str, *, dtype: Any = np.float64) -> np.ndarray:
    col = _frame_resolve_column(df, name)
    if col is None:
        raise KeyError(name)
    return _to_numpy_1d(df[col], dtype=dtype)  # type: ignore[index]


def _frame_copy(df: Any) -> Any:
    if df is None:
        return None
    with contextlib.suppress(Exception):
        return df.copy(deep=True)
    with contextlib.suppress(Exception):
        return df.copy()
    return df


def _frame_slice(df: Any, start: int, end: int) -> Any:
    if df is None:
        return None
    s = max(0, int(start))
    e = max(s, int(end))
    if hasattr(df, "take"):
        with contextlib.suppress(Exception):
            idx_take = np.arange(s, e, dtype=np.int64)
            out = df.take(idx_take)
            out = out.copy() if hasattr(out, "copy") else out
            return _copy_attrs(df, out)
    if hasattr(df, "loc"):
        with contextlib.suppress(Exception):
            idx_src = _frame_index(df)
            if idx_src is not None:
                idx_arr = np.asarray(idx_src).reshape(-1)
                out = df.loc[idx_arr[s:e]]
                out = out.copy() if hasattr(out, "copy") else out
                return _copy_attrs(df, out)
    idx = _frame_index(df)
    idx_arr = np.asarray(idx).reshape(-1) if idx is not None else np.arange(_frame_len(df), dtype=np.int64)
    cols = _frame_columns(df)
    data: dict[str, np.ndarray] = {}
    for col in cols:
        with contextlib.suppress(Exception):
            vals = np.asarray(df[col]).reshape(-1)  # type: ignore[index]
            data[str(col)] = vals[s:e]
    out = _NumpyFrame(data, idx_arr[s:e], attrs=dict(getattr(df, "attrs", {}) or {}))
    return out


def _frame_filter_mask(df: Any, mask: Any) -> Any:
    if df is None:
        return None
    m = np.asarray(mask).reshape(-1).astype(bool, copy=False)
    if hasattr(df, "loc"):
        with contextlib.suppress(Exception):
            out = df.loc[m].copy()
            return _copy_attrs(df, out)
    idx = _frame_index(df)
    idx_arr = np.asarray(idx).reshape(-1) if idx is not None else np.arange(_frame_len(df), dtype=np.int64)
    cols = _frame_columns(df)
    data: dict[str, np.ndarray] = {}
    for col in cols:
        with contextlib.suppress(Exception):
            vals = np.asarray(df[col]).reshape(-1)  # type: ignore[index]
            if vals.size == m.size:
                data[str(col)] = vals[m]
            else:
                k = min(vals.size, m.size)
                data[str(col)] = vals[:k][m[:k]]
    idx_out = idx_arr[m] if idx_arr.size == m.size else idx_arr[: m.size][m[: idx_arr[: m.size].size]]
    out = _NumpyFrame(data, idx_out, attrs=dict(getattr(df, "attrs", {}) or {}))
    return out


def _is_datetime_index(idx: Any) -> bool:
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


def _index_to_ns_int64(idx: Any) -> np.ndarray:
    if idx is None:
        return np.zeros(0, dtype=np.int64)
    try:
        if hasattr(idx, "asi8"):
            return np.asarray(idx.asi8, dtype=np.int64).reshape(-1)
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


def _rust_time_index_arrays(idx: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if _fb is None or not hasattr(_fb, "derive_time_index_arrays"):
        return None
    ns = _index_to_ns_int64(idx)
    if ns.size <= 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    try:
        unix_ms, month_idx, day_idx = _fb.derive_time_index_arrays(np.asarray(ns, dtype=np.int64))
    except Exception:
        return None
    return (
        np.asarray(unix_ms, dtype=np.int64).reshape(-1),
        np.asarray(month_idx, dtype=np.int64).reshape(-1),
        np.asarray(day_idx, dtype=np.int64).reshape(-1),
    )


def _rust_rank_scores_desc(scores: Any, *, absolute: bool = False) -> np.ndarray | None:
    if _fb is None or not hasattr(_fb, "rank_scores_desc"):
        return None
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    try:
        out = _fb.rank_scores_desc(arr, bool(absolute))
    except Exception:
        return None
    order = np.asarray(out, dtype=np.int64).reshape(-1)
    if order.size != arr.size:
        return None
    return order


def _safe_indices(idx: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
    rust = _rust_time_index_arrays(idx)
    if rust is not None:
        _unix_ms, month_idx, day_idx = rust
        return month_idx[:n], day_idx[:n]
    if _is_datetime_index(idx):
        try:
            year = idx.year.astype(np.int64)
            month = idx.month.astype(np.int64)
            day = idx.day.astype(np.int64)
        except Exception:
            arr = np.asarray(idx)
            # numpy datetime64 path
            months_since_epoch = arr.astype("datetime64[M]").astype(np.int64)
            years = months_since_epoch // 12 + 1970
            months = months_since_epoch % 12 + 1
            days_since_epoch = arr.astype("datetime64[D]").astype(np.int64)
            # days to yyyymmdd
            # derive y,m,d by converting back to datetime64 to vectorized datetime
            dt = days_since_epoch.astype("datetime64[D]")
            years = dt.astype("datetime64[Y]").astype(int) + 1970
            start_year = years.astype("datetime64[Y]")
            months = ((dt - start_year).astype("timedelta64[M]").astype(int)) + 1
            start_month = start_year + (months - 1).astype("timedelta64[M]")
            days = ((dt - start_month).astype("timedelta64[D]").astype(int)) + 1
            year = years
            month = months
            day = days
        month_idx = (year * 12 + month).astype(np.int64, copy=False)
        day_idx = (year * 10000 + month * 100 + day).astype(np.int64, copy=False)
        return month_idx[:n], day_idx[:n]
    ns = _index_to_ns_int64(idx)
    if ns.size > 0:
        vmax = int(np.max(np.abs(ns))) if ns.size > 0 else 0
        if vmax > 10**14:
            dt = np.asarray(ns, dtype=np.int64).astype("datetime64[ns]")
            month_idx = dt.astype("datetime64[M]").astype(np.int64)
            day_idx = dt.astype("datetime64[D]").astype(np.int64)
            return month_idx[:n], day_idx[:n]
    seq = np.arange(n, dtype=np.int64)
    return seq, seq


def _datetime_index_to_unix_ms(idx: Any) -> np.ndarray:
    rust = _rust_time_index_arrays(idx)
    if rust is not None:
        unix_ms, _month_idx, _day_idx = rust
        return unix_ms
    ns = _index_to_ns_int64(idx)
    if ns.size <= 0:
        return np.zeros(0, dtype=np.int64)
    return (np.asarray(ns, dtype=np.int64) // 1_000_000).astype(np.int64, copy=False)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _df_reference_prices(df: Any) -> dict[str, float] | None:
    raw = _frame_attr(df, "pip_reference_prices")
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


def _df_pip_metrics(df: Any, close: np.ndarray | None = None) -> tuple[float, float]:
    pip_size = _safe_float(_frame_attr(df, "pip_size"), 0.0)
    pip_val = _safe_float(_frame_attr(df, "pip_value_per_lot"), 0.0)
    if pip_size > 0.0 and pip_val > 0.0:
        return float(pip_size), float(pip_val)

    symbol = str(_frame_attr(df, "symbol", "") or "")
    last_close: float | None = None
    if close is not None and close.size > 0:
        last_close = _safe_float(close[-1], 0.0)
    elif _frame_has_column(df, "close") and _frame_len(df) > 0:
        with contextlib.suppress(Exception):
            close_arr = _frame_column_numpy(df, "close", dtype=np.float64)
            if close_arr.size > 0:
                last_close = _safe_float(close_arr[-1], 0.0)
    if last_close is not None and (not np.isfinite(last_close) or last_close <= 0.0):
        last_close = None

    ref_prices = _df_reference_prices(df)
    pip_size, pip_val = infer_pip_metrics(
        symbol,
        price=last_close,
        account_currency="USD",
        reference_prices=ref_prices,
    )
    return float(pip_size), float(pip_val)


def _history_span_days_months(df: Any) -> tuple[float, float]:
    if _frame_empty(df):
        return 0.0, 0.0
    idx = _frame_index(df)
    if not _is_datetime_index(idx) or len(idx) < 2:
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


def _copy_attrs(src: Any, dst: Any) -> Any:
    try:
        dst.attrs.update(dict(getattr(src, "attrs", {}) or {}))
    except Exception:
        pass
    return dst


def _holdout_cfg(settings: Any) -> tuple[float, int, float, float, float, int, bool, float, float]:
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
    min_tr = int(_get("FOREX_BOT_PROP_HOLDOUT_MIN_TRADES", getattr(settings.models, "prop_search_holdout_min_trades", 15)) or 15)
    years = float(_get("FOREX_BOT_PROP_HOLDOUT_YEARS", getattr(settings.models, "prop_search_holdout_years", 0.0)) or 0.0)
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


def _train_years_cfg(settings: Any) -> float:
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
    # Fallback to system history window when explicit prop-search years are not configured.
    with contextlib.suppress(Exception):
        system_years = float(getattr(getattr(settings, "system", None), "history_years", 0.0) or 0.0)
        if system_years > 0.0:
            return max(0.0, system_years)
    return 0.0


def _trim_to_recent_years(df: Any, years: float) -> Any:
    if _frame_empty(df) or years <= 0.0:
        return df
    idx = _frame_index(df)
    if not _is_datetime_index(idx) or len(idx) < 2:
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
    trimmed = _frame_filter_mask(df, mask)
    trimmed = _copy_attrs(df, trimmed)
    return trimmed


def _holdout_from_cutoff_utc() -> Any | None:
    raw = str(os.environ.get("FOREX_BOT_PROP_HOLDOUT_FROM", "") or "").strip()
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    try:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
    except Exception:
        return None
    return ts


def _split_discovery_holdout(df: Any, settings: Any) -> tuple[Any, Any | None]:
    frac, min_rows, *_base, holdout_years, _min_truth = _holdout_cfg(settings)
    if _frame_empty(df):
        return df, None
    n = _frame_len(df)
    if n < max(1000, min_rows):
        return df, None

    # Optional strict forward cutoff date (e.g. 2025-08-01) for anti-lookahead validation.
    holdout_from = _holdout_from_cutoff_utc()
    idx_base = _frame_index(df)
    if holdout_from is not None and _is_datetime_index(idx_base):
        try:
            idx = idx_base
            if hasattr(idx, "tz_localize"):
                if idx.tz is None:
                    idx2 = idx.tz_localize("UTC")
                else:
                    idx2 = idx.tz_convert("UTC")
                hold_mask = idx2 >= holdout_from
            else:
                idx2 = np.asarray(idx).astype("datetime64[ns]")
                hold_from_ns = np.datetime64(holdout_from.replace(tzinfo=timezone.utc).isoformat())
                hold_mask = idx2 >= hold_from_ns
            hold_n = int(np.count_nonzero(hold_mask))
            split = int(n - hold_n)
            if split >= 500 and hold_n >= max(500, min_rows):
                search_df = _copy_attrs(df, _frame_slice(df, 0, split))
                holdout_df = _copy_attrs(df, _frame_slice(df, split, n))
                return search_df, holdout_df
        except Exception:
            pass

    # Preferred mode: strict calendar holdout (e.g., last 3 years as forward test).
    if holdout_years > 0.0 and _is_datetime_index(idx_base):
        try:
            idx = idx_base
            if hasattr(idx, "tz_localize"):
                if idx.tz is None:
                    idx2 = idx.tz_localize("UTC")
                else:
                    idx2 = idx.tz_convert("UTC")
            else:
                idx2 = np.asarray(idx).astype("datetime64[ns]")
        except Exception:
            idx2 = idx_base
        try:
            if hasattr(idx2, "max") and not isinstance(idx2, np.ndarray):
                split_ts = idx2.max() - timedelta(days=float(holdout_years) * 365.2425)
            else:
                arr_idx = np.asarray(idx2).astype("datetime64[ns]")
                split_ts = arr_idx.max() - np.timedelta64(
                    int(float(holdout_years) * 365.2425 * 24 * 3600 * 1_000_000_000), "ns"
                )
            hold_mask = idx2 >= split_ts
            hold_n = int(np.count_nonzero(hold_mask))
            split = int(n - hold_n)
            if split >= 500 and hold_n >= max(500, min_rows):
                search_df = _copy_attrs(df, _frame_slice(df, 0, split))
                holdout_df = _copy_attrs(df, _frame_slice(df, split, n))
                return search_df, holdout_df
        except Exception:
            pass

    if frac <= 0.0:
        return df, None

    split = int(round(n * (1.0 - min(0.8, max(0.05, frac)))))
    split = max(500, min(n - 500, split))
    if split <= 0 or split >= n:
        return df, None
    search_df = _copy_attrs(df, _frame_slice(df, 0, split))
    holdout_df = _copy_attrs(df, _frame_slice(df, split, n))
    return search_df, holdout_df


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


def _ratio01(num: float, den: float) -> float:
    if den <= 0.0:
        return 1.0 if num > 0.0 else 0.0
    return _clamp01(num / den)


def _truth_probability(
    *,
    in_sample_net: float,
    in_sample_sharpe: float,
    in_sample_win: float,
    in_sample_pf: float,
    holdout_net: float,
    holdout_sharpe: float,
    holdout_win: float,
    holdout_pf: float,
    holdout_trades: float,
    holdout_monthly_profit_pct: float,
    min_sharpe: float,
    min_win: float,
    min_pf: float,
    min_trades: float,
) -> float:
    quality = (
        0.35 * _ratio01(max(0.0, holdout_sharpe), max(1e-9, min_sharpe))
        + 0.25 * _ratio01(max(0.0, holdout_win), max(1e-9, min_win))
        + 0.25 * _ratio01(max(0.0, holdout_pf), max(1e-9, min_pf))
        + 0.15 * _ratio01(max(0.0, holdout_trades), max(1e-9, float(min_trades)))
    )

    stability = (
        0.40 * _ratio01(max(0.0, holdout_net), max(1e-9, max(0.0, in_sample_net) * 0.35))
        + 0.20 * _ratio01(max(0.0, holdout_sharpe), max(1e-9, max(0.0, in_sample_sharpe) * 0.60))
        + 0.20 * _ratio01(max(0.0, holdout_pf), max(1e-9, max(0.0, in_sample_pf) * 0.60))
        + 0.20 * _clamp01(1.0 - (abs(holdout_win - in_sample_win) / 0.25))
    )

    monthly_target = max(_env_float("FOREX_BOT_PROP_KEEP_MIN_MONTHLY_PROFIT_PCT", 0.0), 0.005)
    monthly = _ratio01(max(0.0, holdout_monthly_profit_pct), monthly_target)

    score = 0.50 * quality + 0.35 * stability + 0.15 * monthly
    if holdout_net <= 0.0:
        score *= 0.30
    if holdout_trades < float(min_trades):
        score *= 0.60
    return _clamp01(score)


def _apply_holdout_validation(
    *,
    selected: list[TALibStrategyGene],
    holdout_df: Any | None,
    settings: Any,
    max_dd: float,
    min_profit: float,
    min_trades: float,
    initial_balance: float,
    search_history_months: float | None = None,
) -> list[TALibStrategyGene]:
    if not selected:
        return selected

    frac, _min_rows, min_sharpe, min_win, min_pf, min_tr_holdout, required, holdout_years, min_truth = _holdout_cfg(settings)
    if frac <= 0.0 and holdout_years <= 0.0:
        return selected
    if _frame_empty(holdout_df):
        if required:
            logger.warning("Holdout validation required but holdout split is unavailable; dropping selected strategies.")
            return []
        return selected

    try:
        mixer = TALibStrategyMixer()
        if not mixer.available_indicators:
            logger.warning("Holdout validation skipped: TA-Lib indicators unavailable.")
            return [] if required else selected

        cache = mixer.bulk_calculate_indicators(holdout_df, selected)
        _days, holdout_months = _history_span_days_months(holdout_df)
        passed: list[TALibStrategyGene] = []
        search_months = float(search_history_months or 0.0)
        init_bal = max(1e-9, float(initial_balance))
        for gene in selected:
            in_sample_net = float(getattr(gene, "net_profit", 0.0) or 0.0)
            in_sample_sharpe = float(getattr(gene, "sharpe_ratio", 0.0) or 0.0)
            in_sample_win = float(getattr(gene, "win_rate", 0.0) or 0.0)
            in_sample_pf = float(getattr(gene, "profit_factor", 0.0) or 0.0)
            in_sample_trades = float(getattr(gene, "trades", 0.0) or 0.0)
            in_sample_dd = float(getattr(gene, "max_dd_pct", 0.0) or 0.0)

            g_eval = replace(gene)
            g_eval.in_sample_net_profit = in_sample_net
            g_eval.in_sample_sharpe_ratio = in_sample_sharpe
            g_eval.in_sample_win_rate = in_sample_win
            g_eval.in_sample_profit_factor = in_sample_pf
            g_eval.in_sample_trades = in_sample_trades
            g_eval.in_sample_max_dd_pct = in_sample_dd
            g_eval.in_sample_months = max(0.0, search_months)
            _evaluate_gene(holdout_df, g_eval, mixer, cache, settings)

            holdout_net = float(getattr(g_eval, "net_profit", 0.0) or 0.0)
            holdout_sharpe = float(getattr(g_eval, "sharpe_ratio", 0.0) or 0.0)
            holdout_win = float(getattr(g_eval, "win_rate", 0.0) or 0.0)
            holdout_pf = float(getattr(g_eval, "profit_factor", 0.0) or 0.0)
            holdout_trades = float(getattr(g_eval, "trades", 0.0) or 0.0)
            holdout_dd = float(getattr(g_eval, "max_dd_pct", 0.0) or 0.0)
            holdout_tpm = (holdout_trades / holdout_months) if holdout_months > 0.0 else 0.0
            holdout_monthly_pct = (holdout_net / (init_bal * holdout_months)) if holdout_months > 0.0 else 0.0

            g_eval.holdout_net_profit = holdout_net
            g_eval.holdout_sharpe_ratio = holdout_sharpe
            g_eval.holdout_win_rate = holdout_win
            g_eval.holdout_profit_factor = holdout_pf
            g_eval.holdout_trades = holdout_trades
            g_eval.holdout_max_dd_pct = holdout_dd
            g_eval.holdout_months = holdout_months
            g_eval.holdout_trades_per_month = holdout_tpm
            g_eval.holdout_monthly_profit_pct = holdout_monthly_pct
            g_eval.truth_probability = _truth_probability(
                in_sample_net=in_sample_net,
                in_sample_sharpe=in_sample_sharpe,
                in_sample_win=in_sample_win,
                in_sample_pf=in_sample_pf,
                holdout_net=holdout_net,
                holdout_sharpe=holdout_sharpe,
                holdout_win=holdout_win,
                holdout_pf=holdout_pf,
                holdout_trades=holdout_trades,
                holdout_monthly_profit_pct=holdout_monthly_pct,
                min_sharpe=min_sharpe,
                min_win=min_win,
                min_pf=min_pf,
                min_trades=max(min_trades, float(min_tr_holdout)),
            )

            passed_filters = _strategy_passes_filter(
                g_eval,
                max_dd=max_dd,
                min_profit=min_profit,
                min_trades=max(min_trades, float(min_tr_holdout)),
                history_months=holdout_months,
                initial_balance=initial_balance,
            )
            if float(getattr(g_eval, "sharpe_ratio", 0.0) or 0.0) < float(min_sharpe):
                passed_filters = False
            if float(getattr(g_eval, "win_rate", 0.0) or 0.0) < float(min_win):
                passed_filters = False
            if float(getattr(g_eval, "profit_factor", 0.0) or 0.0) < float(min_pf):
                passed_filters = False
            if float(getattr(g_eval, "truth_probability", 0.0) or 0.0) < float(min_truth):
                passed_filters = False

            g_eval.forward_test_passed = bool(passed_filters)
            if g_eval.forward_test_passed:
                passed.append(g_eval)

        if not passed:
            logger.warning(
                "Holdout validation kept 0/%s strategies (required=%s, min_sharpe=%.2f, min_win=%.2f, min_pf=%.2f, min_truth=%.2f).",
                len(selected),
                required,
                min_sharpe,
                min_win,
                min_pf,
                min_truth,
            )
            return [] if required else selected

        passed = _dedupe_ranked(passed)
        logger.info(
            "Holdout validation kept %s/%s strategies (min_sharpe=%.2f, min_win=%.2f, min_pf=%.2f, min_truth=%.2f).",
            len(passed),
            len(selected),
            min_sharpe,
            min_win,
            min_pf,
            min_truth,
        )
        return passed
    except Exception as exc:
        logger.warning("Holdout validation failed: %s", exc)
        return selected


def _gene_to_dict(gene: TALibStrategyGene) -> dict[str, Any]:
    in_trades = float(getattr(gene, "in_sample_trades", 0.0) or 0.0)
    in_months = float(getattr(gene, "in_sample_months", 0.0) or 0.0)
    hold_trades = float(getattr(gene, "holdout_trades", 0.0) or 0.0)
    hold_months = float(getattr(gene, "holdout_months", 0.0) or 0.0)
    in_net = float(getattr(gene, "in_sample_net_profit", 0.0) or 0.0)
    hold_net = float(getattr(gene, "holdout_net_profit", 0.0) or 0.0)
    bal = max(1e-9, _safe_float(os.environ.get("FOREX_BOT_PROP_INITIAL_BALANCE", 100000.0), 100000.0))

    in_profit_per_trade = (in_net / in_trades) if in_trades > 0.0 else 0.0
    hold_profit_per_trade = (hold_net / hold_trades) if hold_trades > 0.0 else 0.0
    in_tpm = (in_trades / in_months) if in_months > 0.0 else 0.0
    hold_tpm = float(getattr(gene, "holdout_trades_per_month", 0.0) or 0.0)
    if hold_tpm <= 0.0 and hold_months > 0.0:
        hold_tpm = hold_trades / hold_months
    in_monthly_profit_pct = (in_net / (bal * in_months)) if in_months > 0.0 else 0.0
    hold_monthly_profit_pct = float(getattr(gene, "holdout_monthly_profit_pct", 0.0) or 0.0)
    if hold_monthly_profit_pct <= 0.0 and hold_months > 0.0:
        hold_monthly_profit_pct = hold_net / (bal * hold_months)
    in_journal = dict(getattr(gene, "in_sample_journal", {}) or {})
    hold_journal = dict(getattr(gene, "holdout_journal", {}) or {})

    return {
        "indicators": list(gene.indicators),
        "params": gene.params,
        "combination_method": gene.combination_method,
        "long_threshold": float(gene.long_threshold),
        "short_threshold": float(gene.short_threshold),
        "weights": gene.weights,
        "preferred_regime": gene.preferred_regime,
        "strategy_id": gene.strategy_id,
        "fitness": float(getattr(gene, "fitness", 0.0)),
        "sharpe_ratio": float(getattr(gene, "sharpe_ratio", 0.0)),
        "win_rate": float(getattr(gene, "win_rate", 0.0)),
        "net_profit": float(getattr(gene, "net_profit", 0.0)),
        "profit_factor": float(getattr(gene, "profit_factor", 0.0)),
        "expectancy": float(getattr(gene, "expectancy", 0.0)),
        "max_dd_pct": float(getattr(gene, "max_dd_pct", 0.0)),
        "max_drawdown": float(getattr(gene, "max_dd_pct", 0.0)),
        "trades": float(getattr(gene, "trades", 0.0)),
        "trades_count": float(getattr(gene, "trades", 0.0)),
        "use_ob": bool(getattr(gene, "use_ob", False)),
        "use_fvg": bool(getattr(gene, "use_fvg", False)),
        "use_liq_sweep": bool(getattr(gene, "use_liq_sweep", False)),
        "mtf_confirmation": bool(getattr(gene, "mtf_confirmation", False)),
        "use_premium_discount": bool(getattr(gene, "use_premium_discount", False)),
        "use_inducement": bool(getattr(gene, "use_inducement", False)),
        "use_bos": bool(getattr(gene, "use_bos", False)),
        "use_choch": bool(getattr(gene, "use_choch", False)),
        "use_eqh": bool(getattr(gene, "use_eqh", False)),
        "use_eql": bool(getattr(gene, "use_eql", False)),
        "use_displacement": bool(getattr(gene, "use_displacement", False)),
        "tp_pips": float(getattr(gene, "tp_pips", 40.0)),
        "sl_pips": float(getattr(gene, "sl_pips", 20.0)),
        "in_sample_net_profit": in_net,
        "in_sample_sharpe_ratio": float(getattr(gene, "in_sample_sharpe_ratio", 0.0) or 0.0),
        "in_sample_win_rate": float(getattr(gene, "in_sample_win_rate", 0.0) or 0.0),
        "in_sample_profit_factor": float(getattr(gene, "in_sample_profit_factor", 0.0) or 0.0),
        "in_sample_trades": in_trades,
        "in_sample_max_dd_pct": float(getattr(gene, "in_sample_max_dd_pct", 0.0) or 0.0),
        "in_sample_months": in_months,
        "in_sample_trades_per_month": in_tpm,
        "in_sample_profit_per_trade": in_profit_per_trade,
        "in_sample_monthly_profit_pct": in_monthly_profit_pct,
        "holdout_net_profit": hold_net,
        "holdout_sharpe_ratio": float(getattr(gene, "holdout_sharpe_ratio", 0.0) or 0.0),
        "holdout_win_rate": float(getattr(gene, "holdout_win_rate", 0.0) or 0.0),
        "holdout_profit_factor": float(getattr(gene, "holdout_profit_factor", 0.0) or 0.0),
        "holdout_trades": hold_trades,
        "holdout_max_dd_pct": float(getattr(gene, "holdout_max_dd_pct", 0.0) or 0.0),
        "holdout_months": hold_months,
        "holdout_trades_per_month": hold_tpm,
        "holdout_profit_per_trade": hold_profit_per_trade,
        "holdout_monthly_profit_pct": hold_monthly_profit_pct,
        "truth_probability": float(getattr(gene, "truth_probability", 0.0) or 0.0),
        "forward_test_passed": bool(getattr(gene, "forward_test_passed", False)),
        "in_sample_avg_holding_hours": float(in_journal.get("avg_holding_hours", 0.0) or 0.0),
        "holdout_avg_holding_hours": float(hold_journal.get("avg_holding_hours", 0.0) or 0.0),
        "in_sample_trades_per_day": float(in_journal.get("avg_trades_per_day", 0.0) or 0.0),
        "holdout_trades_per_day": float(hold_journal.get("avg_trades_per_day", 0.0) or 0.0),
        "in_sample_wins": float(in_journal.get("wins", 0.0) or 0.0),
        "in_sample_losses": float(in_journal.get("losses", 0.0) or 0.0),
        "holdout_wins": float(hold_journal.get("wins", 0.0) or 0.0),
        "holdout_losses": float(hold_journal.get("losses", 0.0) or 0.0),
        "in_sample_trade_dd_pct": float(in_journal.get("avg_trade_dd_pct", 0.0) or 0.0),
        "holdout_trade_dd_pct": float(hold_journal.get("avg_trade_dd_pct", 0.0) or 0.0),
        "in_sample_journal": in_journal,
        "holdout_journal": hold_journal,
    }


def _journal_summary(genes: list[TALibStrategyGene]) -> dict[str, Any]:
    if not genes:
        return {"count": 0}
    truth = np.asarray([float(getattr(g, "truth_probability", 0.0) or 0.0) for g in genes], dtype=np.float64)
    hold_monthly = np.asarray(
        [float(getattr(g, "holdout_monthly_profit_pct", 0.0) or 0.0) for g in genes],
        dtype=np.float64,
    )
    hold_tpm = np.asarray(
        [float(getattr(g, "holdout_trades_per_month", 0.0) or 0.0) for g in genes],
        dtype=np.float64,
    )
    hold_net = np.asarray([float(getattr(g, "holdout_net_profit", 0.0) or 0.0) for g in genes], dtype=np.float64)
    hold_trades = np.asarray([float(getattr(g, "holdout_trades", 0.0) or 0.0) for g in genes], dtype=np.float64)
    ppt = np.divide(hold_net, np.maximum(hold_trades, 1e-9))
    hold_journals = [dict(getattr(g, "holdout_journal", {}) or {}) for g in genes]
    hold_journals = [j for j in hold_journals if bool(j.get("computed", False))]
    avg_hold_hours = float(
        np.mean(
            np.asarray([float(j.get("avg_holding_hours", 0.0) or 0.0) for j in hold_journals], dtype=np.float64)
        )
    ) if hold_journals else 0.0
    avg_trades_day = float(
        np.mean(
            np.asarray([float(j.get("avg_trades_per_day", 0.0) or 0.0) for j in hold_journals], dtype=np.float64)
        )
    ) if hold_journals else 0.0
    avg_trade_dd_pct = float(
        np.mean(
            np.asarray([float(j.get("avg_trade_dd_pct", 0.0) or 0.0) for j in hold_journals], dtype=np.float64)
        )
    ) if hold_journals else 0.0
    return {
        "count": int(len(genes)),
        "avg_truth_probability": float(np.mean(truth)),
        "min_truth_probability": float(np.min(truth)),
        "avg_holdout_monthly_profit_pct": float(np.mean(hold_monthly)),
        "avg_holdout_trades_per_month": float(np.mean(hold_tpm)),
        "avg_holdout_profit_per_trade": float(np.mean(ppt)),
        "avg_holdout_sharpe_ratio": float(
            np.mean(np.asarray([float(getattr(g, "holdout_sharpe_ratio", 0.0) or 0.0) for g in genes], dtype=np.float64))
        ),
        "avg_holdout_win_rate": float(
            np.mean(np.asarray([float(getattr(g, "holdout_win_rate", 0.0) or 0.0) for g in genes], dtype=np.float64))
        ),
        "avg_holdout_profit_factor": float(
            np.mean(np.asarray([float(getattr(g, "holdout_profit_factor", 0.0) or 0.0) for g in genes], dtype=np.float64))
        ),
        "journal_coverage": float(len(hold_journals)) / float(len(genes)) if genes else 0.0,
        "avg_holdout_holding_hours": avg_hold_hours,
        "avg_holdout_trades_per_day": avg_trades_day,
        "avg_holdout_trade_dd_pct": avg_trade_dd_pct,
    }


def _feature_to_indicator(name: str, available: set[str]) -> str | None:
    if not name:
        return None
    raw = str(name).strip()
    if raw.lower().startswith("ta_"):
        raw = raw[3:]
    cand = raw.upper()
    if cand.startswith("SMC_"):
        return cand
    if cand in available:
        return cand
    base = cand.split("_")[0]
    if base in available:
        return base
    return None


def _convert_rust_gene(
    gene: dict[str, Any],
    feature_names: list[str],
    available: set[str],
    metric: Any | None = None,
) -> TALibStrategyGene | None:
    indices = gene.get("indices") or []
    weights = gene.get("weights") or []
    indicators: list[str] = []
    weight_map: dict[str, float] = {}
    params: dict[str, dict[str, Any]] = {}

    for idx, w in zip(indices, weights):
        try:
            i = int(idx)
        except Exception:
            continue
        if i < 0 or i >= len(feature_names):
            continue
        ind = _feature_to_indicator(feature_names[i], available)
        if not ind:
            continue
        indicators.append(ind)
        weight_map[ind] = float(weight_map.get(ind, 0.0) + float(w))
        params.setdefault(ind, {})

    if not indicators:
        return None

    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    metric_row: list[float] = []
    if isinstance(metric, (list, tuple, np.ndarray)):
        for item in metric:
            try:
                metric_row.append(float(item))
            except Exception:
                metric_row.append(0.0)

    def _metric_at(idx: int, default: float = 0.0) -> float:
        if idx < 0 or idx >= len(metric_row):
            return float(default)
        return float(metric_row[idx])

    max_dd_pct = _to_float(
        gene.get(
            "max_dd_pct",
            gene.get("max_drawdown", gene.get("max_dd", gene.get("drawdown", _metric_at(3, 0.0)))),
        ),
        0.0,
    )
    trades = _to_float(
        gene.get("trades", gene.get("trades_count", gene.get("trade_count", _metric_at(8, 0.0)))),
        0.0,
    )
    net_profit = _to_float(gene.get("net_profit", _metric_at(0, 0.0)), 0.0)
    sharpe_ratio = _to_float(gene.get("sharpe_ratio", _metric_at(1, 0.0)), 0.0)
    win_rate = _to_float(gene.get("win_rate", _metric_at(4, 0.0)), 0.0)
    profit_factor = _to_float(gene.get("profit_factor", _metric_at(5, 0.0)), 0.0)
    expectancy = _to_float(gene.get("expectancy", _metric_at(6, 0.0)), 0.0)

    return TALibStrategyGene(
        indicators=indicators,
        params=params,
        weights=weight_map,
        long_threshold=float(gene.get("long_threshold", 0.66)),
        short_threshold=float(gene.get("short_threshold", -0.66)),
        combination_method=str(gene.get("combination_method", "weighted_vote")),
        preferred_regime=str(gene.get("preferred_regime", "any")),
        strategy_id=str(gene.get("strategy_id", "")),
        fitness=float(gene.get("fitness", 0.0)),
        sharpe_ratio=sharpe_ratio,
        win_rate=win_rate,
        max_dd_pct=max_dd_pct,
        trades=trades,
        net_profit=net_profit,
        profit_factor=profit_factor,
        expectancy=expectancy,
        use_ob=bool(gene.get("use_ob", False)),
        use_fvg=bool(gene.get("use_fvg", False)),
        use_liq_sweep=bool(gene.get("use_liq_sweep", False)),
        mtf_confirmation=bool(gene.get("mtf_confirmation", False)),
        use_premium_discount=bool(gene.get("use_premium_discount", False)),
        use_inducement=bool(gene.get("use_inducement", False)),
        use_bos=bool(gene.get("use_bos", False)),
        use_choch=bool(gene.get("use_choch", False)),
        use_eqh=bool(gene.get("use_eqh", False)),
        use_eql=bool(gene.get("use_eql", False)),
        use_displacement=bool(gene.get("use_displacement", False)),
        tp_pips=float(gene.get("tp_pips", 40.0)),
        sl_pips=float(gene.get("sl_pips", 20.0)),
    )


def _evogp_requested(settings: Any | None) -> bool:
    env = os.environ.get("FOREX_BOT_EVOGP_ENABLED")
    if env is not None and str(env).strip() != "":
        return str(env).strip().lower() in {"1", "true", "yes", "on"}
    try:
        if settings is not None and hasattr(settings, "models"):
            enabled = bool(getattr(settings.models, "evogp_enabled", True))
            if not enabled:
                return False
            device = str(getattr(settings.models, "prop_search_device", "cpu") or "cpu").strip().lower()
            return device in {"gpu", "cuda", "auto"}
    except Exception:
        pass
    return False


def _parse_gpu_devices(raw: str | None) -> list[int]:
    if raw is None:
        return []
    txt = str(raw).strip()
    if not txt:
        return []
    out: list[int] = []
    seen: set[int] = set()
    for tok in txt.split(","):
        token = str(tok).strip()
        if not token:
            continue
        try:
            gid = int(token)
        except Exception:
            continue
        if gid < 0 or gid in seen:
            continue
        seen.add(gid)
        out.append(gid)
    return out


def _convert_gpu_genome(
    *,
    genome: Any,
    fitness: float,
    feature_names: list[str],
    available: set[str],
    max_indicators: int,
    threshold_scale: float,
    threshold_margin: float,
    threshold_clip: float,
    strategy_id: str,
) -> TALibStrategyGene | None:
    arr = np.asarray(genome, dtype=np.float64).reshape(-1)
    n_features = int(len(feature_names))
    if n_features <= 0 or arr.size < (n_features + 3):
        return None

    tf_count = int(arr.size - n_features - 2)
    if tf_count < 1:
        tf_count = 1
    start = tf_count
    end = start + n_features
    if end + 2 > arr.size:
        return None
    logic = arr[start:end]
    if logic.size != n_features:
        return None

    order = _rust_rank_scores_desc(logic, absolute=True)
    if order is None:
        order = np.argsort(np.abs(logic))[::-1]
    indicators: list[str] = []
    weights: dict[str, float] = {}
    params: dict[str, dict[str, Any]] = {}
    cap = max(1, int(max_indicators or 1))
    for idx in order:
        i = int(idx)
        if i < 0 or i >= n_features:
            continue
        ind = _feature_to_indicator(feature_names[i], available)
        if not ind or ind in weights:
            continue
        w = float(logic[i])
        if not np.isfinite(w):
            continue
        indicators.append(ind)
        weights[ind] = w
        params[ind] = {}
        if len(indicators) >= cap:
            break
    if not indicators:
        return None

    denom = float(sum(abs(weights[k]) for k in indicators))
    if not np.isfinite(denom) or denom <= 0.0:
        weights = {k: 1.0 for k in indicators}
    else:
        weights = {k: float(weights[k] / denom) for k in indicators}

    t0 = float(np.clip(arr[end], -threshold_clip, threshold_clip) * threshold_scale)
    t1 = float(np.clip(arr[end + 1], -threshold_clip, threshold_clip) * threshold_scale)
    long_thr = float(np.clip(max(t0, t1) + threshold_margin, 0.05, 1.25))
    short_thr = float(np.clip(min(t0, t1) - threshold_margin, -1.25, -0.05))

    fit = float(fitness) if np.isfinite(float(fitness)) else 0.0
    return TALibStrategyGene(
        indicators=indicators,
        params=params,
        weights=weights,
        long_threshold=long_thr,
        short_threshold=short_thr,
        combination_method="weighted_vote",
        preferred_regime="any",
        strategy_id=strategy_id,
        fitness=fit,
        sharpe_ratio=0.0,
        win_rate=0.0,
        max_dd_pct=0.0,
        trades=0.0,
        net_profit=0.0,
        profit_factor=0.0,
        expectancy=0.0,
        use_ob=False,
        use_fvg=False,
        use_liq_sweep=False,
        mtf_confirmation=False,
        use_premium_discount=False,
        use_inducement=False,
        use_bos=False,
        use_choch=False,
        use_eqh=False,
        use_eql=False,
        use_displacement=False,
        tp_pips=40.0,
        sl_pips=20.0,
    )


def _resolve_sl_tp(
    *,
    gene: TALibStrategyGene,
    settings: Any,
    pip_size: float,
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    atr_values: np.ndarray | None,
) -> tuple[float, float]:
    sl_cfg = None
    tp_cfg = None
    try:
        sl_cfg = getattr(settings.risk, "meta_label_sl_pips", None)
        tp_cfg = getattr(settings.risk, "meta_label_tp_pips", None)
    except Exception:
        sl_cfg = None
        tp_cfg = None

    if sl_cfg is not None or tp_cfg is not None:
        sl_pips = float(sl_cfg) if sl_cfg is not None else float(getattr(gene, "sl_pips", 30.0) or 30.0)
        rr = 2.0
        try:
            rr = float(getattr(settings.risk, "min_risk_reward", 2.0) or 2.0)
        except Exception:
            rr = 2.0
        rr = max(1.5, rr)
        if tp_cfg is None:
            tp_pips = sl_pips * rr
        else:
            tp_pips = max(float(tp_cfg), sl_pips * rr)
        return float(sl_pips), float(tp_pips)

    atr_mult = 1.5
    min_rr = 2.0
    min_dist = 0.0
    try:
        atr_mult = float(getattr(settings.risk, "atr_stop_multiplier", 1.5) or 1.5)
        min_rr = float(getattr(settings.risk, "min_risk_reward", 2.0) or 2.0)
        min_dist = float(getattr(settings.risk, "meta_label_min_dist", 0.0) or 0.0)
    except Exception:
        pass
    min_rr = max(1.5, min_rr)

    auto = infer_sl_tp_pips_auto(
        open_prices=open_prices,
        high_prices=high_prices,
        low_prices=low_prices,
        close_prices=close_prices,
        atr_values=atr_values,
        pip_size=pip_size,
        atr_mult=atr_mult,
        min_rr=min_rr,
        min_dist=min_dist,
        settings=settings,
    )
    if auto:
        return float(auto[0]), float(auto[1])

    sl_pips = float(getattr(gene, "sl_pips", 30.0) or 30.0)
    tp_pips = float(getattr(gene, "tp_pips", 60.0) or 60.0)
    return float(sl_pips), float(tp_pips)


def _timeframe_hours(tf: str) -> float:
    raw = str(tf or "").strip().upper()
    if not raw:
        return 1.0 / 60.0
    try:
        if raw.startswith("MN"):
            n = max(1.0, float(raw[2:] or 1.0))
            return n * 24.0 * 30.4375
        unit = raw[0]
        n = max(1.0, float(raw[1:] or 1.0))
        if unit == "M":
            return n / 60.0
        if unit == "H":
            return n
        if unit == "D":
            return n * 24.0
        if unit == "W":
            return n * 24.0 * 7.0
    except Exception:
        return 1.0 / 60.0
    return 1.0 / 60.0


def _index_ms_and_bar_hours(df: Any) -> tuple[np.ndarray | None, float]:
    idx = _frame_index(df)
    if _is_datetime_index(idx) and len(idx) > 0:
        try:
            if hasattr(idx, "tz_localize"):
                if idx.tz is None:
                    i2 = idx.tz_localize("UTC")
                else:
                    i2 = idx.tz_convert("UTC")
            else:
                i2 = idx
        except Exception:
            i2 = idx
        raw = _index_to_ns_int64(i2)
        if raw.size <= 0:
            return None, _timeframe_hours(str(_frame_attr(df, "timeframe", _frame_attr(df, "tf", "M1"))))
        abs_max = float(np.max(np.abs(raw))) if raw.size > 0 else 0.0
        # Support dataframe int64 datetime representations in ns/us/ms/s.
        if abs_max > 1e16:
            scale_to_ms = 1.0 / 1_000_000.0  # ns -> ms
        elif abs_max > 1e13:
            scale_to_ms = 1.0 / 1_000.0  # us -> ms
        elif abs_max > 1e11:
            scale_to_ms = 1.0  # ms -> ms
        else:
            scale_to_ms = 1_000.0  # s -> ms
        ts_ms = np.asarray(np.round(raw.astype(np.float64) * scale_to_ms), dtype=np.int64)
        if ts_ms.size >= 2:
            delta = np.diff(ts_ms)
            delta = delta[delta > 0]
            if delta.size > 0:
                bar_hours = float(np.median(delta) / 3_600_000.0)
                return ts_ms, max(1e-9, bar_hours)
        return ts_ms, _timeframe_hours(str(_frame_attr(df, "timeframe", _frame_attr(df, "tf", "M1"))))
    return None, _timeframe_hours(str(_frame_attr(df, "timeframe", _frame_attr(df, "tf", "M1"))))


def _journal_month_day_codes(ts_ms: np.ndarray | None, n: int, bar_hours: float) -> tuple[np.ndarray, np.ndarray]:
    if ts_ms is not None and ts_ms.size == n:
        try:
            dt_ms = np.asarray(ts_ms, dtype=np.int64).astype("datetime64[ms]")
            months = np.datetime_as_string(dt_ms.astype("datetime64[M]"), unit="M")
            days = np.datetime_as_string(dt_ms.astype("datetime64[D]"), unit="D")
            month_codes = np.char.replace(months, "-", "").astype(np.int64, copy=False)
            day_codes = np.char.replace(days, "-", "").astype(np.int64, copy=False)
            return month_codes, day_codes
        except Exception:
            pass
    bars_per_month = max(1, int(30.0 * 24.0 / max(1e-9, float(bar_hours))))
    bars_per_day = max(1, int(24.0 / max(1e-9, float(bar_hours))))
    idx = np.arange(max(0, int(n)), dtype=np.int64)
    month_codes = (idx // bars_per_month) + 1
    day_codes = (idx // bars_per_day) + 1
    return month_codes, day_codes


def _trade_journal_from_signals(
    *,
    df: Any,
    signals: np.ndarray,
    sl_pips: float,
    tp_pips: float,
    pip_value: float,
    pip_value_per_lot: float,
    spread_pips: float,
    commission_per_trade: float,
    max_hold_bars: int = 0,
    trailing_enabled: bool = False,
    trailing_atr_multiplier: float = 1.0,
    trailing_be_trigger_r: float = 1.0,
) -> dict[str, Any]:
    n = int(_frame_len(df))
    if n <= 1:
        return {"computed": False, "reason": "insufficient_rows"}

    if not (_RUST_TRADE_JOURNAL and _fb is not None):
        return {"computed": False, "reason": "rust_trade_journal_unavailable"}

    close = _frame_column_numpy(df, "close", dtype=np.float64)
    high = _frame_column_numpy(df, "high", dtype=np.float64)
    low = _frame_column_numpy(df, "low", dtype=np.float64)
    sig = np.asarray(signals, dtype=np.int8)
    if sig.shape[0] != n:
        return {"computed": False, "reason": "shape_mismatch"}

    ts_ms, bar_hours = _index_ms_and_bar_hours(df)
    history_days, history_months = _history_span_days_months(df)
    pip_value = float(max(1e-12, abs(pip_value)))
    cash_per_pip = float(pip_value_per_lot)
    swap_long_per_day = _env_float("FOREX_BOT_PROP_SWAP_LONG_PER_DAY", 0.0)
    swap_short_per_day = _env_float("FOREX_BOT_PROP_SWAP_SHORT_PER_DAY", 0.0)
    month_codes, day_codes = _journal_month_day_codes(ts_ms, n, bar_hours)
    try:
        rust_out = _fb.trade_journal_metrics(
            close,
            high,
            low,
            sig,
            month_codes,
            day_codes,
            float(sl_pips),
            float(tp_pips),
            max_hold_bars=int(max_hold_bars),
            trailing_enabled=bool(trailing_enabled),
            trailing_atr_multiplier=float(trailing_atr_multiplier),
            trailing_be_trigger_r=float(trailing_be_trigger_r),
            pip_value=float(pip_value),
            spread_pips=float(spread_pips),
            commission_per_trade=float(commission_per_trade),
            pip_value_per_lot=float(cash_per_pip),
            history_days=float(history_days),
            history_months=float(history_months),
            bar_hours=float(bar_hours),
            timestamps_ms=(np.asarray(ts_ms, dtype=np.int64) if ts_ms is not None else None),
            swap_long_per_day=float(swap_long_per_day),
            swap_short_per_day=float(swap_short_per_day),
        )
    except Exception:
        return {"computed": False, "reason": "rust_trade_journal_failed"}

    if isinstance(rust_out, dict) and bool(rust_out.get("computed", False)):
        return rust_out
    return {"computed": False, "reason": "rust_trade_journal_unavailable"}


def _attach_trade_journals(
    *,
    selected: list[TALibStrategyGene],
    search_df: Any,
    holdout_df: Any | None,
    settings: Any,
) -> None:
    if not selected:
        return
    top_k = int(max(0.0, _env_float("FOREX_BOT_PROP_JOURNAL_TOP_K", 10.0)))
    if top_k <= 0:
        return

    targets = selected[: min(len(selected), top_k)]
    rust_bulk_available = _fb is not None and hasattr(_fb, "talib_bulk_signals_ohlcv")
    if not rust_bulk_available:
        for gene in targets:
            gene.in_sample_journal = {"computed": False, "reason": "rust_signal_unavailable"}
            if _frame_empty(holdout_df):
                gene.holdout_journal = {"computed": False, "reason": "no_holdout"}
            else:
                gene.holdout_journal = {"computed": False, "reason": "rust_signal_unavailable"}
        return

    try:
        mixer = TALibStrategyMixer()
    except Exception as exc:
        logger.warning("Trade journal precompute failed: %s", exc)
        return

    spread = float(os.environ.get("FOREX_BOT_PROP_EVAL_SPREAD_PIPS", "1.5") or 1.5)
    commission = float(os.environ.get("FOREX_BOT_PROP_EVAL_COMMISSION", "7.0") or 7.0)

    def _rust_bulk_signal_map(local_df: Any | None) -> dict[tuple[Any, ...], np.ndarray] | None:
        if _frame_empty(local_df):
            return None
        if _fb is None or not hasattr(_fb, "talib_bulk_signals_ohlcv"):
            return None
        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset({str(c).lower() for c in _frame_columns(local_df)}):
            return None

        indicator_sets: list[list[str]] = []
        weight_sets: list[list[float]] = []
        long_thr: list[float] = []
        short_thr: list[float] = []
        keys: list[tuple[Any, ...]] = []
        for gene in targets:
            inds = [str(i).upper() for i in (gene.indicators or []) if str(i).strip()]
            if not inds:
                continue
            keys.append(mixer._gene_key(gene))
            indicator_sets.append(inds)
            weights_map = gene.weights or {}
            weight_sets.append(
                [float(weights_map.get(ind, weights_map.get(ind.lower(), 1.0)) or 1.0) for ind in inds]
            )
            long_thr.append(float(gene.long_threshold))
            short_thr.append(float(gene.short_threshold))
        if not indicator_sets:
            return None

        open_arr = _frame_column_numpy(local_df, "open", dtype=np.float64)
        high_arr = _frame_column_numpy(local_df, "high", dtype=np.float64)
        low_arr = _frame_column_numpy(local_df, "low", dtype=np.float64)
        close_arr = _frame_column_numpy(local_df, "close", dtype=np.float64)
        volume_arr = (
            _frame_column_numpy(local_df, "volume", dtype=np.float64)
            if _frame_has_column(local_df, "volume")
            else None
        )
        local_idx = _frame_index(local_df)
        timestamps = _datetime_index_to_unix_ms(local_idx) if _is_datetime_index(local_idx) else None
        try:
            causal_min_bars = max(2, int(os.environ.get("FOREX_BOT_TALIB_CAUSAL_MIN_BARS", "30") or 30))
        except Exception:
            causal_min_bars = 30
        try:
            raw = _fb.talib_bulk_signals_ohlcv(  # type: ignore[attr-defined]
                open_arr,
                high_arr,
                low_arr,
                close_arr,
                indicator_sets=indicator_sets,
                weight_sets=weight_sets,
                long_thresholds=long_thr,
                short_thresholds=short_thr,
                timestamps=timestamps,
                volume=volume_arr,
                include_raw=False,
                causal_min_bars=causal_min_bars,
            )
        except TypeError:
            raw = _fb.talib_bulk_signals_ohlcv(  # type: ignore[attr-defined]
                open_arr,
                high_arr,
                low_arr,
                close_arr,
                indicator_sets=indicator_sets,
                weight_sets=weight_sets,
                long_thresholds=long_thr,
                short_thresholds=short_thr,
                timestamps=timestamps,
                volume=volume_arr,
                include_raw=False,
            )
        except Exception:
            return None

        sig = np.asarray(raw, dtype=np.int8)
        n_rows = _frame_len(local_df)
        n_genes = int(len(keys))
        if sig.ndim != 2:
            return None
        if sig.shape[0] == n_rows and sig.shape[1] == n_genes:
            sig = sig.T
        elif not (sig.shape[0] == n_genes and sig.shape[1] == n_rows):
            return None

        out: dict[tuple[Any, ...], np.ndarray] = {}
        for row_idx, key in enumerate(keys):
            out[key] = np.asarray(sig[row_idx], dtype=np.int8).reshape(-1)
        return out

    search_rust_map = _rust_bulk_signal_map(search_df)
    holdout_rust_map = _rust_bulk_signal_map(holdout_df)

    for gene in targets:
        try:
            rust_key = mixer._gene_key(gene)
            sig_search = None
            if search_rust_map is not None and rust_key in search_rust_map:
                sig_search = np.asarray(search_rust_map[rust_key], dtype=np.int8).reshape(-1)
            if sig_search is None or sig_search.size <= 0:
                gene.in_sample_journal = {"computed": False, "reason": "rust_signal_unavailable"}
                continue
            search_n = _frame_len(search_df)
            if sig_search.shape[0] != search_n:
                sig_search = np.resize(sig_search, search_n).astype(np.int8, copy=False)
            close = _frame_column_numpy(search_df, "close", dtype=np.float64)
            high = _frame_column_numpy(search_df, "high", dtype=np.float64)
            low = _frame_column_numpy(search_df, "low", dtype=np.float64)
            open_ = (
                _frame_column_numpy(search_df, "open", dtype=np.float64)
                if _frame_has_column(search_df, "open")
                else close
            )
            atr_vals = (
                _frame_column_numpy(search_df, "atr", dtype=np.float64)
                if _frame_has_column(search_df, "atr")
                else None
            )
            pip_size, pip_val = _df_pip_metrics(search_df, close=close)
            sl_pips, tp_pips = _resolve_sl_tp(
                gene=gene,
                settings=settings,
                pip_size=pip_size,
                open_prices=open_,
                high_prices=high,
                low_prices=low,
                close_prices=close,
                atr_values=atr_vals,
            )
            gene.in_sample_journal = _trade_journal_from_signals(
                df=search_df,
                signals=sig_search,
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                pip_value=pip_size,
                pip_value_per_lot=pip_val,
                spread_pips=spread,
                commission_per_trade=commission,
            )
        except Exception as exc:
            gene.in_sample_journal = {"computed": False, "reason": f"error:{exc}"}

        if _frame_empty(holdout_df):
            gene.holdout_journal = {"computed": False, "reason": "no_holdout"}
            continue

        try:
            rust_key = mixer._gene_key(gene)
            sig_hold = None
            if holdout_rust_map is not None and rust_key in holdout_rust_map:
                sig_hold = np.asarray(holdout_rust_map[rust_key], dtype=np.int8).reshape(-1)
            if sig_hold is None or sig_hold.size <= 0:
                gene.holdout_journal = {"computed": False, "reason": "rust_signal_unavailable"}
                continue
            holdout_n = _frame_len(holdout_df)
            if sig_hold.shape[0] != holdout_n:
                sig_hold = np.resize(sig_hold, holdout_n).astype(np.int8, copy=False)
            close_h = _frame_column_numpy(holdout_df, "close", dtype=np.float64)
            high_h = _frame_column_numpy(holdout_df, "high", dtype=np.float64)
            low_h = _frame_column_numpy(holdout_df, "low", dtype=np.float64)
            open_h = (
                _frame_column_numpy(holdout_df, "open", dtype=np.float64)
                if _frame_has_column(holdout_df, "open")
                else close_h
            )
            atr_h = (
                _frame_column_numpy(holdout_df, "atr", dtype=np.float64)
                if _frame_has_column(holdout_df, "atr")
                else None
            )
            pip_size_h, pip_val_h = _df_pip_metrics(holdout_df, close=close_h)
            sl_h, tp_h = _resolve_sl_tp(
                gene=gene,
                settings=settings,
                pip_size=pip_size_h,
                open_prices=open_h,
                high_prices=high_h,
                low_prices=low_h,
                close_prices=close_h,
                atr_values=atr_h,
            )
            gene.holdout_journal = _trade_journal_from_signals(
                df=holdout_df,
                signals=sig_hold,
                sl_pips=sl_h,
                tp_pips=tp_h,
                pip_value=pip_size_h,
                pip_value_per_lot=pip_val_h,
                spread_pips=spread,
                commission_per_trade=commission,
            )
        except Exception as exc:
            gene.holdout_journal = {"computed": False, "reason": f"error:{exc}"}


def _evaluate_gene(
    df: Any,
    gene: TALibStrategyGene,
    mixer: TALibStrategyMixer,
    cache: dict[str, Any] | None,
    settings: Any,
) -> float:
    if _RUST_POP_EVAL:
        try:
            metrics_mat = _batch_evaluate_population_rust(
                df=df,
                genes=[gene],
                mixer=mixer,
                cache=cache,
                settings=settings,
            )
            if metrics_mat is not None and np.asarray(metrics_mat).ndim == 2 and len(metrics_mat) >= 1:
                return _apply_metrics_to_gene(gene, np.asarray(metrics_mat[0], dtype=np.float64))
        except Exception as exc:
            logger.debug("Rust single-gene eval failed: %s", exc, exc_info=True)
    return _apply_metrics_to_gene(gene, np.zeros(0, dtype=np.float64))


def _apply_metrics_to_gene(gene: TALibStrategyGene, metrics: np.ndarray) -> float:
    if metrics is None or metrics.size < 11:
        gene.fitness = 0.0
        gene.net_profit = 0.0
        gene.profit_factor = 0.0
        gene.expectancy = 0.0
        gene.trades = 0.0
        gene.sharpe_ratio = 0.0
        gene.max_dd_pct = 0.0
        gene.win_rate = 0.0
        return 0.0
    gene.fitness = float(metrics[0])
    gene.sharpe_ratio = float(metrics[1])
    gene.max_dd_pct = float(metrics[3])
    gene.win_rate = float(metrics[4])
    gene.trades = float(metrics[8])
    gene.net_profit = float(metrics[0])
    gene.profit_factor = float(metrics[5])
    gene.expectancy = float(metrics[6])
    return float(gene.fitness)


def _batch_evaluate_population_rust(
    df: Any,
    genes: list[TALibStrategyGene],
    mixer: TALibStrategyMixer,
    cache: dict[str, Any] | None,
    settings: Any,
) -> np.ndarray | None:
    if not genes or _fb is None:
        return None
    try:
        n = _frame_len(df)
        if n == 0:
            return None
        idx = _frame_index(df)
        month_idx, day_idx = _safe_indices(idx, n)
        close = _frame_column_numpy(df, "close", dtype=np.float64)
        high = _frame_column_numpy(df, "high", dtype=np.float64)
        low = _frame_column_numpy(df, "low", dtype=np.float64)
        open_ = _frame_column_numpy(df, "open", dtype=np.float64) if _frame_has_column(df, "open") else close
        atr_vals = _frame_column_numpy(df, "atr", dtype=np.float64) if _frame_has_column(df, "atr") else None
        volume_arr = _frame_column_numpy(df, "volume", dtype=np.float64) if _frame_has_column(df, "volume") else None
        pip_size, pip_val = _df_pip_metrics(df, close=close)
        spread = float(os.environ.get("FOREX_BOT_PROP_EVAL_SPREAD_PIPS", "1.5") or 1.5)
        commission = float(os.environ.get("FOREX_BOT_PROP_EVAL_COMMISSION", "7.0") or 7.0)
        causal_min_bars = max(2, int(os.environ.get("FOREX_BOT_TALIB_CAUSAL_MIN_BARS", "30") or 30))
        smc_gate_threshold = float(os.environ.get("FOREX_BOT_SMC_GATE_THRESHOLD", "0.0") or 0.0)
        smc_weight_ob = float(os.environ.get("FOREX_BOT_SMC_WEIGHT_OB", "1.0") or 1.0)
        smc_weight_fvg = float(os.environ.get("FOREX_BOT_SMC_WEIGHT_FVG", "1.0") or 1.0)
        smc_weight_liq = float(os.environ.get("FOREX_BOT_SMC_WEIGHT_LIQ", "1.0") or 1.0)
        smc_weight_mtf = float(os.environ.get("FOREX_BOT_SMC_WEIGHT_MTF", "1.0") or 1.0)
        smc_weight_premium = float(os.environ.get("FOREX_BOT_SMC_WEIGHT_PREMIUM", "1.0") or 1.0)
        smc_weight_inducement = float(os.environ.get("FOREX_BOT_SMC_WEIGHT_INDUCEMENT", "1.0") or 1.0)
        smc_weight_bos = float(os.environ.get("FOREX_BOT_SMC_WEIGHT_BOS", "1.0") or 1.0)
        smc_weight_choch = float(os.environ.get("FOREX_BOT_SMC_WEIGHT_CHOCH", "1.0") or 1.0)
        smc_weight_eqh = float(os.environ.get("FOREX_BOT_SMC_WEIGHT_EQH", "1.0") or 1.0)
        smc_weight_eql = float(os.environ.get("FOREX_BOT_SMC_WEIGHT_EQL", "1.0") or 1.0)
        smc_weight_displacement = float(os.environ.get("FOREX_BOT_SMC_WEIGHT_DISPLACEMENT", "1.0") or 1.0)

        # Preferred: full Rust path computing indicators + evaluation in one shot.
        if _RUST_TALIB_POP:
            indicator_sets: list[list[str]] = []
            weight_sets: list[list[float]] = []
            long_thr: list[float] = []
            short_thr: list[float] = []
            sl_arr: list[float] = []
            tp_arr: list[float] = []
            use_ob: list[int] = []
            use_fvg: list[int] = []
            use_liq: list[int] = []
            use_mtf: list[int] = []
            use_premium: list[int] = []
            use_inducement: list[int] = []
            use_bos: list[int] = []
            use_choch: list[int] = []
            use_eqh: list[int] = []
            use_eql: list[int] = []
            use_displacement: list[int] = []
            for gene in genes:
                inds = [str(i).upper() for i in (gene.indicators or []) if str(i).strip()]
                indicator_sets.append(inds)
                weights_map = gene.weights or {}
                weight_sets.append(
                    [
                        float(weights_map.get(ind, weights_map.get(ind.lower(), 1.0)) or 1.0)
                        for ind in inds
                    ]
                    if inds
                    else []
                )
                long_thr.append(float(gene.long_threshold))
                short_thr.append(float(gene.short_threshold))
                sl_pips, tp_pips = _resolve_sl_tp(
                    gene=gene,
                    settings=settings,
                    pip_size=pip_size,
                    open_prices=open_,
                    high_prices=high,
                    low_prices=low,
                    close_prices=close,
                    atr_values=atr_vals,
                )
                sl_arr.append(float(sl_pips))
                tp_arr.append(float(tp_pips))
                use_ob.append(1 if getattr(gene, "use_ob", False) else 0)
                use_fvg.append(1 if getattr(gene, "use_fvg", False) else 0)
                use_liq.append(1 if getattr(gene, "use_liq_sweep", False) else 0)
                use_mtf.append(1 if getattr(gene, "mtf_confirmation", False) else 0)
                use_premium.append(1 if getattr(gene, "use_premium_discount", False) else 0)
                use_inducement.append(1 if getattr(gene, "use_inducement", False) else 0)
                use_bos.append(1 if getattr(gene, "use_bos", False) else 0)
                use_choch.append(1 if getattr(gene, "use_choch", False) else 0)
                use_eqh.append(1 if getattr(gene, "use_eqh", False) else 0)
                use_eql.append(1 if getattr(gene, "use_eql", False) else 0)
                use_displacement.append(1 if getattr(gene, "use_displacement", False) else 0)
            try:
                metrics = _fb.evaluate_population_talib_ohlcv(  # type: ignore[attr-defined]
                    open_,
                    high,
                    low,
                    close,
                    indicator_sets=indicator_sets,
                    weight_sets=weight_sets,
                    long_thresholds=long_thr,
                    short_thresholds=short_thr,
                    sl_pips=sl_arr,
                    tp_pips=tp_arr,
                    use_ob_flags=use_ob,
                    use_fvg_flags=use_fvg,
                    use_liq_flags=use_liq,
                    use_mtf_flags=use_mtf,
                    use_premium_flags=use_premium,
                    use_inducement_flags=use_inducement,
                    timestamps=_datetime_index_to_unix_ms(idx) if _is_datetime_index(idx) else None,
                    volume=volume_arr,
                    include_raw=True,
                    smc_gate_threshold=smc_gate_threshold,
                    smc_weight_ob=smc_weight_ob,
                    smc_weight_fvg=smc_weight_fvg,
                    smc_weight_liq=smc_weight_liq,
                    smc_weight_mtf=smc_weight_mtf,
                    smc_weight_premium=smc_weight_premium,
                    smc_weight_inducement=smc_weight_inducement,
                    max_hold_bars=int(os.environ.get("FOREX_BOT_PROP_MAX_HOLD_BARS", "0") or 0),
                    trailing_enabled=False,
                    trailing_atr_multiplier=1.0,
                    trailing_be_trigger_r=1.0,
                    pip_value=float(pip_size),
                    spread_pips=float(spread),
                    commission_per_trade=float(commission),
                    pip_value_per_lot=float(pip_val),
                    causal_min_bars=causal_min_bars,
                    use_bos_flags=use_bos,
                    use_choch_flags=use_choch,
                    use_eqh_flags=use_eqh,
                    use_eql_flags=use_eql,
                    use_displacement_flags=use_displacement,
                    smc_weight_bos=smc_weight_bos,
                    smc_weight_choch=smc_weight_choch,
                    smc_weight_eqh=smc_weight_eqh,
                    smc_weight_eql=smc_weight_eql,
                    smc_weight_displacement=smc_weight_displacement,
                )
            except TypeError:
                metrics = _fb.evaluate_population_talib_ohlcv(  # type: ignore[attr-defined]
                    open_,
                    high,
                    low,
                    close,
                    indicator_sets=indicator_sets,
                    weight_sets=weight_sets,
                    long_thresholds=long_thr,
                    short_thresholds=short_thr,
                    sl_pips=sl_arr,
                    tp_pips=tp_arr,
                    timestamps=_datetime_index_to_unix_ms(idx) if _is_datetime_index(idx) else None,
                    volume=volume_arr,
                    include_raw=True,
                    max_hold_bars=int(os.environ.get("FOREX_BOT_PROP_MAX_HOLD_BARS", "0") or 0),
                    trailing_enabled=False,
                    trailing_atr_multiplier=1.0,
                    trailing_be_trigger_r=1.0,
                    pip_value=float(pip_size),
                    spread_pips=float(spread),
                    commission_per_trade=float(commission),
                    pip_value_per_lot=float(pip_val),
                )
            return np.asarray(metrics, dtype=np.float64)

        # Fast-path: if Rust mixer already produced per-gene signals, batch-evaluate them in Rust.
        rust_sig_cache = getattr(mixer, "_rust_signal_cache", {}) or {}
        rust_sig_index = getattr(mixer, "_rust_signal_index", None)
        if rust_sig_cache and rust_sig_index is not None and len(rust_sig_index) == n:
            keys = [mixer._gene_key(g) for g in genes]
            if all(k in rust_sig_cache for k in keys):
                signals_mat = np.zeros((len(genes), n), dtype=np.int8)
                for row_idx, key in enumerate(keys):
                    series = rust_sig_cache[key]
                    try:
                        signals_mat[row_idx, :] = signal_to_numpy(
                            series,
                            index=idx,
                            dtype=np.int8,
                            fill_value=0,
                            forward_fill=False,
                        )
                    except Exception:
                        signals_mat[row_idx, :] = 0
                sl_arr: list[float] = []
                tp_arr: list[float] = []
                for g in genes:
                    sl_pips, tp_pips = _resolve_sl_tp(
                        gene=g,
                        settings=settings,
                        pip_size=pip_size,
                        open_prices=open_,
                        high_prices=high,
                        low_prices=low,
                        close_prices=close,
                        atr_values=atr_vals,
                    )
                    sl_arr.append(float(sl_pips))
                    tp_arr.append(float(tp_pips))

                metrics = _fb.batch_evaluate_strategies(  # type: ignore[attr-defined]
                    close,
                    high,
                    low,
                    signals_mat,
                    month_idx,
                    day_idx,
                    np.asarray(sl_arr, dtype=np.float64),
                    np.asarray(tp_arr, dtype=np.float64),
                    int(os.environ.get("FOREX_BOT_PROP_MAX_HOLD_BARS", "0") or 0),
                    False,
                    1.0,
                    1.0,
                    float(pip_size),
                    float(spread),
                    float(commission),
                    float(pip_val),
                )
                return np.asarray(metrics, dtype=np.float64)

        # Bridge path: compute per-gene signals in Rust bulk, then batch-evaluate in Rust.
        if hasattr(_fb, "talib_bulk_signals_ohlcv") and hasattr(_fb, "batch_evaluate_strategies"):
            indicator_sets: list[list[str]] = []
            weight_sets: list[list[float]] = []
            long_thr: list[float] = []
            short_thr: list[float] = []
            sl_arr: list[float] = []
            tp_arr: list[float] = []
            mapped_genes: list[TALibStrategyGene] = []

            for gene in genes:
                inds = [str(i).upper() for i in (gene.indicators or []) if str(i).strip()]
                if not inds:
                    continue
                mapped_genes.append(gene)
                indicator_sets.append(inds)
                weights_map = gene.weights or {}
                weight_sets.append(
                    [
                        float(weights_map.get(ind, weights_map.get(ind.lower(), 1.0)) or 1.0)
                        for ind in inds
                    ]
                )
                long_thr.append(float(gene.long_threshold))
                short_thr.append(float(gene.short_threshold))
                sl_pips, tp_pips = _resolve_sl_tp(
                    gene=gene,
                    settings=settings,
                    pip_size=pip_size,
                    open_prices=open_,
                    high_prices=high,
                    low_prices=low,
                    close_prices=close,
                    atr_values=atr_vals,
                )
                sl_arr.append(float(sl_pips))
                tp_arr.append(float(tp_pips))

            if mapped_genes:
                timestamps = _datetime_index_to_unix_ms(idx) if _is_datetime_index(idx) else None
                try:
                    raw = _fb.talib_bulk_signals_ohlcv(  # type: ignore[attr-defined]
                        open_,
                        high,
                        low,
                        close,
                        indicator_sets=indicator_sets,
                        weight_sets=weight_sets,
                        long_thresholds=long_thr,
                        short_thresholds=short_thr,
                        timestamps=timestamps,
                        volume=volume_arr,
                        include_raw=False,
                        causal_min_bars=causal_min_bars,
                    )
                except TypeError:
                    raw = _fb.talib_bulk_signals_ohlcv(  # type: ignore[attr-defined]
                        open_,
                        high,
                        low,
                        close,
                        indicator_sets=indicator_sets,
                        weight_sets=weight_sets,
                        long_thresholds=long_thr,
                        short_thresholds=short_thr,
                        timestamps=timestamps,
                        volume=volume_arr,
                        include_raw=False,
                    )

                sig = np.asarray(raw, dtype=np.int8)
                n_genes = int(len(mapped_genes))
                if sig.ndim != 2:
                    return None
                if sig.shape[0] == n and sig.shape[1] == n_genes:
                    sig = sig.T
                elif not (sig.shape[0] == n_genes and sig.shape[1] == n):
                    return None

                metrics = _fb.batch_evaluate_strategies(  # type: ignore[attr-defined]
                    close,
                    high,
                    low,
                    sig,
                    month_idx,
                    day_idx,
                    np.asarray(sl_arr, dtype=np.float64),
                    np.asarray(tp_arr, dtype=np.float64),
                    int(os.environ.get("FOREX_BOT_PROP_MAX_HOLD_BARS", "0") or 0),
                    False,
                    1.0,
                    1.0,
                    float(pip_size),
                    float(spread),
                    float(commission),
                    float(pip_val),
                )
                return np.asarray(metrics, dtype=np.float64)

        return None
    except Exception as exc:
        logger.debug("Rust batch population eval failed: %s", exc, exc_info=True)
        return None


def _expand_threshold_variants(
    *,
    df: Any,
    genes: list[TALibStrategyGene],
    settings: Any,
) -> list[TALibStrategyGene]:
    try:
        threshold_steps = int(os.environ.get("FOREX_BOT_PROP_EXPAND_THRESHOLDS", "0") or 0)
    except Exception:
        threshold_steps = 0
    if threshold_steps <= 0:
        return genes

    try:
        max_total = int(os.environ.get("FOREX_BOT_PROP_EXPAND_MAX_TOTAL", "0") or 0)
    except Exception:
        max_total = 0

    mixer = TALibStrategyMixer()
    if not mixer.available_indicators:
        return genes

    base = [g for g in genes if getattr(g, "indicators", None)]
    if not base:
        return genes

    try:
        cache = mixer.bulk_calculate_indicators(df, base)
    except Exception as exc:
        logger.warning("Threshold expansion indicator precompute failed: %s", exc)
        return _dedupe_ranked(genes)
    levels = np.linspace(0.05, 0.75, num=max(1, threshold_steps), dtype=np.float64)

    expanded: list[TALibStrategyGene] = []
    for gene in base:
        try:
            _evaluate_gene(df, gene, mixer, cache, settings)
        except Exception:
            pass
        expanded.append(gene)

    for gene in base:
        sid = str(getattr(gene, "strategy_id", "") or "gene")
        for lvl in levels:
            long_thr = float(lvl)
            short_thr = -float(lvl)
            if abs(float(getattr(gene, "long_threshold", 0.66)) - long_thr) < 1e-12 and abs(
                float(getattr(gene, "short_threshold", -0.66)) - short_thr
            ) < 1e-12:
                continue
            variant = replace(
                gene,
                long_threshold=long_thr,
                short_threshold=short_thr,
                strategy_id=f"{sid}_thr_{long_thr:.3f}",
            )
            try:
                _evaluate_gene(df, variant, mixer, cache, settings)
            except Exception:
                pass
            expanded.append(variant)
            if max_total > 0 and len(expanded) >= max_total:
                break
        if max_total > 0 and len(expanded) >= max_total:
            break

    return _dedupe_ranked(expanded)


def _strategy_keep_limits(settings: Any) -> tuple[float, float, float, int, int]:
    elite = _elite_filter_enabled()
    default_dd = 0.03 if elite else float(getattr(settings.risk, "total_drawdown_limit", 0.07) or 0.07)
    try:
        max_dd = float(
            os.environ.get(
                "FOREX_BOT_PROP_KEEP_MAX_DD",
                default_dd,
            )
            or default_dd
        )
    except Exception:
        max_dd = default_dd
    max_dd = float(min(1.0, max(0.0, max_dd)))

    try:
        min_profit = float(os.environ.get("FOREX_BOT_PROP_KEEP_MIN_PROFIT", "0.0") or 0.0)
    except Exception:
        min_profit = 0.0

    try:
        min_trades = float(os.environ.get("FOREX_BOT_PROP_KEEP_MIN_TRADES", "20" if elite else "1") or (20.0 if elite else 1.0))
    except Exception:
        min_trades = 20.0 if elite else 1.0
    min_trades = float(max(0.0, min_trades))

    try:
        min_keep = int(os.environ.get("FOREX_BOT_PROP_KEEP_MIN_COUNT", "100") or 100)
    except Exception:
        min_keep = 100
    min_keep = max(0, min_keep)

    try:
        portfolio_cap = int(
            os.environ.get(
                "FOREX_BOT_PROP_KEEP_CAP",
                getattr(settings.models, "prop_search_portfolio_size", 3000),
            )
            or 3000
        )
    except Exception:
        portfolio_cap = 3000
    if portfolio_cap < 0:
        portfolio_cap = 0
    if portfolio_cap > 0 and min_keep > portfolio_cap:
        min_keep = portfolio_cap
    return max_dd, min_profit, min_trades, min_keep, portfolio_cap


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _elite_filter_enabled() -> bool:
    runtime = str(os.environ.get("FOREX_BOT_RUNTIME_PROFILE", "") or "").strip().lower()
    default = runtime.startswith("rust")
    return _env_bool("FOREX_BOT_PROP_ELITE_FILTER", default)


def _strategy_is_anomalous(gene: TALibStrategyGene) -> bool:
    if not _env_bool("FOREX_BOT_PROP_ANOMALY_GUARD", True):
        return False

    try:
        profit = float(getattr(gene, "net_profit", 0.0) or 0.0)
    except Exception:
        profit = 0.0
    try:
        dd = float(getattr(gene, "max_dd_pct", 0.0) or 0.0)
    except Exception:
        dd = 0.0
    try:
        trades = float(getattr(gene, "trades", 0.0) or 0.0)
    except Exception:
        trades = 0.0
    try:
        win_rate = float(getattr(gene, "win_rate", 0.0) or 0.0)
    except Exception:
        win_rate = 0.0
    try:
        profit_factor = float(getattr(gene, "profit_factor", 0.0) or 0.0)
    except Exception:
        profit_factor = 0.0

    ppt = (profit / trades) if trades > 0 else 0.0

    min_trades = _env_float("FOREX_BOT_PROP_ANOMALY_MIN_TRADES", 120.0)
    max_dd = _env_float("FOREX_BOT_PROP_ANOMALY_MAX_DD", 0.0025)
    min_win_rate = _env_float("FOREX_BOT_PROP_ANOMALY_MIN_WIN_RATE", 0.92)
    min_profit_factor = _env_float("FOREX_BOT_PROP_ANOMALY_MIN_PF", 12.0)
    min_profit = _env_float("FOREX_BOT_PROP_ANOMALY_MIN_PROFIT", 200_000.0)
    max_profit_per_trade = _env_float("FOREX_BOT_PROP_ANOMALY_MAX_PROFIT_PER_TRADE", 2_000.0)
    ultra_min_trades = _env_float("FOREX_BOT_PROP_ANOMALY_ULTRA_MIN_TRADES", 50.0)
    ultra_max_dd = _env_float("FOREX_BOT_PROP_ANOMALY_ULTRA_MAX_DD", 0.001)
    ultra_min_profit = _env_float("FOREX_BOT_PROP_ANOMALY_ULTRA_MIN_PROFIT", 150_000.0)
    ultra_min_ppt = _env_float("FOREX_BOT_PROP_ANOMALY_ULTRA_MIN_PPT", 1_000.0)
    low_dd_min_trades = _env_float("FOREX_BOT_PROP_ANOMALY_LOW_DD_MIN_TRADES", 80.0)
    low_dd_max_dd = _env_float("FOREX_BOT_PROP_ANOMALY_LOW_DD_MAX_DD", 0.001)
    low_dd_min_profit = _env_float("FOREX_BOT_PROP_ANOMALY_LOW_DD_MIN_PROFIT", 50_000.0)

    suspicious_combo = (
        trades >= min_trades
        and dd <= max_dd
        and win_rate >= min_win_rate
        and profit_factor >= min_profit_factor
        and profit >= min_profit
    )
    suspicious_ppt = (
        trades >= max(40.0, min_trades * 0.5)
        and dd <= max(0.01, max_dd * 2.0)
        and ppt >= max_profit_per_trade
    )
    suspicious_ultra = (
        trades >= ultra_min_trades
        and dd <= ultra_max_dd
        and profit >= ultra_min_profit
        and ppt >= ultra_min_ppt
    )
    suspicious_low_dd = (
        trades >= low_dd_min_trades
        and dd <= low_dd_max_dd
        and profit >= low_dd_min_profit
    )
    return bool(suspicious_combo or suspicious_ppt or suspicious_ultra or suspicious_low_dd)


def _strategy_passes_filter(
    gene: TALibStrategyGene,
    *,
    max_dd: float,
    min_profit: float,
    min_trades: float,
    history_months: float | None = None,
    initial_balance: float | None = None,
) -> bool:
    elite = _elite_filter_enabled()
    min_history_months = _env_float("FOREX_BOT_PROP_MIN_HISTORY_MONTHS", 6.0 if elite else 0.0)
    if min_history_months > 0.0:
        hm = float(history_months) if history_months is not None else _env_float("FOREX_BOT_PROP_HISTORY_MONTHS", 0.0)
        if hm <= 0.0 or hm < min_history_months:
            return False

    profit_metric = str(os.environ.get("FOREX_BOT_PROP_KEEP_PROFIT_METRIC", "fitness") or "fitness").strip().lower()
    if profit_metric in {"net", "net_profit", "pnl"}:
        try:
            profit = float(getattr(gene, "net_profit", 0.0) or 0.0)
        except Exception:
            profit = 0.0
    else:
        try:
            profit = float(getattr(gene, "fitness", 0.0) or 0.0)
        except Exception:
            profit = 0.0
    if profit <= min_profit:
        return False

    try:
        dd = float(getattr(gene, "max_dd_pct", 0.0) or 0.0)
    except Exception:
        dd = 0.0
    if dd > max_dd:
        return False

    try:
        trades = float(getattr(gene, "trades", 0.0) or 0.0)
    except Exception:
        trades = 0.0
    if trades < min_trades:
        return False

    min_sharpe = _env_float("FOREX_BOT_PROP_KEEP_MIN_SHARPE", 1.2 if elite else 0.0)
    if min_sharpe > 0.0:
        try:
            sharpe = float(getattr(gene, "sharpe_ratio", 0.0) or 0.0)
        except Exception:
            sharpe = 0.0
        if sharpe < min_sharpe:
            return False

    min_win_rate = _env_float("FOREX_BOT_PROP_KEEP_MIN_WIN_RATE", 0.52 if elite else 0.0)
    if min_win_rate > 0.0:
        try:
            win_rate = float(getattr(gene, "win_rate", 0.0) or 0.0)
        except Exception:
            win_rate = 0.0
        if win_rate < min_win_rate:
            return False

    min_profit_factor = _env_float("FOREX_BOT_PROP_KEEP_MIN_PROFIT_FACTOR", 1.30 if elite else 0.0)
    if min_profit_factor > 0.0:
        try:
            profit_factor = float(getattr(gene, "profit_factor", 0.0) or 0.0)
        except Exception:
            profit_factor = 0.0
        if profit_factor < min_profit_factor:
            return False

    min_tpm = _env_float("FOREX_BOT_PROP_KEEP_MIN_TRADES_PER_MONTH", 0.0)
    if min_tpm > 0.0:
        hm = float(history_months) if history_months is not None else _env_float("FOREX_BOT_PROP_HISTORY_MONTHS", 0.0)
        if hm > 0.0:
            tpm = trades / hm
            if tpm < min_tpm:
                return False

    min_monthly_pct = _env_float("FOREX_BOT_PROP_KEEP_MIN_MONTHLY_PROFIT_PCT", 0.01 if elite else 0.0)
    if min_monthly_pct > 0.0:
        hm = float(history_months) if history_months is not None else _env_float("FOREX_BOT_PROP_HISTORY_MONTHS", 0.0)
        if hm > 0.0:
            try:
                net_profit = float(getattr(gene, "net_profit", profit) or profit)
            except Exception:
                net_profit = float(profit)
            bal = float(initial_balance) if initial_balance is not None else _env_float("FOREX_BOT_PROP_INITIAL_BALANCE", 100000.0)
            bal = max(1e-9, bal)
            monthly_profit_pct = net_profit / (bal * hm)
            if monthly_profit_pct < min_monthly_pct:
                return False

    if _strategy_is_anomalous(gene):
        return False
    return True


def _dedupe_ranked(genes: list[TALibStrategyGene]) -> list[TALibStrategyGene]:
    out: list[TALibStrategyGene] = []
    seen: set[str] = set()
    for gene in sorted(
        genes,
        key=lambda g: (
            float(getattr(g, "fitness", 0.0) or 0.0),
            float(getattr(g, "sharpe_ratio", 0.0) or 0.0),
            float(getattr(g, "win_rate", 0.0) or 0.0),
        ),
        reverse=True,
    ):
        key = _gene_key(gene)
        if key in seen:
            continue
        seen.add(key)
        out.append(gene)
    return out


def _gene_key(gene: TALibStrategyGene) -> str:
    sid = str(getattr(gene, "strategy_id", "") or "").strip()
    if sid:
        return f"id:{sid}"
    return (
        f"sig:{tuple(gene.indicators)}|{gene.combination_method}|"
        f"{float(gene.long_threshold):.6f}|{float(gene.short_threshold):.6f}"
    )


def _select_ranked(
    candidates: list[TALibStrategyGene],
    *,
    filtered: list[TALibStrategyGene],
    min_keep: int,
    cap: int,
) -> tuple[list[TALibStrategyGene], int, int]:
    ranked_all = _dedupe_ranked(candidates)
    ranked_filtered = _dedupe_ranked(filtered) if filtered else []
    selected = list(ranked_filtered)
    if min_keep > 0 and len(selected) < min_keep:
        seen = {_gene_key(g) for g in selected}
        for gene in ranked_all:
            key = _gene_key(gene)
            if key in seen:
                continue
            selected.append(gene)
            seen.add(key)
            if len(selected) >= min_keep:
                break
    if not selected:
        selected = ranked_all
    if cap > 0:
        selected = selected[:cap]
    return selected, len(ranked_filtered), len(ranked_all)


def run_evo_search(
    df: Any,
    settings: Any,
    population: int,
    generations: int,
    checkpoint: str,
    max_hours: float,
    actual_balance: float,
    max_workers: int | None = None,
) -> None:
    # API compatibility: callers may pass worker hints even though this search currently
    # runs synchronously in-process.
    _ = max_workers
    if _frame_empty(df):
        return
    train_years = _train_years_cfg(settings)
    if train_years > 0.0:
        orig_rows = int(len(df))
        df = _trim_to_recent_years(df, train_years)
        if int(len(df)) != orig_rows:
            logger.info(
                "Prop search train-year window applied: years=%.2f rows=%s->%s",
                float(train_years),
                orig_rows,
                int(len(df)),
            )
    search_df, holdout_df = _split_discovery_holdout(df, settings)
    if not _frame_empty(holdout_df):
        logger.info(
            "Prop search holdout enabled: search_rows=%s holdout_rows=%s",
            _frame_len(search_df),
            _frame_len(holdout_df),
        )
    max_dd, min_profit, min_trades, min_keep, portfolio_cap = _strategy_keep_limits(settings)
    symbol = str(_frame_attr(search_df, "symbol", "") or "")
    timeframe = str(_frame_attr(search_df, "timeframe", _frame_attr(search_df, "tf", "")) or "")
    history_days, history_months = _history_span_days_months(search_df)
    holdout_frac, _holdout_min_rows, holdout_min_sharpe, holdout_min_win, holdout_min_pf, holdout_min_trades, holdout_required, holdout_years, min_truth = _holdout_cfg(settings)
    holdout_from_raw = str(os.environ.get("FOREX_BOT_PROP_HOLDOUT_FROM", "") or "").strip()
    backend_ok = (_RUST_SEARCH or _RUST_GPU_SEARCH) and _fb is not None
    if not backend_ok:
        logger.error("Rust prop-search backend unavailable; skipping prop search.")
        return
    if (_RUST_SEARCH or _RUST_GPU_SEARCH) and _fb is not None:
        try:
            ts = None
            idx = _frame_index(search_df)
            if _is_datetime_index(idx):
                ts = _datetime_index_to_unix_ms(idx)
            close = _frame_column_numpy(search_df, "close", dtype=np.float64)
            high = _frame_column_numpy(search_df, "high", dtype=np.float64)
            low = _frame_column_numpy(search_df, "low", dtype=np.float64)
            open_ = (
                _frame_column_numpy(search_df, "open", dtype=np.float64)
                if _frame_has_column(search_df, "open")
                else close
            )
            volume = (
                _frame_column_numpy(search_df, "volume", dtype=np.float64)
                if _frame_has_column(search_df, "volume")
                else None
            )
            pip_size, pip_val = _df_pip_metrics(search_df, close=close)

            max_indicators = 0
            env_max = os.environ.get("FOREX_BOT_PROP_SEARCH_MAX_INDICATORS")
            if env_max:
                try:
                    max_indicators = int(env_max)
                except Exception:
                    max_indicators = 0
            if max_indicators <= 0:
                try:
                    max_indicators = int(
                        getattr(settings.models, "prop_search_max_indicators", 0) or 0
                    )
                except Exception:
                    max_indicators = 0
            if max_indicators <= 0:
                max_indicators = len(ALL_INDICATORS) or 12

            include_raw = str(os.environ.get("FOREX_BOT_PROP_INCLUDE_RAW_FEATURES", "0") or "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            prev_pip_env = {
                "FOREX_BOT_PROP_PIP_VALUE": os.environ.get("FOREX_BOT_PROP_PIP_VALUE"),
                "FOREX_BOT_PROP_PIP_VALUE_PER_LOT": os.environ.get("FOREX_BOT_PROP_PIP_VALUE_PER_LOT"),
            }
            os.environ["FOREX_BOT_PROP_PIP_VALUE"] = f"{float(pip_size):.12g}"
            os.environ["FOREX_BOT_PROP_PIP_VALUE_PER_LOT"] = f"{float(pip_val):.12g}"
            try:
                use_evogp = bool(_RUST_GPU_SEARCH and _evogp_requested(settings))
                if use_evogp:
                    default_evogp_pop = max(int(population or 0), 4096)
                    default_evogp_gens = max(int(generations or 0), 80)
                    try:
                        default_evogp_pop = int(
                            getattr(settings.models, "evogp_population", default_evogp_pop) or default_evogp_pop
                        )
                    except Exception:
                        pass
                    try:
                        default_evogp_gens = int(
                            getattr(settings.models, "evogp_generations", default_evogp_gens) or default_evogp_gens
                        )
                    except Exception:
                        pass
                    try:
                        gpu_population = int(
                            os.environ.get(
                                "FOREX_BOT_EVOGP_POPULATION",
                                str(default_evogp_pop),
                            )
                            or default_evogp_pop
                        )
                    except Exception:
                        gpu_population = default_evogp_pop
                    try:
                        gpu_generations = int(
                            os.environ.get(
                                "FOREX_BOT_EVOGP_GENERATIONS",
                                str(default_evogp_gens),
                            )
                            or default_evogp_gens
                        )
                    except Exception:
                        gpu_generations = default_evogp_gens

                    elite_fraction = _env_float("FOREX_BOT_EVOGP_ELITE_FRACTION", 0.05)
                    sigma = _env_float("FOREX_BOT_EVOGP_SIGMA", 0.5)
                    crossover = _env_float("FOREX_BOT_EVOGP_CROSSOVER_RATE", 0.35)
                    threshold_scale = _env_float("FOREX_BOT_EVOGP_THRESHOLD_SCALE", 0.10)
                    threshold_margin = _env_float("FOREX_BOT_EVOGP_THRESHOLD_MARGIN", 0.02)
                    threshold_clip = _env_float("FOREX_BOT_EVOGP_THRESHOLD_CLIP", 0.30)
                    window_bars = max(
                        256,
                        int(
                            os.environ.get(
                                "FOREX_BOT_EVOGP_WINDOW_BARS",
                                os.environ.get("FOREX_BOT_PROP_SEARCH_WINDOW_BARS", "190080"),
                            )
                            or 190080
                        ),
                    )
                    segments = max(1, int(os.environ.get("FOREX_BOT_EVOGP_SEGMENTS", "4") or 4))
                    chunk_size = max(
                        128,
                        int(
                            os.environ.get(
                                "FOREX_BOT_EVOGP_CHUNK_SIZE",
                                os.environ.get("FOREX_BOT_GPU_CHUNK_SIZE", "8192"),
                            )
                            or 8192
                        ),
                    )
                    devices = _parse_gpu_devices(
                        os.environ.get("FOREX_BOT_EVOGP_DEVICES")
                        or os.environ.get("FOREX_BOT_GPU_DEVICES")
                    )
                    try:
                        result = _fb.search_evolve_gpu_ohlcv(
                            open_,
                            high,
                            low,
                            close,
                            ts,
                            volume,
                            int(max(16, gpu_population)),
                            int(max(1, gpu_generations)),
                            include_raw,
                            float(np.clip(elite_fraction, 0.01, 0.50)),
                            float(max(0.01, sigma)),
                            float(np.clip(crossover, 0.0, 1.0)),
                            float(max(0.001, threshold_scale)),
                            float(max(0.0, threshold_margin)),
                            float(max(0.01, threshold_clip)),
                            int(window_bars),
                            int(segments),
                            float(_env_float("FOREX_BOT_EVOGP_MIN_TRADES_PER_DAY", 1.0)),
                            float(_env_float("FOREX_BOT_EVOGP_TRADE_PENALTY", 25.0)),
                            float(_env_float("FOREX_BOT_EVOGP_DD_LIMIT", 0.04)),
                            float(_env_float("FOREX_BOT_EVOGP_DD_PENALTY", 200.0)),
                            float(_env_float("FOREX_BOT_EVOGP_ROBUST_WEIGHT", 0.2)),
                            float(_env_float("FOREX_BOT_EVOGP_POS_WINDOW_FRACTION", 0.5)),
                            float(_env_float("FOREX_BOT_EVOGP_POS_PENALTY", 15.0)),
                            int(chunk_size),
                            devices if devices else None,
                        )
                        result["search_mode"] = "evogp_gpu"
                        result["threshold_scale_used"] = float(max(0.001, threshold_scale))
                        result["threshold_margin_used"] = float(max(0.0, threshold_margin))
                        result["threshold_clip_used"] = float(max(0.01, threshold_clip))
                    except Exception as evogp_exc:
                        if _RUST_SEARCH:
                            logger.warning(
                                "EvoGP GPU search failed (%s). Falling back to Rust GA for this run.",
                                evogp_exc,
                            )
                            result = _fb.search_evolve_ohlcv(
                                open_,
                                high,
                                low,
                                close,
                                ts,
                                volume,
                                int(population or 0),
                                int(generations or 0),
                                int(max_indicators),
                                include_raw,
                            )
                            result["search_mode"] = "rust_ga_fallback"
                        else:
                            raise
                elif _RUST_SEARCH:
                    result = _fb.search_evolve_ohlcv(
                        open_,
                        high,
                        low,
                        close,
                        ts,
                        volume,
                        int(population or 0),
                        int(generations or 0),
                        int(max_indicators),
                        include_raw,
                    )
                else:
                    raise RuntimeError("Rust GA binding unavailable for CPU evolve path")
            finally:
                for key, old in prev_pip_env.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old
            feature_names = list(result.get("feature_names") or [])
            search_mode = str(result.get("search_mode", "rust_ga") or "rust_ga")
            genes_raw = list(result.get("genes") or [])
            metrics_raw = list(result.get("metrics") or [])
            available = {str(x).upper() for x in ALL_INDICATORS}
            if not available and feature_names:
                available = {str(x).upper() for x in feature_names}
            best: list[TALibStrategyGene] = []
            if search_mode == "evogp_gpu":
                genomes = list(result.get("genomes") or [])
                fitness_raw = list(result.get("fitness") or [])
                if genomes:
                    ranked_idx = sorted(
                        range(len(genomes)),
                        key=lambda i: float(fitness_raw[i]) if i < len(fitness_raw) else float("-inf"),
                        reverse=True,
                    )
                    try:
                        default_eval_cap = max(512, min(6000, portfolio_cap * 4))
                        try:
                            default_eval_cap = int(
                                getattr(settings.models, "evogp_eval_candidates", default_eval_cap) or default_eval_cap
                            )
                        except Exception:
                            pass
                        eval_cap = int(
                            os.environ.get(
                                "FOREX_BOT_EVOGP_EVAL_CANDIDATES",
                                str(default_eval_cap),
                            )
                            or default_eval_cap
                        )
                    except Exception:
                        eval_cap = default_eval_cap
                    eval_cap = max(64, eval_cap)
                    take_idx = ranked_idx[: min(eval_cap, len(ranked_idx))]
                    thr_scale = float(result.get("threshold_scale_used", _env_float("FOREX_BOT_EVOGP_THRESHOLD_SCALE", 0.10)) or 0.10)
                    thr_margin = float(result.get("threshold_margin_used", _env_float("FOREX_BOT_EVOGP_THRESHOLD_MARGIN", 0.02)) or 0.02)
                    thr_clip = float(result.get("threshold_clip_used", _env_float("FOREX_BOT_EVOGP_THRESHOLD_CLIP", 0.30)) or 0.30)
                    for rank, i in enumerate(take_idx):
                        fit = float(fitness_raw[i]) if i < len(fitness_raw) else 0.0
                        gene = _convert_gpu_genome(
                            genome=genomes[i],
                            fitness=fit,
                            feature_names=feature_names,
                            available=available,
                            max_indicators=max_indicators,
                            threshold_scale=thr_scale,
                            threshold_margin=thr_margin,
                            threshold_clip=thr_clip,
                            strategy_id=f"evogp_{rank}",
                        )
                        if gene is not None:
                            best.append(gene)
            else:
                for idx, g in enumerate(genes_raw):
                    if not isinstance(g, dict):
                        continue
                    metric = metrics_raw[idx] if idx < len(metrics_raw) else None
                    gene = _convert_rust_gene(g, feature_names, available, metric=metric)
                    if gene:
                        best.append(gene)
            if not best:
                raise RuntimeError(f"{search_mode} produced no usable genes")
            best = _dedupe_ranked(best)

            filtered = [
                g
                for g in best
                if _strategy_passes_filter(
                    g,
                    max_dd=max_dd,
                    min_profit=min_profit,
                    min_trades=min_trades,
                    history_months=history_months,
                    initial_balance=actual_balance,
                )
            ]
            selected, strict_kept, ranked_total = _select_ranked(
                best,
                filtered=filtered,
                min_keep=min_keep,
                cap=portfolio_cap,
            )
            selected = _apply_holdout_validation(
                selected=selected,
                holdout_df=holdout_df,
                settings=settings,
                max_dd=max_dd,
                min_profit=min_profit,
                min_trades=min_trades,
                initial_balance=actual_balance,
                search_history_months=history_months,
            )
            _attach_trade_journals(
                selected=selected,
                search_df=search_df,
                holdout_df=holdout_df,
                settings=settings,
            )
            journal = _journal_summary(selected)

            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
                "rows": _frame_len(df),
                "search_rows": _frame_len(search_df),
                "holdout_rows": _frame_len(holdout_df) if holdout_df is not None else 0,
                "history_days": float(history_days),
                "history_months": float(history_months),
                "holdout_fraction": float(holdout_frac),
                "holdout_years": float(holdout_years),
                "holdout_from": holdout_from_raw,
                "holdout_required": bool(holdout_required),
                "holdout_min_sharpe": float(holdout_min_sharpe),
                "holdout_min_win_rate": float(holdout_min_win),
                "holdout_min_profit_factor": float(holdout_min_pf),
                "holdout_min_trades": int(holdout_min_trades),
                "min_truth_probability": float(min_truth),
                "initial_balance": float(actual_balance),
                "journal_summary": journal,
                "best_genes": [_gene_to_dict(g) for g in selected],
            }
            out_path = Path(checkpoint)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            cache_dir = Path("cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            out = cache_dir / "talib_knowledge.json"
            if symbol:
                safe = "".join(c for c in symbol if c.isalnum() or c in ("-", "_"))
                out = cache_dir / f"talib_knowledge_{safe}.json"
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info(
                "Prop search (%s): kept %s/%s genes (strict=%s, min_keep=%s) for %s %s "
                "(profit>%.3f, max_dd<=%.3f, trades>=%.0f, holdout_years=%.2f, min_truth=%.2f). Wrote %s",
                search_mode,
                len(selected),
                ranked_total,
                strict_kept,
                min_keep,
                symbol or "?",
                timeframe or "?",
                min_profit,
                max_dd,
                min_trades,
                holdout_years,
                min_truth,
                out,
            )
            return
        except Exception as exc:
            logger.error("Rust prop search failed; skipping prop search: %s", exc, exc_info=True)
            return

__all__ = ["run_evo_search"]


