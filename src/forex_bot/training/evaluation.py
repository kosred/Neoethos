from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

from ..strategy.fast_backtest import fast_evaluate_strategy, infer_pip_metrics

try:
    import forex_bindings as _fb  # type: ignore
except Exception:
    _fb = None  # type: ignore


def _as_1d(values: Any, *, dtype: np.dtype) -> np.ndarray:
    if values is None:
        return np.zeros(0, dtype=dtype)
    try:
        if hasattr(values, "to_numpy"):
            arr = values.to_numpy(dtype=dtype, copy=False)
        else:
            arr = np.asarray(values, dtype=dtype)
    except Exception:
        arr = np.asarray(values)
        arr = arr.astype(dtype, copy=False)
    arr = np.asarray(arr, dtype=dtype).reshape(-1)
    return arr


def _extract_column(frame: Any, name: str, *, dtype: np.dtype) -> np.ndarray:
    col = None
    with np.errstate(all="ignore"):
        try:
            col = frame[name]
        except Exception:
            if isinstance(frame, dict):
                col = frame.get(name)
    if col is None:
        return np.zeros(0, dtype=dtype)
    return _as_1d(col, dtype=dtype)


def _extract_index(frame: Any, n: int) -> np.ndarray:
    idx = None
    try:
        idx = getattr(frame, "index", None)
    except Exception:
        idx = None
    if idx is None and isinstance(frame, dict):
        idx = frame.get("index")
    if idx is None:
        return np.arange(n, dtype=np.int64)
    try:
        if hasattr(idx, "asi8"):
            arr = np.asarray(idx.asi8, dtype=np.int64).reshape(-1)
            if arr.size == n:
                return arr
    except Exception:
        pass
    try:
        arr = np.asarray(idx).reshape(-1)
    except Exception:
        return np.arange(n, dtype=np.int64)
    if arr.size != n:
        return np.arange(n, dtype=np.int64)
    try:
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
        if arr.dtype.kind in {"i", "u"}:
            return arr.astype(np.int64, copy=False)
        if arr.dtype.kind == "f":
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
    except Exception:
        pass
    out = np.zeros(n, dtype=np.int64)
    for i, value in enumerate(arr.tolist()):
        try:
            ns = getattr(value, "value", None)
            if ns is not None:
                out[i] = int(ns)
            else:
                out[i] = int(np.datetime64(value, "ns").astype(np.int64))
        except Exception:
            out[i] = i
    return out


def _month_day_indices(index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(index).reshape(-1)
    n = arr.size
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    def _rust_from_ns(ns_values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if _fb is None or not hasattr(_fb, "derive_time_index_arrays"):
            return None
        try:
            _unix_ms, month_idx, day_idx = _fb.derive_time_index_arrays(
                np.asarray(ns_values, dtype=np.int64).reshape(-1)
            )
        except Exception:
            return None
        month_arr = np.asarray(month_idx, dtype=np.int64).reshape(-1)
        day_arr = np.asarray(day_idx, dtype=np.int64).reshape(-1)
        if month_arr.size != n or day_arr.size != n:
            return None
        return month_arr, day_arr

    with np.errstate(all="ignore"):
        if np.issubdtype(arr.dtype, np.datetime64):
            dt = arr.astype("datetime64[ns]")
            rust = _rust_from_ns(dt.astype(np.int64, copy=False))
            if rust is not None:
                return rust
            month_idx = dt.astype("datetime64[M]").astype(np.int64)
            day_idx = dt.astype("datetime64[D]").astype(np.int64)
            return month_idx, day_idx

    if arr.dtype.kind in {"i", "u"}:
        ints = arr.astype(np.int64, copy=False)
        if ints.size > 0 and int(np.max(np.abs(ints))) > 10**12:
            rust = _rust_from_ns(ints)
            if rust is not None:
                return rust
            with np.errstate(all="ignore"):
                dt = ints.astype("datetime64[ns]")
                month_idx = dt.astype("datetime64[M]").astype(np.int64)
                day_idx = dt.astype("datetime64[D]").astype(np.int64)
                return month_idx, day_idx
        day_idx = ints
        month_idx = ints // 31
        return month_idx, day_idx

    if arr.dtype.kind == "f":
        ints = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
        day_idx = ints
        month_idx = ints // 31
        return month_idx, day_idx

    day_idx = np.arange(n, dtype=np.int64)
    month_idx = day_idx // 31
    return month_idx, day_idx


def _extract_symbol(frame: Any) -> str:
    symbol = None
    with np.errstate(all="ignore"):
        try:
            attrs = getattr(frame, "attrs", None)
            if isinstance(attrs, dict):
                symbol = attrs.get("symbol")
        except Exception:
            symbol = None
    if symbol is None and isinstance(frame, dict):
        symbol = frame.get("symbol")
    sym = str(symbol or "EURUSD").strip().upper()
    return sym if sym else "EURUSD"


def pad_probs(probs: np.ndarray, classes: list[int] | None = None) -> np.ndarray:
    """
    HPC UNIFIED PROTOCOL: Force output to [Neutral, Buy, Sell].
    Standard indices: 0=Neutral, 1=Buy, 2=Sell.
    """
    if probs is None or len(probs) == 0:
        return np.zeros((0, 3), dtype=float)

    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]
    out = np.zeros((n, 3), dtype=float)

    # 1. Handle Model-Specific Mapping (If classes provided)
    if classes is not None and len(classes) == arr.shape[1]:
        for col, cls_val in enumerate(classes):
            # Map from Project Labels {-1, 0, 1} to Unified Indices {0, 1, 2}
            if cls_val == 0:
                out[:, 0] = arr[:, col]  # Neutral -> 0
            elif cls_val == 1:
                out[:, 1] = arr[:, col]  # Buy -> 1
            elif cls_val == -1 or cls_val == 2:
                out[:, 2] = arr[:, col]  # Sell -> 2
        return out

    # 2. Automated Heuristics (If no class map)
    if arr.shape[1] == 3:
        return arr  # Assume [Neutral, Buy, Sell]
    if arr.shape[1] == 2:
        out[:, 0] = arr[:, 0]  # Neutral
        out[:, 1] = arr[:, 1]  # Buy
        return out
    out[:, 0] = 1.0 - arr[:, 0]  # Neutral
    out[:, 1] = arr[:, 0]  # Buy
    return out


def _rust_probs_to_signals(probs: np.ndarray) -> np.ndarray | None:
    if _fb is None or not hasattr(_fb, "probs_to_signals"):
        return None
    try:
        out = _fb.probs_to_signals(np.asarray(probs, dtype=np.float64))
    except Exception:
        return None
    return np.asarray(out, dtype=np.int8).reshape(-1)


def probs_to_signals(probs: np.ndarray, classes: list[int] | None = None) -> np.ndarray:
    """
    Map probability matrix [neutral, buy, sell] to discrete signals {-1,0,1}.
    Chooses highest prob; ties fall back to neutral.
    """
    p = pad_probs(probs, classes=classes)
    if len(p) == 0:
        return np.zeros(0, dtype=int)
    rust = _rust_probs_to_signals(p)
    if rust is not None:
        return rust.astype(int, copy=False)
    sig_idx = p.argmax(axis=1)
    signals = np.zeros(len(p), dtype=int)
    signals[sig_idx == 1] = 1
    signals[sig_idx == 2] = -1
    return signals


def quick_backtest(frame: Any, signals: Any) -> dict[str, Any]:
    """
    Lightweight backtest: assumes input has 'close', uses next-bar direction as outcome.
    Returns accuracy and simple PnL proxy.
    """
    close = _extract_column(frame, "close", dtype=np.float64)
    sig_arr = _as_1d(signals, dtype=np.int8)
    n = int(min(close.size, sig_arr.size))
    if n <= 1:
        return {}
    close = close[:n]
    sig_arr = sig_arr[:n]
    if _fb is not None and hasattr(_fb, "quick_backtest_metrics"):
        try:
            acc, pnl_score, win_rate, _trades = _fb.quick_backtest_metrics(
                np.asarray(close, dtype=np.float64),
                np.asarray(sig_arr, dtype=np.int8),
            )
            return {
                "accuracy": float(acc),
                "pnl_score": float(pnl_score),
                "win_rate": float(win_rate),
            }
        except Exception:
            pass
    future = np.roll(close, -1)
    ret = future - close

    pnl = np.where(
        sig_arr == 0,
        0.0,
        np.where(sig_arr == 1, np.where(ret > 0, 1.0, -1.0), np.where(ret < 0, 1.0, -1.0)),
    )
    pnl = pnl[:-1]
    acc = float((sig_arr[:-1] == np.sign(ret[:-1])).mean()) if len(ret) > 1 else 0.0
    return {
        "accuracy": acc,
        "pnl_score": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()) if len(pnl) > 0 else 0.0,
    }


def prop_backtest(
    frame: Any,
    signals: Any,
    max_daily_dd_pct: float = 0.05,
    daily_dd_warn_pct: float = 0.03,
    max_trades_per_day: int = 10,
    use_gpu: bool | None = None,
) -> dict[str, Any]:
    """
    HPC Unified Backtest: Uses the master Rust/NumPy engine for identical results.
    """
    close = _extract_column(frame, "close", dtype=np.float64)
    high = _extract_column(frame, "high", dtype=np.float64)
    low = _extract_column(frame, "low", dtype=np.float64)
    sig_arr = _as_1d(signals, dtype=np.int8)

    n = int(min(close.size, high.size, low.size, sig_arr.size))
    if n <= 1:
        return {}
    close = close[:n]
    high = high[:n]
    low = low[:n]
    sig_arr = sig_arr[:n]

    idx = _extract_index(frame, n)
    month_idx, day_idx = _month_day_indices(idx)
    symbol = _extract_symbol(frame)
    pip_size, pip_val_lot = infer_pip_metrics(symbol)

    # Keep defaults for stable cross-model scoring.
    arr = fast_evaluate_strategy(
        close_prices=close,
        high_prices=high,
        low_prices=low,
        signals=sig_arr,
        month_indices=month_idx,
        day_indices=day_idx,
        sl_pips=30.0,
        tp_pips=60.0,
        pip_value=pip_size,
        pip_value_per_lot=pip_val_lot,
        spread_pips=1.5,
        commission_per_trade=7.0,
    )

    keys = [
        "net_profit",
        "sharpe",
        "sortino",
        "max_dd_pct",
        "win_rate",
        "profit_factor",
        "expectancy",
        "sqn",
        "trades",
        "consistency",
        "daily_dd",
    ]
    out = {k: float(v) for k, v in zip(keys, arr.tolist())}
    # Compatibility aliases used by older evaluation aggregators.
    out["pnl_score"] = float(out.get("net_profit", 0.0))
    out["max_dd"] = float(out.get("max_dd_pct", 0.0))
    return out
