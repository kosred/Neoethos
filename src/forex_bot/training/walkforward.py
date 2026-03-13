from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

try:
    import forex_bindings as _fb  # type: ignore
except Exception:
    _fb = None  # type: ignore

logger = logging.getLogger(__name__)


def _is_dataframe_like(values: Any) -> bool:
    return bool(hasattr(values, "columns") and hasattr(values, "index"))


class _NumpyFrame:
    """Minimal frame-like container used for slicing non-dataframe-module datasets."""

    def __init__(self, data: dict[str, Any], index: Any, attrs: dict[str, Any] | None = None) -> None:
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.index = np.asarray(index).reshape(-1)
        self.columns = list(self._data.keys())
        self.attrs = dict(attrs or {})

    @property
    def empty(self) -> bool:
        return int(len(self.index)) <= 0

    def __len__(self) -> int:
        return int(len(self.index))

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[str(key)]


def _frame_columns(values: Any) -> list[str]:
    cols = getattr(values, "columns", None)
    if cols is None:
        return []
    try:
        return [str(c) for c in list(cols)]
    except Exception:
        return []


def _frame_resolve_column(values: Any, name: str) -> str | None:
    target = str(name).strip().lower()
    for col in _frame_columns(values):
        if str(col).strip().lower() == target:
            return col
    return None


def _slice_frame_rows(values: Any, start: int, end: int) -> Any:
    cols = _frame_columns(values)
    s = max(0, int(start))
    e = max(s, int(end))
    if not cols:
        arr = np.asarray(values)
        return arr[s:e]
    data: dict[str, np.ndarray] = {}
    for col in cols:
        try:
            arr = np.asarray(values[col]).reshape(-1)  # type: ignore[index]
            data[str(col)] = arr[s:e]
        except Exception:
            continue
    idx = getattr(values, "index", None)
    if idx is None:
        out_idx = np.arange(max(0, e - s), dtype=np.int64)
    else:
        idx_arr = np.asarray(idx).reshape(-1)
        out_idx = idx_arr[s:e] if idx_arr.size >= e else np.arange(max(0, e - s), dtype=np.int64)
    attrs = getattr(values, "attrs", None)
    return _NumpyFrame(data, out_idx, attrs=(dict(attrs) if isinstance(attrs, dict) else None))


def _slice_rows(values: Any, start: int, end: int) -> Any:
    if _is_dataframe_like(values):
        s = max(0, int(start))
        e = max(s, int(end))
        idx = np.arange(s, e, dtype=np.int64)
        try:
            return values.take(idx)
        except Exception:
            try:
                base_idx = np.asarray(getattr(values, "index")).reshape(-1)
                return values.loc[base_idx[idx]]
            except Exception:
                pass
    if hasattr(values, "columns") and hasattr(values, "__getitem__"):
        return _slice_frame_rows(values, start, end)
    arr = np.asarray(values)
    return arr[start:end]


def _to_numpy_1d(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        arr = values.to_numpy(copy=False)
    else:
        arr = np.asarray(values)
    arr = np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)
    return arr.reshape(-1)


def _extract_close(values: Any) -> np.ndarray:
    try:
        col = _frame_resolve_column(values, "close")
        if col is not None:
            close = values[col]  # type: ignore[index]
            return _to_numpy_1d(close, dtype=np.float64)
    except Exception:
        pass
    arr = np.asarray(values)
    if arr.ndim == 0:
        return arr.reshape(1).astype(np.float64, copy=False)
    if arr.ndim == 1:
        return arr.astype(np.float64, copy=False)
    return np.asarray(arr[:, 0], dtype=np.float64).reshape(-1)


def _extract_index_ns(values: Any, n_rows: int) -> np.ndarray:
    idx = getattr(values, "index", None)
    if idx is None:
        return np.arange(n_rows, dtype=np.int64)
    try:
        if hasattr(idx, "asi8"):
            arr_ns = np.asarray(idx.asi8, dtype=np.int64).reshape(-1)
            if arr_ns.size == n_rows:
                return arr_ns
    except Exception:
        pass
    try:
        arr = np.asarray(idx).reshape(-1)
    except Exception:
        return np.arange(n_rows, dtype=np.int64)
    if arr.size != n_rows:
        return np.arange(n_rows, dtype=np.int64)
    try:
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
        if arr.dtype.kind in {"i", "u"}:
            return arr.astype(np.int64, copy=False)
        if arr.dtype.kind == "f":
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
    except Exception:
        pass
    out = np.zeros(n_rows, dtype=np.int64)
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


def _extract_day_keys(values: Any, n_rows: int) -> np.ndarray:
    idx_ns = _extract_index_ns(values, n_rows)
    if idx_ns.size != n_rows:
        return np.zeros(n_rows, dtype=np.int64)
    is_timestamp_like = idx_ns.size > 0 and int(np.max(np.abs(idx_ns))) > 10**14
    if is_timestamp_like and _fb is not None and hasattr(_fb, "derive_time_index_arrays"):
        try:
            _unix_ms, _month_idx, day_idx = _fb.derive_time_index_arrays(np.asarray(idx_ns, dtype=np.int64))
        except Exception:
            day_idx = None
        if day_idx is not None:
            arr = np.asarray(day_idx, dtype=np.int64).reshape(-1)
            if arr.size == n_rows:
                return arr
    if is_timestamp_like:
        return (np.asarray(idx_ns, dtype=np.int64) // (24 * 60 * 60 * 1_000_000_000)).astype(np.int64, copy=False)
    return np.zeros(n_rows, dtype=np.int64)


def embargoed_walkforward_backtest(
    df: Any,
    signals: Any,
    train_ratio: float = 0.7,
    n_splits: int = 5,
    embargo_minutes: int = 120,
    timeframe_minutes: int = 5,
    max_daily_loss_pct: float = 0.05,
    max_daily_profit_pct: float = 0.50,
    min_trading_days: int = 3,
    max_trades_per_day: int = 15,
    use_gpu: bool | None = None,
) -> dict[str, Any]:
    """
    Walk-forward backtest with embargo and dynamic Meta-Controller simulation.
    Mimics live execution behavior including risk throttling.
    """
    n = len(df)
    if n == 0 or signals is None or len(signals) != n:
        return {"walk_forward_splits": 0}

    window = max(1, n // n_splits)
    # HPC FIX: Robust Embargo (Clear indicators and labels)
    # Most TA-Lib indicators use up to 200 bars. Labels look forward 100.
    auto_embargo = int(os.environ.get("FOREX_BOT_WALKFORWARD_EMBARGO", "300") or 300)
    embargo_bars = max(embargo_minutes // max(1, timeframe_minutes), auto_embargo)
    results = []

    for i in range(n_splits):
        start = i * window
        end = min(n, (i + 1) * window)
        if end - start < 80:
            break

        train_end = start + int(window * train_ratio)
        test_start = train_end + embargo_bars
        if test_start >= end or (train_end - start) < 40 or (end - test_start) < 40:
            continue

        df_test = _slice_rows(df, test_start, end)
        sig = _slice_rows(signals, test_start, end)

        # GPU toggle auto
        if use_gpu is None:
            use_gpu = CUPY_AVAILABLE and bool(int(os.environ.get("GPU_BACKTEST", "1")))

        close = _extract_close(df_test)
        future = np.roll(close, -1)
        ret = (future - close) / close
        if len(ret) > 0:
            ret[-1] = 0.0

        if use_gpu and CUPY_AVAILABLE:
            try:
                c_close = cp.asarray(close, dtype=cp.float32)
                c_future = cp.roll(c_close, -1)
                c_ret = (c_future - c_close) / c_close
                c_ret[-1] = 0.0
                c_ret = cp.nan_to_num(c_ret, nan=0.0, posinf=0.0, neginf=0.0)
                c_sig = cp.asarray(_to_numpy_1d(sig, dtype=np.int8), dtype=cp.int8)
                day_keys = _extract_day_keys(df_test, len(close))
                days = cp.asarray(day_keys, dtype=cp.int64)

                equity = cp.float32(1.0)
                peak = cp.float32(1.0)
                max_dd = cp.float32(0.0)
                max_consec_losses = 0
                consec_losses = 0
                daily_stats = {}
                day_trades = {}
                daily_dd_vals = []
                rolling = cp.zeros(20, dtype=cp.float32)
                rolling_len = 0

                last_day = days[0]
                day_pnl = cp.float32(0.0)
                max_daily_loss = cp.float32(0.0)

                pnl = []

                for i in range(len(c_close) - 1):
                    day = days[i]
                    if day != last_day:
                        daily_dd_vals.append(float(day_pnl))
                        day_pnl = cp.float32(0.0)
                        last_day = day

                    # crude day start equity storage
                    ds = daily_stats.setdefault(
                        int(day), {"equity_start": float(equity), "equity_end": float(equity), "trades": 0}
                    )
                    day_start_eq = ds["equity_start"]
                    current_daily_dd = (day_start_eq - float(equity)) / day_start_eq if day_start_eq > 0 else 0.0
                    current_daily_dd = max(0.0, current_daily_dd)
                    # risk mult proxy
                    risk_mult = 1.0
                    allow_trade = True
                    if current_daily_dd >= max_daily_loss_pct or ds["trades"] >= max_trades_per_day:
                        allow_trade = False

                    trade_pnl = 0.0
                    sigv = int(c_sig[i])
                    bar_ret = float(c_ret[i])
                    if sigv != 0 and allow_trade:
                        trade_ret = sigv * bar_ret
                        base_risk = 0.01
                        realized = base_risk * risk_mult * (1.0 if trade_ret > 0 else -1.0)
                        realized -= 0.0005 * risk_mult
                        trade_pnl = realized
                        ds["trades"] += 1
                        day_trades[int(day)] = ds["trades"]
                        if trade_ret > 0:
                            consec_losses = 0
                            if rolling_len < 20:
                                rolling[rolling_len] = 1.0
                                rolling_len += 1
                            else:
                                rolling[:-1] = rolling[1:]
                                rolling[-1] = 1.0
                        else:
                            consec_losses += 1
                            max_consec_losses = max(max_consec_losses, consec_losses)
                            if rolling_len < 20:
                                rolling[rolling_len] = 0.0
                                rolling_len += 1
                            else:
                                rolling[:-1] = rolling[1:]
                                rolling[-1] = 0.0
                    pnl.append(trade_pnl)
                    equity *= 1.0 + trade_pnl
                    peak = max(peak, equity)
                    if peak > 0:
                        max_dd = max(max_dd, (peak - equity) / peak)
                    ds["equity_end"] = float(equity)
                    day_pnl += trade_pnl
                    max_daily_loss = min(max_daily_loss, day_pnl)

                # finalize day stats
                if day_pnl != 0:
                    daily_dd_vals.append(float(day_pnl))
                pnl_arr = np.array(pnl[:-1]) if len(pnl) > 1 else np.array(pnl)
                trades = int(np.count_nonzero(sig))

                daily_loss_breach = any(v <= -max_daily_loss_pct for v in daily_dd_vals)
                consistency_violation = any(v >= max_daily_profit_pct for v in daily_dd_vals)
                trade_limit_violation = any(v["trades"] > max_trades_per_day for v in daily_stats.values())
                days_with_trades = sum(1 for d in daily_stats.values() if d["trades"] > 0)
                min_days_ok = days_with_trades >= min_trading_days
                daily_min_dd = float(min(daily_dd_vals)) if daily_dd_vals else 0.0
                max_daily_loss_val = float(max_daily_loss)
                score = {
                    "split": i + 1,
                    "trades": trades,
                    "pnl": float(pnl_arr.sum()),
                    "win_rate": float((pnl_arr > 0).mean()) if len(pnl_arr) > 0 else 0.0,
                    "max_dd": float(max_dd),
                    "max_consec_losses": int(max_consec_losses),
                    "daily_min_dd": daily_min_dd,
                    "max_daily_loss": max_daily_loss_val,
                    "daily_loss_breach": bool(daily_loss_breach),
                    "consistency_violation": bool(consistency_violation),
                    "trade_limit_violation": bool(trade_limit_violation),
                    "min_trading_days_ok": bool(min_days_ok),
                    "daily_returns": daily_dd_vals,
                }
                if equity > 0:
                    score["max_daily_dd_pct"] = float(max_daily_loss_val / float(equity))
                    score["prop_compliant"] = bool(max_daily_loss_val > -(0.05 * float(equity)))
                results.append(score)
                continue
            except Exception as exc:
                use_gpu = False
                logger.warning(f"GPU walkforward backtest failed, fallback to CPU: {exc}")

        # CPU path
        pnl: list[float] = []
        equity = 1.0
        peak = equity
        max_dd = 0.0
        max_consec_losses = 0
        consec_losses = 0
        daily_stats: dict[object, dict[str, float | int]] = {}

        day_keys = _extract_day_keys(df_test, len(close))
        sig_arr = _to_numpy_1d(sig, dtype=np.int8)
        for idx in range(max(0, len(close) - 1)):
            day_key = int(day_keys[idx]) if idx < len(day_keys) else 0
            stats = daily_stats.get(day_key)
            if stats is None:
                stats = {"equity_start": equity, "equity_end": equity, "trades": 0}
                daily_stats[day_key] = stats

            # HPC FIX: Unified HPC Backtest Math (CPU/GPU Parity)
            sig_val = int(sig_arr[idx]) if idx < len(sig_arr) else 0
            bar_ret = float(ret[idx])
            
            pnl_i = 0.0
            if sig_val != 0:
                # 1. Base Profit
                trade_ret = sig_val * bar_ret
                # 2. Risk Sizing (Match GPU Path)
                base_risk = 0.01
                pnl_i = base_risk * (1.0 if trade_ret > 0 else -1.0)
                # 3. Cost (Spread/Comm)
                pnl_i -= 0.0005 
                
            pnl.append(pnl_i)
            equity *= 1.0 + pnl_i
            stats["equity_end"] = equity
            if sig_val != 0:
                stats["trades"] = int(stats.get("trades", 0)) + 1

            if equity > peak:
                peak = equity
            if peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak)

            if pnl_i < 0:
                consec_losses += 1
                max_consec_losses = max(max_consec_losses, consec_losses)
            elif pnl_i > 0:
                consec_losses = 0

        pnl_arr = np.array(pnl, dtype=float)
        trades = int(np.count_nonzero(sig_arr[:-1]))

        daily_loss_breach = False
        consistency_violation = False
        trade_limit_violation = False
        daily_returns: list[float] = []
        daily_dd_list: list[float] = []
        for stats in daily_stats.values():
            eq_start = float(stats["equity_start"])
            eq_end = float(stats["equity_end"])
            daily_ret = (eq_end - eq_start) / max(eq_start, 1e-9)
            daily_returns.append(daily_ret)
            daily_dd_list.append(min(0.0, daily_ret))
            if daily_ret <= -max_daily_loss_pct:
                daily_loss_breach = True
            if daily_ret >= max_daily_profit_pct:
                consistency_violation = True
            if int(stats["trades"]) > max_trades_per_day:
                trade_limit_violation = True

        days_with_trades = sum(1 for stats in daily_stats.values() if int(stats["trades"]) > 0)
        daily_min_dd = float(min(daily_dd_list)) if daily_dd_list else 0.0
        min_days_ok = days_with_trades >= min_trading_days
        max_daily_loss = float(min(daily_returns)) if daily_returns else 0.0

        score = {
            "split": i + 1,
            "trades": trades,
            "pnl": float(pnl_arr.sum()),
            "win_rate": float((pnl_arr > 0).mean()) if len(pnl_arr) > 0 else 0.0,
            "max_dd": float(max_dd),
            "max_consec_losses": int(max_consec_losses),
            "daily_min_dd": daily_min_dd,
            "max_daily_loss": max_daily_loss,
            "daily_loss_breach": bool(daily_loss_breach),
            "consistency_violation": bool(consistency_violation),
            "trade_limit_violation": bool(trade_limit_violation),
            "min_trading_days_ok": bool(min_days_ok),
            "daily_returns": daily_returns,
        }
        if equity > 0:
            score["max_daily_dd_pct"] = float(max_daily_loss / equity)
            score["prop_compliant"] = bool(max_daily_loss > -(0.05 * equity))
        results.append(score)

    if not results:
        return {"walk_forward_splits": 0}

    agg = {
        "walk_forward_splits": len(results),
        "avg_pnl": float(np.mean([r["pnl"] for r in results])),
        "avg_win_rate": float(np.mean([r["win_rate"] for r in results])),
        "avg_max_dd": float(np.mean([r["max_dd"] for r in results])),
        "avg_max_consec_losses": float(np.mean([r["max_consec_losses"] for r in results])),
        "avg_daily_min_dd": float(np.mean([r["daily_min_dd"] for r in results])),
        "avg_max_daily_loss": float(np.mean([r["max_daily_loss"] for r in results])),
        "any_daily_loss_breach": any(r["daily_loss_breach"] for r in results),
        "any_consistency_violation": any(r["consistency_violation"] for r in results),
        "any_trade_limit_violation": any(r["trade_limit_violation"] for r in results),
        "all_min_trading_days_ok": all(r["min_trading_days_ok"] for r in results),
        "splits": results,
    }
    return agg

