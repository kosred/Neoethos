from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CUSTOM_SMC_INDICATORS: tuple[str, ...] = (
    "SMC_OB",
    "SMC_FVG",
    "SMC_LIQ",
    "SMC_TREND",
    "SMC_PREMIUM",
    "SMC_INDUCEMENT",
    "SMC_BOS",
    "SMC_CHOCH",
    "SMC_EQH",
    "SMC_EQL",
    "SMC_DISPLACEMENT",
)

try:
    import talib
    from talib import abstract

    TALIB_AVAILABLE = True
    ALL_INDICATORS = sorted({*map(str.upper, talib.get_functions()), *CUSTOM_SMC_INDICATORS})
except Exception:
    talib = None  # type: ignore
    abstract = None  # type: ignore
    TALIB_AVAILABLE = False
    ALL_INDICATORS: list[str] = list(CUSTOM_SMC_INDICATORS)

TALIB_INDICATORS: dict[str, list[str]] = {
    "momentum": ["RSI", "ADX", "MACD"],
    "overlap": ["SMA", "EMA"],
    "volatility": ["ATR", "NATR"],
    "smc": list(CUSTOM_SMC_INDICATORS),
}

try:
    import forex_bindings as _fb  # type: ignore

    _RUST_TALIB_MIXER = hasattr(_fb, "talib_bulk_signals_ohlcv")
except Exception:
    _fb = None
    _RUST_TALIB_MIXER = False


_STRICT_RUST_WARNED = False


def _frame_empty(df: Any) -> bool:
    if df is None:
        return True
    try:
        return bool(df.empty)
    except Exception:
        pass
    try:
        return int(len(df)) <= 0
    except Exception:
        return True


def _frame_len(df: Any) -> int:
    try:
        return int(len(df))
    except Exception:
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


def _frame_has_column(df: Any, name: str) -> bool:
    target = _normalize_indicator_name(name)
    for col in _frame_columns(df):
        if _normalize_indicator_name(col) == target:
            return True
    return False


def _frame_resolve_column(df: Any, name: str) -> str | None:
    target = _normalize_indicator_name(name)
    for col in _frame_columns(df):
        if _normalize_indicator_name(col) == target:
            return col
    return None


def _frame_column_numpy(df: Any, name: str, *, dtype: Any = np.float64) -> np.ndarray:
    col = _frame_resolve_column(df, name)
    if col is None:
        raise KeyError(name)
    values = df[col]  # type: ignore[index]
    if hasattr(values, "to_numpy"):
        try:
            arr = np.asarray(values.to_numpy(dtype=dtype, copy=False), dtype=dtype)
        except TypeError:
            arr = np.asarray(values.to_numpy(dtype=dtype), dtype=dtype)
        except Exception:
            arr = np.asarray(values, dtype=dtype)
    else:
        arr = np.asarray(values, dtype=dtype)
    return arr.reshape(-1)


def _strict_rust_mode_enabled() -> bool:
    return True


def _make_series(values: Any, index: Any):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if index is None:
        return arr
    try:
        n = int(len(index))
    except Exception:
        return arr
    if arr.size < n:
        out = np.zeros(n, dtype=np.float64)
        out[: arr.size] = arr
        return out
    if arr.size > n:
        return arr[:n]
    return arr


def _index_to_ns(values: Any, n_rows: int) -> np.ndarray | None:
    if values is None:
        return None
    try:
        if hasattr(values, "asi8"):
            arr_ns = np.asarray(values.asi8, dtype=np.int64).reshape(-1)
            if arr_ns.size != int(n_rows):
                return None
            return arr_ns
    except Exception:
        pass
    try:
        if hasattr(values, "view"):
            viewed = values.view("int64")
            if hasattr(viewed, "to_numpy"):
                arr_ns = np.asarray(viewed.to_numpy(dtype=np.int64, copy=False), dtype=np.int64).reshape(-1)
            else:
                arr_ns = np.asarray(viewed, dtype=np.int64).reshape(-1)
            if arr_ns.size != int(n_rows):
                return None
            return arr_ns
    except Exception:
        pass
    try:
        arr = np.asarray(values).reshape(-1)
    except Exception:
        return None
    if arr.size != int(n_rows):
        return None
    try:
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
        if arr.dtype.kind in {"i", "u"}:
            return arr.astype(np.int64, copy=False)
        if arr.dtype.kind == "f":
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
        return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
    except Exception:
        return None


def _rust_align_by_ns(
    src_ns: np.ndarray,
    src_vals: np.ndarray,
    tgt_ns: np.ndarray,
    *,
    fill_value: float,
    forward_fill: bool,
) -> np.ndarray | None:
    if _fb is None:
        return None
    fn_name = "align_ffill_values_by_ns" if forward_fill else "align_exact_values_by_ns"
    if not hasattr(_fb, fn_name):
        return None
    try:
        fn = getattr(_fb, fn_name)
        out = fn(
            np.asarray(src_ns, dtype=np.int64),
            np.asarray(src_vals, dtype=np.float64),
            np.asarray(tgt_ns, dtype=np.int64),
            float(fill_value),
        )
    except Exception:
        return None
    arr = np.asarray(out, dtype=np.float64).reshape(-1)
    if arr.size != int(np.asarray(tgt_ns).size):
        return None
    return arr


def _rust_sorted_index_order(index_like: Any) -> np.ndarray | None:
    if _fb is None or not hasattr(_fb, "sorted_index_order"):
        return None
    idx_ns = np.asarray(index_like, dtype=np.int64).reshape(-1)
    if idx_ns.size <= 0:
        return None
    try:
        out = _fb.sorted_index_order(idx_ns)
    except Exception:
        return None
    order = np.asarray(out, dtype=np.int64).reshape(-1)
    if order.size != idx_ns.size:
        return None
    return order


def _sorted_time_order(index_like: Any) -> np.ndarray | None:
    idx_ns = np.asarray(index_like, dtype=np.int64).reshape(-1)
    if idx_ns.size <= 1:
        return None
    if not bool(np.any(idx_ns[1:] < idx_ns[:-1])):
        return None
    order = _rust_sorted_index_order(idx_ns)
    if order is not None:
        return order
    return np.argsort(idx_ns, kind="mergesort")


def signal_to_numpy(
    signal: Any,
    *,
    index: Any | None = None,
    dtype: Any = np.float64,
    fill_value: float = 0.0,
    forward_fill: bool = True,
) -> np.ndarray:
    if signal is None:
        n = int(len(index)) if index is not None else 0
        return np.full(n, fill_value, dtype=dtype)

    obj = signal
    arr: np.ndarray | None = None
    if hasattr(obj, "to_numpy"):
        try:
            arr = np.asarray(obj.to_numpy(dtype=dtype, copy=False), dtype=dtype)
        except TypeError:
            arr = np.asarray(obj.to_numpy(dtype=dtype), dtype=dtype)
        except Exception:
            arr = np.asarray(obj, dtype=dtype)
    else:
        arr = np.asarray(obj, dtype=dtype)

    arr = arr.reshape(-1)
    if index is not None:
        n = int(len(index))
        src_ns = _index_to_ns(getattr(obj, "index", None), arr.size)
        tgt_ns = _index_to_ns(index, n)
        if src_ns is not None and tgt_ns is not None:
            m = min(src_ns.size, arr.size)
            src_ns = src_ns[:m]
            src_vals = np.asarray(arr[:m], dtype=np.float64)

            valid_mask = np.isfinite(src_vals)
            if not np.all(valid_mask):
                src_ns = src_ns[valid_mask]
                src_vals = src_vals[valid_mask]

            aligned = _rust_align_by_ns(
                src_ns,
                src_vals,
                tgt_ns,
                fill_value=fill_value,
                forward_fill=forward_fill,
            )
            if aligned is None:
                order = _sorted_time_order(src_ns)
                if order is not None:
                    src_ns = src_ns[order]
                    src_vals = src_vals[order]
                if forward_fill:
                    pos = np.searchsorted(src_ns, tgt_ns, side="right") - 1
                    aligned = np.full(tgt_ns.shape, float(fill_value), dtype=np.float64)
                    valid = pos >= 0
                    if np.any(valid):
                        aligned[valid] = src_vals[np.clip(pos[valid], 0, src_vals.size - 1)]
                else:
                    pos = np.searchsorted(src_ns, tgt_ns, side="left")
                    aligned = np.full(tgt_ns.shape, float(fill_value), dtype=np.float64)
                    valid = pos < src_ns.size
                    if np.any(valid):
                        matched = np.zeros(tgt_ns.shape[0], dtype=bool)
                        vp = pos[valid]
                        matched[valid] = src_ns[vp] == tgt_ns[valid]
                        take = valid & matched
                        if np.any(take):
                            aligned[take] = src_vals[pos[take]]
            arr = aligned.astype(dtype, copy=False)
        else:
            if arr.size < n:
                padded = np.full(n, fill_value, dtype=dtype)
                padded[: arr.size] = arr
                arr = padded
            elif arr.size > n:
                arr = arr[:n]

    if forward_fill and arr.size > 0 and np.issubdtype(np.asarray(arr).dtype, np.floating):
        ff = np.asarray(arr, dtype=np.float64).reshape(-1).copy()
        for i in range(1, ff.size):
            if not np.isfinite(ff[i]):
                ff[i] = ff[i - 1]
        arr = ff.astype(dtype, copy=False)

    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.copy()
        arr[~np.isfinite(arr)] = fill_value
    return arr


def signal_shift_prev(
    signal: Any,
    *,
    index: Any | None = None,
    dtype: Any = np.float64,
    fill_value: float = 0.0,
) -> np.ndarray:
    arr = signal_to_numpy(signal, index=index, dtype=dtype, fill_value=fill_value, forward_fill=False)
    if arr.size <= 0:
        return arr
    out = np.empty_like(arr)
    out[0] = fill_value
    out[1:] = arr[:-1]
    return out


def _normalize_indicator_name(name: str) -> str:
    return str(name or "").strip().upper()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _causal_tanh_zscore(values: np.ndarray, *, min_periods: int) -> np.ndarray:
    """
    Strictly causal normalization:
    - stats are built from historical values only (shifted by one bar)
    - no future values influence current signal
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr.astype(np.float64, copy=False)
    out = np.zeros(arr.shape[0], dtype=np.float64)
    needed = max(2, int(min_periods))
    count = 0
    mean = 0.0
    m2 = 0.0

    for i, val in enumerate(arr):
        if count >= needed:
            var = m2 / max(1, count)
            std = float(np.sqrt(var)) if var > 0.0 else 0.0
            z = (val - mean) / std if std > 1e-12 else (val - mean)
            if np.isfinite(z):
                out[i] = np.tanh(z)
        if not np.isfinite(val):
            continue
        count += 1
        delta = val - mean
        mean += delta / count
        delta2 = val - mean
        m2 += delta * delta2
    return out


def _coerce_ohlc_numpy(df: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def _col(name: str, default: np.ndarray) -> np.ndarray:
        try:
            out = _frame_column_numpy(df, name, dtype=np.float64)
            if out.shape[0] == default.shape[0]:
                return out
        except Exception:
            pass
        return default

    try:
        close = _frame_column_numpy(df, "close", dtype=np.float64)
    except Exception:
        n = _frame_len(df)
        close = np.zeros(n, dtype=np.float64)
    open_ = _col("open", close)
    high = _col("high", np.maximum(open_, close))
    low = _col("low", np.minimum(open_, close))
    return open_, high, low, close


def _compute_smc_indicator_numpy(df: Any, indicator: str) -> np.ndarray:
    name = _normalize_indicator_name(indicator)
    open_, high, low, close = _coerce_ohlc_numpy(df)
    n = int(close.shape[0])
    out = np.zeros(n, dtype=np.float64)
    if n <= 0:
        return out

    if name == "SMC_TREND":
        lookback = 12
        for i in range(1, n):
            ref = i - lookback if i >= lookback else i - 1
            d = close[i] - close[ref]
            out[i] = 1.0 if d > 0.0 else (-1.0 if d < 0.0 else 0.0)
        return out

    if name == "SMC_PREMIUM":
        rng = np.maximum(high - low, 1e-12)
        rel = (close - low) / rng
        out = np.where(rel <= 0.5, 1.0, -1.0).astype(np.float64, copy=False)
        return out

    if name == "SMC_FVG":
        for i in range(2, n):
            if low[i] > high[i - 2]:
                out[i] = 1.0
            elif high[i] < low[i - 2]:
                out[i] = -1.0
        return out

    if name == "SMC_LIQ":
        for i in range(3, n):
            prev_low = float(np.min(low[i - 3 : i]))
            prev_high = float(np.max(high[i - 3 : i]))
            if low[i] < prev_low and close[i] > prev_low:
                out[i] = 1.0
            elif high[i] > prev_high and close[i] < prev_high:
                out[i] = -1.0
        return out

    if name == "SMC_OB":
        for i in range(1, n):
            bull = close[i] > open_[i] and close[i - 1] < open_[i - 1] and close[i] >= high[i - 1]
            bear = close[i] < open_[i] and close[i - 1] > open_[i - 1] and close[i] <= low[i - 1]
            out[i] = 1.0 if bull else (-1.0 if bear else 0.0)
        return out

    if name == "SMC_INDUCEMENT":
        body = np.abs(close - open_)
        upper = high - np.maximum(open_, close)
        lower = np.minimum(open_, close) - low
        mask = (body > 1e-12) & ((upper / np.maximum(body, 1e-12) > 2.0) | (lower / np.maximum(body, 1e-12) > 2.0))
        out[mask] = 1.0
        return out

    if name == "SMC_BOS":
        lookback = 20
        for i in range(2, n):
            lb = max(0, i - lookback)
            prev_high = float(np.max(high[lb:i]))
            prev_low = float(np.min(low[lb:i]))
            if close[i] > prev_high:
                out[i] = 1.0
            elif close[i] < prev_low:
                out[i] = -1.0
        return out

    if name == "SMC_CHOCH":
        trend = _compute_smc_indicator_numpy(df, "SMC_TREND")
        bos = _compute_smc_indicator_numpy(df, "SMC_BOS")
        for i in range(1, n):
            if bos[i] > 0.0 and trend[i - 1] < 0.0:
                out[i] = 1.0
            elif bos[i] < 0.0 and trend[i - 1] > 0.0:
                out[i] = -1.0
        return out

    if name in {"SMC_EQH", "SMC_EQL"}:
        lookback = 20
        for i in range(1, n):
            lb = max(0, i - lookback)
            atr_proxy = float(np.mean(np.abs(high[lb : i + 1] - low[lb : i + 1])))
            tol = max(atr_proxy * 0.1, 1e-6)
            if name == "SMC_EQH":
                for j in range(lb, i):
                    if abs(high[i] - high[j]) <= tol:
                        out[i] = -1.0
                        break
            else:
                for j in range(lb, i):
                    if abs(low[i] - low[j]) <= tol:
                        out[i] = 1.0
                        break
        return out

    if name == "SMC_DISPLACEMENT":
        lookback = 20
        body = np.abs(close - open_)
        for i in range(lookback, n):
            avg_body = float(np.mean(body[i - lookback : i]))
            if avg_body <= 1e-12:
                continue
            if body[i] >= (1.8 * avg_body):
                out[i] = 1.0 if close[i] > open_[i] else (-1.0 if close[i] < open_[i] else 0.0)
        return out

    return out


def _parse_synergy_key(key: str) -> tuple[str, str] | None:
    if not key:
        return None
    parts = str(key).split("_")
    if len(parts) != 2:
        return None
    return _normalize_indicator_name(parts[0]), _normalize_indicator_name(parts[1])


@dataclass(slots=True)
class TALibStrategyGene:
    indicators: list[str]
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    combination_method: str = "weighted_vote"
    long_threshold: float = 0.66
    short_threshold: float = -0.66
    weights: dict[str, float] = field(default_factory=dict)
    preferred_regime: str = "any"
    strategy_id: str = ""
    fitness: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_dd_pct: float = 0.0
    trades: float = 0.0
    source_symbol: str = ""
    source_timeframe: str = ""
    use_ob: bool = False
    use_fvg: bool = False
    use_liq_sweep: bool = False
    mtf_confirmation: bool = False
    use_premium_discount: bool = False
    use_inducement: bool = False
    use_bos: bool = False
    use_choch: bool = False
    use_eqh: bool = False
    use_eql: bool = False
    use_displacement: bool = False
    tp_pips: float = 40.0
    sl_pips: float = 20.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    # Journal metrics (optional): in-sample and forward-holdout diagnostics.
    in_sample_net_profit: float = 0.0
    in_sample_sharpe_ratio: float = 0.0
    in_sample_win_rate: float = 0.0
    in_sample_profit_factor: float = 0.0
    in_sample_trades: float = 0.0
    in_sample_max_dd_pct: float = 0.0
    in_sample_months: float = 0.0
    holdout_net_profit: float = 0.0
    holdout_sharpe_ratio: float = 0.0
    holdout_win_rate: float = 0.0
    holdout_profit_factor: float = 0.0
    holdout_trades: float = 0.0
    holdout_max_dd_pct: float = 0.0
    holdout_months: float = 0.0
    holdout_trades_per_month: float = 0.0
    holdout_monthly_profit_pct: float = 0.0
    truth_probability: float = 0.0
    forward_test_passed: bool = False
    in_sample_journal: dict[str, Any] = field(default_factory=dict)
    holdout_journal: dict[str, Any] = field(default_factory=dict)


class TALibStrategyMixer:
    def __init__(self, *, device: str = "cpu", use_volume_features: bool = False) -> None:
        self.device = device
        self.use_volume_features = use_volume_features
        self.available_indicators = [_normalize_indicator_name(i) for i in ALL_INDICATORS]
        self.indicator_synergy_matrix: dict[tuple[str, str], float] = {}
        self.regime_performance: dict[str, dict[str, float]] = {}
        self._strict_rust = _strict_rust_mode_enabled()
        self._rust_signal_cache: dict[tuple[Any, ...], Any] = {}
        self._rust_signal_index: Any | None = None

    @staticmethod
    def _gene_key(gene: TALibStrategyGene) -> tuple[Any, ...]:
        indicators = tuple(_normalize_indicator_name(i) for i in (gene.indicators or []))
        weights = tuple(float((gene.weights or {}).get(ind, 1.0)) for ind in indicators)
        return (
            indicators,
            weights,
            float(gene.long_threshold),
            float(gene.short_threshold),
            str(gene.combination_method or "weighted_vote").lower(),
        )

    @staticmethod
    def _has_custom_params(gene: TALibStrategyGene) -> bool:
        params = gene.params or {}
        if not params:
            return False
        for value in params.values():
            if isinstance(value, dict) and len(value) == 0:
                continue
            if value:
                return True
        return False

    def _try_rust_bulk_signal_cache(self, df: Any, population: list[TALibStrategyGene]) -> None:
        self._rust_signal_cache = {}
        self._rust_signal_index = None
        if self._strict_rust and not _RUST_TALIB_MIXER:
            global _STRICT_RUST_WARNED
            if not _STRICT_RUST_WARNED:
                logger.warning(
                    "Strict Rust mode: talib_bulk_signals_ohlcv is unavailable; "
                    "signals will default to zero."
                )
                _STRICT_RUST_WARNED = True
            return
        bulk_default = True
        if not _env_bool("FOREX_BOT_TALIB_RUST_BULK_SIGNALS", bulk_default):
            return
        if not _RUST_TALIB_MIXER or _fb is None:
            return
        if _frame_empty(df) or not population:
            return
        required_cols = {"open", "high", "low", "close"}
        if not all(_frame_has_column(df, col) for col in required_cols):
            return

        eligible: list[tuple[tuple[Any, ...], TALibStrategyGene]] = []
        indicator_sets: list[list[str]] = []
        weight_sets: list[list[float]] = []
        long_thresholds: list[float] = []
        short_thresholds: list[float] = []
        for gene in population:
            if self._has_custom_params(gene):
                continue
            inds = [_normalize_indicator_name(i) for i in (gene.indicators or []) if _normalize_indicator_name(i)]
            if not inds:
                continue
            key = self._gene_key(gene)
            eligible.append((key, gene))
            indicator_sets.append(inds)
            weight_sets.append([float((gene.weights or {}).get(ind, 1.0)) for ind in inds])
            long_thresholds.append(float(gene.long_threshold))
            short_thresholds.append(float(gene.short_threshold))
        if not eligible:
            return

        open_arr = _frame_column_numpy(df, "open", dtype=np.float64)
        high_arr = _frame_column_numpy(df, "high", dtype=np.float64)
        low_arr = _frame_column_numpy(df, "low", dtype=np.float64)
        close_arr = _frame_column_numpy(df, "close", dtype=np.float64)
        volume_arr = None
        if self.use_volume_features and _frame_has_column(df, "volume"):
            volume_arr = _frame_column_numpy(df, "volume", dtype=np.float64)

        try:
            causal_min_bars = max(2, _env_int("FOREX_BOT_TALIB_CAUSAL_MIN_BARS", 30))
            try:
                raw = _fb.talib_bulk_signals_ohlcv(
                    open_arr,
                    high_arr,
                    low_arr,
                    close_arr,
                    indicator_sets=indicator_sets,
                    weight_sets=weight_sets,
                    long_thresholds=long_thresholds,
                    short_thresholds=short_thresholds,
                    volume=volume_arr,
                    include_raw=False,
                    causal_min_bars=causal_min_bars,
                )
            except TypeError:
                # Backward-compat for older bindings that do not expose causal_min_bars.
                raw = _fb.talib_bulk_signals_ohlcv(
                    open_arr,
                    high_arr,
                    low_arr,
                    close_arr,
                    indicator_sets=indicator_sets,
                    weight_sets=weight_sets,
                    long_thresholds=long_thresholds,
                    short_thresholds=short_thresholds,
                    volume=volume_arr,
                    include_raw=False,
                )
            signals = np.asarray(raw, dtype=np.float64)
        except Exception as exc:
            logger.debug("Rust TALib bulk signals failed; signal cache remains empty: %s", exc)
            return

        n_rows = _frame_len(df)
        n_genes = int(len(eligible))
        if signals.ndim != 2:
            logger.debug(
                "Rust TALib bulk signals shape mismatch (got=%s expected=(%s,%s)); signal cache remains empty.",
                signals.shape,
                n_rows,
                n_genes,
            )
            return
        if signals.shape[0] == n_genes and signals.shape[1] == n_rows:
            signals = signals.T
        if signals.shape[0] != n_rows or signals.shape[1] != n_genes:
            logger.debug(
                "Rust TALib bulk signals shape mismatch (got=%s expected=(%s,%s)); signal cache remains empty.",
                signals.shape,
                n_rows,
                n_genes,
            )
            return
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int8, copy=False)

        idx = _frame_index(df)
        for col_idx, (key, _gene) in enumerate(eligible):
            self._rust_signal_cache[key] = _make_series(
                signals[:, col_idx].astype(np.float64, copy=False),
                index=idx,
            )
        self._rust_signal_index = idx

    def _try_rust_signal_cache(self, df: Any, gene: TALibStrategyGene) -> None:
        key = self._gene_key(gene)
        if key in self._rust_signal_cache:
            return
        prev_cache = dict(self._rust_signal_cache)
        prev_index = self._rust_signal_index
        self._try_rust_bulk_signal_cache(df, [gene])
        cached = self._rust_signal_cache.get(key)
        self._rust_signal_cache = prev_cache
        self._rust_signal_index = prev_index
        if cached is not None:
            self._rust_signal_cache[key] = cached
            if self._rust_signal_index is None:
                self._rust_signal_index = _frame_index(df)

    def generate_random_strategy(self, *, max_indicators: int = 5) -> TALibStrategyGene:
        inds = [i for i in self.available_indicators if i]
        if not inds:
            return TALibStrategyGene(indicators=[])
        if max_indicators <= 0:
            max_indicators = len(inds)
        k = max(1, min(max_indicators, len(inds)))
        selected = random.sample(inds, k=k)
        weights = {i: float(random.uniform(0.5, 1.5)) for i in selected}
        params = {i: {} for i in selected}
        gene = TALibStrategyGene(
            indicators=selected,
            params=params,
            weights=weights,
            long_threshold=float(random.uniform(0.4, 1.0)),
            short_threshold=float(random.uniform(-1.0, -0.4)),
            strategy_id=f"gene_{random.randint(0, 1_000_000)}",
        )
        return gene

    def _compute_indicator(self, df: Any, indicator: str, params: dict[str, Any] | None):
        norm = _normalize_indicator_name(indicator)
        idx = _frame_index(df)
        if norm in CUSTOM_SMC_INDICATORS:
            return _make_series(_compute_smc_indicator_numpy(df, norm), index=idx)
        if abstract is None:
            raise RuntimeError("TA-Lib not available")
        func = abstract.Function(indicator)
        try:
            info = getattr(func, "info", {}) or {}
            defaults = info.get("parameters", {}) if isinstance(info, dict) else {}
        except Exception:
            defaults = {}
        merged = dict(defaults)
        if params:
            merged.update(params)
        output = func(df, **merged)
        if hasattr(output, "columns") and hasattr(output, "__getitem__"):
            out_cols = _frame_columns(output)
            if out_cols:
                try:
                    return output[out_cols[0]]
                except Exception:
                    pass
        if hasattr(output, "index") and hasattr(output, "to_numpy"):
            return output
        return _make_series(np.asarray(output), index=idx)

    def bulk_calculate_indicators(self, df: Any, population: list[TALibStrategyGene]) -> dict[str, Any]:
        cache: dict[str, Any] = {}
        self._rust_signal_cache = {}
        self._rust_signal_index = None
        if _frame_empty(df):
            return cache
        if not population:
            return cache
        self._try_rust_bulk_signal_cache(df, population)
        return cache

    def compute_signals(
        self,
        df: Any,
        gene: TALibStrategyGene,
        *,
        cache: dict[str, Any] | None = None,
    ):
        idx = _frame_index(df)
        n_rows = _frame_len(df)
        if df is None:
            return _make_series(np.zeros(0, dtype=float), index=None)
        if _frame_empty(df):
            return _make_series(np.zeros(0, dtype=float), index=idx)
        indicators = [_normalize_indicator_name(i) for i in gene.indicators]
        if not indicators:
            return _make_series(np.zeros(n_rows, dtype=float), index=idx)
        rust_key = self._gene_key(gene)
        if rust_key not in self._rust_signal_cache:
            self._try_rust_signal_cache(df, gene)
        if rust_key in self._rust_signal_cache:
            cached = self._rust_signal_cache[rust_key]
            cached_arr = signal_to_numpy(
                cached,
                index=idx,
                dtype=np.float64,
                fill_value=0.0,
                forward_fill=False,
            )
            return _make_series(cached_arr, index=idx)
        return _make_series(np.zeros(n_rows, dtype=float), index=idx)

    def load_knowledge(self, path: str | Path) -> None:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load TA-Lib knowledge: %s", exc)
            return
        matrix = payload.get("synergy_matrix", {}) if isinstance(payload, dict) else {}
        for key, value in dict(matrix).items():
            pair = _parse_synergy_key(key)
            if pair is None:
                continue
            try:
                self.indicator_synergy_matrix[pair] = float(value)
            except Exception:
                continue
        regime = payload.get("regime_performance", {}) if isinstance(payload, dict) else {}
        if isinstance(regime, dict):
            self.regime_performance = {k: dict(v) if isinstance(v, dict) else {} for k, v in regime.items()}

    def save_knowledge(self, path: str | Path) -> None:
        out = {
            "synergy_matrix": {f"{a}_{b}": v for (a, b), v in self.indicator_synergy_matrix.items()},
            "regime_performance": self.regime_performance,
        }
        Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")


__all__ = [
    "TALIB_AVAILABLE",
    "ALL_INDICATORS",
    "TALIB_INDICATORS",
    "TALibStrategyGene",
    "TALibStrategyMixer",
    "signal_shift_prev",
    "signal_to_numpy",
]

