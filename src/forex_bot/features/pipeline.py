from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..core.config import ALL_TIMEFRAMES, Settings
from ..domain.events import PreparedDataset

logger = logging.getLogger(__name__)
_RUST_FEATURES_BACKEND_OK: bool | None = None
_RUST_FEATURES_WARNED_UNAVAILABLE = False
_RUST_LABELS_BACKEND_OK: bool | None = None
_RUST_LABELS_WARNED_UNAVAILABLE = False


def _tabular_module(*, required: bool = True):
    _ = required
    return None


class _NumpyFrame:
    """Minimal frame-like metadata container for frame-native pipelines."""

    def __init__(
        self,
        data: dict[str, Any],
        *,
        index: Any | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        n_rows = 0
        if self._data:
            try:
                n_rows = int(next(iter(self._data.values())).shape[0])
            except Exception:
                n_rows = 0
        if index is None:
            self.index = np.arange(n_rows, dtype=np.int64)
        else:
            idx = np.asarray(index).reshape(-1)
            if idx.size == n_rows:
                self.index = idx
            elif idx.size <= 0:
                self.index = np.arange(n_rows, dtype=np.int64)
            elif idx.size > n_rows:
                self.index = idx[:n_rows]
            else:
                pad = np.full(n_rows - idx.size, idx[-1], dtype=idx.dtype)
                self.index = np.concatenate([idx, pad])
        self.columns = list(self._data.keys())
        self.attrs = dict(attrs or {})

    @property
    def empty(self) -> bool:
        return int(len(self.index)) <= 0

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[str(key)]

    def copy(self, deep: bool = False) -> "_NumpyFrame":
        _ = deep
        return _NumpyFrame(
            {k: np.asarray(v).copy() for k, v in self._data.items()},
            index=np.asarray(self.index).copy(),
            attrs=dict(self.attrs),
        )

    def to_numpy(self, dtype: Any | None = None, copy: bool = False) -> np.ndarray:
        if not self.columns:
            arr = np.zeros((len(self), 0), dtype=np.float32)
        else:
            arr = np.column_stack([np.asarray(self._data[col]).reshape(-1) for col in self.columns])
        if dtype is not None:
            arr = np.asarray(arr, dtype=dtype)
        return np.array(arr, copy=True) if copy else np.asarray(arr)


def _make_series(
    values: Any,
    *,
    index: Any | None = None,
    dtype: Any | None = None,
    template: Any | None = None,
) -> Any:
    if template is not None:
        ctor = getattr(template, "__class__", None)
        if ctor is not None:
            with contextlib.suppress(Exception):
                if dtype is None:
                    return ctor(values, index=index)
                return ctor(values, index=index, dtype=dtype)
    arr = np.asarray(values)
    if dtype is not None:
        with contextlib.suppress(Exception):
            arr = arr.astype(dtype, copy=False)
    return arr.reshape(-1)


def _is_datetime_index(value: Any) -> bool:
    return bool(
        value is not None
        and hasattr(value, "tz")
        and hasattr(value, "tz_localize")
        and hasattr(value, "tz_convert")
        and (hasattr(value, "year") or hasattr(value, "asi8"))
    )


def _rust_features_backend_available(*, force_log: bool = False) -> bool:
    global _RUST_FEATURES_BACKEND_OK, _RUST_FEATURES_WARNED_UNAVAILABLE
    if _RUST_FEATURES_BACKEND_OK is None:
        try:
            import forex_bindings  # type: ignore

            _RUST_FEATURES_BACKEND_OK = hasattr(forex_bindings, "load_symbol_features")
        except Exception:
            _RUST_FEATURES_BACKEND_OK = False
    if force_log and not _RUST_FEATURES_BACKEND_OK and not _RUST_FEATURES_WARNED_UNAVAILABLE:
        logger.warning(
            "Rust features backend requested but forex_bindings.load_symbol_features is unavailable."
        )
        _RUST_FEATURES_WARNED_UNAVAILABLE = True
    return bool(_RUST_FEATURES_BACKEND_OK)


def _disable_rust_features_backend() -> None:
    global _RUST_FEATURES_BACKEND_OK
    _RUST_FEATURES_BACKEND_OK = False


def _rust_labels_backend_available(*, force_log: bool = False) -> bool:
    global _RUST_LABELS_BACKEND_OK, _RUST_LABELS_WARNED_UNAVAILABLE
    if _RUST_LABELS_BACKEND_OK is None:
        try:
            import forex_bindings  # type: ignore

            _RUST_LABELS_BACKEND_OK = hasattr(forex_bindings, "triple_barrier_labels")
        except Exception:
            _RUST_LABELS_BACKEND_OK = False
    if force_log and not _RUST_LABELS_BACKEND_OK and not _RUST_LABELS_WARNED_UNAVAILABLE:
        logger.warning(
            "Rust labels backend requested but forex_bindings.triple_barrier_labels is unavailable."
        )
        _RUST_LABELS_WARNED_UNAVAILABLE = True
    return bool(_RUST_LABELS_BACKEND_OK)


def _disable_rust_labels_backend() -> None:
    global _RUST_LABELS_BACKEND_OK
    _RUST_LABELS_BACKEND_OK = False


def _ensure_datetime_index(df: Any) -> Any:
    if df is None or df.empty:
        return df
    out = df.copy()
    if not _is_datetime_index(out.index):
        ts_col = None
        for candidate in ("timestamp", "time", "datetime", "date"):
            if candidate in out.columns:
                ts_col = candidate
                break
        idx_np: np.ndarray | None = None
        if ts_col is not None:
            idx_np = _index_to_ns_like(out[ts_col])
        else:
            idx_np = _index_to_ns_like(out.index)
        if idx_np is not None:
            out.index = idx_np.astype("datetime64[ns]")
    else:
        idx_np = _index_to_ns_like(out.index)
        if idx_np is not None:
            out.index = idx_np.astype("datetime64[ns]")
    return out


def _index_to_ns_like(index: Any) -> np.ndarray | None:
    try:
        if hasattr(index, "asi8"):
            arr = np.asarray(index.asi8, dtype=np.int64).reshape(-1)
            return arr if arr.size > 0 else np.zeros(0, dtype=np.int64)
    except Exception:
        pass
    try:
        arr = np.asarray(index).reshape(-1)
    except Exception:
        return None
    if arr.size <= 0:
        return np.zeros(0, dtype=np.int64)
    with np.errstate(all="ignore"):
        try:
            if np.issubdtype(arr.dtype, np.datetime64):
                return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
        except Exception:
            pass
    if arr.dtype.kind in {"i", "u"}:
        vals = arr.astype(np.int64, copy=False)
        vmax = int(np.max(np.abs(vals))) if vals.size > 0 else 0
        if vmax > 10**14:
            return vals
        if vmax > 10**11:
            return vals * 1_000_000
        return vals * 1_000_000_000
    if arr.dtype.kind == "f":
        vals = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
        return vals * 1_000_000_000
    out = np.zeros(arr.size, dtype=np.int64)
    for i, v in enumerate(arr):
        try:
            ns = getattr(v, "value", None)
            if ns is not None:
                out[i] = int(ns)
                continue
            out[i] = int(np.datetime64(v, "ns").astype(np.int64))
        except Exception:
            out[i] = 0
    return out


def _rust_align_by_ts(
    source_ts: np.ndarray,
    values: object,
    target_ts: np.ndarray,
    *,
    default: float,
    forward_fill: bool,
) -> np.ndarray | None:
    try:
        import forex_bindings as _fb  # type: ignore
    except Exception:
        return None
    fn_name = "align_ffill_values_by_ns" if forward_fill else "align_exact_values_by_ns"
    if not hasattr(_fb, fn_name):
        return None
    try:
        fn = getattr(_fb, fn_name)
        out = fn(
            np.asarray(source_ts, dtype=np.int64),
            np.asarray(values, dtype=np.float64),
            np.asarray(target_ts, dtype=np.int64),
            float(default),
        )
    except Exception:
        return None
    arr = np.asarray(out, dtype=np.float64).reshape(-1)
    if arr.size != int(np.asarray(target_ts).size):
        return None
    return arr


def _rust_sorted_index_order(index_like: Any) -> np.ndarray | None:
    try:
        import forex_bindings as _fb  # type: ignore
    except Exception:
        return None
    if not hasattr(_fb, "sorted_index_order"):
        return None
    idx_ns = _index_to_ns_like(index_like)
    if idx_ns is None:
        return None
    try:
        out = _fb.sorted_index_order(np.asarray(idx_ns, dtype=np.int64))
    except Exception:
        return None
    order = np.asarray(out, dtype=np.int64).reshape(-1)
    if order.size != idx_ns.size:
        return None
    return order


def _rust_rank_scores_desc(scores: Any, *, absolute: bool = False) -> np.ndarray | None:
    try:
        import forex_bindings as _fb  # type: ignore
    except Exception:
        return None
    if not hasattr(_fb, "rank_scores_desc"):
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


def _sorted_time_order(index_like: Any) -> np.ndarray | None:
    idx_ns = _index_to_ns_like(index_like)
    if idx_ns is None or idx_ns.size <= 1:
        return None
    if not bool(np.any(idx_ns[1:] < idx_ns[:-1])):
        return None
    order = _rust_sorted_index_order(idx_ns)
    if order is not None:
        return order
    return np.argsort(idx_ns, kind="mergesort")


def _ema(series: Any, span: int) -> Any:
    return series.ewm(span=span, adjust=False).mean()


def _compute_rsi(series: Any, period: int = 14) -> Any:
    delta = series.diff()
    gain = delta.where(delta > 0.0, 0.0)
    loss = -delta.where(delta < 0.0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _compute_macd(series: Any) -> tuple[Any, Any, Any]:
    ema12 = _ema(series, 12)
    ema26 = _ema(series, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


def _compute_atr(high: Any, low: Any, close: Any, period: int = 14) -> Any:
    close_arr = np.asarray(close, dtype=np.float64).reshape(-1)
    n = int(close_arr.shape[0])
    idx = getattr(close, "index", None)
    if n <= 0:
        return _make_series(np.zeros(0, dtype=np.float64), index=idx, template=close)

    hi_arr = np.asarray(high, dtype=np.float64).reshape(-1)
    lo_arr = np.asarray(low, dtype=np.float64).reshape(-1)
    if hi_arr.size != n:
        if hi_arr.size <= 0:
            hi_arr = np.copy(close_arr)
        elif hi_arr.size > n:
            hi_arr = hi_arr[:n]
        else:
            hi_arr = np.concatenate([hi_arr, np.full(n - hi_arr.size, float(hi_arr[-1]), dtype=np.float64)])
    if lo_arr.size != n:
        if lo_arr.size <= 0:
            lo_arr = np.copy(close_arr)
        elif lo_arr.size > n:
            lo_arr = lo_arr[:n]
        else:
            lo_arr = np.concatenate([lo_arr, np.full(n - lo_arr.size, float(lo_arr[-1]), dtype=np.float64)])

    atr_np = _compute_atr_numpy(hi_arr, lo_arr, close_arr, period=period)
    return _make_series(atr_np, index=idx, template=close)


def _compute_atr_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    h = np.asarray(high, dtype=np.float64).reshape(-1)
    l = np.asarray(low, dtype=np.float64).reshape(-1)
    c = np.asarray(close, dtype=np.float64).reshape(-1)
    n = int(c.shape[0])
    if n <= 0:
        return np.zeros(0, dtype=np.float64)

    prev_close = np.empty_like(c)
    prev_close[0] = c[0]
    if n > 1:
        prev_close[1:] = c[:-1]
    tr = np.maximum.reduce(
        (
            np.abs(h - l),
            np.abs(h - prev_close),
            np.abs(l - prev_close),
        )
    )
    p = int(max(2, period))
    atr = np.full(n, np.nan, dtype=np.float64)
    if n >= p:
        kernel = np.ones(p, dtype=np.float64)
        vals = np.convolve(tr, kernel, mode="valid") / float(p)
        atr[p - 1 :] = vals
    valid = np.flatnonzero(np.isfinite(atr))
    if valid.size > 0 and valid[0] > 0:
        atr[: valid[0]] = atr[valid[0]]
    return np.nan_to_num(atr, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_adx_numba(high: Iterable[float], low: Iterable[float], close: Iterable[float], period: int = 14) -> np.ndarray:
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    close_arr = np.asarray(close, dtype=np.float64)
    n = close_arr.shape[0]
    adx = np.zeros(n, dtype=np.float64)
    if n <= period:
        return adx

    tr = np.zeros(n, dtype=np.float64)
    pdm = np.zeros(n, dtype=np.float64)
    mdm = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        up = high_arr[i] - high_arr[i - 1]
        down = low_arr[i - 1] - low_arr[i]
        pdm[i] = up if (up > down and up > 0.0) else 0.0
        mdm[i] = down if (down > up and down > 0.0) else 0.0
        tr[i] = max(
            high_arr[i] - low_arr[i],
            abs(high_arr[i] - close_arr[i - 1]),
            abs(low_arr[i] - close_arr[i - 1]),
        )

    atr = np.zeros(n, dtype=np.float64)
    pdm_sm = np.zeros(n, dtype=np.float64)
    mdm_sm = np.zeros(n, dtype=np.float64)

    atr[period] = tr[1 : period + 1].sum()
    pdm_sm[period] = pdm[1 : period + 1].sum()
    mdm_sm[period] = mdm[1 : period + 1].sum()

    for i in range(period + 1, n):
        atr[i] = atr[i - 1] - (atr[i - 1] / period) + tr[i]
        pdm_sm[i] = pdm_sm[i - 1] - (pdm_sm[i - 1] / period) + pdm[i]
        mdm_sm[i] = mdm_sm[i - 1] - (mdm_sm[i - 1] / period) + mdm[i]

    for i in range(period, n):
        if atr[i] <= 0.0:
            continue
        pdi = 100.0 * (pdm_sm[i] / atr[i])
        mdi = 100.0 * (mdm_sm[i] / atr[i])
        denom = pdi + mdi
        if denom <= 0.0:
            dx = 0.0
        else:
            dx = 100.0 * abs(pdi - mdi) / denom
        if i == period:
            adx[i] = dx
        else:
            adx[i] = (adx[i - 1] * (period - 1) + dx) / period
    return adx


@dataclass(slots=True)
class _LabelConfig:
    horizon: int
    min_dist: float
    use_triple_barrier: bool
    max_hold: int
    sl_pips: float | None
    tp_pips: float | None


class FeatureEngineer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def _tf_minutes(tf: str) -> int:
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
        }.get(str(tf or "").upper(), 10**9)

    def _resolved_timeframes(self, base_tf: str) -> list[str]:
        tfs: list[str] = [str(base_tf or "M1").upper()]
        if bool(getattr(self.settings.system, "multi_resolution_enabled", True)):
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
        tfs = sorted(set(tfs), key=self._tf_minutes)
        drop_lower = str(os.environ.get("FOREX_BOT_DROP_LOWER_TFS", "0") or "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if drop_lower:
            base_mins = self._tf_minutes(base_tf)
            tfs = [tf for tf in tfs if self._tf_minutes(tf) >= base_mins]
        if base_tf not in tfs:
            tfs.insert(0, base_tf)
        return tfs

    def _downcast_training_float32(self) -> bool:
        raw = os.environ.get("FOREX_BOT_DOWNCAST_FLOAT32")
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip().lower() in {"1", "true", "yes", "on"}
        return bool(getattr(self.settings.system, "downcast_training_float32", True))

    def _maybe_downcast_features(self, df: Any) -> Any:
        if df is None or df.empty or not self._downcast_training_float32():
            return df
        try:
            # Pandas 3+ deprecates `copy=` for astype in favor of CoW semantics.
            return df.astype(np.float32)
        except Exception:
            return df

    @staticmethod
    def _use_rust_backend() -> bool:
        raw = os.environ.get("FOREX_BOT_RUST_FEATURES")
        if raw is not None and str(raw).strip() != "":
            mode = str(raw).strip().lower()
            if mode in {"auto", "detect"}:
                return _rust_features_backend_available()
            enabled = mode in {"1", "true", "yes", "on", "rust"}
            return enabled and _rust_features_backend_available(force_log=True)
        mode = str(os.environ.get("FOREX_BOT_FEATURES_BACKEND", "auto")).strip().lower()
        if mode in {"rust", "rs", "1", "true", "yes", "on"}:
            return _rust_features_backend_available(force_log=True)
        if mode in {"python", "py", "0", "false", "no", "off"}:
            return False
        return _rust_features_backend_available()

    @staticmethod
    def _rust_only_enabled() -> bool:
        raw = str(os.environ.get("FOREX_BOT_RUST_ONLY", "") or "").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        backend = str(os.environ.get("FOREX_BOT_FEATURES_BACKEND", "") or "").strip().lower()
        if backend in {"rust_strict", "strict_rust", "rust_only", "rust-only"}:
            return True
        tree_backend = str(os.environ.get("FOREX_BOT_TREE_BACKEND", "") or "").strip().lower()
        if tree_backend in {"rust_strict", "strict_rust", "rust_only", "rust-only"}:
            return True
        runtime_profile = str(os.environ.get("FOREX_BOT_RUNTIME_PROFILE", "") or "").strip().lower()
        if runtime_profile.startswith("rust"):
            return True
        return False

    @staticmethod
    def _empty_numpy_dataset() -> PreparedDataset:
        x = np.zeros((0, 0), dtype=np.float32)
        y = np.zeros(0, dtype=np.int8)
        idx = np.zeros(0, dtype=np.int64)
        return PreparedDataset(X=x, y=y, index=idx, feature_names=[], metadata=None, labels=y)

    def _label_config(self) -> _LabelConfig:
        horizon = 1
        for key in ("FOREX_BOT_LABEL_HORIZON", "FOREX_BOT_LABEL_HORIZON_BARS"):
            raw = os.environ.get(key)
            if raw is None or str(raw).strip() == "":
                continue
            try:
                val = int(str(raw).strip())
            except Exception:
                continue
            if val > 0:
                horizon = val
                break
        try:
            min_dist = float(getattr(self.settings.risk, "meta_label_min_dist", 0.0) or 0.0)
        except Exception:
            min_dist = 0.0
        try:
            max_hold = int(getattr(self.settings.risk, "triple_barrier_max_bars", 0) or 0)
        except Exception:
            max_hold = 0
        if max_hold <= 0:
            try:
                max_hold = int(getattr(self.settings.risk, "meta_label_max_hold_bars", 0) or 0)
            except Exception:
                max_hold = 0
        max_hold = max(0, max_hold)

        raw_tb = os.environ.get("FOREX_BOT_LABEL_TRIPLE_BARRIER", "1")
        use_triple = str(raw_tb).strip().lower() in {"1", "true", "yes", "on"}
        use_triple = bool(use_triple and max_hold > 0)

        try:
            sl_pips = getattr(self.settings.risk, "meta_label_sl_pips", None)
            sl_pips = float(sl_pips) if sl_pips is not None else None
        except Exception:
            sl_pips = None
        try:
            tp_pips = getattr(self.settings.risk, "meta_label_tp_pips", None)
            tp_pips = float(tp_pips) if tp_pips is not None else None
        except Exception:
            tp_pips = None

        return _LabelConfig(
            horizon=max(1, horizon),
            min_dist=max(0.0, min_dist),
            use_triple_barrier=use_triple,
            max_hold=max_hold,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
        )

    @staticmethod
    def _infer_pip_size(symbol: str | None) -> float:
        sym = str(symbol or "").upper()
        if sym.startswith("XAU") or sym.startswith("XAG"):
            return 0.01
        if "BTC" in sym or "ETH" in sym or "LTC" in sym:
            return 1.0
        if sym.endswith("JPY") or sym.startswith("JPY"):
            return 0.01
        return 0.0001

    def _compute_basic_features(self, df: Any, *, use_gpu: bool = False) -> Any:
        if df is None or df.empty:
            return df
        out = df.copy()
        close = out["close"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        out["rsi"] = _compute_rsi(close)
        macd, macd_signal, macd_hist = _compute_macd(close)
        out["macd"] = macd
        out["macd_signal"] = macd_signal
        out["macd_hist"] = macd_hist
        out["adx"] = _compute_adx_numba(high.to_numpy(), low.to_numpy(), close.to_numpy())
        return out

    def _compute_volatility_features(self, df: Any) -> Any:
        if df is None or df.empty:
            return df
        out = df.copy()
        close = out["close"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        out["returns"] = close.pct_change().fillna(0.0)
        out["atr14"] = _compute_atr(high, low, close, period=14)
        ma = close.rolling(20, min_periods=1).mean()
        std = close.rolling(20, min_periods=1).std().fillna(0.0)
        upper = ma + 2.0 * std
        lower = ma - 2.0 * std
        out["bb_width"] = ((upper - lower) / (ma.replace(0.0, np.nan))).fillna(0.0)
        return out

    def _compute_volume_profile_features(self, df: Any) -> Any:
        if df is None or df.empty:
            return df
        out = df.copy()
        if "volume" in out.columns:
            volume = out["volume"].astype(float)
        else:
            volume = _make_series(np.ones(len(out), dtype=float), index=out.index)
        close = out["close"].astype(float)
        window = 20
        vol_sum = volume.rolling(window, min_periods=1).sum().replace(0.0, np.nan)
        poc = (close * volume).rolling(window, min_periods=1).sum() / vol_sum
        poc = poc.bfill().fillna(close)
        out["dist_to_poc"] = (close - poc).fillna(0.0)
        std = close.rolling(window, min_periods=1).std().fillna(0.0)
        out["in_value_area"] = ((close >= (poc - std)) & (close <= (poc + std))).astype(float)
        return out

    def _compute_obi_features(self, df: Any, *, use_gpu: bool = False) -> Any:
        if df is None or df.empty:
            return df
        out = df.copy()
        open_ = out["open"].astype(float)
        close = out["close"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        rng = (high - low).replace(0.0, np.nan)
        if "volume" in out.columns:
            volume = out["volume"].astype(float)
        else:
            volume = _make_series(np.ones(len(out), dtype=float), index=out.index)
        imbalance = ((close - open_) / rng).fillna(0.0) * volume
        out["vol_imbalance"] = imbalance.fillna(0.0)
        out["obi_mom3"] = out["vol_imbalance"].rolling(3, min_periods=1).mean().fillna(0.0)
        out["obi_seq_up5"] = (out["vol_imbalance"] > 0).astype(float).rolling(5, min_periods=1).mean().fillna(0.0)
        out["obi_seq_dn5"] = (out["vol_imbalance"] < 0).astype(float).rolling(5, min_periods=1).mean().fillna(0.0)
        return out

    def _compute_session_features(self, df: Any) -> Any:
        if df is None or df.empty:
            return df
        out = df.copy()
        if not _is_datetime_index(out.index):
            return out
        try:
            idx_utc = out.index.tz_convert("UTC") if out.index.tz is not None else out.index.tz_localize("UTC")
        except Exception:
            return out
        hour = idx_utc.hour
        out["session_asia"] = ((hour >= 0) & (hour < 7)).astype(float)
        out["session_london"] = ((hour >= 7) & (hour < 13)).astype(float)
        out["session_newyork"] = ((hour >= 13) & (hour < 21)).astype(float)
        out["hour_sin"] = np.sin((2.0 * np.pi * hour) / 24.0)
        out["hour_cos"] = np.cos((2.0 * np.pi * hour) / 24.0)
        if {"high", "low", "close"}.issubset(out.columns):
            day = _make_series(idx_utc.date, index=out.index)
            asia_mask = (hour >= 0) & (hour < 7)
            asia_high = out["high"].where(asia_mask).groupby(day).transform("max")
            asia_low = out["low"].where(asia_mask).groupby(day).transform("min")
            out["asia_range_width"] = (asia_high - asia_low).fillna(0.0)
            london_mask = (hour >= 7) & (hour < 13)
            out["london_break_above_asia"] = (london_mask & (out["close"] > asia_high)).astype(float)
            out["london_break_below_asia"] = (london_mask & (out["close"] < asia_low)).astype(float)
        return out

    @staticmethod
    def _safe_symbol_tag(symbol: str) -> str:
        safe = "".join(c for c in str(symbol or "") if c.isalnum() or c in ("-", "_"))
        return safe or "GLOBAL"

    def _prop_gene_artifact_paths(self, symbol: str | None) -> list[Path]:
        safe = self._safe_symbol_tag(symbol or "")
        paths: list[Path] = []
        cache_dir = Path(getattr(self.settings.system, "cache_dir", "cache") or "cache")
        paths.append(cache_dir / f"talib_knowledge_{safe}.json")
        paths.append(cache_dir / "talib_knowledge.json")

        checkpoint = str(
            getattr(
                getattr(self.settings, "models", None),
                "prop_search_checkpoint",
                "models/strategy_evo_checkpoint.json",
            )
            or "models/strategy_evo_checkpoint.json"
        )
        ckpt = Path(checkpoint)
        paths.append(ckpt)
        try:
            for candidate in ckpt.parent.glob(f"{ckpt.stem}_{safe}_*{ckpt.suffix}"):
                paths.append(candidate)
        except Exception:
            pass

        uniq: list[Path] = []
        seen: set[str] = set()
        for p in paths:
            key = str(p.resolve()) if p.exists() else str(p)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)

        existing = [p for p in uniq if p.exists() and p.is_file()]
        existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return existing

    @staticmethod
    def _parse_discovered_gene(
        *,
        raw: dict[str, Any],
        available: set[str] | None,
        payload_symbol: str,
        payload_tf: str,
        TALibStrategyGene: Any,
    ) -> Any | None:
        def _to_float(key: str, default: float) -> float:
            try:
                return float(raw.get(key, default) or default)
            except Exception:
                return float(default)

        def _to_bool(key: str, default: bool = False) -> bool:
            val = raw.get(key, default)
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return float(val) != 0.0
            return str(val).strip().lower() in {"1", "true", "yes", "on"}

        inds_raw = raw.get("indicators") or []
        indicators: list[str] = []
        for ind in inds_raw:
            name = str(ind).strip().upper()
            if name and (not available or name in available) and name not in indicators:
                indicators.append(name)
        if not indicators:
            return None

        params_raw = raw.get("params") if isinstance(raw.get("params"), dict) else {}
        params: dict[str, dict[str, Any]] = {}
        for ind in indicators:
            val = params_raw.get(ind) or params_raw.get(ind.lower()) or params_raw.get(ind.upper()) or {}
            params[ind] = dict(val) if isinstance(val, dict) else {}

        weights_raw = raw.get("weights") if isinstance(raw.get("weights"), dict) else {}
        weights: dict[str, float] = {}
        for ind in indicators:
            w = weights_raw.get(ind)
            if w is None:
                w = weights_raw.get(ind.lower(), 1.0)
            try:
                weights[ind] = float(w)
            except Exception:
                weights[ind] = 1.0

        try:
            return TALibStrategyGene(
                indicators=indicators,
                params=params,
                combination_method=str(raw.get("combination_method", "weighted_vote") or "weighted_vote"),
                long_threshold=_to_float("long_threshold", 0.66),
                short_threshold=_to_float("short_threshold", -0.66),
                weights=weights,
                preferred_regime=str(raw.get("preferred_regime", "any") or "any"),
                strategy_id=str(raw.get("strategy_id", "") or ""),
                fitness=_to_float("fitness", 0.0),
                sharpe_ratio=_to_float("sharpe_ratio", 0.0),
                win_rate=_to_float("win_rate", 0.0),
                max_dd_pct=_to_float(
                    "max_dd_pct",
                    _to_float("max_drawdown", _to_float("max_dd", _to_float("drawdown", 0.0))),
                ),
                trades=_to_float("trades", _to_float("trades_count", _to_float("trade_count", 0.0))),
                net_profit=_to_float("net_profit", 0.0),
                profit_factor=_to_float("profit_factor", 0.0),
                expectancy=_to_float("expectancy", 0.0),
                use_ob=_to_bool("use_ob", False),
                use_fvg=_to_bool("use_fvg", False),
                use_liq_sweep=_to_bool("use_liq_sweep", False),
                mtf_confirmation=_to_bool("mtf_confirmation", False),
                use_premium_discount=_to_bool("use_premium_discount", False),
                use_inducement=_to_bool("use_inducement", False),
                tp_pips=_to_float("tp_pips", 40.0),
                sl_pips=_to_float("sl_pips", 20.0),
                source_symbol=payload_symbol,
                source_timeframe=payload_tf,
                in_sample_net_profit=_to_float("in_sample_net_profit", 0.0),
                in_sample_sharpe_ratio=_to_float("in_sample_sharpe_ratio", 0.0),
                in_sample_win_rate=_to_float("in_sample_win_rate", 0.0),
                in_sample_profit_factor=_to_float("in_sample_profit_factor", 0.0),
                in_sample_trades=_to_float("in_sample_trades", 0.0),
                in_sample_max_dd_pct=_to_float("in_sample_max_dd_pct", 0.0),
                in_sample_months=_to_float("in_sample_months", 0.0),
                holdout_net_profit=_to_float("holdout_net_profit", 0.0),
                holdout_sharpe_ratio=_to_float("holdout_sharpe_ratio", 0.0),
                holdout_win_rate=_to_float("holdout_win_rate", 0.0),
                holdout_profit_factor=_to_float("holdout_profit_factor", 0.0),
                holdout_trades=_to_float("holdout_trades", 0.0),
                holdout_max_dd_pct=_to_float(
                    "holdout_max_dd_pct",
                    _to_float("holdout_max_drawdown", _to_float("holdout_max_dd", 0.0)),
                ),
                holdout_months=_to_float("holdout_months", 0.0),
                holdout_trades_per_month=_to_float("holdout_trades_per_month", 0.0),
                holdout_monthly_profit_pct=_to_float("holdout_monthly_profit_pct", 0.0),
                truth_probability=_to_float("truth_probability", 0.0),
                forward_test_passed=_to_bool("forward_test_passed", _to_bool("holdout_passed", False)),
                in_sample_journal=dict(raw.get("in_sample_journal", {}) or {})
                if isinstance(raw.get("in_sample_journal"), dict)
                else {},
                holdout_journal=dict(raw.get("holdout_journal", {}) or {})
                if isinstance(raw.get("holdout_journal"), dict)
                else {},
            )
        except Exception:
            return None

    def _load_discovered_base_signal_genes(self, symbol: str | None, max_genes: int = 100) -> list[Any]:
        try:
            from .talib_mixer import TALIB_AVAILABLE, TALibStrategyGene, TALibStrategyMixer
        except Exception:
            return []
        rust_talib_available = False
        with contextlib.suppress(Exception):
            import forex_bindings  # type: ignore

            rust_talib_available = bool(hasattr(forex_bindings, "talib_bulk_signals_ohlcv"))
        if not TALIB_AVAILABLE and not rust_talib_available:
            return []

        candidates = self._prop_gene_artifact_paths(symbol)
        if not candidates:
            return []

        target_symbol = str(symbol or "").upper().strip()
        strict_symbol = str(os.environ.get("FOREX_BOT_PROP_SYMBOL_STRICT", "1") or "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        available: set[str] | None = None
        if TALIB_AVAILABLE:
            mixer = TALibStrategyMixer(
                device="cpu",
                use_volume_features=bool(getattr(self.settings.system, "use_volume_features", False)),
            )
            candidate_available = {str(i).upper() for i in getattr(mixer, "available_indicators", [])}
            if candidate_available:
                available = candidate_available

        try:
            max_dd = float(
                os.environ.get(
                    "FOREX_BOT_PROP_BASE_SIGNAL_MAX_DD",
                    getattr(self.settings.risk, "total_drawdown_limit", 0.07),
                )
                or 0.07
            )
        except Exception:
            max_dd = 0.07
        max_dd = float(min(1.0, max(0.0, max_dd)))
        try:
            min_profit = float(os.environ.get("FOREX_BOT_PROP_BASE_SIGNAL_MIN_PROFIT", "0.0") or 0.0)
        except Exception:
            min_profit = 0.0
        try:
            min_trades = float(os.environ.get("FOREX_BOT_PROP_BASE_SIGNAL_MIN_TRADES", "5") or 5.0)
        except Exception:
            min_trades = 5.0
        min_trades = float(max(0.0, min_trades))
        runtime_profile = str(os.environ.get("FOREX_BOT_RUNTIME_PROFILE", "") or "").strip().lower()
        elite_default = runtime_profile.startswith("rust")
        elite_filter = str(
            os.environ.get("FOREX_BOT_PROP_ELITE_FILTER", "1" if elite_default else "0") or ("1" if elite_default else "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        require_forward = str(
            os.environ.get("FOREX_BOT_PROP_REQUIRE_FORWARD_PASS", "1" if elite_filter else "0") or ("1" if elite_filter else "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        try:
            min_holdout_months = float(os.environ.get("FOREX_BOT_PROP_MIN_HOLDOUT_MONTHS", "6.0" if elite_filter else "0.0") or (6.0 if elite_filter else 0.0))
        except Exception:
            min_holdout_months = 6.0 if elite_filter else 0.0
        min_holdout_months = float(max(0.0, min_holdout_months))
        try:
            holdout_max_dd = float(os.environ.get("FOREX_BOT_PROP_HOLDOUT_MAX_DD", "0.03" if elite_filter else "1.0") or (0.03 if elite_filter else 1.0))
        except Exception:
            holdout_max_dd = 0.03 if elite_filter else 1.0
        holdout_max_dd = float(min(1.0, max(0.0, holdout_max_dd)))
        try:
            min_truth = float(
                os.environ.get(
                    "FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY",
                    os.environ.get(
                        "FOREX_BOT_MIN_TRUTH_PROBABILITY",
                        str(getattr(self.settings.models, "prop_search_holdout_min_truth_probability", 0.0) or 0.0),
                    ),
                )
                or 0.0
            )
        except Exception:
            min_truth = 0.0
        if min_truth > 1.0:
            min_truth *= 0.01
        min_truth = float(min(1.0, max(0.0, min_truth)))
        strict_prefilter = str(
            os.environ.get("FOREX_BOT_PROP_BASE_SIGNAL_STRICT_FILTER", "1" if elite_filter else "0")
            or ("1" if elite_filter else "0")
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        parsed_all: list[Any] = []
        parsed_filtered: list[Any] = []
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            payload_symbol = str(payload.get("symbol", "") or "").upper().strip()
            if strict_symbol and payload_symbol and target_symbol and payload_symbol != target_symbol:
                continue
            payload_tf = str(payload.get("timeframe", payload.get("tf", "")) or "").upper().strip()
            raw_genes = payload.get("best_genes")
            if not isinstance(raw_genes, list) or not raw_genes:
                continue

            for raw in raw_genes:
                if not isinstance(raw, dict):
                    continue
                gene = self._parse_discovered_gene(
                    raw=raw,
                    available=available,
                    payload_symbol=payload_symbol,
                    payload_tf=payload_tf,
                    TALibStrategyGene=TALibStrategyGene,
                )
                if gene is not None:
                    parsed_all.append(gene)
                    try:
                        dd = float(getattr(gene, "max_dd_pct", 0.0) or 0.0)
                    except Exception:
                        dd = 1.0
                    try:
                        profit = float(getattr(gene, "net_profit", 0.0) or 0.0)
                    except Exception:
                        profit = 0.0
                    try:
                        trades = float(getattr(gene, "trades", 0.0) or 0.0)
                    except Exception:
                        trades = 0.0
                    try:
                        truth = float(getattr(gene, "truth_probability", 0.0) or 0.0)
                    except Exception:
                        truth = 0.0
                    if truth > 1.0:
                        truth *= 0.01
                    try:
                        hold_months = float(getattr(gene, "holdout_months", 0.0) or 0.0)
                    except Exception:
                        hold_months = 0.0
                    try:
                        hold_dd = float(getattr(gene, "holdout_max_dd_pct", 0.0) or 0.0)
                    except Exception:
                        hold_dd = 0.0
                    forward_ok = bool(getattr(gene, "forward_test_passed", False))
                    if (
                        dd <= max_dd
                        and profit > min_profit
                        and trades >= min_trades
                        and (not require_forward or forward_ok)
                        and truth >= min_truth
                        and (min_holdout_months <= 0.0 or hold_months >= min_holdout_months)
                        and (holdout_max_dd >= 1.0 or hold_dd <= holdout_max_dd)
                    ):
                        parsed_filtered.append(gene)

        if not parsed_all:
            return []
        parsed = parsed_filtered if parsed_filtered else ([] if strict_prefilter else parsed_all)
        if not parsed:
            return []

        dedup: dict[str, Any] = {}
        for gene in sorted(
            parsed,
            key=lambda g: (
                float(getattr(g, "fitness", 0.0) or 0.0),
                float(getattr(g, "sharpe_ratio", 0.0) or 0.0),
                float(getattr(g, "net_profit", 0.0) or 0.0),
                -float(getattr(g, "max_dd_pct", 0.0) or 0.0),
            ),
            reverse=True,
        ):
            sid = str(getattr(gene, "strategy_id", "") or "").strip()
            if sid:
                key = f"id:{sid}"
            else:
                key = (
                    f"sig:{tuple(gene.indicators)}|{gene.combination_method}|"
                    f"{float(gene.long_threshold):.6f}|{float(gene.short_threshold):.6f}"
                )
            if key in dedup:
                continue
            dedup[key] = gene

        out = list(dedup.values())
        out.sort(
            key=lambda g: (
                float(getattr(g, "fitness", 0.0) or 0.0),
                float(getattr(g, "sharpe_ratio", 0.0) or 0.0),
                float(getattr(g, "win_rate", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return out[: max(1, int(max_genes))]

    def _compute_discovered_base_signal(self, df: Any, *, symbol: str | None) -> np.ndarray | None:
        if df is None or df.empty:
            return None
        rust_signal: np.ndarray | None = None
        with contextlib.suppress(Exception):
            volume_arr = (
                np.asarray(df["volume"], dtype=np.float64)
                if bool(getattr(self.settings.system, "use_volume_features", False)) and "volume" in df.columns
                else None
            )
            rust_signal = self._compute_discovered_base_signal_ohlcv_numpy(
                open_arr=np.asarray(df["open"], dtype=np.float64),
                high_arr=np.asarray(df["high"], dtype=np.float64),
                low_arr=np.asarray(df["low"], dtype=np.float64),
                close_arr=np.asarray(df["close"], dtype=np.float64),
                volume_arr=volume_arr,
                symbol=symbol,
            )
        if rust_signal is None:
            return None
        out = np.asarray(rust_signal, dtype=np.int8).reshape(-1)
        if out.size != len(df):
            out = self._fit_len(out, len(df), default=0.0, dtype=np.int8)
        return out

    def _discovered_base_signal_settings(self) -> tuple[int, float, float, float]:
        try:
            default_genes = int(getattr(getattr(self.settings, "models", None), "prop_search_portfolio_size", 4) or 4)
        except Exception:
            default_genes = 4
        try:
            max_genes = int(os.environ.get("FOREX_BOT_PROP_BASE_SIGNAL_GENES", str(default_genes)) or default_genes)
        except Exception:
            max_genes = default_genes
        max_genes = max(1, max_genes)

        try:
            threshold = float(os.environ.get("FOREX_BOT_PROP_BASE_SIGNAL_THRESHOLD", "0.15") or 0.15)
        except Exception:
            threshold = 0.15
        threshold = float(min(0.95, max(0.0, threshold)))
        try:
            min_coverage = float(os.environ.get("FOREX_BOT_PROP_BASE_SIGNAL_MIN_COVERAGE", "0.02") or 0.02)
        except Exception:
            min_coverage = 0.02
        try:
            max_coverage = float(os.environ.get("FOREX_BOT_PROP_BASE_SIGNAL_MAX_COVERAGE", "1.0") or 1.0)
        except Exception:
            max_coverage = 1.0
        min_coverage = float(min(0.95, max(0.0, min_coverage)))
        max_coverage = float(min(0.99, max(min_coverage, max_coverage)))
        return max_genes, threshold, min_coverage, max_coverage

    @staticmethod
    def _score_to_discovered_signal(
        score: np.ndarray,
        *,
        threshold: float,
        min_coverage: float,
        max_coverage: float,
    ) -> np.ndarray:
        score_arr = np.asarray(score, dtype=np.float64).reshape(-1)
        if score_arr.size <= 0:
            return np.zeros(0, dtype=np.int8)

        abs_score = np.abs(score_arr)
        signal = np.where(score_arr >= threshold, 1, np.where(score_arr <= -threshold, -1, 0)).astype(np.int8)
        n = int(signal.shape[0])
        if n <= 0:
            return signal

        target_min = int(round(min_coverage * n))
        target_max = int(round(max_coverage * n))
        target_min = max(0, min(target_min, n))
        target_max = max(target_min, min(target_max, n))

        active_now = int(np.count_nonzero(signal))
        if target_max > 0 and active_now > target_max:
            top_idx = _rust_rank_scores_desc(abs_score, absolute=False)
            if top_idx is None:
                top_idx = np.argsort(abs_score)[::-1]
            top_idx = np.asarray(top_idx, dtype=np.int64)[:target_max]
            trimmed = np.zeros(n, dtype=np.int8)
            sel = score_arr[top_idx]
            trimmed[top_idx] = np.where(sel > 0.0, 1, np.where(sel < 0.0, -1, 0)).astype(np.int8)
            return trimmed
        if target_min > 0 and active_now < target_min:
            top_idx = _rust_rank_scores_desc(abs_score, absolute=False)
            if top_idx is None:
                top_idx = np.argsort(abs_score)[::-1]
            top_idx = np.asarray(top_idx, dtype=np.int64)[:target_min]
            boosted = np.zeros(n, dtype=np.int8)
            sel = score_arr[top_idx]
            boosted[top_idx] = np.where(sel > 0.0, 1, np.where(sel < 0.0, -1, 0)).astype(np.int8)
            return boosted
        return signal

    @staticmethod
    def _gene_has_custom_params(gene: Any) -> bool:
        params = getattr(gene, "params", {}) or {}
        if not isinstance(params, dict) or not params:
            return False
        for value in params.values():
            if isinstance(value, dict):
                if len(value) == 0:
                    continue
                return True
            if value:
                return True
        return False

    def _compute_discovered_base_signal_ohlcv_numpy(
        self,
        *,
        open_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
        volume_arr: np.ndarray | None,
        symbol: str | None,
    ) -> np.ndarray | None:
        try:
            import forex_bindings  # type: ignore
        except Exception:
            return None
        if not hasattr(forex_bindings, "talib_bulk_signals_ohlcv"):
            return None

        max_genes, threshold, min_coverage, max_coverage = self._discovered_base_signal_settings()
        genes = self._load_discovered_base_signal_genes(symbol, max_genes=max_genes)
        if not genes:
            return None

        o = np.asarray(open_arr, dtype=np.float64).reshape(-1)
        h = np.asarray(high_arr, dtype=np.float64).reshape(-1)
        l = np.asarray(low_arr, dtype=np.float64).reshape(-1)
        c = np.asarray(close_arr, dtype=np.float64).reshape(-1)
        n = int(min(o.size, h.size, l.size, c.size))
        if n <= 0:
            return None
        o = np.nan_to_num(o[:n], nan=0.0, posinf=0.0, neginf=0.0)
        h = np.nan_to_num(h[:n], nan=0.0, posinf=0.0, neginf=0.0)
        l = np.nan_to_num(l[:n], nan=0.0, posinf=0.0, neginf=0.0)
        c = np.nan_to_num(c[:n], nan=0.0, posinf=0.0, neginf=0.0)

        use_volume = bool(getattr(self.settings.system, "use_volume_features", False))
        vol: np.ndarray | None = None
        if use_volume and volume_arr is not None:
            raw_vol = np.asarray(volume_arr, dtype=np.float64).reshape(-1)
            if raw_vol.size > 0:
                vol = np.nan_to_num(
                    self._fit_len(raw_vol, n, default=0.0, dtype=np.float64),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

        indicator_sets: list[list[str]] = []
        weight_sets: list[list[float]] = []
        long_thresholds: list[float] = []
        short_thresholds: list[float] = []
        gene_weights: list[float] = []

        for gene in genes:
            if self._gene_has_custom_params(gene):
                continue
            indicators: list[str] = []
            for raw_ind in getattr(gene, "indicators", []) or []:
                name = str(raw_ind).strip().upper()
                if name and name not in indicators:
                    indicators.append(name)
            if not indicators:
                continue

            weights_raw = getattr(gene, "weights", {}) or {}
            wset: list[float] = []
            for ind in indicators:
                val = weights_raw.get(ind, weights_raw.get(ind.lower(), 1.0))
                try:
                    w = float(val)
                except Exception:
                    w = 1.0
                if not np.isfinite(w):
                    w = 1.0
                wset.append(w)

            try:
                long_thr = float(getattr(gene, "long_threshold", 0.66) or 0.66)
            except Exception:
                long_thr = 0.66
            try:
                short_thr = float(getattr(gene, "short_threshold", -0.66) or -0.66)
            except Exception:
                short_thr = -0.66

            w_gene = float(getattr(gene, "fitness", 0.0) or 0.0)
            if not np.isfinite(w_gene) or w_gene <= 0.0:
                w_gene = 1.0

            indicator_sets.append(indicators)
            weight_sets.append(wset)
            long_thresholds.append(long_thr)
            short_thresholds.append(short_thr)
            gene_weights.append(w_gene)

        if not indicator_sets:
            return None

        try:
            causal_min_bars = int(os.environ.get("FOREX_BOT_TALIB_CAUSAL_MIN_BARS", "30") or 30)
        except Exception:
            causal_min_bars = 30
        causal_min_bars = max(2, causal_min_bars)

        try:
            raw = forex_bindings.talib_bulk_signals_ohlcv(
                o,
                h,
                l,
                c,
                indicator_sets=indicator_sets,
                weight_sets=weight_sets,
                long_thresholds=long_thresholds,
                short_thresholds=short_thresholds,
                volume=vol,
                include_raw=False,
                causal_min_bars=causal_min_bars,
            )
        except TypeError:
            raw = forex_bindings.talib_bulk_signals_ohlcv(
                o,
                h,
                l,
                c,
                indicator_sets=indicator_sets,
                weight_sets=weight_sets,
                long_thresholds=long_thresholds,
                short_thresholds=short_thresholds,
                volume=vol,
                include_raw=False,
            )
        except Exception as exc:
            logger.debug("Rust discovered base-signal bulk call failed: %s", exc)
            return None

        signals = np.asarray(raw, dtype=np.float64)
        expected_genes = len(gene_weights)
        if signals.ndim != 2:
            return None
        if signals.shape[0] == expected_genes and signals.shape[1] == n:
            signals = signals.T
        if signals.shape[0] != n or signals.shape[1] != expected_genes:
            logger.debug(
                "Rust discovered base-signal shape mismatch (got=%s expected=(%s,%s)).",
                signals.shape,
                n,
                expected_genes,
            )
            return None

        w = np.asarray(gene_weights, dtype=np.float64).reshape(-1)
        weight_sum = float(np.sum(np.abs(w)))
        if weight_sum <= 0.0:
            return None

        sig_mat = np.nan_to_num(np.asarray(signals, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        score = (sig_mat @ w) / weight_sum
        return self._score_to_discovered_signal(
            score,
            threshold=threshold,
            min_coverage=min_coverage,
            max_coverage=max_coverage,
        )

    def _compute_base_signal(self, df: Any, *, symbol: str | None = None) -> Any:
        if df is None or df.empty:
            return df
        out = df.copy()

        signal_source = str(os.environ.get("FOREX_BOT_BASE_SIGNAL_SOURCE", "discovery_first") or "discovery_first").strip().lower()
        use_discovery = signal_source in {"discovery", "discovery_first", "prop", "talib", "mixer", "auto"}
        target_symbol = symbol
        if not target_symbol:
            target_symbol = str(getattr(out, "attrs", {}).get("symbol", "") or "").strip()
        discovered_signal = None
        if use_discovery:
            discovered_signal = self._compute_discovered_base_signal(out, symbol=target_symbol)
        if discovered_signal is not None:
            out["base_signal"] = discovered_signal.astype(int, copy=False)
            return out
        out["base_signal"] = 0
        return out

    @staticmethod
    def _find_rsi_feature_column(columns: Iterable[str]) -> str | None:
        cols = [str(c) for c in columns]
        if "rsi" in cols:
            return "rsi"
        if "ta_rsi" in cols:
            return "ta_rsi"
        matches = [c for c in cols if c.lower().endswith("ta_rsi")]
        if matches:
            return min(matches, key=len)
        return None

    @staticmethod
    def _find_macd_hist_feature_column(columns: Iterable[str]) -> str | None:
        cols = [str(c) for c in columns]
        if "macd_hist" in cols:
            return "macd_hist"
        preferred = (
            "ta_macd_outmacdhist",
            "ta_macdext_outmacdhist",
            "ta_macdfix_outmacdhist",
        )
        for name in preferred:
            if name in cols:
                return name
        matches = [c for c in cols if c.lower().endswith("outmacdhist")]
        if matches:
            return min(matches, key=len)
        return None

    def _compute_labels(
        self,
        close: Any,
        cfg: _LabelConfig,
        *,
        high: Any | None = None,
        low: Any | None = None,
        symbol: str | None = None,
        base_signal: Any | None = None,
    ) -> Any:
        close_idx = getattr(close, "index", None)
        close_arr = np.asarray(close, dtype=np.float64).reshape(-1)
        n = int(close_arr.shape[0])
        if n <= 0:
            return _make_series(np.zeros(0, dtype=np.int8), index=close_idx, template=close)
        close_ts = self._to_timestamp_ns(close_idx, n)

        def _aligned(values: Any | None, *, default: float) -> np.ndarray:
            if values is None:
                return np.full(n, float(default), dtype=np.float64)
            src = values
            arr = np.asarray(src, dtype=np.float64).reshape(-1)
            if close_idx is not None:
                src_idx = getattr(src, "index", None)
                if src_idx is not None:
                    src_ts = self._to_timestamp_ns(src_idx, arr.shape[0])
                    arr = self._align_series_exact_by_ts(
                        close_ts,
                        src_ts,
                        arr,
                        default=float(default),
                        dtype=np.float64,
                    )
            return self._fit_len(arr, n, default=float(default), dtype=np.float64)

        close_arr = self._fit_len(close_arr, n, default=0.0, dtype=np.float64)
        hi_arr = _aligned(high if high is not None else close_arr, default=float(close_arr[-1]))
        lo_arr = _aligned(low if low is not None else close_arr, default=float(close_arr[-1]))

        sig_arr: np.ndarray | None = None
        if base_signal is not None:
            sig_src = base_signal
            sig_np = np.asarray(sig_src, dtype=np.float64).reshape(-1)
            if close_idx is not None:
                sig_idx = getattr(sig_src, "index", None)
                if sig_idx is not None:
                    sig_ts = self._to_timestamp_ns(sig_idx, sig_np.shape[0])
                    sig_np = self._align_series_exact_by_ts(
                        close_ts,
                        sig_ts,
                        sig_np,
                        default=0.0,
                        dtype=np.float64,
                    )
            sig_arr = self._fit_len(sig_np, n, default=0.0, dtype=np.float64)
            sig_arr = np.nan_to_num(sig_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int8, copy=False)

        labels = self._compute_labels_numpy(
            close_arr,
            cfg,
            high=hi_arr,
            low=lo_arr,
            symbol=symbol,
            base_signal=sig_arr,
        )
        return _make_series(
            np.asarray(labels, dtype=np.int8).astype(int, copy=False),
            index=close_idx,
            template=close,
        )

    @staticmethod
    def _to_timestamp_ns(values: object, rows: int) -> np.ndarray:
        n = max(0, int(rows))
        if values is None:
            return np.arange(n, dtype=np.int64)
        try:
            if hasattr(values, "asi8"):
                out = np.asarray(values.asi8, dtype=np.int64).reshape(-1)
                if out.size < n:
                    if out.size <= 0:
                        return np.arange(n, dtype=np.int64)
                    pad = np.full(n - out.size, int(out[-1]), dtype=np.int64)
                    return np.concatenate([out, pad])
                if out.size > n:
                    return out[:n]
                return out
        except Exception:
            pass
        arr = np.asarray(values).reshape(-1)
        if arr.size == 0:
            return np.arange(n, dtype=np.int64)
        out: np.ndarray
        try:
            if np.issubdtype(arr.dtype, np.datetime64):
                out = arr.astype("datetime64[ns]").astype(np.int64, copy=False)
            elif arr.dtype.kind in {"i", "u"}:
                out = arr.astype(np.int64, copy=False)
            elif arr.dtype.kind == "f":
                out = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
            else:
                out = np.zeros(arr.size, dtype=np.int64)
                for i, v in enumerate(arr):
                    try:
                        ns = getattr(v, "value", None)
                        if ns is not None:
                            out[i] = int(ns)
                            continue
                        out[i] = int(np.datetime64(v, "ns").astype(np.int64))
                    except Exception:
                        out[i] = 0
        except Exception:
            out = np.arange(arr.size, dtype=np.int64)
        if out.size < n:
            if out.size <= 0:
                return np.arange(n, dtype=np.int64)
            pad = np.full(n - out.size, int(out[-1]), dtype=np.int64)
            return np.concatenate([out, pad])
        if out.size > n:
            return out[:n]
        return out

    @staticmethod
    def _align_series_by_ts(
        target_ts: np.ndarray,
        source_ts: np.ndarray,
        values: object,
        *,
        default: float = 0.0,
        dtype: Any = np.float32,
    ) -> np.ndarray:
        tgt = np.asarray(target_ts, dtype=np.int64).reshape(-1)
        src = np.asarray(source_ts, dtype=np.int64).reshape(-1)
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        if tgt.size == 0:
            return np.zeros(0, dtype=dtype)
        if vals.size == 0 or src.size == 0:
            return np.full(tgt.shape, float(default), dtype=dtype)

        m = min(src.size, vals.size)
        src = src[:m]
        vals = vals[:m]
        if m <= 0:
            return np.full(tgt.shape, float(default), dtype=dtype)

        out = _rust_align_by_ts(src, vals, tgt, default=float(default), forward_fill=True)
        if out is None:
            order = _sorted_time_order(src)
            src_sorted = src if order is None else src[order]
            vals_sorted = vals if order is None else vals[order]
            pos = np.searchsorted(src_sorted, tgt, side="right") - 1
            out = np.full(tgt.shape, float(default), dtype=np.float64)
            valid = pos >= 0
            if valid.any():
                idx = np.clip(pos[valid], 0, vals_sorted.size - 1)
                out[valid] = vals_sorted[idx]
        return np.nan_to_num(out, nan=float(default), posinf=float(default), neginf=float(default)).astype(
            dtype, copy=False
        )

    @staticmethod
    def _align_series_exact_by_ts(
        target_ts: np.ndarray,
        source_ts: np.ndarray,
        values: object,
        *,
        default: float = 0.0,
        dtype: Any = np.float32,
    ) -> np.ndarray:
        tgt = np.asarray(target_ts, dtype=np.int64).reshape(-1)
        src = np.asarray(source_ts, dtype=np.int64).reshape(-1)
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        if tgt.size == 0:
            return np.zeros(0, dtype=dtype)
        if vals.size == 0 or src.size == 0:
            return np.full(tgt.shape, float(default), dtype=dtype)

        m = min(src.size, vals.size)
        src = src[:m]
        vals = vals[:m]
        if m <= 0:
            return np.full(tgt.shape, float(default), dtype=dtype)

        out = _rust_align_by_ts(src, vals, tgt, default=float(default), forward_fill=False)
        if out is None:
            order = _sorted_time_order(src)
            src_sorted = src if order is None else src[order]
            vals_sorted = vals if order is None else vals[order]
            pos = np.searchsorted(src_sorted, tgt, side="left")
            out = np.full(tgt.shape, float(default), dtype=np.float64)
            valid = pos < src_sorted.size
            if valid.any():
                matched = np.zeros(tgt.shape[0], dtype=bool)
                valid_pos = pos[valid]
                matched[valid] = src_sorted[valid_pos] == tgt[valid]
                take = valid & matched
                if take.any():
                    out[take] = vals_sorted[pos[take]]
        return np.nan_to_num(out, nan=float(default), posinf=float(default), neginf=float(default)).astype(
            dtype, copy=False
        )

    @staticmethod
    def _fit_len(arr: object, n: int, *, default: float = 0.0, dtype: Any = np.float64) -> np.ndarray:
        out = np.asarray(arr, dtype=dtype).reshape(-1)
        target = max(0, int(n))
        if out.size == target:
            return out
        if out.size <= 0:
            return np.full(target, default, dtype=dtype)
        if out.size > target:
            return out[:target]
        pad = np.full(target - out.size, float(out[-1]), dtype=dtype)
        return np.concatenate([out, pad])

    def _compute_labels_numpy(
        self,
        close: np.ndarray,
        cfg: _LabelConfig,
        *,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
        symbol: str | None = None,
        base_signal: np.ndarray | None = None,
    ) -> np.ndarray:
        close_arr = self._fit_len(close, len(np.asarray(close).reshape(-1)), default=0.0, dtype=np.float64)
        n = int(close_arr.shape[0])
        if n <= 0:
            return np.zeros(0, dtype=np.int8)

        if not cfg.use_triple_barrier:
            h = int(max(1, cfg.horizon))
            future = np.empty_like(close_arr)
            if h < n:
                future[:-h] = close_arr[h:]
                future[-h:] = np.nan
            else:
                future[:] = np.nan
            delta = future - close_arr
            up = delta > float(cfg.min_dist)
            down = delta < -float(cfg.min_dist)
            return np.where(up, 1, np.where(down, -1, 0)).astype(np.int8, copy=False)

        hi = self._fit_len(high if high is not None else close_arr, n, default=0.0, dtype=np.float64)
        lo = self._fit_len(low if low is not None else close_arr, n, default=0.0, dtype=np.float64)
        if n <= 2:
            return np.zeros(n, dtype=np.int8)

        pip_size = self._infer_pip_size(symbol)
        sl_pips = cfg.sl_pips if (cfg.sl_pips is not None and cfg.sl_pips > 0) else None
        tp_pips = cfg.tp_pips if (cfg.tp_pips is not None and cfg.tp_pips > 0) else None
        rr = float(getattr(self.settings.risk, "min_risk_reward", 2.0) or 2.0)
        atr_mult = float(getattr(self.settings.risk, "atr_stop_multiplier", 1.5) or 1.5)
        atr_period = max(2, int(getattr(self.settings.risk, "atr_period", 14) or 14))
        atr = _compute_atr_numpy(hi, lo, close_arr, period=atr_period)

        if sl_pips is not None:
            sl_dist = np.full(n, max(0.0, sl_pips * pip_size), dtype=np.float64)
        else:
            sl_dist = np.maximum(np.asarray(atr, dtype=np.float64) * max(0.1, atr_mult), float(cfg.min_dist))

        if tp_pips is not None:
            tp_dist = np.full(n, max(0.0, tp_pips * pip_size), dtype=np.float64)
        else:
            tp_dist = np.maximum(sl_dist * max(0.1, rr), float(cfg.min_dist))

        sig_arr: np.ndarray | None = None
        if base_signal is not None:
            sig_arr = self._fit_len(base_signal, n, default=0.0, dtype=np.float64).astype(np.int8, copy=False)

        max_hold = int(max(cfg.horizon, cfg.max_hold))
        max_hold = max(1, max_hold)

        if _rust_labels_backend_available(force_log=True):
            try:
                import forex_bindings  # type: ignore

                sig_arg = sig_arr.astype(np.int8, copy=False) if sig_arr is not None else None
                labels_rs = np.asarray(
                    forex_bindings.triple_barrier_labels(
                        close_arr,
                        hi,
                        lo,
                        sl_dist,
                        tp_dist,
                        int(max_hold),
                        sig_arg,
                    ),
                    dtype=np.int8,
                )
                if labels_rs.shape[0] == n:
                    return labels_rs.astype(np.int8, copy=False)
                logger.error(
                    "Rust triple-barrier labels shape mismatch (got=%s expected=%s); returning neutral labels.",
                    labels_rs.shape,
                    n,
                )
            except Exception as exc:
                _disable_rust_labels_backend()
                logger.error("Rust triple-barrier labels failed; returning neutral labels: %s", exc)
        else:
            logger.error("Rust triple-barrier labels backend unavailable; returning neutral labels.")
        return np.zeros(n, dtype=np.int8)

    def _prepare_rust_features_numpy(
        self,
        *,
        payload: dict[str, Any],
        features: np.ndarray,
        feature_names: list[str],
        symbol: str | None,
    ) -> PreparedDataset:
        X = np.asarray(features, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        rows = int(X.shape[0]) if X.ndim == 2 else 0
        if rows <= 0 or X.ndim != 2:
            return self._empty_numpy_dataset()

        ts = self._to_timestamp_ns(payload.get("timestamps"), rows)

        open_raw = np.asarray(payload.get("open"), dtype=np.float64).reshape(-1)
        high_raw = np.asarray(payload.get("high"), dtype=np.float64).reshape(-1)
        low_raw = np.asarray(payload.get("low"), dtype=np.float64).reshape(-1)
        close_raw = np.asarray(payload.get("close"), dtype=np.float64).reshape(-1)
        volume_raw_val = payload.get("volume")
        volume_raw: np.ndarray | None = None
        if volume_raw_val is not None:
            volume_raw = np.asarray(volume_raw_val, dtype=np.float64).reshape(-1)
        base_rows = int(close_raw.shape[0]) if close_raw.size > 0 else rows
        base_ts_values = payload.get("base_timestamps")
        if base_ts_values is None:
            base_ts_values = payload.get("timestamps")
        base_ts = self._to_timestamp_ns(base_ts_values, base_rows)

        open_arr = self._align_series_by_ts(ts, base_ts, payload.get("open"), default=0.0, dtype=np.float32)
        high_arr = self._align_series_by_ts(ts, base_ts, payload.get("high"), default=0.0, dtype=np.float32)
        low_arr = self._align_series_by_ts(ts, base_ts, payload.get("low"), default=0.0, dtype=np.float32)
        close_arr = self._align_series_by_ts(ts, base_ts, payload.get("close"), default=0.0, dtype=np.float32)
        volume_arr = None
        if volume_raw is not None:
            volume_arr = self._align_series_by_ts(ts, base_ts, volume_raw, default=0.0, dtype=np.float32)

        cols = [str(c) for c in feature_names]
        col_pos = {name: i for i, name in enumerate(cols)}
        use_base_signal = str(os.environ.get("FOREX_BOT_BASE_SIGNAL", "1") or "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        base_signal_np: np.ndarray | None = None
        base_col_idx = col_pos.get("base_signal")
        if base_col_idx is not None:
            base_signal_np = np.asarray(X[:, base_col_idx], dtype=np.int8)
        elif use_base_signal:
            signal_source = str(os.environ.get("FOREX_BOT_BASE_SIGNAL_SOURCE", "discovery_first") or "discovery_first").strip().lower()
            use_discovery = signal_source in {"discovery", "discovery_first", "prop", "talib", "mixer", "auto"}

            discovered_signal_np: np.ndarray | None = None
            if use_discovery:
                discovered_raw = self._compute_discovered_base_signal_ohlcv_numpy(
                    open_arr=open_raw,
                    high_arr=high_raw,
                    low_arr=low_raw,
                    close_arr=close_raw,
                    volume_arr=volume_raw,
                    symbol=symbol,
                )
                if discovered_raw is not None and discovered_raw.size > 0:
                    discovered_signal_np = self._align_series_by_ts(
                        ts,
                        base_ts,
                        discovered_raw,
                        default=0.0,
                        dtype=np.float32,
                    ).astype(np.int8, copy=False)

            if discovered_signal_np is not None:
                base_signal_np = discovered_signal_np

            if base_signal_np is None:
                base_signal_np = np.zeros(rows, dtype=np.int8)
            X = np.column_stack([X, base_signal_np.astype(np.float32, copy=False)]).astype(np.float32, copy=False)
            cols.append("base_signal")

        label_cfg = self._label_config()
        labels = self._compute_labels_numpy(
            close_arr.astype(np.float64, copy=False),
            label_cfg,
            high=high_arr.astype(np.float64, copy=False),
            low=low_arr.astype(np.float64, copy=False),
            symbol=symbol,
            base_signal=base_signal_np,
        )
        trim = int(max(label_cfg.horizon, label_cfg.max_hold if label_cfg.use_triple_barrier else label_cfg.horizon))
        if trim > 0 and rows > trim:
            end = rows - trim
            X = X[:end]
            labels = labels[:end]
            ts = ts[:end]
            open_arr = open_arr[:end]
            high_arr = high_arr[:end]
            low_arr = low_arr[:end]
            close_arr = close_arr[:end]
            if volume_arr is not None:
                volume_arr = volume_arr[:end]
        elif trim > 0:
            return self._empty_numpy_dataset()

        metadata_data: dict[str, Any] = {
            "open": np.asarray(open_arr, dtype=np.float32),
            "high": np.asarray(high_arr, dtype=np.float32),
            "low": np.asarray(low_arr, dtype=np.float32),
            "close": np.asarray(close_arr, dtype=np.float32),
        }
        if volume_arr is not None:
            metadata_data["volume"] = np.asarray(volume_arr, dtype=np.float32)
        metadata = _NumpyFrame(
            metadata_data,
            index=np.asarray(ts, dtype=np.int64),
            attrs={"symbol": str(symbol)} if symbol else None,
        )

        return PreparedDataset(
            X=np.asarray(X, dtype=np.float32),
            y=np.asarray(labels, dtype=np.int8),
            index=np.asarray(ts, dtype=np.int64),
            feature_names=list(cols),
            metadata=metadata,
            labels=np.asarray(labels, dtype=np.int8),
        )

    def _prepare_rust_features(
        self,
        *,
        news_features: Any | None = None,
        symbol: str | None = None,
    ) -> PreparedDataset | None:
        if not symbol:
            return None

        root = str(getattr(self.settings.system, "data_dir", "data") or "data")
        try:
            import forex_bindings  # type: ignore
        except Exception as exc:
            _disable_rust_features_backend()
            logger.error("Rust feature bindings unavailable; feature preparation is blocked: %s", exc)
            return None
        if not hasattr(forex_bindings, "load_symbol_features"):
            _disable_rust_features_backend()
            return None

        base_tf = str(getattr(self.settings.system, "base_timeframe", "M1") or "M1").upper()
        all_tfs = self._resolved_timeframes(base_tf)
        higher = [tf for tf in all_tfs if tf != base_tf]

        include_raw = str(os.environ.get("FOREX_BOT_RUST_INCLUDE_RAW", "1") or "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        resample_missing = str(os.environ.get("FOREX_BOT_RUST_RESAMPLE", "1") or "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        arrow_tensor = str(os.environ.get("FOREX_BOT_RUST_ARROW", "1") or "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        def _parse_pos_int_env(name: str, default: int = 0) -> int:
            raw = os.environ.get(name)
            if raw is None or str(raw).strip() == "":
                return int(default)
            try:
                value = int(str(raw).strip())
            except Exception:
                return int(default)
            return int(max(0, value))

        use_all_features = str(os.environ.get("FOREX_BOT_USE_ALL_FEATURES", "0") or "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        profile_raw = str(os.environ.get("FOREX_BOT_RUST_FEATURE_PROFILE", "auto") or "auto").strip().lower()
        valid_profiles = {"full", "core", "compact"}
        if use_all_features:
            feature_profile = "full"
        elif profile_raw in valid_profiles:
            feature_profile = profile_raw
        else:
            feature_profile = "full"
            with contextlib.suppress(Exception):
                import psutil

                total_gb = float(psutil.virtual_memory().total) / float(1024**3)
                if total_gb <= 96.0 and (len(all_tfs) >= 4 or self._tf_minutes(base_tf) <= 5):
                    feature_profile = "core"

        htf_profile_raw = str(os.environ.get("FOREX_BOT_RUST_HTF_FEATURE_PROFILE", "auto") or "auto").strip().lower()
        if use_all_features:
            htf_profile = "full"
        elif htf_profile_raw in valid_profiles:
            htf_profile = htf_profile_raw
        elif htf_profile_raw in {"auto", "", "detect"}:
            htf_profile = "compact" if feature_profile in {"core", "compact"} else feature_profile
        else:
            htf_profile = feature_profile

        default_max_features = 0
        if use_all_features:
            default_max_features = 0
        elif feature_profile == "core":
            default_max_features = 96
        elif feature_profile == "compact":
            default_max_features = 64
        max_features = _parse_pos_int_env("FOREX_BOT_RUST_MAX_FEATURES", default_max_features)

        default_max_htf = 0
        if use_all_features:
            default_max_htf = 0
        elif htf_profile == "core":
            default_max_htf = 24
        elif htf_profile == "compact":
            default_max_htf = 12
        max_htf_features = _parse_pos_int_env("FOREX_BOT_RUST_MAX_HTF_FEATURES", default_max_htf)
        tail_rows = _parse_pos_int_env(
            "FOREX_BOT_RUST_TAIL_ROWS",
            _parse_pos_int_env("FOREX_BOT_GLOBAL_MAX_ROWS_PER_SYMBOL", 0),
        )
        logger.info(
            "Rust feature profile: symbol=%s base=%s(%s) htf=%s(%s) tail_rows=%s",
            symbol,
            feature_profile,
            max_features if max_features > 0 else "unlimited",
            htf_profile,
            max_htf_features if max_htf_features > 0 else "unlimited",
            tail_rows if tail_rows > 0 else "all",
        )

        cache_dir = str(getattr(self.settings.system, "cache_dir", "cache") or "cache")
        cache_enabled = bool(getattr(self.settings.system, "cache_enabled", False))
        cache_override = os.environ.get("FOREX_BOT_RUST_FEATURE_CACHE")
        if cache_override is not None and str(cache_override).strip() != "":
            cache_enabled = str(cache_override).strip().lower() in {"1", "true", "yes", "on"}
        try:
            cache_ttl = int(getattr(self.settings.system, "cache_max_age_minutes", 0) or 0)
        except Exception:
            cache_ttl = 0

        try:
            payload = forex_bindings.load_symbol_features(
                root=root,
                symbol=symbol,
                base_tf=base_tf,
                higher_tfs=higher or None,
                include_raw=include_raw,
                cache_dir=cache_dir,
                cache_ttl_minutes=cache_ttl,
                cache_enabled=cache_enabled,
                resample_missing=resample_missing,
                arrow_tensor=arrow_tensor,
                feature_profile=feature_profile,
                htf_feature_profile=htf_profile,
                max_features=max_features,
                max_htf_features=max_htf_features,
                tail_rows=tail_rows,
            )
        except TypeError:
            try:
                payload = forex_bindings.load_symbol_features(
                    root=root,
                    symbol=symbol,
                    base_tf=base_tf,
                    higher_tfs=higher or None,
                    include_raw=include_raw,
                    cache_dir=cache_dir,
                    cache_ttl_minutes=cache_ttl,
                    cache_enabled=cache_enabled,
                    resample_missing=resample_missing,
                    arrow_tensor=arrow_tensor,
                )
            except TypeError:
                payload = forex_bindings.load_symbol_features(
                    root=root,
                    symbol=symbol,
                    base_tf=base_tf,
                    higher_tfs=higher or None,
                    include_raw=include_raw,
                    cache_dir=cache_dir,
                    cache_ttl_minutes=cache_ttl,
                    cache_enabled=cache_enabled,
                    resample_missing=resample_missing,
                )
        except Exception as exc:
            _disable_rust_features_backend()
            logger.error("Rust feature load failed; feature preparation is blocked: %s", exc)
            return None

        try:
            feature_names = list(payload.get("feature_names") or [])
            features_obj = payload.get("features")
            arrow_obj = payload.get("features_arrow_tensor")
            if arrow_obj is not None:
                with contextlib.suppress(Exception):
                    features_obj = arrow_obj.to_numpy()
            features = np.asarray(features_obj, dtype=np.float32)
            rows_hint = 0
            with contextlib.suppress(Exception):
                rows_hint = int(len(payload.get("timestamps") or []))
            # Some Arrow paths expose transposed 2D tensors (features, rows).
            # Align to canonical shape: (rows, features).
            if (
                features.ndim == 2
                and rows_hint > 0
                and features.shape[0] != rows_hint
                and features.shape[1] == rows_hint
            ):
                features = features.T
            if (
                features.ndim == 2
                and feature_names
                and features.shape[1] != len(feature_names)
                and features.shape[0] == len(feature_names)
            ):
                features = features.T
        except Exception as exc:
            logger.error("Rust feature payload malformed; feature preparation is blocked: %s", exc)
            return None

        if features.ndim != 2 or not feature_names or features.shape[1] != len(feature_names):
            logger.error("Rust feature payload shape mismatch; feature preparation is blocked.")
            return None

        # Pandas fallback removed: always use numpy/Rust prepared dataset.
        if news_features is not None:
            with contextlib.suppress(Exception):
                if hasattr(news_features, "empty") and not news_features.empty:
                    logger.info("Pandas-free rust path: news feature merge is skipped.")
        return self._prepare_rust_features_numpy(
            payload=payload,
            features=features,
            feature_names=feature_names,
            symbol=symbol,
        )

    def prepare(
        self,
        frames: dict[str, Any],
        *,
        news_features: Any | None = None,
        symbol: str | None = None,
    ) -> PreparedDataset:
        use_rust = self._use_rust_backend()
        if not use_rust:
            logger.error("Rust feature path requires rust backend.")
            return self._empty_numpy_dataset()
        if use_rust and frames:
            sources = {
                str(getattr(df, "attrs", {}).get("source", "")).strip().lower()
                for df in (frames or {}).values()
                if df is not None
            }
            if not any(sources):
                logger.error("Rust feature path cannot use in-memory Python frames.")
                return self._empty_numpy_dataset()
            live_source = False
            try:
                for df in (frames or {}).values():
                    if getattr(df, "attrs", {}).get("source") == "mt5":
                        live_source = True
                        break
            except Exception:
                live_source = False
            if live_source:
                logger.error("Rust feature path cannot use live MT5 Python frames.")
                return self._empty_numpy_dataset()
        rust_ds = self._prepare_rust_features(news_features=news_features, symbol=symbol)
        if rust_ds is not None:
            return rust_ds
        logger.error("Rust feature preparation failed; feature pipeline is blocked.")
        return self._empty_numpy_dataset()


__all__ = [
    "FeatureEngineer",
    "_compute_adx_numba",
]


