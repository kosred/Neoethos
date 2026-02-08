from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ..core.config import Settings
from ..domain.events import PreparedDataset

logger = logging.getLogger(__name__)
_RUST_FEATURES_BACKEND_OK: bool | None = None
_RUST_FEATURES_WARNED_UNAVAILABLE = False


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
            "Rust features backend requested but forex_bindings.load_symbol_features is unavailable; using Python features."
        )
        _RUST_FEATURES_WARNED_UNAVAILABLE = True
    return bool(_RUST_FEATURES_BACKEND_OK)


def _disable_rust_features_backend() -> None:
    global _RUST_FEATURES_BACKEND_OK
    _RUST_FEATURES_BACKEND_OK = False


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        ts_col = None
        for candidate in ("timestamp", "time", "datetime", "date"):
            if candidate in out.columns:
                ts_col = candidate
                break
        if ts_col is not None:
            idx = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
            out = out.set_index(idx)
        else:
            out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    else:
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
    return out


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0.0, 0.0)
    loss = -delta.where(delta < 0.0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _compute_macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = _ema(series, 12)
    ema26 = _ema(series, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    return atr.bfill().fillna(0.0)


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


class FeatureEngineer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

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
        return _LabelConfig(horizon=max(1, horizon), min_dist=max(0.0, min_dist))

    def _compute_basic_features(self, df: pd.DataFrame, *, use_gpu: bool = False) -> pd.DataFrame:
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

    def _compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _compute_volume_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        if "volume" in out.columns:
            volume = out["volume"].astype(float)
        else:
            volume = pd.Series(np.ones(len(out), dtype=float), index=out.index)
        close = out["close"].astype(float)
        window = 20
        vol_sum = volume.rolling(window, min_periods=1).sum().replace(0.0, np.nan)
        poc = (close * volume).rolling(window, min_periods=1).sum() / vol_sum
        poc = poc.bfill().fillna(close)
        out["dist_to_poc"] = (close - poc).fillna(0.0)
        std = close.rolling(window, min_periods=1).std().fillna(0.0)
        out["in_value_area"] = ((close >= (poc - std)) & (close <= (poc + std))).astype(float)
        return out

    def _compute_obi_features(self, df: pd.DataFrame, *, use_gpu: bool = False) -> pd.DataFrame:
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
            volume = pd.Series(np.ones(len(out), dtype=float), index=out.index)
        imbalance = ((close - open_) / rng).fillna(0.0) * volume
        out["vol_imbalance"] = imbalance.fillna(0.0)
        return out

    def _compute_base_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        rsi = out.get("rsi")
        macd_hist = out.get("macd_hist")
        if rsi is None or macd_hist is None:
            out["base_signal"] = 0
            return out
        buy = (rsi < 30.0) & (macd_hist > 0)
        sell = (rsi > 70.0) & (macd_hist < 0)
        signal = np.where(buy, 1, np.where(sell, -1, 0))
        out["base_signal"] = signal.astype(int)
        return out

    def _compute_labels(self, close: pd.Series, cfg: _LabelConfig) -> pd.Series:
        future = close.shift(-cfg.horizon)
        delta = (future - close).astype(float)
        up = delta > cfg.min_dist
        down = delta < -cfg.min_dist
        labels = np.where(up, 1, np.where(down, -1, 0)).astype(int)
        return pd.Series(labels, index=close.index)

    def _prepare_rust_features(
        self,
        *,
        news_features: pd.DataFrame | None = None,
        symbol: str | None = None,
    ) -> PreparedDataset | None:
        if not symbol:
            return None

        root = str(getattr(self.settings.system, "data_dir", "data") or "data")
        try:
            import forex_bindings  # type: ignore
        except Exception as exc:
            _disable_rust_features_backend()
            logger.warning("Rust bindings unavailable; falling back to Python features: %s", exc)
            return None
        if not hasattr(forex_bindings, "load_symbol_features"):
            _disable_rust_features_backend()
            return None

        base_tf = str(getattr(self.settings.system, "base_timeframe", "M1") or "M1")
        higher = list(getattr(self.settings.system, "higher_timeframes", []) or [])
        required = list(getattr(self.settings.system, "required_timeframes", []) or [])
        for tf in required:
            if tf != base_tf and tf not in higher:
                higher.append(tf)

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
            )
        except Exception as exc:
            _disable_rust_features_backend()
            logger.warning("Rust feature load failed; falling back to Python: %s", exc)
            return None

        try:
            feature_names = list(payload.get("feature_names") or [])
            features = np.asarray(payload.get("features"), dtype=np.float32)
        except Exception as exc:
            logger.warning("Rust feature payload malformed; falling back: %s", exc)
            return None

        if features.ndim != 2 or not feature_names or features.shape[1] != len(feature_names):
            logger.warning("Rust feature payload shape mismatch; falling back.")
            return None

        def _to_index(values: object) -> pd.Index:
            if values is None:
                return pd.RangeIndex(features.shape[0])
            arr = np.asarray(values)
            if arr.size == 0:
                return pd.RangeIndex(0)
            try:
                return pd.to_datetime(arr.astype("int64", copy=False), utc=True)
            except Exception:
                return pd.to_datetime(arr, utc=True, errors="coerce")

        idx = _to_index(payload.get("timestamps"))
        X = pd.DataFrame(features, columns=feature_names, index=idx)

        base_ts = payload.get("base_timestamps") or payload.get("timestamps")
        base_idx = _to_index(base_ts)
        base_df = pd.DataFrame(
            {
                "open": np.asarray(payload.get("open"), dtype=np.float64),
                "high": np.asarray(payload.get("high"), dtype=np.float64),
                "low": np.asarray(payload.get("low"), dtype=np.float64),
                "close": np.asarray(payload.get("close"), dtype=np.float64),
            },
            index=base_idx,
        )
        if "volume" in payload:
            base_df["volume"] = np.asarray(payload.get("volume"), dtype=np.float64)

        base_df = base_df.reindex(X.index, method="ffill").fillna(0.0)
        try:
            if symbol:
                base_df.attrs["symbol"] = symbol
        except Exception:
            pass

        use_base_signal = str(os.environ.get("FOREX_BOT_BASE_SIGNAL", "1") or "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if use_base_signal and "base_signal" not in X.columns:
            try:
                tmp = self._compute_basic_features(base_df, use_gpu=False)
                tmp = self._compute_base_signal(tmp)
                sig = tmp.get("base_signal")
                if sig is not None:
                    X["base_signal"] = sig.reindex(X.index).fillna(0).astype(int)
            except Exception:
                X["base_signal"] = 0

        if news_features is not None and not news_features.empty:
            nf = _ensure_datetime_index(news_features)
            nf = nf.reindex(X.index, method="ffill").fillna(0.0)
            nf = nf.rename(columns={c: f"news_{c}" for c in nf.columns})
            X = pd.concat([X, nf], axis=1)

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        label_cfg = self._label_config()
        labels = self._compute_labels(base_df["close"].astype(float), label_cfg)
        trim = int(label_cfg.horizon)
        if trim > 0:
            X = X.iloc[:-trim]
            labels = labels.iloc[:-trim]
            meta = base_df.iloc[:-trim]
        else:
            meta = base_df

        return PreparedDataset(
            X=X,
            y=labels,
            index=X.index,
            feature_names=list(X.columns),
            metadata=meta,
            labels=labels,
        )

    def prepare(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        news_features: pd.DataFrame | None = None,
        symbol: str | None = None,
    ) -> PreparedDataset:
        if self._use_rust_backend():
            live_source = False
            try:
                for df in (frames or {}).values():
                    if getattr(df, "attrs", {}).get("source") == "mt5":
                        live_source = True
                        break
            except Exception:
                live_source = False
            if live_source:
                logger.info("Live MT5 frames detected; using Python feature pipeline.")
            else:
                rust_ds = self._prepare_rust_features(news_features=news_features, symbol=symbol)
                if rust_ds is not None:
                    return rust_ds

        if frames is None or len(frames) == 0:
            empty = pd.DataFrame()
            return PreparedDataset(X=empty, y=pd.Series(dtype=int), index=empty.index, feature_names=[], metadata=None, labels=None)

        base_tf = str(getattr(self.settings.system, "base_timeframe", "M1") or "M1")
        base_df = frames.get(base_tf)
        if base_df is None:
            base_df = frames.get("M1")
        if base_df is None:
            base_df = next(iter(frames.values()))
        base_df = _ensure_datetime_index(base_df)
        base_df = base_df.sort_index()
        try:
            if symbol:
                base_df.attrs["symbol"] = symbol
        except Exception:
            pass

        features = self._compute_basic_features(base_df, use_gpu=False)
        features = self._compute_volatility_features(features)
        features = self._compute_volume_profile_features(features)
        features = self._compute_obi_features(features, use_gpu=False)
        features = self._compute_base_signal(features)

        if news_features is not None and not news_features.empty:
            nf = _ensure_datetime_index(news_features)
            nf = nf.reindex(features.index, method="ffill").fillna(0.0)
            nf = nf.rename(columns={c: f"news_{c}" for c in nf.columns})
            features = pd.concat([features, nf], axis=1)

        higher = list(getattr(self.settings.system, "higher_timeframes", []) or [])
        required = list(getattr(self.settings.system, "required_timeframes", []) or [])
        for tf in required:
            if tf not in higher:
                higher.append(tf)

        for tf in higher:
            if tf == base_tf:
                continue
            htf = frames.get(tf)
            if htf is None or htf.empty:
                continue
            htf = _ensure_datetime_index(htf).sort_index()
            htf = self._compute_basic_features(htf, use_gpu=False)
            htf = self._compute_volatility_features(htf)
            htf = self._compute_volume_profile_features(htf)
            htf = self._compute_obi_features(htf, use_gpu=False)
            htf_shifted = htf.shift(1)
            htf_shifted.columns = [f"{tf}_{c}" for c in htf_shifted.columns]
            aligned = htf_shifted.reindex(features.index, method="ffill").fillna(0.0)
            features = features.join(aligned, how="left")

        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        label_cfg = self._label_config()
        labels = self._compute_labels(base_df["close"].astype(float), label_cfg)
        trim = int(label_cfg.horizon)
        if trim > 0:
            features = features.iloc[:-trim]
            labels = labels.iloc[:-trim]
            meta = base_df.iloc[:-trim]
        else:
            meta = base_df

        return PreparedDataset(
            X=features,
            y=labels,
            index=features.index,
            feature_names=list(features.columns),
            metadata=meta,
            labels=labels,
        )


__all__ = [
    "FeatureEngineer",
    "_compute_adx_numba",
]
