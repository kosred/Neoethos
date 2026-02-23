from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..core.config import Settings
from ..domain.events import PreparedDataset

logger = logging.getLogger(__name__)
_RUST_FEATURES_BACKEND_OK: bool | None = None
_RUST_FEATURES_WARNED_UNAVAILABLE = False
_RUST_LABELS_BACKEND_OK: bool | None = None
_RUST_LABELS_WARNED_UNAVAILABLE = False


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
            "Rust labels backend requested but forex_bindings.triple_barrier_labels is unavailable; using Python labels."
        )
        _RUST_LABELS_WARNED_UNAVAILABLE = True
    return bool(_RUST_LABELS_BACKEND_OK)


def _disable_rust_labels_backend() -> None:
    global _RUST_LABELS_BACKEND_OK
    _RUST_LABELS_BACKEND_OK = False


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
        tfs = sorted(set(tfs), key=self._tf_minutes)
        if base_tf not in tfs:
            tfs.insert(0, base_tf)
        return tfs

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
        out["obi_mom3"] = out["vol_imbalance"].rolling(3, min_periods=1).mean().fillna(0.0)
        out["obi_seq_up5"] = (out["vol_imbalance"] > 0).astype(float).rolling(5, min_periods=1).mean().fillna(0.0)
        out["obi_seq_dn5"] = (out["vol_imbalance"] < 0).astype(float).rolling(5, min_periods=1).mean().fillna(0.0)
        return out

    def _compute_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
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
            day = pd.Series(idx_utc.date, index=out.index)
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
        available: set[str],
        payload_symbol: str,
        payload_tf: str,
        TALibStrategyGene: Any,
    ) -> Any | None:
        def _to_float(key: str, default: float) -> float:
            try:
                return float(raw.get(key, default) or default)
            except Exception:
                return float(default)

        inds_raw = raw.get("indicators") or []
        indicators: list[str] = []
        for ind in inds_raw:
            name = str(ind).strip().upper()
            if name and name in available and name not in indicators:
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
                use_ob=bool(raw.get("use_ob", False)),
                use_fvg=bool(raw.get("use_fvg", False)),
                use_liq_sweep=bool(raw.get("use_liq_sweep", False)),
                mtf_confirmation=bool(raw.get("mtf_confirmation", False)),
                use_premium_discount=bool(raw.get("use_premium_discount", False)),
                use_inducement=bool(raw.get("use_inducement", False)),
                tp_pips=_to_float("tp_pips", 40.0),
                sl_pips=_to_float("sl_pips", 20.0),
                source_symbol=payload_symbol,
                source_timeframe=payload_tf,
            )
        except Exception:
            return None

    def _load_discovered_base_signal_genes(self, symbol: str | None, max_genes: int = 100) -> list[Any]:
        try:
            from .talib_mixer import TALIB_AVAILABLE, TALibStrategyGene, TALibStrategyMixer
        except Exception:
            return []
        if not TALIB_AVAILABLE:
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

        mixer = TALibStrategyMixer(
            device="cpu",
            use_volume_features=bool(getattr(self.settings.system, "use_volume_features", False)),
        )
        available = {str(i).upper() for i in getattr(mixer, "available_indicators", [])}
        if not available:
            return []

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
        strict_prefilter = str(os.environ.get("FOREX_BOT_PROP_BASE_SIGNAL_STRICT_FILTER", "0") or "0").strip().lower() in {
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
                    if dd <= max_dd and profit > min_profit and trades >= min_trades:
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

    def _compute_discovered_base_signal(self, df: pd.DataFrame, *, symbol: str | None) -> np.ndarray | None:
        if df is None or df.empty:
            return None
        try:
            from .talib_mixer import TALIB_AVAILABLE, TALibStrategyMixer
        except Exception:
            return None
        if not TALIB_AVAILABLE:
            return None

        try:
            max_genes = int(os.environ.get("FOREX_BOT_PROP_BASE_SIGNAL_GENES", "100") or 100)
        except Exception:
            max_genes = 100
        max_genes = max(1, max_genes)

        genes = self._load_discovered_base_signal_genes(symbol, max_genes=max_genes)
        if not genes:
            return None

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

        mixer = TALibStrategyMixer(
            device="cpu",
            use_volume_features=bool(getattr(self.settings.system, "use_volume_features", False)),
        )
        cache = mixer.bulk_calculate_indicators(df, genes)

        score_sum = np.zeros(len(df), dtype=np.float64)
        weight_sum = 0.0
        for gene in genes:
            try:
                sig = mixer.compute_signals(df, gene, cache=cache)
                aligned = sig.reindex(df.index).ffill().fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            except Exception:
                continue

            w = float(getattr(gene, "fitness", 0.0) or 0.0)
            if not np.isfinite(w) or w <= 0.0:
                w = 1.0
            score_sum += w * aligned
            weight_sum += abs(w)

        if weight_sum <= 0.0:
            return None

        score = score_sum / weight_sum
        abs_score = np.abs(score)
        signal = np.where(score >= threshold, 1, np.where(score <= -threshold, -1, 0)).astype(np.int8)
        n = int(len(signal))
        if n <= 0:
            return signal

        target_min = int(round(min_coverage * n))
        target_max = int(round(max_coverage * n))
        target_min = max(0, min(target_min, n))
        target_max = max(target_min, min(target_max, n))

        active_now = int(np.count_nonzero(signal))
        if target_max > 0 and active_now > target_max:
            keep = np.zeros(n, dtype=bool)
            top_idx = np.argsort(abs_score)[::-1][:target_max]
            keep[top_idx] = True
            trimmed = np.zeros(n, dtype=np.int8)
            sel = score[top_idx]
            trimmed[top_idx] = np.where(sel > 0.0, 1, np.where(sel < 0.0, -1, 0)).astype(np.int8)
            signal = trimmed
        elif target_min > 0 and active_now < target_min:
            keep = np.zeros(n, dtype=bool)
            top_idx = np.argsort(abs_score)[::-1][:target_min]
            keep[top_idx] = True
            boosted = np.zeros(n, dtype=np.int8)
            sel = score[top_idx]
            boosted[top_idx] = np.where(sel > 0.0, 1, np.where(sel < 0.0, -1, 0)).astype(np.int8)
            signal = boosted
        return signal

    def _compute_base_signal(self, df: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()

        signal_source = str(os.environ.get("FOREX_BOT_BASE_SIGNAL_SOURCE", "discovery_first") or "discovery_first").strip().lower()
        use_discovery = signal_source in {"discovery", "discovery_first", "prop", "talib", "mixer", "auto"}
        strict_discovery = signal_source in {"discovery", "prop", "talib", "mixer"}

        target_symbol = symbol
        if not target_symbol:
            target_symbol = str(getattr(out, "attrs", {}).get("symbol", "") or "").strip()
        discovered_signal = None
        if use_discovery:
            discovered_signal = self._compute_discovered_base_signal(out, symbol=target_symbol)
        if discovered_signal is not None:
            cov = float(np.count_nonzero(discovered_signal)) / float(max(1, len(discovered_signal)))
            try:
                min_cov = float(os.environ.get("FOREX_BOT_DISCOVERY_BASE_SIGNAL_MIN_COVERAGE", "0.005") or 0.005)
            except Exception:
                min_cov = 0.005
            if signal_source == "discovery_first" and cov < max(0.0, min(1.0, min_cov)):
                logger.warning(
                    "Discovery base_signal coverage too low (%.3f%% < %.3f%%); falling back to classic rules.",
                    cov * 100.0,
                    min_cov * 100.0,
                )
            else:
                out["base_signal"] = discovered_signal.astype(int, copy=False)
                return out
        if strict_discovery:
            out["base_signal"] = 0
            return out

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

    def _compute_labels(
        self,
        close: pd.Series,
        cfg: _LabelConfig,
        *,
        high: pd.Series | None = None,
        low: pd.Series | None = None,
        symbol: str | None = None,
        base_signal: pd.Series | None = None,
    ) -> pd.Series:
        close_f = close.astype(float)
        if not cfg.use_triple_barrier:
            future = close_f.shift(-cfg.horizon)
            delta = (future - close_f).astype(float)
            up = delta > cfg.min_dist
            down = delta < -cfg.min_dist
            labels = np.where(up, 1, np.where(down, -1, 0)).astype(int)
            return pd.Series(labels, index=close_f.index)

        hi = (high.astype(float) if high is not None else close_f).reindex(close_f.index)
        lo = (low.astype(float) if low is not None else close_f).reindex(close_f.index)
        n = len(close_f)
        if n <= 2:
            return pd.Series(np.zeros(n, dtype=int), index=close_f.index)

        pip_size = self._infer_pip_size(symbol)
        sl_pips = cfg.sl_pips if (cfg.sl_pips is not None and cfg.sl_pips > 0) else None
        tp_pips = cfg.tp_pips if (cfg.tp_pips is not None and cfg.tp_pips > 0) else None
        rr = float(getattr(self.settings.risk, "min_risk_reward", 2.0) or 2.0)
        atr_mult = float(getattr(self.settings.risk, "atr_stop_multiplier", 1.5) or 1.5)

        atr = _compute_atr(hi, lo, close_f, period=max(2, int(getattr(self.settings.risk, "atr_period", 14) or 14)))
        atr = atr.ffill().fillna(0.0)

        if sl_pips is not None:
            sl_dist = np.full(n, max(0.0, sl_pips * pip_size), dtype=np.float64)
        else:
            sl_dist = np.maximum(np.asarray(atr, dtype=np.float64) * max(0.1, atr_mult), cfg.min_dist)

        if tp_pips is not None:
            tp_dist = np.full(n, max(0.0, tp_pips * pip_size), dtype=np.float64)
        else:
            tp_dist = np.maximum(sl_dist * max(0.1, rr), cfg.min_dist)

        close_arr = np.asarray(close_f, dtype=np.float64)
        high_arr = np.asarray(hi, dtype=np.float64)
        low_arr = np.asarray(lo, dtype=np.float64)
        sig_arr: np.ndarray | None = None
        if base_signal is not None:
            try:
                sig_arr = np.asarray(base_signal.reindex(close_f.index).fillna(0).astype(int), dtype=np.int8)
            except Exception:
                sig_arr = None

        max_hold = int(max(cfg.horizon, cfg.max_hold))
        max_hold = max(1, max_hold)

        if _rust_labels_backend_available():
            try:
                import forex_bindings  # type: ignore

                sig_arg = sig_arr.astype(np.int8, copy=False) if sig_arr is not None else None
                labels_rs = np.asarray(
                    forex_bindings.triple_barrier_labels(
                        close_arr,
                        high_arr,
                        low_arr,
                        sl_dist,
                        tp_dist,
                        int(max_hold),
                        sig_arg,
                    ),
                    dtype=np.int8,
                )
                if labels_rs.shape[0] == n:
                    return pd.Series(labels_rs.astype(int, copy=False), index=close_f.index)
                logger.debug(
                    "Rust triple-barrier labels shape mismatch (got=%s expected=%s); falling back to Python.",
                    labels_rs.shape,
                    n,
                )
            except Exception as exc:
                _disable_rust_labels_backend()
                logger.debug("Rust triple-barrier labels failed; falling back to Python: %s", exc)

        labels = np.zeros(n, dtype=np.int8)

        for i in range(n):
            if not np.isfinite(close_arr[i]):
                continue
            j_end = min(n - 1, i + max_hold)
            if j_end <= i:
                continue

            entry = close_arr[i]
            s = int(sig_arr[i]) if sig_arr is not None else 0
            sd = float(max(sl_dist[i], cfg.min_dist))
            td = float(max(tp_dist[i], cfg.min_dist))
            if sd <= 0.0 and td <= 0.0:
                continue

            if s > 0:
                tp_lvl = entry + td
                sl_lvl = entry - sd
                out = 0
                for j in range(i + 1, j_end + 1):
                    hit_tp = high_arr[j] >= tp_lvl
                    hit_sl = low_arr[j] <= sl_lvl
                    if hit_tp and hit_sl:
                        out = 1 if close_arr[j] >= entry else -1
                        break
                    if hit_tp:
                        out = 1
                        break
                    if hit_sl:
                        out = -1
                        break
                labels[i] = np.int8(out)
            elif s < 0:
                tp_lvl = entry - td
                sl_lvl = entry + sd
                out = 0
                for j in range(i + 1, j_end + 1):
                    hit_tp = low_arr[j] <= tp_lvl
                    hit_sl = high_arr[j] >= sl_lvl
                    if hit_tp and hit_sl:
                        out = 1 if close_arr[j] <= entry else -1
                        break
                    if hit_tp:
                        out = 1
                        break
                    if hit_sl:
                        out = -1
                        break
                labels[i] = np.int8(out)
            else:
                up_lvl = entry + td
                dn_lvl = entry - sd
                out = 0
                for j in range(i + 1, j_end + 1):
                    hit_up = high_arr[j] >= up_lvl
                    hit_dn = low_arr[j] <= dn_lvl
                    if hit_up and hit_dn:
                        out = 1 if close_arr[j] >= entry else -1
                        break
                    if hit_up:
                        out = 1
                        break
                    if hit_dn:
                        out = -1
                        break
                labels[i] = np.int8(out)

        return pd.Series(labels.astype(int), index=close_f.index)

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
            logger.warning("Rust feature load failed; falling back to Python: %s", exc)
            return None

        try:
            feature_names = list(payload.get("feature_names") or [])
            features_obj = payload.get("features")
            arrow_obj = payload.get("features_arrow_tensor")
            if arrow_obj is not None:
                with contextlib.suppress(Exception):
                    features_obj = arrow_obj.to_numpy()
            features = np.asarray(features_obj, dtype=np.float32)
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
                tmp = self._compute_base_signal(tmp, symbol=symbol)
                sig = tmp.get("base_signal")
                if sig is not None:
                    X["base_signal"] = sig.reindex(X.index).fillna(0).astype(int)
            except Exception:
                X["base_signal"] = 0

        feature_blocks: list[pd.DataFrame] = [X]
        try:
            session_src = base_df[["high", "low", "close"]]
            session_feats = self._compute_session_features(session_src)
            new_cols = [c for c in session_feats.columns if c not in {"high", "low", "close"} and c not in X.columns]
            if new_cols:
                feature_blocks.append(session_feats[new_cols].reindex(X.index).fillna(0.0))
        except Exception:
            pass

        if news_features is not None and not news_features.empty:
            nf = _ensure_datetime_index(news_features)
            nf = nf.reindex(X.index, method="ffill").fillna(0.0)
            nf = nf.rename(columns={c: f"news_{c}" for c in nf.columns})
            feature_blocks.append(nf)

        if len(feature_blocks) > 1:
            X = pd.concat(feature_blocks, axis=1, copy=False)

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        label_cfg = self._label_config()
        labels = self._compute_labels(
            base_df["close"].astype(float),
            label_cfg,
            high=base_df["high"].astype(float) if "high" in base_df.columns else None,
            low=base_df["low"].astype(float) if "low" in base_df.columns else None,
            symbol=symbol,
            base_signal=X["base_signal"] if "base_signal" in X.columns else None,
        )
        trim = int(max(label_cfg.horizon, label_cfg.max_hold if label_cfg.use_triple_barrier else label_cfg.horizon))
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
        use_rust = self._use_rust_backend()
        if use_rust and frames:
            # If caller passed in-memory/manual frames (no source marker), honor them directly.
            # This keeps tests and custom pipelines deterministic and avoids unexpected disk reloads.
            sources = {
                str(getattr(df, "attrs", {}).get("source", "")).strip().lower()
                for df in (frames or {}).values()
                if df is not None
            }
            if not any(sources):
                use_rust = False
        if use_rust:
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

        base_tf = str(getattr(self.settings.system, "base_timeframe", "M1") or "M1").upper()
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
        features = self._compute_session_features(features)
        features = self._compute_base_signal(features, symbol=symbol)

        base_blocks: list[pd.DataFrame] = [features]
        if news_features is not None and not news_features.empty:
            nf = _ensure_datetime_index(news_features)
            nf = nf.reindex(features.index, method="ffill").fillna(0.0)
            nf = nf.rename(columns={c: f"news_{c}" for c in nf.columns})
            base_blocks.append(nf)
        if len(base_blocks) > 1:
            features = pd.concat(base_blocks, axis=1, copy=False)

        all_tfs = self._resolved_timeframes(base_tf)
        prefix_base = bool(getattr(self.settings.system, "multi_resolution_prefix_base", False))
        if prefix_base:
            base_pref = features.add_prefix(f"{base_tf}_")
        else:
            base_pref = None

        aligned_higher: list[pd.DataFrame] = []
        for tf in all_tfs:
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
            htf = self._compute_session_features(htf)
            htf_shifted = htf.shift(1)
            htf_shifted.columns = [f"{tf}_{c}" for c in htf_shifted.columns]
            aligned = htf_shifted.reindex(features.index, method="ffill").fillna(0.0)
            aligned_higher.append(aligned)

        final_blocks: list[pd.DataFrame] = [features]
        if base_pref is not None:
            final_blocks.append(base_pref)
        if aligned_higher:
            final_blocks.extend(aligned_higher)
        if len(final_blocks) > 1:
            features = pd.concat(final_blocks, axis=1, copy=False)

        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        label_cfg = self._label_config()
        labels = self._compute_labels(
            base_df["close"].astype(float),
            label_cfg,
            high=base_df["high"].astype(float) if "high" in base_df.columns else None,
            low=base_df["low"].astype(float) if "low" in base_df.columns else None,
            symbol=symbol,
            base_signal=features["base_signal"] if "base_signal" in features.columns else None,
        )
        trim = int(max(label_cfg.horizon, label_cfg.max_hold if label_cfg.use_triple_barrier else label_cfg.horizon))
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
