from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import talib
    from talib import abstract

    TALIB_AVAILABLE = True
    ALL_INDICATORS = sorted(talib.get_functions())
except Exception:
    talib = None  # type: ignore
    abstract = None  # type: ignore
    TALIB_AVAILABLE = False
    ALL_INDICATORS: list[str] = []

TALIB_INDICATORS: dict[str, list[str]] = {
    "momentum": ["RSI", "ADX", "MACD"],
    "overlap": ["SMA", "EMA"],
    "volatility": ["ATR", "NATR"],
}

try:
    import forex_bindings as _fb  # type: ignore

    _RUST_TALIB_MIXER = hasattr(_fb, "talib_bulk_signals_ohlcv")
except Exception:
    _fb = None
_RUST_TALIB_MIXER = False


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
    if values.size == 0:
        return values.astype(np.float64, copy=False)
    s = pd.Series(values, copy=False)
    hist_mean = s.expanding(min_periods=max(2, int(min_periods))).mean().shift(1)
    hist_std = s.expanding(min_periods=max(2, int(min_periods))).std(ddof=0).shift(1)
    z = (s - hist_mean) / hist_std.replace(0.0, np.nan)
    out = z.to_numpy(dtype=np.float64, copy=False)
    if not out.flags.writeable:
        out = out.copy()
    out[~np.isfinite(out)] = 0.0
    return np.tanh(out)


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
        self._rust_signal_cache: dict[tuple[Any, ...], pd.Series] = {}
        self._rust_signal_index: pd.Index | None = None

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

    def _try_rust_bulk_signal_cache(self, df: pd.DataFrame, population: list[TALibStrategyGene]) -> None:
        self._rust_signal_cache = {}
        self._rust_signal_index = None
        # Disabled by default: current Rust bulk mixer normalizes indicators using full-sample
        # statistics, which introduces look-ahead leakage for backtests.
        if not _env_bool("FOREX_BOT_TALIB_RUST_BULK_SIGNALS", False):
            return
        if not _RUST_TALIB_MIXER or _fb is None:
            return
        if df is None or df.empty or not population:
            return
        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(set(df.columns)):
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

        open_arr = np.asarray(df["open"], dtype=np.float64)
        high_arr = np.asarray(df["high"], dtype=np.float64)
        low_arr = np.asarray(df["low"], dtype=np.float64)
        close_arr = np.asarray(df["close"], dtype=np.float64)
        volume_arr = None
        if self.use_volume_features and "volume" in df.columns:
            volume_arr = np.asarray(df["volume"], dtype=np.float64)

        try:
            signals = np.asarray(
                _fb.talib_bulk_signals_ohlcv(
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
                ),
                dtype=np.int8,
            )
        except Exception as exc:
            logger.debug("Rust TALib bulk signals failed; fallback to Python mixer: %s", exc)
            return

        if signals.ndim != 2 or signals.shape[0] != len(df) or signals.shape[1] != len(eligible):
            logger.debug(
                "Rust TALib bulk signals shape mismatch (got=%s expected=(%s,%s)); fallback to Python mixer.",
                signals.shape,
                len(df),
                len(eligible),
            )
            return

        idx = df.index
        for col_idx, (key, _gene) in enumerate(eligible):
            self._rust_signal_cache[key] = pd.Series(
                signals[:, col_idx].astype(np.float64, copy=False),
                index=idx,
            )
        self._rust_signal_index = idx

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

    def _compute_indicator(self, df: pd.DataFrame, indicator: str, params: dict[str, Any] | None) -> pd.Series:
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
        if isinstance(output, pd.DataFrame):
            return output.iloc[:, 0]
        if isinstance(output, pd.Series):
            return output
        return pd.Series(np.asarray(output), index=df.index)

    def bulk_calculate_indicators(self, df: pd.DataFrame, population: list[TALibStrategyGene]) -> dict[str, pd.Series]:
        cache: dict[str, pd.Series] = {}
        self._rust_signal_cache = {}
        self._rust_signal_index = None
        if df is None or df.empty:
            return cache
        if not population:
            return cache
        self._try_rust_bulk_signal_cache(df, population)
        # Build a first-seen parameter map once, avoiding O(indicators * population) scans.
        params_by_indicator: dict[str, dict[str, Any]] = {}
        for gene in population:
            if self._gene_key(gene) in self._rust_signal_cache:
                continue
            if not gene.params:
                continue
            for key, value in gene.params.items():
                norm = _normalize_indicator_name(key)
                if norm and norm not in params_by_indicator:
                    params_by_indicator[norm] = value
        needed: set[str] = set()
        for gene in population:
            if self._gene_key(gene) in self._rust_signal_cache:
                continue
            for ind in gene.indicators:
                norm = _normalize_indicator_name(ind)
                if norm:
                    needed.add(norm)
        for ind in needed:
            try:
                params = params_by_indicator.get(ind)
                cache[ind] = self._compute_indicator(df, ind, params)
            except Exception as exc:
                logger.debug("Indicator %s failed: %s", ind, exc)
                cache[ind] = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
        return cache

    def compute_signals(
        self,
        df: pd.DataFrame,
        gene: TALibStrategyGene,
        *,
        cache: dict[str, pd.Series] | None = None,
    ) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(np.zeros(0, dtype=float), index=df.index)
        indicators = [_normalize_indicator_name(i) for i in gene.indicators]
        if not indicators:
            return pd.Series(np.zeros(len(df), dtype=float), index=df.index)
        rust_key = self._gene_key(gene)
        if rust_key in self._rust_signal_cache:
            cached = self._rust_signal_cache[rust_key]
            try:
                if self._rust_signal_index is not None and cached.index.equals(df.index):
                    return cached
                return cached.reindex(df.index).fillna(0.0)
            except Exception:
                pass

        votes = np.zeros(len(df), dtype=np.float64)
        weight_total = 0.0
        causal_min_bars = max(2, _env_int("FOREX_BOT_TALIB_CAUSAL_MIN_BARS", 30))

        for ind in indicators:
            if not ind:
                continue
            series = None
            if cache is not None:
                series = cache.get(ind)
            if series is None:
                try:
                    params = None
                    if gene.params:
                        params = gene.params.get(ind)
                        if params is None:
                            params = gene.params.get(ind.lower())
                        if params is None:
                            params = gene.params.get(ind.upper())
                    series = self._compute_indicator(df, ind, params)
                except Exception as exc:
                    logger.debug("Indicator %s compute failed: %s", ind, exc)
                    series = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
            arr = np.asarray(series, dtype=np.float64)
            score = _causal_tanh_zscore(arr, min_periods=causal_min_bars)
            w = float(gene.weights.get(ind, 1.0)) if gene.weights else 1.0
            votes += w * score
            weight_total += abs(w)

        if weight_total <= 0.0:
            weight_total = 1.0
        combined = votes / weight_total
        long_thr = float(gene.long_threshold)
        short_thr = float(gene.short_threshold)
        signals = np.where(combined > long_thr, 1.0, np.where(combined < short_thr, -1.0, 0.0))
        return pd.Series(signals, index=df.index)

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
]
