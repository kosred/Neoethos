from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..features.talib_mixer import TALibStrategyGene, TALibStrategyMixer, signal_to_numpy
from .base import ExpertModel
from ..strategy.fast_backtest import batch_evaluate_strategies, infer_pip_metrics

logger = logging.getLogger(__name__)


def _is_dataframe(obj: Any) -> bool:
    return bool(hasattr(obj, "columns") and hasattr(obj, "index") and hasattr(obj, "__getitem__"))


@dataclass(slots=True)
class GeneticStrategyExpert(ExpertModel):
    """
    Genetic Algorithm for evolving TA-Lib indicator combinations.
    Uses OHLC data (metadata) to generate signals, not just pre-computed features.
    """

    population_size: int = 50
    generations: int = 10
    # 0 = allow full indicator set (no hard cap)
    max_indicators: int = 0
    best_gene: TALibStrategyGene | None = None
    mixer: TALibStrategyMixer | None = None
    portfolio: list[TALibStrategyGene] = field(default_factory=list)

    def __post_init__(self):
        self.mixer = TALibStrategyMixer()
        if not self.mixer.available_indicators:
            return
        max_indicators = 0
        env_max = os.environ.get("FOREX_BOT_PROP_SEARCH_MAX_INDICATORS") or os.environ.get(
            "FOREX_BOT_DISCOVERY_MAX_INDICATORS"
        )
        if env_max:
            try:
                max_indicators = int(env_max)
            except Exception:
                max_indicators = 0
        if max_indicators <= 0:
            max_indicators = int(self.max_indicators or 0)
        if max_indicators <= 0:
            max_indicators = len(self.mixer.available_indicators)
        self.max_indicators = max(
            2, min(max_indicators, len(self.mixer.available_indicators))
        )

    @staticmethod
    def _strict_rust_requested() -> bool:
        # Genetic expert training/inference is Rust-first and Rust-only.
        return True

    def _validate_gene(self, gene: TALibStrategyGene) -> TALibStrategyGene | None:
        """
        Ensure gene only references available indicators and sane thresholds.
        Returns a sanitized gene or None if it cannot be used.
        """
        if not self.mixer or not self.mixer.available_indicators:
            return None

        valid_set = set(self.mixer.available_indicators)
        filtered_inds = [ind for ind in gene.indicators if ind in valid_set]
        if not filtered_inds:
            return None

        gene.indicators = filtered_inds
        # Drop params/weights for missing indicators
        gene.params = {k: v for k, v in gene.params.items() if k in valid_set}
        gene.weights = {k: float(v) for k, v in (gene.weights or {}).items() if k in filtered_inds}

        # Clamp thresholds to reasonable bounds to avoid degenerate always-long/always-short
        gene.long_threshold = float(np.clip(gene.long_threshold, 0.1, 1.5))
        gene.short_threshold = float(np.clip(gene.short_threshold, -1.5, -0.1))
        return gene

    @staticmethod
    def _is_datetime_index(idx: Any) -> bool:
        if idx is None:
            return False
        if hasattr(idx, "year") and hasattr(idx, "month") and hasattr(idx, "day"):
            return True
        with np.errstate(all="ignore"):
            try:
                arr = np.asarray(idx).reshape(-1)
            except Exception:
                return False
        if arr.size <= 0:
            return False
        if np.issubdtype(arr.dtype, np.datetime64):
            return True
        if arr.dtype.kind == "O":
            for item in arr.tolist():
                if item is None:
                    continue
                if hasattr(item, "year") and hasattr(item, "month") and hasattr(item, "day"):
                    return True
                try:
                    np.datetime64(item, "ns")
                    return True
                except Exception:
                    continue
        return False

    @staticmethod
    def _datetime_index_to_unix_ms(idx: Any) -> np.ndarray | None:
        if not GeneticStrategyExpert._is_datetime_index(idx):
            return None
        rust = GeneticStrategyExpert._rust_time_index_arrays(idx)
        if rust is not None:
            unix_ms, _month_idx, _day_idx = rust
            return unix_ms
        ns = GeneticStrategyExpert._index_to_ns(idx)
        if ns is None or ns.size <= 0:
            return np.zeros(0, dtype=np.int64)
        return (np.asarray(ns, dtype=np.int64) // 1_000_000).astype(np.int64, copy=False)

    @staticmethod
    def _rust_time_index_arrays(idx: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        try:
            import forex_bindings as _fb  # type: ignore
        except Exception:
            return None
        if not hasattr(_fb, "derive_time_index_arrays"):
            return None
        ns = GeneticStrategyExpert._index_to_ns(idx)
        if ns is None:
            return None
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

    @staticmethod
    def _index_to_ns(idx: Any) -> np.ndarray | None:
        if idx is None:
            return None
        try:
            if hasattr(idx, "asi8"):
                return np.asarray(idx.asi8, dtype=np.int64).reshape(-1)
            arr = np.asarray(idx).reshape(-1)
            if arr.size <= 0:
                return None
            if np.issubdtype(arr.dtype, np.datetime64):
                return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
            if arr.dtype.kind in {"i", "u"}:
                return arr.astype(np.int64, copy=False)
            if arr.dtype.kind == "f":
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
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
                        try:
                            out[i] = int(value)
                        except Exception:
                            out[i] = 0
                return out
            if hasattr(idx, "view"):
                viewed = idx.view("int64")
                if hasattr(viewed, "to_numpy"):
                    return np.asarray(viewed.to_numpy(dtype=np.int64, copy=False), dtype=np.int64).reshape(-1)
                return np.asarray(viewed, dtype=np.int64).reshape(-1)
            return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
        except Exception:
            return None

    @staticmethod
    def _month_day_indices(idx: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
        rust = GeneticStrategyExpert._rust_time_index_arrays(idx)
        if rust is not None:
            _unix_ms, month_idx, day_idx = rust
            return month_idx[:n], day_idx[:n]
        if GeneticStrategyExpert._is_datetime_index(idx):
            if hasattr(idx, "year") and hasattr(idx, "month") and hasattr(idx, "day"):
                month_idx = (idx.year.astype(np.int32) * 12 + idx.month.astype(np.int32)).to_numpy(dtype=np.int64)
                day_idx = (
                    idx.year.astype(np.int32) * 10000 + idx.month.astype(np.int32) * 100 + idx.day.astype(np.int32)
                ).to_numpy(dtype=np.int64)
                return month_idx[:n], day_idx[:n]
            arr_idx = np.asarray(idx).reshape(-1)
            if arr_idx.size > 0 and np.issubdtype(arr_idx.dtype, np.datetime64):
                month_idx = arr_idx.astype("datetime64[M]").astype(np.int64)
                day_idx = arr_idx.astype("datetime64[D]").astype(np.int64)
                return month_idx[:n], day_idx[:n]
        idx_ns = GeneticStrategyExpert._index_to_ns(idx)
        if idx_ns is not None and idx_ns.size > 0:
            try:
                if int(np.max(np.abs(idx_ns))) > 10**14:
                    dt = np.asarray(idx_ns, dtype=np.int64).astype("datetime64[ns]")
                    month_idx = dt.astype("datetime64[M]").astype(np.int64)
                    day_idx = dt.astype("datetime64[D]").astype(np.int64)
                    return month_idx[:n], day_idx[:n]
            except Exception:
                pass
        seq = np.arange(n, dtype=np.int64)
        return seq, seq

    @staticmethod
    def _rust_align_ffill_by_ns(
        src_idx: np.ndarray,
        src_vals: np.ndarray,
        tgt_idx: np.ndarray,
    ) -> np.ndarray | None:
        try:
            import forex_bindings as _fb  # type: ignore
        except Exception:
            return None
        if not hasattr(_fb, "align_ffill_values_by_ns"):
            return None
        try:
            out = _fb.align_ffill_values_by_ns(
                np.asarray(src_idx, dtype=np.int64),
                np.asarray(src_vals, dtype=np.float64),
                np.asarray(tgt_idx, dtype=np.int64),
                0.0,
            )
        except Exception:
            return None
        arr = np.asarray(out, dtype=np.float64).reshape(-1)
        if arr.size != int(np.asarray(tgt_idx).size):
            return None
        return arr

    @staticmethod
    def _rust_sorted_index_order(idx: Any) -> np.ndarray | None:
        try:
            import forex_bindings as _fb  # type: ignore
        except Exception:
            return None
        if not hasattr(_fb, "sorted_index_order"):
            return None
        idx_ns = GeneticStrategyExpert._index_to_ns(idx)
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

    @staticmethod
    def _sorted_time_order(idx: Any, n_rows: int) -> np.ndarray | None:
        idx_ns = GeneticStrategyExpert._index_to_ns(idx)
        if idx_ns is None or idx_ns.size != int(n_rows) or idx_ns.size <= 1:
            return None
        if not bool(np.any(idx_ns[1:] < idx_ns[:-1])):
            return None
        order = GeneticStrategyExpert._rust_sorted_index_order(idx_ns)
        if order is not None:
            return order
        return np.argsort(idx_ns, kind="mergesort")

    @staticmethod
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

    @staticmethod
    def _align_ffill_by_ns(
        src_idx: np.ndarray | None,
        src_vals: np.ndarray,
        tgt_idx: np.ndarray | None,
        *,
        dtype: Any = np.float64,
    ) -> np.ndarray | None:
        if src_idx is None or tgt_idx is None:
            return None
        s_idx = np.asarray(src_idx, dtype=np.int64).reshape(-1)
        t_idx = np.asarray(tgt_idx, dtype=np.int64).reshape(-1)
        vals = np.asarray(src_vals, dtype=np.float64).reshape(-1)
        if s_idx.size <= 0 or t_idx.size <= 0 or vals.size <= 0:
            return np.zeros(t_idx.size, dtype=dtype)
        m = min(s_idx.size, vals.size)
        s_idx = s_idx[:m]
        vals = vals[:m]
        rust = GeneticStrategyExpert._rust_align_ffill_by_ns(s_idx, vals, t_idx)
        if rust is not None:
            return rust.astype(dtype, copy=False)
        order = GeneticStrategyExpert._sorted_time_order(s_idx, s_idx.size)
        if order is not None:
            s_idx = s_idx[order]
            vals = vals[order]
        pos = np.searchsorted(s_idx, t_idx, side="right") - 1
        out = np.zeros(t_idx.size, dtype=np.float64)
        valid = pos >= 0
        if np.any(valid):
            out[valid] = vals[np.clip(pos[valid], 0, vals.size - 1)]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(dtype, copy=False)

    def _evaluate_population_rust(
        self,
        df: Any,
        population: list[TALibStrategyGene],
    ) -> list[float] | None:
        try:
            import forex_bindings as _fb  # type: ignore
        except Exception:
            return None
        has_pop_eval = hasattr(_fb, "evaluate_population_talib_ohlcv")
        has_bulk = hasattr(_fb, "talib_bulk_signals_ohlcv")
        if not (has_pop_eval or has_bulk):
            return None
        if df is None or not _is_dataframe(df):
            return None
        try:
            if bool(df.empty):  # type: ignore[attr-defined]
                return None
        except Exception:
            if len(df) <= 0:
                return None

        try:
            close = np.asarray(df["close"], dtype=np.float64)
            high = np.asarray(df["high"], dtype=np.float64) if "high" in df.columns else close
            low = np.asarray(df["low"], dtype=np.float64) if "low" in df.columns else close
            open_ = np.asarray(df["open"], dtype=np.float64) if "open" in df.columns else close
            volume = np.asarray(df["volume"], dtype=np.float64) if "volume" in df.columns else None
            timestamps = self._datetime_index_to_unix_ms(df.index)
        except Exception:
            return None

        symbol = ""
        with np.errstate(all="ignore"):
            try:
                symbol = str(getattr(df, "attrs", {}).get("symbol", "") or "")
            except Exception:
                symbol = ""
        if not symbol:
            symbol = str(os.environ.get("FOREX_BOT_SYMBOL", "") or "")
        pip_size, pip_val = infer_pip_metrics(symbol)
        try:
            spread = float(os.environ.get("FOREX_BOT_PROP_EVAL_SPREAD_PIPS", "1.5") or 1.5)
        except Exception:
            spread = 1.5
        try:
            commission = float(os.environ.get("FOREX_BOT_PROP_EVAL_COMMISSION", "7.0") or 7.0)
        except Exception:
            commission = 7.0
        try:
            max_hold_bars = int(os.environ.get("FOREX_BOT_PROP_MAX_HOLD_BARS", "0") or 0)
        except Exception:
            max_hold_bars = 0
        try:
            causal_min_bars = int(os.environ.get("FOREX_BOT_TALIB_CAUSAL_MIN_BARS", "30") or 30)
        except Exception:
            causal_min_bars = 30
        causal_min_bars = max(2, causal_min_bars)
        try:
            smc_gate_threshold = float(os.environ.get("FOREX_BOT_SMC_GATE_THRESHOLD", "0.0") or 0.0)
        except Exception:
            smc_gate_threshold = 0.0
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

        indicator_sets: list[list[str]] = []
        weight_sets: list[list[float]] = []
        long_thresholds: list[float] = []
        short_thresholds: list[float] = []
        sl_pips: list[float] = []
        tp_pips: list[float] = []
        use_ob_flags: list[int] = []
        use_fvg_flags: list[int] = []
        use_liq_flags: list[int] = []
        use_mtf_flags: list[int] = []
        use_premium_flags: list[int] = []
        use_inducement_flags: list[int] = []
        use_bos_flags: list[int] = []
        use_choch_flags: list[int] = []
        use_eqh_flags: list[int] = []
        use_eql_flags: list[int] = []
        use_displacement_flags: list[int] = []
        map_idx: list[int] = []

        for idx, gene in enumerate(population):
            inds = [str(i).upper() for i in (getattr(gene, "indicators", []) or []) if str(i).strip()]
            if not inds:
                continue
            weights_map = getattr(gene, "weights", {}) or {}
            indicator_sets.append(inds)
            weight_sets.append(
                [float(weights_map.get(ind, weights_map.get(ind.lower(), 1.0)) or 1.0) for ind in inds]
            )
            long_thresholds.append(float(getattr(gene, "long_threshold", 0.66) or 0.66))
            short_thresholds.append(float(getattr(gene, "short_threshold", -0.66) or -0.66))
            sl_pips.append(float(getattr(gene, "sl_pips", 30.0) or 30.0))
            tp_pips.append(float(getattr(gene, "tp_pips", 60.0) or 60.0))
            use_ob_flags.append(1 if bool(getattr(gene, "use_ob", False)) else 0)
            use_fvg_flags.append(1 if bool(getattr(gene, "use_fvg", False)) else 0)
            use_liq_flags.append(1 if bool(getattr(gene, "use_liq_sweep", False)) else 0)
            use_mtf_flags.append(1 if bool(getattr(gene, "mtf_confirmation", False)) else 0)
            use_premium_flags.append(1 if bool(getattr(gene, "use_premium_discount", False)) else 0)
            use_inducement_flags.append(1 if bool(getattr(gene, "use_inducement", False)) else 0)
            use_bos_flags.append(1 if bool(getattr(gene, "use_bos", False)) else 0)
            use_choch_flags.append(1 if bool(getattr(gene, "use_choch", False)) else 0)
            use_eqh_flags.append(1 if bool(getattr(gene, "use_eqh", False)) else 0)
            use_eql_flags.append(1 if bool(getattr(gene, "use_eql", False)) else 0)
            use_displacement_flags.append(1 if bool(getattr(gene, "use_displacement", False)) else 0)
            map_idx.append(idx)

        if not indicator_sets:
            return None

        idx_obj = getattr(df, "index", None)
        month_idx, day_idx = self._month_day_indices(idx_obj, len(df))

        try:
            if has_pop_eval:
                try:
                    metrics = _fb.evaluate_population_talib_ohlcv(  # type: ignore[attr-defined]
                        open_,
                        high,
                        low,
                        close,
                        indicator_sets=indicator_sets,
                        weight_sets=weight_sets,
                        long_thresholds=long_thresholds,
                        short_thresholds=short_thresholds,
                        sl_pips=sl_pips,
                        tp_pips=tp_pips,
                        use_ob_flags=use_ob_flags,
                        use_fvg_flags=use_fvg_flags,
                        use_liq_flags=use_liq_flags,
                        use_mtf_flags=use_mtf_flags,
                        use_premium_flags=use_premium_flags,
                        use_inducement_flags=use_inducement_flags,
                        use_bos_flags=use_bos_flags,
                        use_choch_flags=use_choch_flags,
                        use_eqh_flags=use_eqh_flags,
                        use_eql_flags=use_eql_flags,
                        use_displacement_flags=use_displacement_flags,
                        timestamps=timestamps,
                        volume=volume,
                        include_raw=True,
                        smc_gate_threshold=float(smc_gate_threshold),
                        smc_weight_ob=float(smc_weight_ob),
                        smc_weight_fvg=float(smc_weight_fvg),
                        smc_weight_liq=float(smc_weight_liq),
                        smc_weight_mtf=float(smc_weight_mtf),
                        smc_weight_premium=float(smc_weight_premium),
                        smc_weight_inducement=float(smc_weight_inducement),
                        smc_weight_bos=float(smc_weight_bos),
                        smc_weight_choch=float(smc_weight_choch),
                        smc_weight_eqh=float(smc_weight_eqh),
                        smc_weight_eql=float(smc_weight_eql),
                        smc_weight_displacement=float(smc_weight_displacement),
                        max_hold_bars=int(max_hold_bars),
                        trailing_enabled=False,
                        trailing_atr_multiplier=1.0,
                        trailing_be_trigger_r=1.0,
                        pip_value=float(pip_size),
                        spread_pips=float(spread),
                        commission_per_trade=float(commission),
                        pip_value_per_lot=float(pip_val),
                        causal_min_bars=int(causal_min_bars),
                    )
                except TypeError:
                    metrics = _fb.evaluate_population_talib_ohlcv(  # type: ignore[attr-defined]
                        open_,
                        high,
                        low,
                        close,
                        indicator_sets=indicator_sets,
                        weight_sets=weight_sets,
                        long_thresholds=long_thresholds,
                        short_thresholds=short_thresholds,
                        sl_pips=sl_pips,
                        tp_pips=tp_pips,
                        timestamps=timestamps,
                        volume=volume,
                        include_raw=True,
                        max_hold_bars=int(max_hold_bars),
                        trailing_enabled=False,
                        trailing_atr_multiplier=1.0,
                        trailing_be_trigger_r=1.0,
                        pip_value=float(pip_size),
                        spread_pips=float(spread),
                        commission_per_trade=float(commission),
                        pip_value_per_lot=float(pip_val),
                    )
            else:
                try:
                    raw = _fb.talib_bulk_signals_ohlcv(  # type: ignore[attr-defined]
                        open_,
                        high,
                        low,
                        close,
                        indicator_sets=indicator_sets,
                        weight_sets=weight_sets,
                        long_thresholds=long_thresholds,
                        short_thresholds=short_thresholds,
                        timestamps=timestamps,
                        volume=volume,
                        include_raw=False,
                        causal_min_bars=int(causal_min_bars),
                    )
                except TypeError:
                    raw = _fb.talib_bulk_signals_ohlcv(  # type: ignore[attr-defined]
                        open_,
                        high,
                        low,
                        close,
                        indicator_sets=indicator_sets,
                        weight_sets=weight_sets,
                        long_thresholds=long_thresholds,
                        short_thresholds=short_thresholds,
                        timestamps=timestamps,
                        volume=volume,
                        include_raw=False,
                    )
                sig = np.asarray(raw, dtype=np.int8)
                n_bars = int(len(close))
                n_genes = int(len(map_idx))
                if sig.ndim != 2:
                    return None
                if sig.shape[0] == n_bars and sig.shape[1] == n_genes:
                    sig = sig.T
                elif not (sig.shape[0] == n_genes and sig.shape[1] == n_bars):
                    return None
                metrics = batch_evaluate_strategies(
                    close_prices=close,
                    high_prices=high,
                    low_prices=low,
                    signals=sig,
                    month_indices=month_idx,
                    day_indices=day_idx,
                    sl_pips=np.asarray(sl_pips, dtype=np.float64),
                    tp_pips=np.asarray(tp_pips, dtype=np.float64),
                    max_hold_bars=int(max_hold_bars),
                    trailing_enabled=False,
                    trailing_atr_multiplier=1.0,
                    trailing_be_trigger_r=1.0,
                    pip_value=float(pip_size),
                    spread_pips=float(spread),
                    commission_per_trade=float(commission),
                    pip_value_per_lot=float(pip_val),
                )
            arr = np.asarray(metrics, dtype=np.float64)
        except Exception:
            return None

        if arr.ndim != 2 or arr.shape[0] != len(map_idx) or arr.shape[1] < 9:
            return None

        out = [float("-inf")] * len(population)
        for i, pop_i in enumerate(map_idx):
            sharpe = float(arr[i, 1]) if np.isfinite(arr[i, 1]) else float("-inf")
            trades = float(arr[i, 8]) if np.isfinite(arr[i, 8]) else 0.0
            out[pop_i] = sharpe if trades > 10.0 else -1.0
        return out

    def _predict_votes_rust_bulk(
        self,
        source: Any,
        *,
        target_index: Any | None,
        n_target: int,
    ) -> np.ndarray | None:
        try:
            import forex_bindings as _fb  # type: ignore
        except Exception:
            return None
        if not hasattr(_fb, "talib_bulk_signals_ohlcv"):
            return None
        if source is None or not _is_dataframe(source):
            return None
        try:
            if bool(source.empty):  # type: ignore[attr-defined]
                return None
        except Exception:
            if len(source) <= 0:
                return None
        required = {"open", "high", "low", "close"}
        if not required.issubset({str(c).lower() for c in source.columns}):
            return None

        indicator_sets: list[list[str]] = []
        weight_sets: list[list[float]] = []
        long_thresholds: list[float] = []
        short_thresholds: list[float] = []
        for gene in self.portfolio:
            inds = [str(i).upper() for i in (getattr(gene, "indicators", []) or []) if str(i).strip()]
            if not inds:
                continue
            weights_map = getattr(gene, "weights", {}) or {}
            indicator_sets.append(inds)
            weight_sets.append(
                [float(weights_map.get(ind, weights_map.get(ind.lower(), 1.0)) or 1.0) for ind in inds]
            )
            long_thresholds.append(float(getattr(gene, "long_threshold", 0.66) or 0.66))
            short_thresholds.append(float(getattr(gene, "short_threshold", -0.66) or -0.66))
        if not indicator_sets:
            return None

        open_arr = np.asarray(source["open"], dtype=np.float64)
        high_arr = np.asarray(source["high"], dtype=np.float64)
        low_arr = np.asarray(source["low"], dtype=np.float64)
        close_arr = np.asarray(source["close"], dtype=np.float64)
        volume_arr = np.asarray(source["volume"], dtype=np.float64) if "volume" in source.columns else None
        timestamps = self._datetime_index_to_unix_ms(source.index)
        try:
            causal_min_bars = int(os.environ.get("FOREX_BOT_TALIB_CAUSAL_MIN_BARS", "30") or 30)
        except Exception:
            causal_min_bars = 30
        causal_min_bars = max(2, causal_min_bars)

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
                timestamps=timestamps,
                volume=volume_arr,
                include_raw=False,
                causal_min_bars=causal_min_bars,
            )
        except TypeError:
            raw = _fb.talib_bulk_signals_ohlcv(
                open_arr,
                high_arr,
                low_arr,
                close_arr,
                indicator_sets=indicator_sets,
                weight_sets=weight_sets,
                long_thresholds=long_thresholds,
                short_thresholds=short_thresholds,
                timestamps=timestamps,
                volume=volume_arr,
                include_raw=False,
            )
        except Exception:
            return None

        signals = np.asarray(raw, dtype=np.float64)
        n_src = int(len(source))
        n_genes = int(len(indicator_sets))
        if signals.ndim != 2:
            return None
        if signals.shape[0] == n_genes and signals.shape[1] == n_src:
            signals = signals.T
        if signals.shape[0] != n_src or signals.shape[1] != n_genes:
            return None

        avg_src = np.nan_to_num(np.mean(signals, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
        obj: Any = avg_src
        if target_index is not None:
            src_ns = self._index_to_ns(source.index)
            tgt_ns = self._index_to_ns(target_index)
            aligned = self._align_ffill_by_ns(src_ns, avg_src, tgt_ns, dtype=np.float64)
            if aligned is not None:
                obj = aligned
        out = signal_to_numpy(obj, index=target_index, dtype=np.float64, fill_value=0.0, forward_fill=False)
        if out.size != n_target:
            if out.size > n_target:
                out = out[:n_target]
            else:
                pad = np.zeros(max(0, n_target - out.size), dtype=np.float64)
                out = np.concatenate([out, pad])
        return np.clip(out, -1.0, 1.0).astype(np.float64, copy=False)

    def fit(self, x: Any, y: Any, metadata: Any | None = None) -> None:
        if metadata is None:
            logger.warning("GeneticStrategyExpert requires metadata (OHLC) to fit. Skipping.")
            return

        if not self.mixer or not self.mixer.available_indicators:
            logger.warning("TALib mixer not available/empty. Skipping.")
            return

        self.portfolio = []  # List of TALibStrategyGene

        # BRIDGE: Try to load from the advanced Discovery Engine first
        try:
            import json
            from pathlib import Path

            cache_dir = Path("cache")
            symbol = ""
            try:
                if _is_dataframe(metadata):
                    symbol = str(metadata.attrs.get("symbol", "") or "")
            except Exception:
                symbol = ""
            if not symbol:
                symbol = str(os.environ.get("FOREX_BOT_SYMBOL", "") or "")
            knowledge_path = cache_dir / "talib_knowledge.json"
            if symbol:
                sym_tag = "".join(c for c in symbol if c.isalnum() or c in ("-", "_"))
                sym_path = cache_dir / f"talib_knowledge_{sym_tag}.json"
                if sym_path.exists():
                    knowledge_path = sym_path

            if knowledge_path.exists():
                content = knowledge_path.read_text()
                if content.strip():
                    data = json.loads(content)

                    # Support both new Portfolio format and old Single format
                    genes_data: list[dict] = []
                    if "best_genes" in data:
                        genes_data = list(data["best_genes"] or [])
                    elif "best_gene" in data:
                        genes_data = [data["best_gene"]]

                    use_opp = str(os.environ.get("FOREX_BOT_USE_OPPORTUNISTIC", "1")).strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }
                    if use_opp:
                        opp_path = cache_dir / "talib_knowledge_opportunistic.json"
                        if symbol:
                            opp_sym_path = cache_dir / f"talib_knowledge_opportunistic_{sym_tag}.json"
                            if opp_sym_path.exists():
                                opp_path = opp_sym_path
                        if opp_path.exists():
                            try:
                                opp_data = json.loads(opp_path.read_text())
                                opp_genes = list(opp_data.get("best_genes", []) or [])
                                if opp_genes:
                                    genes_data.extend(opp_genes)
                            except Exception as exc:
                                logger.warning(f"Failed to load opportunistic knowledge: {exc}")

                    for bg_data in genes_data:
                        try:
                            # Reconstruct Gene
                            gene = TALibStrategyGene(
                                indicators=bg_data.get("indicators", []),
                                params=bg_data.get("params", {}),
                                combination_method=bg_data.get("combination_method", "weighted_vote"),
                                long_threshold=bg_data.get("long_threshold", 0.6),
                                short_threshold=bg_data.get("short_threshold", -0.6),
                                weights=bg_data.get("weights", {}),
                                preferred_regime=bg_data.get("preferred_regime", "any"),
                                strategy_id=bg_data.get("strategy_id", "imported"),
                                # SMC Flags
                                use_ob=bg_data.get("use_ob", False),
                                use_fvg=bg_data.get("use_fvg", False),
                                use_liq_sweep=bg_data.get("use_liq_sweep", False),
                                mtf_confirmation=bg_data.get("mtf_confirmation", False),
                                use_premium_discount=bg_data.get("use_premium_discount", False),
                                use_inducement=bg_data.get("use_inducement", False),
                                tp_pips=bg_data.get("tp_pips", 40.0),
                                sl_pips=bg_data.get("sl_pips", 20.0),
                            )
                            gene.fitness = bg_data.get("fitness", 0.0)
                            gene = self._validate_gene(gene)
                            if gene:
                                self.portfolio.append(gene)
                            else:
                                logger.debug("Discovery Engine gene rejected during validation.")
                        except Exception as exc:
                            logger.debug(
                                "Skipping invalid Discovery Engine gene payload: %s", exc, exc_info=True
                            )
                            continue

                    if self.portfolio:
                        logger.info(f"BRIDGE: Loaded {len(self.portfolio)} strategies from Discovery Engine.")
                        self.best_gene = self.portfolio[0]  # Backwards compat
                        return
        except Exception as e:
            logger.warning(f"Failed to bridge Discovery Engine strategy: {e}", exc_info=True)

        logger.info("Evolving strategies over %s generations (Rust-only mode).", self.generations)
        # ... Fallback legacy code ...
        population: list[TALibStrategyGene] = []
        while len(population) < self.population_size:
            gene = self.mixer.generate_random_strategy(max_indicators=self.max_indicators)
            gene = self._validate_gene(gene)
            if gene:
                population.append(gene)

        df = metadata.copy()
        best_fitness = -np.inf

        for _gen in range(self.generations):
            scores = []
            rust_scores = self._evaluate_population_rust(df, population)
            if rust_scores is not None and len(rust_scores) == len(population):
                for gene, fitness in zip(population, rust_scores, strict=False):
                    gene.fitness = float(fitness)
                    scores.append((float(fitness), gene))
            else:
                for gene in population:
                    gene.fitness = -1.0
                    scores.append((-1.0, gene))

            if scores:
                order = self._rust_rank_scores_desc(np.asarray([score for score, _gene in scores], dtype=np.float64))
                if order is not None:
                    scores = [scores[int(i)] for i in order.tolist()]
                else:
                    scores.sort(key=lambda x: x[0], reverse=True)
                if scores[0][0] > best_fitness:
                    best_fitness = scores[0][0]
                    self.best_gene = scores[0][1]

                survivors = [s[1] for s in scores[: self.population_size // 2]]
                new_pop = survivors[:]
                while len(new_pop) < self.population_size:
                    new_pop.append(self.mixer.generate_random_strategy(max_indicators=self.max_indicators))
                population = new_pop

        if self.best_gene:
            self.portfolio = [self.best_gene]

    def predict_proba(self, x: Any, metadata: Any | None = None) -> np.ndarray:
        n = len(x)
        neutral = np.full((n, 3), 1.0 / 3.0, dtype=float)
        if not self.portfolio or self.mixer is None:
            return neutral
        target_index = getattr(x, "index", None)
        source = metadata if metadata is not None else x
        if source is None or not _is_dataframe(source):
            return neutral
        required = {"open", "high", "low", "close"}
        if not required.issubset({str(c).lower() for c in source.columns}):
            return neutral

        try:
            avg_vote = self._predict_votes_rust_bulk(source, target_index=target_index, n_target=n)
            if avg_vote is None:
                return neutral

            # Map average vote into smooth 3-class probabilities.
            # - Neutral dominates near 0 vote
            # - Buy/Sell dominate as vote moves away from 0
            avg_vote = np.clip(avg_vote, -1.0, 1.0).astype(np.float32, copy=False)
            k = 3.0
            neutral_base = 2.0
            neutral_alpha = 4.0
            neutral_logit = neutral_base - neutral_alpha * np.abs(avg_vote)
            logits = np.stack([neutral_logit, k * avg_vote, -k * avg_vote], axis=1).astype(np.float32, copy=False)
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-12)
            return probs.astype(float, copy=False)
        except Exception as e:
            logger.error(f"Genetic predict failed: {e}", exc_info=True)
            return np.zeros((n, 3))

    def save(self, path: str) -> None:
        if self.best_gene is None:
            return
        import joblib

        p = os.path.join(path, "genetic_expert.joblib")
        joblib.dump(self.best_gene, p)

    def load(self, path: str) -> None:
        import joblib

        p = os.path.join(path, "genetic_expert.joblib")
        if os.path.exists(p):
            try:
                self.best_gene = joblib.load(p)
                logger.info(f"Loaded Genetic Strategy: {self.best_gene.indicators}")
            except Exception as e:
                logger.warning(f"Failed to load Genetic Expert: {e}")


