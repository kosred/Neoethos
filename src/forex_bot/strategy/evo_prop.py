from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..features.talib_mixer import (
    ALL_INDICATORS,
    TALibStrategyGene,
    TALibStrategyMixer,
)
from .fast_backtest import (
    fast_evaluate_strategy,
    infer_pip_metrics,
    infer_sl_tp_pips_auto,
)

logger = logging.getLogger(__name__)

try:
    import forex_bindings as _fb  # type: ignore

    _RUST_SEARCH = hasattr(_fb, "search_evolve_ohlcv")
    _RUST_GPU_SEARCH = hasattr(_fb, "search_evolve_gpu_ohlcv")
except Exception:
    _fb = None  # type: ignore
    _RUST_SEARCH = False
    _RUST_GPU_SEARCH = False


def _safe_indices(idx: pd.Index, n: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(idx, pd.DatetimeIndex):
        month_idx = (idx.year.astype(np.int32) * 12 + idx.month.astype(np.int32)).to_numpy(dtype=np.int64)
        day_idx = (idx.year.astype(np.int32) * 10000 + idx.month.astype(np.int32) * 100 + idx.day.astype(np.int32)).to_numpy(dtype=np.int64)
        return month_idx[:n], day_idx[:n]
    seq = np.arange(n, dtype=np.int64)
    return seq, seq


def _datetime_index_to_unix_ms(idx: pd.DatetimeIndex) -> np.ndarray:
    """
    Convert DatetimeIndex to int64 unix milliseconds robustly across pandas versions.

    Some pandas versions return ndarray from `view("int64")`, others return an Index-like
    object with `.to_numpy()`. We normalize both to plain NumPy int64.
    """
    idx_i64 = idx.view("int64")
    if hasattr(idx_i64, "to_numpy"):
        idx_i64 = idx_i64.to_numpy(dtype=np.int64, copy=False)
    else:
        idx_i64 = np.asarray(idx_i64, dtype=np.int64)
    return (np.asarray(idx_i64, dtype=np.int64) // 1_000_000).astype(np.int64, copy=False)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _df_reference_prices(df: pd.DataFrame) -> dict[str, float] | None:
    raw = df.attrs.get("pip_reference_prices")
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


def _df_pip_metrics(df: pd.DataFrame, close: np.ndarray | None = None) -> tuple[float, float]:
    pip_size = _safe_float(df.attrs.get("pip_size"), 0.0)
    pip_val = _safe_float(df.attrs.get("pip_value_per_lot"), 0.0)
    if pip_size > 0.0 and pip_val > 0.0:
        return float(pip_size), float(pip_val)

    symbol = str(df.attrs.get("symbol", "") or "")
    last_close: float | None = None
    if close is not None and close.size > 0:
        last_close = _safe_float(close[-1], 0.0)
    elif "close" in df.columns and len(df) > 0:
        last_close = _safe_float(df["close"].iloc[-1], 0.0)
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


def _history_span_days_months(df: pd.DataFrame) -> tuple[float, float]:
    if df is None or df.empty:
        return 0.0, 0.0
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return 0.0, 0.0
    try:
        if idx.tz is None:
            i2 = idx.tz_localize("UTC")
        else:
            i2 = idx.tz_convert("UTC")
    except Exception:
        i2 = idx
    try:
        span_days = float((i2.max() - i2.min()).total_seconds() / 86400.0)
    except Exception:
        span_days = 0.0
    span_days = max(0.0, span_days)
    span_months = (span_days / 30.4375) if span_days > 0.0 else 0.0
    return float(span_days), float(span_months)


def _copy_attrs(src: pd.DataFrame, dst: pd.DataFrame) -> pd.DataFrame:
    try:
        dst.attrs.update(dict(getattr(src, "attrs", {}) or {}))
    except Exception:
        pass
    return dst


def _holdout_cfg(settings: Any) -> tuple[float, int, float, float, float, int, bool]:
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
    required = str(
        _get("FOREX_BOT_PROP_HOLDOUT_REQUIRED", getattr(settings.models, "prop_search_holdout_required", False))
    ).strip().lower() in {"1", "true", "yes", "on"}
    return frac, max(0, min_rows), min_sharpe, min_win, min_pf, max(0, min_tr), required


def _split_discovery_holdout(df: pd.DataFrame, settings: Any) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    frac, min_rows, *_ = _holdout_cfg(settings)
    if frac <= 0.0 or df is None or df.empty:
        return df, None
    n = len(df)
    if n < max(1000, min_rows):
        return df, None
    split = int(round(n * (1.0 - min(0.8, max(0.05, frac)))))
    split = max(500, min(n - 500, split))
    if split <= 0 or split >= n:
        return df, None
    search_df = _copy_attrs(df, df.iloc[:split].copy())
    holdout_df = _copy_attrs(df, df.iloc[split:].copy())
    return search_df, holdout_df


def _apply_holdout_validation(
    *,
    selected: list[TALibStrategyGene],
    holdout_df: pd.DataFrame | None,
    settings: Any,
    max_dd: float,
    min_profit: float,
    min_trades: float,
    initial_balance: float,
) -> list[TALibStrategyGene]:
    if not selected or holdout_df is None or holdout_df.empty:
        return selected

    frac, _min_rows, min_sharpe, min_win, min_pf, min_tr_holdout, required = _holdout_cfg(settings)
    if frac <= 0.0:
        return selected

    try:
        mixer = TALibStrategyMixer()
        if not mixer.available_indicators:
            logger.warning("Holdout validation skipped: TA-Lib indicators unavailable.")
            return [] if required else selected

        cache = mixer.bulk_calculate_indicators(holdout_df, selected)
        _days, holdout_months = _history_span_days_months(holdout_df)
        passed: list[TALibStrategyGene] = []
        for gene in selected:
            g_eval = replace(gene)
            _evaluate_gene(holdout_df, g_eval, mixer, cache, settings)
            if not _strategy_passes_filter(
                g_eval,
                max_dd=max_dd,
                min_profit=min_profit,
                min_trades=max(min_trades, float(min_tr_holdout)),
                history_months=holdout_months,
                initial_balance=initial_balance,
            ):
                continue
            if float(getattr(g_eval, "sharpe_ratio", 0.0) or 0.0) < float(min_sharpe):
                continue
            if float(getattr(g_eval, "win_rate", 0.0) or 0.0) < float(min_win):
                continue
            if float(getattr(g_eval, "profit_factor", 0.0) or 0.0) < float(min_pf):
                continue
            passed.append(g_eval)

        if not passed:
            logger.warning(
                "Holdout validation kept 0/%s strategies (required=%s, min_sharpe=%.2f, min_win=%.2f, min_pf=%.2f).",
                len(selected),
                required,
                min_sharpe,
                min_win,
                min_pf,
            )
            return [] if required else selected

        passed = _dedupe_ranked(passed)
        logger.info(
            "Holdout validation kept %s/%s strategies (min_sharpe=%.2f, min_win=%.2f, min_pf=%.2f).",
            len(passed),
            len(selected),
            min_sharpe,
            min_win,
            min_pf,
        )
        return passed
    except Exception as exc:
        logger.warning("Holdout validation failed: %s", exc)
        return selected


def _gene_to_dict(gene: TALibStrategyGene) -> dict[str, Any]:
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
        "tp_pips": float(getattr(gene, "tp_pips", 40.0)),
        "sl_pips": float(getattr(gene, "sl_pips", 20.0)),
    }


def _feature_to_indicator(name: str, available: set[str]) -> str | None:
    if not name:
        return None
    raw = str(name).strip()
    if raw.lower().startswith("ta_"):
        raw = raw[3:]
    cand = raw.upper()
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


def _evaluate_gene(
    df: pd.DataFrame,
    gene: TALibStrategyGene,
    mixer: TALibStrategyMixer,
    cache: dict[str, pd.Series] | None,
    settings: Any,
) -> float:
    try:
        sig = mixer.compute_signals(df, gene, cache=cache).fillna(0.0).to_numpy(dtype=np.int8)
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        open_ = df["open"].to_numpy(dtype=np.float64) if "open" in df.columns else close
        atr_vals = df["atr"].to_numpy(dtype=np.float64) if "atr" in df.columns else None
        month_idx, day_idx = _safe_indices(df.index, len(df))
        pip_size, pip_val = _df_pip_metrics(df, close=close)
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
        metrics = fast_evaluate_strategy(
            close_prices=close,
            high_prices=high,
            low_prices=low,
            signals=sig,
            month_indices=month_idx,
            day_indices=day_idx,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            pip_value=pip_size,
            pip_value_per_lot=pip_val,
            spread_pips=float(os.environ.get("FOREX_BOT_PROP_EVAL_SPREAD_PIPS", "1.5") or 1.5),
            commission_per_trade=float(os.environ.get("FOREX_BOT_PROP_EVAL_COMMISSION", "7.0") or 7.0),
        )
        if metrics is None or len(metrics) < 9:
            gene.fitness = 0.0
            gene.net_profit = 0.0
            gene.profit_factor = 0.0
            gene.expectancy = 0.0
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
    except Exception as exc:
        logger.debug("Prop search eval failed: %s", exc)
        return 0.0


def _expand_threshold_variants(
    *,
    df: pd.DataFrame,
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
    try:
        max_dd = float(
            os.environ.get(
                "FOREX_BOT_PROP_KEEP_MAX_DD",
                getattr(settings.risk, "total_drawdown_limit", 0.07),
            )
            or 0.07
        )
    except Exception:
        max_dd = 0.07
    max_dd = float(min(1.0, max(0.0, max_dd)))

    try:
        min_profit = float(os.environ.get("FOREX_BOT_PROP_KEEP_MIN_PROFIT", "0.0") or 0.0)
    except Exception:
        min_profit = 0.0

    try:
        min_trades = float(os.environ.get("FOREX_BOT_PROP_KEEP_MIN_TRADES", "1") or 1.0)
    except Exception:
        min_trades = 1.0
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

    min_sharpe = _env_float("FOREX_BOT_PROP_KEEP_MIN_SHARPE", 0.0)
    if min_sharpe > 0.0:
        try:
            sharpe = float(getattr(gene, "sharpe_ratio", 0.0) or 0.0)
        except Exception:
            sharpe = 0.0
        if sharpe < min_sharpe:
            return False

    min_win_rate = _env_float("FOREX_BOT_PROP_KEEP_MIN_WIN_RATE", 0.0)
    if min_win_rate > 0.0:
        try:
            win_rate = float(getattr(gene, "win_rate", 0.0) or 0.0)
        except Exception:
            win_rate = 0.0
        if win_rate < min_win_rate:
            return False

    min_profit_factor = _env_float("FOREX_BOT_PROP_KEEP_MIN_PROFIT_FACTOR", 0.0)
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

    min_monthly_pct = _env_float("FOREX_BOT_PROP_KEEP_MIN_MONTHLY_PROFIT_PCT", 0.0)
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
    df: pd.DataFrame,
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
    if df is None or df.empty:
        return
    search_df, holdout_df = _split_discovery_holdout(df, settings)
    if holdout_df is not None and not holdout_df.empty:
        logger.info(
            "Prop search holdout enabled: search_rows=%s holdout_rows=%s",
            len(search_df),
            len(holdout_df),
        )
    max_dd, min_profit, min_trades, min_keep, portfolio_cap = _strategy_keep_limits(settings)
    symbol = str(search_df.attrs.get("symbol", "") or "")
    timeframe = str(search_df.attrs.get("timeframe", search_df.attrs.get("tf", "")) or "")
    history_days, history_months = _history_span_days_months(search_df)
    if (_RUST_SEARCH or _RUST_GPU_SEARCH) and _fb is not None:
        try:
            ts = None
            idx = search_df.index
            if isinstance(idx, pd.DatetimeIndex):
                ts = _datetime_index_to_unix_ms(idx)
            close = search_df["close"].to_numpy(dtype=np.float64)
            high = search_df["high"].to_numpy(dtype=np.float64)
            low = search_df["low"].to_numpy(dtype=np.float64)
            open_ = search_df["open"].to_numpy(dtype=np.float64) if "open" in search_df.columns else close
            volume = search_df["volume"].to_numpy(dtype=np.float64) if "volume" in search_df.columns else None
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
                    if best:
                        try:
                            mixer = TALibStrategyMixer()
                            if mixer.available_indicators:
                                cache = mixer.bulk_calculate_indicators(search_df, best)
                                rescored: list[TALibStrategyGene] = []
                                for g in best:
                                    try:
                                        _evaluate_gene(search_df, g, mixer, cache, settings)
                                        rescored.append(g)
                                    except Exception:
                                        continue
                                best = _dedupe_ranked(rescored) if rescored else []
                        except Exception as exc:
                            logger.warning("EvoGP GPU rescoring failed; using raw fitness ranking: %s", exc)
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
            try:
                best = _expand_threshold_variants(df=search_df, genes=best, settings=settings)
            except Exception as exc:
                logger.warning("Threshold expansion failed after %s; continuing without expansion: %s", search_mode, exc)
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
            )

            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
                "rows": int(len(df)),
                "search_rows": int(len(search_df)),
                "holdout_rows": int(len(holdout_df)) if holdout_df is not None else 0,
                "history_days": float(history_days),
                "history_months": float(history_months),
                "initial_balance": float(actual_balance),
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
                "(profit>%.3f, max_dd<=%.3f, trades>=%.0f). Wrote %s",
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
                out,
            )
            return
        except Exception as exc:
            logger.warning("Rust prop search path failed, falling back to Python: %s", exc, exc_info=True)

    mixer = TALibStrategyMixer()
    if not mixer.available_indicators:
        logger.warning("Prop search: no TA-Lib indicators available.")
        return

    pop = max(2, int(population or 0))
    gens = max(1, int(generations or 0))
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
        max_indicators = len(mixer.available_indicators)
    max_indicators = max(2, min(max_indicators, len(mixer.available_indicators)))

    genes = [mixer.generate_random_strategy(max_indicators=max_indicators) for _ in range(pop)]
    best: list[TALibStrategyGene] = []
    accepted: list[TALibStrategyGene] = []

    start = time.time()
    for _ in range(gens):
        cache = mixer.bulk_calculate_indicators(search_df, genes)
        scored: list[tuple[float, TALibStrategyGene]] = []
        for gene in genes:
            score = _evaluate_gene(search_df, gene, mixer, cache, settings)
            gene.fitness = score
            scored.append((score, gene))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _score, gene in scored:
            if _strategy_passes_filter(
                gene,
                max_dd=max_dd,
                min_profit=min_profit,
                min_trades=min_trades,
                history_months=history_months,
                initial_balance=actual_balance,
            ):
                accepted.append(gene)
        survivors = [g for _, g in scored[: max(1, pop // 2)]]
        best = survivors
        while len(survivors) < pop:
            survivors.append(mixer.generate_random_strategy(max_indicators=max_indicators))
        genes = survivors
        if max_hours > 0 and (time.time() - start) > max_hours * 3600.0:
            break

    try:
        expanded = _expand_threshold_variants(df=search_df, genes=accepted + best, settings=settings)
    except Exception as exc:
        logger.warning("Threshold expansion failed; using unexpanded population: %s", exc)
        expanded = _dedupe_ranked(accepted + best)
    filtered = [
        g
        for g in expanded
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
        expanded,
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
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": int(len(df)),
        "search_rows": int(len(search_df)),
        "holdout_rows": int(len(holdout_df)) if holdout_df is not None else 0,
        "history_days": float(history_days),
        "history_months": float(history_months),
        "initial_balance": float(actual_balance),
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
        "Prop search: kept %s/%s genes (strict=%s, min_keep=%s) for %s %s "
        "(profit>%.3f, max_dd<=%.3f, trades>=%.0f). Wrote %s",
        len(selected),
        ranked_total,
        strict_kept,
        min_keep,
        symbol or "?",
        timeframe or "?",
        min_profit,
        max_dd,
        min_trades,
        out,
    )


__all__ = ["run_evo_search"]
