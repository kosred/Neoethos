from __future__ import annotations

import json
import logging
import os
import time
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
except Exception:
    _fb = None  # type: ignore
    _RUST_SEARCH = False


def _safe_indices(idx: pd.Index, n: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(idx, pd.DatetimeIndex):
        month_idx = (idx.year.astype(np.int32) * 12 + idx.month.astype(np.int32)).to_numpy(dtype=np.int64)
        day_idx = (idx.year.astype(np.int32) * 10000 + idx.month.astype(np.int32) * 100 + idx.day.astype(np.int32)).to_numpy(dtype=np.int64)
        return month_idx[:n], day_idx[:n]
    seq = np.arange(n, dtype=np.int64)
    return seq, seq


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
        "max_dd_pct": float(getattr(gene, "max_dd_pct", 0.0)),
        "trades": float(getattr(gene, "trades", 0.0)),
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


def _convert_rust_gene(gene: dict[str, Any], feature_names: list[str], available: set[str]) -> TALibStrategyGene | None:
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

    try:
        max_dd_pct = float(
            gene.get("max_dd_pct", gene.get("max_dd", gene.get("drawdown", 0.0))) or 0.0
        )
    except Exception:
        max_dd_pct = 0.0
    try:
        trades = float(gene.get("trades", gene.get("trade_count", 0.0)) or 0.0)
    except Exception:
        trades = 0.0

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
        sharpe_ratio=float(gene.get("sharpe_ratio", 0.0)),
        win_rate=float(gene.get("win_rate", 0.0)),
        max_dd_pct=max_dd_pct,
        trades=trades,
        use_ob=bool(gene.get("use_ob", False)),
        use_fvg=bool(gene.get("use_fvg", False)),
        use_liq_sweep=bool(gene.get("use_liq_sweep", False)),
        mtf_confirmation=bool(gene.get("mtf_confirmation", False)),
        use_premium_discount=bool(gene.get("use_premium_discount", False)),
        use_inducement=bool(gene.get("use_inducement", False)),
        tp_pips=float(gene.get("tp_pips", 40.0)),
        sl_pips=float(gene.get("sl_pips", 20.0)),
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
        pip_size, pip_val = infer_pip_metrics(str(df.attrs.get("symbol", "") or ""))
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
            spread_pips=1.5,
            commission_per_trade=7.0,
        )
        if metrics is None or len(metrics) < 9:
            gene.fitness = 0.0
            return 0.0
        gene.fitness = float(metrics[0])
        gene.sharpe_ratio = float(metrics[1])
        gene.max_dd_pct = float(metrics[3])
        gene.win_rate = float(metrics[4])
        gene.trades = float(metrics[8])
        return float(gene.fitness)
    except Exception as exc:
        logger.debug("Prop search eval failed: %s", exc)
        return 0.0


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


def _strategy_passes_filter(
    gene: TALibStrategyGene,
    *,
    max_dd: float,
    min_profit: float,
    min_trades: float,
) -> bool:
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
    max_dd, min_profit, min_trades, min_keep, portfolio_cap = _strategy_keep_limits(settings)
    symbol = str(df.attrs.get("symbol", "") or "")
    timeframe = str(df.attrs.get("timeframe", df.attrs.get("tf", "")) or "")
    if _RUST_SEARCH and _fb is not None:
        try:
            ts = None
            idx = df.index
            if isinstance(idx, pd.DatetimeIndex):
                ts = (idx.view("int64") // 1_000_000).to_numpy(dtype=np.int64)
            close = df["close"].to_numpy(dtype=np.float64)
            high = df["high"].to_numpy(dtype=np.float64)
            low = df["low"].to_numpy(dtype=np.float64)
            open_ = df["open"].to_numpy(dtype=np.float64) if "open" in df.columns else close
            volume = df["volume"].to_numpy(dtype=np.float64) if "volume" in df.columns else None

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
                True,
            )
            feature_names = list(result.get("feature_names") or [])
            genes_raw = list(result.get("genes") or [])
            available = {str(x).upper() for x in ALL_INDICATORS}
            best: list[TALibStrategyGene] = []
            for g in genes_raw:
                if not isinstance(g, dict):
                    continue
                gene = _convert_rust_gene(g, feature_names, available)
                if gene:
                    best.append(gene)
            if not best:
                raise RuntimeError("Rust search produced no usable genes")

            filtered = [
                g
                for g in best
                if _strategy_passes_filter(
                    g,
                    max_dd=max_dd,
                    min_profit=min_profit,
                    min_trades=min_trades,
                )
            ]
            selected, strict_kept, ranked_total = _select_ranked(
                best,
                filtered=filtered,
                min_keep=min_keep,
                cap=portfolio_cap,
            )

            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
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
                "Prop search (Rust): kept %s/%s genes (strict=%s, min_keep=%s) for %s %s "
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
            return
        except Exception as exc:
            logger.warning("Rust prop search failed, falling back to Python: %s", exc, exc_info=True)

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
        cache = mixer.bulk_calculate_indicators(df, genes)
        scored: list[tuple[float, TALibStrategyGene]] = []
        for gene in genes:
            score = _evaluate_gene(df, gene, mixer, cache, settings)
            gene.fitness = score
            scored.append((score, gene))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _score, gene in scored:
            if _strategy_passes_filter(
                gene,
                max_dd=max_dd,
                min_profit=min_profit,
                min_trades=min_trades,
            ):
                accepted.append(gene)
        survivors = [g for _, g in scored[: max(1, pop // 2)]]
        best = survivors
        while len(survivors) < pop:
            survivors.append(mixer.generate_random_strategy(max_indicators=max_indicators))
        genes = survivors
        if max_hours > 0 and (time.time() - start) > max_hours * 3600.0:
            break

    selected, strict_kept, ranked_total = _select_ranked(
        accepted + best,
        filtered=accepted,
        min_keep=min_keep,
        cap=portfolio_cap,
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
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
