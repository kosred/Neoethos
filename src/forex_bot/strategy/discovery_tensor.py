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

    _RUST_DISCOVERY = hasattr(_fb, "search_discovery_ohlcv")
except Exception:
    _fb = None  # type: ignore
    _RUST_DISCOVERY = False


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
    settings: Any | None,
    pip_size: float,
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    atr_values: np.ndarray | None,
) -> tuple[float, float]:
    sl_cfg = None
    tp_cfg = None
    if settings is not None:
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
            if settings is not None:
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
    if settings is not None:
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


def _gene_key(gene: TALibStrategyGene) -> str:
    sid = str(getattr(gene, "strategy_id", "") or "").strip()
    if sid:
        return f"id:{sid}"
    return (
        f"sig:{tuple(gene.indicators)}|{gene.combination_method}|"
        f"{float(gene.long_threshold):.6f}|{float(gene.short_threshold):.6f}"
    )


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


def _strategy_keep_limits(settings: Any | None, default_cap: int) -> tuple[float, float, float, int, int]:
    try:
        risk_dd = getattr(settings.risk, "total_drawdown_limit", 0.07) if settings is not None else 0.07
        keep_max_dd = float(
            os.environ.get(
                "FOREX_BOT_DISCOVERY_KEEP_MAX_DD",
                risk_dd,
            )
            or 0.07
        )
    except Exception:
        keep_max_dd = 0.07
    keep_max_dd = float(min(1.0, max(0.0, keep_max_dd)))

    try:
        keep_min_profit = float(os.environ.get("FOREX_BOT_DISCOVERY_KEEP_MIN_PROFIT", "0.0") or 0.0)
    except Exception:
        keep_min_profit = 0.0

    try:
        keep_min_trades = float(os.environ.get("FOREX_BOT_DISCOVERY_KEEP_MIN_TRADES", "1.0") or 1.0)
    except Exception:
        keep_min_trades = 1.0
    keep_min_trades = float(max(0.0, keep_min_trades))

    try:
        keep_min_count = int(os.environ.get("FOREX_BOT_DISCOVERY_KEEP_MIN_COUNT", "100") or 100)
    except Exception:
        keep_min_count = 100
    keep_min_count = max(0, keep_min_count)

    fallback_cap = max(1, int(default_cap or 1))
    try:
        keep_cap = int(os.environ.get("FOREX_BOT_DISCOVERY_PORTFOLIO", str(fallback_cap)) or fallback_cap)
    except Exception:
        keep_cap = fallback_cap
    keep_cap = max(1, keep_cap)
    if keep_min_count > keep_cap:
        keep_min_count = keep_cap

    return keep_max_dd, keep_min_profit, keep_min_trades, keep_min_count, keep_cap


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


class TensorDiscoveryEngine:
    def __init__(
        self,
        *,
        device: str = "cpu",
        n_experts: int = 100,
        timeframes: list[str] | None = None,
        max_rows: int = 0,
        stream_mode: bool = False,
        auto_cap: bool = True,
        settings: Any | None = None,
    ) -> None:
        self.device = device
        self.n_experts = int(n_experts or 0)
        self.timeframes = timeframes or []
        self.max_rows = int(max_rows or 0)
        self.stream_mode = bool(stream_mode)
        self.auto_cap = bool(auto_cap)
        self.settings = settings
        self._last_payload: dict[str, Any] | None = None

    def run_unsupervised_search(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        iterations: int = 1000,
        news_features: pd.DataFrame | None = None,
    ) -> None:
        if frames is None or len(frames) == 0:
            return
        base_tf = self.timeframes[0] if self.timeframes else next(iter(frames.keys()))
        base_df = frames.get(base_tf)
        if base_df is None or base_df.empty:
            return
        df = base_df.copy()
        if self.max_rows > 0 and len(df) > self.max_rows:
            df = df.tail(self.max_rows)
        if len(df) < 50:
            return
        iter_budget = max(1, int(iterations or 1))

        def _env_int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw is None or str(raw).strip() == "":
                return int(default)
            try:
                return int(raw)
            except Exception:
                return int(default)

        if _RUST_DISCOVERY and _fb is not None:
            try:
                ts = None
                idx = df.index
                if isinstance(idx, pd.DatetimeIndex):
                    idx_i64 = idx.view("int64")
                    if hasattr(idx_i64, "to_numpy"):
                        idx_i64 = idx_i64.to_numpy(dtype=np.int64, copy=False)
                    else:
                        idx_i64 = np.asarray(idx_i64, dtype=np.int64)
                    ts = (np.asarray(idx_i64, dtype=np.int64) // 1_000_000).astype(np.int64, copy=False)
                close = df["close"].to_numpy(dtype=np.float64)
                high = df["high"].to_numpy(dtype=np.float64)
                low = df["low"].to_numpy(dtype=np.float64)
                open_ = df["open"].to_numpy(dtype=np.float64) if "open" in df.columns else close
                volume = df["volume"].to_numpy(dtype=np.float64) if "volume" in df.columns else None

                default_pop = max(8, min(100, iter_budget))
                default_gens = max(1, min(5, (iter_budget + 19) // 20))
                default_candidates = max(10, min(200, default_pop * 2))
                default_portfolio = 3000
                if self.settings is not None:
                    try:
                        default_portfolio = int(
                            getattr(self.settings.models, "prop_search_portfolio_size", default_portfolio)
                            or default_portfolio
                        )
                    except Exception:
                        default_portfolio = 3000
                default_portfolio = max(100, default_portfolio)
                default_candidates = max(default_candidates, min(10_000, max(100, default_portfolio // 2)))

                result = _fb.search_discovery_ohlcv(
                    open_,
                    high,
                    low,
                    close,
                    ts,
                    volume,
                    max(4, _env_int("FOREX_BOT_DISCOVERY_POP", default_pop)),
                    max(1, _env_int("FOREX_BOT_DISCOVERY_GENS", default_gens)),
                    max(2, _env_int("FOREX_BOT_DISCOVERY_MAX_INDICATORS", 12)),
                    max(10, _env_int("FOREX_BOT_DISCOVERY_CANDIDATES", default_candidates)),
                    max(5, _env_int("FOREX_BOT_DISCOVERY_PORTFOLIO", default_portfolio)),
                    float(os.environ.get("FOREX_BOT_DISCOVERY_CORR", "0.7") or 0.7),
                    float(os.environ.get("FOREX_BOT_DISCOVERY_MIN_TRADES", "1.0") or 1.0),
                    True,
                )

                feature_names = list(result.get("feature_names") or [])
                portfolio = list(result.get("portfolio") or [])
                available = {str(x).upper() for x in ALL_INDICATORS}
                best: list[TALibStrategyGene] = []
                for g in portfolio:
                    if not isinstance(g, dict):
                        continue
                    gene = _convert_rust_gene(g, feature_names, available)
                    if gene:
                        best.append(gene)
                if not best:
                    raise RuntimeError("Rust discovery produced no usable genes")

                keep_max_dd, keep_min_profit, keep_min_trades, keep_min_count, keep_cap = _strategy_keep_limits(
                    self.settings,
                    default_cap=default_portfolio,
                )

                filtered = [
                    g
                    for g in best
                    if float(getattr(g, "fitness", 0.0) or 0.0) > keep_min_profit
                    and float(getattr(g, "max_dd_pct", 0.0) or 0.0) <= keep_max_dd
                    and float(getattr(g, "trades", 0.0) or 0.0) >= keep_min_trades
                ]
                selected, strict_kept, ranked_total = _select_ranked(
                    best,
                    filtered=filtered,
                    min_keep=keep_min_count,
                    cap=keep_cap,
                )

                payload = {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "best_genes": [_gene_to_dict(g) for g in selected],
                }
                out_dir = Path("cache")
                out_dir.mkdir(parents=True, exist_ok=True)
                symbol = str(df.attrs.get("symbol", "") or "")
                out_path = out_dir / "talib_knowledge.json"
                if symbol:
                    safe = "".join(c for c in symbol if c.isalnum() or c in ("-", "_"))
                    out_path = out_dir / f"talib_knowledge_{safe}.json"
                out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                self._last_payload = payload
                logger.info(
                    "Discovery (Rust): kept %s/%s genes (strict=%s, min_keep=%s) "
                    "(profit>%.3f, max_dd<=%.3f, trades>=%.0f). Wrote %s",
                    len(selected),
                    ranked_total,
                    strict_kept,
                    keep_min_count,
                    keep_min_profit,
                    keep_max_dd,
                    keep_min_trades,
                    out_path,
                )
                return
            except Exception as exc:
                logger.warning("Rust discovery failed, falling back to Python: %s", exc, exc_info=True)
        mixer = TALibStrategyMixer()
        if not mixer.available_indicators:
            logger.warning("Discovery: no TA-Lib indicators available.")        
            return

        n = max(1, min(self.n_experts, int(os.environ.get("FOREX_BOT_DISCOVERY_SAMPLE", "64") or 64)))
        max_indicators = 0
        env_max = os.environ.get("FOREX_BOT_DISCOVERY_MAX_INDICATORS") or os.environ.get(
            "FOREX_BOT_PROP_SEARCH_MAX_INDICATORS"
        )
        if env_max:
            try:
                max_indicators = int(env_max)
            except Exception:
                max_indicators = 0
        if max_indicators <= 0:
            try:
                max_indicators = int(
                    getattr(self.settings.models, "prop_search_max_indicators", 0) or 0
                )
            except Exception:
                max_indicators = 0
        if max_indicators <= 0:
            max_indicators = len(mixer.available_indicators)
        max_indicators = max(2, min(max_indicators, len(mixer.available_indicators)))
        genes = [mixer.generate_random_strategy(max_indicators=max_indicators) for _ in range(n)]
        cache = mixer.bulk_calculate_indicators(df, genes)

        symbol = str(df.attrs.get("symbol", "") or "")
        pip_size, pip_val = infer_pip_metrics(symbol)
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        open_ = df["open"].to_numpy(dtype=np.float64) if "open" in df.columns else close
        atr_vals = df["atr"].to_numpy(dtype=np.float64) if "atr" in df.columns else None
        month_idx, day_idx = _safe_indices(df.index, len(df))
        default_cap = 3000
        if self.settings is not None:
            try:
                default_cap = int(
                    getattr(self.settings.models, "prop_search_portfolio_size", default_cap) or default_cap
                )
            except Exception:
                default_cap = 3000
        keep_max_dd, keep_min_profit, keep_min_trades, keep_min_count, keep_cap = _strategy_keep_limits(
            self.settings,
            default_cap=default_cap,
        )

        scored: list[tuple[float, TALibStrategyGene]] = []
        for gene in genes:
            try:
                sig = mixer.compute_signals(df, gene, cache=cache).fillna(0.0).to_numpy(dtype=np.int8)
                sl_pips, tp_pips = _resolve_sl_tp(
                    gene=gene,
                    settings=self.settings,
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
                    score = 0.0
                    gene.sharpe_ratio = 0.0
                    gene.max_dd_pct = 0.0
                    gene.win_rate = 0.0
                    gene.trades = 0.0
                else:
                    score = float(metrics[0])
                    gene.sharpe_ratio = float(metrics[1])
                    gene.max_dd_pct = float(metrics[3])
                    gene.win_rate = float(metrics[4])
                    gene.trades = float(metrics[8])
            except Exception as exc:
                logger.debug("Discovery: gene eval failed: %s", exc)
                score = 0.0
            gene.fitness = score
            scored.append((score, gene))

        scored.sort(key=lambda x: x[0], reverse=True)
        merged = [g for _, g in scored]
        filtered = [
            g
            for g in merged
            if float(getattr(g, "fitness", 0.0) or 0.0) > keep_min_profit
            and float(getattr(g, "max_dd_pct", 0.0) or 0.0) <= keep_max_dd
            and float(getattr(g, "trades", 0.0) or 0.0) >= keep_min_trades
        ]
        best, strict_kept, ranked_total = _select_ranked(
            merged,
            filtered=filtered,
            min_keep=keep_min_count,
            cap=keep_cap,
        )

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "best_genes": [_gene_to_dict(g) for g in best],
        }
        out_dir = Path("cache")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "talib_knowledge.json"
        if symbol:
            safe = "".join(c for c in symbol if c.isalnum() or c in ("-", "_"))
            out_path = out_dir / f"talib_knowledge_{safe}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._last_payload = payload
        logger.info(
            "Discovery: kept %s/%s genes (strict=%s, min_keep=%s) "
            "(profit>%.3f, max_dd<=%.3f, trades>=%.0f). Wrote %s",
            len(best),
            ranked_total,
            strict_kept,
            keep_min_count,
            keep_min_profit,
            keep_max_dd,
            keep_min_trades,
            out_path,
        )

    def save_experts(self, path: str) -> None:
        """
        Backward-compatible artifact writer expected by TrainingService.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._last_payload or {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "best_genes": [],
        }
        try:
            import torch

            torch.save(payload, out_path)
        except Exception:
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Discovery: saved experts artifact %s", out_path)


__all__ = ["TensorDiscoveryEngine"]
