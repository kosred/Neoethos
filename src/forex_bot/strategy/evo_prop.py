from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .evo_prop_discovery import run_strategy_discovery
from .evo_prop_search import evaluate_gene
from .evo_prop_types import (
    frame_attr,
    frame_empty,
    frame_len,
    holdout_cfg,
    to_numpy_1d,
    train_years_cfg,
    trim_to_recent_years,
)
from ..training.evaluation import prop_backtest, split_discovery_holdout

logger = logging.getLogger(__name__)

class PropSearch:
    """
    Refined strategy search engine.
    Now allows keeping hundreds or thousands of strategies that meet basic profitability rules.
    """

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.train_years = train_years_cfg(settings)

    def search(self, df: Any, symbol: str, timeframe: str) -> list[dict[str, Any]]:
        if frame_empty(df):
            return []

        # 1. Trim to recent history
        if self.train_years > 0.0:
            df = trim_to_recent_years(df, self.train_years)

        # 2. Configure holdout
        (
            h_frac,
            h_min_rows,
            min_sharpe,
            min_win,
            min_pf,
            min_trades,
            h_required,
            h_years,
            min_truth,
        ) = holdout_cfg(self.settings)

        # 3. Split data
        discovery_df, holdout_df = split_discovery_holdout(
            df, holdout_frac=h_frac, min_rows=h_min_rows, holdout_years=h_years
        )

        # 4. Run discovery (Relaxed Filter)
        # We now pass max_strategies=2000 to keep more candidates
        population = run_strategy_discovery(
            discovery_df, self.settings, symbol, timeframe, max_strategies=2000
        )

        if not population:
            logger.warning(f"No strategies found for {symbol} {timeframe}")
            return []

        if not frame_empty(holdout_df):
            logger.info(f"Validating {len(population)} genes on holdout ({frame_len(holdout_df)} rows)")
            validated = []
            dropped = 0
            for gene in population:
                # Use prop_backtest for standardized validation
                # (Need to pass signals, but gene evaluation usually produces signals)
                # For now, evaluate_gene is a specialized light evaluator for discovery
                m = evaluate_gene(gene, holdout_df)
                if not m:
                    dropped += 1
                    continue
                
                metrics = m.get("metrics", {})
                h_sharpe = float(metrics.get("sharpe_ratio", 0.0))
                h_win = float(metrics.get("win_rate", 0.0))
                h_pf = float(metrics.get("profit_factor", 0.0))
                h_trades = int(metrics.get("trade_count", 0))
                h_profit = float(metrics.get("total_profit_pips", 0.0))
                
                # Consistency Check (from metrics)
                h_consistency = float(metrics.get("consistency", 0.0))
                # Add truth prob
                from ..training.evaluation import truth_probability
                gene["truth_prob"] = truth_probability(metrics)
                
                if (
                    h_sharpe >= min_sharpe
                    and h_win >= min_win
                    and h_pf >= min_pf
                    and h_trades >= min_trades
                    and h_profit > 0.0
                    and h_consistency >= 0.4
                ):
                    gene["holdout_metrics"] = metrics
                    gene["is_validated"] = True
                    validated.append(gene)
                else:
                    dropped += 1
            population = validated
            logger.info(f"Holdout summary: {len(population)} validated, {dropped} dropped.")
        elif h_required:
            logger.error(f"Holdout required but not available for {symbol}. Dropping all.")
            return []

        # 6. Final Filter for Truth Probability
        if min_truth > 0.0:
            population = [p for p in population if float(p.get("truth_prob", 0.0)) >= min_truth]

        # 7. Quality Sorting (Sharpe Ratio on Full Data or Discovery Data)
        population.sort(key=lambda x: float(x.get("metrics", {}).get("sharpe_ratio", 0.0)), reverse=True)

        logger.info(f"PropSearch complete: {len(population)} strategies retained for {symbol}.")
        return population
