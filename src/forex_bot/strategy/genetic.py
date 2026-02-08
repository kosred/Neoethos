"""Compatibility shim for legacy genetic strategy interfaces."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import fast_backtest as fb


@dataclass(slots=True)
class GeneticGene:
    indicators: list[str]
    params: dict[str, dict[str, Any]]
    weights: dict[str, float]
    fitness: float = 0.0
    evaluated: bool = False


class GeneticStrategyEvolution:
    """
    Legacy wrapper retained for compatibility with older strategy tests/integrations.
    """

    def __init__(self, population_size: int = 50, mixer: Any | None = None) -> None:
        self.population_size = max(1, int(population_size or 1))
        self.mixer = mixer
        self.population: list[GeneticGene] = []

    def _evaluate_population(self, df: pd.DataFrame, population: list[GeneticGene] | None = None) -> None:
        genes = population if population is not None else self.population
        if not genes:
            return

        if df is None or df.empty:
            for gene in genes:
                gene.fitness = float("-inf")
                gene.evaluated = True
            return

        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64) if "high" in df.columns else close
        low = df["low"].to_numpy(dtype=np.float64) if "low" in df.columns else close
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            month_idx = (idx.year.astype(np.int32) * 12 + idx.month.astype(np.int32)).to_numpy(dtype=np.int64)
            day_idx = (
                idx.year.astype(np.int32) * 10000 + idx.month.astype(np.int32) * 100 + idx.day.astype(np.int32)
            ).to_numpy(dtype=np.int64)
        else:
            seq = np.arange(len(df), dtype=np.int64)
            month_idx = seq
            day_idx = seq

        symbol = str(df.attrs.get("symbol", "") or "")
        pip_size, pip_val = fb.infer_pip_metrics(symbol)

        for gene in genes:
            try:
                if self.mixer is None:
                    raise RuntimeError("GeneticStrategyEvolution requires a mixer with compute_signals")
                sig = self.mixer.compute_signals(df, gene)
                if isinstance(sig, pd.Series):
                    sig_arr = sig.fillna(0).to_numpy(dtype=np.int8)
                else:
                    sig_arr = np.asarray(sig, dtype=np.int8)
                if len(sig_arr) != len(df):
                    sig_arr = np.resize(sig_arr, len(df))

                metrics = fb.fast_evaluate_strategy(
                    close_prices=close,
                    high_prices=high,
                    low_prices=low,
                    signals=sig_arr,
                    month_indices=month_idx,
                    day_indices=day_idx,
                    sl_pips=30.0,
                    tp_pips=60.0,
                    pip_value=pip_size,
                    pip_value_per_lot=pip_val,
                    spread_pips=1.5,
                    commission_per_trade=7.0,
                )
                gene.fitness = float(metrics[0]) if metrics is not None and len(metrics) > 0 else float("-inf")
            except Exception:
                gene.fitness = float("-inf")
            gene.evaluated = bool(np.isfinite(gene.fitness))


__all__ = ["GeneticGene", "GeneticStrategyEvolution"]
