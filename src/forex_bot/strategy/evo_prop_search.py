from __future__ import annotations

import logging
import random
import time
from typing import Any

import numpy as np

from .evo_prop_types import (
    df_pip_metrics,
    env_float,
    frame_column_numpy,
    frame_has_column,
    frame_len,
    history_span_days_months,
)
from .fast_backtest import fast_pip_backtest

logger = logging.getLogger(__name__)

def evaluate_gene(gene: dict[str, Any], df: Any, fee_pips: float | None = None) -> dict[str, Any] | None:
    if not isinstance(gene, dict) or "expression" not in gene:
        return None
    try:
        if not frame_has_column(df, "close"):
            return None
        close = frame_column_numpy(df, "close")
        if close.size < 100:
            return None
        
        # In a real implementation, 'expression' would be evaluated via numexpr or a custom Rust engine.
        # For refactoring purposes, we keep the logic structure.
        # We assume discovery_tensor or similar provides the signal mask.
        mask = gene.get("_signal_mask")
        if mask is None:
            return None
        
        pip_size, pip_val = df_pip_metrics(df, close)
        if fee_pips is None:
            fee_pips = float(env_float("FOREX_BOT_PROP_SEARCH_FEE_PIPS", 1.5))

        res = fast_pip_backtest(
            close,
            mask,
            pip_size=pip_size,
            pip_value_per_lot=pip_val,
            fee_pips=fee_pips,
        )
        if not res or "metrics" not in res:
            return None
        
        m = res["metrics"]
        days, months = history_span_days_months(df)
        m["span_days"] = days
        m["span_months"] = months
        
        out = gene.copy()
        out["metrics"] = m
        return out
    except Exception as e:
        logger.error(f"Error evaluating gene: {e}")
        return None

def evolve_generation(
    population: list[dict[str, Any]],
    df: Any,
    size: int = 100,
    mutation_rate: float = 0.1,
) -> list[dict[str, Any]]:
    if not population:
        return []
    
    # Sort by sharpe ratio (or a composite fitness)
    population.sort(key=lambda x: float(x.get("metrics", {}).get("sharpe_ratio", 0.0)), reverse=True)
    
    elite_count = max(1, size // 10)
    new_pop = population[:elite_count]
    
    while len(new_pop) < size:
        if random.random() < 0.3:
            # Randomly create or mutate
            parent = random.choice(population[:size//2])
            child = parent.copy()
            # Mutation logic here...
            evaluated = evaluate_gene(child, df)
            if evaluated:
                new_pop.append(evaluated)
        else:
            # Crossover
            p1 = random.choice(population[:size//2])
            p2 = random.choice(population[:size//2])
            child = p1.copy() # Placeholder for crossover
            evaluated = evaluate_gene(child, df)
            if evaluated:
                new_pop.append(evaluated)
                
    return new_pop
