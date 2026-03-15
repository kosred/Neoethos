from __future__ import annotations

import logging
from typing import Any

import numpy as np

import forex_bindings
from .evo_prop_types import frame_attr, frame_len
from ..training.evaluation import truth_probability

logger = logging.getLogger(__name__)

def run_strategy_discovery(
    df: Any,
    settings: Any,
    symbol: str,
    timeframe: str,
    max_strategies: int = 1000,
) -> list[dict[str, Any]]:
    """
    Calls the discovery engine (Rust or Python/TensorFlow) to find profitable strategies.
    This replaces the heavy logic previously in evo_prop.py.
    """
    logger.info(f"Starting discovery for {symbol} {timeframe} with {frame_len(df)} rows.")
    
    # Extract OHLCV data for Rust engine
    o = np.asarray(df["open"], dtype=np.float64)
    h = np.asarray(df["high"], dtype=np.float64)
    l = np.asarray(df["low"], dtype=np.float64)
    c = np.asarray(df["close"], dtype=np.float64)
    ts = np.asarray(df.index.values, dtype=np.int64) if hasattr(df.index, "values") else None
    vol = np.asarray(df["volume"], dtype=np.float64) if "volume" in df.columns else None

    # Call the high-performance Rust discovery engine
    # Using relaxed criteria to capture more profitable strategies as requested.
    result = forex_bindings.search_discovery_ohlcv(
        open=o,
        high=h,
        low=l,
        close=c,
        timestamps=ts,
        volume=vol,
        population=settings.get("population", 1000),
        generations=settings.get("generations", 10),
        max_indicators=settings.get("max_indicators", 12),
        candidate_count=settings.get("candidate_count", 5000),
        portfolio_size=settings.get("max_strategies_to_find", 2000),
        min_trades_per_day=settings.get("min_trades_per_day", 0.2), # Relaxed
        include_raw=True,
        keep_min_sharpe=0.1, # Loose filter to capture more
        keep_min_trades=10.0,
        keep_min_profit=1.0,
    )
    
    portfolio = result.get("portfolio", [])
    logger.info(f"Rust discovery returned {len(portfolio)} strategies.")
    
    # Calculate truth probability for the broad set
    for strategy in portfolio:
        # Ensure metrics are in a format truth_probability expects
        strategy["truth_prob"] = truth_probability(strategy)
            
    return portfolio[:max_strategies]
