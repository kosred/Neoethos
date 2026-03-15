from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .pipeline_base import LabelConfig

logger = logging.getLogger(__name__)

def compute_labels_triple_barrier(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    max_hold: int,
    tp_pips: float,
    sl_pips: float,
    pip_size: float = 0.0001,
) -> np.ndarray:
    n = close.shape[0]
    labels = np.zeros(n, dtype=np.int8)
    
    tp_delta = tp_pips * pip_size
    sl_delta = sl_pips * pip_size
    
    for i in range(n - 1):
        # Look ahead up to max_hold
        end = min(i + max_hold, n)
        for j in range(i + 1, end):
            # Check TP/SL
            if high[j] >= close[i] + tp_delta:
                labels[i] = 1
                break
            if low[j] <= close[i] - sl_delta:
                labels[i] = -1
                break
        # If no barrier hit, label based on final return or leave as 0 (Neutral)
            
    return labels

def compute_labels(df: Any, cfg: LabelConfig) -> np.ndarray:
    if df is None or len(df) == 0:
        return np.zeros(0, dtype=np.int8)
    
    close = np.asarray(df["close"], dtype=np.float64)
    high = np.asarray(df["high"], dtype=np.float64)
    low = np.asarray(df["low"], dtype=np.float64)
    
    if cfg.use_triple_barrier:
        return compute_labels_triple_barrier(
            close, high, low, cfg.max_hold, cfg.tp_pips or 40.0, cfg.sl_pips or 20.0
        )
    
    # Standard future return labeling
    fut_return = (np.roll(close, -cfg.horizon) - close) / np.where(close != 0, close, 1.0)
    labels = np.where(fut_return > cfg.min_dist, 1, np.where(fut_return < -cfg.min_dist, -1, 0))
    labels[-cfg.horizon:] = 0
    return labels.astype(np.int8)
