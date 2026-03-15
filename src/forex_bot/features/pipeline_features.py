from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .enhanced_indicators import (
    enhanced_macd,
    enhanced_rsi,
    fisher_transform,
    super_smoother,
    vortex_indicator,
    wavetrend,
)
from .pipeline_base import align_series_by_ts, index_to_ns_like

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, settings: Any) -> None:
        self.settings = settings

    def compute_enhanced_features(self, df: Any) -> Any:
        if df is None or len(df) <= 0:
            return df
        
        out = df.copy()
        close = np.asarray(out["close"], dtype=np.float64)
        high = np.asarray(out["high"], dtype=np.float64)
        low = np.asarray(out["low"], dtype=np.float64)
        
        # 1. Enhanced RSI (includes Divergence and Normalization)
        rsi_data = enhanced_rsi(close)
        out["rsi"] = rsi_data["rsi"]
        out["rsi_norm"] = rsi_data["rsi_norm"]
        out["rsi_div"] = rsi_data["rsi_div"]
        
        # 2. Enhanced MACD
        macd_data = enhanced_macd(close)
        out["macd"] = macd_data["macd"]
        out["macd_signal"] = macd_data["macd_signal"]
        out["macd_hist"] = macd_data["macd_hist"]
        out["macd_norm"] = macd_data["macd_norm"]
        out["hist_norm"] = macd_data["hist_norm"]
        out["macd_div"] = macd_data["macd_div"]
        
        # 3. Advanced Oscillators & Filters
        out["fisher"] = fisher_transform(close, period=10)
        out["supersmoother"] = super_smoother(close, period=10)
        
        vi_plus, vi_minus = vortex_indicator(high, low, close, period=14)
        out["vi_plus"] = vi_plus
        out["vi_minus"] = vi_minus
        out["vi_diff"] = vi_plus - vi_minus
        
        wt1, wt2 = wavetrend(close, n1=10, n2=21)
        out["wt1"] = wt1
        out["wt2"] = wt2
        out["wt_diff"] = wt1 - wt2
        
        # 4. Basic OHLC Features
        out["returns"] = np.diff(close, prepend=close[0]) / np.where(close != 0, close, 1.0)
        out["hl_range"] = (high - low) / np.where(close != 0, close, 1.0)
        
        return out
