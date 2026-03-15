from __future__ import annotations
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def calculate_hurst_exponent(price: np.ndarray, max_lags: int = 100) -> float:
    """
    Calculates the Hurst Exponent to identify regime:
    H < 0.5: Mean-reverting (Anti-persistent)
    H = 0.5: Random Walk
    H > 0.5: Trending (Persistent)
    """
    if len(price) < max_lags * 2:
        return 0.5
    
    lags = range(2, max_lags)
    tau = [np.sqrt(np.std(np.subtract(price[lag:], price[:-lag]))) for lag in lags]
    
    # Use polyfit to find the slope of the log-log plot
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def kalman_filter(price: np.ndarray, process_variance: float = 1e-5, estimated_measurement_variance: float = 0.1**2) -> np.ndarray:
    """
    Kalman Filter for price smoothing with minimal lag.
    process_variance: Q (how fast the system changes)
    estimated_measurement_variance: R (how much noise in the input)
    """
    n = len(price)
    post_estimate = np.zeros(n)
    post_error_estimate = np.zeros(n)
    
    # Init
    post_estimate[0] = price[0]
    post_error_estimate[0] = 1.0
    
    for i in range(1, n):
        # Prediction
        prior_estimate = post_estimate[i-1]
        prior_error_estimate = post_error_estimate[i-1] + process_variance
        
        # Update
        blending_factor = prior_error_estimate / (prior_error_estimate + estimated_measurement_variance)
        post_estimate[i] = prior_estimate + blending_factor * (price[i] - prior_estimate)
        post_error_estimate[i] = (1 - blending_factor) * prior_error_estimate
        
    return post_estimate

def vertical_horizontal_filter(close: np.ndarray, period: int = 28) -> np.ndarray:
    """
    VHF: Identifies if price is trending vs congestion.
    """
    n = len(close)
    vhf = np.zeros(n)
    if n < period:
        return vhf
    
    for i in range(period, n):
        win = close[i-period+1:i+1]
        p_max = np.max(win)
        p_min = np.min(win)
        numerator = abs(p_max - p_min)
        denominator = np.sum(np.abs(np.diff(win)))
        if denominator != 0:
            vhf[i] = numerator / denominator
        else:
            vhf[i] = 0.0
    return vhf

def chande_momentum_oscillator(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    CMO: Similar to RSI but more sensitive.
    Uses pure momentum instead of gains/losses.
    """
    n = len(close)
    cmo = np.zeros(n)
    if n < period:
        return cmo
    
    diff = np.diff(close, prepend=close[0])
    for i in range(period, n):
        win = diff[i-period+1 : i+1]
        sum_gains = np.sum(win[win > 0])
        sum_losses = np.sum(np.abs(win[win < 0]))
        denom = sum_gains + sum_losses
        if denom != 0:
            cmo[i] = 100.0 * (sum_gains - sum_losses) / denom
        else:
            cmo[i] = 0.0
    return cmo

def regime_detector(price: np.ndarray, period: int = 100) -> dict[str, Any]:
    """
    Consolidated regime detection using Hurst and VHF.
    """
    h_exp = calculate_hurst_exponent(price)
    vhf_val = vertical_horizontal_filter(price, period=28)[-1]
    
    regime = "uncertain"
    if h_exp > 0.55 and vhf_val > 0.4:
        regime = "trending"
    elif h_exp < 0.45 and vhf_val < 0.3:
        regime = "mean_reverting"
        
    return {
        "hurst_exponent": h_exp,
        "vhf": vhf_val,
        "regime": regime
    }
