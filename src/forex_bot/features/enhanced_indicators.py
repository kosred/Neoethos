from __future__ import annotations

import contextlib
import logging
from typing import Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

def causal_tanh_zscore(values: np.ndarray, *, min_periods: int = 30) -> np.ndarray:
    """
    Strictly causal normalization using rolling mean and variance.
    Prioritizes Rust implementation via forex_bindings if available.
    """
    with contextlib.suppress(Exception):
        import forex_bindings  # type: ignore
        if hasattr(forex_bindings, "causal_tanh_zscore"):
            return forex_bindings.causal_tanh_zscore(values, min_periods)

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    
    out = np.zeros(arr.shape[0], dtype=np.float64)
    mean = 0.0
    m2 = 0.0
    count = 0
    
    for i, val in enumerate(arr):
        if count >= min_periods:
            var = m2 / count
            std = np.sqrt(var) if var > 0 else 0.0
            z = (val - mean) / std if std > 1e-12 else (val - mean)
            out[i] = np.tanh(z)
            
        if np.isfinite(val):
            count += 1
            delta = val - mean
            mean += delta / count
            delta2 = val - mean
            m2 += delta * delta2
            
    return out

def detect_divergence(price: np.ndarray, indicator: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Detects simple regular divergence.
    Prioritizes Rust implementation via forex_bindings if available.
    """
    with contextlib.suppress(Exception):
        import forex_bindings  # type: ignore
        if hasattr(forex_bindings, "detect_divergence"):
            return forex_bindings.detect_divergence(price, indicator, window)

    n = price.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n < window:
        return out
    
    for i in range(window, n):
        # Look for local peaks/troughs in the window
        p_window = price[i-window:i+1]
        ind_window = indicator[i-window:i+1]
        
        # Bullish Divergence (Higher Low in Indicator, Lower Low in Price)
        if price[i] < np.min(price[i-window:i]) and indicator[i] > np.min(indicator[i-window:i]):
            out[i] = 1.0
            
        # Bearish Divergence (Lower High in Indicator, Higher High in Price)
        elif price[i] > np.max(price[i-window:i]) and indicator[i] < np.max(indicator[i-window:i]):
            out[i] = -1.0
            
    return out

def enhanced_rsi(close: np.ndarray, period: int = 14) -> dict[str, np.ndarray]:
    """
    Returns base RSI, normalized RSI, and divergence signal.
    """
    n = close.shape[0]
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Use exponential moving average for RSI (Wilder's style)
    alpha = 1.0 / period
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    curr_gain = 0.0
    curr_loss = 0.0
    for i in range(n):
        curr_gain = (alpha * gain[i]) + ((1 - alpha) * curr_gain)
        curr_loss = (alpha * loss[i]) + ((1 - alpha) * curr_loss)
        avg_gain[i] = curr_gain
        avg_loss[i] = curr_loss
        
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Normalization
    rsi_norm = causal_tanh_zscore(rsi)
    
    # Divergence
    div = detect_divergence(close, rsi)
    
    return {
        "rsi": rsi,
        "rsi_norm": rsi_norm,
        "rsi_div": div
    }

def enhanced_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal_period: int = 9) -> dict[str, np.ndarray]:
    """
    Returns MACD, Signal, Histogram, and normalized versions.
    """
    def ema(arr, p):
        out = np.zeros_like(arr)
        alpha = 2.0 / (p + 1)
        curr = arr[0]
        for i in range(len(arr)):
            curr = (alpha * arr[i]) + ((1 - alpha) * curr)
            out[i] = curr
        return out

    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd = ema_fast - ema_slow
    signal = ema(macd, signal_period)
    hist = macd - signal
    
    # Normalized versions (Z-score of MACD and Hist)
    macd_norm = causal_tanh_zscore(macd)
    hist_norm = causal_tanh_zscore(hist)
    
    # Divergence of MACD with Price
    div = detect_divergence(close, macd)
    
    return {
        "macd": macd,
        "signal": signal,
        "hist": hist,
        "macd_norm": macd_norm,
        "hist_norm": hist_norm,
        "macd_div": div
    }

def super_smoother(price: np.ndarray, period: int = 10) -> np.ndarray:
    """Ehlers SuperSmoother Filter."""
    n = len(price)
    out = price.copy()
    if n < 2: return out
    a1 = np.exp(-np.sqrt(2) * np.pi / period)
    b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    for i in range(2, n):
        out[i] = c1 * (price[i] + price[i-1]) / 2 + c2 * out[i-1] + c3 * out[i-2]
    return out

def vortex_indicator(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """Vortex Indicator (VI). Returns (VI_plus, VI_minus)."""
    with contextlib.suppress(Exception):
        import forex_bindings
        if hasattr(forex_bindings, "vortex_indicator"):
            vp, vm = forex_bindings.vortex_indicator(high, low, close, period)
            return np.asarray(vp), np.asarray(vm)
            
    n = len(close)
    vi_plus = np.full(n, 1.0)
    vi_minus = np.full(n, 1.0)
    if n < period + 1: return vi_plus, vi_minus
    vm_plus = np.abs(high[1:] - low[:-1])
    vm_minus = np.abs(low[1:] - high[:-1])
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    def rolling_sum(arr, w):
        res = np.zeros(len(arr))
        c = np.cumsum(arr)
        res[w-1] = c[w-1]
        res[w:] = c[w:] - c[:-w]
        return res
    vmp_padded = np.pad(vm_plus, (1, 0), constant_values=0)
    vmm_padded = np.pad(vm_minus, (1, 0), constant_values=0)
    tr_padded = np.pad(tr, (1, 0), constant_values=0)
    s_vmp = rolling_sum(vmp_padded, period)
    s_vmm = rolling_sum(vmm_padded, period)
    s_tr = rolling_sum(tr_padded, period)
    mask = (s_tr > 0)
    vi_plus[mask] = s_vmp[mask] / s_tr[mask]
    vi_minus[mask] = s_vmm[mask] / s_tr[mask]
    return vi_plus, vi_minus

def fisher_transform(price: np.ndarray, period: int = 10) -> np.ndarray:
    """Ehlers Fisher Transform."""
    with contextlib.suppress(Exception):
        import forex_bindings
        if hasattr(forex_bindings, "fisher_transform"):
            return np.asarray(forex_bindings.fisher_transform(price, period))
            
    n = len(price)
    fisher = np.zeros(n)
    if n < period: return fisher
    value = np.zeros(n)
    for i in range(period, n):
        win = price[i-period+1:i+1]
        p_min = np.min(win)
        p_max = np.max(win)
        if p_max != p_min:
            val = 0.66 * ((price[i] - p_min) / (p_max - p_min) - 0.5) + 0.67 * value[i-1]
        else:
            val = 0.0
        val = max(-0.99, min(0.99, val))
        value[i] = val
        fisher[i] = 0.5 * np.log((1 + val) / (1 - val)) + 0.5 * fisher[i-1]
    return fisher

def wavetrend(price: np.ndarray, n1: int = 10, n2: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    """WaveTrend Oscillator."""
    def ema(arr, p):
        out = np.zeros_like(arr)
        alpha = 2.0 / (p + 1)
        curr = arr[0]
        for i in range(len(arr)):
            curr = (alpha * arr[i]) + ((1 - alpha) * curr)
            out[i] = curr
        return out
    esa = ema(price, n1)
    d = ema(np.abs(price - esa), n1)
    ci = (price - esa) / (0.015 * d + 1e-12)
    wt1 = ema(ci, n2)
    wt2 = np.zeros_like(wt1)
    for i in range(4, len(wt1)):
        wt2[i] = np.mean(wt1[i-3:i+1])
    return wt1, wt2
