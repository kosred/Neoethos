from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.config import Settings

try:
    import forex_bindings as _fb  # type: ignore
except Exception:  # pragma: no cover - optional native extension
    _fb = None


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean().iloc[-1]
    try:
        return float(atr)
    except Exception:
        return 0.0


def _adx_last(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    n = int(min(len(high), len(low), len(close)))
    if n <= period + 1:
        return float("nan")
    h = high.to_numpy(dtype=float, copy=False)
    l = low.to_numpy(dtype=float, copy=False)
    c = close.to_numpy(dtype=float, copy=False)

    tr = np.zeros(n, dtype=float)
    plus_dm = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)
    for i in range(1, n):
        tr1 = h[i] - l[i]
        tr2 = abs(h[i] - c[i - 1])
        tr3 = abs(l[i] - c[i - 1])
        tr[i] = max(tr1, tr2, tr3)
        up = h[i] - h[i - 1]
        down = l[i - 1] - l[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    tr_sum = float(np.nansum(tr[1 : period + 1]))
    plus_sum = float(np.nansum(plus_dm[1 : period + 1]))
    minus_sum = float(np.nansum(minus_dm[1 : period + 1]))
    dx: list[float] = []
    p = float(period)
    for i in range(period + 1, n):
        tr_sum = tr_sum - (tr_sum / p) + tr[i]
        plus_sum = plus_sum - (plus_sum / p) + plus_dm[i]
        minus_sum = minus_sum - (minus_sum / p) + minus_dm[i]
        if tr_sum <= 0:
            dx.append(0.0)
            continue
        plus_di = 100.0 * plus_sum / tr_sum
        minus_di = 100.0 * minus_sum / tr_sum
        denom = max(1e-9, plus_di + minus_di)
        dx.append(100.0 * abs(plus_di - minus_di) / denom)

    if len(dx) < period:
        return float("nan")
    adx = float(np.nanmean(np.asarray(dx[:period], dtype=float)))
    for val in dx[period:]:
        adx = ((adx * (p - 1.0)) + float(val)) / p
    return adx


def _infer_regime(high: pd.Series, low: pd.Series, close: pd.Series, settings: Settings) -> str:
    atr_period = int(getattr(settings.risk, "atr_period", 14) or 14)
    trend_thr = float(getattr(settings.risk, "regime_adx_trend", 25.0) or 25.0)
    range_thr = float(getattr(settings.risk, "regime_adx_range", 20.0) or 20.0)
    adx = _adx_last(high, low, close, period=max(5, atr_period))
    if np.isfinite(adx):
        if adx >= trend_thr:
            return "trend"
        if adx <= range_thr:
            return "range"
    return "neutral"


def _swing_levels(high: pd.Series, low: pd.Series, *, lookback: int, swing_window: int) -> tuple[float, float] | None:
    span = max(3, (2 * max(1, swing_window)) + 1)
    lb = max(span + 2, int(lookback))
    h = high.tail(lb)
    l = low.tail(lb)
    if len(h) < span or len(l) < span:
        return None

    roll_hi = h.rolling(span, center=True, min_periods=span).max()
    roll_lo = l.rolling(span, center=True, min_periods=span).min()
    eps = 1e-12
    swing_highs = h[(roll_hi.notna()) & (h >= (roll_hi - eps))]
    swing_lows = l[(roll_lo.notna()) & (l <= (roll_lo + eps))]

    resistance = float(swing_highs.iloc[-1]) if len(swing_highs) > 0 else float(h.max())
    support = float(swing_lows.iloc[-1]) if len(swing_lows) > 0 else float(l.min())
    if not np.isfinite(resistance) or not np.isfinite(support):
        return None
    return support, resistance


def _atr_distances(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    settings: Settings,
    regime: str,
) -> tuple[float, float, float] | None:
    atr_period = int(getattr(settings.risk, "atr_period", 14) or 14)
    atr = _atr(high, low, close, period=max(5, atr_period))
    if not np.isfinite(atr) or atr <= 0.0:
        return None
    atr_mult = float(getattr(settings.risk, "atr_stop_multiplier", 1.5) or 1.5)
    min_dist = float(getattr(settings.risk, "meta_label_min_dist", 0.0) or 0.0)
    sl_dist = max(atr * max(atr_mult, 0.1), min_dist)
    if sl_dist <= 0:
        return None
    rr = float(getattr(settings.risk, "min_risk_reward", 2.0) or 2.0)
    if regime == "trend":
        rr = float(getattr(settings.risk, "rr_trend", rr) or rr)
    elif regime == "range":
        rr = float(getattr(settings.risk, "rr_range", rr) or rr)
    else:
        rr = float(getattr(settings.risk, "rr_neutral", rr) or rr)
    rr = max(0.1, rr)
    return float(sl_dist), float(sl_dist * rr), float(rr)


def _structure_distances(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    settings: Settings,
    signal: int | None,
    regime: str,
) -> tuple[float, float, float] | None:
    lookback = int(getattr(settings.risk, "structure_lookback_bars", 120) or 120)
    swing_window = int(getattr(settings.risk, "structure_swing_window", 2) or 2)
    lvls = _swing_levels(high, low, lookback=max(20, lookback), swing_window=max(1, swing_window))
    if lvls is None:
        return None
    support, resistance = lvls
    px = float(close.iloc[-1])
    if not np.isfinite(px) or px <= 0:
        return None

    s = int(signal or 0)
    if s > 0:
        sl_raw = px - support
        tp_raw = resistance - px
    elif s < 0:
        sl_raw = resistance - px
        tp_raw = px - support
    else:
        down = px - support
        up = resistance - px
        sl_raw = min(max(0.0, down), max(0.0, up))
        tp_raw = max(max(0.0, down), max(0.0, up))
    if not np.isfinite(sl_raw) or sl_raw <= 0.0:
        return None

    atr_period = int(getattr(settings.risk, "atr_period", 14) or 14)
    atr = _atr(high, low, close, period=max(5, atr_period))
    min_dist = float(getattr(settings.risk, "meta_label_min_dist", 0.0) or 0.0)
    min_mult = float(getattr(settings.risk, "structure_min_atr_mult", 0.8) or 0.8)
    max_mult = float(getattr(settings.risk, "structure_max_atr_mult", 4.0) or 4.0)
    if np.isfinite(atr) and atr > 0.0:
        lo = max(min_dist, atr * max(0.1, min_mult))
        hi = max(lo, atr * max(min_mult, max_mult))
        sl_dist = float(np.clip(sl_raw, lo, hi))
    else:
        sl_dist = max(sl_raw, min_dist)

    rr_floor = float(getattr(settings.risk, "min_risk_reward", 2.0) or 2.0)
    rr_regime = rr_floor
    if regime == "trend":
        rr_regime = float(getattr(settings.risk, "rr_trend", rr_floor) or rr_floor)
    elif regime == "range":
        rr_regime = float(getattr(settings.risk, "rr_range", rr_floor) or rr_floor)
    else:
        rr_regime = float(getattr(settings.risk, "rr_neutral", rr_floor) or rr_floor)

    rr_struct = tp_raw / max(sl_dist, 1e-9) if (np.isfinite(tp_raw) and tp_raw > 0.0) else rr_regime
    rr = max(0.1, min(6.0, max(rr_floor * 0.75, rr_struct)))
    tp_dist = max(sl_dist * rr, tp_raw if (np.isfinite(tp_raw) and tp_raw > 0.0) else 0.0)
    if tp_dist <= 0.0:
        tp_dist = sl_dist * rr_regime
        rr = rr_regime
    return float(sl_dist), float(tp_dist), float(max(0.1, rr))


def _rust_vol_tail_pips(
    df: pd.DataFrame,
    settings: Settings,
    pip_size: float,
) -> tuple[float, float, float] | None:
    if _fb is None or not hasattr(_fb, "infer_stop_target_pips_ohlcv"):
        return None
    try:
        open_ = df["open"].astype(float).to_numpy(dtype=np.float64, copy=False)
        high = df["high"].astype(float).to_numpy(dtype=np.float64, copy=False)
        low = df["low"].astype(float).to_numpy(dtype=np.float64, copy=False)
        close = df["close"].astype(float).to_numpy(dtype=np.float64, copy=False)
        return _fb.infer_stop_target_pips_ohlcv(
            open_,
            high,
            low,
            close,
            float(pip_size),
            str(getattr(settings.risk, "vol_estimator", "ensemble") or "ensemble"),
            int(getattr(settings.risk, "vol_window", 50) or 50),
            float(getattr(settings.risk, "ewma_lambda", 0.94) or 0.94),
            int(getattr(settings.risk, "vol_horizon_bars", 5) or 5),
            int(getattr(settings.risk, "tail_window", 100) or 100),
            float(getattr(settings.risk, "tail_alpha", 0.975) or 0.975),
            int(getattr(settings.risk, "tail_step", 5) or 5),
            int(getattr(settings.risk, "tail_max_bars", 300_000) or 300_000),
            float(getattr(settings.risk, "stop_k_vol", 1.0) or 1.0),
            float(getattr(settings.risk, "stop_k_tail", 1.25) or 1.25),
            float(getattr(settings.risk, "meta_label_min_dist", 0.0) or 0.0),
            float(getattr(settings.risk, "regime_adx_trend", 25.0) or 25.0),
            float(getattr(settings.risk, "regime_adx_range", 20.0) or 20.0),
            int(getattr(settings.risk, "hurst_window", 100) or 100),
            float(getattr(settings.risk, "hurst_trend", 0.55) or 0.55),
            float(getattr(settings.risk, "hurst_range", 0.45) or 0.45),
            float(getattr(settings.risk, "rr_trend", 2.5) or 2.5),
            float(getattr(settings.risk, "rr_range", 1.5) or 1.5),
            float(getattr(settings.risk, "rr_neutral", 2.0) or 2.0),
            int(getattr(settings.risk, "ema_fast_period", 20) or 20),
            int(getattr(settings.risk, "ema_slow_period", 50) or 50),
            int(getattr(settings.risk, "atr_period", 14) or 14),
        )
    except Exception:
        return None


def infer_stop_target_pips(
    df: pd.DataFrame,
    *,
    settings: Settings,
    pip_size: float,
    signal: int | None = None,
) -> tuple[float, float, float] | None:
    if df is None or df.empty:
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    if pip_size <= 0:
        return None
    regime = _infer_regime(high, low, close, settings)
    mode = str(getattr(settings.risk, "stop_target_mode", "blend") or "blend").strip().lower()
    if mode in {"smart", "hybrid", "adaptive", "auto"}:
        mode = "blend"

    rust_auto = None
    if mode not in {"structure", "market_structure", "swing"}:
        rust_auto = _rust_vol_tail_pips(df, settings, pip_size)
    vol_tail = None
    if rust_auto is not None:
        slp, tpp, rrp = float(rust_auto[0]), float(rust_auto[1]), float(rust_auto[2])
        if np.isfinite(slp) and np.isfinite(tpp) and np.isfinite(rrp) and slp > 0 and tpp > 0 and rrp > 0:
            vol_tail = slp * pip_size, tpp * pip_size, rrp

    atr = _atr_distances(high, low, close, settings, regime)
    structure = _structure_distances(high, low, close, settings, signal, regime)
    base = vol_tail or atr

    selected: tuple[float, float, float] | None = None
    if mode in {"structure", "market_structure", "swing"}:
        selected = structure or atr
    elif mode in {"atr", "atr_only"}:
        selected = atr or base or structure
    else:
        if structure is not None and base is not None:
            if regime == "trend":
                w_struct = 0.70
            elif regime == "range":
                w_struct = 0.35
            else:
                w_struct = 0.55
            w_atr = 1.0 - w_struct
            sl_dist = (w_struct * structure[0]) + (w_atr * base[0])
            rr = (w_struct * structure[2]) + (w_atr * base[2])
            rr = max(0.1, rr)
            tp_dist = max(sl_dist * rr, (w_struct * structure[1]) + (w_atr * base[1]))
            selected = float(sl_dist), float(tp_dist), float(rr)
        else:
            selected = structure or base or atr

    if selected is None:
        return None
    sl_dist, tp_dist, rr = selected
    if sl_dist <= 0.0 or tp_dist <= 0.0:
        return None
    sl_pips = float(sl_dist / max(pip_size, 1e-9))
    tp_pips = float(tp_dist / max(pip_size, 1e-9))
    rr_out = float(tp_pips / max(sl_pips, 1e-9))
    return sl_pips, tp_pips, rr_out


__all__ = ["infer_stop_target_pips"]
