from __future__ import annotations
from typing import Any, Iterable, Mapping

import numpy as np

from ..core.config import Settings

try:
    import forex_bindings as _fb  # type: ignore
except Exception:  # pragma: no cover - optional native extension
    _fb = None

_RUST_STOP_FULL_SUPPORT: bool | None = None


def _as_f64(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr


def _extract_ohlcv_arrays(data: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    def _pull(name: str) -> np.ndarray | None:
        src: Any = None
        if isinstance(data, Mapping):
            src = data.get(name)
        else:
            try:
                src = data[name]  # type: ignore[index]
            except Exception:
                src = None
        if src is None:
            return None
        try:
            if hasattr(src, "to_numpy"):
                return _as_f64(src.to_numpy(dtype=np.float64, copy=False))
            return _as_f64(src)
        except Exception:
            return None

    open_ = _pull("open")
    high = _pull("high")
    low = _pull("low")
    close = _pull("close")
    if open_ is None or high is None or low is None or close is None:
        return None
    n = int(min(open_.size, high.size, low.size, close.size))
    if n <= 0:
        return None
    return open_[-n:], high[-n:], low[-n:], close[-n:]


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    n = int(min(high.size, low.size, close.size))
    if n < 2:
        return 0.0
    h = high[-n:]
    l = low[-n:]
    c = close[-n:]
    tr = np.empty(n, dtype=np.float64)
    tr[0] = float(abs(h[0] - l[0]))
    prev_close = c[:-1]
    tr[1:] = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - prev_close), np.abs(l[1:] - prev_close)),
    )
    window = max(2, int(period))
    if tr.size < window:
        atr = float(np.nanmean(tr))
    else:
        atr = float(np.nanmean(tr[-window:]))
    if not np.isfinite(atr):
        return 0.0
    return atr


def _adx_last(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    n = int(min(high.size, low.size, close.size))
    if n <= period + 1:
        return float("nan")
    h = high[-n:]
    l = low[-n:]
    c = close[-n:]

    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
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
    adx = float(np.nanmean(np.asarray(dx[:period], dtype=np.float64)))
    for val in dx[period:]:
        adx = ((adx * (p - 1.0)) + float(val)) / p
    return adx


def _infer_regime(high: np.ndarray, low: np.ndarray, close: np.ndarray, settings: Settings) -> str:
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


def _swing_levels(high: np.ndarray, low: np.ndarray, *, lookback: int, swing_window: int) -> tuple[float, float] | None:
    n = int(min(high.size, low.size))
    if n <= 0:
        return None
    half = max(1, int(swing_window))
    span = (2 * half) + 1
    lb = max(span + 2, int(lookback))
    lb = min(lb, n)
    if lb < span:
        return None
    hs = high[-lb:]
    ls = low[-lb:]
    eps = 1e-12

    swing_highs: list[float] = []
    swing_lows: list[float] = []
    for i in range(half, lb - half):
        window_high = hs[(i - half) : (i + half + 1)]
        window_low = ls[(i - half) : (i + half + 1)]
        max_w = float(np.nanmax(window_high))
        min_w = float(np.nanmin(window_low))
        if np.isfinite(hs[i]) and np.isfinite(max_w) and float(hs[i]) >= (max_w - eps):
            swing_highs.append(float(hs[i]))
        if np.isfinite(ls[i]) and np.isfinite(min_w) and float(ls[i]) <= (min_w + eps):
            swing_lows.append(float(ls[i]))

    resistance = swing_highs[-1] if swing_highs else float(np.nanmax(hs))
    support = swing_lows[-1] if swing_lows else float(np.nanmin(ls))
    if not np.isfinite(resistance) or not np.isfinite(support):
        return None
    return support, resistance


def _atr_distances(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
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
    rr = max(1.5, rr)
    return float(sl_dist), float(sl_dist * rr), float(rr)


def _structure_distances(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
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
    px = float(close[-1])
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

    rr_floor = max(1.5, float(getattr(settings.risk, "min_risk_reward", 2.0) or 2.0))
    rr_regime = rr_floor
    if regime == "trend":
        rr_regime = float(getattr(settings.risk, "rr_trend", rr_floor) or rr_floor)
    elif regime == "range":
        rr_regime = float(getattr(settings.risk, "rr_range", rr_floor) or rr_floor)
    else:
        rr_regime = float(getattr(settings.risk, "rr_neutral", rr_floor) or rr_floor)
    rr_regime = max(rr_floor, rr_regime)

    rr_struct = tp_raw / max(sl_dist, 1e-9) if (np.isfinite(tp_raw) and tp_raw > 0.0) else rr_regime
    rr = max(rr_floor, min(6.0, max(rr_struct, rr_regime)))
    tp_dist = max(sl_dist * rr, tp_raw if (np.isfinite(tp_raw) and tp_raw > 0.0) else 0.0)
    if tp_dist <= 0.0:
        tp_dist = sl_dist * rr_regime
        rr = max(rr_floor, rr_regime)
    return float(sl_dist), float(tp_dist), float(max(rr_floor, rr))


def _rust_vol_tail_pips(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    settings: Settings,
    pip_size: float,
    *,
    signal: int | None,
    mode: str,
) -> tuple[float, float, float] | None:
    global _RUST_STOP_FULL_SUPPORT
    if _fb is None or not hasattr(_fb, "infer_stop_target_pips_ohlcv"):
        return None
    try:
        common_args = (
            _as_f64(open_),
            _as_f64(high),
            _as_f64(low),
            _as_f64(close),
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
        if _RUST_STOP_FULL_SUPPORT is not False:
            try:
                out = _fb.infer_stop_target_pips_ohlcv(
                    *common_args,
                    str(mode or "blend"),
                    int(signal or 0),
                    float(getattr(settings.risk, "atr_stop_multiplier", 1.5) or 1.5),
                    float(getattr(settings.risk, "min_risk_reward", 2.0) or 2.0),
                    int(getattr(settings.risk, "structure_lookback_bars", 120) or 120),
                    int(getattr(settings.risk, "structure_swing_window", 2) or 2),
                    float(getattr(settings.risk, "structure_min_atr_mult", 0.8) or 0.8),
                    float(getattr(settings.risk, "structure_max_atr_mult", 4.0) or 4.0),
                )
                _RUST_STOP_FULL_SUPPORT = True
                return out
            except TypeError:
                _RUST_STOP_FULL_SUPPORT = False
        if mode in {"structure", "market_structure", "swing"}:
            return None
        return _fb.infer_stop_target_pips_ohlcv(*common_args)
    except Exception:
        return None


def infer_stop_target_pips_ohlcv(
    open_prices: Iterable[Any],
    high_prices: Iterable[Any],
    low_prices: Iterable[Any],
    close_prices: Iterable[Any],
    *,
    settings: Settings,
    pip_size: float,
    signal: int | None = None,
) -> tuple[float, float, float] | None:
    open_ = _as_f64(open_prices)
    high = _as_f64(high_prices)
    low = _as_f64(low_prices)
    close = _as_f64(close_prices)
    n = int(min(open_.size, high.size, low.size, close.size))
    if n <= 0 or pip_size <= 0:
        return None
    open_ = open_[-n:]
    high = high[-n:]
    low = low[-n:]
    close = close[-n:]

    mode = str(getattr(settings.risk, "stop_target_mode", "blend") or "blend").strip().lower()
    if mode in {"smart", "hybrid", "adaptive", "auto"}:
        mode = "blend"

    rust_auto = _rust_vol_tail_pips(open_, high, low, close, settings, pip_size, signal=signal, mode=mode)
    if rust_auto is not None:
        slp, tpp, rrp = float(rust_auto[0]), float(rust_auto[1]), float(rust_auto[2])
        if np.isfinite(slp) and np.isfinite(tpp) and np.isfinite(rrp) and slp > 0 and tpp > 0 and rrp > 0:
            return slp, tpp, rrp
    return None


def infer_stop_target_pips(
    df: object,
    *,
    settings: Settings,
    pip_size: float,
    signal: int | None = None,
) -> tuple[float, float, float] | None:
    arrays = _extract_ohlcv_arrays(df)
    if arrays is None or pip_size <= 0:
        return None
    open_, high, low, close = arrays
    return infer_stop_target_pips_ohlcv(
        open_,
        high,
        low,
        close,
        settings=settings,
        pip_size=pip_size,
        signal=signal,
    )


__all__ = ["infer_stop_target_pips", "infer_stop_target_pips_ohlcv"]
